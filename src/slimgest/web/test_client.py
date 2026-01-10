#!/usr/bin/env python3
"""
Enhanced test client for the slim-gest FastAPI server with client-side PDF processing.

This client:
- Uses pypdfium2 to render PDFs locally
- Batches pages into chunks of 32
- Sends base64-encoded PNGs to the server
- Provides rich progress tracking and metrics
- Generates performance charts
"""
import sys
import json
import time
import base64
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import defaultdict
import io

import requests
import pypdfium2 as pdfium
from PIL import Image
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


console = Console()


@dataclass
class PageMetrics:
    """Metrics for a single page."""
    page_number: int
    render_time: float
    upload_time: float
    processing_time: float
    total_time: float


@dataclass
class PDFMetrics:
    """Metrics for a single PDF."""
    pdf_path: Path
    file_size_bytes: int
    total_pages: int
    render_time: float = 0.0
    upload_time: float = 0.0
    processing_time: float = 0.0
    total_time: float = 0.0
    pages: List[PageMetrics] = field(default_factory=list)
    ocr_text: str = ""
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class GlobalMetrics:
    """Global metrics across all PDFs."""
    total_pdfs: int = 0
    completed_pdfs: int = 0
    total_pages: int = 0
    rendered_pages: int = 0
    processed_pages: int = 0
    total_bytes: int = 0
    bytes_read: int = 0
    batches_sent: int = 0
    batches_in_flight: int = 0
    batches_completed: int = 0
    max_batches_in_flight: int = 0
    peak_batches_in_flight: int = 0  # Actual max achieved during processing
    start_time: float = 0.0
    total_render_time: float = 0.0
    total_processing_time: float = 0.0
    time_below_max_capacity: float = 0.0  # Time with batches_in_flight < max
    time_at_max_capacity: float = 0.0     # Time with batches_in_flight == max
    last_capacity_check: float = 0.0
    render_pages_per_second_history: List[Tuple[float, float]] = field(default_factory=list)
    processing_pages_per_second_history: List[Tuple[float, float]] = field(default_factory=list)
    pdf_metrics: List[PDFMetrics] = field(default_factory=list)
    
    def pages_per_second(self) -> float:
        """Calculate current overall pages per second."""
        if self.start_time == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.processed_pages / elapsed
    
    def render_pages_per_second(self) -> float:
        """Calculate rendering pages per second."""
        if self.total_render_time == 0:
            return 0.0
        return self.rendered_pages / self.total_render_time
    
    def processing_pages_per_second(self) -> float:
        """Calculate server processing pages per second."""
        if self.total_processing_time == 0:
            return 0.0
        return self.processed_pages / self.total_processing_time


class ProgressTracker:
    """Thread-safe progress tracker."""
    
    def __init__(self, global_metrics: GlobalMetrics):
        self.metrics = global_metrics
        self.lock = Lock()
        self.last_update = time.time()
        
    def _update_capacity_time(self):
        """Update time spent at various capacity levels."""
        current_time = time.time()
        if self.metrics.last_capacity_check == 0.0:
            self.metrics.last_capacity_check = current_time
            return
        
        elapsed = current_time - self.metrics.last_capacity_check
        
        # Check if we were at max capacity
        if self.metrics.max_batches_in_flight > 0:
            if self.metrics.batches_in_flight >= self.metrics.max_batches_in_flight:
                self.metrics.time_at_max_capacity += elapsed
            else:
                self.metrics.time_below_max_capacity += elapsed
        
        self.metrics.last_capacity_check = current_time
        
    def update_rendered_pages(self, count: int = 1, render_time: float = 0.0):
        """Update rendered pages count."""
        with self.lock:
            self.metrics.rendered_pages += count
            self.metrics.total_render_time += render_time
            current_time = time.time()
            if current_time - self.last_update >= 0.5:  # Record every 0.5 seconds
                render_pps = self.metrics.render_pages_per_second()
                elapsed = current_time - self.metrics.start_time
                self.metrics.render_pages_per_second_history.append((elapsed, render_pps))
                self.last_update = current_time
    
    def update_pages(self, count: int = 1, processing_time: float = 0.0):
        """Update processed pages count."""
        with self.lock:
            self.metrics.processed_pages += count
            self.metrics.total_processing_time += processing_time
            current_time = time.time()
            if current_time - self.last_update >= 0.5:  # Record every 0.5 seconds
                processing_pps = self.metrics.processing_pages_per_second()
                elapsed = current_time - self.metrics.start_time
                self.metrics.processing_pages_per_second_history.append((elapsed, processing_pps))
                self.last_update = current_time
    
    def update_batches_sent(self):
        """Increment batches sent counter."""
        with self.lock:
            self._update_capacity_time()
            self.metrics.batches_sent += 1
            self.metrics.batches_in_flight += 1
            
            # Track peak batches in flight
            if self.metrics.batches_in_flight > self.metrics.peak_batches_in_flight:
                self.metrics.peak_batches_in_flight = self.metrics.batches_in_flight
    
    def update_batches_completed(self):
        """Increment batches completed counter."""
        with self.lock:
            self._update_capacity_time()
            self.metrics.batches_in_flight -= 1
            self.metrics.batches_completed += 1
    
    def add_pdf_metrics(self, pdf_metrics: PDFMetrics):
        """Add completed PDF metrics."""
        with self.lock:
            self.metrics.pdf_metrics.append(pdf_metrics)
            self.metrics.completed_pdfs += 1
    
    def finalize_capacity_tracking(self):
        """Finalize capacity tracking at the end of processing."""
        with self.lock:
            self._update_capacity_time()


def render_pdf_pages_to_base64(
    pdf_path: Path,
    dpi: float = 150.0,
) -> List[Tuple[int, str, float]]:
    """
    Render all pages of a PDF to base64-encoded PNG images.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: DPI for rendering
    
    Returns:
        List of tuples: (page_number, base64_string, render_time)
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    pages_data = []
    
    scale = dpi / 72.0  # Convert DPI to scale factor
    
    for page_idx in range(len(pdf)):
        page_start = time.time()
        page = pdf[page_idx]
        
        # Render page to bitmap
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        
        # Convert to PNG and base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        png_bytes = buffer.getvalue()
        base64_str = base64.b64encode(png_bytes).decode('utf-8')
        
        render_time = time.time() - page_start
        pages_data.append((page_idx + 1, base64_str, render_time))
    
    pdf.close()
    return pages_data


def batch_pages(pages: List[Tuple[int, str, float]], batch_size: int = 32) -> List[List[Tuple[int, str, float]]]:
    """Split pages into batches of specified size."""
    batches = []
    for i in range(0, len(pages), batch_size):
        batches.append(pages[i:i + batch_size])
    return batches


def send_batch_to_server(
    batch: List[Tuple[int, str, float]],
    base_url: str,
    tracker: ProgressTracker,
) -> List[Dict]:
    """
    Send a batch of pages to the server and collect results.
    
    Args:
        batch: List of (page_number, base64_string, render_time) tuples
        base_url: Base URL of the API server
        tracker: Progress tracker
    
    Returns:
        List of page results
    """
    # Prepare request payload
    images = [
        {"page_number": page_num, "image_base64": base64_str}
        for page_num, base64_str, _ in batch
    ]
    
    payload = {"images": images}
    
    batch_start = time.time()
    tracker.update_batches_sent()
    
    try:
        # Send batch to server
        response = requests.post(
            f"{base_url}/process-batch-stream",
            json=payload,
            stream=True,
            timeout=600,
        )
        
        if response.status_code != 200:
            console.print(f"[red]Error: {response.status_code}[/red]")
            console.print(f"[red]Response: {response.text}[/red]")
            tracker.update_batches_completed()
            return []
        
        # Process SSE stream
        results = []
        event_type = None
        page_count = 0
        
        for line in response.iter_lines():
            if not line:
                continue
            
            line = line.decode('utf-8')
            
            if line.startswith('event:'):
                event_type = line.split(':', 1)[1].strip()
            elif line.startswith('data:'):
                data_str = line.split(':', 1)[1].strip()
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    console.print(f"[yellow]Failed to parse JSON: {data_str[:100]}[/yellow]")
                    continue
                
                if event_type == 'page':
                    page_count += 1
                    # Debug: Check if OCR text is present
                    ocr_text = data.get('ocr_text', '')
                    if not ocr_text:
                        console.print(f"[yellow]Warning: Page {data.get('page_number')} has empty OCR text[/yellow]")
                        console.print(f"[yellow]Raw data keys: {list(data.keys())}[/yellow]")
                    results.append(data)
                    tracker.update_pages(1)
                elif event_type == 'complete':
                    pass
                elif event_type == 'error':
                    console.print(f"[red]Batch error: {data.get('error')}[/red]")
        
        batch_time = time.time() - batch_start
        
        # Debug output
        if not results:
            console.print(f"[red]Warning: Batch returned no results![/red]")
        else:
            total_chars = sum(len(r.get('ocr_text', '')) for r in results)
            if total_chars == 0:
                console.print(f"[yellow]Warning: Batch processed {len(results)} pages but extracted 0 characters[/yellow]")
        
        tracker.update_batches_completed()
        return results
    
    except Exception as e:
        console.print(f"[red]Exception sending batch: {e}[/red]")
        import traceback
        traceback.print_exc()
        tracker.update_batches_completed()
        return []


def process_single_pdf(
    pdf_path: Path,
    base_url: str,
    dpi: float,
    batch_size: int,
    tracker: ProgressTracker,
    output_dir: Optional[Path] = None,
) -> PDFMetrics:
    """
    Process a single PDF file.
    
    Args:
        pdf_path: Path to PDF file
        base_url: Base URL of API server
        dpi: DPI for rendering
        batch_size: Number of pages per batch
        tracker: Progress tracker
        output_dir: Optional output directory for markdown files
    
    Returns:
        PDFMetrics object with timing information
    """
    pdf_start_time = time.time()
    
    # Get file size
    file_size = pdf_path.stat().st_size
    
    # Initialize metrics
    metrics = PDFMetrics(
        pdf_path=pdf_path,
        file_size_bytes=file_size,
        total_pages=0,
        start_time=pdf_start_time,
    )
    
    try:
        # Step 1: Render all pages to base64
        render_start = time.time()
        pages_data = render_pdf_pages_to_base64(pdf_path, dpi)
        render_time = time.time() - render_start
        
        metrics.total_pages = len(pages_data)
        metrics.render_time = render_time
        
        # Step 2: Batch pages
        batches = batch_pages(pages_data, batch_size)
        
        # Step 3: Send batches to server and collect results
        all_results = []
        processing_start = time.time()
        
        for batch in batches:
            batch_results = send_batch_to_server(batch, base_url, tracker)
            all_results.extend(batch_results)
        
        processing_time = time.time() - processing_start
        metrics.processing_time = processing_time
        
        # Combine OCR results
        all_results.sort(key=lambda x: x['page_number'])
        ocr_texts = [r['ocr_text'] for r in all_results]
        full_text = "\n\n".join(ocr_texts)
        metrics.ocr_text = full_text
        
        # Step 4: Write to markdown if output directory specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            md_filename = pdf_path.stem + ".md"
            md_path = output_dir / md_filename
            
            md_content = f"# {pdf_path.name}\n\n"
            md_content += f"**Total Pages:** {metrics.total_pages}\n\n"
            md_content += f"**File Size:** {metrics.file_size_bytes:,} bytes\n\n"
            md_content += f"**Render Time:** {metrics.render_time:.2f}s\n\n"
            md_content += f"**Processing Time:** {metrics.processing_time:.2f}s\n\n"
            md_content += f"**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            md_content += "---\n\n"
            
            for idx, result in enumerate(all_results):
                md_content += f"\n\n## Page {result['page_number']}\n\n"
                md_content += result['ocr_text']
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
        
        metrics.end_time = time.time()
        metrics.total_time = metrics.end_time - metrics.start_time
        
        return metrics
    
    except Exception as e:
        console.print(f"[red]Error processing {pdf_path.name}: {e}[/red]")
        metrics.end_time = time.time()
        metrics.total_time = metrics.end_time - metrics.start_time
        return metrics


def generate_performance_chart(metrics: GlobalMetrics, output_path: Path):
    """Generate a performance chart showing pages/second over time (render vs processing)."""
    has_render_data = len(metrics.render_pages_per_second_history) > 0
    has_processing_data = len(metrics.processing_pages_per_second_history) > 0
    
    if not has_render_data and not has_processing_data:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Render performance chart
    if has_render_data:
        render_times = [t for t, _ in metrics.render_pages_per_second_history]
        render_pps = [p for _, p in metrics.render_pages_per_second_history]
        
        ax1.plot(render_times, render_pps, linewidth=2, color='#e74c3c', label='Render')
        ax1.fill_between(render_times, render_pps, alpha=0.3, color='#e74c3c')
        
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Pages per Second', fontsize=12)
        ax1.set_title('Client-Side Rendering Performance', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add stats
        avg_render = sum(render_pps) / len(render_pps) if render_pps else 0
        max_render = max(render_pps) if render_pps else 0
        stats_text = f'Avg: {avg_render:.2f} pages/s\nPeak: {max_render:.2f} pages/s'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#ffcccb', alpha=0.7))
    
    # Processing performance chart
    if has_processing_data:
        proc_times = [t for t, _ in metrics.processing_pages_per_second_history]
        proc_pps = [p for _, p in metrics.processing_pages_per_second_history]
        
        ax2.plot(proc_times, proc_pps, linewidth=2, color='#3498db', label='Processing')
        ax2.fill_between(proc_times, proc_pps, alpha=0.3, color='#3498db')
        
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Pages per Second', fontsize=12)
        ax2.set_title('Server-Side OCR Processing Performance', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add stats
        avg_proc = sum(proc_pps) / len(proc_pps) if proc_pps else 0
        max_proc = max(proc_pps) if proc_pps else 0
        stats_text = f'Avg: {avg_proc:.2f} pages/s\nPeak: {max_proc:.2f} pages/s'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_summary_report(metrics: GlobalMetrics, output_dir: Optional[Path] = None):
    """Print a detailed summary report."""
    console.print("\n")
    console.print("=" * 80)
    console.print("[bold green]PROCESSING COMPLETE[/bold green]")
    console.print("=" * 80)
    
    elapsed = time.time() - metrics.start_time
    
    # Overall statistics
    table = Table(title="Overall Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Total PDFs Processed", f"{metrics.completed_pdfs} / {metrics.total_pdfs}")
    table.add_row("Total Pages Rendered", f"{metrics.rendered_pages:,}")
    table.add_row("Total Pages Processed", f"{metrics.processed_pages:,}")
    table.add_row("Total Bytes Read", f"{metrics.bytes_read:,}")
    table.add_row("Total Batches Sent", f"{metrics.batches_sent:,}")
    table.add_row("Total Time", f"{elapsed:.2f}s")
    table.add_row("Total Render Time", f"{metrics.total_render_time:.2f}s")
    table.add_row("Total Processing Time", f"{metrics.total_processing_time:.2f}s")
    table.add_row("Render Pages/Second", f"{metrics.render_pages_per_second():.2f}")
    table.add_row("Processing Pages/Second", f"{metrics.processing_pages_per_second():.2f}")
    table.add_row("Overall Pages/Second", f"{metrics.pages_per_second():.2f}")
    
    # Capacity utilization
    if metrics.max_batches_in_flight > 0:
        total_capacity_time = metrics.time_below_max_capacity + metrics.time_at_max_capacity
        if total_capacity_time > 0:
            utilization_pct = (metrics.time_at_max_capacity / total_capacity_time) * 100
            table.add_row("", "")  # Empty row for spacing
            table.add_row("Max Concurrent Batches", f"{metrics.max_batches_in_flight}")
            table.add_row("Peak Batches In Flight", f"{metrics.peak_batches_in_flight}")
            table.add_row("Time Below Max Capacity", f"{metrics.time_below_max_capacity:.2f}s ({(metrics.time_below_max_capacity/total_capacity_time)*100:.1f}%)")
            table.add_row("Time At Max Capacity", f"{metrics.time_at_max_capacity:.2f}s ({utilization_pct:.1f}%)")
            
            # Add interpretation
            efficiency = (metrics.peak_batches_in_flight / metrics.max_batches_in_flight) * 100
            
            if utilization_pct < 30:
                recommendation = f"⚠️ Client-side bottleneck (rendering too slow)"
                if efficiency < 80:
                    recommendation += f"\n   💡 Reduce --max-batches-in-flight to {metrics.peak_batches_in_flight + 2}"
            elif utilization_pct > 70:
                recommendation = f"⚠️ Server-side bottleneck"
                if efficiency > 90:
                    recommendation += f"\n   💡 Increase --max-batches-in-flight to {metrics.max_batches_in_flight * 2}"
            else:
                recommendation = "✓ Balanced workload"
                if efficiency < 60:
                    recommendation += f"\n   💡 Consider --max-batches-in-flight={metrics.peak_batches_in_flight + 2}"
            
            table.add_row("Bottleneck Analysis", recommendation)
    
    console.print(table)
    console.print()
    
    # Top 100 slowest PDFs
    if metrics.pdf_metrics:
        slowest = sorted(metrics.pdf_metrics, key=lambda x: x.total_time, reverse=True)[:100]
        
        slowest_table = Table(
            title="Top 100 Slowest PDFs",
            show_header=True,
            header_style="bold yellow"
        )
        slowest_table.add_column("Rank", style="yellow", width=6)
        slowest_table.add_column("PDF Name", style="white", width=40)
        slowest_table.add_column("Pages", style="cyan", width=8)
        slowest_table.add_column("Size (bytes)", style="cyan", width=12)
        slowest_table.add_column("Render (s)", style="green", width=12)
        slowest_table.add_column("Process (s)", style="blue", width=12)
        slowest_table.add_column("Total (s)", style="red", width=12)
        
        for rank, pdf_metric in enumerate(slowest, 1):
            slowest_table.add_row(
                str(rank),
                pdf_metric.pdf_path.name[:40],
                str(pdf_metric.total_pages),
                f"{pdf_metric.file_size_bytes:,}",
                f"{pdf_metric.render_time:.2f}",
                f"{pdf_metric.processing_time:.2f}",
                f"{pdf_metric.total_time:.2f}",
            )
        
        console.print(slowest_table)
    
    # Generate performance chart
    has_chart_data = (len(metrics.render_pages_per_second_history) > 0 or 
                      len(metrics.processing_pages_per_second_history) > 0)
    if output_dir and has_chart_data:
        chart_path = output_dir / "performance_chart.png"
        generate_performance_chart(metrics, chart_path)
        console.print(f"\n[green]✓[/green] Performance chart saved to: {chart_path}")
    
    console.print("\n" + "=" * 80 + "\n")


def process_directory(
    directory: Path,
    base_url: str,
    dpi: float = 150.0,
    batch_size: int = 32,
    output_dir: Optional[Path] = None,
    max_workers: int = 4,
):
    """
    Process all PDF files in a directory with concurrent processing.
    
    Args:
        directory: Directory containing PDF files
        base_url: Base URL of the API server
        dpi: DPI for PDF rendering
        batch_size: Number of pages per batch
        output_dir: Directory to save markdown outputs
        max_workers: Maximum number of concurrent PDF processing threads
    """
    # Find all PDF files
    pdf_files = sorted([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == '.pdf'])
    
    if not pdf_files:
        console.print(f"[red]No PDF files found in {directory}[/red]")
        return
    
    # Calculate total bytes and pages (approximate)
    total_bytes = sum(f.stat().st_size for f in pdf_files)
    
    # Initialize global metrics
    global_metrics = GlobalMetrics(
        total_pdfs=len(pdf_files),
        total_bytes=total_bytes,
        start_time=time.time(),
    )
    
    tracker = ProgressTracker(global_metrics)
    
    console.print(f"\n[bold cyan]Starting batch processing[/bold cyan]")
    console.print(f"  PDFs: {len(pdf_files)}")
    console.print(f"  Total size: {total_bytes:,} bytes")
    console.print(f"  Batch size: {batch_size} pages")
    console.print(f"  Max concurrent PDFs: {max_workers}")
    console.print(f"  DPI: {dpi}")
    console.print()
    
    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:
        
        pdf_task = progress.add_task(
            "[cyan]Processing PDFs...",
            total=len(pdf_files)
        )
        
        # Process PDFs concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_pdf,
                    pdf,
                    base_url,
                    dpi,
                    batch_size,
                    tracker,
                    output_dir
                ): pdf
                for pdf in pdf_files
            }
            
            for future in as_completed(futures):
                pdf = futures[future]
                try:
                    pdf_metrics = future.result()
                    tracker.add_pdf_metrics(pdf_metrics)
                    
                    # Update progress
                    progress.update(pdf_task, advance=1)
                    
                    # Show current stats
                    pps = global_metrics.pages_per_second()
                    progress.console.print(
                        f"[green]✓[/green] {pdf.name[:50]:<50} | "
                        f"Pages: {pdf_metrics.total_pages:>4} | "
                        f"Time: {pdf_metrics.total_time:>6.2f}s | "
                        f"Current: {pps:.2f} pages/s"
                    )
                    
                except Exception as e:
                    progress.console.print(f"[red]✗ {pdf.name}: {e}[/red]")
                    progress.update(pdf_task, advance=1)
    
    # Print summary report
    print_summary_report(global_metrics, output_dir)


def test_health_check(base_url: str):
    """Test the health check endpoint."""
    console.print("\n[bold]Testing health check...[/bold]")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            console.print(f"[green]✓[/green] Server is healthy")
            console.print(f"  Status: {data.get('status')}")
            console.print(f"  Models loaded: {data.get('models_loaded')}")
        else:
            console.print(f"[red]✗[/red] Server returned status {response.status_code}")
            sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Health check failed: {e}")
        sys.exit(1)
    console.print()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        console.print("[bold]Usage:[/bold]")
        console.print("  python test_client.py <pdf_file_or_directory> [options]")
        console.print("\n[bold]Options:[/bold]")
        console.print("  --output-dir <dir>   Directory to save markdown files (default: ./output)")
        console.print("  --dpi <dpi>          DPI for PDF rendering (default: 150.0)")
        console.print("  --batch-size <n>     Pages per batch (default: 32)")
        console.print("  --workers <n>        Max concurrent PDFs (default: 4)")
        console.print("  --url <url>          Base URL of API server (default: http://localhost:7670)")
        console.print("\n[bold]Examples:[/bold]")
        console.print("  # Process a single PDF")
        console.print("  python test_client.py document.pdf --output-dir ./output")
        console.print()
        console.print("  # Process all PDFs in a directory")
        console.print("  python test_client.py ./pdfs/ --output-dir ./output --workers 8")
        sys.exit(1)
    
    # Parse arguments
    base_url = "http://localhost:7670"
    input_path = Path(sys.argv[1])
    output_dir = Path("./output")
    dpi = 150.0
    batch_size = 32
    max_workers = 4
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--output-dir' and i + 1 < len(sys.argv):
            output_dir = Path(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--dpi' and i + 1 < len(sys.argv):
            dpi = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--batch-size' and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--workers' and i + 1 < len(sys.argv):
            max_workers = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--url' and i + 1 < len(sys.argv):
            base_url = sys.argv[i + 1].rstrip("/")
            i += 2
        else:
            i += 1
    
    # Validate input path
    if not input_path.exists():
        console.print(f"[red]Error: Path not found: {input_path}[/red]")
        sys.exit(1)
    
    # Test health check
    test_health_check(base_url)
    
    # Process based on input type
    if input_path.is_file():
        if not input_path.suffix.lower() == '.pdf':
            console.print(f"[red]Error: Not a PDF file: {input_path}[/red]")
            sys.exit(1)
        
        # Process single file as a "directory" with one file
        global_metrics = GlobalMetrics(
            total_pdfs=1,
            total_bytes=input_path.stat().st_size,
            start_time=time.time(),
        )
        tracker = ProgressTracker(global_metrics)
        
        pdf_metrics = process_single_pdf(
            input_path,
            base_url,
            dpi,
            batch_size,
            tracker,
            output_dir,
        )
        tracker.add_pdf_metrics(pdf_metrics)
        print_summary_report(global_metrics, output_dir)
    
    elif input_path.is_dir():
        process_directory(input_path, base_url, dpi, batch_size, output_dir, max_workers)
    
    else:
        console.print(f"[red]Error: Invalid input path: {input_path}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
