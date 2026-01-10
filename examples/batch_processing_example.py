#!/usr/bin/env python3
"""
Example script demonstrating programmatic use of the batch processing client.

This shows how to integrate the batch processing functionality into your own scripts.

Usage:
    python batch_processing_example.py <directory>
    python batch_processing_example.py <directory> --output-dir ./output --batch-size 32
"""
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import sys
import time
import argparse
import io
import base64
import asyncio

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slimgest.web.test_client import (
    GlobalMetrics,
    ProgressTracker,
    process_single_pdf,
    generate_performance_chart,
    print_summary_report,
    render_pdf_pages_to_base64,
    batch_pages,
    send_batch_to_server,
    PDFMetrics,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from PIL import Image as PILImage
import httpx


def render_image_to_base64(
    image_path: Path,
) -> Tuple[int, str, float]:
    """
    Load an image file and convert to base64-encoded PNG.
    
    Args:
        image_path: Path to the image file (JPEG or PNG)
    
    Returns:
        Tuple of (page_number=1, base64_string, load_time)
    """
    load_start = time.time()
    
    # Open image and ensure it's RGB
    img = PILImage.open(str(image_path)).convert('RGB')
    
    # Convert to PNG and base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    png_bytes = buffer.getvalue()
    base64_str = base64.b64encode(png_bytes).decode('utf-8')
    
    load_time = time.time() - load_start
    
    return (1, base64_str, load_time)


async def process_single_file_concurrent_async(
    file_path: Path,
    base_url: str,
    dpi: float,
    batch_size: int,
    tracker: ProgressTracker,
    output_dir: Optional[Path] = None,
    max_batches_in_flight: int = 16,
) -> PDFMetrics:
    """
    Process a single file (PDF, JPEG, or PNG) with async concurrent batch sending.
    
    Args:
        file_path: Path to file (PDF, JPEG, or PNG)
        base_url: Base URL of API server
        dpi: DPI for PDF rendering (ignored for images)
        batch_size: Number of pages per batch
        tracker: Progress tracker
        output_dir: Optional output directory for markdown files
        max_batches_in_flight: Maximum number of batches to send concurrently
    
    Returns:
        PDFMetrics object with timing information
    """
    file_start_time = time.time()
    
    # Get file size
    file_size = file_path.stat().st_size
    
    # Initialize metrics
    metrics = PDFMetrics(
        pdf_path=file_path,
        file_size_bytes=file_size,
        total_pages=0,
        start_time=file_start_time,
    )
    
    try:
        # Determine file type
        file_extension = file_path.suffix.lower()
        is_image = file_extension in ['.jpg', '.jpeg', '.png']
        
        # Step 1: Render/Load pages to base64
        render_start = time.time()
        if is_image:
            # For images, just load and convert to base64
            pages_data = [render_image_to_base64(file_path)]
            file_type = "image"
        else:
            # For PDFs, render all pages
            pages_data = render_pdf_pages_to_base64(file_path, dpi)
            file_type = "PDF"
        
        render_time = time.time() - render_start
        
        metrics.total_pages = len(pages_data)
        metrics.render_time = render_time
        
        # Update render metrics
        tracker.update_rendered_pages(len(pages_data), render_time)
        
        # Step 2: Batch pages
        batches = batch_pages(pages_data, batch_size)
        
        render_pps = len(pages_data) / render_time if render_time > 0 else 0
        print(f"  Loaded {len(pages_data)} page(s) from {file_type} in {render_time:.2f}s ({render_pps:.2f} pages/s)")
        print(f"  Created {len(batches)} batch(es) (max {max_batches_in_flight} concurrent)")
        
        # Step 3: Send batches asynchronously with controlled concurrency
        all_results = []
        processing_start = time.time()
        
        # Create async HTTP client
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=10.0),
            limits=httpx.Limits(max_connections=max_batches_in_flight * 2, max_keepalive_connections=max_batches_in_flight),
        ) as client:
            # Create semaphore to limit concurrent batches
            semaphore = asyncio.Semaphore(max_batches_in_flight)
            
            async def send_with_semaphore(batch, batch_idx):
                async with semaphore:
                    results = await send_batch_to_server(batch, base_url, tracker, client)
                    return (batch_idx, results)
            
            # Send all batches concurrently (limited by semaphore)
            tasks = [send_with_semaphore(batch, idx) for idx, batch in enumerate(batches)]
            batch_results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for result in batch_results_list:
                if isinstance(result, Exception):
                    print(f"    Batch failed with exception: {result}")
                else:
                    batch_idx, batch_results = result
                    all_results.extend(batch_results)
                    print(f"    Batch {batch_idx + 1}/{len(batches)} completed: {len(batch_results)} pages")
        
        processing_time = time.time() - processing_start
        metrics.processing_time = processing_time
        
        print(f"  Processed {len(all_results)} pages in {processing_time:.2f}s")
        
        # Sort results by page number
        all_results.sort(key=lambda x: x['page_number'])
        
        # Combine OCR results
        ocr_texts = [r.get('ocr_text', '') for r in all_results]
        full_text = "\n\n".join(ocr_texts)
        metrics.ocr_text = full_text
        
        # Debug: Check if we got OCR text
        total_ocr_chars = sum(len(t) for t in ocr_texts)
        print(f"  Total OCR characters: {total_ocr_chars:,}")
        if total_ocr_chars == 0:
            print(f"  ⚠️  WARNING: No OCR text extracted! Check server logs.")
        
        # Step 4: Write to markdown if output directory specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            md_filename = file_path.stem + ".md"
            md_path = output_dir / md_filename
            
            md_content = f"# {file_path.name}\n\n"
            md_content += f"**File Type:** {file_type}\n\n"
            md_content += f"**Total Pages:** {metrics.total_pages}\n\n"
            md_content += f"**File Size:** {metrics.file_size_bytes:,} bytes\n\n"
            md_content += f"**Render Time:** {metrics.render_time:.2f}s\n\n"
            md_content += f"**Processing Time:** {metrics.processing_time:.2f}s\n\n"
            md_content += f"**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            md_content += "---\n\n"
            
            for idx, result in enumerate(all_results):
                page_num = result.get('page_number', idx + 1)
                page_text = result.get('ocr_text', '')
                md_content += f"\n\n## Page {page_num}\n\n"
                if page_text:
                    md_content += page_text
                else:
                    md_content += "*[No text extracted]*\n"
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            print(f"  ✓ Saved markdown to: {md_path}")
        
        metrics.end_time = time.time()
        metrics.total_time = metrics.end_time - metrics.start_time
        
        return metrics
    
    except Exception as e:
        print(f"  ✗ Error processing {file_path.name}: {e}")
        import traceback
        traceback.print_exc()
        metrics.end_time = time.time()
        metrics.total_time = metrics.end_time - metrics.start_time
        return metrics


def process_single_file_concurrent(
    file_path: Path,
    base_url: str,
    dpi: float,
    batch_size: int,
    tracker: ProgressTracker,
    output_dir: Optional[Path] = None,
    max_batches_in_flight: int = 16,
) -> PDFMetrics:
    """
    Process a single file (PDF, JPEG, or PNG) with concurrent batch sending (sync wrapper).
    
    Args:
        file_path: Path to file (PDF, JPEG, or PNG)
        base_url: Base URL of API server
        dpi: DPI for PDF rendering (ignored for images)
        batch_size: Number of pages per batch
        tracker: Progress tracker
        output_dir: Optional output directory for markdown files
        max_batches_in_flight: Maximum number of batches to send concurrently
    
    Returns:
        PDFMetrics object with timing information
    """
    return asyncio.run(process_single_file_concurrent_async(
        file_path, base_url, dpi, batch_size, tracker, output_dir, max_batches_in_flight
    ))


def process_files_from_directory(
    directory: Path,
    base_url: str = "http://localhost:7670",
    dpi: float = 150.0,
    batch_size: int = 32,
    output_dir: Path = Path("./example_output"),
    max_batches_in_flight: int = 16,
):
    """
    Process all files (PDF, JPEG, PNG) in a directory programmatically.
    
    Args:
        directory: Directory containing files
        base_url: URL of the OCR server
        dpi: DPI for PDF rendering (ignored for images)
        batch_size: Pages per batch
        output_dir: Directory to save output files
        max_batches_in_flight: Maximum concurrent batches per file
    """
    # Find all supported files in the directory
    supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png']
    all_files = [
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
    
    if not all_files:
        print(f"❌ No supported files (PDF, JPEG, PNG) found in {directory}")
        return
    
    # Sort by file size (largest first)
    all_files = sorted(all_files, key=lambda f: f.stat().st_size, reverse=True)
    
    # Count file types
    pdfs = [f for f in all_files if f.suffix.lower() == '.pdf']
    images = [f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    print(f"\n{'='*80}")
    print(f"Found {len(all_files)} files in {directory}")
    print(f"  - {len(pdfs)} PDF(s)")
    print(f"  - {len(images)} image(s) (JPEG/PNG)")
    print(f"Sorted by size (largest first)")
    print(f"{'='*80}\n")
    
    # Initialize metrics
    total_bytes = sum(f.stat().st_size for f in all_files)
    global_metrics = GlobalMetrics(
        total_pdfs=len(all_files),
        total_bytes=total_bytes,
        max_batches_in_flight=max_batches_in_flight,
    )
    
    # Start timing
    global_metrics.start_time = time.time()
    
    # Create tracker
    tracker = ProgressTracker(global_metrics)
    
    # Process each file
    for idx, file_path in enumerate(all_files, 1):
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        file_type = "PDF" if file_path.suffix.lower() == '.pdf' else "Image"
        print(f"\n[{idx}/{len(all_files)}] Processing {file_path.name} ({file_type}, {file_size_mb:.2f} MB)...")
        
        try:
            # Process the file with concurrent batch sending
            file_metrics = process_single_file_concurrent(
                file_path=file_path,
                base_url=base_url,
                dpi=dpi,
                batch_size=batch_size,
                tracker=tracker,
                output_dir=output_dir,
                max_batches_in_flight=max_batches_in_flight,
            )
            
            # Add to global metrics
            tracker.add_pdf_metrics(file_metrics)
            
            # Print results
            render_pps = file_metrics.total_pages / file_metrics.render_time if file_metrics.render_time > 0 else 0
            proc_pps = file_metrics.total_pages / file_metrics.processing_time if file_metrics.processing_time > 0 else 0
            
            print(f"✓ Completed {file_path.name}")
            print(f"  Pages: {file_metrics.total_pages}")
            print(f"  Render time: {file_metrics.render_time:.2f}s ({render_pps:.2f} pages/s)")
            print(f"  Processing time: {file_metrics.processing_time:.2f}s ({proc_pps:.2f} pages/s)")
            print(f"  Total time: {file_metrics.total_time:.2f}s")
            print(f"  OCR text length: {len(file_metrics.ocr_text):,} characters")
            
            if len(file_metrics.ocr_text) == 0:
                print(f"  ⚠️  WARNING: No OCR text extracted!")
            
        except Exception as e:
            print(f"✗ Failed to process {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Finalize capacity tracking
    tracker.finalize_capacity_tracking()
    
    # Generate summary report
    print("\n" + "=" * 80)
    print_summary_report(global_metrics, output_dir)
    
    # Access metrics programmatically
    print("\n" + "=" * 80)
    print("Programmatic Access to Metrics:")
    print("=" * 80)
    print(f"  Total pages rendered: {global_metrics.rendered_pages}")
    print(f"  Total pages processed: {global_metrics.processed_pages}")
    print(f"  Render pages/second: {global_metrics.render_pages_per_second():.2f}")
    print(f"  Processing pages/second: {global_metrics.processing_pages_per_second():.2f}")
    print(f"  Overall pages/second: {global_metrics.pages_per_second():.2f}")
    print(f"  Number of PDFs completed: {global_metrics.completed_pdfs}")
    print(f"  Total bytes processed: {global_metrics.bytes_read:,}")
    print(f"  Total render time: {global_metrics.total_render_time:.2f}s")
    print(f"  Total processing time: {global_metrics.total_processing_time:.2f}s")
    
    # Capacity utilization analysis
    total_capacity_time = global_metrics.time_below_max_capacity + global_metrics.time_at_max_capacity
    if total_capacity_time > 0:
        utilization_pct = (global_metrics.time_at_max_capacity / total_capacity_time) * 100
        efficiency = (global_metrics.peak_batches_in_flight / global_metrics.max_batches_in_flight) * 100
        
        print(f"\nCapacity Utilization:")
        print(f"  Max concurrent batches allowed: {global_metrics.max_batches_in_flight}")
        print(f"  Peak batches in flight: {global_metrics.peak_batches_in_flight} ({efficiency:.1f}% of max)")
        print(f"  Time below max capacity: {global_metrics.time_below_max_capacity:.2f}s ({(global_metrics.time_below_max_capacity/total_capacity_time)*100:.1f}%)")
        print(f"  Time at max capacity: {global_metrics.time_at_max_capacity:.2f}s ({utilization_pct:.1f}%)")
        
        # Provide recommendations
        print(f"\n  Analysis & Recommendations:")
        if utilization_pct < 30:
            print(f"    ⚠️  Bottleneck: Client-side rendering is too slow")
            if efficiency < 80:
                recommended = global_metrics.peak_batches_in_flight + 2
                print(f"    💡 Reduce --max-batches-in-flight to {recommended} (you never exceeded {global_metrics.peak_batches_in_flight})")
        elif utilization_pct > 70:
            print(f"    ⚠️  Bottleneck: Server-side processing")
            if efficiency > 90:
                recommended = global_metrics.max_batches_in_flight * 2
                print(f"    💡 Increase --max-batches-in-flight to {recommended} (you're maxing out capacity)")
        else:
            print(f"    ✓ Balanced workload between client and server")
            if efficiency < 60:
                recommended = global_metrics.peak_batches_in_flight + 2
                print(f"    💡 Consider --max-batches-in-flight={recommended} for better efficiency")
    
    # Access individual file metrics
    if global_metrics.pdf_metrics:
        slowest = max(global_metrics.pdf_metrics, key=lambda x: x.total_time)
        fastest = min(global_metrics.pdf_metrics, key=lambda x: x.total_time)
        
        print(f"\nSlowest File:")
        print(f"  Name: {slowest.pdf_path.name}")
        print(f"  Time: {slowest.total_time:.2f}s")
        print(f"  Pages: {slowest.total_pages}")
        print(f"  Size: {slowest.file_size_bytes:,} bytes")
        
        print(f"\nFastest File:")
        print(f"  Name: {fastest.pdf_path.name}")
        print(f"  Time: {fastest.total_time:.2f}s")
        print(f"  Pages: {fastest.total_pages}")
        print(f"  Size: {fastest.file_size_bytes:,} bytes")
    
    # Generate chart
    has_chart_data = (len(global_metrics.render_pages_per_second_history) > 0 or 
                      len(global_metrics.processing_pages_per_second_history) > 0)
    if output_dir and has_chart_data:
        chart_path = output_dir / "performance_chart.png"
        generate_performance_chart(global_metrics, chart_path)
        print(f"\n✓ Performance chart saved to: {chart_path}")
    
    print("\n" + "=" * 80)


def process_with_custom_progress_callback():
    """Example showing how to add custom progress callbacks."""
    
    # You can extend ProgressTracker to add custom callbacks
    class CustomProgressTracker(ProgressTracker):
        def update_pages(self, count: int = 1):
            super().update_pages(count)
            # Custom callback - could send to monitoring system
            print(f"Custom callback: {self.metrics.processed_pages} pages processed")
    
    # Use the custom tracker
    metrics = GlobalMetrics(total_pdfs=1, total_bytes=0)
    import time
    metrics.start_time = time.time()
    
    tracker = CustomProgressTracker(metrics)
    
    # ... rest of processing code


def access_results_programmatically():
    """Example showing how to access and use the results."""
    
    # After processing, you can access all the data
    pdf_metrics_example = {
        'pdf_path': Path("document.pdf"),
        'file_size_bytes': 1234567,
        'total_pages': 45,
        'render_time': 12.5,
        'processing_time': 23.8,
        'total_time': 36.3,
        'ocr_text': "Full extracted text here...",
    }
    
    # You can:
    # 1. Store results in a database
    # 2. Send to an API
    # 3. Generate custom reports
    # 4. Trigger downstream processing
    # 5. Calculate custom statistics
    
    # Example: Calculate cost based on pages
    cost_per_page = 0.01  # $0.01 per page
    total_cost = pdf_metrics_example['total_pages'] * cost_per_page
    print(f"Processing cost: ${total_cost:.2f}")
    
    # Example: Calculate efficiency
    pages_per_second = pdf_metrics_example['total_pages'] / pdf_metrics_example['total_time']
    print(f"Efficiency: {pages_per_second:.2f} pages/second")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Batch process PDFs from a directory using programmatic API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files (PDF, JPEG, PNG) in a directory (sorted by size, largest first)
  python batch_processing_example.py ./documents/
  
  # Specify custom output directory
  python batch_processing_example.py ./documents/ --output-dir ./my_output
  
  # Adjust batch size and concurrent batches
  python batch_processing_example.py ./documents/ --batch-size 64 --max-batches-in-flight 32
  
  # High quality PDF rendering with lower concurrency
  python batch_processing_example.py ./documents/ --dpi 300 --max-batches-in-flight 8
  
  # Use remote server
  python batch_processing_example.py ./documents/ --url http://gpu-server:7670

Supported file types: PDF, JPEG (.jpg/.jpeg), PNG (.png)
Images are processed as single-page documents.
        """
    )
    
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing files to process (PDF, JPEG, PNG)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./example_output"),
        help="Directory to save output files (default: ./example_output)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:7670",
        help="URL of the OCR server (default: http://localhost:7670)"
    )
    parser.add_argument(
        "--dpi",
        type=float,
        default=150.0,
        help="DPI for PDF rendering (default: 150.0)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of pages per batch (default: 32)"
    )
    parser.add_argument(
        "--max-batches-in-flight",
        type=int,
        default=16,
        help="Maximum number of concurrent batches per PDF (default: 16)"
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not args.directory.exists():
        print(f"❌ Error: Directory not found: {args.directory}")
        sys.exit(1)
    
    if not args.directory.is_dir():
        print(f"❌ Error: Not a directory: {args.directory}")
        sys.exit(1)
    
    print("=" * 80)
    print("Batch Processing Client - Programmatic Example")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input directory: {args.directory}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Server URL: {args.url}")
    print(f"  DPI: {args.dpi}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max batches in flight: {args.max_batches_in_flight}")
    print(f"  Files will be processed largest to smallest")
    print(f"  Supported types: PDF, JPEG, PNG")
    
    # Run the processing
    try:
        process_files_from_directory(
            directory=args.directory,
            base_url=args.url,
            dpi=args.dpi,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            max_batches_in_flight=args.max_batches_in_flight,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. The server is running (python -m slimgest.web)")
        print("  2. The directory contains supported files (PDF, JPEG, PNG)")
        print("  3. All dependencies are installed")
        print("  4. The server URL is correct")
        import traceback
        traceback.print_exc()
        sys.exit(1)