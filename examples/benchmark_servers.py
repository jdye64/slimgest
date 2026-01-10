#!/usr/bin/env python3
"""
Benchmark script to compare performance between Python FastAPI and Rust servers.

This script:
1. Submits PDF files to both servers
2. Measures latency and throughput
3. Compares performance metrics
4. Generates a detailed report
"""
import os
import sys
import time
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


@dataclass
class RequestResult:
    """Result of a single request"""
    server: str
    filename: str
    success: bool
    latency: float  # seconds
    total_pages: int
    error: str = ""


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results"""
    server: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_latency: float
    avg_latency: float
    min_latency: float
    max_latency: float
    total_pages: int
    pages_per_second: float
    requests_per_second: float


async def process_pdf_file(
    client: httpx.AsyncClient,
    server_url: str,
    pdf_path: Path,
    dpi: float = 150.0,
) -> RequestResult:
    """Process a single PDF file through the server"""
    server_name = "Python" if "7670" in server_url else "Rust"
    
    start_time = time.time()
    
    try:
        with open(pdf_path, "rb") as f:
            files = {"file": (pdf_path.name, f, "application/pdf")}
            data = {"dpi": str(dpi)}
            
            response = await client.post(
                f"{server_url}/process-pdf",
                files=files,
                data=data,
                timeout=300.0,  # 5 minutes timeout
            )
            response.raise_for_status()
            
        latency = time.time() - start_time
        result_data = response.json()
        
        # Extract total pages from response
        total_pages = result_data.get("total_pages_processed", 0)
        
        return RequestResult(
            server=server_name,
            filename=pdf_path.name,
            success=True,
            latency=latency,
            total_pages=total_pages,
        )
        
    except Exception as e:
        latency = time.time() - start_time
        return RequestResult(
            server=server_name,
            filename=pdf_path.name,
            success=False,
            latency=latency,
            total_pages=0,
            error=str(e),
        )


async def benchmark_server(
    server_url: str,
    pdf_files: List[Path],
    dpi: float,
    concurrent: int = 1,
) -> List[RequestResult]:
    """Benchmark a server with the given PDF files"""
    server_name = "Python" if "7670" in server_url else "Rust"
    results = []
    
    async with httpx.AsyncClient() as client:
        # Process files with controlled concurrency
        semaphore = asyncio.Semaphore(concurrent)
        
        async def process_with_semaphore(pdf_path: Path):
            async with semaphore:
                return await process_pdf_file(client, server_url, pdf_path, dpi)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Processing {len(pdf_files)} files on {server_name} server...",
                total=len(pdf_files)
            )
            
            # Create tasks for all files
            tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_files]
            
            # Process and collect results
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                progress.update(task, advance=1)
    
    return results


def calculate_statistics(results: List[RequestResult]) -> BenchmarkResults:
    """Calculate aggregate statistics from results"""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    if not successful:
        return BenchmarkResults(
            server=results[0].server if results else "Unknown",
            total_requests=len(results),
            successful_requests=0,
            failed_requests=len(failed),
            total_latency=0,
            avg_latency=0,
            min_latency=0,
            max_latency=0,
            total_pages=0,
            pages_per_second=0,
            requests_per_second=0,
        )
    
    latencies = [r.latency for r in successful]
    total_latency = sum(latencies)
    total_pages = sum(r.total_pages for r in successful)
    
    return BenchmarkResults(
        server=results[0].server,
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_latency=total_latency,
        avg_latency=sum(latencies) / len(latencies),
        min_latency=min(latencies),
        max_latency=max(latencies),
        total_pages=total_pages,
        pages_per_second=total_pages / total_latency if total_latency > 0 else 0,
        requests_per_second=len(successful) / total_latency if total_latency > 0 else 0,
    )


def print_results(python_results: BenchmarkResults, rust_results: BenchmarkResults):
    """Print comparison table of results"""
    console.print("\n[bold cyan]Benchmark Results[/bold cyan]")
    
    # Create comparison table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Python FastAPI", justify="right", style="yellow")
    table.add_column("Rust Axum", justify="right", style="green")
    table.add_column("Difference", justify="right", style="red")
    
    # Add rows
    table.add_row(
        "Total Requests",
        str(python_results.total_requests),
        str(rust_results.total_requests),
        "-"
    )
    
    table.add_row(
        "Successful Requests",
        str(python_results.successful_requests),
        str(rust_results.successful_requests),
        "-"
    )
    
    table.add_row(
        "Failed Requests",
        str(python_results.failed_requests),
        str(rust_results.failed_requests),
        "-"
    )
    
    table.add_row(
        "Total Pages Processed",
        str(python_results.total_pages),
        str(rust_results.total_pages),
        "-"
    )
    
    # Average latency
    python_avg = python_results.avg_latency
    rust_avg = rust_results.avg_latency
    improvement = ((python_avg - rust_avg) / python_avg * 100) if python_avg > 0 else 0
    table.add_row(
        "Avg Latency (seconds)",
        f"{python_avg:.3f}",
        f"{rust_avg:.3f}",
        f"{improvement:+.1f}%"
    )
    
    # Min latency
    table.add_row(
        "Min Latency (seconds)",
        f"{python_results.min_latency:.3f}",
        f"{rust_results.min_latency:.3f}",
        "-"
    )
    
    # Max latency
    table.add_row(
        "Max Latency (seconds)",
        f"{python_results.max_latency:.3f}",
        f"{rust_results.max_latency:.3f}",
        "-"
    )
    
    # Pages per second
    python_pps = python_results.pages_per_second
    rust_pps = rust_results.pages_per_second
    pps_improvement = ((rust_pps - python_pps) / python_pps * 100) if python_pps > 0 else 0
    table.add_row(
        "Pages per Second",
        f"{python_pps:.2f}",
        f"{rust_pps:.2f}",
        f"{pps_improvement:+.1f}%"
    )
    
    # Requests per second
    python_rps = python_results.requests_per_second
    rust_rps = rust_results.requests_per_second
    rps_improvement = ((rust_rps - python_rps) / python_rps * 100) if python_rps > 0 else 0
    table.add_row(
        "Requests per Second",
        f"{python_rps:.2f}",
        f"{rust_rps:.2f}",
        f"{rps_improvement:+.1f}%"
    )
    
    console.print(table)
    
    # Print summary
    console.print("\n[bold]Summary:[/bold]")
    if improvement > 0:
        console.print(f"✓ Rust server is [green]{improvement:.1f}% faster[/green] on average")
    elif improvement < 0:
        console.print(f"✗ Python server is [yellow]{-improvement:.1f}% faster[/yellow] on average")
    else:
        console.print("○ Both servers have similar performance")
    
    if pps_improvement > 0:
        console.print(f"✓ Rust server processes [green]{pps_improvement:.1f}% more pages per second[/green]")
    elif pps_improvement < 0:
        console.print(f"✗ Python server processes [yellow]{-pps_improvement:.1f}% more pages per second[/yellow]")


def save_detailed_report(
    python_results: List[RequestResult],
    rust_results: List[RequestResult],
    python_stats: BenchmarkResults,
    rust_stats: BenchmarkResults,
    output_file: Path,
):
    """Save detailed JSON report"""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": {
            "statistics": asdict(python_stats),
            "individual_results": [asdict(r) for r in python_results],
        },
        "rust": {
            "statistics": asdict(rust_stats),
            "individual_results": [asdict(r) for r in rust_results],
        },
    }
    
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    
    console.print(f"\n[green]Detailed report saved to:[/green] {output_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Python vs Rust web servers for PDF processing"
    )
    parser.add_argument(
        "pdf_directory",
        type=Path,
        help="Directory containing PDF files to process"
    )
    parser.add_argument(
        "--python-url",
        default="http://localhost:7670",
        help="Python server URL (default: http://localhost:7670)"
    )
    parser.add_argument(
        "--rust-url",
        default="http://localhost:7671",
        help="Rust server URL (default: http://localhost:7671)"
    )
    parser.add_argument(
        "--dpi",
        type=float,
        default=150.0,
        help="DPI for PDF rendering (default: 150.0)"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.json"),
        help="Output file for detailed results (default: benchmark_results.json)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of PDF files to process (default: all)"
    )
    
    args = parser.parse_args()
    
    # Find PDF files
    pdf_files = sorted(args.pdf_directory.glob("*.pdf"))
    
    if not pdf_files:
        console.print(f"[red]No PDF files found in {args.pdf_directory}[/red]")
        sys.exit(1)
    
    if args.max_files:
        pdf_files = pdf_files[:args.max_files]
    
    console.print(f"[bold]Found {len(pdf_files)} PDF files to process[/bold]")
    console.print(f"Python server: {args.python_url}")
    console.print(f"Rust server: {args.rust_url}")
    console.print(f"DPI: {args.dpi}")
    console.print(f"Concurrent requests: {args.concurrent}\n")
    
    # Check server availability
    console.print("[yellow]Checking server availability...[/yellow]")
    async with httpx.AsyncClient() as client:
        try:
            python_resp = await client.get(f"{args.python_url}/", timeout=5.0)
            console.print(f"✓ Python server is running: {python_resp.json()}")
        except Exception as e:
            console.print(f"[red]✗ Python server is not available: {e}[/red]")
            sys.exit(1)
        
        try:
            rust_resp = await client.get(f"{args.rust_url}/", timeout=5.0)
            console.print(f"✓ Rust server is running: {rust_resp.json()}")
        except Exception as e:
            console.print(f"[red]✗ Rust server is not available: {e}[/red]")
            sys.exit(1)
    
    console.print()
    
    # Benchmark Python server
    console.print("[bold cyan]Benchmarking Python FastAPI server...[/bold cyan]")
    python_results = await benchmark_server(
        args.python_url,
        pdf_files,
        args.dpi,
        args.concurrent,
    )
    python_stats = calculate_statistics(python_results)
    
    # Small delay between benchmarks
    await asyncio.sleep(2)
    
    # Benchmark Rust server
    console.print("\n[bold cyan]Benchmarking Rust Axum server...[/bold cyan]")
    rust_results = await benchmark_server(
        args.rust_url,
        pdf_files,
        args.dpi,
        args.concurrent,
    )
    rust_stats = calculate_statistics(rust_results)
    
    # Print comparison
    print_results(python_stats, rust_stats)
    
    # Save detailed report
    save_detailed_report(
        python_results,
        rust_results,
        python_stats,
        rust_stats,
        args.output,
    )


if __name__ == "__main__":
    asyncio.run(main())
