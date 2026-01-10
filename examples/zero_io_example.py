#!/usr/bin/env python3
"""
Zero-IO batch processing example - pre-loads all images into memory before processing.

This script demonstrates maximum throughput by eliminating I/O bottlenecks:
1. Loads ALL images from directory into memory first
2. Creates batches of 32 images
3. Sends batches concurrently to REST service
4. Measures time spent at max capacity to evaluate server throughput

Usage:
    python zero_io_example.py <image_directory>
    python zero_io_example.py <image_directory> --max-batches-in-flight 64
    python zero_io_example.py <image_directory> --batch-size 16 --max-batches-in-flight 32
"""

import argparse
import base64
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from PIL import Image as PILImage

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slimgest.web.test_client import (
    GlobalMetrics,
    ProgressTracker,
    batch_pages,
    send_batch_to_server,
)


def load_image_to_base64(image_path: Path) -> Tuple[int, str, float]:
    """
    Load an image file and convert to base64-encoded PNG.
    
    Args:
        image_path: Path to the image file (JPEG or PNG)
    
    Returns:
        Tuple of (page_number, base64_string, load_time)
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
    
    # Return with sequential page number (will be assigned during load)
    return (0, base64_str, load_time)


def preload_all_images(image_dir: Path, supported_extensions: List[str]) -> Tuple[List[Tuple[int, str, float]], float]:
    """
    Pre-load ALL images from directory into memory.
    
    Args:
        image_dir: Directory containing images
        supported_extensions: List of supported file extensions (e.g., ['.jpg', '.jpeg', '.png'])
    
    Returns:
        Tuple of (list of (page_number, base64_string, load_time), total_load_time)
    """
    # Find all image files
    image_files = [
        f for f in sorted(image_dir.iterdir())
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir} with extensions {supported_extensions}")
    
    print(f"\n{'='*80}")
    print(f"PRE-LOADING PHASE: Loading {len(image_files)} images into memory...")
    print(f"{'='*80}")
    
    preload_start = time.time()
    loaded_images = []
    
    for idx, image_path in enumerate(image_files, 1):
        page_num, base64_str, load_time = load_image_to_base64(image_path)
        # Assign sequential page number
        loaded_images.append((idx, base64_str, load_time))
        
        # Progress indicator
        if idx % 10 == 0 or idx == len(image_files):
            print(f"  Loaded {idx}/{len(image_files)} images...", end='\r')
    
    total_load_time = time.time() - preload_start
    
    print(f"\n✓ Pre-loaded {len(loaded_images)} images in {total_load_time:.2f}s")
    print(f"  Average load time: {total_load_time/len(loaded_images):.3f}s per image")
    print(f"  Total memory footprint: ~{sum(len(img[1]) for img in loaded_images) / (1024*1024):.1f} MB (base64)")
    print(f"{'='*80}\n")
    
    return loaded_images, total_load_time


def process_preloaded_images(
    preloaded_images: List[Tuple[int, str, float]],
    base_url: str,
    batch_size: int,
    max_batches_in_flight: int,
) -> GlobalMetrics:
    """
    Process pre-loaded images with concurrent batch sending.
    
    This eliminates I/O bottlenecks and measures pure processing throughput.
    
    Args:
        preloaded_images: List of (page_number, base64_string, load_time) tuples
        base_url: Base URL of API server
        batch_size: Number of images per batch
        max_batches_in_flight: Maximum number of batches to send concurrently
    
    Returns:
        GlobalMetrics object with performance statistics
    """
    total_images = len(preloaded_images)
    
    # Initialize metrics
    global_metrics = GlobalMetrics(
        total_pdfs=1,  # Treating all images as one logical "document"
        total_bytes=0,  # Not tracking individual file sizes
        max_batches_in_flight=max_batches_in_flight,
    )
    global_metrics.start_time = time.time()
    
    tracker = ProgressTracker(global_metrics)
    
    # Create batches
    print(f"{'='*80}")
    print(f"BATCHING PHASE: Creating batches of {batch_size} images...")
    print(f"{'='*80}")
    
    batches = batch_pages(preloaded_images, batch_size)
    
    print(f"✓ Created {len(batches)} batches from {total_images} images")
    print(f"  Batch size: {batch_size}")
    print(f"  Max concurrent batches: {max_batches_in_flight}")
    print(f"{'='*80}\n")
    
    # Send batches concurrently
    print(f"{'='*80}")
    print(f"PROCESSING PHASE: Sending batches to {base_url}")
    print(f"{'='*80}")
    
    all_results = []
    processing_start = time.time()
    
    with ThreadPoolExecutor(max_workers=max_batches_in_flight) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(send_batch_to_server, batch, base_url, tracker): batch_idx
            for batch_idx, batch in enumerate(batches)
        }
        
        print(f"Submitted {len(batches)} batches to executor...")
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                completed += 1
                
                # Progress indicator
                elapsed = time.time() - processing_start
                pages_so_far = len(all_results)
                pps = pages_so_far / elapsed if elapsed > 0 else 0
                
                print(f"  [{completed}/{len(batches)}] Batch {batch_idx + 1} completed: "
                      f"{len(batch_results)} images | "
                      f"Total: {pages_so_far}/{total_images} | "
                      f"{pps:.1f} img/s", 
                      end='\r')
                
            except Exception as e:
                print(f"\n  ✗ Batch {batch_idx + 1} failed: {e}")
    
    processing_time = time.time() - processing_start
    
    print(f"\n\n✓ Processed {len(all_results)} images in {processing_time:.2f}s")
    print(f"  Average throughput: {len(all_results)/processing_time:.2f} images/second")
    print(f"{'='*80}\n")
    
    # Finalize metrics
    tracker.finalize_capacity_tracking()
    global_metrics.end_time = time.time()
    global_metrics.total_processing_time = processing_time
    
    # Sort results by page number
    all_results.sort(key=lambda x: x['page_number'])
    
    # Check OCR extraction
    total_ocr_chars = sum(len(r.get('ocr_text', '')) for r in all_results)
    print(f"OCR Results:")
    print(f"  Total characters extracted: {total_ocr_chars:,}")
    if total_ocr_chars == 0:
        print(f"  ⚠️  WARNING: No OCR text extracted! Check server logs.")
    else:
        print(f"  ✓ OCR extraction successful")
    
    return global_metrics


def print_performance_analysis(metrics: GlobalMetrics, preload_time: float):
    """
    Print detailed performance analysis focusing on capacity utilization.
    
    Args:
        metrics: GlobalMetrics with performance data
        preload_time: Time spent pre-loading images
    """
    print(f"\n{'='*80}")
    print(f"PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Time breakdown
    total_time = metrics.end_time - metrics.start_time
    processing_time = metrics.total_processing_time
    
    print(f"\nTime Breakdown:")
    print(f"  Pre-load time:    {preload_time:>8.2f}s")
    print(f"  Processing time:  {processing_time:>8.2f}s")
    print(f"  Total time:       {total_time:>8.2f}s")
    print(f"  Processing/Total: {(processing_time/total_time)*100:>7.1f}%")
    
    # Throughput metrics
    print(f"\nThroughput:")
    print(f"  Pages processed:     {metrics.processed_pages}")
    print(f"  Processing rate:     {metrics.processing_pages_per_second():.2f} images/s")
    print(f"  Overall rate:        {metrics.pages_per_second():.2f} images/s")
    
    # Capacity utilization analysis
    total_capacity_time = metrics.time_below_max_capacity + metrics.time_at_max_capacity
    
    if total_capacity_time > 0:
        utilization_pct = (metrics.time_at_max_capacity / total_capacity_time) * 100
        efficiency_pct = (metrics.peak_batches_in_flight / metrics.max_batches_in_flight) * 100
        
        print(f"\nCapacity Utilization:")
        print(f"  Max concurrent batches:   {metrics.max_batches_in_flight}")
        print(f"  Peak batches in flight:   {metrics.peak_batches_in_flight} ({efficiency_pct:.1f}% of max)")
        print(f"  Time below max capacity:  {metrics.time_below_max_capacity:.2f}s ({(metrics.time_below_max_capacity/total_capacity_time)*100:.1f}%)")
        print(f"  Time at max capacity:     {metrics.time_at_max_capacity:.2f}s ({utilization_pct:.1f}%)")
        
        # Analysis
        print(f"\n  📊 Analysis:")
        
        if utilization_pct >= 90:
            print(f"    ✓ EXCELLENT: Sustained max capacity {utilization_pct:.1f}% of processing time")
            print(f"    ✓ Zero-IO strategy is effective - server is the bottleneck")
            if efficiency_pct >= 95:
                recommended = metrics.max_batches_in_flight * 2
                print(f"    💡 Consider increasing --max-batches-in-flight to {recommended}")
                print(f"       (consistently maxing out at {metrics.peak_batches_in_flight} batches)")
        elif utilization_pct >= 70:
            print(f"    ✓ GOOD: High capacity utilization ({utilization_pct:.1f}%)")
            print(f"    ✓ Server keeping up well with pre-loaded images")
        elif utilization_pct >= 40:
            print(f"    ⚠️  MODERATE: Medium capacity utilization ({utilization_pct:.1f}%)")
            print(f"    ⚠️  Server processing is keeping up, but not fully saturated")
            if efficiency_pct < 70:
                recommended = metrics.peak_batches_in_flight + 4
                print(f"    💡 Reduce --max-batches-in-flight to {recommended}")
                print(f"       (peak was only {metrics.peak_batches_in_flight} batches)")
        else:
            print(f"    ❌ LOW: Poor capacity utilization ({utilization_pct:.1f}%)")
            print(f"    ❌ Server processing very fast, batch submission too slow")
            print(f"    💡 Potential issues:")
            print(f"       - Network latency")
            print(f"       - Batch size too small")
            print(f"       - max_batches_in_flight too low")
            if efficiency_pct < 50:
                recommended = max(4, metrics.peak_batches_in_flight + 2)
                print(f"    💡 Reduce --max-batches-in-flight to {recommended}")
        
        # Efficiency score
        efficiency_score = min(utilization_pct, 100)
        print(f"\n  🎯 Zero-IO Efficiency Score: {efficiency_score:.1f}/100")
        print(f"     (Measures how well pre-loading eliminates I/O bottlenecks)")
    
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-IO batch processing - pre-loads all images into memory before processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in a directory (default: 32 images/batch, 16 concurrent batches)
  python zero_io_example.py ./images/
  
  # Maximize throughput with high concurrency
  python zero_io_example.py ./images/ --max-batches-in-flight 64
  
  # Smaller batches, higher concurrency
  python zero_io_example.py ./images/ --batch-size 16 --max-batches-in-flight 32
  
  # Use remote server
  python zero_io_example.py ./images/ --url http://gpu-server:7670

This script is optimized for measuring pure server throughput by:
  1. Pre-loading ALL images into memory (eliminates read I/O)
  2. Batching all images before sending (eliminates batching overhead)
  3. Concurrent batch submission (maximizes server utilization)
  
Use this to benchmark your server's maximum processing capacity.
        """
    )
    
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing images to process (JPEG, PNG)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:7670",
        help="URL of the OCR server (default: http://localhost:7670)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images per batch (default: 32)"
    )
    parser.add_argument(
        "--max-batches-in-flight",
        type=int,
        default=16,
        help="Maximum number of concurrent batches (default: 16)"
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
    print("ZERO-IO BATCH PROCESSING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input directory:          {args.directory}")
    print(f"  Server URL:               {args.url}")
    print(f"  Batch size:               {args.batch_size}")
    print(f"  Max batches in flight:    {args.max_batches_in_flight}")
    print(f"  Strategy:                 Pre-load all images before processing")
    
    try:
        # Phase 1: Pre-load all images into memory
        supported_extensions = ['.jpg', '.jpeg', '.png']
        preloaded_images, preload_time = preload_all_images(
            args.directory,
            supported_extensions
        )
        
        # Phase 2: Process all pre-loaded images
        metrics = process_preloaded_images(
            preloaded_images=preloaded_images,
            base_url=args.url,
            batch_size=args.batch_size,
            max_batches_in_flight=args.max_batches_in_flight,
        )
        
        # Phase 3: Analyze performance
        print_performance_analysis(metrics, preload_time)
        
        print("\n✓ Zero-IO processing complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. The server is running (python -m slimgest.web)")
        print("  2. The directory contains images (JPEG, PNG)")
        print("  3. All dependencies are installed")
        print("  4. The server URL is correct")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
