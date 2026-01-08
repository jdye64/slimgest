#!/usr/bin/env python3
"""
Example script demonstrating the new PDF rendering utilities.

This shows how to use the generator-based PDF processing functions for
memory-efficient page-by-page processing.
"""

import sys
from pathlib import Path
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from slimgest.pdf import (
    iter_pdf_page_bitmaps,
    iter_pdf_page_tensors,
    load_pdf_page_tensors,
    crop_tensor_with_bbox,
)
import numpy as np


def example_1_iterate_bitmaps(pdf_path: str):
    """Example 1: Iterate through PDF pages as bitmaps."""
    print("\n" + "="*60)
    print("Example 1: Iterate Through PDF Pages as Bitmaps")
    print("="*60)
    
    for page_bitmap in iter_pdf_page_bitmaps(pdf_path, dpi=150):
        print(f"\nPage {page_bitmap.page_number}:")
        print(f"  Size: {page_bitmap.width}x{page_bitmap.height} pixels")
        
        # Convert to numpy array
        arr = page_bitmap.to_numpy()
        print(f"  NumPy array shape: {arr.shape}")
        print(f"  Data type: {arr.dtype}")
        print(f"  Value range: [{arr.min()}, {arr.max()}]")


def example_2_iterate_tensors_cpu(pdf_path: str):
    """Example 2: Iterate through PDF pages as CPU tensors."""
    print("\n" + "="*60)
    print("Example 2: Iterate Through PDF Pages as CPU Tensors")
    print("="*60)
    
    for page_tensor in iter_pdf_page_tensors(pdf_path, dpi=150, device="cpu"):
        print(f"\nPage {page_tensor.page_number}:")
        print(f"  Tensor shape: {page_tensor.tensor.shape}")  # [C, H, W]
        print(f"  Original size: {page_tensor.original_width}x{page_tensor.original_height}")
        print(f"  Device: {page_tensor.device}")
        print(f"  Data type: {page_tensor.tensor.dtype}")
        print(f"  Value range: [{page_tensor.tensor.min()}, {page_tensor.tensor.max()}]")


def example_3_iterate_tensors_gpu(pdf_path: str):
    """Example 3: Iterate through PDF pages as GPU tensors."""
    print("\n" + "="*60)
    print("Example 3: Iterate Through PDF Pages as GPU Tensors")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU example")
        return
    
    for page_tensor in iter_pdf_page_tensors(pdf_path, dpi=150, device="cuda"):
        print(f"\nPage {page_tensor.page_number}:")
        print(f"  Tensor shape: {page_tensor.tensor.shape}")
        print(f"  Original size: {page_tensor.original_width}x{page_tensor.original_height}")
        print(f"  Device: {page_tensor.device}")
        print(f"  Memory allocated on GPU: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")


def example_4_load_all_pages(pdf_path: str):
    """Example 4: Load all pages at once (non-streaming)."""
    print("\n" + "="*60)
    print("Example 4: Load All Pages at Once")
    print("="*60)
    
    tensors, shapes = load_pdf_page_tensors(pdf_path, dpi=150, device="cpu")
    
    print(f"\nLoaded {len(tensors)} pages total")
    for i, (tensor, shape) in enumerate(zip(tensors, shapes)):
        print(f"  Page {i}: tensor shape={tensor.shape}, original shape={shape}")


def example_5_crop_from_tensor(pdf_path: str):
    """Example 5: Crop regions from a page tensor."""
    print("\n" + "="*60)
    print("Example 5: Crop Regions from Page Tensor")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get first page
    for page_tensor in iter_pdf_page_tensors(pdf_path, dpi=150, device=device):
        print(f"\nProcessing page {page_tensor.page_number}:")
        print(f"  Original tensor shape: {page_tensor.tensor.shape}")
        
        # Define some example bounding boxes (normalized coordinates)
        # Format: [xmin, ymin, xmax, ymax] in range [0, 1]
        example_bboxes = [
            np.array([0.1, 0.1, 0.4, 0.4]),  # Top-left region
            np.array([0.6, 0.6, 0.9, 0.9]),  # Bottom-right region
            np.array([0.3, 0.4, 0.7, 0.6]),  # Center region
        ]
        
        original_shape = (page_tensor.original_height, page_tensor.original_width)
        
        # For this example, assume the tensor is already at its original size
        # (no resizing done yet)
        resized_shape = original_shape
        
        print(f"  Original shape: {original_shape}")
        print(f"  Cropping {len(example_bboxes)} regions:")
        
        for i, bbox in enumerate(example_bboxes):
            cropped = crop_tensor_with_bbox(
                page_tensor.tensor,
                bbox,
                original_shape,
                resized_shape
            )
            print(f"    Crop {i}: bbox={bbox.tolist()}, shape={cropped.shape}")
        
        # Only process first page for this example
        break


def main():
    """Run all examples."""
    if len(sys.argv) < 2:
        print("Usage: python pdf_render_example.py <path_to_pdf>")
        print("\nThis script demonstrates the new PDF rendering utilities.")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Processing PDF: {pdf_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Run examples
    try:
        example_1_iterate_bitmaps(pdf_path)
        example_2_iterate_tensors_cpu(pdf_path)
        example_3_iterate_tensors_gpu(pdf_path)
        example_4_load_all_pages(pdf_path)
        example_5_crop_from_tensor(pdf_path)
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
