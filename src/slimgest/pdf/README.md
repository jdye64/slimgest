# PDF Processing Utilities

This module provides reusable utilities for processing PDF documents, with a focus on efficient memory usage and GPU acceleration.

## Overview

The module is organized into several components:

- **`render.py`**: PDF page rendering to bitmaps and tensors
- **`tensor_ops.py`**: Tensor manipulation operations (cropping, coordinate transforms)
- **`convert.py`**: PDF to image conversion utilities
- **`extract.py`**: PDF content extraction
- **`examine.py`**: PDF structure analysis

## Key Features

### Generator-Based Processing

All rendering functions use Python generators to process PDF pages one at a time, avoiding the need to load entire documents into memory at once.

### GPU Acceleration

Tensors can be directly loaded onto GPU devices (CUDA) for immediate processing without intermediate CPU copies.

## Usage Examples

### Example 1: Iterate Through PDF Pages as Tensors

```python
from slimgest.pdf import iter_pdf_page_tensors

# Process pages one at a time on GPU
for page_tensor in iter_pdf_page_tensors("document.pdf", device="cuda", dpi=150):
    print(f"Page {page_tensor.page_number}")
    print(f"  Tensor shape: {page_tensor.tensor.shape}")  # [3, H, W]
    print(f"  Original size: {page_tensor.original_width}x{page_tensor.original_height}")
    print(f"  Device: {page_tensor.device}")
    
    # Process the tensor immediately
    # tensor is already on GPU, ready for model inference
    result = your_model(page_tensor.tensor)
```

### Example 2: Iterate Through PDF Pages as Bitmaps

```python
from slimgest.pdf import iter_pdf_page_bitmaps

# Process pages as bitmaps (useful for CPU-only processing)
for page_bitmap in iter_pdf_page_bitmaps("document.pdf", dpi=150):
    print(f"Page {page_bitmap.page_number}")
    print(f"  Size: {page_bitmap.width}x{page_bitmap.height}")
    
    # Convert to numpy array when needed
    arr = page_bitmap.to_numpy()  # Shape: [H, W, 3]
    
    # Process with your pipeline
    process_image(arr)
```

### Example 3: Load All Pages (Non-Streaming)

```python
from slimgest.pdf import load_pdf_page_tensors

# Load all pages at once (for small PDFs)
tensors, shapes = load_pdf_page_tensors("document.pdf", device="cuda")

print(f"Loaded {len(tensors)} pages")
for i, (tensor, shape) in enumerate(zip(tensors, shapes)):
    print(f"Page {i}: tensor shape={tensor.shape}, original shape={shape}")
```

### Example 4: Crop Regions from Tensors

```python
from slimgest.pdf import iter_pdf_page_tensors, crop_tensor_with_bbox
import numpy as np

for page_tensor in iter_pdf_page_tensors("document.pdf", device="cuda"):
    # Assume we have detected bounding boxes (normalized coordinates)
    bbox = np.array([0.1, 0.2, 0.5, 0.8])  # [xmin, ymin, xmax, ymax]
    
    # Original shape before any resizing
    original_shape = (page_tensor.original_height, page_tensor.original_width)
    
    # If tensor was resized to (1024, 1024)
    resized_shape = (1024, 1024)
    
    # Crop the region (handles coordinate transformation automatically)
    cropped = crop_tensor_with_bbox(
        page_tensor.tensor,
        bbox,
        original_shape,
        resized_shape
    )
    
    print(f"Cropped tensor shape: {cropped.shape}")
```

### Example 5: Batch Crop Multiple Regions

```python
from slimgest.pdf import batch_crop_tensors
import numpy as np

# Multiple bounding boxes (e.g., from object detection)
bboxes = [
    np.array([0.1, 0.1, 0.3, 0.3]),
    np.array([0.5, 0.5, 0.9, 0.9]),
    np.array([0.2, 0.6, 0.4, 0.8]),
]

# Crop all regions at once
crops = batch_crop_tensors(
    image_tensor,
    bboxes,
    original_shape=(800, 600),
    resized_shape=(1024, 1024),
    clone=True  # Clone each crop to avoid memory aliasing
)

print(f"Created {len(crops)} crops")
for i, crop in enumerate(crops):
    print(f"Crop {i}: {crop.shape}")
```

## API Reference

### render.py

#### `iter_pdf_page_bitmaps(pdf_path, dpi=150.0, rotation=0, grayscale=False)`

Generator that yields `PageBitmapWithText` objects for each page.

**Parameters:**
- `pdf_path`: Path to PDF file
- `dpi`: Resolution for rendering (default 150)
- `rotation`: Rotation angle in degrees (0, 90, 180, 270)
- `grayscale`: If True, render in grayscale

**Yields:** `PageBitmapWithText` with attributes:
- `page_number`: 0-indexed page number
- `bitmap`: pypdfium2 bitmap object
- `width`, `height`: Bitmap dimensions
- `text`: Raw embedded text extracted from the PDF page via PDFium (not OCR)
- `to_numpy()`: Method to convert to numpy array

#### `iter_pdf_page_tensors(pdf_path, dpi=150.0, device="cpu", dtype=torch.uint8, ...)`

Generator that yields `PageTensor` objects for each page.

**Parameters:**
- `pdf_path`: Path to PDF file
- `dpi`: Resolution for rendering (default 150)
- `device`: Target device ("cpu", "cuda", or torch.device)
- `dtype`: Tensor data type (default torch.uint8)
- `rotation`: Rotation angle
- `grayscale`: If True, render in grayscale
- `non_blocking`: If True, use non-blocking GPU transfer

**Yields:** `PageTensor` with attributes:
- `page_number`: 0-indexed page number
- `tensor`: PyTorch tensor of shape [C, H, W]
- `original_width`, `original_height`: Original dimensions
- `device`: Tensor device

#### `load_pdf_page_tensors(pdf_path, ...)`

Load all pages at once (non-streaming).

**Returns:** Tuple of (tensors, shapes)
- `tensors`: List of tensors, each [C, H, W]
- `shapes`: List of (height, width) tuples

### tensor_ops.py

#### `crop_tensor_with_bbox(image_tensor, bbox, original_shape, resized_shape)`

Crop a tensor using a normalized bounding box, handling coordinate transformation.

**Parameters:**
- `image_tensor`: Tensor of shape [C, H, W]
- `bbox`: Normalized bbox [xmin, ymin, xmax, ymax] in [0, 1]
- `original_shape`: (height, width) before resize
- `resized_shape`: (height, width) after resize

**Returns:** Cropped tensor [C, cropped_H, cropped_W]

#### `batch_crop_tensors(image_tensor, bboxes, original_shape, resized_shape, clone=True)`

Crop multiple regions from a single tensor.

**Parameters:**
- `image_tensor`: Tensor of shape [C, H, W]
- `bboxes`: List of normalized bounding boxes
- `original_shape`: (height, width) before resize
- `resized_shape`: (height, width) after resize
- `clone`: If True, clone each crop

**Returns:** List of cropped tensors

#### `normalize_bbox(bbox, image_shape)`

Convert pixel coordinates to normalized [0, 1] coordinates.

#### `denormalize_bbox(bbox, image_shape)`

Convert normalized [0, 1] coordinates to pixel coordinates.

## Integration with Existing Code

The refactored `simple_all_gpu.py` demonstrates how these utilities integrate into a real pipeline:

1. **PDF Loading**: Uses `iter_pdf_page_tensors()` for memory-efficient page-by-page processing
2. **GPU Processing**: Tensors are loaded directly to GPU
3. **Cropping**: Uses `crop_tensor_with_bbox()` for extracting detected regions
4. **Streaming**: Generator-based approach processes pages as they're rendered

## Performance Benefits

- **Memory Efficiency**: Pages are processed one at a time
- **GPU Acceleration**: Direct loading to GPU without intermediate copies
- **Streaming**: Start processing immediately without waiting for entire PDF to load
- **Reusability**: Common operations abstracted into reusable functions
