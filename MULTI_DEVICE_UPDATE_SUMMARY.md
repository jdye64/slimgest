# Multi-Device Support Update Summary

This document summarizes the changes made to support multiple GPU devices and device lists throughout the slimgest codebase.

## Overview

The slimgest codebase has been updated to support distributing workload across multiple GPU devices. Functions that previously accepted a single `device` parameter now accept either a single device or a list of devices, enabling automatic load balancing across multiple GPUs.

## Key Changes

### 1. PDF Rendering with Device Lists (`src/slimgest/pdf/render.py`)

#### `iter_pdf_page_tensors()`
- **Updated signature**: `device: Union[str, torch.device, list]`
- **Behavior**: When a list of devices is provided, pages are distributed round-robin across devices
- **Example**:
  ```python
  # Single device (original behavior)
  for page in iter_pdf_page_tensors("doc.pdf", device="cuda:0"):
      pass
  
  # Multiple devices (new feature)
  for page in iter_pdf_page_tensors("doc.pdf", device=["cuda:0", "cuda:1"]):
      # page 0 on cuda:0, page 1 on cuda:1, page 2 on cuda:0, etc.
      pass
  ```

#### `load_pdf_page_tensors()`
- **Updated signature**: `device: Union[str, torch.device, list]`
- **Behavior**: Delegates to `iter_pdf_page_tensors()` for device distribution
- **Example**:
  ```python
  tensors, shapes = load_pdf_page_tensors("doc.pdf", device=["cuda:0", "cuda:1"])
  # Returns list of tensors distributed across devices
  ```

### 2. In-Memory Batch Pipeline (`src/slimgest/local/in_memory.py`)

#### Major Refactoring
- **Removed hardcoded 2-GPU logic**: Previously only supported cuda:0 and cuda:1
- **Added flexible device list support**: Now supports any number of devices
- **New command-line option**: `--devices` parameter (comma-separated list)

#### `load_and_prepare_bitmaps()`
- **New parameter**: `devices: list` (instead of single `device`)
- **Behavior**: Loads PDF pages and distributes tensors across devices using round-robin
- **Example**:
  ```python
  # Automatically uses device list for load balancing
  bitmaps = load_and_prepare_bitmaps(
      input_dir, 
      dpi=150, 
      devices=["cuda:0", "cuda:1", "cuda:2"]
  )
  ```

#### `run()` Command
- **New option**: `--devices` (e.g., `--devices "cuda:0,cuda:1,cuda:2"`)
- **Auto-detection**: If `--devices` not specified, automatically detects all available CUDA devices
- **Dynamic model loading**: Loads one set of models per device
- **Parallel processing**: Creates one thread per device for parallel batch processing
- **Example**:
  ```bash
  # Use specific devices
  python -m slimgest.local.in_memory input_dir --devices "cuda:0,cuda:1"
  
  # Auto-detect all GPUs
  python -m slimgest.local.in_memory input_dir
  ```

#### Threading Architecture
- Creates one thread per device
- Each thread processes batches assigned to its device
- Batches are distributed by grouping pages that were loaded to each device
- Results and timings are aggregated from all threads

### 3. Simple All-GPU Pipeline (`src/slimgest/local/simple_all_gpu.py`)

#### `run()` Command
- **New option**: `--device` (single device parameter)
- **Behavior**: Processes PDFs sequentially using specified device
- **Example**:
  ```bash
  python -m slimgest.local.simple_all_gpu input_dir --device cuda:1
  ```

### 4. Research Scripts

#### `world_class_pdf_with_crops.py`
- **Updated**: `DocumentEngine` constructor now uses `self.device` throughout
- **New option**: `--device` command-line parameter
- **Example**:
  ```bash
  python world_class_pdf_with_crops.py input_dir --device cuda:0
  ```

#### `benchmark_pdf_pipeline.py`
- **Already supported**: Device parameter via `--device` flag
- **No changes needed**: Already properly implemented

### 5. Web Worker (`src/slimgest/web/worker.py`)

#### `PDFWorker` Class
- **New parameter**: `device: str` in constructor
- **Updated**: All model loading uses specified device
- **Purpose**: Allows web service to spawn multiple workers on different GPUs

#### `start_worker()` Function
- **New parameter**: `device: str = "cuda"`
- **Usage**: Web service can create workers like:
  ```python
  # Worker 0 on cuda:0
  worker_0 = Process(target=start_worker, args=(0, req_q, res_q, model_dir, "cuda:0"))
  
  # Worker 1 on cuda:1
  worker_1 = Process(target=start_worker, args=(1, req_q, res_q, model_dir, "cuda:1"))
  ```

## Usage Examples

### Example 1: Automatic Multi-GPU Detection

```bash
# Auto-detects all available GPUs and distributes load
python -m slimgest.local.in_memory /path/to/pdfs --batch-size 32
```

### Example 2: Specific GPU Selection

```bash
# Use only GPUs 0 and 2
python -m slimgest.local.in_memory /path/to/pdfs --devices "cuda:0,cuda:2"
```

### Example 3: Single GPU

```bash
# Use only GPU 1
python -m slimgest.local.in_memory /path/to/pdfs --devices "cuda:1"
```

### Example 4: CPU Fallback

```bash
# Use CPU if no GPUs available
python -m slimgest.local.in_memory /path/to/pdfs --devices "cpu"
```

### Example 5: Programmatic Usage

```python
from slimgest.pdf.render import iter_pdf_page_tensors

# Distribute pages across 4 GPUs
devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
for page_tensor in iter_pdf_page_tensors("large_doc.pdf", device=devices):
    # Each page automatically placed on next GPU in round-robin fashion
    process_page(page_tensor.tensor)
```

## Architecture Benefits

### 1. **Scalability**
- Support for any number of GPUs (not limited to 2)
- Easy to scale from 1 to N GPUs

### 2. **Load Balancing**
- Round-robin distribution ensures even load across devices
- Automatic batching by device minimizes cross-device transfers

### 3. **Backward Compatibility**
- Single device usage remains unchanged
- Default behavior preserved (auto-detect or "cuda")

### 4. **Flexibility**
- Mix of GPU and CPU devices possible
- Easy to assign specific workers to specific devices

### 5. **Performance**
- Parallel processing across multiple GPUs
- Reduced per-GPU memory pressure
- Better throughput for large batches

## Implementation Details

### Round-Robin Device Assignment

When loading PDF pages with multiple devices:

```python
devices = ["cuda:0", "cuda:1"]
device_idx = 0

for page in pdf_pages:
    target_device = devices[device_idx % len(devices)]
    page_tensor = page_tensor.to(target_device)
    device_idx += 1
```

### Model Instantiation Per Device

For parallel processing:

```python
models_per_device = []
for device in device_list:
    models_per_device.append({
        "page_elements": define_model(...).to(device),
        "table_structure": define_model(...).to(device),
        "ocr": NemotronOCR(..., device=device),
    })
```

### Thread-Safe Processing

Each device gets its own:
- Model instances
- Thread
- Timing structures
- Results list

Results are aggregated after all threads complete.

## Migration Guide

### Updating Existing Code

**Before:**
```python
# Old: Single device only
tensors, shapes = load_pdf_page_tensors("doc.pdf", device="cuda")
```

**After:**
```python
# New: Still works the same
tensors, shapes = load_pdf_page_tensors("doc.pdf", device="cuda")

# New: Can also use multiple devices
tensors, shapes = load_pdf_page_tensors("doc.pdf", device=["cuda:0", "cuda:1"])
```

### Updating Command-Line Scripts

**Before:**
```bash
# Old: Hardcoded 2 GPUs or single GPU
python -m slimgest.local.in_memory input_dir
```

**After:**
```bash
# New: Auto-detect all GPUs
python -m slimgest.local.in_memory input_dir

# New: Specify exact devices
python -m slimgest.local.in_memory input_dir --devices "cuda:0,cuda:1,cuda:2"
```

## Performance Considerations

### When to Use Multiple Devices

✅ **Good use cases:**
- Large batch processing (100+ pages)
- Multiple PDFs to process
- High-throughput scenarios
- Systems with multiple GPUs

❌ **Less beneficial:**
- Small batches (< 10 pages)
- Single small PDF
- Limited GPU memory per device
- Inter-GPU communication overhead concerns

### Memory Management

Each device maintains its own:
- Model weights (replicated)
- Page tensors (distributed)
- Intermediate activations

**Memory requirement per device:** ~4-6GB depending on model and batch size

### Optimal Batch Sizes

- **Single GPU**: 32-64 pages per batch
- **Multiple GPUs**: 16-32 pages per batch per GPU
- Adjust based on available GPU memory

## Testing

### Verify Multi-GPU Setup

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
```

### Test Device Distribution

```python
from slimgest.pdf.render import load_pdf_page_tensors

tensors, _ = load_pdf_page_tensors("test.pdf", device=["cuda:0", "cuda:1"])
for i, tensor in enumerate(tensors):
    print(f"Page {i}: device={tensor.device}")
```

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size or reduce number of devices

### Issue: Uneven GPU Utilization
**Solution**: Ensure batch sizes are similar across devices

### Issue: Slow Performance with Multiple GPUs
**Solution**: Check for cross-device transfers, ensure data is pre-distributed

## Files Modified

1. `src/slimgest/pdf/render.py` - Device list support for PDF loading
2. `src/slimgest/local/in_memory.py` - Complete refactor for N-device support
3. `src/slimgest/local/simple_all_gpu.py` - Device parameter option
4. `src/slimgest/research/world_class_pdf_with_crops.py` - Device parameter option
5. `src/slimgest/web/worker.py` - Device parameter for web workers

## Future Enhancements

Potential improvements for future versions:

1. **Dynamic load balancing**: Adjust distribution based on GPU utilization
2. **Heterogeneous devices**: Mix CPU and GPU devices intelligently
3. **Memory-aware scheduling**: Consider available GPU memory when distributing
4. **Pipeline optimization**: Overlap I/O, transfer, and computation
5. **Distributed training**: Extend to multi-node setups

## Conclusion

The multi-device support update provides flexible, scalable GPU utilization for the slimgest pipeline. The implementation maintains backward compatibility while enabling significant performance improvements for multi-GPU systems.
