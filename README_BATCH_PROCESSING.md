# Slim-Gest Batch Processing - Complete Guide

## 🚀 Quick Start

### 1. Start the Server
```bash
python -m slimgest.web --host 0.0.0.0 --port 7670 --workers 1
```

### 2. Process Your PDFs
```bash
# Single PDF
python src/slimgest/web/test_client.py document.pdf --output-dir ./output

# Directory of PDFs with 8 concurrent workers
python src/slimgest/web/test_client.py ./pdfs/ --output-dir ./output --workers 8
```

### 3. View Your Results
- **Markdown files**: `./output/*.md` (one per PDF)
- **Performance chart**: `./output/performance_chart.png`
- **Terminal**: Detailed statistics and top 100 slowest PDFs

---

## 📋 What's New?

This enhanced version transforms the PDF processing workflow with:

✅ **Client-side PDF rendering** - Uses pypdfium2 for local page rendering  
✅ **Intelligent batching** - Groups pages into batches of 32 (configurable)  
✅ **Rich progress tracking** - Real-time metrics and beautiful terminal UI  
✅ **Performance insights** - Detailed timing breakdown and charts  
✅ **Concurrent processing** - Process multiple PDFs simultaneously  
✅ **Comprehensive reports** - Top 100 slowest PDFs with analysis  
✅ **Production-ready** - Thread-safe, error handling, and logging  

---

## 📊 What You'll See

### Real-Time Progress Display

```
Processing PDFs... ━━━━━━━━━━━━━━━━━━━━━━ 45/100 • 0:02:30 • 0:01:15

✓ technical_manual.pdf        | Pages:   87 | Time:  44.0s | 13.5 pages/s
✓ user_guide.pdf              | Pages:   42 | Time:  21.2s | 14.1 pages/s
✓ reference_doc.pdf           | Pages:  156 | Time:  78.9s | 12.8 pages/s
```

### Final Report

**Overall Statistics:**
- Total PDFs Processed: 100 / 100
- Total Pages: 2,450
- Total Time: 180.45s
- Average: 13.58 pages/second

**Top 100 Slowest PDFs:**
Ranked table showing which documents took longest, with file size and page count.

**Performance Chart:**
Visual graph of pages/second over time with statistics.

---

## 🎯 Key Features

### 1. Client-Side Processing

The client handles:
- PDF loading and page enumeration
- Rendering pages to PNG images (pypdfium2)
- Batching pages into groups
- Base64 encoding for HTTP transport
- Progress tracking and metrics collection
- Chart generation and report writing

### 2. Server-Side Processing

The server focuses on:
- Receiving batches of pre-rendered images
- Running OCR pipeline (page elements, tables, graphics, text)
- Streaming results via Server-Sent Events
- Pure OCR workload (no PDF rendering overhead)

### 3. Comprehensive Metrics

**Per-Page Metrics:**
- Render time
- Upload time
- Processing time
- Total time

**Per-PDF Metrics:**
- File size (bytes)
- Total pages
- Render time breakdown
- Processing time breakdown
- Full OCR text

**Global Metrics:**
- Total PDFs processed
- Total pages processed
- Total bytes read
- Batches sent/in-flight/completed
- Real-time pages per second
- Performance history for charting

---

## 🔧 Configuration

### Command-Line Options

```bash
python src/slimgest/web/test_client.py <path> [options]

Required:
  <path>                   PDF file or directory

Options:
  --output-dir <dir>       Output directory (default: ./output)
  --dpi <float>           Rendering DPI (default: 150.0)
  --batch-size <int>      Pages per batch (default: 32, max: 64)
  --workers <int>         Concurrent PDFs (default: 4)
  --url <url>             Server URL (default: http://localhost:7670)
```

### Performance Tuning

**For Speed:**
```bash
--workers 8 --batch-size 64 --dpi 150
```

**For Quality:**
```bash
--workers 2 --batch-size 32 --dpi 300
```

**For Stability:**
```bash
--workers 1 --batch-size 32 --dpi 150
```

---

## 📁 Project Structure

### Modified Files

- **`src/slimgest/local/simple_all_gpu.py`**
  - Added `base64_to_tensor()` function
  - Added `process_image_batch()` function
  - Supports processing pre-rendered images

- **`src/slimgest/web/__main__.py`**
  - Added `/process-batch-stream` endpoint
  - Added Pydantic models for batch requests
  - Supports SSE streaming for batch results

- **`src/slimgest/web/test_client.py`**
  - Completely rewritten with new architecture
  - Client-side PDF rendering
  - Batch processing logic
  - Rich progress tracking
  - Performance charting
  - Detailed metrics collection

- **`pyproject.toml`**
  - Added matplotlib dependency

### New Files

- **`CLIENT_BATCH_PROCESSING.md`** - Comprehensive documentation
- **`QUICK_START.md`** - Quick reference guide
- **`CHANGES_SUMMARY.md`** - Detailed change log
- **`WORKFLOW_DIAGRAM.txt`** - Visual workflow diagram
- **`examples/batch_processing_example.py`** - Programmatic usage example
- **`README_BATCH_PROCESSING.md`** - This file

---

## 🏗️ Architecture

### Old Workflow
```
Client → Upload PDF → Server → Render → Process → Stream Results → Client
```

### New Workflow
```
Client → Render Pages → Batch → Upload Images → Server → Process → Stream → Client
```

**Benefits:**
- Server focuses on OCR (its core strength)
- Client handles rendering (distributed across clients)
- Better scalability and load distribution
- Detailed metrics at every stage
- Easier to tune and optimize

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `README_BATCH_PROCESSING.md` | This file - overview and quick start |
| `QUICK_START.md` | Fast reference for common commands |
| `CLIENT_BATCH_PROCESSING.md` | Complete technical documentation |
| `CHANGES_SUMMARY.md` | Detailed change log and migration guide |
| `WORKFLOW_DIAGRAM.txt` | Visual workflow and data flow diagrams |
| `examples/batch_processing_example.py` | Code examples for programmatic use |

---

## 🔌 API Reference

### New Endpoint: `/process-batch-stream`

**POST** `/process-batch-stream`

**Request Body:**
```json
{
  "images": [
    {
      "page_number": 1,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    },
    {
      "page_number": 2,
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    }
  ]
}
```

**Response:** Server-Sent Events (text/event-stream)

**Events:**
- `start` - Processing has begun
- `page` - A page has been processed (includes OCR text)
- `page_error` - Error processing a specific page
- `complete` - Batch processing complete
- `error` - Critical error occurred

**Constraints:**
- Max 64 images per batch
- Images must be base64-encoded PNG
- Requires models to be loaded on server

---

## 💡 Usage Examples

### Basic Usage

```bash
# Process a single PDF
python src/slimgest/web/test_client.py document.pdf

# Process a directory
python src/slimgest/web/test_client.py ./pdfs/

# Specify output directory
python src/slimgest/web/test_client.py ./pdfs/ --output-dir ./results
```

### Advanced Usage

```bash
# High-resolution processing
python src/slimgest/web/test_client.py ./pdfs/ --dpi 300 --output-dir ./high_res

# Maximum throughput
python src/slimgest/web/test_client.py ./pdfs/ --batch-size 64 --workers 16

# Remote server
python src/slimgest/web/test_client.py ./pdfs/ --url http://gpu-server:7670

# Custom configuration
python src/slimgest/web/test_client.py ./pdfs/ \
  --output-dir ./output \
  --dpi 200 \
  --batch-size 48 \
  --workers 8 \
  --url http://192.168.1.100:7670
```

### Programmatic Usage

```python
from pathlib import Path
from slimgest.web.test_client import (
    process_single_pdf,
    GlobalMetrics,
    ProgressTracker,
)

# Initialize metrics
metrics = GlobalMetrics(total_pdfs=1, total_bytes=0)
metrics.start_time = time.time()
tracker = ProgressTracker(metrics)

# Process a PDF
pdf_metrics = process_single_pdf(
    pdf_path=Path("document.pdf"),
    base_url="http://localhost:7670",
    dpi=150.0,
    batch_size=32,
    tracker=tracker,
    output_dir=Path("./output"),
)

# Access results
print(f"Pages: {pdf_metrics.total_pages}")
print(f"Time: {pdf_metrics.total_time:.2f}s")
print(f"OCR text length: {len(pdf_metrics.ocr_text)}")
```

See `examples/batch_processing_example.py` for more examples.

---

## 🐛 Troubleshooting

### Common Issues

**"Health check failed"**
- Server is not running or not accessible
- Check server URL with `--url` parameter
- Verify server is listening on the correct port

**"Timeout errors"**
- Reduce `--batch-size` to lower payload size
- Increase timeout in client code if needed
- Check network connectivity

**"Out of memory"**
- Lower `--dpi` setting
- Reduce `--batch-size`
- Reduce `--workers` count
- Check available RAM

**"Models not loaded yet" (503)**
- Wait for server startup to complete
- Models take time to load on first start

**Slow processing**
- Check `pages/second` metric
- Review performance chart
- Check slowest PDFs report for patterns
- Tune `--batch-size` and `--workers`

---

## 📈 Performance Tips

### Optimal Settings

**For Most Cases:**
- DPI: 150
- Batch size: 32
- Workers: 4
- These defaults work well for typical documents

**For Speed:**
- Lower DPI (100-150)
- Larger batches (48-64)
- More workers (8-16)
- Trade some quality for throughput

**For Quality:**
- Higher DPI (200-300)
- Moderate batches (32)
- Fewer workers (2-4)
- Better for scanned documents

**For Stability:**
- Default DPI (150)
- Default batch size (32)
- Fewer workers (1-2)
- Easier debugging and monitoring

### Monitoring Performance

1. **Watch real-time pages/second** - Shows current throughput
2. **Check performance chart** - Identifies bottlenecks over time
3. **Review slowest PDFs** - Find problematic documents
4. **Monitor batch in-flight count** - Shows network saturation

---

## 🔍 Output Files

### Markdown Files

Each PDF generates a markdown file with:
- Document metadata (size, pages, times)
- Processing statistics
- Full OCR text organized by page

Example: `output/document.md`
```markdown
# document.pdf

**Total Pages:** 87
**File Size:** 4,567,890 bytes
**Render Time:** 15.3s
**Processing Time:** 28.7s
**Processed:** 2026-01-09 10:30:45

---

## Page 1

[OCR text for page 1]

## Page 2

[OCR text for page 2]
```

### Performance Chart

PNG image showing:
- Pages per second over time
- Average performance line
- Peak performance indicator
- Statistical summary

Saved as: `output/performance_chart.png`

---

## 🎓 Learning More

1. **Start with** `QUICK_START.md` for basic commands
2. **Read** `CLIENT_BATCH_PROCESSING.md` for full technical details
3. **View** `WORKFLOW_DIAGRAM.txt` for visual workflow
4. **Check** `examples/batch_processing_example.py` for code examples
5. **Review** `CHANGES_SUMMARY.md` for migration guide

---

## 🤝 Contributing

This is a production-ready implementation with:
- Clean, documented code
- Type hints throughout
- Thread-safe operations
- Comprehensive error handling
- Extensive documentation

Feel free to extend or modify for your needs!

---

## 📝 License

MIT License - See project root for details

---

## 🙏 Credits

**Technologies Used:**
- **pypdfium2** - PDF rendering
- **FastAPI** - Web server framework
- **Rich** - Terminal UI
- **matplotlib** - Performance charts
- **Nemotron OCR** - OCR models
- **PyTorch** - Deep learning framework

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the comprehensive documentation
3. Examine the performance chart for bottlenecks
4. Check the slowest PDFs report for patterns

---

**Version:** 0.2.0  
**Last Updated:** 2026-01-09  
**Status:** Production Ready ✅
