# Client-Side Batch Processing Workflow

This document describes the enhanced PDF processing workflow where the client handles PDF rendering and page batching, sending pre-rendered images to the server for OCR processing.

## Overview

The new workflow separates concerns:
- **Client**: Handles PDF rendering, page batching, progress tracking, and metrics collection
- **Server**: Focuses on OCR processing of pre-rendered page images

This design provides better scalability, detailed progress tracking, and comprehensive performance metrics.

## Architecture

```
┌─────────────┐                                    ┌─────────────┐
│   Client    │                                    │   Server    │
│             │                                    │             │
│ 1. Load PDF │                                    │             │
│ 2. Render   │──── Batch of 32 PNG images ───────▶│ 1. Receive  │
│    pages    │       (base64 encoded)             │    batch    │
│    locally  │                                    │ 2. Run OCR  │
│             │                                    │    pipeline │
│ 3. Batch    │◀──── SSE Stream (page results) ────│ 3. Stream   │
│    pages    │                                    │    results  │
│             │                                    │             │
│ 4. Track    │                                    │             │
│    metrics  │                                    │             │
│             │                                    │             │
│ 5. Generate │                                    │             │
│    charts   │                                    │             │
└─────────────┘                                    └─────────────┘
```

## Key Features

### Client Features

1. **Local PDF Rendering**: Uses `pypdfium2` to render PDF pages to PNG images
2. **Intelligent Batching**: Groups pages into batches of 32 (configurable)
3. **Base64 Encoding**: Converts PNG images to base64 for HTTP transport
4. **Rich Progress Tracking**: Real-time display of processing status
5. **Performance Metrics**: Tracks detailed timing information
6. **Concurrent Processing**: Processes multiple PDFs simultaneously
7. **Performance Charts**: Generates visual performance graphs
8. **Detailed Reports**: Shows top 100 slowest PDFs with analysis

### Server Features

1. **Batch Endpoint**: New `/process-batch-stream` endpoint
2. **SSE Streaming**: Real-time results via Server-Sent Events
3. **Image Processing**: Accepts pre-rendered images (skips PDF rendering)
4. **Same OCR Pipeline**: Uses existing detection and OCR models

## API Changes

### New Endpoint: `/process-batch-stream`

**Method:** POST  
**Content-Type:** application/json

**Request Body:**
```json
{
  "images": [
    {
      "page_number": 1,
      "image_base64": "iVBORw0KGgo..."
    },
    {
      "page_number": 2,
      "image_base64": "iVBORw0KGgo..."
    }
  ]
}
```

**Response:** Server-Sent Events (SSE) stream

**Events:**
- `start`: Processing has begun
- `page`: A page has been processed (includes OCR text)
- `page_error`: Error processing a specific page
- `complete`: Batch processing complete
- `error`: Critical error occurred

**Example SSE Stream:**
```
event: start
data: {"status": "processing", "batch_size": 32}

event: page
data: {"page_number": 1, "ocr_text": "...", "raw_ocr_results": [...]}

event: page
data: {"page_number": 2, "ocr_text": "...", "raw_ocr_results": [...]}

event: complete
data: {"status": "complete", "pages_processed": 32}
```

## Client Usage

### Installation

First, ensure you have the required dependencies:

```bash
pip install pypdfium2 matplotlib rich requests
```

Or install from pyproject.toml:

```bash
pip install -e .
```

### Basic Usage

**Process a single PDF:**
```bash
python src/slimgest/web/test_client.py document.pdf --output-dir ./output
```

**Process a directory of PDFs:**
```bash
python src/slimgest/web/test_client.py ./pdfs/ --output-dir ./output
```

### Advanced Options

```bash
python src/slimgest/web/test_client.py <path> [options]

Options:
  --output-dir <dir>   Directory to save markdown files (default: ./output)
  --dpi <dpi>          DPI for PDF rendering (default: 150.0)
  --batch-size <n>     Pages per batch (default: 32)
  --workers <n>        Max concurrent PDFs (default: 4)
  --url <url>          Base URL of API server (default: http://localhost:7670)
```

### Examples

**High-resolution processing:**
```bash
python src/slimgest/web/test_client.py ./pdfs/ --dpi 300.0 --output-dir ./high_res_output
```

**Larger batches for better throughput:**
```bash
python src/slimgest/web/test_client.py ./pdfs/ --batch-size 64 --output-dir ./output
```

**More concurrent PDFs:**
```bash
python src/slimgest/web/test_client.py ./pdfs/ --workers 8 --output-dir ./output
```

**Custom server URL:**
```bash
python src/slimgest/web/test_client.py ./pdfs/ --url http://gpu-server:7670
```

## Performance Tracking

### Real-Time Metrics

The client displays comprehensive real-time metrics:

- **Current Processing Status**: Which PDFs are being processed
- **Total Files**: Number of PDFs queued and completed
- **Pages Processed**: Running count of pages completed
- **Bytes Read**: Total file size processed
- **Batches Sent/In-Flight**: Network activity monitoring
- **Pages per Second**: Real-time throughput calculation

### Progress Display

Uses Rich library for beautiful terminal output:
- Progress bars for overall completion
- Spinner for active processing
- Color-coded status indicators
- Time elapsed and remaining estimates

### Post-Processing Reports

After completion, the client generates:

1. **Overall Statistics Table**
   - Total PDFs and pages processed
   - Total bytes read
   - Total batches sent
   - Total time and average pages/second

2. **Top 100 Slowest PDFs Table**
   - Ranked by total processing time
   - Shows: Name, Pages, File Size, Render Time, Processing Time, Total Time
   - Helps identify problematic documents

3. **Performance Chart** (PNG image)
   - Graph of pages/second over time
   - Shows processing speed trends
   - Includes average and peak statistics
   - Saved to output directory as `performance_chart.png`

## Metrics Breakdown

### Per-PDF Metrics

For each PDF, the client tracks:

- **File Size**: Bytes
- **Total Pages**: Count
- **Render Time**: Time to convert PDF pages to PNG images
- **Processing Time**: Time for server-side OCR processing
- **Total Time**: End-to-end time
- **OCR Text**: Full extracted text

### Global Metrics

Across all PDFs:

- **Total PDFs**: Queued and completed counts
- **Total Pages**: Processed page count
- **Total Bytes**: Sum of all file sizes
- **Batches**: Sent, in-flight, and completed counts
- **Pages per Second**: Real-time and historical tracking
- **Performance History**: Time-series data for charting

## Output Files

### Markdown Files

For each PDF, a markdown file is generated containing:

```markdown
# document.pdf

**Total Pages:** 45

**File Size:** 1,234,567 bytes

**Render Time:** 12.34s

**Processing Time:** 23.45s

**Processed:** 2026-01-09 10:30:45

---

## Page 1

[OCR text for page 1]

## Page 2

[OCR text for page 2]
```

### Performance Chart

A PNG image showing processing performance over time:
- X-axis: Time (seconds)
- Y-axis: Pages per second
- Includes average and peak statistics
- Filename: `performance_chart.png`

## Performance Tuning

### Batch Size

**Default: 32 pages**

- **Smaller batches (16-32)**: Better for limited network bandwidth, more frequent updates
- **Larger batches (64)**: Better throughput, fewer HTTP requests, but larger payloads

### Concurrent Workers

**Default: 4 PDFs**

- **Lower (1-2)**: Less memory usage, simpler debugging
- **Higher (8-16)**: Better throughput with multiple GPUs or high-capacity server

### DPI Settings

**Default: 150.0**

- **Lower (72-100)**: Faster rendering, smaller images, adequate for clean documents
- **Higher (200-300)**: Better quality for scanned documents, slower rendering

### Optimization Tips

1. **Balance batch size with network latency**: Larger batches reduce overhead but increase latency
2. **Match workers to server capacity**: More workers only help if server can handle parallel requests
3. **Monitor pages/second**: Use the real-time metric to find optimal settings
4. **Check slowest PDFs**: Identify patterns (size, page count) affecting performance

## Server Configuration

### Starting the Server

```bash
python -m slimgest.web --host 0.0.0.0 --port 7670 --workers 1
```

**Note:** Use `--workers 1` for GPU workloads to avoid GPU memory conflicts.

### Environment Variables

- `NEMOTRON_OCR_MODEL_DIR`: Path to OCR model checkpoints

## Troubleshooting

### Client Issues

**Problem:** "No PDF files found"
- **Solution:** Check directory path and ensure files have `.pdf` extension

**Problem:** "Health check failed"
- **Solution:** Verify server is running and URL is correct

**Problem:** Slow rendering
- **Solution:** Lower DPI setting or check CPU resources

### Server Issues

**Problem:** "Models not loaded yet" (503 error)
- **Solution:** Wait for server startup to complete (models take time to load)

**Problem:** "Batch size too large" (400 error)
- **Solution:** Reduce `--batch-size` parameter (max 64)

**Problem:** Out of memory
- **Solution:** Reduce batch size or concurrent workers

### Network Issues

**Problem:** Timeout errors
- **Solution:** Check network connectivity, increase timeout in client code

**Problem:** Large upload times
- **Solution:** Reduce DPI or batch size to decrease payload size

## Migration from Old Client

### Old Workflow
```bash
# Client uploads entire PDF, server handles everything
python test_client.py document.pdf --url http://localhost:7670
```

### New Workflow
```bash
# Client renders pages, server processes images
python test_client.py document.pdf --url http://localhost:7670 --batch-size 32
```

### Key Differences

| Aspect | Old | New |
|--------|-----|-----|
| PDF Rendering | Server-side | Client-side |
| Page Batching | None | 32 pages (configurable) |
| Progress Tracking | Basic | Comprehensive with metrics |
| Performance Charts | None | Automatic generation |
| Concurrent PDFs | Limited | Configurable (default 4) |
| Network Efficiency | Upload full PDF | Upload batched images |

### Benefits

1. **Better Scalability**: Server focuses on OCR, clients handle rendering
2. **Detailed Metrics**: Track every aspect of processing
3. **Visual Feedback**: Rich progress bars and real-time stats
4. **Performance Analysis**: Charts and reports for optimization
5. **Flexible Batching**: Tune for your network and server capacity

## Code Structure

### Client Code Organization

```
test_client.py
├── PageMetrics: Per-page timing data
├── PDFMetrics: Per-PDF timing and results
├── GlobalMetrics: Cross-PDF statistics
├── ProgressTracker: Thread-safe metric updates
├── render_pdf_pages_to_base64(): PDF rendering with pypdfium2
├── batch_pages(): Page batching logic
├── send_batch_to_server(): HTTP communication
├── process_single_pdf(): PDF processing workflow
├── generate_performance_chart(): Chart generation
├── print_summary_report(): Final report
└── main(): CLI entry point
```

### Server Code Organization

```
__main__.py
├── PageImage: Pydantic model for page data
├── BatchProcessRequest: Pydantic model for batch request
├── process_batch_stream_generator(): SSE stream generator
└── /process-batch-stream: HTTP endpoint
```

## Future Enhancements

Potential improvements:

1. **Adaptive Batching**: Automatically adjust batch size based on performance
2. **Resume Support**: Save progress and resume interrupted sessions
3. **Distributed Rendering**: Spread PDF rendering across multiple clients
4. **Real-time Dashboard**: Web-based monitoring interface
5. **Error Recovery**: Automatic retry logic for failed batches
6. **Compression**: Compress images before base64 encoding
7. **Streaming Rendering**: Send pages as they're rendered (don't wait for full PDF)

## Benchmarks

Example performance on a typical workload:

**Configuration:**
- Server: 1x NVIDIA GPU
- Client: 8-core CPU
- Network: 1 Gbps LAN
- PDFs: 100 documents, avg 20 pages each
- DPI: 150
- Batch size: 32
- Workers: 4

**Results:**
- Total time: 180 seconds
- Average: 11.1 pages/second
- Peak: 15.3 pages/second
- Total pages: 2,000
- Total batches: 63

**Breakdown:**
- Rendering: 35% of time
- Upload: 10% of time
- OCR processing: 55% of time

## Support

For issues or questions:
1. Check this documentation
2. Review the troubleshooting section
3. Examine the performance chart for bottlenecks
4. Check the slowest PDFs report for patterns

---

**Version:** 0.2.0  
**Last Updated:** 2026-01-09
