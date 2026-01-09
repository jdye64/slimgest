# Slim-Gest FastAPI Server

This module provides a FastAPI server for processing PDF files using the slim-gest OCR pipeline.

## Features

- **Single PDF Processing**: Process one PDF at a time
- **Batch Processing**: Process multiple PDFs in a single request
- **🆕 Streaming Processing**: Stream page results as they're processed using Server-Sent Events (SSE)
- **GPU Acceleration**: Utilizes GPU models for fast processing
- **Automatic Model Loading**: Models are loaded once on startup
- **Concurrent Processing**: Support for concurrent requests

## Running the Server

### Using Python Module

```bash
python -m slimgest.web --host 0.0.0.0 --port 8000
```

### Using uvicorn directly

```bash
uvicorn slimgest.web.__main__:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check

```bash
GET /
```

Returns server status and whether models are loaded.

**Example:**
```bash
curl http://localhost:8000/
```

### Process Single PDF

```bash
POST /process-pdf
```

**Parameters:**
- `file`: PDF file (multipart/form-data)
- `dpi`: DPI for rendering (optional, default: 150.0)

**Example:**
```bash
curl -X POST http://localhost:8000/process-pdf \
  -F "file=@document.pdf" \
  -F "dpi=150"
```

### Process Multiple PDFs

```bash
POST /process-pdfs
```

**Parameters:**
- `files`: List of PDF files (multipart/form-data)
- `dpi`: DPI for rendering (optional, default: 150.0)

**Example:**
```bash
curl -X POST http://localhost:8000/process-pdfs \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf" \
  -F "dpi=150"
```

### 🆕 Process PDF with Streaming (SSE)

```bash
POST /process-pdf-stream
```

This endpoint uses Server-Sent Events (SSE) to stream results as each page is processed, allowing real-time progress monitoring.

**Parameters:**
- `file`: PDF file (multipart/form-data)
- `dpi`: DPI for rendering (optional, default: 150.0)

**SSE Events:**
- `start`: Processing has begun
- `page`: A page has been processed (includes page number and text)
- `complete`: All pages processed (includes full results)
- `error`: An error occurred

**Example (curl):**
```bash
curl -N -X POST http://localhost:8000/process-pdf-stream \
  -F "file=@document.pdf" \
  -F "dpi=150"
```

**Example Output:**
```
event: start
data: {"status": "processing", "pdf": "document.pdf"}

event: page
data: {"page_number": 1, "page_text": "Text from page 1...", "total_pages_so_far": 1}

event: page
data: {"page_number": 2, "page_text": "Text from page 2...", "total_pages_so_far": 2}

event: complete
data: {"status": "complete", "total_pages": 2, "pages": [...], "pdf_name": "document.pdf"}
```

## Response Format

The API returns a JSON response with the following structure:

```json
{
  "total_pages_processed": 5,
  "total_pdfs": 1,
  "elapsed_seconds": 12.34,
  "results": [
    {
      "pdf_path": "/tmp/slimgest_xyz/document.pdf",
      "pages_processed": 5,
      "ocr_text": "Full extracted text from all pages...",
      "raw_ocr_results": [
        "page 1 text...",
        "page 2 text...",
        ...
      ]
    }
  ]
}
```

## Test Client

A test client is provided that supports SSE streaming, directory processing, and automatic markdown generation.

### Single PDF Processing

```bash
python -m slimgest.web.test_client document.pdf --output-dir ./output
```

### Directory Processing (Concurrent)

Process all PDFs in a directory with concurrent requests (16 concurrent by default):

```bash
python -m slimgest.web.test_client ./pdfs/ --output-dir ./output
```

Adjust concurrency level:

```bash
python -m slimgest.web.test_client ./pdfs/ --output-dir ./output --workers 32
```

### Options

- `--output-dir <dir>`: Directory to save markdown files (default: ./output)
- `--dpi <dpi>`: DPI for PDF rendering (default: 150.0)
- `--workers <n>`: Max concurrent requests for directory processing (default: 16)

### Features

- **Real-time Progress**: See each page as it's processed
- **Concurrent Processing**: Process multiple PDFs simultaneously
- **Automatic Markdown Generation**: Saves OCR results as formatted markdown files
- **Error Handling**: Gracefully handles errors and continues processing

## Python Client Example

### Standard JSON Response

```python
import requests

# Process a single PDF
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/process-pdf", files=files)
    result = response.json()
    print(result["results"][0]["ocr_text"])
```

### Streaming SSE Response

```python
import requests
import json

with open("document.pdf", "rb") as f:
    files = {"file": ("document.pdf", f, "application/pdf")}
    response = requests.post(
        "http://localhost:8000/process-pdf-stream",
        files=files,
        stream=True
    )
    
    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode('utf-8')
        
        if line.startswith('event:'):
            event_type = line.split(':', 1)[1].strip()
        elif line.startswith('data:'):
            data = json.loads(line.split(':', 1)[1].strip())
            
            if event_type == 'page':
                print(f"Page {data['page_number']}: {data['page_text'][:100]}...")
            elif event_type == 'complete':
                print(f"Complete! Total pages: {data['total_pages']}")
```

## Configuration

The server currently uses a hardcoded path for the OCR model checkpoints:
```
/home/jdyer/Development/slim-gest/models/nemotron-ocr-v1/checkpoints
```

You may need to update this path in `__main__.py` to match your installation.

## Development

### Interactive API Documentation

Once the server is running, you can access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
