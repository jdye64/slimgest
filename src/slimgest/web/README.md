# Slim-Gest FastAPI Server

This module provides a FastAPI server for processing PDF files using the slim-gest OCR pipeline.

## Features

- **Single PDF Processing**: Process one PDF at a time
- **Batch Processing**: Process multiple PDFs in a single request
- **GPU Acceleration**: Utilizes GPU models for fast processing
- **Automatic Model Loading**: Models are loaded once on startup

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

## Python Client Example

```python
import requests

# Process a single PDF
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/process-pdf", files=files)
    result = response.json()
    print(result["results"][0]["ocr_text"])

# Process multiple PDFs
files = [
    ("files", open("doc1.pdf", "rb")),
    ("files", open("doc2.pdf", "rb"))
]
response = requests.post("http://localhost:8000/process-pdfs", files=files)
results = response.json()
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
