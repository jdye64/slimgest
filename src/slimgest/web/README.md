# Slim-Gest Web Service

FastAPI-based web service for PDF processing using the Slim-Gest pipeline with multiprocessing workers for concurrent request handling.

## Architecture

The service uses a multiprocessing architecture to handle concurrent requests efficiently:

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
│  - Receives HTTP requests                                   │
│  - Creates jobs and manages job lifecycle                   │
│  - Non-blocking request handling                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ├─► Request Queue (IPC)
                  │
    ┌─────────────┴──────────────┬──────────────┬─────────────┐
    │                            │              │             │
    ▼                            ▼              ▼             ▼
┌─────────┐                ┌─────────┐    ┌─────────┐   ┌─────────┐
│ Worker 0│                │ Worker 1│    │ Worker 2│   │ Worker N│
│  - GPU  │                │  - GPU  │    │  - GPU  │   │  - GPU  │
│  - OCR  │                │  - OCR  │    │  - OCR  │   │  - OCR  │
│ Models  │                │ Models  │    │ Models  │   │ Models  │
└────┬────┘                └────┬────┘    └────┬────┘   └────┬────┘
     │                          │              │             │
     └──────────────┬───────────┴──────────────┴─────────────┘
                    │
                    ▼
              Result Queue (IPC)
                    │
                    ▼
          ┌─────────────────────┐
          │  Result Processor   │
          │  - Updates jobs     │
          │  - Streams events   │
          └─────────────────────┘
```

### Key Components

1. **FastAPI Server** (`__main__.py`):
   - Handles HTTP requests asynchronously
   - Manages job lifecycle and state
   - Submits work to worker processes via IPC queues
   - Returns immediately without blocking on processing

2. **Worker Processes** (`worker.py`):
   - Each worker loads its own copy of the ML models
   - Processes PDFs independently
   - Communicates results back via result queue
   - Supports batch and streaming processing modes

3. **Job Manager** (`job_manager.py`):
   - Tracks job state and results
   - Provides async interfaces for job status queries
   - Manages streaming event queues for SSE endpoints

4. **IPC Queues**:
   - `request_queue`: FastAPI → Workers (job requests)
   - `result_queue`: Workers → FastAPI (processing results)

## Configuration

Environment variables:

- `SLIMGEST_NUM_WORKERS`: Number of worker processes (default: 2)
- `NEMOTRON_OCR_MODEL_DIR`: Path to OCR model directory

## API Endpoints

### Health Check

```bash
GET /
```

Returns service status and worker information.

### Process PDF (Batch)

```bash
POST /process-pdf
POST /process-pdfs
```

Process one or more PDFs and return complete results. The server waits for processing to complete before returning.

**Parameters:**
- `file` or `files`: PDF file(s) to process
- `dpi`: Resolution for PDF rendering (default: 150.0)

**Response:**
```json
{
  "pdfs": [
    {
      "filename": "document.pdf",
      "pages": [...],
      ...
    }
  ]
}
```

### Process PDF (Streaming)

```bash
POST /process-pdf-stream
```

Process a PDF and stream results as Server-Sent Events. Results are sent as each page is processed.

**Parameters:**
- `file`: PDF file to process
- `dpi`: Resolution for PDF rendering (default: 150.0)

**Response:** Server-Sent Events stream

**Events:**
- `start`: Processing has begun
- `page`: A page has been processed
- `complete`: All pages processed
- `error`: An error occurred

### Check Job Status

```bash
GET /jobs/{job_id}
```

Get the status of a specific job.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "created_at": "2026-01-10T...",
  "completed_at": "2026-01-10T...",
  "has_result": true,
  "error": null
}
```

## Running the Service

### Start the server

```bash
# Using the module directly
python -m slimgest.web

# Or with custom settings
SLIMGEST_NUM_WORKERS=4 python -m slimgest.web --port 8000 --host 0.0.0.0
```

### Using the example client

```bash
python -m slimgest.web.example_client /path/to/document.pdf
```

## Benefits of Multiprocessing Architecture

1. **Non-blocking**: FastAPI server never blocks on heavy processing
2. **Concurrent**: Multiple requests can be processed simultaneously
3. **Scalable**: Add more workers to handle more load
4. **Isolated**: Each worker has its own GPU memory and model instances
5. **Resilient**: Worker crashes don't affect the server or other workers
6. **Fair**: Request queue ensures FIFO processing

## Performance Considerations

- **Workers per GPU**: Typically 1-2 workers per GPU to avoid memory issues
- **Queue size**: Limited to 100 pending requests to prevent memory exhaustion
- **Timeouts**: Batch requests have a 10-minute timeout
- **Cleanup**: Background tasks handle temporary file cleanup

## Development

Run with auto-reload:

```bash
uvicorn slimgest.web.__main__:app --reload --port 8000
```

Note: Auto-reload may not work well with multiprocessing. For development, consider using `workers=1`.
