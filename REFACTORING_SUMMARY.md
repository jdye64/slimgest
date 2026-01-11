# Refactoring Summary: Multiprocessing Architecture

## Overview

The FastAPI server has been refactored from a synchronous, blocking architecture to a multiprocessing-based, non-blocking architecture using IPC (Inter-Process Communication) queues.

## Changes Made

### New Files Created

1. **`src/slimgest/web/worker.py`**
   - `PDFWorker` class: Handles PDF processing in separate processes
   - `start_worker()` function: Entry point for worker processes
   - Loads models once per worker process
   - Processes batch and streaming requests
   - Communicates results via IPC queues

2. **`src/slimgest/web/job_manager.py`**
   - `JobManager` class: Tracks job lifecycle and state
   - `Job` dataclass: Represents a processing job
   - `JobStatus` enum: Job states (PENDING, PROCESSING, COMPLETED, FAILED, STREAMING)
   - Async interfaces for job queries and streaming events

3. **`src/slimgest/web/example_client.py`**
   - Example Python client demonstrating API usage
   - Shows batch and streaming modes
   - Health check and job status queries

4. **`src/slimgest/web/test_workers.py`**
   - Test script for validating worker setup
   - Tests IPC queues and worker startup

5. **`src/slimgest/web/README.md`**
   - Comprehensive documentation of the new architecture
   - API endpoint documentation
   - Configuration guide

### Modified Files

1. **`src/slimgest/web/__main__.py`**
   - **Before**: Loaded models in main process, blocking on processing
   - **After**: Spawns worker processes, non-blocking request handling
   
   Key changes:
   - Removed direct model loading (moved to workers)
   - Added `process_result_queue()` background task
   - Refactored `lifespan()` to manage worker processes
   - Updated all endpoints to use job submission pattern
   - Added `BackgroundTasks` for cleanup
   - Added `/jobs/{job_id}` endpoint for status queries

## Architecture Comparison

### Before (Synchronous)
```
Request → FastAPI → Load Models → Process PDF → Return Response
                    (blocking)    (blocking)
```

### After (Multiprocessing)
```
Request → FastAPI → Submit Job → Wait for Completion → Return Response
                        ↓              ↑
                   Request Queue   Result Queue
                        ↓              ↑
                   Worker Processes (async)
                   - Load Models
                   - Process PDFs
```

## Benefits

1. **Non-blocking**: FastAPI never blocks on heavy processing
2. **Concurrent**: Multiple requests processed simultaneously  
3. **Scalable**: Adjust worker count via `SLIMGEST_NUM_WORKERS`
4. **Isolated**: Worker crashes don't affect server
5. **Efficient**: Models loaded once per worker, shared across requests

## Configuration

New environment variables:
- `SLIMGEST_NUM_WORKERS`: Number of worker processes (default: 2)
- `NEMOTRON_OCR_MODEL_DIR`: OCR model directory (unchanged)

## API Changes

### Endpoints (unchanged paths, modified behavior)

- `POST /process-pdf`: Now submits job and waits for completion
- `POST /process-pdfs`: Now submits job and waits for completion  
- `POST /process-pdf-stream`: Now submits job and streams from queue

### New Endpoints

- `GET /jobs/{job_id}`: Query job status and results

### Response Format

Response formats remain the same, ensuring backward compatibility.

## Testing

Run the test suite:
```bash
python -m slimgest.web.test_workers
```

Test with example client:
```bash
python -m slimgest.web.example_client /path/to/test.pdf
```

## Performance Considerations

- **Request Queue**: Limited to 100 pending jobs
- **Timeout**: 10 minutes for batch processing
- **Workers**: Typically 1-2 per GPU to avoid memory issues
- **Cleanup**: Background tasks handle temporary file cleanup

## Migration Notes

The refactoring maintains API compatibility:
- Same endpoint paths
- Same request formats
- Same response formats

Existing clients should continue to work without modification.
