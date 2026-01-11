"""
FastAPI server for PDF processing using the slim-gest pipeline.
Uses multiprocessing workers for non-blocking request handling.
"""
import os
import tempfile
import shutil
import json
import asyncio
from pathlib import Path
from typing import List, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from multiprocessing import Queue, Process

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from slimgest.web.worker import start_worker
from slimgest.web.job_manager import JobManager, JobStatus


# Global state
worker_processes = []
request_queue = None
result_queue = None
job_manager = None
result_processor_task = None


async def process_result_queue():
    """Background task to process results from worker processes."""
    global result_queue, job_manager
    
    print("[RESULT-PROCESSOR] Starting...")
    
    while True:
        try:
            # Check if there are any results (non-blocking)
            if not result_queue.empty():
                result = result_queue.get_nowait()
                
                job_id = result["job_id"]
                print(f"[RESULT-PROCESSOR] Received result for job {job_id}")
                
                # Handle different result types
                if "type" in result:
                    # Streaming result
                    result_type = result["type"]
                    print(f"[RESULT-PROCESSOR] Streaming result type: {result_type}")
                    
                    if result_type == "start":
                        await job_manager.update_job_status(job_id, JobStatus.STREAMING)
                        await job_manager.queue_stream_event(job_id, result)
                        
                    elif result_type == "page":
                        await job_manager.queue_stream_event(job_id, result)
                        
                    elif result_type == "complete":
                        await job_manager.set_job_result(job_id, result["data"])
                        await job_manager.queue_stream_event(job_id, result)
                        
                    elif result_type == "error":
                        error_msg = result["data"].get("error", "Unknown error")
                        print(f"[RESULT-PROCESSOR] Job {job_id} streaming error: {error_msg}")
                        await job_manager.set_job_error(job_id, error_msg)
                        await job_manager.queue_stream_event(job_id, result)
                else:
                    # Batch result
                    print(f"[RESULT-PROCESSOR] Batch result status: {result['status']}")
                    if result["status"] == "success":
                        print(f"[RESULT-PROCESSOR] Job {job_id} succeeded")
                        await job_manager.set_job_result(job_id, result["results"])
                    else:
                        error_msg = result.get("error", "Unknown error")
                        print(f"[RESULT-PROCESSOR] Job {job_id} failed: {error_msg}")
                        if "traceback" in result:
                            print(f"[RESULT-PROCESSOR] Traceback:\n{result['traceback']}")
                        await job_manager.set_job_error(job_id, error_msg)
            
            # Small sleep to prevent busy waiting
            await asyncio.sleep(0.01)
            
        except Exception as e:
            print(f"[RESULT-PROCESSOR] ERROR: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(0.1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start worker processes on startup and cleanup on shutdown."""
    global worker_processes, request_queue, result_queue, job_manager, result_processor_task
    
    print("Starting PDF processing service...")
    
    # Initialize job manager
    job_manager = JobManager()
    
    # Create IPC queues
    request_queue = Queue(maxsize=100)
    result_queue = Queue()
    
    # Get configuration from environment
    num_workers = int(os.environ.get("SLIMGEST_NUM_WORKERS", "2"))
    ocr_model_dir = os.environ.get(
        "NEMOTRON_OCR_MODEL_DIR",
        "/home/jdyer/Development/slim-gest/models/nemotron-ocr-v1/checkpoints"
    )
    
    print(f"Starting {num_workers} worker processes...")
    
    # Start worker processes
    for i in range(num_workers):
        p = Process(
            target=start_worker,
            args=(i, request_queue, result_queue, ocr_model_dir),
            daemon=True,
        )
        p.start()
        worker_processes.append(p)
        print(f"Started worker {i} (PID: {p.pid})")
    
    # Start result processor task
    result_processor_task = asyncio.create_task(process_result_queue())
    
    print(f"Service ready with {num_workers} workers")
    
    yield
    
    # Cleanup
    print("Shutting down...")
    
    # Cancel result processor
    if result_processor_task:
        result_processor_task.cancel()
        try:
            await result_processor_task
        except asyncio.CancelledError:
            pass
    
    # Send shutdown signal to workers
    for _ in worker_processes:
        request_queue.put({"type": "shutdown"})
    
    # Wait for workers to finish (with timeout)
    import time
    timeout = 10
    start = time.time()
    for p in worker_processes:
        remaining = max(0, timeout - (time.time() - start))
        p.join(timeout=remaining)
        if p.is_alive():
            print(f"Worker {p.pid} did not shut down gracefully, terminating...")
            p.terminate()
    
    worker_processes.clear()
    print("Shutdown complete")


app = FastAPI(
    title="Slim-Gest PDF Processing API",
    description="API for processing PDFs with OCR, table detection, and element extraction",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check endpoint."""
    alive_workers = sum(1 for p in worker_processes if p.is_alive())
    return {
        "status": "ok",
        "message": "Slim-Gest PDF Processing API is running",
        "workers": {
            "total": len(worker_processes),
            "alive": alive_workers,
        },
        "queue_size": request_queue.qsize() if request_queue else 0,
    }


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a job.
    
    Args:
        job_id: Job ID to check
        
    Returns:
        Job status information
    """
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "has_result": job.result is not None,
        "error": job.error,
    }


@app.post("/process-pdfs")
async def process_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="PDF files to process"),
    dpi: float = 150.0,
):
    """
    Process one or more PDF files through the OCR pipeline.
    
    This endpoint submits the job to worker processes and waits for completion.
    
    Args:
        files: List of PDF files to process
        dpi: Resolution for PDF rendering (default: 150.0)
    
    Returns:
        JSON response with OCR results and metadata
    """
    if not worker_processes or not any(p.is_alive() for p in worker_processes):
        raise HTTPException(status_code=503, detail="No workers available")

    print(f"[FASTAPI] Processing {len(files)} files")

    print(f"[FASTAPI] Files: {files}")
    # Validate that all files are PDFs
    for file in files:
        print(f"[FASTAPI] Validating file: {file.filename}")
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Only PDF files are accepted."
            )
    
    # Create a temporary directory for uploaded files
    temp_dir = tempfile.mkdtemp(prefix="slimgest_")
    print(f"[FASTAPI] Created temporary directory: {temp_dir}")
    temp_path = Path(temp_dir)
    print(f"[FASTAPI] Temporary directory: {temp_dir}")
    
    try:
        # Save uploaded files to temporary directory
        pdf_paths = []
        print(f"[FASTAPI] Saving {len(files)} uploaded file(s) to temp directory: {temp_dir}")
        for file in files:
            # Extract just the basename to avoid path issues when client sends full paths
            import os as _check_os
            filename_base = _check_os.path.basename(file.filename)
            file_path = temp_path / filename_base
            
            print(f"[FASTAPI] Processing file: {file.filename}")
            print(f"[FASTAPI]   - Original filename: {file.filename}")
            print(f"[FASTAPI]   - Basename: {filename_base}")
            print(f"[FASTAPI]   - Save path: {file_path}")
            
            # Read content from the uploaded file
            content = await file.read()
            content_size = len(content)
            print(f"[FASTAPI]   - Content read from upload: {content_size} bytes")
            
            # Write to local temporary file
            with open(file_path, "wb") as f:
                f.write(content)
            
            pdf_paths.append(str(file_path))
            
            # Verify file was saved correctly
            file_exists = _check_os.path.exists(file_path)
            file_size = _check_os.path.getsize(file_path) if file_exists else 0
            print(f"[FASTAPI] Saved to {file_path}")
            print(f"[FASTAPI]   - File exists: {file_exists}")
            print(f"[FASTAPI]   - File size: {file_size} bytes")
            print(f"[FASTAPI]   - Absolute path: {_check_os.path.abspath(file_path)}")
        
        # Create job
        job_id = job_manager.create_job(job_type="batch")
        print(f"[FASTAPI] Created job {job_id}")
        print(f"[FASTAPI] Submitting to worker queue with paths: {pdf_paths}")
        print(f"[FASTAPI] Current queue size: {request_queue.qsize()}")

        # Submit to worker queue
        request_queue.put({
            "job_id": job_id,
            "type": "batch",
            "pdf_paths": pdf_paths,
            "dpi": dpi,
        })
        print(f"[FASTAPI] Job {job_id} submitted to queue")
        
        await job_manager.update_job_status(job_id, JobStatus.PROCESSING)
        print(f"[FASTAPI] Waiting for job {job_id} to complete...")
        
        # Wait for completion (with timeout)
        try:
            job = await job_manager.wait_for_completion(job_id, timeout=600)  # 10 minute timeout
            print(f"[FASTAPI] Job {job_id} completed with status: {job.status}")
        except asyncio.TimeoutError:
            print(f"[FASTAPI] Job {job_id} timed out!")
            raise HTTPException(status_code=504, detail="Processing timed out")
        
        # Schedule cleanup
        background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
        
        # Return results
        if job.status == JobStatus.COMPLETED:
            print(f"[FASTAPI] Job {job_id} succeeded, returning results")
            return JSONResponse(content=job.result)
        else:
            print(f"[FASTAPI] Job {job_id} failed with error: {job.error}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {job.error}")
    
    except HTTPException:
        # Clean up on HTTP errors
        print(f"[FASTAPI] HTTPException occurred, cleaning up temp dir: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    
    except Exception as e:
        # Clean up on other errors
        print(f"[FASTAPI] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")


@app.post("/process-pdf")
async def process_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to process"),
    dpi: float = 150.0,
):
    """
    Process a single PDF file through the OCR pipeline.
    
    Args:
        file: PDF file to process
        dpi: Resolution for PDF rendering (default: 150.0)
    
    Returns:
        JSON response with OCR results and metadata
    """
    # Use the multi-file endpoint with a single file
    return await process_pdfs(background_tasks=background_tasks, files=[file], dpi=dpi)


async def process_pdf_stream_generator(
    job_id: str,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE events from worker process.
    
    Args:
        job_id: Job ID to stream events for
    
    Yields:
        SSE-formatted messages for each page
    """
    try:
        # Stream events from the job manager
        async for event in job_manager.get_stream_events(job_id):
            event_type = event["type"]
            event_data = event["data"]
            
            # Format as SSE
            yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
            
    except Exception as e:
        # Send error event
        error_data = {
            "status": "error",
            "error": str(e),
        }
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"


@app.post("/process-pdf-stream")
async def process_pdf_stream(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to process"),
    dpi: float = 150.0,
):
    """
    Process a single PDF file through the OCR pipeline with Server-Sent Events streaming.
    
    This endpoint streams results as each page is processed, allowing clients to receive
    incremental updates instead of waiting for all pages to complete.
    
    Args:
        file: PDF file to process
        dpi: Resolution for PDF rendering (default: 150.0)
    
    Returns:
        Server-Sent Events stream with page results
        
    Events:
        - start: Processing has begun
        - page: A single page has been processed
        - complete: All pages have been processed
        - error: An error occurred
    """
    if not worker_processes or not any(p.is_alive() for p in worker_processes):
        raise HTTPException(status_code=503, detail="No workers available")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.filename}. Only PDF files are accepted."
        )
    
    # Create a temporary directory for the uploaded file
    temp_dir = tempfile.mkdtemp(prefix="slimgest_stream_")
    temp_path = Path(temp_dir)
    print(f"[FASTAPI-STREAM] Created temporary directory: {temp_dir}")
    
    try:
        # Save uploaded file to temporary directory
        # Extract just the basename to avoid path issues when client sends full paths
        import os as _check_os
        filename_base = _check_os.path.basename(file.filename)
        file_path = temp_path / filename_base
        
        print(f"[FASTAPI-STREAM] Processing file: {file.filename}")
        print(f"[FASTAPI-STREAM]   - Original filename: {file.filename}")
        print(f"[FASTAPI-STREAM]   - Basename: {filename_base}")
        print(f"[FASTAPI-STREAM]   - Save path: {file_path}")
        
        # Read content from the uploaded file
        content = await file.read()
        content_size = len(content)
        print(f"[FASTAPI-STREAM]   - Content read from upload: {content_size} bytes")
        
        # Write to local temporary file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Verify file was saved correctly
        file_exists = _check_os.path.exists(file_path)
        file_size = _check_os.path.getsize(file_path) if file_exists else 0
        print(f"[FASTAPI-STREAM] Saved to {file_path}")
        print(f"[FASTAPI-STREAM]   - File exists: {file_exists}")
        print(f"[FASTAPI-STREAM]   - File size: {file_size} bytes")
        print(f"[FASTAPI-STREAM]   - Absolute path: {_check_os.path.abspath(file_path)}")
        
        # Create job
        job_id = job_manager.create_job(job_type="stream")
        print(f"[FASTAPI-STREAM] Created job {job_id}")
        print(f"[FASTAPI-STREAM] Submitting to worker queue with path: {file_path}")
        
        # Submit to worker queue
        request_queue.put({
            "job_id": job_id,
            "type": "stream",
            "pdf_path": str(file_path),
            "dpi": dpi,
        })
        print(f"[FASTAPI-STREAM] Job {job_id} submitted to queue")
        
        await job_manager.update_job_status(job_id, JobStatus.PROCESSING)
        
        # Schedule cleanup after streaming completes
        background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
        
        # Return streaming response
        return StreamingResponse(
            process_pdf_stream_generator(job_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    
    except Exception as e:
        # Clean up on error
        print(f"[FASTAPI-STREAM] Error occurred: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


def main(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8000)
        workers: Number of worker processes (default: 1). Use 1 for GPU workloads.
    """
    # Note: For GPU workloads, workers should typically be 1 to avoid GPU memory issues
    # For CPU-only or if models are small enough, more workers can improve concurrency
    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == "__main__":
    import typer
    typer.run(main)
