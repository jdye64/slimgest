"""
FastAPI server for PDF processing using the slim-gest pipeline.
"""
import os
import tempfile
import shutil
import json
import asyncio
from pathlib import Path
from typing import List, Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from torch import nn

from nemotron_page_elements_v3.model import define_model as define_model_page_elements
from nemotron_table_structure_v1.model import define_model as define_model_table_structure
from nemotron_graphic_elements_v1.model import define_model as define_model_graphic_elements
from nemotron_ocr.inference.pipeline import NemotronOCR

from slimgest.local.simple_all_gpu import run_pipeline, process_pdf_pages, process_image_batch, base64_to_tensor


# Global models - initialized on startup
models = {}


# Request models for batch processing
class PageImage(BaseModel):
    page_number: int
    image_base64: str


class BatchProcessRequest(BaseModel):
    images: List[PageImage]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and cleanup on shutdown."""
    print("Loading models...")
    
    # Load all models
    models["page_elements"] = define_model_page_elements("page_element_v3")
    models["table_structure"] = define_model_table_structure("table_structure_v1")
    models["graphic_elements"] = define_model_graphic_elements("graphic_elements_v1")
    
    # Get OCR model directory from environment or use default
    ocr_model_dir = os.environ.get(
        "NEMOTRON_OCR_MODEL_DIR",
        "/home/jdyer/Development/slim-gest/models/nemotron-ocr-v1/checkpoints"
    )
    models["ocr"] = NemotronOCR(model_dir=ocr_model_dir)
    
    print(f"Models loaded:")
    print(f"  - Page Elements (device: {models['page_elements'].device})")
    print(f"  - Table Structure (device: {models['table_structure'].device})")
    print(f"  - Graphic Elements (device: {models['graphic_elements'].device})")
    print(f"  - OCR Model")
    
    yield
    
    # Cleanup (if needed)
    print("Shutting down...")
    models.clear()


app = FastAPI(
    title="Slim-Gest PDF Processing API",
    description="API for processing PDFs with OCR, table detection, and element extraction",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Slim-Gest PDF Processing API is running",
        "models_loaded": len(models) > 0,
    }


@app.post("/process-pdfs")
async def process_pdfs(
    files: List[UploadFile] = File(..., description="PDF files to process"),
    dpi: float = 150.0,
):
    """
    Process one or more PDF files through the OCR pipeline.
    
    Args:
        files: List of PDF files to process
        dpi: Resolution for PDF rendering (default: 150.0)
    
    Returns:
        JSON response with OCR results and metadata
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    # Validate that all files are PDFs
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Only PDF files are accepted."
            )
    
    # Create a temporary directory for uploaded files
    temp_dir = tempfile.mkdtemp(prefix="slimgest_")
    temp_path = Path(temp_dir)
    
    try:
        # Save uploaded files to temporary directory
        pdf_paths = []
        for file in files:
            file_path = temp_path / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            pdf_paths.append(str(file_path))
        
        # Process PDFs using the pipeline
        results = run_pipeline(
            pdf_files=pdf_paths,
            page_elements_model=models["page_elements"],
            table_structure_model=models["table_structure"],
            graphic_elements_model=models["graphic_elements"],
            ocr_model=models["ocr"],
            dpi=dpi,
            return_results=True,
        )
        
        return JSONResponse(content=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temp directory {temp_dir}: {e}")


@app.post("/process-pdf")
async def process_pdf(
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
    return await process_pdfs(files=[file], dpi=dpi)


async def process_pdf_stream_generator(
    pdf_path: str,
    dpi: float,
) -> AsyncGenerator[str, None]:
    """
    Async generator that processes PDF pages and yields SSE events.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for PDF rendering
    
    Yields:
        SSE-formatted messages for each page
    """
    try:
        # Send start event
        yield f"event: start\ndata: {json.dumps({'status': 'processing', 'pdf': Path(pdf_path).name})}\n\n"
        
        all_pages_data = []
        page_count = 0
        
        # Run the synchronous generator in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def run_sync_generator():
            """Run the synchronous generator and collect results."""
            results = []
            for page_number, tensor, page_ocr_results, page_raw_ocr_results in process_pdf_pages(
                pdf_path,
                models["page_elements"],
                models["table_structure"],
                models["graphic_elements"],
                models["ocr"],
                device="cuda",
                dpi=dpi,
            ):
                results.append((page_number, page_ocr_results, page_raw_ocr_results))
            return results
        
        # Execute in thread pool
        results = await loop.run_in_executor(None, run_sync_generator)
        
        # Yield results as SSE events
        for page_number, page_ocr_results, page_raw_ocr_results in results:
            page_count += 1
            page_text = " ".join(page_ocr_results)
            
            page_data = {
                "page_number": page_number,
                "ocr_text": page_text,
                "raw_ocr_results": page_raw_ocr_results,
            }
            all_pages_data.append(page_data)
            
            # Send page completion event
            event_data = {
                "page_number": page_number,
                "page_text": page_text,
                "total_pages_so_far": page_count,
            }
            yield f"event: page\ndata: {json.dumps(event_data)}\n\n"
            
            # Small delay to ensure events are sent
            await asyncio.sleep(0.01)
        
        # Send completion event
        completion_data = {
            "status": "complete",
            "total_pages": page_count,
            "pages": all_pages_data,
            "pdf_name": Path(pdf_path).name,
        }
        yield f"event: complete\ndata: {json.dumps(completion_data)}\n\n"
        
    except Exception as e:
        # Send error event
        error_data = {
            "status": "error",
            "error": str(e),
        }
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"


@app.post("/process-pdf-stream")
async def process_pdf_stream(
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
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.filename}. Only PDF files are accepted."
        )
    
    # Create a temporary directory for the uploaded file
    temp_dir = tempfile.mkdtemp(prefix="slimgest_stream_")
    temp_path = Path(temp_dir)
    
    try:
        # Save uploaded file to temporary directory
        file_path = temp_path / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Return streaming response
        return StreamingResponse(
            process_pdf_stream_generator(str(file_path), dpi),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    
    except Exception as e:
        # Clean up on error
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    # Note: temp_dir cleanup happens after the stream completes
    # We could use a background task to clean it up, but for now we'll leave it


async def process_batch_stream_generator(
    images: List[PageImage],
) -> AsyncGenerator[str, None]:
    """
    Async generator that processes a batch of pre-rendered page images and yields SSE events.
    
    Args:
        images: List of PageImage objects containing page numbers and base64-encoded PNG images
    
    Yields:
        SSE-formatted messages for each page
    """
    try:
        # Send start event
        yield f"event: start\ndata: {json.dumps({'status': 'processing', 'batch_size': len(images)})}\n\n"
        
        # Convert base64 images to tensors
        image_tensors = []
        page_numbers = []
        
        for page_img in images:
            try:
                tensor = base64_to_tensor(page_img.image_base64, device="cuda")
                image_tensors.append(tensor)
                page_numbers.append(page_img.page_number)
            except Exception as e:
                # Send error event for this specific image
                error_data = {
                    "status": "error",
                    "page_number": page_img.page_number,
                    "error": f"Failed to decode image: {str(e)}",
                }
                yield f"event: page_error\ndata: {json.dumps(error_data)}\n\n"
                continue
        
        if not image_tensors:
            yield f"event: error\ndata: {json.dumps({'status': 'error', 'error': 'No valid images in batch'})}\n\n"
            return
        
        # Process batch
        loop = asyncio.get_event_loop()
        
        def run_sync_batch_processing():
            """Run the synchronous batch processor and collect results."""
            results = []
            for page_number, page_ocr_results, page_raw_ocr_results in process_image_batch(
                image_tensors,
                page_numbers,
                models["page_elements"],
                models["table_structure"],
                models["graphic_elements"],
                models["ocr"],
                device="cuda",
            ):
                results.append((page_number, page_ocr_results, page_raw_ocr_results))
            return results
        
        # Execute in thread pool
        results = await loop.run_in_executor(None, run_sync_batch_processing)
        
        # Yield results as SSE events
        for page_number, page_ocr_results, page_raw_ocr_results in results:
            page_text = " ".join(page_ocr_results)
            
            # Debug: Log OCR extraction
            if not page_text:
                print(f"WARNING: Page {page_number} extracted no text. OCR results count: {len(page_ocr_results)}")
            else:
                print(f"Page {page_number}: Extracted {len(page_text)} characters from {len(page_ocr_results)} OCR blocks")
            
            page_data = {
                "page_number": page_number,
                "ocr_text": page_text,
                "raw_ocr_results": page_raw_ocr_results,
            }
            
            # Send page completion event
            yield f"event: page\ndata: {json.dumps(page_data)}\n\n"
            
            # Small delay to ensure events are sent
            await asyncio.sleep(0.001)
        
        # Send batch completion event
        completion_data = {
            "status": "complete",
            "pages_processed": len(results),
        }
        yield f"event: complete\ndata: {json.dumps(completion_data)}\n\n"
        
    except Exception as e:
        # Send error event
        error_data = {
            "status": "error",
            "error": str(e),
        }
        yield f"event: error\ndata: {json.dumps(error_data)}\n\n"


@app.post("/process-batch-stream")
async def process_batch_stream(
    batch: BatchProcessRequest = Body(..., description="Batch of page images to process"),
):
    """
    Process a batch of pre-rendered page images through the OCR pipeline with Server-Sent Events streaming.
    
    This endpoint accepts base64-encoded PNG images and streams results as each page is processed.
    The client is responsible for rendering PDF pages to images and batching them.
    
    Args:
        batch: BatchProcessRequest containing a list of page images with page numbers
    
    Returns:
        Server-Sent Events stream with page results
        
    Events:
        - start: Processing has begun
        - page: A single page has been processed
        - page_error: An error occurred processing a specific page
        - complete: All pages in the batch have been processed
        - error: A critical error occurred
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    if not batch.images:
        raise HTTPException(status_code=400, detail="No images provided in batch")
    
    if len(batch.images) > 64:
        raise HTTPException(status_code=400, detail="Batch size too large (max 64 images)")
    
    # Return streaming response
    return StreamingResponse(
        process_batch_stream_generator(batch.images),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


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
