#!/usr/bin/env python3
"""FastAPI server for document ingestion."""

import base64
import logging
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app declaration
app = FastAPI(
    title="Slimgest Document Ingestion Service",
    description="Service for ingesting and processing documents",
    version="0.1.0",
    docs_url="/docs",
)


@app.get("/")
async def root():
    """Root endpoint providing service information."""
    return {
        "service": "Slimgest Document Ingestion Service",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "submit": "/ingest",
            "docs": "/docs",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.post(
    "/ingest",
    responses={
        200: {"description": "Submission was successful"},
        400: {"description": "Invalid file or request"},
        500: {"description": "Error encountered during submission"},
    },
    tags=["Ingestion"],
    summary="Submit document for processing",
    operation_id="ingest",
)
async def ingest(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Submit a document for processing.
    
    This endpoint accepts a file upload and returns a job ID for tracking.
    Currently supports PDF documents.
    
    Args:
        file: The uploaded file (PDF format)
        
    Returns:
        Dict containing job_id, filename, status, and file_size
        
    Raises:
        HTTPException: If file processing fails
    """
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file to local filesystem and get its Path
        from tempfile import NamedTemporaryFile
        from pathlib import Path

        with NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            file.file.seek(0)
            while True:
                chunk = file.file.read(8192)
                if not chunk:
                    break
                tmp_file.write(chunk)
            tmp_file_path = Path(tmp_file.name)

        # Validate file size
        file_size = tmp_file_path.stat().st_size
        if file_size == 0:
            tmp_file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # tmp_file_path is a Path object pointing to the written file on disk
        
        # Determine document type from filename
        file_suffix = Path(file.filename).suffix.lower()
        supported_types = {".pdf": "pdf", ".txt": "text", ".html": "html"}
        
        document_type = supported_types.get(file_suffix, "unknown")
        if document_type == "unknown":
            logger.warning(f"Unknown file type: {file_suffix}, treating as PDF")
            document_type = "pdf"
        
        logger.info(
            f"Job {job_id}: Received file '{file.filename}' "
            f"({file_size} bytes, type: {document_type})"
        )
        
        from ingestor import process_files

        # Call the processing function with the uploaded file's location as input directory
        # We'll use the parent directory of the temp file as the data_dir and scratch/output/db as subdirs
        # This processes ALL files in the temp dir, but since this is an upload API, usually only one file will exist

        data_dir = tmp_file_path.parent
        scratch_dir = data_dir / "scratch"
        output_dir = data_dir / "extracts"
        db_path = data_dir / "lancedb"

        # The process_files function will process files matching .pdf, .txt, .html; we assume only the uploaded file is present
        process_response = process_files(
            data_dir=str(data_dir),
            scratch_dir=str(scratch_dir),
            output_dir=str(output_dir),
            db_path=str(db_path)
        )

        # Include processing response (if any) in output
        
        return {
            "job_id": job_id,
            "filename": file.filename,
            "document_type": document_type,
            "file_size": str(file_size),
            "status": "submitted",
            "message": f"File '{file.filename}' successfully submitted for processing",
            "timestamp": str(time.time()),
            "processing_response": process_response,
        }
        
    except HTTPException:
        raise
    except Exception as ex:
        logger.exception(f"Error submitting job: {str(ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(ex)}"
        )


def main():
    """Run the FastAPI server using uvicorn."""
    import uvicorn
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run the Slimgest Document Ingestion API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"],
                       help="Log level (default: info)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Slimgest Document Ingestion API server on {args.host}:{args.port}")
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
