"""
FastAPI server for PDF processing using the slim-gest pipeline.
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from torch import nn

from nemotron_page_elements_v3.model import define_model as define_model_page_elements
from nemotron_table_structure_v1.model import define_model as define_model_table_structure
from nemotron_graphic_elements_v1.model import define_model as define_model_graphic_elements
from nemotron_ocr.inference.pipeline import NemotronOCR

from slimgest.local.simple_all_gpu import run_pipeline


# Global models - initialized on startup
models = {}


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


def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import typer
    typer.run(main)
