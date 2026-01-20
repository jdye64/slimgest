from pathlib import Path
from typing import List, Tuple, Optional
from rich.console import Console
from rich.traceback import install
from torch import nn
import torch
import time
import json
import io
from PIL import Image
import numpy as np

from slimgest.model.local.nemotron_page_elements_v3 import NemotronPageElementsV3
from slimgest.model.local.nemotron_table_structure_v1 import NemotronTableStructureV1
from slimgest.model.local.nemotron_graphic_elements_v1 import NemotronGraphicElementsV1
from slimgest.model.local.nemotron_ocr_v1 import NemotronOCRV1

import typer

# Import our new PDF processing utilities
from slimgest.pdf.render import iter_pdf_page_tensors
from slimgest.pdf.tensor_ops import crop_tensor_with_bbox

app = typer.Typer(help="Simpliest pipeline with limited CPU parallelism while using maximum GPU possible")
install(show_locals=False)
console = Console()


def process_pdf_pages(
    pdf_path,
    page_elements,
    table_structure,
    graphic_elements,
    ocr,
    device="cuda",
    dpi=150.0,
):
    """
    Generator that processes PDF pages one at a time, yielding results for each page.
    
    This is memory-efficient as it processes pages as they're rendered without loading
    all pages into memory at once.
    
    Yields:
        Tuple of (page_number, processed_tensor, ocr_results, raw_ocr_results)
    """
    
    # Use the new generator to iterate through PDF pages
    for page_tensor_info in iter_pdf_page_tensors(pdf_path, dpi=dpi, device=device):
        page_number = page_tensor_info.page_number
        tensor = page_tensor_info.tensor  # Shape: [3, H, W]
        bitmap_shape = (page_tensor_info.original_height, page_tensor_info.original_width)
        
        page_ocr_results = []
        page_raw_ocr_results = []
        
        with torch.inference_mode():
            resized_tensor = page_elements.preprocess(tensor)
            preds = page_elements.invoke(resized_tensor, bitmap_shape)
            boxes, labels, scores = page_elements.postprocess(preds)
            
            # Process detected elements (tables and graphics)
            for label, box in zip(labels, boxes):
                if label == 0:  # Table
                    cropped = crop_tensor_with_bbox(
                        resized_tensor, box, bitmap_shape, table_structure.input_shape
                    ).clone()
                    crop_shape = (cropped.shape[1], cropped.shape[2])
                    input_tensor = table_structure.preprocess(cropped, crop_shape)
                    table_preds = table_structure.invoke(input_tensor, crop_shape)
                elif label in [1, 2, 3]:  # Graphic elements
                    cropped = crop_tensor_with_bbox(
                        resized_tensor, box, bitmap_shape, table_structure.input_shape
                    ).clone()
                    crop_shape = (cropped.shape[1], cropped.shape[2])
                    input_tensor = graphic_elements.preprocess(cropped)
                    graphic_preds = graphic_elements.invoke(input_tensor, crop_shape)
            
            # Run OCR on the original (un-resized) tensor
            ocr_preds = ocr.invoke(tensor)

            if isinstance(ocr_preds, list):
                for pred in ocr_preds:
                    page_ocr_results.append(str(pred['text']))
                    page_raw_ocr_results.append(str(pred))
            else:
                page_ocr_results.append(str(ocr_preds['text']))
                page_raw_ocr_results.append(str(ocr_preds))
        
        yield page_number, resized_tensor, page_ocr_results, page_raw_ocr_results

def run_pipeline(
    pdf_files: List[str],
    page_elements: NemotronPageElementsV3,
    table_structure: NemotronTableStructureV1,
    graphic_elements: NemotronGraphicElementsV1,
    ocr: NemotronOCRV1,
    raw_output_dir: Optional[Path] = None,
    dpi: float = 150.0,
):
    start_time = time.time()
    total_pages_processed = 0
    results = []
    
    for pdf_idx, pdf_path in enumerate(pdf_files, start=1):
        console.print(f"[bold cyan]Processing:[/bold cyan] {pdf_path}")
        
        # Collect results for this PDF
        all_page_ocr_results = []
        all_page_raw_ocr_results = []
        pages_in_pdf = 0
        
        # Process pages one at a time using the generator
        for page_number, tensor, page_ocr_results, page_raw_ocr_results in process_pdf_pages(
            pdf_path,
            page_elements,
            table_structure,
            graphic_elements,
            ocr,
            device="cuda",
            dpi=dpi,
        ):
            pages_in_pdf += 1
            total_pages_processed += 1
            
            # Collect OCR results
            all_page_ocr_results.extend(page_ocr_results)
            all_page_raw_ocr_results.extend(page_raw_ocr_results)
            
            # Show progress
            console.print(
                f"  Processed page {page_number} | "
                f"Tensor shape: {list(tensor.shape)} | "
                f"Device: {tensor.device}"
            )
        
        # Summary for this PDF
        console.print(
            f"Completed {pages_in_pdf} pages from {pdf_path}. "
            f"PDF {pdf_idx} of {len(pdf_files)}. "
            f"Total pages processed: {total_pages_processed}. "
            f"Current Runtime: {time.time() - start_time:.2f} seconds"
        )
        
        # Combine all OCR results for this PDF
        ocr_final_result = " ".join(all_page_ocr_results)
        console.print(f"OCR final result: {ocr_final_result}", markup=False)
        
        # Save raw OCR results if requested
        if raw_output_dir is not None:
            raw_output_dir = Path(raw_output_dir)
            raw_output_dir.mkdir(parents=True, exist_ok=True)
            pdf_path_obj = Path(pdf_path)
            output_json_path = raw_output_dir / pdf_path_obj.with_suffix('.page_raw_ocr_results.json').name
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(all_page_raw_ocr_results, f, ensure_ascii=False, indent=2)
            console.print(f"Saved page_raw_ocr_results to {output_json_path}")
        
        # Store results for this PDF
        results.append({
            "pdf_path": pdf_path,
            "pages_processed": pages_in_pdf,
            "ocr_text": ocr_final_result,
            "raw_ocr_results": all_page_raw_ocr_results,
        })
    
    elapsed = time.time() - start_time
    console.print(
        f"[bold green]Processed {total_pages_processed} pages from {len(pdf_files)} PDF(s) "
        f"in {elapsed:.2f} seconds[/bold green]"
    )
    
    return {
            "total_pages_processed": total_pages_processed,
            "total_pdfs": len(pdf_files),
            "elapsed_seconds": elapsed,
            "results": results,
        }

@app.command()
def run(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=True),
    raw_output_dir: Optional[Path] = typer.Option(None, help="Directory to save raw OCR results (optional)."),
):
    # Load Models
    page_elements = NemotronPageElementsV3()
    table_structure = NemotronTableStructureV1()
    graphic_elements = NemotronGraphicElementsV1()
    ocr = NemotronOCRV1(model_dir="/home/jdyer/Development/slim-gest/models/nemotron-ocr-v1/checkpoints")

    
    if input_dir.is_file():
        pdf_files = [input_dir]
    else:
        pdf_files = [
            str(f) for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".pdf"
        ]

    console.print(f"Processing {len(pdf_files)} PDFs")
    console.print(f"Using page_elements_model device: {page_elements.model.device}")
    # console.print(f"Using table_structure_model device: {table_structure_model.device}")
    # console.print(f"Using graphic_elements_model device: {graphic_elements_model.device}")

    run_pipeline(
        pdf_files,
        page_elements,
        table_structure,
        graphic_elements,
        ocr,
        raw_output_dir=raw_output_dir,
    )

    console.print("[bold green]Done![/bold green]")