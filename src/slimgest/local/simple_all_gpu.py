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

from nemotron_page_elements_v3.model import define_model as define_model_page_elements
from nemotron_page_elements_v3.model import resize_pad as resize_pad_page_elements
from nemotron_page_elements_v3.utils import postprocess_preds_page_element as postprocess_preds_page_element
from nemotron_table_structure_v1.model import define_model as define_model_table_structure
from nemotron_table_structure_v1.model import resize_pad as resize_pad_table_structure
from nemotron_graphic_elements_v1.model import define_model as define_model_graphic_elements
from nemotron_graphic_elements_v1.model import resize_pad as resize_pad_graphic_elements
from nemotron_ocr.inference.pipeline import NemotronOCR

import typer
import base64

# Import our new PDF processing utilities
from slimgest.pdf.render import iter_pdf_page_tensors
from slimgest.pdf.tensor_ops import crop_tensor_with_bbox

app = typer.Typer(help="Simpliest pipeline with limited CPU parallelism while using maximum GPU possible")
install(show_locals=False)
console = Console()


def tensor_to_pil_image(tensor):
    """
    Converts a 3xHxW torch tensor [0,1] or [0,255] on cpu/gpu to PIL Image (RGB).
    Assumes tensor is [C,H,W] and in standard format, does NOT do normalization undoing.
    """
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    # Clamp and convert to uint8
    arr = tensor.detach().numpy()
    if arr.max() <= 1.0:
        arr = arr * 255.
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # HWC
    img = Image.fromarray(arr, mode='RGB')
    return img


def base64_to_tensor(base64_str: str, device: str = "cuda") -> torch.Tensor:
    """
    Convert a base64-encoded PNG image to a torch tensor [3, H, W].
    
    Args:
        base64_str: Base64-encoded PNG image string
        device: Device to place the tensor on
    
    Returns:
        Tensor of shape [3, H, W] with values in range [0, 255]
    """
    # Decode base64 to bytes
    img_bytes = base64.b64decode(base64_str)
    
    # Load as PIL Image
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # Convert to numpy array [H, W, 3]
    arr = np.array(img, dtype=np.float32)
    
    # Convert to tensor [3, H, W]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    
    # Move to device
    tensor = tensor.to(device)
    
    return tensor

def process_image_batch(
    image_tensors: List[torch.Tensor],
    page_numbers: List[int],
    page_elements_model,
    table_structure_model,
    graphic_element_model,
    ocr_model,
    device="cuda",
):
    """
    Process a batch of pre-rendered page images (as tensors).
    
    Args:
        image_tensors: List of image tensors [3, H, W] already on the specified device
        page_numbers: List of page numbers corresponding to each tensor
        page_elements_model, table_structure_model, graphic_element_model, ocr_model: Models to use
        device: Device to run inference on
    
    Yields:
        Tuple of (page_number, ocr_results, raw_ocr_results) for each image
    """
    page_elements_input_shape = (1024, 1024)
    table_structure_input_shape = (1024, 1024)
    graphic_elements_input_shape = (1024, 1024)
    
    for tensor, page_number in zip(image_tensors, page_numbers):
        tensor = tensor.to(device)
        bitmap_shape = (tensor.shape[1], tensor.shape[2])  # H, W from [3, H, W]
        
        page_ocr_results = []
        page_raw_ocr_results = []
        
        with torch.inference_mode():
            # Resize for page elements detection
            resized_tensor = resize_pad_page_elements(tensor, page_elements_input_shape)
            preds = page_elements_model(resized_tensor, bitmap_shape)[0]
            boxes, labels, scores = postprocess_preds_page_element(
                preds, page_elements_model.thresholds_per_class, page_elements_model.labels
            )
            
            # Process detected elements (tables and graphics)
            for label, box in zip(labels, boxes):
                if label == 0:  # Table
                    cropped = crop_tensor_with_bbox(
                        resized_tensor, box, bitmap_shape, page_elements_input_shape
                    ).clone()
                    crop_shape = (cropped.shape[1], cropped.shape[2])
                    cropped_resized = resize_pad_table_structure(cropped, table_structure_input_shape)
                    table_preds = table_structure_model(cropped_resized, crop_shape)[0]
                    
                elif label in [1, 2, 3]:  # Graphic elements
                    cropped = crop_tensor_with_bbox(
                        resized_tensor, box, bitmap_shape, page_elements_input_shape
                    ).clone()
                    crop_shape = (cropped.shape[1], cropped.shape[2])
                    cropped_resized = resize_pad_graphic_elements(cropped, graphic_elements_input_shape)
                    graphic_preds = graphic_element_model(cropped_resized, crop_shape)[0]
            
            # Run OCR on the tensor
            print(f"  Running OCR on page {page_number}, tensor shape: {tensor.shape}, device: {tensor.device}")
            ocr_preds = ocr_model(tensor.clone().to(device="cuda"))
            print(f"  OCR returned {len(ocr_preds) if ocr_preds else 0} predictions")

            for pred in ocr_preds:
                text = str(pred.get('text', ''))
                if text:
                    page_ocr_results.append(text)
                page_raw_ocr_results.append(str(pred))
            
            print(f"  Page {page_number}: Extracted {len(page_ocr_results)} text blocks, total {sum(len(t) for t in page_ocr_results)} characters")
        
        yield page_number, page_ocr_results, page_raw_ocr_results


def process_pdf_pages(
    pdf_path,
    page_elements_model,
    table_structure_model,
    graphic_element_model,
    ocr_model,
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
    page_elements_input_shape = (1024, 1024)
    table_structure_input_shape = (1024, 1024)
    graphic_elements_input_shape = (1024, 1024)
    
    # Use the new generator to iterate through PDF pages
    for page_tensor_info in iter_pdf_page_tensors(pdf_path, dpi=dpi, device=device):
        page_number = page_tensor_info.page_number
        tensor = page_tensor_info.tensor  # Shape: [3, H, W]
        bitmap_shape = (page_tensor_info.original_height, page_tensor_info.original_width)
        
        # Keep a reference to the original tensor for OCR
        original_tensor = tensor
        
        page_ocr_results = []
        page_raw_ocr_results = []
        
        with torch.inference_mode():
            # Resize for page elements detection
            resized_tensor = resize_pad_page_elements(tensor, page_elements_input_shape)
            preds = page_elements_model(resized_tensor, bitmap_shape)[0]
            boxes, labels, scores = postprocess_preds_page_element(
                preds, page_elements_model.thresholds_per_class, page_elements_model.labels
            )
            
            # Process detected elements (tables and graphics)
            for label, box in zip(labels, boxes):
                if label == 0:  # Table
                    cropped = crop_tensor_with_bbox(
                        resized_tensor, box, bitmap_shape, page_elements_input_shape
                    ).clone()
                    crop_shape = (cropped.shape[1], cropped.shape[2])
                    cropped_resized = resize_pad_table_structure(cropped, table_structure_input_shape)
                    table_preds = table_structure_model(cropped_resized, crop_shape)[0]
                    print(f"Page {page_number} - Table structure results: {table_preds}")
                    
                elif label in [1, 2, 3]:  # Graphic elements
                    cropped = crop_tensor_with_bbox(
                        resized_tensor, box, bitmap_shape, page_elements_input_shape
                    ).clone()
                    crop_shape = (cropped.shape[1], cropped.shape[2])
                    cropped_resized = resize_pad_graphic_elements(cropped, graphic_elements_input_shape)
                    graphic_preds = graphic_element_model(cropped_resized, crop_shape)[0]
                    print(f"Page {page_number} - Graphic elements results: {graphic_preds}")
            
            # Run OCR on the original (un-resized) tensor
            # Convert the tensor to a PIL image, then to a BytesIO JPEG for OCR model
            # pil_img = tensor_to_pil_image(original_tensor)
            # img_bytesio = io.BytesIO()
            # pil_img.save(img_bytesio, format="JPEG")
            # img_bytesio.seek(0)
            ocr_preds = ocr_model(tensor.clone().to(device="cuda"))

            for pred in ocr_preds:
                page_ocr_results.append(str(pred['text']))
                page_raw_ocr_results.append(str(pred))
        
        yield page_number, resized_tensor, page_ocr_results, page_raw_ocr_results

def run_pipeline(
    pdf_files: List[str],
    page_elements_model: nn.Module,
    table_structure_model: nn.Module,
    graphic_elements_model: nn.Module,
    ocr_model: NemotronOCR,
    raw_output_dir: Optional[Path] = None,
    dpi: float = 150.0,
    return_results: bool = False,
):
    """
    Process PDF files using a streaming pipeline that handles one page at a time.
    
    Args:
        pdf_files: List of PDF file paths as strings.
        page_elements_model, table_structure_model, graphic_elements_model, ocr_model: Models to use.
        raw_output_dir: Directory to save raw OCR results. If None, does not save.
        dpi: DPI for PDF rendering (default 150).
        return_results: If True, returns results as dict instead of just printing.
    
    Returns:
        If return_results is True, returns a dict with all results.
    """
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
            page_elements_model,
            table_structure_model,
            graphic_elements_model,
            ocr_model,
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
    
    if return_results:
        return {
            "total_pages_processed": total_pages_processed,
            "total_pdfs": len(pdf_files),
            "elapsed_seconds": elapsed,
            "results": results,
        }
    return None

@app.command()
def run(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=True),
    raw_output_dir: Optional[Path] = typer.Option(None, help="Directory to save raw OCR results (optional)."),
):
    # Load Page Elements model
    page_elements_model = define_model_page_elements("page_element_v3")
    table_structure_model = define_model_table_structure("table_structure_v1")
    graphic_elements_model = define_model_graphic_elements("graphic_elements_v1")
    ocr_model = NemotronOCR(model_dir="/home/jdyer/Development/slim-gest/models/nemotron-ocr-v1/checkpoints")

    
    if input_dir.is_file():
        pdf_files = [input_dir]
    else:
        pdf_files = [
            str(f) for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".pdf"
        ]

    console.print(f"Processing {len(pdf_files)} PDFs")
    console.print(f"Using page_elements_model device: {page_elements_model.device}")
    console.print(f"Using table_structure_model device: {table_structure_model.device}")
    console.print(f"Using graphic_elements_model device: {graphic_elements_model.device}")
    
    import time

    for i in range(3, 0, -1):
        console.print(f"[bold yellow]{i}[/bold yellow]", end='\r')
        time.sleep(1)
    console.print("[bold green]Go![/bold green]")

    run_pipeline(
        pdf_files,
        page_elements_model,
        table_structure_model,
        graphic_elements_model,
        ocr_model,
        raw_output_dir=raw_output_dir,
    )

    console.print("[bold green]Done![/bold green]")