from pathlib import Path
from typing import List
from rich.console import Console
from rich.traceback import install
from torch import nn
import torch
import time
import pypdfium2 as pdfium

from nemotron_page_elements_v3.model import define_model as define_model_page_elements
from nemotron_page_elements_v3.model import resize_pad as resize_pad_page_elements
from nemotron_page_elements_v3.utils import postprocess_preds_page_element as postprocess_preds_page_element
from nemotron_table_structure_v1.model import define_model as define_model_table_structure
from nemotron_graphic_elements_v1.model import define_model as define_model_graphic_elements
from nemotron_ocr.inference.pipeline import NemotronOCR

import typer

app = typer.Typer(help="Simpliest pipeline with limited CPU parallelism while using maximum GPU possible")
install(show_locals=False)
console = Console()

def crop_tensor_on_gpu(image_tensor: torch.Tensor, bbox: List[int]) -> torch.Tensor:
    """
    Crops a tensor image on the GPU using the given bounding box.

    Args:
        image_tensor (torch.Tensor): Image tensor of shape [C, H, W] on GPU.
        bbox (List[int] or torch.Tensor): Bounding box [xmin, ymin, xmax, ymax].

    Returns:
        torch.Tensor: Cropped image tensor [C, cropped_H, cropped_W] on the same device.
    """
    # Ensure bbox is in integer format and tensor if needed
    if not torch.is_tensor(bbox):
        bbox = torch.tensor(bbox, dtype=torch.long, device=image_tensor.device)
    else:
        bbox = bbox.to(dtype=torch.long, device=image_tensor.device)
    xmin, ymin, xmax, ymax = bbox.tolist()
    # Clamp the bounds in case they exceed image size
    _, H, W = image_tensor.shape
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(W, xmax)
    ymax = min(H, ymax)
    cropped = image_tensor[:, ymin:ymax, xmin:xmax]
    return cropped


def pdf_to_page_tensors(pdf_path, page_elements_model, table_structure_model, graphic_element_model, ocr_model, device="cuda"):
    """
    Loads a PDF, yields CUDA tensors for each page (shape [3,H,W], dtype=torch.uint8).
    """
    pdf = pdfium.PdfDocument(pdf_path)
    num_pages = len(pdf)
    result = []
    for idx in range(num_pages):
        page = pdf.get_page(idx)
        # Render at 150 DPI, RGB
        bitmap = page.render(scale=150/72.0, rotation=0, grayscale=False)
        # Get shape of bitmap (height and width)
        bitmap_shape = (bitmap.height, bitmap.width)  # bitmap.height and bitmap.width are available
        arr = bitmap.to_numpy()  # shape [H,W,3 or 4]
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        # Shape: [H, W, 3] -> [3, H, W]
        tensor = torch.from_numpy(arr).permute(2,0,1).contiguous()
        tensor = tensor.to(device, non_blocking=True)

        with torch.inference_mode():
            tensor = resize_pad_page_elements(tensor, (1024, 1024))
            preds = page_elements_model(tensor, bitmap_shape)[0]
            boxes, labels, scores = postprocess_preds_page_element(preds, page_elements_model.thresholds_per_class, page_elements_model.labels)

            # Iterate over all of the labels and crop the tensors that are tables
            for label, box in zip(labels, boxes):
                if label == 0: # table from page_elements_model.labels
                    cropped = crop_tensor_on_gpu(tensor, box)
                    # Run the table structure model on the table tensors
                    # Get the shape of the cropped tensor (H, W)
                    crop_shape = (cropped.shape[1], cropped.shape[2])
                    preds = table_structure_model(cropped, crop_shape)[0]
                elif label == 1 or label == 2 or label == 3:
                    cropped = crop_tensor_on_gpu(tensor, box)
                    # Run the graphic elements model on the cropped tensors
                    # Get the shape of the cropped tensor (H, W)
                    crop_shape = (cropped.shape[1], cropped.shape[2])
                    preds = graphic_element_model(cropped, crop_shape)[0]

            # # Send the tensor to the OCR model
            # preds = ocr_model(tensor)

        result.append(tensor)
        page.close()
    pdf.close()
    return result

def run_pipeline(pdf_files: List[str],
    page_elements_model: nn.Module,
    table_structure_model: nn.Module,
    graphic_elements_model: nn.Module,
    ocr_model: NemotronOCR,
):

    gpu_tensors = []
    idx = 1
    total_pages_loaded = 0

    start_time = time.time()
    for pdf_path in pdf_files:
        console.print(f"[bold cyan]Processing:[/bold cyan] {pdf_path}")

        # Efficiently load all pages to GPU tensors (list of [3,H,W] uint8 cuda tensors)
        page_tensors = pdf_to_page_tensors(pdf_path, page_elements_model, table_structure_model, graphic_elements_model, ocr_model)
        # gpu_tensors.extend(page_tensors)
        num_pages = len(page_tensors)
        total_pages_loaded += num_pages

        # Example: just show count and shape, or (optionally) process with models
        console.print(
            f"Loaded {num_pages} pages from {pdf_path}. "
            f"PDF {idx} of {len(pdf_files)}. "
            f"Total pages loaded so far: {total_pages_loaded}. "
            f"Current Runtime: {time.time() - start_time:.2f} seconds"
        )
        idx += 1

    elapsed = time.time() - start_time
    console.print(f"[bold green]Loaded {len(gpu_tensors)} pages from {pdf_files} in {elapsed:.2f} seconds[/bold green]")
    for i, t in enumerate(gpu_tensors):
        console.print(f"- Page {i}: tensor shape {list(t.shape)}, dtype={t.dtype}, device={t.device}")

@app.command()
def run(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False),
):
    # Load Page Elements model
    page_elements_model = define_model_page_elements("page_element_v3")
    table_structure_model = define_model_table_structure("table_structure_v1")
    graphic_elements_model = define_model_graphic_elements("graphic_elements_v1")
    ocr_model = NemotronOCR(model_dir="/home/jdyer/Development/slim-gest/models/nemotron-ocr-v1/checkpoints")

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

    run_pipeline(pdf_files, page_elements_model, table_structure_model, graphic_elements_model, ocr_model)

    console.print("[bold green]Done![/bold green]")