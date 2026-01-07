from pathlib import Path
from typing import List, Tuple, Optional
from rich.console import Console
from rich.traceback import install
from torch import nn
import torch
import time
import pypdfium2 as pdfium
import json

from nemotron_page_elements_v3.model import define_model as define_model_page_elements
from nemotron_page_elements_v3.model import resize_pad as resize_pad_page_elements
from nemotron_page_elements_v3.utils import postprocess_preds_page_element as postprocess_preds_page_element
from nemotron_table_structure_v1.model import define_model as define_model_table_structure
from nemotron_table_structure_v1.model import resize_pad as resize_pad_table_structure
from nemotron_graphic_elements_v1.model import define_model as define_model_graphic_elements
from nemotron_graphic_elements_v1.model import resize_pad as resize_pad_graphic_elements
from nemotron_ocr.inference.pipeline import NemotronOCR

import typer

app = typer.Typer(help="Simpliest pipeline with limited CPU parallelism while using maximum GPU possible")
install(show_locals=False)
console = Console()

def crop_tensor_on_gpu(image_tensor: torch.Tensor, bbox: List[int], bitmap_shape: Tuple[int, int], page_elements_input_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Crops a tensor image on the GPU using the given bounding box.
    The bbox is in normalized coordinates (0-1) relative to the original bitmap.
    The image_tensor has been resized/padded to page_elements_input_shape, so we transform
    the bbox coordinates to match the resized tensor space.

    Args:
        image_tensor (torch.Tensor): Resized image tensor of shape [C, H, W] on GPU.
        bbox (np.ndarray): Normalized bounding box [xmin, ymin, xmax, ymax] in range [0, 1].
        bitmap_shape (Tuple[int, int]): Original bitmap shape (height, width) before resize.
        page_elements_input_shape (Tuple[int, int]): Target shape (height, width) the image was resized to.

    Returns:
        torch.Tensor: Cropped image tensor [C, cropped_H, cropped_W] on the same device.
    """

    # Get dimensions
    orig_h, orig_w = bitmap_shape
    input_h, input_w = page_elements_input_shape
    
    # Calculate scale and padding (matching resize_pad_page_elements logic)
    scale = min(input_h / orig_h, input_w / orig_w)
    scaled_h = int(orig_h * scale)
    scaled_w = int(orig_w * scale)
    pad_y = (input_h - scaled_h) / 2
    pad_x = (input_w - scaled_w) / 2
    
    # Convert normalized bbox to pixel coordinates in original image
    boxes_plot = bbox.copy()
    boxes_plot[0] *= orig_w  # xmin
    boxes_plot[2] *= orig_w  # xmax
    boxes_plot[1] *= orig_h  # ymin
    boxes_plot[3] *= orig_h  # ymax
    
    # Scale to resized coordinates and add padding offset
    xmin = int(boxes_plot[0] * scale + pad_x)
    ymin = int(boxes_plot[1] * scale + pad_y)
    xmax = int(boxes_plot[2] * scale + pad_x)
    ymax = int(boxes_plot[3] * scale + pad_y)
    
    # Clamp the bounds to the actual tensor dimensions
    _, H, W = image_tensor.shape
    xmin = max(0, min(xmin, W - 1))
    ymin = max(0, min(ymin, H - 1))
    xmax = max(xmin + 1, min(xmax, W))  # Ensure xmax > xmin
    ymax = max(ymin + 1, min(ymax, H))  # Ensure ymax > ymin
    
    cropped = image_tensor[:, ymin:ymax, xmin:xmax]
    return cropped


def pdf_to_page_tensors(pdf_path, page_elements_model, table_structure_model, graphic_element_model, ocr_model, device="cuda"):
    """
    Loads a PDF, yields CUDA tensors for each page (shape [3,H,W], dtype=torch.uint8).
    """
    pdf = pdfium.PdfDocument(pdf_path)
    num_pages = len(pdf)
    tensors = []
    result = []
    raw_results = []
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

        # cloned_tensor = tensor.clone()
        cloned_tensor = tensor

        page_elements_input_shape = (1024, 1024)
        table_structure_input_shape = (1024, 1024)
        graphic_elements_input_shape = (1024, 1024)

        with torch.inference_mode():
            # tensor = resize_pad_page_elements(tensor, page_elements_input_shape)
            # preds = page_elements_model(tensor, bitmap_shape)[0]
            # boxes, labels, scores = postprocess_preds_page_element(preds, page_elements_model.thresholds_per_class, page_elements_model.labels)

            # # Iterate over all of the labels and crop the tensors that are tables
            # for label, box in zip(labels, boxes):
            #     if label == 0: # table from page_elements_model.labels
            #         cropped = crop_tensor_on_gpu(tensor, box, bitmap_shape, page_elements_input_shape).clone()
            #         # Get the shape of the cropped tensor BEFORE resizing (H, W)
            #         crop_shape = (cropped.shape[1], cropped.shape[2])
            #         # Resize the cropped tensor to the expected input size (1024, 1024)
            #         cropped_resized = resize_pad_table_structure(cropped, table_structure_input_shape)
            #         # Run the table structure model on the resized cropped tensor
            #         preds = table_structure_model(cropped_resized, crop_shape)[0]
            #         print(f"Table structure results: {preds}")
            #     elif label == 1 or label == 2 or label == 3:
            #         cropped = crop_tensor_on_gpu(tensor, box, bitmap_shape, page_elements_input_shape).clone()
            #         # Get the shape of the cropped tensor BEFORE resizing (H, W)
            #         crop_shape = (cropped.shape[1], cropped.shape[2])
            #         # Resize the cropped tensor to the expected input size (1024, 1024)
            #         cropped_resized = resize_pad_graphic_elements(cropped, graphic_elements_input_shape)
            #         # Run the graphic elements model on the resized cropped tensor
            #         preds = graphic_element_model(cropped_resized, crop_shape)[0]
            #         print(f"Graphic elements results: {preds}")

            # Send the tensor to the OCR model
            preds = ocr_model(cloned_tensor)

            for pred in preds:
                result.append(str(pred['text']))
                raw_results.append(str(pred))

        tensors.append(tensor)
        page.close()
    pdf.close()
    return tensors, result, raw_results

def run_pipeline(
    pdf_files: List[str],
    page_elements_model: nn.Module,
    table_structure_model: nn.Module,
    graphic_elements_model: nn.Module,
    ocr_model: NemotronOCR,
    raw_output_dir: Optional[Path] = None,
):
    """
    Args:
        pdf_files: List of PDF file paths as strings.
        page_elements_model, table_structure_model, graphic_elements_model, ocr_model: Models to use.
        raw_output_dir: Directory to save raw OCR results. If None, does not save.
    """

    gpu_tensors = []
    idx = 1
    total_pages_loaded = 0

    start_time = time.time()
    for pdf_path in pdf_files:
        console.print(f"[bold cyan]Processing:[/bold cyan] {pdf_path}")

        # Efficiently load all pages to GPU tensors (list of [3,H,W] uint8 cuda tensors)
        page_tensors, page_ocr_results, page_raw_ocr_results = pdf_to_page_tensors(
            pdf_path, page_elements_model, table_structure_model, graphic_elements_model, ocr_model
        )
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
        ocr_final_result = " ".join(page_ocr_results)
        console.print(f"OCR final result: {ocr_final_result}", markup=False)

        if raw_output_dir is not None:
            raw_output_dir = Path(raw_output_dir)
            raw_output_dir.mkdir(parents=True, exist_ok=True)
            pdf_path_obj = Path(pdf_path)
            output_json_path = raw_output_dir / pdf_path_obj.with_suffix('.page_raw_ocr_results.json').name
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(page_raw_ocr_results, f, ensure_ascii=False, indent=2)
            console.print(f"Saved page_raw_ocr_results to {output_json_path}")

    elapsed = time.time() - start_time
    console.print(f"[bold green]Loaded {len(gpu_tensors)} pages from {pdf_files} in {elapsed:.2f} seconds[/bold green]")
    for i, t in enumerate(gpu_tensors):
        console.print(f"- Page {i}: tensor shape {list(t.shape)}, dtype={t.dtype}, device={t.device}")

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