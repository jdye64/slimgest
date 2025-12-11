# world_class_pdf_with_crops.py
import os
from pathlib import Path
import time
from threading import Lock
from typing import Optional, List, Tuple, Generator

import torch
import torch.nn.functional as F
import pypdfium2 as pdfium
import typer

# -----------------------
# TUNABLES
# -----------------------
DEVICE = "cuda:0"
TARGET_DET_SIZE = (640, 640)       # detector input H,W
TARGET_CROP_SIZE = (224, 224)      # final crop size for model_crop
PDF_RENDER_DPI = 150               # controls page raster resolution
VERBOSE = True

# -----------------------
# UTILS
# -----------------------
def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

PDFIUM_LOCK = Lock()

def render_pdf_page_to_pinned_tensor(pdf_path: str, page_index: int, dpi: int = PDF_RENDER_DPI):
    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_path)
        page = pdf.get_page(page_index)
        scale = dpi / 72.0
        bitmap = page.render(scale=scale, rotation=0, grayscale=False)
        arr = bitmap.to_numpy()  # H x W x (3 or 4)
        page.close()
        pdf.close()
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    t = t.pin_memory()
    return t

def crop_from_boxes(full_image, boxes, out_size):
    """
    Crop regions from a single image tensor, shaped (C, H, W),
    using Nx4 box tensor (x1, y1, x2, y2) in pixel units.
    Returns tensor shape (N, C, out_H, out_W).
    """
    device = full_image.device
    C, H_full, W_full = full_image.shape
    if boxes.numel() == 0:
        return torch.empty((0, C, out_size[0], out_size[1]), device=device, dtype=full_image.dtype)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    N = boxes.shape[0]
    H_out, W_out = out_size
    x1n = (x1 / (W_full - 1)) * 2.0 - 1.0
    x2n = (x2 / (W_full - 1)) * 2.0 - 1.0
    y1n = (y1 / (H_full - 1)) * 2.0 - 1.0
    y2n = (y2 / (H_full - 1)) * 2.0 - 1.0
    grids = []
    for i in range(N):
        gy = torch.linspace(y1n[i], y2n[i], H_out, device=device)
        gx = torch.linspace(x1n[i], x2n[i], W_out, device=device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grids.append(grid)
    grids = torch.stack(grids, dim=0)
    input_image = full_image.unsqueeze(0).expand(N, -1, -1, -1)
    crops = F.grid_sample(input_image, grids, mode='bilinear', padding_mode='zeros', align_corners=True)
    return crops

def page_pipeline_generator(pdf_path: str, model_detector, model_crop, ocr_model) -> Generator:
    """
    Processes a single PDF document, yielding results for each page as it is finished.
    Each yield is a tuple: (page_index, ocr_results_for_that_page, metadata_for_that_page)
    """
    device = torch.device(DEVICE)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_path)
        n_pages = len(pdf)
    for page_idx in range(n_pages):
        log(f"Processing PDF: {pdf_path}  Page: {page_idx}/{n_pages}")

        # Load and upload image
        t = render_pdf_page_to_pinned_tensor(pdf_path, page_idx)    # C x H x W
        c, h, w = t.shape

        # Upload to GPU
        t_gpu = t.to(device, non_blocking=True).float().div_(255.0).unsqueeze(0)  # 1 x C x H x W

        # Prepare input for detector
        det_in = F.interpolate(t_gpu, size=TARGET_DET_SIZE, mode='bilinear', align_corners=False)
        det_in = (det_in - mean) / std

        orig_size = [[h, w]]
        with torch.no_grad():
            det_out = model_detector(det_in, orig_size)   # typically outputs list[dict] or list[boxes]

        # Get boxes for this page (detects per page, assumes batch=1)
        boxes = None
        if isinstance(det_out, (list, tuple)):
            bi = det_out[0]
            if isinstance(bi, dict) and 'boxes' in bi:
                bi = bi['boxes']
            if bi is None or bi.numel() == 0:
                log(f"No detected boxes for pdf={pdf_path} page={page_idx}")
                yield (page_idx, [], {"pdf_path": pdf_path, "page_index": page_idx, "boxes": None, "crops": None})
                continue
            DET_H, DET_W = TARGET_DET_SIZE
            if bi.max() <= 1.0001:
                bi_px = bi * torch.tensor([DET_W, DET_H, DET_W, DET_H], device=bi.device)
            else:
                bi_px = bi
            scale_x = w / DET_W
            scale_y = h / DET_H
            x1 = (bi_px[:, 0] * scale_x).clamp(0, w-1)
            y1 = (bi_px[:, 1] * scale_y).clamp(0, h-1)
            x2 = (bi_px[:, 2] * scale_x).clamp(0, w-1)
            y2 = (bi_px[:, 3] * scale_y).clamp(0, h-1)
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
        elif isinstance(det_out, torch.Tensor):
            # If it's a tensor of Nx4/5 boxes
            if det_out.ndim == 3 and det_out.shape[0] == 1:
                bi = det_out[0]
                if bi is None or bi.numel() == 0:
                    log(f"No detected boxes for pdf={pdf_path} page={page_idx}")
                    yield (page_idx, [], {"pdf_path": pdf_path, "page_index": page_idx, "boxes": None, "crops": None})
                    continue
                DET_H, DET_W = TARGET_DET_SIZE
                if bi.max() <= 1.0001:
                    bi_px = bi * torch.tensor([DET_W, DET_H, DET_W, DET_H], device=bi.device)
                else:
                    bi_px = bi
                scale_x = w / DET_W
                scale_y = h / DET_H
                x1 = (bi_px[:, 0] * scale_x).clamp(0, w-1)
                y1 = (bi_px[:, 1] * scale_y).clamp(0, h-1)
                x2 = (bi_px[:, 2] * scale_x).clamp(0, w-1)
                y2 = (bi_px[:, 3] * scale_y).clamp(0, h-1)
                boxes = torch.stack([x1, y1, x2, y2], dim=1)
            elif det_out.ndim == 2 and det_out.size(1)>=4:
                boxes = det_out[:, :4]
            else:
                boxes = det_out
        else:
            log(f"Unhandled detector model output for pdf={pdf_path} page={page_idx}: {type(det_out)}")
            yield (page_idx, [], {"pdf_path": pdf_path, "page_index": page_idx, "boxes": None, "crops": None})
            continue

        # Filter boxes for width/height
        widths_box = boxes[:, 2] - boxes[:, 0]
        heights_box = boxes[:, 3] - boxes[:, 1]
        valid_mask = (widths_box > 1.0) & (heights_box > 1.0)
        boxes = boxes[valid_mask]
        if boxes.numel() == 0:
            log(f"No valid boxes for pdf={pdf_path} page={page_idx}")
            yield (page_idx, [], {"pdf_path": pdf_path, "page_index": page_idx, "boxes": None, "crops": None})
            continue

        # Cropping
        t_crop = (t_gpu[0] if t_gpu.ndim == 4 else t_gpu) # (C,H,W)
        crops = crop_from_boxes(t_crop, boxes, TARGET_CROP_SIZE)
        crops = (crops - mean) / std

        # Model crop head, if provided
        if model_crop is not None:
            crop_orig_sizes = [[TARGET_CROP_SIZE[0], TARGET_CROP_SIZE[1]]] * crops.shape[0]
            with torch.no_grad():
                crops = model_crop(crops, crop_orig_sizes)

        # OCR model expects 3xHxW, so loop over crops one by one
        ocr_results = []
        if crops.shape[0] > 0:
            for crop_img in crops:
                try:
                    ocr_result = ocr_model(crop_img.cpu())
                    ocr_results.append(ocr_result)
                except Exception as ex:
                    log("OCR error on crop:", ex)
                    ocr_results.append(None)
        meta = {
            "pdf_path": pdf_path,
            "page_index": page_idx,
            "boxes": boxes.cpu(),
            "n_crops": crops.shape[0] if crops is not None else 0
        }
        yield (page_idx, ocr_results, meta)

app = typer.Typer(help="Process PDFs locally using shared pipeline")

@app.command()
def run(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False),
):
    import torch
    from nemotron_page_elements_v3.model import define_model
    from nemotron_ocr.inference.pipeline import NemotronOCR
    from nemotron_table_structure_v1.model import define_model as define_model_table_structure
    from nemotron_graphic_elements_v1.model import define_model as define_model_graphic_elements

    page_elements_model = define_model("page_element_v3")
    table_structure_model = define_model_table_structure("table_structure_v1")
    graphic_elements_model = define_model_graphic_elements("graphic_elements_v1")
    ocr_model = NemotronOCR(model_dir="/home/jdyer/Development/slim-gest/models/nemotron-ocr-v1/checkpoints")
    pdf_files = [
        str(f) for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".pdf"
    ]
    for pdf_path in pdf_files:
        for page_idx, ocr_results, meta in page_pipeline_generator(
            pdf_path, model_detector=page_elements_model, model_crop=page_elements_model, ocr_model=ocr_model
        ):
            print(f"PDF: {pdf_path}, Page: {page_idx}, OCR Results: {ocr_results}")
            print(f"Meta: {meta}")
