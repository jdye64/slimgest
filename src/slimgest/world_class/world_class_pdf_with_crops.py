# world_class_pdf_with_crops.py
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Thread, Event, Lock
from typing import Optional

from rich.console import Console
import torch
import torch.nn.functional as F
import pypdfium2 as pdfium
import typer

# -----------------------
# TUNABLES
# -----------------------
DEVICE = "cuda:0"
BATCH_SIZE = 32                    # pages per detector batch (tune)
TARGET_DET_SIZE = (640, 640)       # detector input H,W (bigger gives better boxes)
TARGET_CROP_SIZE = (224, 224)      # final crop size for model_crop
NUM_DECODE_WORKERS = 12
PREFETCH_PAGES = 512
QUEUE_MAXSIZE = PREFETCH_PAGES
PDF_RENDER_DPI = 150               # controls page raster resolution
VERBOSE = True

# -----------------------
# UTILS
# -----------------------
def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# Global lock to protect pdfium operations if not thread-safe
PDFIUM_LOCK = Lock()

# Fast PDF -> pinned CHW uint8 tensor
def render_pdf_page_to_pinned_tensor(pdf_path: str, page_index: int, dpi: int = PDF_RENDER_DPI):
    with PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_path)
        page = pdf.get_page(page_index)
        scale = dpi / 72.0
        bitmap = page.render(scale=scale, rotation=0, color=True, grayscale=False)
        arr = bitmap.to_numpy()  # H x W x (3 or 4)
        page.close()
        pdf.close()
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W] uint8
    t = t.pin_memory()
    return t

# Render page to pinned tensor from an already-open PdfDocument (safer per-doc reuse)
def render_pdf_page_to_pinned_tensor_from_doc(pdf, page_index: int, dpi: int = PDF_RENDER_DPI):
    with PDFIUM_LOCK:
        page = pdf.get_page(page_index)
        scale = dpi / 72.0
        bitmap = page.render(scale=scale, rotation=0, grayscale=False)
        arr = bitmap.to_numpy()
        page.close()
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    t = t.pin_memory()
    return t

# Producer: decode pages and put (path, page_idx, pinned_tensor) into queue
def producer_feed(queue: Queue, pdf_paths: list, stop_event: Event):
    def process_document(path: str):
        # Open the document once, iterate pages in this worker
        try:
            with PDFIUM_LOCK:
                pdf_local = pdfium.PdfDocument(path)
                n_pages = len(pdf_local)
        except Exception as e:
            log("Failed opening PDF", path, e)
            return
        try:
            for page_idx in range(n_pages):
                # Allow graceful shutdown and queue backpressure
                while queue.qsize() >= QUEUE_MAXSIZE and not stop_event.is_set():
                    time.sleep(0.001)
                if stop_event.is_set():
                    break
                try:
                    t = render_pdf_page_to_pinned_tensor_from_doc(pdf_local, page_idx)
                    queue.put((path, page_idx, t))
                except Exception as e:
                    log("Decode error", path, page_idx, e)
        finally:
            with PDFIUM_LOCK:
                try:
                    pdf_local.close()
                except Exception:
                    pass

    with ThreadPoolExecutor(max_workers=NUM_DECODE_WORKERS) as ex:
        for p in pdf_paths:
            if stop_event.is_set():
                break
            ex.submit(process_document, p)

    queue.put((None, None, None))
    log("Producer finished scheduling decodes.")

# gather CPU batch (list of pinned CHW uint8 tensors)
def gather_page_batch(queue: Queue, batch_size: int, timeout=2.0):
    batch = []
    meta = []
    while len(batch) < batch_size:
        try:
            path, page_idx, tensor = queue.get(timeout=timeout)
        except Empty:
            break
        if path is None and page_idx is None and tensor is None:
            queue.put((None, None, None))
            break
        batch.append(tensor)
        meta.append((path, page_idx))
    return meta, batch

# vectorized batch crop via grid_sample
def batch_crop_from_boxes(full_images, boxes, out_size):
    """
    full_images: [B, C, H_full, W_full] float tensor on device (0..1)
    boxes: list of per-image boxes. We'll accept `boxes` as a tensor shaped [M,5]
           or [B, K, 4] depending on upstream. We expect final to be a 2D tensor:
           boxes_all: [N, 5] with columns (img_index, x1, y1, x2, y2) in pixel coords of full_images.
    out_size: (H_out, W_out)
    Returns: crops [N, C, H_out, W_out]
    """
    device = full_images.device
    C = full_images.shape[1]
    H_full = full_images.shape[2]
    W_full = full_images.shape[3]

    # boxes_all: tensor [N,5] -> (img_idx, x1, y1, x2, y2)
    # Expect boxes as tensor already prepared by caller.
    boxes_all = boxes  # assume already on device

    if boxes_all.numel() == 0:
        # no boxes -> return empty tensor
        return torch.empty((0, C, out_size[0], out_size[1]), device=device, dtype=full_images.dtype)

    img_indices = boxes_all[:, 0].long()      # [N]
    x1 = boxes_all[:, 1]
    y1 = boxes_all[:, 2]
    x2 = boxes_all[:, 3]
    y2 = boxes_all[:, 4]

    N = boxes_all.shape[0]
    H_out, W_out = out_size

    # Normalize to -1..1 coordinates expected by grid_sample
    # x_norm = (x / (W_full-1)) * 2 - 1
    x1n = (x1 / (W_full - 1)) * 2.0 - 1.0
    x2n = (x2 / (W_full - 1)) * 2.0 - 1.0
    y1n = (y1 / (H_full - 1)) * 2.0 - 1.0
    y2n = (y2 / (H_full - 1)) * 2.0 - 1.0

    # build sampling grids per box
    # We'll create [N, H_out, W_out, 2] grid
    grids = []
    # create linspace arrays for each box
    for i in range(N):
        gy = torch.linspace(y1n[i], y2n[i], H_out, device=device)
        gx = torch.linspace(x1n[i], x2n[i], W_out, device=device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # [H_out, W_out]
        grid = torch.stack((grid_x, grid_y), dim=-1)           # [H_out, W_out, 2]
        grids.append(grid)
    grids = torch.stack(grids, dim=0)  # [N, H_out, W_out, 2]

    # Prepare batched full_image repeated per box
    # We must select correct image for each box: easiest is to index-select and repeat per-box
    image_per_box = full_images[img_indices]  # [N, C, H_full, W_full]

    crops = F.grid_sample(image_per_box, grids, mode='bilinear', padding_mode='zeros', align_corners=True)
    return crops  # [N, C, H_out, W_out]

# -----------------------
# CONSUMER: upload -> detector -> map boxes -> crop -> model_crop
# -----------------------
def consumer_loop_with_crops(queue: Queue, model_detector, model_crop, stop_event: Event):
    device = torch.device(DEVICE)
    upload_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)

    # Preallocate scratch buffers
    # For detector input (BATCH_SIZE,3,DET_H,DET_W)
    det_scratch = torch.empty((BATCH_SIZE, 3, TARGET_DET_SIZE[0], TARGET_DET_SIZE[1]), dtype=torch.float32, device=device)
    # For full-resolution images: to avoid too-large prealloc complexity we'll allocate per-batch max
    # We'll create these lazily on first batch based on observed full H,W
    gpu_full_images = None

    # Normalization tensors for detector and model_crop (if same, reuse)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    total_pages = 0
    start_time = time.time()
    finished = False

    while not stop_event.is_set() and not finished:
        meta, cpu_batch = gather_page_batch(queue, BATCH_SIZE)
        if len(cpu_batch) == 0:
            try:
                p, _, _ = queue.get_nowait()
                if p is None:
                    finished = True
                    break
            except Empty:
                time.sleep(0.001)
                continue

        actual_b = len(cpu_batch)
        # stack pinned CPU tensors -> requires same H,W across batch
        # If your pages may vary size, see notes below to pad or render to fixed size.
        cpu_stack = torch.stack(cpu_batch, dim=0)  # uint8 pinned [B,3,H_full, W_full]

        # Async H->D and create full-resolution float image on upload_stream
        with torch.cuda.stream(upload_stream):
            gpu_uint8_full = cpu_stack.to(device, non_blocking=True)          # [B,3,Hf,Wf] uint8
            gpu_full = gpu_uint8_full.float().div_(255.0)                     # float 0..1
            # Optionally keep gpu_full for later cropping; do not resize it here
            # We'll also prepare detector input by resizing gpu_full -> DET size
            gpu_det_in = F.interpolate(gpu_full, size=TARGET_DET_SIZE, mode='bilinear', align_corners=False)
            # normalize detector input
            gpu_det_in = (gpu_det_in - mean) / std
            # copy into det_scratch first actual_b slots to reuse buffer
            det_scratch[:actual_b].copy_(gpu_det_in)
            # record event to signal compute stream
            upload_event = torch.cuda.Event()
            upload_event.record()

        # Compute stream: run detector on resized inputs
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(upload_event)
            detector_input = det_scratch[:actual_b]  # [B,3,DET_H,DET_W]

            # model_detector should be on device and set to eval()
            with torch.no_grad():
                # It's common for detectors to return per-image lists of boxes; adapt below
                det_out = model_detector(detector_input)  # user-specific interface

            # Normalize understanding of detector output:
            # We expect det_out to be either:
            #  - a list of length B with each element a tensor [K,4] of boxes in pixel coords (relative to DET size)
            #  - a tensor [B, K, 4]
            #  - a tensor [total_boxes, 6] with (img_idx, x1, y1, x2, y2, score) (less common)
            # We'll convert to a consolidated boxes_all tensor: [N,5] with (img_idx, x1, y1, x2, y2) in full-image pixel coords.

            # --- NORMALIZE detector outputs into boxes_all ---
            boxes_list = []  # will collect tensors on device
            if isinstance(det_out, (list, tuple)):
                # list of per-image boxes or dicts as in some libs
                for i in range(actual_b):
                    bi = det_out[i]
                    # if dict like {'boxes': tensor, 'scores':..., ...}
                    if isinstance(bi, dict) and 'boxes' in bi:
                        bi = bi['boxes']
                    if bi is None:
                        continue
                    if bi.numel() == 0:
                        continue
                    # bi expected [K,4] in pixel coords relative to DET input
                    # Map from DET coords to full image coords:
                    # full_x = x * (W_full / DET_W)
                    _, Hf, Wf = gpu_full.shape  # careful: gpu_full shape is [B,3,Hf,Wf]; but that's per-batch, we stacked, so...
                    # Actually gpu_full has shape [actual_b,3,Hf,Wf]; we need per-image Hf,Wf; assume all same Hf,Wf
                    _, Hf, Wf = gpu_full.shape[1], gpu_full.shape[2], gpu_full.shape[3] if False else (gpu_full.shape[1], gpu_full.shape[2], gpu_full.shape[3])
                    # Simpler: get Hf,Wf from gpu_full (it is [B,C,Hf,Wf])
                    Hf = gpu_full.shape[2]
                    Wf = gpu_full.shape[3]

                    # DET_W, DET_H
                    DET_H, DET_W = TARGET_DET_SIZE
                    # Convert pixel coords
                    # If boxes appear normalized (max <= 1) treat as normalized
                    if bi.max() <= 1.0001:
                        # normalized coords [0,1] -> convert to DET pixel coords
                        bi_px = bi * torch.tensor([DET_W, DET_H, DET_W, DET_H], device=bi.device)
                    else:
                        bi_px = bi

                    # scale to full resolution
                    scale_x = Wf / DET_W
                    scale_y = Hf / DET_H
                    x1 = bi_px[:, 0] * scale_x
                    y1 = bi_px[:, 1] * scale_y
                    x2 = bi_px[:, 2] * scale_x
                    y2 = bi_px[:, 3] * scale_y

                    img_idx = torch.full((x1.shape[0], 1), i, device=bi.device)
                    boxes_img = torch.stack((img_idx[:,0], x1, y1, x2, y2), dim=1)  # [K,5]
                    boxes_list.append(boxes_img)

            elif isinstance(det_out, torch.Tensor):
                # Could be [B,K,4] or [N,5] etc.
                if det_out.ndim == 3 and det_out.shape[0] == actual_b:
                    # [B,K,4]
                    B0, K, _ = det_out.shape
                    for i in range(B0):
                        bi = det_out[i]  # [K,4]
                        if bi.numel() == 0:
                            continue
                        # handle normalized vs pixel like above
                        if bi.max() <= 1.0001:
                            bi_px = bi * torch.tensor([TARGET_DET_SIZE[1], TARGET_DET_SIZE[0],
                                                       TARGET_DET_SIZE[1], TARGET_DET_SIZE[0]], device=bi.device)
                        else:
                            bi_px = bi
                        Hf = gpu_full.shape[2]
                        Wf = gpu_full.shape[3]
                        scale_x = Wf / TARGET_DET_SIZE[1]
                        scale_y = Hf / TARGET_DET_SIZE[0]
                        x1 = bi_px[:, 0] * scale_x
                        y1 = bi_px[:, 1] * scale_y
                        x2 = bi_px[:, 2] * scale_x
                        y2 = bi_px[:, 3] * scale_y
                        img_idx = torch.full((x1.shape[0], 1), i, device=bi.device)
                        boxes_img = torch.stack((img_idx[:,0], x1, y1, x2, y2), dim=1)
                        boxes_list.append(boxes_img)
                else:
                    # assume already [N,5] (img_idx, x1,y1,x2,y2)
                    boxes_list.append(det_out)

            else:
                log("Unhandled detector output type:", type(det_out))

            if len(boxes_list) == 0:
                # no detections in this batch -> continue
                total_pages += actual_b
                continue

            boxes_all = torch.cat(boxes_list, dim=0).to(device)

            # Clip boxes to image bounds
            boxes_all[:, 1].clamp_(0, gpu_full.shape[3] - 1)  # x1
            boxes_all[:, 3].clamp_(0, gpu_full.shape[3] - 1)  # x2
            boxes_all[:, 2].clamp_(0, gpu_full.shape[2] - 1)  # y1
            boxes_all[:, 4].clamp_(0, gpu_full.shape[2] - 1)  # y2

            # Remove degenerate boxes where x2<=x1 or y2<=y1
            widths = boxes_all[:, 3] - boxes_all[:, 1]
            heights = boxes_all[:, 4] - boxes_all[:, 2]
            valid_mask = (widths > 1.0) & (heights > 1.0)
            boxes_all = boxes_all[valid_mask]
            if boxes_all.numel() == 0:
                total_pages += actual_b
                continue

            # Crop full-resolution images using grid_sample in a vectorized way
            # We need full_images tensor [B,3,Hf,Wf] on device (gpu_full)
            # gpu_full currently is [B,3,Hf,Wf] in upload_stream - but we are on compute_stream and waited on the upload_event,
            # so gpu_full is available here. Ensure it's named correctly:
            full_images = gpu_full  # [B,3,Hf,Wf] float 0..1

            crops = batch_crop_from_boxes(full_images, boxes_all, TARGET_CROP_SIZE)  # [N,3,CH,CW]

            # Normalize crops (in-place)
            crops = (crops - mean) / std  # broadcasting works

            # Run model_crop on the crops (batched)
            with torch.no_grad():
                out_crops = model_crop(crops) if model_crop is not None else None

            # optional: store or push outputs to some sink
            # For example, map boxes_all + out_crops to CPU results (but avoid unless needed)
            # out_cpu = None
            # if out_crops is not None:
            #     out_cpu = out_crops.detach().cpu()  # avoid unless necessary

        total_pages += actual_b

    torch.cuda.synchronize(device)
    elapsed = time.time() - start_time
    log(f"Processed {total_pages} pages in {elapsed:.2f}s -> {total_pages/elapsed:.1f} pages/s")

# Entrypoint that starts producer and consumer
def run_pdf_pipeline_with_crops(pdf_paths, model_detector, model_crop):
    q = Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = Event()

    prod_thread = Thread(target=producer_feed, args=(q, pdf_paths, stop_event), daemon=True)
    prod_thread.start()

    try:
        consumer_loop_with_crops(q, model_detector, model_crop, stop_event)
    finally:
        stop_event.set()
        prod_thread.join(timeout=2.0)


app = typer.Typer(help="Process PDFs locally using shared pipeline")
console = Console()

@app.command()
def run(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False),
):
    import torch
    import torch.nn as nn

    # dummy model for benchmarking (make sure it's on the same device)
    dummy_model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(64, 128)
    ).to("cuda:0").eval()

    pdf_files = [
        str(f) for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".pdf"
    ]
    run_pdf_pipeline_with_crops(pdf_files, model_detector=dummy_model, model_crop=dummy_model)