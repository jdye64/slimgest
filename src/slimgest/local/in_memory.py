from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
from rich.console import Console
from rich.traceback import install
from rich.table import Table
from torch import nn
import torch
import time
import json
import io
from PIL import Image
import numpy as np
import mimetypes
from collections import defaultdict
import threading
import queue
import sys
import subprocess
from datetime import datetime

from nemotron_page_elements_v3.model import define_model as define_model_page_elements
from nemotron_page_elements_v3.model import resize_pad as resize_pad_page_elements
from nemotron_page_elements_v3.utils import postprocess_preds_page_element as postprocess_preds_page_element
from nemotron_table_structure_v1.model import define_model as define_model_table_structure
from nemotron_table_structure_v1.model import resize_pad as resize_pad_table_structure
from nemotron_graphic_elements_v1.model import define_model as define_model_graphic_elements
from nemotron_graphic_elements_v1.model import resize_pad as resize_pad_graphic_elements
from nemotron_ocr.inference.pipeline import NemotronOCR

import typer

# Import PDF render utilities
from slimgest.pdf.render import iter_pdf_page_tensors
from slimgest.pdf.tensor_ops import crop_tensor_with_bbox

app = typer.Typer(help="In-memory batch pipeline with batched page/image processing and detailed performance breakdown")
install(show_locals=False)
console = Console()

def get_git_sha():
    """Get the current git commit SHA, or 'unknown' if not in a git repo."""
    try:
        sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], 
                                      stderr=subprocess.DEVNULL).decode('ascii').strip()
        return sha
    except:
        return 'unknown'

def get_log_filename():
    """Generate log filename in format YYYY-MM-DD-HH-MM-SS-git-sha.log"""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    git_sha = get_git_sha()
    return f"{timestamp}-{git_sha}.log"


def tensor_to_pil_image(tensor):
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    arr = tensor.detach().numpy()
    if arr.max() <= 1.0:
        arr = arr * 255.
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # HWC
    img = Image.fromarray(arr, mode='RGB')
    return img

def pil_image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32)
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr) / 255.0  # Normalize to [0,1]

def process_batch_bitmaps(
    batch: List[Dict[str, Any]],
    page_elements_model,
    table_structure_model,
    graphic_element_model,
    ocr_model,
    device="cuda",
    timings=None,
    graphic_elements_batch_size=32,
    table_structure_batch_size=32,
):
    """
    Processes a batch of page/image tensors with optimized batching and parallelization.
    
    Optimizations:
    - Batch processes all page elements at once
    - Runs OCR in parallel with page elements detection
    - Batches table structure and graphic elements processing
    - Uses memory pinning for faster CPU->GPU transfers
    
    Args:
        batch: List of page/image dictionaries
        page_elements_model: Model for detecting page elements
        table_structure_model: Model for table structure detection
        graphic_element_model: Model for graphic element detection
        ocr_model: OCR model
        device: Device to run inference on
        timings: Optional timing dictionary
        graphic_elements_batch_size: Batch size for graphic elements processing
        table_structure_batch_size: Batch size for table structure processing
    """

    # For timing breakdown
    page_elements_times = []
    table_structure_times = []
    graphic_elements_times = []
    ocr_times = []
    postprocess_times = []

    page_elements_input_shape = (1024, 1024)
    table_structure_input_shape = (1024, 1024)
    graphic_elements_input_shape = (1024, 1024)

    # For model invocation counts
    pe_invocations = 0
    table_invocations = 0
    graphics_invocations = 0
    ocr_invocations = 0

    # STEP 1: Prepare all tensors and batch them for page elements
    t_prep = time.perf_counter()
    batch_tensors = []
    batch_shapes = []
    batch_metadata = []
    
    for page in batch:
        tensor = page["tensor"].to(device, non_blocking=True)
        batch_tensors.append(tensor)
        batch_shapes.append((tensor.shape[1], tensor.shape[2]))
        batch_metadata.append({
            "input_id": page["input_id"],
            "pdf_path": page.get("pdf_path"),
            "image_path": page.get("image_path"),
            "page_number": page.get("page_number"),
        })
    
    # STEP 2: Run Page Elements detection (sequential to avoid CUDA threading issues)
    t0_pe = time.perf_counter()
    with torch.inference_mode():
        # Resize and pad all tensors
        resized_tensors = []
        for tensor, shape in zip(batch_tensors, batch_shapes):
            resized = resize_pad_page_elements(tensor, page_elements_input_shape)
            resized_tensors.append(resized)
        
        # Batch process all pages through page elements model
        all_preds = []
        with torch.cuda.device(device):
            for resized_tensor, bitmap_shape in zip(resized_tensors, batch_shapes):
                preds = page_elements_model(resized_tensor, bitmap_shape)[0]
                all_preds.append(preds)
    
    page_elements_time = time.perf_counter() - t0_pe
    page_elements_times.append(page_elements_time)
    pe_invocations += len(batch)
    
    # STEP 3: Run OCR on all pages (sequential to avoid CUDA threading issues)
    t0_ocr = time.perf_counter()
    all_ocr_results = []
    for tensor in batch_tensors:
        ocr_preds = ocr_model(tensor)
        ocr_page_results = []
        ocr_page_raw = []
        for pred in ocr_preds:
            ocr_page_results.append(str(pred['text']))
            ocr_page_raw.append(str(pred))
        all_ocr_results.append({
            'text': " ".join(ocr_page_results),
            'raw': ocr_page_raw
        })
    
    ocr_time = time.perf_counter() - t0_ocr
    ocr_times.extend([ocr_time / len(batch)] * len(batch))  # Distribute time across pages
    ocr_invocations += len(batch)
    
    # STEP 4: Postprocess page elements and collect all crops
    t_postproc = time.perf_counter()
    all_table_crops = []
    all_graphic_crops = []
    page_element_metadata = []  # Track which page each element belongs to
    
    for idx, (preds, resized_tensor, bitmap_shape) in enumerate(
        zip(all_preds, resized_tensors, batch_shapes)
    ):
        boxes, labels, scores = postprocess_preds_page_element(
            preds, page_elements_model.thresholds_per_class, page_elements_model.labels
        )
        
        # Collect crops for batched processing
        for label, box in zip(labels, boxes):
            cropped = crop_tensor_with_bbox(
                resized_tensor, box, bitmap_shape, page_elements_input_shape
            ).clone()
            crop_shape = (cropped.shape[1], cropped.shape[2])
            
            if label == 0:  # Table
                cropped_resized = resize_pad_table_structure(cropped, table_structure_input_shape)
                all_table_crops.append({
                    'tensor': cropped_resized,
                    'shape': crop_shape,
                    'page_idx': idx
                })
            elif label in [1, 2, 3]:  # Graphics
                cropped_resized = resize_pad_graphic_elements(cropped, graphic_elements_input_shape)
                all_graphic_crops.append({
                    'tensor': cropped_resized,
                    'shape': crop_shape,
                    'page_idx': idx
                })
    
    postproc_elapsed = time.perf_counter() - t_postproc
    postprocess_times.append(postproc_elapsed)
    
    # STEP 5: Batch process table structures
    t_table = time.perf_counter()
    table_results_by_page = [[] for _ in range(len(batch))]
    
    if all_table_crops:
        with torch.inference_mode():
            with torch.cuda.device(device):
                for i in range(0, len(all_table_crops), table_structure_batch_size):
                    batch_crops = all_table_crops[i:i + table_structure_batch_size]
                    for crop_data in batch_crops:
                        res = table_structure_model(crop_data['tensor'], crop_data['shape'])[0]
                        table_results_by_page[crop_data['page_idx']].append(res)
                        table_invocations += 1
    
    table_elapsed = time.perf_counter() - t_table
    table_structure_times.append(table_elapsed)
    
    # STEP 6: Batch process graphic elements
    t_graphics = time.perf_counter()
    graphic_results_by_page = [[] for _ in range(len(batch))]
    
    if all_graphic_crops:
        with torch.inference_mode():
            with torch.cuda.device(device):
                for i in range(0, len(all_graphic_crops), graphic_elements_batch_size):
                    batch_crops = all_graphic_crops[i:i + graphic_elements_batch_size]
                    for crop_data in batch_crops:
                        res = graphic_element_model(crop_data['tensor'], crop_data['shape'])[0]
                        graphic_results_by_page[crop_data['page_idx']].append(res)
                        graphics_invocations += 1
    
    graphics_elapsed = time.perf_counter() - t_graphics
    graphic_elements_times.append(graphics_elapsed)
    
    # STEP 7: Assemble final results
    results = []
    for idx, metadata in enumerate(batch_metadata):
        ocr_result = all_ocr_results[idx]
        result = {
            "input_id": metadata["input_id"],
            "pdf_path": metadata["pdf_path"],
            "image_path": metadata["image_path"],
            "page_number": metadata["page_number"],
            "ocr_text": ocr_result['text'],
            "raw_ocr_results": ocr_result['raw'],
            # Optionally: more outputs from table_results_by_page or graphic_results_by_page
        }
        results.append(result)
        
        if timings is not None:
            timings_per_page = {
                "page_elements": page_elements_time / len(batch),
                "page_elements_postproc": postproc_elapsed / len(batch),
                "table_structure": table_elapsed / len(batch) if all_table_crops else 0,
                "graphic_elements": graphics_elapsed / len(batch) if all_graphic_crops else 0,
                "ocr": ocr_time / len(batch),
            }
            timings["per_page"].append(timings_per_page)

    # Record batch timings for global stats + invocation counts
    if timings is not None:
        timings["page_elements"].extend(page_elements_times)
        timings["page_elements_postproc"].extend(postprocess_times)
        timings["table_structure"].extend(table_structure_times)
        timings["graphic_elements"].extend(graphic_elements_times)
        timings["ocr"].extend(ocr_times)
        timings["total_pages"] += len(batch)
        # Invocation tracking (accumulate counts)
        timings.setdefault("invocations", {})
        inv = timings["invocations"]
        inv["page_elements"] = inv.get("page_elements", 0) + pe_invocations
        inv["table_structure"] = inv.get("table_structure", 0) + table_invocations
        inv["graphic_elements"] = inv.get("graphic_elements", 0) + graphics_invocations
        inv["ocr"] = inv.get("ocr", 0) + ocr_invocations

    return results

def iter_documents_and_pages(input_dir: Path, dpi: float = 150):
    """
    Generator that yields documents (PDFs and images) one at a time.
    For PDFs, yields all pages of that PDF together.
    For images, yields the single image.
    
    Yields:
        Tuple of (file_path, pages_data) where pages_data is a list of page info dicts
    """
    import mimetypes
    from PIL import Image

    files = sorted([Path(f) for f in input_dir.iterdir() if f.is_file()])
    
    for f in files:
        mime, _ = mimetypes.guess_type(str(f))
        ext = f.suffix.lower()
        
        if mime == "application/pdf" or ext == ".pdf":
            # PDF: yield all pages together
            pages = []
            # Use CPU as temporary device for loading, will be moved to target GPU later
            for page_info in iter_pdf_page_tensors(str(f), dpi=dpi, device=['cpu']):
                pages.append({
                    "input_id": f"{f.name}__page_{page_info.page_number}",
                    "pdf_path": str(f),
                    "page_number": page_info.page_number,
                    "tensor": page_info.tensor.detach().clone().float(),  # shape [3,H,W]
                    "device": None,  # Will be assigned by loader
                })
            if pages:
                yield (f, pages)
                
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            # Image file: yield single image
            try:
                pil_img = Image.open(f).convert("RGB")
                tensor = pil_image_to_tensor(pil_img)
                page_data = [{
                    "input_id": f"{f.name}",
                    "image_path": str(f),
                    "page_number": None,
                    "tensor": tensor,
                    "device": None,  # Will be assigned by loader
                }]
                yield (f, page_data)
            except Exception as e:
                console.print(f"[red]Error loading image {f}: {e}[/red]")
        else:
            console.print(f"[yellow]Skipping unsupported file: {f}[/yellow]")


def load_and_prepare_bitmaps(input_dir: Path, dpi: float = 150, devices: list = None, timings=None) -> List[Dict[str, Any]]:
    """
    Loads all PDF pages and image files as tensors in memory, returning a unified list of dict objects.
    
    Args:
        input_dir: Directory containing PDFs and images
        dpi: DPI for rendering PDFs
        devices: List of device strings (e.g., ["cuda:0", "cuda:1"]) for load balancing. If None, uses single device.
        timings: Optional timing dict to record load time
    """
    import mimetypes
    from PIL import Image

    if timings is not None:
        t0 = time.perf_counter()
    
    if devices is None:
        devices = ['cuda:0' if torch.cuda.is_available() else "cpu"]

    page_entries = []
    page_counter = 0

    for file_path, pages in iter_documents_and_pages(input_dir, dpi=dpi):
        for page in pages:
            # Distribute across devices using round-robin
            target_device = devices[page_counter % len(devices)]
            page["tensor"] = page["tensor"].to(target_device)
            page["device"] = torch.device(target_device)
            page_entries.append(page)
            page_counter += 1

    if timings is not None:
        timings["io_load"] = time.perf_counter() - t0
    return page_entries

def print_timing_breakdown(timings, total_wall):
    # PPS and per-step breakdown, formatted with rich
    total_pages = timings.get("total_pages", None) or len(timings.get("per_page", []))
    breakdown_table = Table(title="Pipeline Wall Time Breakdown", show_lines=True)
    breakdown_table.add_column("Stage", style="bold")
    breakdown_table.add_column("Total Time [s]")
    breakdown_table.add_column("Avg/Page [ms]")
    breakdown_table.add_column("Pct of Wall")
    breakdown_table.add_column("# Model Invocations")

    def stage_row(stage, label=None, invocation_key=None):
        total = sum(timings.get(stage, []))
        ms = 1000. * (total/total_pages) if total_pages else 0
        pct = (total/total_wall*100) if total_wall > 0 else 0
        # Show invocation count if key is present
        invocations = None
        if "invocations" in timings and invocation_key:
            invocations = timings["invocations"].get(invocation_key, None)
        breakdown_table.add_row(
            label or stage,
            f"{total:.2f}",
            f"{ms:.1f}",
            f"{pct:.1f}%",
            str(invocations) if invocations is not None else "-"
        )

    io_time = timings.get("io_load", 0.0)
    breakdown_table.add_row("File/Page IO", f"{io_time:.2f}", "-", f"{io_time/total_wall*100:.1f}%" if total_wall > 0 else "-", "-")

    stage_row("page_elements", "Page Elements Det.", "page_elements")
    stage_row("page_elements_postproc", "Postprocess Elements", None)
    stage_row("table_structure", "Table Struct. Det.", "table_structure")
    stage_row("graphic_elements", "Graphics Det.", "graphic_elements")
    stage_row("ocr", "OCR", "ocr")
    # Aggregate
    sum_stages = sum(sum(timings.get(stage, [])) for stage in ["page_elements", "page_elements_postproc", "table_structure", "graphic_elements", "ocr"])
    idle_time = total_wall - (sum_stages + io_time)
    breakdown_table.add_row("Other/Idle", f"{idle_time:.2f}", "-", f"{idle_time/total_wall*100:.1f}%" if total_wall > 0 else "-", "-")

    breakdown_table.add_section()
    breakdown_table.add_row(
        "TOTAL",
        f"{total_wall:.2f}",
        f"{1000.*total_wall/total_pages:.1f}" if total_pages else "-",
        "100%",
        "-",  # No invocation count for total
    )

    console.print(breakdown_table)
    pps = total_pages / total_wall if total_wall > 0 else 0
    console.print(f"[bold green]Overall throughput: {pps:.2f} pages/sec ({total_pages} pages, {total_wall:.2f}s wall)[/bold green]")

def document_loader_thread(
    input_dir: Path,
    dpi: float,
    device_list: List[str],
    document_queues: Dict[str, queue.Queue],
    pages_per_gpu: int,
    stop_event: threading.Event,
    stats: Dict[str, Any],
    page_semaphores: Dict[str, threading.Semaphore],
):
    """
    Thread that loads documents and distributes them to device queues.
    Uses semaphores to maintain pages_per_gpu page limit per GPU.
    """
    console.print(f"[cyan]Starting document loader thread (max {pages_per_gpu} pages/GPU)...[/cyan]")
    
    total_docs_loaded = 0
    total_pages_loaded = 0
    doc_counter = 0
    
    try:
        for file_path, pages in iter_documents_and_pages(input_dir, dpi=dpi):
            if stop_event.is_set():
                break
            
            # Assign to device in round-robin fashion
            target_device = device_list[doc_counter % len(device_list)]
            doc_counter += 1
            
            num_pages = len(pages)
            
            # Acquire semaphore slots for this many pages (blocks if not enough space)
            # Acquire one at a time to avoid deadlock
            for _ in range(num_pages):
                page_semaphores[target_device].acquire()
            
            # Move pages to target device
            for page in pages:
                page["tensor"] = page["tensor"].to(target_device)
                page["device"] = torch.device(target_device)
            
            # Put document (all its pages) into the device's queue
            document_queues[target_device].put((file_path, pages, num_pages))
            total_docs_loaded += 1
            total_pages_loaded += num_pages
            
            if total_docs_loaded % 10 == 0:
                console.print(f"[cyan]Loader: {total_docs_loaded} documents ({total_pages_loaded} pages) loaded[/cyan]")
        
        # Signal completion by putting None sentinel to all queues
        for device in device_list:
            document_queues[device].put(None)
        
        console.print(f"[green]Document loader completed: {total_docs_loaded} documents ({total_pages_loaded} pages)[/green]")
        stats["total_docs_loaded"] = total_docs_loaded
        stats["total_pages_loaded"] = total_pages_loaded
        
    except Exception as e:
        console.print(f"[red]Error in document loader: {e}[/red]")
        import traceback
        traceback.print_exc()
        stop_event.set()


def gpu_process_batches_streaming(
    document_queue: queue.Queue,
    page_elements_model,
    table_structure_model,
    graphic_element_model,
    ocr_model,
    device,
    timings,
    results_list,
    batch_size,
    thread_id=0,
    graphic_elements_batch_size=32,
    table_structure_batch_size=32,
    stop_event=None,
    pages_counter=None,
    counter_lock=None,
    start_time=None,
    page_semaphore=None,
):
    """
    Process documents from a queue in a streaming fashion.
    Pulls documents from queue, batches pages, and processes them.
    Releases semaphore slots as pages are processed.
    """
    # Set the CUDA device for this thread
    torch.cuda.set_device(device)
    
    local_results = []
    batch_idx = 0
    total_pages = 0
    
    # Track start time for wall clock
    if start_time is None:
        start_time = time.perf_counter()
    
    # Accumulate pages into batches
    current_batch = []
    
    while True:
        if stop_event and stop_event.is_set():
            break
        
        try:
            # Get document from queue (non-blocking with timeout)
            item = document_queue.get(timeout=1.0)
            
            if item is None:  # Sentinel value indicating no more documents
                # Process any remaining pages in current batch
                if current_batch:
                    batch_start = time.perf_counter()
                    batch_results = process_batch_bitmaps(
                        batch=current_batch,
                        page_elements_model=page_elements_model,
                        table_structure_model=table_structure_model,
                        graphic_element_model=graphic_element_model,
                        ocr_model=ocr_model,
                        device=device,
                        timings=timings,
                        graphic_elements_batch_size=graphic_elements_batch_size,
                        table_structure_batch_size=table_structure_batch_size,
                    )
                    batch_end = time.perf_counter()
                    batch_pps = len(current_batch) / (batch_end - batch_start) if (batch_end - batch_start) > 0 else 0
                    local_results.extend(batch_results)
                    num_processed = len(current_batch)
                    total_pages += num_processed
                    
                    # Release semaphore slots for processed pages
                    if page_semaphore is not None:
                        for _ in range(num_processed):
                            page_semaphore.release()
                    
                    # Update global counter
                    wall_elapsed = batch_end - start_time
                    if pages_counter is not None and counter_lock is not None:
                        with counter_lock:
                            pages_counter['value'] += num_processed
                            running_total = pages_counter['value']
                        console.print(f"[Thread {thread_id}] Final batch ({num_processed} pages) in {batch_end-batch_start:.2f}s [{batch_pps:.2f} pages/sec] | Total: {running_total} pages | Wall: {wall_elapsed:.1f}s")
                    else:
                        console.print(f"[Thread {thread_id}] Final batch ({num_processed} pages) in {batch_end-batch_start:.2f}s [{batch_pps:.2f} pages/sec] | Wall: {wall_elapsed:.1f}s")
                break
            
            file_path, pages, num_pages = item
            
            # Add pages to current batch
            for page in pages:
                current_batch.append(page)
                
                # When batch is full, process it
                if len(current_batch) >= batch_size:
                    batch_start = time.perf_counter()
                    batch_results = process_batch_bitmaps(
                        batch=current_batch,
                        page_elements_model=page_elements_model,
                        table_structure_model=table_structure_model,
                        graphic_element_model=graphic_element_model,
                        ocr_model=ocr_model,
                        device=device,
                        timings=timings,
                        graphic_elements_batch_size=graphic_elements_batch_size,
                        table_structure_batch_size=table_structure_batch_size,
                    )
                    batch_end = time.perf_counter()
                    batch_pps = len(current_batch) / (batch_end - batch_start) if (batch_end - batch_start) > 0 else 0
                    batch_idx += 1
                    local_results.extend(batch_results)
                    num_processed = len(current_batch)
                    total_pages += num_processed
                    
                    # Release semaphore slots for processed pages
                    if page_semaphore is not None:
                        for _ in range(num_processed):
                            page_semaphore.release()
                    
                    # Update global counter
                    wall_elapsed = batch_end - start_time
                    if pages_counter is not None and counter_lock is not None:
                        with counter_lock:
                            pages_counter['value'] += num_processed
                            running_total = pages_counter['value']
                        console.print(f"[Thread {thread_id}] Batch {batch_idx} ({num_processed} pages) in {batch_end-batch_start:.2f}s [{batch_pps:.2f} pages/sec] | Total: {running_total} pages | Wall: {wall_elapsed:.1f}s")
                    else:
                        console.print(f"[Thread {thread_id}] Batch {batch_idx} ({num_processed} pages) in {batch_end-batch_start:.2f}s [{batch_pps:.2f} pages/sec] | Wall: {wall_elapsed:.1f}s")
                    
                    current_batch = []
                    
                    # Clear cache after each batch
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
            
            # Mark task as done
            document_queue.task_done()
            
        except queue.Empty:
            # Timeout occurred, check if we should continue
            continue
        except Exception as e:
            console.print(f"[red]Error in GPU processing thread {thread_id}: {e}[/red]")
            import traceback
            traceback.print_exc()
            if stop_event:
                stop_event.set()
            break
    
    results_list.extend(local_results)
    console.print(f"[green]Thread {thread_id} completed: {total_pages} pages processed[/green]")


def gpu_process_batches(
    batches: List[List[Dict[str, Any]]],
    page_elements_model,
    table_structure_model,
    graphic_element_model,
    ocr_model,
    device,
    timings,
    results_list,
    thread_id=0,
    graphic_elements_batch_size=32,
    table_structure_batch_size=32,
):
    # Set the CUDA device for this thread
    torch.cuda.set_device(device)
    
    local_results = []
    for idx, batch in enumerate(batches):
        batch_start = time.perf_counter()
        batch_results = process_batch_bitmaps(
            batch=batch,
            page_elements_model=page_elements_model,
            table_structure_model=table_structure_model,
            graphic_element_model=graphic_element_model,
            ocr_model=ocr_model,
            device=device,
            timings=timings,
            graphic_elements_batch_size=graphic_elements_batch_size,
            table_structure_batch_size=table_structure_batch_size,
        )
        batch_end = time.perf_counter()
        batch_pps = len(batch) / (batch_end - batch_start)
        console.print(f"[Thread {thread_id}] Processed batch {idx+1} ({len(batch)} pages) in {batch_end-batch_start:.2f}s [{batch_pps:.2f} pages/sec]")
        local_results.extend(batch_results)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    results_list.extend(local_results)

@app.command()
def run(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=True),
    raw_output_dir: Optional[Path] = typer.Option(None, help="Directory to save raw OCR results (optional)."),
    batch_size: int = typer.Option(32, help="Batch size for processing (default=32, safest for GPU RAM)."),
    devices: Optional[str] = typer.Option(None, help="Comma-separated list of devices (e.g., 'cuda:0,cuda:1'). If not specified, auto-detects available GPUs."),
    graphic_elements_batch_size: int = typer.Option(32, help="Batch size for graphic elements model processing."),
    table_structure_batch_size: int = typer.Option(32, help="Batch size for table structure model processing."),
    pages_per_gpu: int = typer.Option(1000, help="Maximum number of pages to keep in memory per GPU (streaming mode)."),
    streaming: bool = typer.Option(True, help="Use streaming mode to avoid OOM on large datasets."),
    log_dir: Optional[Path] = typer.Option(None, help="Directory to save log file. If not specified, logs to current directory."),
):
    import time
    
    # Setup log file
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = Path(".")
    
    log_file = log_dir / get_log_filename()
    
    # Start recording console output - create new console with recording
    from rich.console import Console as RichConsole
    recording_console = RichConsole(record=True)
    
    # Use the recording console for output
    global console
    original_console = console
    console = recording_console
    
    console.print(f"[bold cyan]Logging to: {log_file}[/bold cyan]")

    # Parse devices parameter or auto-detect
    if devices:
        device_list = [d.strip() for d in devices.split(',')]
    else:
        # Auto-detect: use all available CUDA devices or fall back to single device
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                device_list = [f"cuda:{i}" for i in range(num_gpus)]
            else:
                device_list = ["cuda"]
        else:
            device_list = ["cpu"]
    
    console.print(f"[bold cyan]Using devices: {device_list}[/bold cyan]")
    console.print(f"[bold cyan]Mode: {'Streaming' if streaming else 'Load All'} ({pages_per_gpu if streaming else 'all'} pages/GPU)[/bold cyan]")

    # Load models for each device
    models_per_device = []
    for device in device_list:
        page_elements_model = define_model_page_elements("page_element_v3")
        page_elements_model = page_elements_model.to(torch.device(device))
        # Manually update the model's device attribute (models have custom .device attribute)
        if hasattr(page_elements_model, 'device'):
            page_elements_model.device = torch.device(device)

        table_structure_model = define_model_table_structure("table_structure_v1").to(torch.device(device))
        if hasattr(table_structure_model, 'device'):
            table_structure_model.device = torch.device(device)
            
        graphic_elements_model = define_model_graphic_elements("graphic_elements_v1").to(torch.device(device))
        if hasattr(graphic_elements_model, 'device'):
            graphic_elements_model.device = torch.device(device)
            
        ocr_model = NemotronOCR(model_dir="/home/jdyer/Development/slim-gest/models/nemotron-ocr-v1/checkpoints", device=device)
        
        models_per_device.append({
            "device": device,
            "page_elements": page_elements_model,
            "table_structure": table_structure_model,
            "graphic_elements": graphic_elements_model,
            "ocr": ocr_model,
        })
    
    console.print(f"[bold green]Loaded models on {len(models_per_device)} device(s)[/bold green]")

    dpi = 150.0

    console.print(f"Scanning files in directory: {input_dir}")

    # Create timing structures for each device
    timings_per_device = []
    for i in range(len(device_list)):
        timings_per_device.append({
            "io_load": 0.0,
            "page_elements": [],
            "page_elements_postproc": [],
            "table_structure": [],
            "graphic_elements": [],
            "ocr": [],
            "per_page": [],
            "total_pages": 0,
            "invocations": {
                "page_elements": 0,
                "table_structure": 0,
                "graphic_elements": 0,
                "ocr": 0,
            },
        })

    t_all_start = time.perf_counter()

    # Progress warmup
    for i in range(3, 0, -1):
        console.print(f"[bold yellow]{i}[/bold yellow]", end='\r')
        time.sleep(1)
    console.print("[bold green]Go![/bold green]")

    results_per_device = [[] for _ in device_list]
    t_pipeline_start = time.perf_counter()

    if streaming:
        # --- STREAMING MODE: Load documents on-demand ---
        console.print(f"[bold cyan]Starting streaming pipeline (max {pages_per_gpu * len(device_list)} pages in memory)[/bold cyan]")
        
        # Create unbounded queues and semaphores for each device
        document_queues = {device: queue.Queue() for device in device_list}
        page_semaphores = {device: threading.Semaphore(pages_per_gpu) for device in device_list}
        
        # Shared state
        stop_event = threading.Event()
        loader_stats = {}
        pages_counter = {'value': 0}  # Shared counter for pages processed
        counter_lock = threading.Lock()
        
        # Track loading and processing start time
        t_load_start = time.perf_counter()
        t_process_start = time.perf_counter()
        
        # Start document loader thread
        loader_thread = threading.Thread(
            target=document_loader_thread,
            args=(
                input_dir,
                dpi,
                device_list,
                document_queues,
                pages_per_gpu,
                stop_event,
                loader_stats,
                page_semaphores,
            ),
            daemon=True,
        )
        loader_thread.start()
        
        # Start processing threads for each device
        threads = []
        for idx, (device, models) in enumerate(zip(device_list, models_per_device)):
            th = threading.Thread(
                target=gpu_process_batches_streaming,
                args=(
                    document_queues[device],
                    models["page_elements"],
                    models["table_structure"],
                    models["graphic_elements"],
                    models["ocr"],
                    device,
                    timings_per_device[idx],
                    results_per_device[idx],
                    batch_size,
                    idx,
                    graphic_elements_batch_size,
                    table_structure_batch_size,
                    stop_event,
                    pages_counter,
                    counter_lock,
                    t_process_start,
                    page_semaphores[device],
                ),
            )
            threads.append(th)
        
        # Start all processing threads
        for th in threads:
            th.start()
        
        # Wait for loader to finish
        loader_thread.join()
        t_load_end = time.perf_counter()
        
        # Wait for all processing threads to complete
        for th in threads:
            th.join()
        
        # Record loading time (distributed across devices)
        load_time = t_load_end - t_load_start
        timings_per_device[0]["io_load"] = load_time
        
        console.print(f"[green]Loader stats: {loader_stats}[/green]")
        
    else:
        # --- LOAD ALL MODE: Original behavior ---
        t_io_start = time.perf_counter()
        all_bitmaps = load_and_prepare_bitmaps(input_dir, dpi=dpi, devices=device_list, timings=timings_per_device[0])
        t_io_end = time.perf_counter()
        timings_per_device[0]["io_load"] = t_io_end - t_io_start
        total_pages = len(all_bitmaps)
        console.print(f"[bold cyan]Found {total_pages} total pages/images for processing.[/bold cyan]")

        # --- SPLIT BATCHES ACROSS ALL DEVICES ---
        # Group bitmaps by their assigned device
        bitmaps_by_device = {device: [] for device in device_list}
        for bitmap in all_bitmaps:
            bitmap_device = str(bitmap.get("device", device_list[0]))
            # Normalize device string
            for dev in device_list:
                if bitmap_device == dev or bitmap_device == str(torch.device(dev)):
                    bitmaps_by_device[dev].append(bitmap)
                    break

        # Create batches for each device
        batches_per_device = []
        for device in device_list:
            device_bitmaps = bitmaps_by_device[device]
            device_batches = [device_bitmaps[i:i+batch_size] for i in range(0, len(device_bitmaps), batch_size)]
            batches_per_device.append(device_batches)
            console.print(f"[cyan]Device {device}: {len(device_bitmaps)} pages in {len(device_batches)} batches[/cyan]")

        threads = []

        # Create a thread for each device
        for idx, (device, models, batches) in enumerate(zip(device_list, models_per_device, batches_per_device)):
            if len(batches) > 0:  # Only create thread if there are batches to process
                print(f"Models: {models}, device: {device}, batches: {len(batches)}")
                th = threading.Thread(
                    target=gpu_process_batches,
                    args=(
                        batches,
                        models["page_elements"],
                        models["table_structure"],
                        models["graphic_elements"],
                        models["ocr"],
                        device,
                        timings_per_device[idx],
                        results_per_device[idx],
                        idx,
                        graphic_elements_batch_size,
                        table_structure_batch_size,
                    ))
                threads.append(th)

        # Start all threads
        for th in threads:
            th.start()

        # Wait for all threads to complete
        for th in threads:
            th.join()

    t_pipeline_end = time.perf_counter()
    pipeline_wall_time = t_pipeline_end - t_pipeline_start

    # Combine results and timings from all devices
    results = []
    for device_results in results_per_device:
        results.extend(device_results)
    
    total_timings = {}
    for k in ["page_elements", "page_elements_postproc", "table_structure", "graphic_elements", "ocr", "per_page"]:
        total_timings[k] = []
        for device_timings in timings_per_device:
            total_timings[k].extend(device_timings.get(k, []))
    
    total_timings["io_load"] = timings_per_device[0].get("io_load", 0.0)
    total_timings["total_pages"] = sum(dt.get("total_pages", 0) for dt in timings_per_device)
    
    # Sum invocation counts across all devices
    total_timings["invocations"] = {}
    for metric in ["page_elements", "table_structure", "graphic_elements", "ocr"]:
        total_timings["invocations"][metric] = sum(
            dt["invocations"].get(metric, 0) for dt in timings_per_device
        )

    # Save per-page raw OCR output if requested
    if raw_output_dir is not None:
        raw_output_dir = Path(raw_output_dir)
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        for res in results:
            if res["pdf_path"]:
                name = Path(res["pdf_path"]).with_suffix('').name
                page_suffix = f".page{res['page_number']:04d}" if res["page_number"] is not None else ""
                fname = f"{name}{page_suffix}_raw_ocr_results.json"
            else:
                fname = Path(res["image_path"]).with_suffix('').name + "_raw_ocr_results.json"
            output_json_path = raw_output_dir / fname
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(res["raw_ocr_results"], f, ensure_ascii=False, indent=2)
        console.print(f"[blue]Saved per-page raw OCR results to {raw_output_dir}[/blue]")

    t_all_end = time.perf_counter()
    total_wall = t_all_end - t_all_start

    # --- Report results & timings ---
    console.print(f"[bold green]Processed {len(results)} total pages/images in {total_wall:.2f} seconds.[/bold green]")
    print_timing_breakdown(total_timings, total_wall)

    console.print("[bold green]Done![/bold green]")
    
    # Save console output to log file
    try:
        console_text = console.export_text()
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(console_text)
        console.print(f"[bold cyan]Log saved to: {log_file}[/bold cyan]")
        # Re-export to include the last message
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(console.export_text())
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save log file: {e}[/yellow]")
    finally:
        # Restore original console
        console = original_console