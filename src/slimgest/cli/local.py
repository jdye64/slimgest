from __future__ import annotations

import typer
from pathlib import Path
import shutil
import os
import time
import json
import yaml
from typing import Dict, List, Any
from rich.console import Console
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

import pypdfium2

from .utils.process_monitor import ProcessMonitor

from nemotron_page_elements_v3.model import define_model as define_page_elements_model

from nemotron_table_structure_v1.table_structure_v1 import Exp
from nemotron_table_structure_v1.model import YoloXWrapper
from nemotron_table_structure_v1.utils import postprocess_preds_table_structure

from nemotron_graphic_elements_v1.graphic_element_v1 import Exp as graphic_element_exp
from nemotron_graphic_elements_v1.model import YoloXWrapper as graphic_element_model
from nemotron_graphic_elements_v1.utils import postprocess_preds_graphic_element

from nemotron_table_structure_v1 import define_model as define_table_model
from nemotron_graphic_elements_v1 import define_model as define_graphic_elements_model

app = typer.Typer(help="Process PDFs, extract per-page images, run page-elements extraction, and profile timings.")

console = Console()

# Global variables for process-local model storage
_process_models = None


def _initialize_worker_process():
    """Initialize models once per worker process.
    
    This is called once when each worker process starts up,
    avoiding the need to load models for every PDF.
    """
    global _process_models
    
    from transformers import logging
    import torch
    logging.set_verbosity_error()
    
    # Force GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    page_elements_model = define_page_elements_model("page_element_v3")
    page_elements_model = page_elements_model.to(device).eval()
    
    table_structure_model = define_table_model("table_structure_v1")
    table_structure_model = table_structure_model.to(device).eval()
    
    graphic_elements_model = define_graphic_elements_model("graphic_elements_v1")
    graphic_elements_model = graphic_elements_model.to(device).eval()
    
    _process_models = {
        'page_elements': page_elements_model,
        'table_structure': table_structure_model,
        'graphic_elements': graphic_elements_model,
        'device': device
    }


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks for config.yaml in current directory.
        
    Returns:
        Configuration dictionary with default values if file not found.
    """
    default_config = {
        "parallel_workers": 10
    }
    
    if config_path is None:
        config_path = Path("config.yaml")
    
    if not config_path.exists():
        console.print(f"[yellow]Config file not found at {config_path}, using defaults[/yellow]")
        return default_config
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Merge with defaults for missing keys
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        return config
    except Exception as e:
        console.print(f"[red]Error loading config: {e}. Using defaults.[/red]")
        return default_config


def make_scratch_dirs(scratch_dir: Path) -> Dict[str, Path]:
    pdf_pages_dir = scratch_dir / "pdf_pages"
    page_images_dir = scratch_dir / "page_images"
    elements_dir = scratch_dir / "page_elements"
    cropped_elements_dir = scratch_dir / "cropped_elements"
    table_structure_dir = scratch_dir / "table_structure"
    graphic_elements_dir = scratch_dir / "graphic_elements"
    graphic_ocr_crops_dir = scratch_dir / "graphic_ocr_crops"
    input_pdf_dir = scratch_dir / "input_pdf"
    metrics_dir = scratch_dir / "metrics"

    pdf_pages_dir.mkdir(parents=True, exist_ok=True)
    page_images_dir.mkdir(parents=True, exist_ok=True)
    elements_dir.mkdir(parents=True, exist_ok=True)
    cropped_elements_dir.mkdir(parents=True, exist_ok=True)
    input_pdf_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    table_structure_dir.mkdir(parents=True, exist_ok=True)
    graphic_elements_dir.mkdir(parents=True, exist_ok=True)
    graphic_ocr_crops_dir.mkdir(parents=True, exist_ok=True)

    return {
        "pdf_pages": pdf_pages_dir,
        "page_images": page_images_dir,
        "page_elements": elements_dir,
        "cropped_elements": cropped_elements_dir,
        "table_structure": table_structure_dir,
        "graphic_elements": graphic_elements_dir,
        "graphic_ocr_crops": graphic_ocr_crops_dir,
        "input_pdf": input_pdf_dir,
        "metrics": metrics_dir,
    }


def split_pdf_to_pages(pdf_path: Path, out_dir: Path) -> List[Path]:
    """Split PDF into single-page PDFs; return list of paths.
    
    Raises:
        Exception: If PDF cannot be loaded or processed
    """
    try:
        pdf = pypdfium2.PdfDocument(str(pdf_path))
    except Exception as e:
        raise Exception(f"Failed to load PDF '{pdf_path.name}': {str(e)}. File may be corrupted or password-protected.")
    
    try:
        page_paths = []
        for page_index in range(len(pdf)):
            out_path = out_dir / f"{pdf_path.stem}_page{page_index+1}.pdf"
            # Create a new PDF with just this one page
            new_pdf = pypdfium2.PdfDocument.new()
            try:
                new_pdf.import_pages(pdf, pages=[page_index])
                new_pdf.save(str(out_path))
                page_paths.append(out_path)
            finally:
                new_pdf.close()
        return page_paths
    finally:
        pdf.close()


def pdf_page_to_png(page_pdf_path: Path, img_dir: Path) -> Path:
    """Convert single-page PDF to PNG, scaled to 1024x1024 with 300dpi."""
    pdf = pypdfium2.PdfDocument(str(page_pdf_path))
    try:
        page = pdf[0]

        # (1) Render at high enough scale to preserve quality for resizing
        # We'll render at a scale to get roughly 1024x1024, then resize to exactly that
        orig_width, orig_height = page.get_size()
        # To target a min 1024x1024 output, cover the max dimension and keep aspect ratio
        scale_x = 1024 / orig_width
        scale_y = 1024 / orig_height
        scale = max(scale_x, scale_y)
        scale = max(scale, 2)  # Ensure at least previous quality

        # Render and convert to PIL, ensuring bitmap is properly closed
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        bitmap.close()

        # (2) Resize and pad/crop to 1024x1024, preserving aspect ratio
        pil_image = pil_image.convert("RGB")
        pil_image = _resize_and_pad(pil_image, (1024, 1024))

        img_name = page_pdf_path.stem + ".png"
        img_path = img_dir / img_name

        # Set DPI and save
        pil_image.save(img_path, dpi=(300, 300))

        return img_path
    finally:
        pdf.close()

def _resize_and_pad(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Resizes and pads the image to the target size (centered), preserving aspect ratio."""
    orig_w, orig_h = img.size
    target_w, target_h = target_size

    # Resize proportionally
    scale = min(target_w / orig_w, target_h / orig_h)
    resized_w, resized_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((resized_w, resized_h), Image.LANCZOS)

    # Create background and paste resized image in center
    new_img = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    left = (target_w - resized_w) // 2
    top = (target_h - resized_h) // 2
    new_img.paste(img, (left, top))
    return new_img


def run_nemotron_page_elements(img_path: Path, out_dir: Path, model=None) -> Path:
    """Run nemotron-page-elements-v3 on given image, save results."""

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    orig_size = img_array.shape
    
    # Preprocess and run inference
    x = model.preprocess(img_array)
    x = x.unsqueeze(0)  # Add batch dimension
    
    with torch.inference_mode():
        # Call with orig_sizes as keyword argument
        preds = model(x, orig_sizes=[orig_size])[0]
    
    # Post-process predictions to get boxes, labels, scores
    boxes = preds["boxes"].cpu().numpy() if "boxes" in preds else []
    labels = preds["labels"].cpu().numpy() if "labels" in preds else []
    scores = preds["scores"].cpu().numpy() if "scores" in preds else []
    
    # Convert to detections list
    detections = []
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        # Assuming labels is an index into model.labels
        label_name = str(int(label)) if not hasattr(model, 'labels') else model.labels[int(label)]
        detections.append({
            "label": label_name,
            "box": box.tolist() if hasattr(box, 'tolist') else list(box),
            "score": float(score)
        })
    
    # Save detections as JSON
    json_file = out_dir / f"{img_path.stem}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "detections": detections,
            "num_detections": len(detections),
            "image": str(img_path)
        }, f, ensure_ascii=False, indent=2)

    return json_file


def crop_page_elements(
    page_img_path: Path,
    elements_json_path: Path,
    out_dir: Path
) -> List[Dict[str, Any]]:
    """
    Crop individual elements from page image based on bounding boxes.
    Returns list of dicts with info about each cropped image.
    """
    # Load the page image
    page_img = Image.open(page_img_path).convert("RGB")
    img_width, img_height = page_img.size
    
    # Load the detections JSON
    with open(elements_json_path, "r") as f:
        elements_data = json.load(f)
    
    cropped_info = []
    page_stem = page_img_path.stem  # e.g., "1016445_page1"
    
    # Create a unique subdirectory for this page's cropped elements
    page_crop_dir = out_dir / page_stem
    page_crop_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, detection in enumerate(elements_data["detections"]):
        label = detection["label"]
        box = detection["box"]  # [x1, y1, x2, y2] in normalized coords (0-1)
        score = detection["score"]
        
        # Convert normalized coords to pixel coords
        x1 = int(box[0] * img_width)
        y1 = int(box[1] * img_height)
        x2 = int(box[2] * img_width)
        y2 = int(box[3] * img_height)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, img_width))
        x2 = max(0, min(x2, img_width))
        y1 = max(0, min(y1, img_height))
        y2 = max(0, min(y2, img_height))
        
        # Skip if box is invalid
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Crop the element
        cropped = page_img.crop((x1, y1, x2, y2))
        
        # Save with descriptive name: page_element{idx}_{label}.png
        crop_filename = f"page_element{idx:03d}_{label}.png"
        crop_path = page_crop_dir / crop_filename
        cropped.save(crop_path)
        
        cropped_info.append({
            "crop_path": str(crop_path),
            "element_index": idx,
            "label": label,
            "score": score,
            "original_page": str(page_img_path),
            "box_pixels": [x1, y1, x2, y2],
            "box_normalized": box
        })
    
    return cropped_info


def run_nemotron_table_structure(cropped_img_path: Path, output_dir: Path, model: YoloXWrapper) -> None:
    """Map function to process table images and detect structure.
    
    Uses nemotron-table-structure-v1 model to detect table structure
    including rows, columns, and cells.
    """
    from nemotron_table_structure_v1.utils import postprocess_preds_table_structure
    
    # Load and preprocess image
    img = Image.open(cropped_img_path).convert("RGB")
    img_array = np.array(img)
    orig_size = img_array.shape
    
    # Preprocess and run inference
    start_time = time.time()
    x = model.preprocess(img_array)
    
    with torch.inference_mode():
        preds = model(x, orig_size)[0]
    
    model_time = time.time() - start_time
    
    # Post-process predictions
    boxes, labels, scores = postprocess_preds_table_structure(
        preds, model.threshold, model.labels
    )
    
    # Convert to detections list
    table_elements = []
    for box, label, score in zip(boxes, labels, scores):
        label_name = model.labels[int(label)]
        table_elements.append({
            "label": label_name,
            "bbox": box.tolist() if hasattr(box, 'tolist') else box,
            "score": float(score)
        })

    return table_elements


def run_nemotron_ocr(cropped_detection_img_path: Path, ocr_counter: Dict[str, int]) -> str:
    """Skeleton function for OCR on cropped detections.
    
    This function will send the image to nemotron_ocr to perform OCR
    and return the detected text.
    
    Args:
        cropped_detection_img_path: Path to the cropped detection image
        ocr_counter: Dictionary to track OCR invocation count (local to process)
        
    Returns:
        Detected text from the image
    """
    # Increment OCR invocation counter (process-local, no lock needed)
    ocr_counter['count'] += 1
    
    # TODO: Implement actual OCR using nemotron_ocr
    # print(f"run_nemotron_ocr: {cropped_detection_img_path.name}")
    return ""


def crop_detection_from_image(
    source_img_path: Path,
    bbox: List[float],
    output_path: Path
) -> None:
    """Crop a detection bounding box from source image and save it.
    
    Args:
        source_img_path: Path to the source image
        bbox: Normalized bounding box [x1, y1, x2, y2] in range 0-1
        output_path: Path to save the cropped image
    """
    img = Image.open(source_img_path).convert("RGB")
    width, height = img.size
    
    # Convert normalized bbox to pixel coordinates
    x1 = int(bbox[0] * width)
    y1 = int(bbox[1] * height)
    x2 = int(bbox[2] * width)
    y2 = int(bbox[3] * height)
    
    # Crop and save
    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(output_path)


def run_nemotron_graphic_elements(cropped_img_path: Path, output_dir: Path, model: graphic_element_model) -> list:
    """Process graphic/figure images and detect elements.
    
    Uses nemotron-graphic-elements-v1 model to detect graphic elements.
    """
    from nemotron_graphic_elements_v1.utils import postprocess_preds_graphic_element
    
    # Load and preprocess image
    img = Image.open(cropped_img_path).convert("RGB")
    img_array = np.array(img)
    orig_size = img_array.shape
    
    # Preprocess and run inference
    start_time = time.time()
    x = model.preprocess(img_array)
    
    with torch.inference_mode():
        preds = model(x, orig_size)[0]
    
    model_time = time.time() - start_time
    
    # Post-process predictions
    boxes, labels, scores = postprocess_preds_graphic_element(
        preds, model.threshold, model.labels
    )
    
    # Convert to detections list
    graphic_elements = []
    for box, label, score in zip(boxes, labels, scores):
        label_name = model.labels[int(label)]
        graphic_elements.append({
            "label": label_name,
            "bbox": box.tolist() if hasattr(box, 'tolist') else box,
            "score": float(score)
        })

    return graphic_elements


def save_metrics(metrics: Dict[str, Any], out_dir: Path, pdf_name: str):
    out_path = out_dir / f"{pdf_name}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)


def process_single_pdf(
    pdf_path: Path,
    scratch_dict: Dict[str, str]
) -> tuple[str, Dict[str, Any], Dict[str, int]]:
    """Process a single PDF file and return its metrics.
    
    This function runs in a worker process with models already loaded
    via the process initializer.
    
    Args:
        pdf_path: Path to the PDF file
        scratch_dict: Dictionary of scratch directory paths (as strings)
        
    Returns:
        Tuple of (pdf_name, pdf_metrics, counters) where counters contains
        page_count, table_count, and graphic_count
    """
    global _process_models
    
    # Convert scratch dict back to Path objects
    scratch = {k: Path(v) for k, v in scratch_dict.items()}
    
    # Get models from process-local storage
    page_elements_model = _process_models['page_elements']
    table_structure_model = _process_models['table_structure']
    graphic_elements_model = _process_models['graphic_elements']
    
    pdf_metrics = {}
    
    # Initialize counters for this PDF
    counters = {
        'page_count': 0,
        'table_count': 0,
        'graphic_count': 0,
        'ocr_count': 0,
        'time_pdf_split': 0.0,
        'time_page_to_png': 0.0,
        'time_page_elements': 0.0,
        'time_crop_elements': 0.0,
        'time_table_structure': 0.0,
        'time_graphic_elements': 0.0
    }
    
    # Local OCR counter for this PDF
    local_ocr_counter = {'count': 0}
    
    # Copy input PDF to scratch directory
    input_pdf_copy = scratch["input_pdf"] / pdf_path.name
    shutil.copy2(pdf_path, input_pdf_copy)
    pdf_metrics["input_pdf_copy"] = str(input_pdf_copy)
    
    # Split PDF into pages
    t0 = time.perf_counter()
    page_pdfs = split_pdf_to_pages(pdf_path, scratch["pdf_pages"])
    t1 = time.perf_counter()
    split_time = t1 - t0
    pdf_metrics["split_pdf_to_pages"] = {"duration_sec": split_time, "num_pages": len(page_pdfs)}
    
    # Update page counter and timing
    counters['page_count'] = len(page_pdfs)
    counters['time_pdf_split'] = split_time

    page_metrics = {}

    for page_idx, page_pdf_path in enumerate(page_pdfs):
        metrics = {"page_num": page_idx+1}
        # PDF → PNG step
        step1_start = time.perf_counter()
        img_path = pdf_page_to_png(page_pdf_path, scratch["page_images"])
        step1_end = time.perf_counter()
        png_time = step1_end - step1_start
        metrics["pdf_page_to_png"] = {"duration_sec": png_time, "image": str(img_path)}
        counters['time_page_to_png'] += png_time

        # PNG → page elements step
        step2_start = time.perf_counter()
        elem_json_path = run_nemotron_page_elements(img_path, scratch["page_elements"], page_elements_model)
        step2_end = time.perf_counter()
        page_elements_time = step2_end - step2_start
        metrics["run_nemotron_page_elements"] = {
            "duration_sec": page_elements_time,
            "elements_json": str(elem_json_path),
        }
        counters['time_page_elements'] += page_elements_time

        # Crop individual elements from page
        step3_start = time.perf_counter()
        cropped_elements = crop_page_elements(img_path, elem_json_path, scratch["cropped_elements"])
        step3_end = time.perf_counter()
        crop_time = step3_end - step3_start
        metrics["crop_page_elements"] = {
            "duration_sec": crop_time,
            "num_cropped": len(cropped_elements),
            "cropped_elements": cropped_elements
        }
        counters['time_crop_elements'] += crop_time

        # Process each cropped element based on its type
        table_structure_results = []
        table_structure_total_time = 0.0
        graphic_elements_results = []
        graphic_elements_total_time = 0.0
        
        for cropped_info in cropped_elements:
            crop_path = Path(cropped_info["crop_path"])
            label = cropped_info["label"]
            element_index = cropped_info["element_index"]
            
            # Call appropriate skeleton function based on label
            if label == "table":
                step4_start = time.perf_counter()
                table_elements = run_nemotron_table_structure(crop_path, scratch["table_structure"], table_structure_model)
                step4_end = time.perf_counter()
                table_duration = step4_end - step4_start
                table_structure_total_time += table_duration
                
                # Increment table counter and timing
                counters['table_count'] += 1
                counters['time_table_structure'] += table_duration
                
                # Save table structure to JSON file
                json_filename = f"{page_pdf_path.stem}_element_{element_index:03d}_table_structure.json"
                json_file = scratch["table_structure"] / json_filename
                
                table_result = {
                    "source_image": str(crop_path),
                    "detections": table_elements,
                    "num_detections": len(table_elements),
                    "page_number": page_idx + 1,
                    "element_index": element_index,
                    "duration_sec": table_duration
                }
                
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(table_result, f, ensure_ascii=False, indent=2)
                
                table_structure_results.append({
                    "json_file": str(json_file),
                    "duration_sec": table_duration,
                    "num_detections": len(table_elements)
                })
                
            elif label in ["figure", "graphic", "image", "chart"]:
                step5_start = time.perf_counter()
                graphic_elements = run_nemotron_graphic_elements(crop_path, scratch["graphic_elements"], graphic_elements_model)
                step5_end = time.perf_counter()
                graphic_duration = step5_end - step5_start
                graphic_elements_total_time += graphic_duration
                
                # Increment graphic counter and timing
                counters['graphic_count'] += 1
                counters['time_graphic_elements'] += graphic_duration
                
                # Create subdirectory for this graphic's OCR crops
                graphic_ocr_subdir = scratch["graphic_ocr_crops"] / f"{page_pdf_path.stem}_element_{element_index:03d}"
                graphic_ocr_subdir.mkdir(parents=True, exist_ok=True)
                
                # Crop each detection and run OCR
                for det_idx, detection in enumerate(graphic_elements):
                    bbox = detection["bbox"]
                    det_label = detection["label"]
                    
                    # Crop the detection from the graphic image
                    crop_filename = f"detection_{det_idx:03d}_{det_label}.png"
                    crop_output_path = graphic_ocr_subdir / crop_filename
                    crop_detection_from_image(crop_path, bbox, crop_output_path)
                    
                    # Run OCR on the cropped detection
                    ocr_text = run_nemotron_ocr(crop_output_path, local_ocr_counter)
                    
                    # Add OCR results to detection
                    detection["ocr_crop_path"] = str(crop_output_path)
                    detection["ocr_text"] = ocr_text
                
                # Save graphic elements to JSON file
                json_filename = f"{page_pdf_path.stem}_element_{element_index:03d}_graphic_elements.json"
                json_file = scratch["graphic_elements"] / json_filename
                
                graphic_result = {
                    "source_image": str(crop_path),
                    "detections": graphic_elements,
                    "num_detections": len(graphic_elements),
                    "page_number": page_idx + 1,
                    "element_index": element_index,
                    "duration_sec": graphic_duration
                }
                
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(graphic_result, f, ensure_ascii=False, indent=2)
                
                graphic_elements_results.append({
                    "json_file": str(json_file),
                    "duration_sec": graphic_duration,
                    "num_detections": len(graphic_elements)
                })
            # For other labels (text, title, etc.) we don't process yet
        
        # Add table structure metrics to page metrics
        if table_structure_results:
            metrics["run_nemotron_table_structure"] = {
                "duration_sec": table_structure_total_time,
                "num_tables": len(table_structure_results),
                "tables": table_structure_results
            }
        
        # Add graphic elements metrics to page metrics
        if graphic_elements_results:
            metrics["run_nemotron_graphic_elements"] = {
                "duration_sec": graphic_elements_total_time,
                "num_graphics": len(graphic_elements_results),
                "graphics": graphic_elements_results
            }
        
        page_metrics[f"page_{page_idx+1}"] = metrics

    pdf_metrics["total_pages"] = len(page_pdfs)
    pdf_metrics["pages"] = page_metrics

    # Save per-PDF metrics
    save_metrics(pdf_metrics, scratch["metrics"], pdf_path.stem)
    
    # Add OCR count to counters
    counters['ocr_count'] = local_ocr_counter['count']
    
    return (pdf_path.name, pdf_metrics, counters)


@app.command()
def process(
    input_dir: Path = typer.Argument(..., help="Directory with input PDFs"),
    scratch_dir: Path = typer.Argument(..., help="Directory to save all temp/intermediate/results"),
    config_path: Path = typer.Option(None, help="Path to config.yaml file"),
):
    """
    Process each PDF in `input_dir`, extract individual pages, save page images, 
    run nemotron-page-elements-v3, and profile timings for every step.
    
    Multiple PDFs are processed in parallel based on config settings.
    """
    # Load configuration
    config = load_config(config_path)
    parallel_workers = config["parallel_workers"]
    console.print(f"[bold cyan]Using {parallel_workers} parallel workers for PDF processing[/bold cyan]")
    console.print(f"[bold]Models will be loaded once per worker process[/bold]")
    
    # Initialize process monitoring
    console.print(f"[bold cyan]Starting process monitoring for CPU, memory, and disk I/O[/bold cyan]")
    monitor = ProcessMonitor(sample_interval=1.0)
    monitor.start()
    monitor.annotate_stage("Initialization")
    
    # Track total runtime and OCR invocations
    total_start_time = time.perf_counter()
    
    scratch = make_scratch_dirs(scratch_dir)
    
    # Convert scratch paths to strings for pickling across processes
    scratch_dict = {k: str(v) for k, v in scratch.items()}

    all_pdf_metrics = {}

    pdf_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() == ".pdf"])
    if not pdf_files:
        console.print("[red]No PDF files found in input directory![/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Found {len(pdf_files)} PDF files to process[/bold]")
    
    # Initialize global counters for statistics
    global_counters = {
        'total_pages': 0,
        'total_tables': 0,
        'total_graphics': 0,
        'total_ocr': 0,
        'time_pdf_split': 0.0,
        'time_page_to_png': 0.0,
        'time_page_elements': 0.0,
        'time_crop_elements': 0.0,
        'time_table_structure': 0.0,
        'time_graphic_elements': 0.0
    }
    
    # Track failed PDFs
    failed_pdfs = []
    
    # Use multiprocessing for parallel processing (pypdfium2 is not thread-safe)
    # Each process loads models once via initializer
    with ProcessPoolExecutor(
        max_workers=parallel_workers,
        initializer=_initialize_worker_process
    ) as executor:
        # Note: Model loading happens inside worker initializer
        monitor.annotate_stage("Loading Models")
        
        # Submit all PDF processing jobs
        future_to_pdf = {
            executor.submit(
                process_single_pdf,
                pdf_path,
                scratch_dict
            ): pdf_path
            for pdf_path in pdf_files
        }
        
        # Process completed jobs with progress bar
        first_result = True
        with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                
                # Annotate PDF processing stage after first result starts
                if first_result:
                    monitor.annotate_stage("PDF Processing")
                    first_result = False
                
                try:
                    pdf_name, pdf_metrics, counters = future.result()
                    all_pdf_metrics[pdf_name] = pdf_metrics
                    
                    # Aggregate global counters
                    global_counters['total_pages'] += counters['page_count']
                    global_counters['total_tables'] += counters['table_count']
                    global_counters['total_graphics'] += counters['graphic_count']
                    global_counters['total_ocr'] += counters['ocr_count']
                    
                    # Aggregate timing information
                    global_counters['time_pdf_split'] += counters['time_pdf_split']
                    global_counters['time_page_to_png'] += counters['time_page_to_png']
                    global_counters['time_page_elements'] += counters['time_page_elements']
                    global_counters['time_crop_elements'] += counters['time_crop_elements']
                    global_counters['time_table_structure'] += counters['time_table_structure']
                    global_counters['time_graphic_elements'] += counters['time_graphic_elements']
                    
                    console.print(f"[green]✓[/green] Completed: {pdf_name}")
                except Exception as exc:
                    error_msg = str(exc)
                    # Simplify common error messages
                    if "Failed to load document" in error_msg or "PDFium" in error_msg:
                        error_msg = "Corrupted or invalid PDF file"
                    elif "password" in error_msg.lower():
                        error_msg = "Password-protected PDF"
                    
                    console.print(f"[red]✗[/red] {pdf_path.name}: {error_msg}")
                    failed_pdfs.append({
                        'filename': pdf_path.name,
                        'error': error_msg
                    })
                finally:
                    pbar.update(1)

    # Save overall metrics
    monitor.annotate_stage("Saving Results")
    save_metrics(all_pdf_metrics, scratch["metrics"], "ALL_PDFS")

    # Calculate total runtime
    total_end_time = time.perf_counter()
    total_runtime = total_end_time - total_start_time
    
    # Print summary
    console.print("\n" + "="*70)
    console.print(f"[bold green]Processing complete![/bold green]")
    console.print("="*70)
    
    # Success/failure summary
    successful_count = len(all_pdf_metrics)
    failed_count = len(failed_pdfs)
    total_pdfs = len(pdf_files)
    
    console.print(f"\n[bold]PDF Processing Summary:[/bold]")
    console.print(f"  • Total PDFs: {total_pdfs}")
    console.print(f"  • [green]Successful: {successful_count}[/green]")
    if failed_count > 0:
        console.print(f"  • [red]Failed: {failed_count}[/red]")
    
    # Calculate total cumulative processing time first
    total_cumulative_time = (
        global_counters['time_pdf_split'] +
        global_counters['time_page_to_png'] +
        global_counters['time_page_elements'] +
        global_counters['time_crop_elements'] +
        global_counters['time_table_structure'] +
        global_counters['time_graphic_elements']
    )
    
    # Statistics
    console.print(f"\n[bold]Processing Statistics:[/bold]")
    console.print(f"  • Total pages processed: {global_counters['total_pages']:,}")
    console.print(f"  • Total tables processed: {global_counters['total_tables']:,}")
    console.print(f"  • Total graphics processed: {global_counters['total_graphics']:,}")
    console.print(f"  • Total OCR invocations: {global_counters['total_ocr']:,}")
    console.print(f"\n[bold]Runtime:[/bold]")
    console.print(f"  • Wall time (actual): {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    console.print(f"  • Cumulative compute time: {total_cumulative_time:.2f} seconds ({total_cumulative_time/60:.2f} minutes)")
    if total_runtime > 0:
        speedup = total_cumulative_time / total_runtime
        console.print(f"  • Parallel speedup: {speedup:.1f}x with {parallel_workers} workers")
    
    # Timing breakdown with percentages
    console.print(f"\n[bold]Time Breakdown (by processing step):[/bold]")
    
    # Display each step with time and percentage of cumulative time
    steps = [
        ("PDF Splitting", global_counters['time_pdf_split']),
        ("Page to PNG", global_counters['time_page_to_png']),
        ("Page Elements Detection", global_counters['time_page_elements']),
        ("Element Cropping", global_counters['time_crop_elements']),
        ("Table Structure Analysis", global_counters['time_table_structure']),
        ("Graphic Elements Analysis", global_counters['time_graphic_elements'])
    ]
    
    for step_name, step_time in steps:
        if total_cumulative_time > 0:
            percentage = (step_time / total_cumulative_time) * 100
            console.print(f"  • {step_name}: {step_time:.2f}s ({percentage:.1f}%)")
    
    # Failed PDFs details
    if failed_pdfs:
        console.print(f"\n[bold yellow]Failed PDFs ({len(failed_pdfs)}):[/bold yellow]")
        for failed in failed_pdfs[:10]:  # Show first 10
            console.print(f"  • {failed['filename']}: {failed['error']}")
        if len(failed_pdfs) > 10:
            console.print(f"  ... and {len(failed_pdfs) - 10} more")
    
    console.print(f"\n[bold]Results saved to:[/bold] {scratch_dir}")
    
    # Stop monitoring and save results
    console.print(f"\n[bold cyan]Stopping process monitoring and saving results...[/bold cyan]")
    monitor.stop()
    
    # Save monitoring results to scratch directory
    monitoring_metrics_path = scratch_dir / "process_monitoring_metrics.json"
    monitoring_graph_path = scratch_dir / "process_monitoring_graph.png"
    
    monitor.save_metrics(monitoring_metrics_path)
    monitor.generate_graphs(monitoring_graph_path)
    
    console.print(f"[green]✓[/green] Saved process monitoring metrics to: {monitoring_metrics_path}")
    console.print(f"[green]✓[/green] Saved process monitoring graph to: {monitoring_graph_path}")
    console.print("="*70 + "\n")


if __name__ == "__main__":
    app()
