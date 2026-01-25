from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from rich.console import Console
from rich.traceback import install
import torch
import torch.nn.functional as F
import time
import json
import numpy as np
import pypdfium2 as pdfium

from slimgest.model.local.nemotron_page_elements_v3 import NemotronPageElementsV3
from slimgest.model.local.nemotron_table_structure_v1 import NemotronTableStructureV1
from slimgest.model.local.nemotron_graphic_elements_v1 import NemotronGraphicElementsV1
from slimgest.model.local.nemotron_ocr_v1 import NemotronOCRV1

import llama_nemotron_embed_1b_v2

import typer

from slimgest.pdf.render import iter_pdf_page_tensors

app = typer.Typer(help="Simpliest pipeline with limited CPU parallelism while using maximum GPU possible")
install(show_locals=False)
console = Console()

def _empty_run_metrics() -> Dict[str, Any]:
    return {
        "pages_processed": 0,
        "models": {
            # Each entry: {"calls": int, "seconds": float, "items": int}
        },
        "counts": {
            "table_regions": 0,
            "graphic_regions": 0,
            "table_structure_detections": 0,
            "graphic_elements_detections": 0,
            "ocr_crops": 0,
            "embedding_segments": 0,
        },
        "timings": {
            # Aggregate stage timing (seconds)
            "page_total": 0.0,
            "page_text": 0.0,
            "page_elements": 0.0,
            "table_structure": 0.0,
            "graphic_elements": 0.0,
            "ocr": 0.0,
            "embedding": 0.0,
        },
    }


def _metrics_add(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    dst["pages_processed"] += int(src.get("pages_processed", 0))

    for k, v in (src.get("counts", {}) or {}).items():
        dst["counts"][k] = int(dst["counts"].get(k, 0)) + int(v or 0)

    for k, v in (src.get("timings", {}) or {}).items():
        dst["timings"][k] = float(dst["timings"].get(k, 0.0)) + float(v or 0.0)

    for model_name, m in (src.get("models", {}) or {}).items():
        cur = dst["models"].setdefault(model_name, {"calls": 0, "seconds": 0.0, "items": 0})
        cur["calls"] += int(m.get("calls", 0) or 0)
        cur["items"] += int(m.get("items", 0) or 0)
        cur["seconds"] += float(m.get("seconds", 0.0) or 0.0)


def _metrics_model_add(metrics: Dict[str, Any], model_name: str, seconds: float, items: int = 0, calls: int = 1) -> None:
    m = metrics["models"].setdefault(model_name, {"calls": 0, "seconds": 0.0, "items": 0})
    m["calls"] += int(calls)
    m["seconds"] += float(seconds)
    m["items"] += int(items)


def _fmt_secs(s: float) -> str:
    if not np.isfinite(s) or s < 0:
        return "unknown"
    s_i = int(s)
    h = s_i // 3600
    m = (s_i % 3600) // 60
    sec = s_i % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _print_metrics_report(
    *,
    scope: str,
    metrics: Dict[str, Any],
    total_expected_pages: Optional[int] = None,
) -> None:
    pages = int(metrics.get("pages_processed", 0))
    elapsed_total = float(metrics.get("timings", {}).get("page_total", 0.0) or 0.0)
    rate = (pages / elapsed_total) if elapsed_total > 0 else 0.0

    header = f"[bold yellow]{scope} metrics[/bold yellow] pages={pages}"
    if total_expected_pages is not None:
        header += f"/{int(total_expected_pages)}"
    header += f" elapsed={_fmt_secs(elapsed_total)} rate={rate:.3f} pages/s"
    console.print(header, highlight=False)

    # Stage timing breakdown
    t = metrics.get("timings", {}) or {}
    console.print(
        "  stages: "
        f"text={t.get('page_text', 0.0):.2f}s, "
        f"page_elements={t.get('page_elements', 0.0):.2f}s, "
        f"table_structure={t.get('table_structure', 0.0):.2f}s, "
        f"graphic_elements={t.get('graphic_elements', 0.0):.2f}s, "
        f"ocr={t.get('ocr', 0.0):.2f}s, "
        f"embedding={t.get('embedding', 0.0):.2f}s",
        highlight=False,
    )

    # Model invocation breakdown (sorted by total time desc)
    models = metrics.get("models", {}) or {}
    for name, m in sorted(models.items(), key=lambda kv: float(kv[1].get("seconds", 0.0) or 0.0), reverse=True):
        calls = int(m.get("calls", 0) or 0)
        secs = float(m.get("seconds", 0.0) or 0.0)
        items = int(m.get("items", 0) or 0)
        per_call = (secs / calls) if calls else 0.0
        console.print(
            f"  model={name} calls={calls} items={items} seconds={secs:.2f} per_call={per_call:.3f}s",
            highlight=False,
        )

    c = metrics.get("counts", {}) or {}
    console.print(
        "  counts: "
        f"table_regions={int(c.get('table_regions', 0) or 0)}, "
        f"graphic_regions={int(c.get('graphic_regions', 0) or 0)}, "
        f"table_structure_detections={int(c.get('table_structure_detections', 0) or 0)}, "
        f"graphic_elements_detections={int(c.get('graphic_elements_detections', 0) or 0)}, "
        f"ocr_crops={int(c.get('ocr_crops', 0) or 0)}, "
        f"embedding_segments={int(c.get('embedding_segments', 0) or 0)}",
        highlight=False,
    )


def _extract_pdfium_page_text(pdf: pdfium.PdfDocument, page_idx: int) -> str:
    """
    Extract embedded text from a PDF page using PDFium (not OCR).
    """
    try:
        page = pdf.get_page(page_idx)
        try:
            textpage = page.get_textpage()
            try:
                # pypdfium2: get_text_range() returns full page text when called without args
                return textpage.get_text_range() or ""
            finally:
                # Some pypdfium2 versions expose close(); others rely on GC.
                close_fn = getattr(textpage, "close", None)
                if callable(close_fn):
                    close_fn()
        finally:
            page.close()
    except Exception:
        # Keep pipeline resilient; OCR is handled elsewhere.
        return ""


def _to_bbox_list(bbox: Any) -> List[float]:
    if isinstance(bbox, torch.Tensor):
        return [float(x) for x in bbox.detach().cpu().tolist()]
    if isinstance(bbox, np.ndarray):
        return [float(x) for x in bbox.tolist()]
    if isinstance(bbox, (list, tuple)):
        return [float(x) for x in bbox]
    # Fallback: best-effort cast
    return [float(x) for x in list(bbox)]

def _to_scalar_int(v: Any) -> Any:
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return int(v.item())
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def _to_scalar_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        return float(v.item())
    if isinstance(v, (np.floating, float, int)):
        return float(v)
    return None


def _crop_tensor_normalized_xyxy(image_tensor: torch.Tensor, bbox_xyxy_norm: Any) -> torch.Tensor:
    """
    Crop a CHW image tensor using normalized [xmin, ymin, xmax, ymax] in [0, 1].
    """
    bbox = _to_bbox_list(bbox_xyxy_norm)
    xmin_n, ymin_n, xmax_n, ymax_n = bbox
    _, H, W = image_tensor.shape
    x1 = int(xmin_n * W)
    y1 = int(ymin_n * H)
    x2 = int(xmax_n * W)
    y2 = int(ymax_n * H)

    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))
    return image_tensor[:, y1:y2, x1:x2]


def _iter_detection_dicts(preds: Any) -> Iterable[Dict[str, Any]]:
    """
    Normalize model outputs into an iterable of dicts with 'boxes'/'labels'/'scores'.
    Many Nemotron HF wrappers return List[Dict[...]]; some return Dict[...].
    """
    if preds is None:
        return []
    if isinstance(preds, dict):
        return [preds]
    if isinstance(preds, list):
        return preds
    return []

def _chunked(seq: List[Any], batch_size: int) -> Iterable[List[Any]]:
    bs = max(1, int(batch_size))
    for i in range(0, len(seq), bs):
        yield seq[i : i + bs]


def _resize_pad_tensor(
    img: torch.Tensor, target_hw: Tuple[int, int] = (1024, 1024), pad_value: float = 114.0
) -> torch.Tensor:
    """
    Resize+pad CHW image tensor to fixed size (preserve aspect ratio).
    Returns same dtype as input.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(img.shape)}")
    C, H, W = img.shape
    th, tw = target_hw
    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid image shape: {tuple(img.shape)}")

    orig_dtype = img.dtype
    x = img.float()
    scale = min(th / H, tw / W)
    nh = max(1, int(H * scale))
    nw = max(1, int(W * scale))
    x = F.interpolate(x.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False).squeeze(0)
    x = torch.clamp(x, 0, 255)
    pad_b = th - nh
    pad_r = tw - nw
    x = F.pad(x, (0, pad_r, 0, pad_b), value=float(pad_value))
    return x.to(dtype=orig_dtype)


def process_pdf_pages(
    pdf_path,
    page_elements,
    table_structure,
    graphic_elements,
    ocr,
    tokenizer,
    embedding_model,
    device="cuda",
    dpi=150.0,
    ocr_batch_size: int = 32,
    table_structure_batch_size: int = 16,
    graphic_elements_batch_size: int = 16,
    embedding_batch_size: int = 16,
):
    """
    Generator that processes PDF pages one at a time, yielding results for each page.
    
    This is memory-efficient as it processes pages as they're rendered without loading
    all pages into memory at once.
    
    Yields:
        Tuple of (page_number, per-page result dict)
    """
    
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        # Use the generator to iterate through rendered PDF pages
        for page_tensor_info in iter_pdf_page_tensors(pdf_path, dpi=dpi, device=device):
            page_t0 = time.perf_counter()
            page_metrics = _empty_run_metrics()
            page_metrics["pages_processed"] = 1

            page_number = page_tensor_info.page_number
            tensor = page_tensor_info.tensor  # Shape: [3, H, W]
            bitmap_shape = (page_tensor_info.original_height, page_tensor_info.original_width)

            t0 = time.perf_counter()
            page_text = _extract_pdfium_page_text(pdf, page_number)
            page_metrics["timings"]["page_text"] += time.perf_counter() - t0

            # OCR is ONLY performed on detections from table_structure/graphic_elements.
            table_structure_text: List[str] = []
            graphic_elements_text: List[str] = []
            table_structure_detections: List[Dict[str, Any]] = []
            graphic_elements_detections: List[Dict[str, Any]] = []

            with torch.inference_mode():
                # Page element detection to find candidate table/graphic regions
                t0 = time.perf_counter()
                det_input = page_elements.preprocess(tensor)
                det_preds = page_elements.invoke(det_input, bitmap_shape)
                boxes, labels, scores = page_elements.postprocess(det_preds)
                dt = time.perf_counter() - t0
                page_metrics["timings"]["page_elements"] += dt
                _metrics_model_add(page_metrics, "page_elements", dt, items=1, calls=1)

                # Split into table regions vs graphic regions first (so we can micro-batch downstream work)
                table_regions: List[Dict[str, Any]] = []
                graphic_regions: List[Dict[str, Any]] = []

                for label, box, score in zip(labels, boxes, scores):
                    label_i = int(_to_scalar_int(label))
                    region = {
                        "page_element_bbox": box,
                        "page_element_score": score,
                    }
                    if label_i == 0:
                        region["crop"] = _crop_tensor_normalized_xyxy(tensor, box)
                        table_regions.append(region)
                    elif label_i in [1, 2, 3]:
                        region["crop"] = _crop_tensor_normalized_xyxy(tensor, box)
                        graphic_regions.append(region)

                page_metrics["counts"]["table_regions"] += len(table_regions)
                page_metrics["counts"]["graphic_regions"] += len(graphic_regions)

                # Run table_structure in micro-batches (default 4), then queue OCR crops
                ocr_tasks: List[Dict[str, Any]] = []

                for batch in _chunked(table_regions, table_structure_batch_size):
                    for region in batch:
                        table_crop = region["crop"]
                        crop_shape = (int(table_crop.shape[1]), int(table_crop.shape[2]))
                        t0 = time.perf_counter()
                        ts_input = table_structure.preprocess(table_crop, crop_shape)
                        ts_preds = table_structure.invoke(ts_input, crop_shape)
                        dt = time.perf_counter() - t0
                        page_metrics["timings"]["table_structure"] += dt
                        _metrics_model_add(page_metrics, "table_structure", dt, items=1, calls=1)

                        for det in _iter_detection_dicts(ts_preds):
                            det_boxes = det.get("boxes", [])
                            det_labels = det.get("labels", [])
                            det_scores = det.get("scores", [])
                            n = min(len(det_boxes), len(det_labels)) if hasattr(det_boxes, "__len__") else 0
                            for i in range(n):
                                cell_box = det_boxes[i]
                                cell_label = det_labels[i]
                                cell_score = det_scores[i] if i < len(det_scores) else None

                                # Create detection record now; fill ocr_text after batched OCR
                                det_rec: Dict[str, Any] = {
                                    "page_element_bbox": _to_bbox_list(region["page_element_bbox"]),
                                    "page_element_score": _to_scalar_float(region["page_element_score"]),
                                    "detection_bbox": _to_bbox_list(cell_box),
                                    "detection_label": _to_scalar_int(cell_label),
                                    "detection_score": _to_scalar_float(cell_score),
                                    "ocr_text": "",
                                }
                                table_structure_detections.append(det_rec)
                                ocr_tasks.append(
                                    {
                                        "kind": "table_structure",
                                        "crop": _crop_tensor_normalized_xyxy(table_crop, cell_box),
                                        "det_rec": det_rec,
                                    }
                                )

                # Run graphic_elements in micro-batches (default 4), then queue OCR crops
                for batch in _chunked(graphic_regions, graphic_elements_batch_size):
                    for region in batch:
                        graphic_crop = region["crop"]
                        crop_shape = (int(graphic_crop.shape[1]), int(graphic_crop.shape[2]))
                        t0 = time.perf_counter()
                        ge_input = graphic_elements.preprocess(graphic_crop)
                        ge_preds = graphic_elements.invoke(ge_input, crop_shape)
                        dt = time.perf_counter() - t0
                        page_metrics["timings"]["graphic_elements"] += dt
                        _metrics_model_add(page_metrics, "graphic_elements", dt, items=1, calls=1)

                        for det in _iter_detection_dicts(ge_preds):
                            det_boxes = det.get("boxes", [])
                            det_labels = det.get("labels", [])
                            det_scores = det.get("scores", [])
                            n = min(len(det_boxes), len(det_labels)) if hasattr(det_boxes, "__len__") else 0
                            for i in range(n):
                                el_box = det_boxes[i]
                                el_label = det_labels[i]
                                el_score = det_scores[i] if i < len(det_scores) else None

                                det_rec = {
                                    "page_element_bbox": _to_bbox_list(region["page_element_bbox"]),
                                    "page_element_score": _to_scalar_float(region["page_element_score"]),
                                    "detection_bbox": _to_bbox_list(el_box),
                                    "detection_label": _to_scalar_int(el_label),
                                    "detection_score": _to_scalar_float(el_score),
                                    "ocr_text": "",
                                }
                                graphic_elements_detections.append(det_rec)
                                ocr_tasks.append(
                                    {
                                        "kind": "graphic_elements",
                                        "crop": _crop_tensor_normalized_xyxy(graphic_crop, el_box),
                                        "det_rec": det_rec,
                                    }
                                )

                page_metrics["counts"]["table_structure_detections"] += len(table_structure_detections)
                page_metrics["counts"]["graphic_elements_detections"] += len(graphic_elements_detections)
                page_metrics["counts"]["ocr_crops"] += len(ocr_tasks)

                # OCR in batches (default 8). Try true batching; fall back to per-crop if unsupported.
                for batch in _chunked(ocr_tasks, ocr_batch_size):
                    # Normalize crop shapes so we can stack into a real batch
                    batch_imgs = [_resize_pad_tensor(t["crop"], target_hw=(1024, 1024)) for t in batch]
                    batch_tensor = torch.stack(batch_imgs, dim=0)

                    batch_texts: List[str] = []
                    try:
                        t0 = time.perf_counter()
                        ocr_out = ocr.invoke(batch_tensor)
                        dt = time.perf_counter() - t0
                        page_metrics["timings"]["ocr"] += dt
                        _metrics_model_add(page_metrics, "ocr", dt, items=len(batch), calls=1)
                        # Common cases:
                        # - batched: list length B, each entry list[dict] or dict with 'text'
                        # - non-batched: list[dict] for single image (then we'll fallback)
                        if isinstance(ocr_out, list) and len(ocr_out) == len(batch):
                            for entry in ocr_out:
                                if isinstance(entry, list):
                                    batch_texts.append(
                                        " ".join([str(p.get("text", "")) for p in entry]).strip()
                                    )
                                elif isinstance(entry, dict):
                                    batch_texts.append(str(entry.get("text", "")).strip())
                                else:
                                    batch_texts.append(str(entry).strip())
                        else:
                            raise RuntimeError("OCR output did not match batch size (fallback to per-crop).")
                    except Exception:
                        # Fallback: run OCR per crop (still chunked for progress/structure)
                        batch_texts = []
                        for t in batch:
                            t0 = time.perf_counter()
                            out = ocr.invoke(t["crop"])
                            dt = time.perf_counter() - t0
                            page_metrics["timings"]["ocr"] += dt
                            _metrics_model_add(page_metrics, "ocr", dt, items=1, calls=1)
                            if isinstance(out, list):
                                batch_texts.append(" ".join([str(p.get("text", "")) for p in out]).strip())
                            elif isinstance(out, dict):
                                batch_texts.append(str(out.get("text", "")).strip())
                            else:
                                batch_texts.append(str(out).strip())

                    for t, txt in zip(batch, batch_texts):
                        t["det_rec"]["ocr_text"] = txt
                        if txt:
                            if t["kind"] == "table_structure":
                                table_structure_text.append(txt)
                            else:
                                graphic_elements_text.append(txt)

                # Build embeddings in micro-batches (default 4) over text segments
                text_segments = [t for t in [page_text] + table_structure_text + graphic_elements_text if t]
                page_metrics["counts"]["embedding_segments"] += len(text_segments)
                embedding_results: List[Any] = []
                for seg_batch in _chunked(text_segments, embedding_batch_size):
                    batch_documents = tokenizer(
                        seg_batch, padding=True, truncation=True, return_tensors="pt"
                    ).to("cuda")
                    t0 = time.perf_counter()
                    out = embedding_model(**batch_documents)
                    dt = time.perf_counter() - t0
                    page_metrics["timings"]["embedding"] += dt
                    _metrics_model_add(page_metrics, "embedding", dt, items=len(seg_batch), calls=1)
                    embedding_results.append(out)

            page_metrics["timings"]["page_total"] += time.perf_counter() - page_t0

            yield (
                page_number,
                {
                    "page_text": page_text,
                    "table_structure_text": table_structure_text,
                    "graphic_elements_text": graphic_elements_text,
                    "table_structure_detections": table_structure_detections,
                    "graphic_elements_detections": graphic_elements_detections,
                    "embedding": embedding_results,
                    "metrics": page_metrics,
                },
            )
    finally:
        pdf.close()

def run_pipeline(
    pdf_files: List[str],
    page_elements: NemotronPageElementsV3,
    table_structure: NemotronTableStructureV1,
    graphic_elements: NemotronGraphicElementsV1,
    ocr: NemotronOCRV1,
    tokenizer,
    embedding_model,
    raw_output_dir: Optional[Path] = None,
    dpi: float = 150.0,
    expected_total_pages: int = 54730,
    eta_print_interval_seconds: float = 5.0,
):
    start_time = time.time()
    last_eta_print = start_time
    total_pages_processed = 0
    results = []
    run_metrics = _empty_run_metrics()
    
    for pdf_idx, pdf_path in enumerate(pdf_files, start=1):
        console.print(f"[bold cyan]Processing:[/bold cyan] {pdf_path}")
        
        # Collect results for this PDF
        pages: List[Dict[str, Any]] = []
        pdf_metrics = _empty_run_metrics()
        pages_in_pdf = 0
        
        # Process pages one at a time using the generator
        for page_number, page_result in process_pdf_pages(
            pdf_path,
            page_elements,
            table_structure,
            graphic_elements,
            ocr,
            tokenizer,
            embedding_model,
            device="cuda",
            dpi=dpi,
        ):
            pages_in_pdf += 1
            total_pages_processed += 1
            pages.append(
                {
                    "page_number": int(page_number),
                    "page_text": page_result["page_text"],
                    "table_structure_text": page_result["table_structure_text"],
                    "graphic_elements_text": page_result["graphic_elements_text"],
                    "table_structure_detections": page_result["table_structure_detections"],
                    "graphic_elements_detections": page_result["graphic_elements_detections"],
                }
            )

            # Update metrics (per page + per pdf + global)
            page_metrics = page_result.get("metrics") or _empty_run_metrics()
            _metrics_add(pdf_metrics, page_metrics)
            _metrics_add(run_metrics, page_metrics)

            # Per-page progress logging (preview + detection counts)
            ts_n = len(page_result.get("table_structure_detections", []) or [])
            ge_n = len(page_result.get("graphic_elements_detections", []) or [])
            preview_src = " ".join(
                [
                    (page_result.get("page_text") or "").replace("\n", " ").strip(),
                    " ".join(page_result.get("table_structure_text", []) or []),
                    " ".join(page_result.get("graphic_elements_text", []) or []),
                ]
            ).strip()
            preview = (preview_src[:200] + ("…" if len(preview_src) > 200 else "")).strip()
            console.print(
                f"[cyan]Page[/cyan] {int(page_number) + 1} "
                f"(idx={int(page_number)}): "
                f"table_structure={ts_n}, graphic_elements={ge_n} | "
                f"preview='{preview}'",
                markup=True,
                highlight=False,
            )

            # Per-page metrics report (delta + cumulative totals)
            _print_metrics_report(scope="Last page", metrics=page_metrics)
            _print_metrics_report(scope="Cumulative run", metrics=run_metrics, total_expected_pages=expected_total_pages)

            # Periodic ETA (every ~5 seconds by default)
            now = time.time()
            if (now - last_eta_print) >= float(eta_print_interval_seconds):
                elapsed = now - start_time
                rate = (total_pages_processed / elapsed) if elapsed > 0 else 0.0  # pages/sec
                remaining_pages = max(0, int(expected_total_pages) - int(total_pages_processed))
                eta_seconds = (remaining_pages / rate) if rate > 0 else float("inf")

                console.print(
                    f"[magenta]ETA[/magenta] elapsed={_fmt_secs(elapsed)} "
                    f"processed={total_pages_processed}/{expected_total_pages} "
                    f"rate={rate:.3f} pages/s "
                    f"remaining={remaining_pages} "
                    f"eta={_fmt_secs(eta_seconds)}",
                    markup=True,
                    highlight=False,
                )
                last_eta_print = now
            
        # Summary for this PDF
        console.print(
            f"Completed {pages_in_pdf} pages from {pdf_path}. "
            f"PDF {pdf_idx} of {len(pdf_files)}. "
            f"Total pages processed: {total_pages_processed}. "
            f"Current Runtime: {time.time() - start_time:.2f} seconds"
        )
        _print_metrics_report(scope=f"PDF {pdf_idx} summary", metrics=pdf_metrics)

        # Dictionary with results ...
        pdf_results = {"pages": pages}
        
        # Save raw OCR results if requested
        if raw_output_dir is not None:
            raw_output_dir = Path(raw_output_dir)
            raw_output_dir.mkdir(parents=True, exist_ok=True)
            pdf_path_obj = Path(pdf_path)
            output_json_path = raw_output_dir / pdf_path_obj.with_suffix('.page_raw_ocr_results.json').name
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(pdf_results, f, ensure_ascii=False, indent=2)
        
        # Store results for this PDF
        results.append({
            "pdf_path": pdf_path,
            "pages_processed": pages_in_pdf,
            "pages": pages,
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
    ocr = NemotronOCRV1(model_dir="/raid/jdyer/slimgest/models/nemotron-ocr-v1/checkpoints")
    hf_cache_dir = str(Path.home() / ".cache" / "huggingface")
    tokenizer = llama_nemotron_embed_1b_v2.load_tokenizer(cache_dir=hf_cache_dir, force_download=False)
    embedding_model = llama_nemotron_embed_1b_v2.load_model(
        device="cuda", trust_remote_code=True, cache_dir=hf_cache_dir, force_download=False
    )


    
    if input_dir.is_file():
        pdf_files = [input_dir]
    else:
        pdf_files = [
            str(f) for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".pdf"
        ]

    console.print(f"Processing {len(pdf_files)} PDFs")

    run_pipeline(
        pdf_files,
        page_elements,
        table_structure,
        graphic_elements,
        ocr,
        tokenizer,
        embedding_model,
        raw_output_dir=raw_output_dir,
    )