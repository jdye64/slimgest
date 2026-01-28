from __future__ import annotations

import concurrent.futures as cf
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pypdfium2 as pdfium
import torch
import torch.nn.functional as F
import typer
from rich.console import Console
from tqdm import tqdm

from slimgest.local.pipeline_utils import TimerBank, extract_text_best_effort, fmt_seconds_hms, resize_pad_tensor
from slimgest.local.stages._io import (
    PAGE_ELEMENT_LABELS,
    bbox_region_to_page,
    crop_tensor_normalized_xyxy,
    iter_detection_dicts,
    to_bbox_list,
    to_scalar_float,
    to_scalar_int,
    write_json,
)
from slimgest.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder
from slimgest.model.local.nemotron_graphic_elements_v1 import NemotronGraphicElementsV1
from slimgest.model.local.nemotron_ocr_v1 import NemotronOCRV1
from slimgest.model.local.nemotron_page_elements_v3 import NemotronPageElementsV3
from slimgest.model.local.nemotron_table_structure_v1 import NemotronTableStructureV1
from slimgest.pdf.render import iter_pdf_page_tensors

app = typer.Typer(
    help="Simple high-performance PDF -> embeddings pipeline (in-memory stages + detailed metrics + tqdm progress)."
)

# NOTE: The legacy implementation (below) is still present for now; it expects these symbols.
console = Console()

#
# New implementation (2026-01): in-memory stage2-6 pipeline with detailed metrics.
#

def sp_chunked(seq: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    bs = max(1, int(batch_size))
    for i in range(0, len(seq), bs):
        yield list(seq[i : i + bs])


def sp_extract_pdfium_page_text(pdf: pdfium.PdfDocument, page_idx: int) -> str:
    """
    Extract embedded PDF text via PDFium (not OCR).
    """
    try:
        page = pdf.get_page(int(page_idx))
        try:
            textpage = page.get_textpage()
            try:
                return (textpage.get_text_range() or "").strip()
            finally:
                close_fn = getattr(textpage, "close", None)
                if callable(close_fn):
                    close_fn()
        finally:
            page.close()
    except Exception:
        return ""


def sp_new_model_metrics() -> Dict[str, Any]:
    return {"calls": 0, "seconds": 0.0, "items": 0, "batch_size_hist": {}}


def sp_model_add(
    page_metrics: Dict[str, Any],
    *,
    model: str,
    seconds: float,
    items: int,
    batch_size: Optional[int] = None,
    calls: int = 1,
) -> None:
    models = page_metrics.setdefault("models", {})
    m = models.setdefault(model, sp_new_model_metrics())
    m["calls"] = int(m.get("calls", 0)) + int(calls)
    m["seconds"] = float(m.get("seconds", 0.0)) + float(seconds)
    m["items"] = int(m.get("items", 0)) + int(items)
    if batch_size is not None:
        hist = m.setdefault("batch_size_hist", {})
        k = str(int(batch_size))
        hist[k] = int(hist.get(k, 0)) + 1


def sp_empty_page_metrics(page_index: int) -> Dict[str, Any]:
    return {"page_index": int(page_index), "counts": {}, "timing_s": {}, "models": {}, "errors": [], "outputs": {}}


def sp_ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sp_list_pdfs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])


def sp_count_pages(pdf_path: Path) -> int:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return int(len(pdf))
    finally:
        pdf.close()


def sp_embed_out_path(output_embeddings_dir: Path, pdf_path: Path, page_index: int) -> Path:
    return output_embeddings_dir / f"{pdf_path.stem}_page{int(page_index):04d}.embeddings.pt"


def sp_pdf_metrics_out_path(output_metrics_dir: Path, pdf_path: Path) -> Path:
    return output_metrics_dir / f"{pdf_path.stem}.metrics.json"


def sp_stage2_page_elements(
    *,
    model: NemotronPageElementsV3,
    page_tensor: torch.Tensor,
    bitmap_shape_hw: Tuple[int, int],
    page_metrics: Dict[str, Any],
    timers: TimerBank,
) -> List[Dict[str, Any]]:
    with torch.inference_mode():
        with timers.timed("stage2.page_elements.preprocess"):
            inp = model.preprocess(page_tensor)
        t0 = time.perf_counter()
        preds = model.invoke(inp, bitmap_shape_hw)
        dt = time.perf_counter() - t0
        sp_model_add(page_metrics, model="page_elements_v3", seconds=dt, items=1, calls=1)
        with timers.timed("stage2.page_elements.postprocess"):
            boxes, labels, scores = model.postprocess(preds)

    dets: List[Dict[str, Any]] = []
    for box, lab, score in zip(boxes, labels, scores):
        lab_i = int(to_scalar_int(lab))
        dets.append(
            {
                "bbox_xyxy_norm": to_bbox_list(box),
                "label": int(lab_i),
                "label_name": PAGE_ELEMENT_LABELS.get(int(lab_i), f"unknown_{int(lab_i)}"),
                "score": to_scalar_float(score),
            }
        )
    page_metrics["counts"]["stage2_page_elements"] = int(len(dets))
    return dets


def sp_stage3_graphic_elements(
    *,
    model: NemotronGraphicElementsV1,
    page_tensor: torch.Tensor,
    graphic_regions: Sequence[Dict[str, Any]],
    page_metrics: Dict[str, Any],
    timers: TimerBank,
    region_batch_size: int,
) -> None:
    det_count = 0
    with torch.inference_mode():
        for batch in sp_chunked(list(graphic_regions), region_batch_size):
            inputs: List[torch.Tensor] = []
            shapes: List[Tuple[int, int]] = []
            region_bboxes: List[List[float]] = []
            for r in batch:
                bbox = r["bbox_xyxy_norm"]
                with timers.timed("stage3.graphic.crop"):
                    crop = crop_tensor_normalized_xyxy(page_tensor, bbox)
                with timers.timed("stage3.graphic.preprocess"):
                    inputs.append(model.preprocess(crop))
                shapes.append((int(crop.shape[1]), int(crop.shape[2])))
                region_bboxes.append([float(x) for x in bbox])

            t0 = time.perf_counter()
            preds_list: List[Any] = [model.invoke(inp, sh) for inp, sh in zip(inputs, shapes)]
            dt = time.perf_counter() - t0
            sp_model_add(
                page_metrics,
                model="graphic_elements_v1",
                seconds=dt,
                items=len(inputs),
                batch_size=len(inputs),
                calls=1,
            )

            for preds, region_bbox in zip(preds_list, region_bboxes):
                for det in iter_detection_dicts(preds):
                    # NOTE: torch.Tensor does not support truthiness (e.g. `x or []`),
                    # so only fall back to [] when the value is actually None.
                    boxes = det.get("boxes")
                    if boxes is None:
                        boxes = []
                    labels = det.get("labels")
                    if labels is None:
                        labels = []
                    scores = det.get("scores")
                    if scores is None:
                        scores = []
                    n = min(len(boxes), len(labels)) if hasattr(boxes, "__len__") else 0
                    det_count += int(n)
                    # Convert to page coords for correctness, even if we only count.
                    for j in range(int(n)):
                        _ = bbox_region_to_page(
                            region_bbox_xyxy_norm_in_page=region_bbox,
                            det_bbox_xyxy_norm_in_region=to_bbox_list(boxes[j]),
                        )
                        _ = to_scalar_int(labels[j])
                        _ = to_scalar_float(scores[j] if j < len(scores) else None)

    page_metrics["counts"]["stage3_graphic_regions"] = int(len(graphic_regions))
    page_metrics["counts"]["stage3_graphic_detections"] = int(det_count)


def sp_stage4_table_structure(
    *,
    model: NemotronTableStructureV1,
    page_tensor: torch.Tensor,
    table_regions: Sequence[Dict[str, Any]],
    page_metrics: Dict[str, Any],
    timers: TimerBank,
    region_batch_size: int,
) -> None:
    det_count = 0
    with torch.inference_mode():
        for batch in sp_chunked(list(table_regions), region_batch_size):
            inputs: List[torch.Tensor] = []
            shapes: List[Tuple[int, int]] = []
            region_bboxes: List[List[float]] = []
            for r in batch:
                bbox = r["bbox_xyxy_norm"]
                with timers.timed("stage4.table.crop"):
                    crop = crop_tensor_normalized_xyxy(page_tensor, bbox)
                crop_shape = (int(crop.shape[1]), int(crop.shape[2]))
                with timers.timed("stage4.table.preprocess"):
                    inputs.append(model.preprocess(crop, crop_shape))
                shapes.append(crop_shape)
                region_bboxes.append([float(x) for x in bbox])

            t0 = time.perf_counter()
            preds_list: List[Any] = [model.invoke(inp, sh) for inp, sh in zip(inputs, shapes)]
            dt = time.perf_counter() - t0
            sp_model_add(
                page_metrics,
                model="table_structure_v1",
                seconds=dt,
                items=len(inputs),
                batch_size=len(inputs),
                calls=1,
            )

            for preds, region_bbox in zip(preds_list, region_bboxes):
                for det in iter_detection_dicts(preds):
                    # NOTE: torch.Tensor does not support truthiness (e.g. `x or []`),
                    # so only fall back to [] when the value is actually None.
                    boxes = det.get("boxes")
                    if boxes is None:
                        boxes = []
                    labels = det.get("labels")
                    if labels is None:
                        labels = []
                    scores = det.get("scores")
                    if scores is None:
                        scores = []
                    n = min(len(boxes), len(labels)) if hasattr(boxes, "__len__") else 0
                    det_count += int(n)
                    for j in range(int(n)):
                        _ = bbox_region_to_page(
                            region_bbox_xyxy_norm_in_page=region_bbox,
                            det_bbox_xyxy_norm_in_region=to_bbox_list(boxes[j]),
                        )
                        _ = to_scalar_int(labels[j])
                        _ = to_scalar_float(scores[j] if j < len(scores) else None)

    page_metrics["counts"]["stage4_table_regions"] = int(len(table_regions))
    page_metrics["counts"]["stage4_table_detections"] = int(det_count)


def sp_stage5_ocr(
    *,
    ocr: NemotronOCRV1,
    page_tensor: torch.Tensor,
    ocr_regions: Sequence[Dict[str, Any]],
    page_metrics: Dict[str, Any],
    timers: TimerBank,
    ocr_batch_size: int,
    resize_to_1024: bool,
) -> List[Dict[str, Any]]:
    if not ocr_regions:
        page_metrics["counts"]["stage5_ocr_regions"] = 0
        page_metrics["counts"]["stage5_ocr_nonempty"] = 0
        return []

    crops: List[torch.Tensor] = []
    region_meta: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for r in ocr_regions:
            bbox = r["bbox_xyxy_norm"]
            with timers.timed("stage5.ocr.crop"):
                crop = crop_tensor_normalized_xyxy(page_tensor, bbox)
            if resize_to_1024:
                with timers.timed("stage5.ocr.resize_pad"):
                    crop = resize_pad_tensor(crop, target_hw=(1024, 1024))
            crops.append(crop)
            region_meta.append({"bbox_xyxy_norm_in_page": [float(x) for x in bbox], "source": r})

        outs: List[Any] = []
        # Remote endpoint: true batching; local: per-crop invocations (but still grouped for reporting).
        for batch in sp_chunked(crops, ocr_batch_size):
            if not batch:
                continue
            if getattr(ocr, "_endpoint", None) is not None:
                t0 = time.perf_counter()
                b = torch.stack([t if t.ndim == 3 else t.squeeze(0) for t in batch], dim=0)
                out = ocr.invoke(b)
                dt = time.perf_counter() - t0
                outs.extend(list(out))
                sp_model_add(
                    page_metrics,
                    model="nemotron_ocr_v1",
                    seconds=dt,
                    items=len(batch),
                    batch_size=len(batch),
                    calls=1,
                )
            else:
                for t in batch:
                    t0 = time.perf_counter()
                    out = ocr.invoke(t)
                    dt = time.perf_counter() - t0
                    outs.append(out)
                    sp_model_add(page_metrics, model="nemotron_ocr_v1", seconds=dt, items=1, batch_size=1, calls=1)

    if len(outs) != len(region_meta):
        page_metrics["errors"].append(f"stage5_ocr_output_mismatch outs={len(outs)} regions={len(region_meta)}")

    n = min(len(outs), len(region_meta))
    regions_out: List[Dict[str, Any]] = []
    nonempty = 0
    with timers.timed("stage5.ocr.postprocess"):
        for meta, raw in zip(region_meta[:n], outs[:n]):
            txt = extract_text_best_effort(raw)
            if txt:
                nonempty += 1
            src = meta["source"]
            regions_out.append(
                {
                    "bbox_xyxy_norm_in_page": meta["bbox_xyxy_norm_in_page"],
                    "ocr_text": txt,
                    "label": src.get("label"),
                    "label_name": src.get("label_name"),
                    "score": src.get("score"),
                }
            )
    page_metrics["counts"]["stage5_ocr_regions"] = int(len(ocr_regions))
    page_metrics["counts"]["stage5_ocr_nonempty"] = int(nonempty)
    return regions_out


def sp_stage6_embed_and_save(
    *,
    embedder: LlamaNemotronEmbed1BV2Embedder,
    texts: List[str],
    text_kinds: List[str],
    bboxes_xyxy_norm_in_page: List[List[float]],
    out_path: Path,
    page_metrics: Dict[str, Any],
    timers: TimerBank,
    embedding_batch_size: int,
    pdf_path: Path,
    page_index: int,
) -> None:
    sp_ensure_dir(out_path.parent)
    vectors: List[torch.Tensor] = []
    with torch.inference_mode():
        t0 = time.perf_counter()
        emb = embedder.embed(texts, batch_size=int(embedding_batch_size)) if texts else torch.empty((0, 0))
        dt = time.perf_counter() - t0
        sp_model_add(
            page_metrics,
            model="llama_nemotron_embed_1b_v2",
            seconds=dt,
            items=len(texts),
            batch_size=int(embedding_batch_size),
            calls=1 if texts else 0,
        )
        with timers.timed("stage6.embed.normalize"):
            if isinstance(emb, torch.Tensor) and emb.ndim == 2 and int(emb.shape[0]) == int(len(texts)):
                for i in range(int(emb.shape[0])):
                    v = emb[i].float()
                    v = v / (v.norm(p=2) + 1e-12)
                    vectors.append(v.cpu())

    with timers.timed("stage6.embed.save"):
        torch.save(
            {
                "schema_version": 1,
                "stage": 6,
                "model": "llama_nemotron_embed_1b_v2",
                "pdf_path": str(pdf_path),
                "page_index": int(page_index),
                "texts": texts,
                "text_kinds": text_kinds,
                "bboxes_xyxy_norm_in_page": bboxes_xyxy_norm_in_page,
                "embeddings": vectors,
            },
            out_path,
        )


def sp_compute_run_summary_and_suggestions(all_pdf_metrics: List[Dict[str, Any]]) -> str:
    model_totals: Dict[str, Dict[str, float]] = {}
    stage_totals: Dict[str, float] = {}
    pages = 0
    pdfs = len(all_pdf_metrics)
    for pm in all_pdf_metrics:
        for p in pm.get("pages", []) or []:
            pages += 1
            for k, v in (p.get("timing_s") or {}).items():
                stage_totals[k] = float(stage_totals.get(k, 0.0) + float(v))
            for name, m in (p.get("models") or {}).items():
                cur = model_totals.setdefault(name, {"seconds": 0.0, "calls": 0.0, "items": 0.0})
                cur["seconds"] += float(m.get("seconds", 0.0) or 0.0)
                cur["calls"] += float(m.get("calls", 0) or 0.0)
                cur["items"] += float(m.get("items", 0) or 0.0)

    lines: List[str] = []
    lines.append("slimgest simple pipeline run summary")
    lines.append(f"pdfs={pdfs} pages={pages}")
    lines.append("")
    lines.append("Top model hotspots (by total seconds):")
    for name, m in sorted(model_totals.items(), key=lambda kv: kv[1]["seconds"], reverse=True)[:10]:
        calls = int(m["calls"])
        items = int(m["items"])
        secs = float(m["seconds"])
        per_call = secs / calls if calls else 0.0
        per_item = secs / items if items else 0.0
        lines.append(
            f"- {name}: seconds={secs:.2f} calls={calls} items={items} per_call_s={per_call:.3f} per_item_s={per_item:.6f}"
        )
    lines.append("")
    lines.append("Top timed components (by total seconds):")
    for k, v in sorted(stage_totals.items(), key=lambda kv: kv[1], reverse=True)[:15]:
        lines.append(f"- {k}: {float(v):.2f}s")
    lines.append("")
    lines.append("Suggestions (rule-of-thumb):")
    ocr_s = float(model_totals.get("nemotron_ocr_v1", {}).get("seconds", 0.0) or 0.0)
    emb_s = float(model_totals.get("llama_nemotron_embed_1b_v2", {}).get("seconds", 0.0) or 0.0)
    pe_s = float(model_totals.get("page_elements_v3", {}).get("seconds", 0.0) or 0.0)
    ts_s = float(model_totals.get("table_structure_v1", {}).get("seconds", 0.0) or 0.0)
    ge_s = float(model_totals.get("graphic_elements_v1", {}).get("seconds", 0.0) or 0.0)
    if max(ocr_s, emb_s, pe_s, ts_s, ge_s) <= 0:
        lines.append("- No model timings captured (did the run exit early?).")
        return "\n".join(lines).strip() + "\n"
    top = max(
        [
            ("nemotron_ocr_v1", ocr_s),
            ("llama_nemotron_embed_1b_v2", emb_s),
            ("page_elements_v3", pe_s),
            ("table_structure_v1", ts_s),
            ("graphic_elements_v1", ge_s),
        ],
        key=lambda kv: kv[1],
    )[0]
    if top == "nemotron_ocr_v1":
        lines.append("- OCR dominates: try increasing `--ocr-batch-size` and/or using `--ocr-endpoint` for true batching.")
        lines.append("- If resize/pad dominates: try `--no-resize-to-1024` (quality trade-off).")
    if top == "llama_nemotron_embed_1b_v2":
        lines.append("- Embeddings dominate: try increasing `--embedding-batch-size` and/or using a remote embedding endpoint.")
    if top in ("table_structure_v1", "graphic_elements_v1"):
        lines.append("- Region models dominate: try increasing `--region-batch-size` (within-page microbatches).")
        lines.append("- If GPU is saturated, try `--no-parallel-stages` to reduce contention.")
    if top == "page_elements_v3":
        lines.append("- Page elements dominates: consider lowering `--dpi` or using a remote endpoint.")
    return "\n".join(lines).strip() + "\n"


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
    embedder,
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
                    t0 = time.perf_counter()
                    out = embedder.embed(seg_batch, batch_size=embedding_batch_size)
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
    embedder,
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
            embedder,
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
    input_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=True),
    output_dir: Path = typer.Option(Path("./outputs/simple"), "--output-dir", help="Directory to write embeddings + metrics."),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Torch device for rendering + models."),
    dpi: float = typer.Option(150.0, "--dpi", help="PDF render DPI."),
    parallel_stages: bool = typer.Option(
        True,
        "--parallel-stages/--no-parallel-stages",
        help="Run graphic_elements, table_structure, and OCR concurrently per page.",
    ),
    region_batch_size: int = typer.Option(16, "--region-batch-size", min=1, help="Microbatch size for region crops."),
    ocr_batch_size: int = typer.Option(16, "--ocr-batch-size", min=1, help="OCR crops per batch (best with --ocr-endpoint)."),
    resize_to_1024: bool = typer.Option(True, "--resize-to-1024/--no-resize-to-1024", help="Resize+pad OCR crops to 1024x1024."),
    embedding_batch_size: int = typer.Option(64, "--embedding-batch-size", min=1, help="Texts per embedding batch."),
    page_elements_endpoint: Optional[str] = typer.Option(None, "--page-elements-endpoint", help="Optional page elements endpoint URL."),
    table_structure_endpoint: Optional[str] = typer.Option(None, "--table-structure-endpoint", help="Optional table structure endpoint URL."),
    graphic_elements_endpoint: Optional[str] = typer.Option(None, "--graphic-elements-endpoint", help="Optional graphic elements endpoint URL."),
    ocr_endpoint: Optional[str] = typer.Option(None, "--ocr-endpoint", help="Optional OCR endpoint URL."),
    embedding_endpoint: Optional[str] = typer.Option(None, "--embedding-endpoint", help="Optional embedding endpoint URL."),
    embedding_model_name: Optional[str] = typer.Option(None, "--embedding-model-name", help="Optional remote embedding model name."),
    ocr_model_dir: Path = typer.Option(
        Path("/raid/jdyer/slimgest/models/nemotron-ocr-v1/checkpoints"),
        "--ocr-model-dir",
        help="Local OCR checkpoints directory (ignored if --ocr-endpoint is set).",
    ),
):
    pdfs = sp_list_pdfs(Path(input_path))
    if not pdfs:
        raise typer.BadParameter(f"No PDFs found under {input_path}")

    output_dir = Path(output_dir)
    embeddings_dir = output_dir / "embeddings"
    metrics_dir = output_dir / "metrics"
    sp_ensure_dir(embeddings_dir)
    sp_ensure_dir(metrics_dir)

    dev = torch.device(device)

    # Load models once.
    page_elements = NemotronPageElementsV3(endpoint=page_elements_endpoint, remote_batch_size=32)
    table_structure = NemotronTableStructureV1(endpoint=table_structure_endpoint, remote_batch_size=32)
    graphic_elements = NemotronGraphicElementsV1(endpoint=graphic_elements_endpoint, remote_batch_size=32)
    ocr = NemotronOCRV1(
        model_dir=str(ocr_model_dir),
        endpoint=str(ocr_endpoint).strip() if ocr_endpoint else None,
        remote_batch_size=32,
    )
    embedder = LlamaNemotronEmbed1BV2Embedder(endpoint=embedding_endpoint, model_name=embedding_model_name, normalize=True)

    # Pre-scan for global page progress.
    pdf_page_counts = [(p, sp_count_pages(p)) for p in pdfs]
    total_pages = int(sum(n for _, n in pdf_page_counts))

    run_t0 = time.perf_counter()
    pages_done = 0
    all_pdf_metrics: List[Dict[str, Any]] = []

    p_pdfs = tqdm(total=len(pdfs), desc="PDFs", unit="pdf", position=0)
    p_pages = tqdm(total=total_pages, desc="Pages", unit="page", position=1)
    try:
        for pdf_idx, (pdf_path, n_pages) in enumerate(pdf_page_counts, start=1):
            pdf_t0 = time.perf_counter()
            pdf_metrics: Dict[str, Any] = {
                "schema_version": 1,
                "pdf_path": str(pdf_path),
                "pdf_index": int(pdf_idx),
                "pdfs_total": int(len(pdfs)),
                "pages_expected": int(n_pages),
                "run_config": {
                    "device": str(dev),
                    "dpi": float(dpi),
                    "parallel_stages": bool(parallel_stages),
                    "region_batch_size": int(region_batch_size),
                    "ocr_batch_size": int(ocr_batch_size),
                    "resize_to_1024": bool(resize_to_1024),
                    "embedding_batch_size": int(embedding_batch_size),
                    "endpoints": {
                        "page_elements": page_elements_endpoint,
                        "table_structure": table_structure_endpoint,
                        "graphic_elements": graphic_elements_endpoint,
                        "ocr": ocr_endpoint,
                        "embedding": embedding_endpoint,
                    },
                },
                "pages": [],
                "totals": {},
                "wall_s": 0.0,
            }

            pdf_doc = pdfium.PdfDocument(str(pdf_path))
            try:
                for page_info in iter_pdf_page_tensors(str(pdf_path), dpi=float(dpi), device=str(dev)):
                    page_idx = int(page_info.page_number)
                    page_tensor = page_info.tensor
                    bitmap_shape = (int(page_info.original_height), int(page_info.original_width))

                    timers = TimerBank()
                    page_metrics = sp_empty_page_metrics(page_idx)
                    page_t0 = time.perf_counter()

                    with timers.timed("stage0.pdfium_text"):
                        pdfium_text = sp_extract_pdfium_page_text(pdf_doc, page_idx)

                    # Stage2: page elements
                    stage2_dets = sp_stage2_page_elements(
                        model=page_elements,
                        page_tensor=page_tensor,
                        bitmap_shape_hw=bitmap_shape,
                        page_metrics=page_metrics,
                        timers=timers,
                    )

                    # Partition detections (mirrors stage scripts).
                    table_regions = [d for d in stage2_dets if int(d.get("label", -1)) == 0]
                    graphic_regions = [d for d in stage2_dets if int(d.get("label", -1)) in (1, 3)]  # chart + infographic
                    ocr_regions = [d for d in stage2_dets if int(d.get("label", -1)) in (0, 1, 3)]  # table + chart + infographic

                    if parallel_stages:
                        with cf.ThreadPoolExecutor(max_workers=3) as ex:
                            fut_ge = ex.submit(
                                sp_stage3_graphic_elements,
                                model=graphic_elements,
                                page_tensor=page_tensor,
                                graphic_regions=graphic_regions,
                                page_metrics=page_metrics,
                                timers=timers,
                                region_batch_size=int(region_batch_size),
                            )
                            fut_ts = ex.submit(
                                sp_stage4_table_structure,
                                model=table_structure,
                                page_tensor=page_tensor,
                                table_regions=table_regions,
                                page_metrics=page_metrics,
                                timers=timers,
                                region_batch_size=int(region_batch_size),
                            )
                            fut_ocr = ex.submit(
                                sp_stage5_ocr,
                                ocr=ocr,
                                page_tensor=page_tensor,
                                ocr_regions=ocr_regions,
                                page_metrics=page_metrics,
                                timers=timers,
                                ocr_batch_size=int(ocr_batch_size),
                                resize_to_1024=bool(resize_to_1024),
                            )
                            fut_ge.result()
                            fut_ts.result()
                            ocr_regions_out = fut_ocr.result()
                    else:
                        sp_stage3_graphic_elements(
                            model=graphic_elements,
                            page_tensor=page_tensor,
                            graphic_regions=graphic_regions,
                            page_metrics=page_metrics,
                            timers=timers,
                            region_batch_size=int(region_batch_size),
                        )
                        sp_stage4_table_structure(
                            model=table_structure,
                            page_tensor=page_tensor,
                            table_regions=table_regions,
                            page_metrics=page_metrics,
                            timers=timers,
                            region_batch_size=int(region_batch_size),
                        )
                        ocr_regions_out = sp_stage5_ocr(
                            ocr=ocr,
                            page_tensor=page_tensor,
                            ocr_regions=ocr_regions,
                            page_metrics=page_metrics,
                            timers=timers,
                            ocr_batch_size=int(ocr_batch_size),
                            resize_to_1024=bool(resize_to_1024),
                        )

                    # Stage6: embeddings (.pt output)
                    texts: List[str] = []
                    bboxes: List[List[float]] = []
                    kinds: List[str] = []
                    if pdfium_text:
                        texts.append(pdfium_text)
                        bboxes.append([0.0, 0.0, 1.0, 1.0])
                        kinds.append("pdfium_page_text")
                    for r in ocr_regions_out:
                        txt = (r.get("ocr_text") or "").strip()
                        if not txt:
                            continue
                        texts.append(txt)
                        bbox = r.get("bbox_xyxy_norm_in_page")
                        if isinstance(bbox, list) and len(bbox) == 4:
                            bboxes.append([float(x) for x in bbox])
                        else:
                            bboxes.append([0.0, 0.0, 0.0, 0.0])
                        kinds.append("ocr_region")

                    page_metrics["counts"]["stage6_embedding_texts"] = int(len(texts))
                    emb_path = sp_embed_out_path(embeddings_dir, pdf_path, page_idx)
                    sp_stage6_embed_and_save(
                        embedder=embedder,
                        texts=texts,
                        text_kinds=kinds,
                        bboxes_xyxy_norm_in_page=bboxes,
                        out_path=emb_path,
                        page_metrics=page_metrics,
                        timers=timers,
                        embedding_batch_size=int(embedding_batch_size),
                        pdf_path=pdf_path,
                        page_index=page_idx,
                    )

                    page_metrics["timing_s"] = timers.as_dict()
                    page_metrics["timing_s"]["page_total"] = float(time.perf_counter() - page_t0)
                    page_metrics["outputs"]["embeddings_pt"] = str(emb_path)
                    pdf_metrics["pages"].append(page_metrics)

                    pages_done += 1
                    p_pages.update(1)
                    p_pages.set_postfix(pages=f"{pages_done}/{total_pages}")

            finally:
                pdf_doc.close()

            pdf_metrics["wall_s"] = float(time.perf_counter() - pdf_t0)

            # Small per-PDF totals (fast to parse later).
            totals: Dict[str, Any] = {"pages": int(len(pdf_metrics["pages"]))}
            totals_models: Dict[str, Dict[str, float]] = {}
            for p in pdf_metrics["pages"]:
                for name, m in (p.get("models") or {}).items():
                    cur = totals_models.setdefault(name, {"seconds": 0.0, "calls": 0.0, "items": 0.0})
                    cur["seconds"] += float(m.get("seconds", 0.0) or 0.0)
                    cur["calls"] += float(m.get("calls", 0) or 0.0)
                    cur["items"] += float(m.get("items", 0) or 0.0)
            totals["models"] = totals_models
            pdf_metrics["totals"] = totals

            metrics_path = sp_pdf_metrics_out_path(metrics_dir, pdf_path)
            write_json(metrics_path, pdf_metrics)
            all_pdf_metrics.append(pdf_metrics)

            p_pdfs.update(1)
            elapsed = max(1e-9, time.perf_counter() - run_t0)
            p_pdfs.set_postfix(
                pdf=f"{pdf_idx}/{len(pdfs)}",
                pages=f"{pages_done}/{total_pages}",
                pdf_s=f"{pdf_idx/elapsed:.2f}",
                pages_s=f"{pages_done/elapsed:.2f}",
                last_pdf_wall=fmt_seconds_hms(pdf_metrics["wall_s"]),
            )
    finally:
        p_pdfs.close()
        p_pages.close()

    # End-of-run results.txt (hotspots + suggestions).
    results_txt = sp_compute_run_summary_and_suggestions(all_pdf_metrics)
    (output_dir / "results.txt").write_text(results_txt, encoding="utf-8")

    total_wall = float(time.perf_counter() - run_t0)
    tqdm.write(f"Done. pdfs={len(pdfs)} pages={pages_done} wall_s={total_wall:.2f} results={output_dir/'results.txt'}")