from __future__ import annotations

import concurrent.futures as cf
import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pypdfium2 as pdfium
import torch
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


app = typer.Typer(help="Multi-GPU version of `slimgest local simple run` (one worker process per GPU).")
console = Console()


def _chunked(seq: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    bs = max(1, int(batch_size))
    for i in range(0, len(seq), bs):
        yield list(seq[i : i + bs])


def _is_cuda_oom(exc: BaseException) -> bool:
    # torch.cuda.OutOfMemoryError exists in most modern torch, but RuntimeError strings still happen.
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        s = str(exc).lower()
        return (
            "cuda out of memory" in s
            or "cublas_status_alloc_failed" in s
            or "cuda error: out of memory" in s
            or "hip out of memory" in s
        )
    return False


def _cuda_oom_recover() -> None:
    # Best-effort: free cached blocks, then prompt GC.
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Synchronize to surface async errors early and reduce fragmentation surprise.
            torch.cuda.synchronize()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


def _run_batch_with_oom_retry(
    *,
    items: Sequence[Any],
    target_batch_size: int,
    min_batch_size: int,
    fn,  # Callable[[Sequence[Any]], List[Any]]
    what: str,
    page_metrics: Dict[str, Any],
) -> List[Any]:
    """
    Run `fn` over `items` in batches of `target_batch_size`.

    If a CUDA OOM occurs for a specific window, retry that window with smaller
    batches (halving until `min_batch_size`), then resume using the original
    `target_batch_size` for subsequent windows.
    """
    out: List[Any] = []
    n = len(items)
    i = 0
    tbs = max(1, int(target_batch_size))
    mbs = max(1, int(min_batch_size))
    while i < n:
        bs = min(tbs, n - i)
        window = items[i : i + bs]
        try:
            out.extend(list(fn(window)))
            i += bs
            continue
        except Exception as e:
            if not _is_cuda_oom(e) or bs <= mbs:
                raise
            page_metrics.setdefault("errors", []).append(
                f"cuda_oom:{what}:retry window={i}:{i+bs} bs={bs} -> smaller"
            )
            _cuda_oom_recover()

        # Retry the *same* window in smaller chunks, but do NOT change global batch size.
        sub_bs = max(mbs, bs // 2)
        j = 0
        while j < len(window):
            cur = min(sub_bs, len(window) - j)
            try:
                out.extend(list(fn(window[j : j + cur])))
                j += cur
            except Exception as e:
                if not _is_cuda_oom(e) or cur <= mbs:
                    raise
                page_metrics.setdefault("errors", []).append(
                    f"cuda_oom:{what}:retry window={i}:{i+bs} sub={j}:{j+cur} bs={cur} -> smaller"
                )
                _cuda_oom_recover()
                sub_bs = max(mbs, cur // 2)
        i += bs
    return out


def _extract_pdfium_page_text(pdf: pdfium.PdfDocument, page_idx: int) -> str:
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


def _embed_out_path(output_embeddings_dir: Path, pdf_path: Path, page_index: int) -> Path:
    return output_embeddings_dir / f"{pdf_path.stem}_page{int(page_index):04d}.embeddings.pt"


def _pdf_metrics_out_path(output_metrics_dir: Path, pdf_path: Path) -> Path:
    return output_metrics_dir / f"{pdf_path.stem}.metrics.json"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _list_pdfs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])


def _count_pages(pdf_path: Path) -> int:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return int(len(pdf))
    finally:
        pdf.close()


def _new_model_metrics() -> Dict[str, Any]:
    return {"calls": 0, "seconds": 0.0, "items": 0, "batch_size_hist": {}}


def _model_add(
    page_metrics: Dict[str, Any],
    *,
    model: str,
    seconds: float,
    items: int,
    batch_size: Optional[int] = None,
    calls: int = 1,
) -> None:
    models = page_metrics.setdefault("models", {})
    m = models.setdefault(model, _new_model_metrics())
    m["calls"] = int(m.get("calls", 0)) + int(calls)
    m["seconds"] = float(m.get("seconds", 0.0)) + float(seconds)
    m["items"] = int(m.get("items", 0)) + int(items)
    if batch_size is not None:
        hist = m.setdefault("batch_size_hist", {})
        k = str(int(batch_size))
        hist[k] = int(hist.get(k, 0)) + 1


def _empty_page_metrics(page_index: int) -> Dict[str, Any]:
    return {"page_index": int(page_index), "counts": {}, "timing_s": {}, "models": {}, "errors": [], "outputs": {}}


def _stage2_page_elements(
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
        _model_add(page_metrics, model="page_elements_v3", seconds=dt, items=1, calls=1)
        with timers.timed("stage2.page_elements.postprocess"):
            boxes, labels, scores = model.postprocess(preds)  # type: ignore[arg-type]

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


def _stage3_graphic_elements(
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
        for batch in _chunked(list(graphic_regions), region_batch_size):
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

            def _invoke_many(window: Sequence[int]) -> List[Any]:
                # window are indices into inputs/shapes; return preds for those.
                preds: List[Any] = []
                t0 = time.perf_counter()
                for idx in window:
                    preds.append(model.invoke(inputs[idx], shapes[idx]))
                dt = time.perf_counter() - t0
                _model_add(
                    page_metrics,
                    model="graphic_elements_v1",
                    seconds=dt,
                    items=len(window),
                    batch_size=len(window),
                    calls=1,
                )
                return preds

            idxs = list(range(len(inputs)))
            preds_list = _run_batch_with_oom_retry(
                items=idxs,
                target_batch_size=len(idxs),
                min_batch_size=1,
                fn=_invoke_many,
                what="stage3.graphic.invoke",
                page_metrics=page_metrics,
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


def _stage4_table_structure(
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
        for batch in _chunked(list(table_regions), region_batch_size):
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

            def _invoke_many(window: Sequence[int]) -> List[Any]:
                preds: List[Any] = []
                t0 = time.perf_counter()
                for idx in window:
                    preds.append(model.invoke(inputs[idx], shapes[idx]))
                dt = time.perf_counter() - t0
                _model_add(
                    page_metrics,
                    model="table_structure_v1",
                    seconds=dt,
                    items=len(window),
                    batch_size=len(window),
                    calls=1,
                )
                return preds

            idxs = list(range(len(inputs)))
            preds_list = _run_batch_with_oom_retry(
                items=idxs,
                target_batch_size=len(idxs),
                min_batch_size=1,
                fn=_invoke_many,
                what="stage4.table.invoke",
                page_metrics=page_metrics,
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


def _stage5_ocr_streaming(
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

    regions_out: List[Dict[str, Any]] = []
    nonempty = 0

    with torch.inference_mode():
        # Process in micro-batches so we never hold all crops in GPU memory at once.
        for region_batch in _chunked(list(ocr_regions), ocr_batch_size):
            batch_crops: List[torch.Tensor] = []
            batch_meta: List[Dict[str, Any]] = []
            for r in region_batch:
                bbox = r["bbox_xyxy_norm"]
                with timers.timed("stage5.ocr.crop"):
                    crop = crop_tensor_normalized_xyxy(page_tensor, bbox)
                if resize_to_1024:
                    with timers.timed("stage5.ocr.resize_pad"):
                        crop = resize_pad_tensor(crop, target_hw=(1024, 1024))
                batch_crops.append(crop)
                batch_meta.append({"bbox_xyxy_norm_in_page": [float(x) for x in bbox], "source": r})

            if not batch_crops:
                continue

            def _invoke_batch(window: Sequence[int]) -> List[Any]:
                t0 = time.perf_counter()
                if getattr(ocr, "_endpoint", None) is not None:
                    b = torch.stack([batch_crops[i] for i in window], dim=0)
                    out = list(ocr.invoke(b))
                else:
                    out = []
                    for i in window:
                        out.append(ocr.invoke(batch_crops[i]))
                dt = time.perf_counter() - t0
                _model_add(
                    page_metrics,
                    model="nemotron_ocr_v1",
                    seconds=dt,
                    items=len(window),
                    batch_size=len(window),
                    calls=1,
                )
                return out

            idxs = list(range(len(batch_crops)))
            outs = _run_batch_with_oom_retry(
                items=idxs,
                target_batch_size=len(idxs),
                min_batch_size=1,
                fn=_invoke_batch,
                what="stage5.ocr.invoke",
                page_metrics=page_metrics,
            )

            n = min(len(outs), len(batch_meta))
            with timers.timed("stage5.ocr.postprocess"):
                for meta, raw in zip(batch_meta[:n], outs[:n]):
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


def _stage6_embed_and_save_oom_safe(
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
    _ensure_dir(out_path.parent)
    vectors: List[torch.Tensor] = []
    if not texts:
        with timers.timed("stage6.embed.save"):
            torch.save(
                {
                    "schema_version": 1,
                    "stage": 6,
                    "model": "llama_nemotron_embed_1b_v2",
                    "pdf_path": str(pdf_path),
                    "page_index": int(page_index),
                    "texts": [],
                    "text_kinds": [],
                    "bboxes_xyxy_norm_in_page": [],
                    "embeddings": [],
                },
                out_path,
            )
        return

    idxs = list(range(len(texts)))

    def _embed_window(window: Sequence[int]) -> List[torch.Tensor]:
        # Embed this chunk and return CPU vectors (already normalized by wrapper).
        chunk_texts = [texts[i] for i in window]
        t0 = time.perf_counter()
        emb = embedder.embed(chunk_texts, batch_size=max(1, int(len(chunk_texts))))
        dt = time.perf_counter() - t0
        _model_add(
            page_metrics,
            model="llama_nemotron_embed_1b_v2",
            seconds=dt,
            items=len(chunk_texts),
            batch_size=len(chunk_texts),
            calls=1,
        )
        outs: List[torch.Tensor] = []
        if isinstance(emb, torch.Tensor) and emb.ndim == 2 and int(emb.shape[0]) == int(len(chunk_texts)):
            for i in range(int(emb.shape[0])):
                outs.append(emb[i].float().cpu())
        return outs

    # Drive our own batching so we can OOM-split, then resume with the original batch size.
    all_vecs = _run_batch_with_oom_retry(
        items=idxs,
        target_batch_size=int(embedding_batch_size),
        min_batch_size=1,
        fn=_embed_window,
        what="stage6.embed",
        page_metrics=page_metrics,
    )
    vectors.extend(all_vecs)

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


@dataclass(frozen=True)
class _WorkerConfig:
    worker_idx: int
    gpu_id: int
    pdfs: List[Path]
    output_dir: Path
    dpi: float
    parallel_stages: bool
    region_batch_size: int
    ocr_batch_size: int
    resize_to_1024: bool
    embedding_batch_size: int
    page_elements_endpoint: Optional[str]
    table_structure_endpoint: Optional[str]
    graphic_elements_endpoint: Optional[str]
    ocr_endpoint: Optional[str]
    embedding_endpoint: Optional[str]
    embedding_model_name: Optional[str]
    ocr_model_dir: Path
    skip_existing: bool


def _worker_main(cfg: _WorkerConfig) -> None:
    # IMPORTANT: pin this process to a single GPU by masking visibility.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(cfg.gpu_id))
    # Reduce allocator fragmentation sensitivity a bit (safe even if unsupported).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass

    embeddings_dir = cfg.output_dir / "embeddings"
    metrics_dir = cfg.output_dir / "metrics"
    _ensure_dir(embeddings_dir)
    _ensure_dir(metrics_dir)

    # Load models once per worker (on that worker's single visible GPU).
    page_elements = NemotronPageElementsV3(endpoint=cfg.page_elements_endpoint, remote_batch_size=32)
    table_structure = NemotronTableStructureV1(endpoint=cfg.table_structure_endpoint, remote_batch_size=32)
    graphic_elements = NemotronGraphicElementsV1(endpoint=cfg.graphic_elements_endpoint, remote_batch_size=32)
    ocr = NemotronOCRV1(
        model_dir=str(cfg.ocr_model_dir),
        endpoint=str(cfg.ocr_endpoint).strip() if cfg.ocr_endpoint else None,
        remote_batch_size=32,
    )
    embedder = LlamaNemotronEmbed1BV2Embedder(
        endpoint=cfg.embedding_endpoint, model_name=cfg.embedding_model_name, normalize=True
    )

    run_t0 = time.perf_counter()
    p_pdfs = tqdm(total=len(cfg.pdfs), desc=f"GPU{cfg.gpu_id} PDFs", unit="pdf", position=cfg.worker_idx * 2)
    try:
        for pdf_idx, pdf_path in enumerate(cfg.pdfs, start=1):
            pdf_t0 = time.perf_counter()
            n_pages = _count_pages(pdf_path)

            pdf_metrics: Dict[str, Any] = {
                "schema_version": 1,
                "pdf_path": str(pdf_path),
                "pdf_index_worker": int(pdf_idx),
                "pdfs_total_worker": int(len(cfg.pdfs)),
                "pages_expected": int(n_pages),
                "worker": {"gpu_id": int(cfg.gpu_id), "worker_idx": int(cfg.worker_idx)},
                "run_config": {
                    "device": "cuda",
                    "dpi": float(cfg.dpi),
                    "parallel_stages": bool(cfg.parallel_stages),
                    "region_batch_size": int(cfg.region_batch_size),
                    "ocr_batch_size": int(cfg.ocr_batch_size),
                    "resize_to_1024": bool(cfg.resize_to_1024),
                    "embedding_batch_size": int(cfg.embedding_batch_size),
                    "endpoints": {
                        "page_elements": cfg.page_elements_endpoint,
                        "table_structure": cfg.table_structure_endpoint,
                        "graphic_elements": cfg.graphic_elements_endpoint,
                        "ocr": cfg.ocr_endpoint,
                        "embedding": cfg.embedding_endpoint,
                    },
                },
                "pages": [],
                "totals": {},
                "wall_s": 0.0,
            }

            pdf_doc = pdfium.PdfDocument(str(pdf_path))
            try:
                for page_info in iter_pdf_page_tensors(str(pdf_path), dpi=float(cfg.dpi), device="cuda"):
                    page_idx = int(page_info.page_number)
                    emb_path = _embed_out_path(embeddings_dir, pdf_path, page_idx)
                    if cfg.skip_existing and emb_path.exists():
                        continue

                    page_tensor = page_info.tensor
                    bitmap_shape = (int(page_info.original_height), int(page_info.original_width))

                    timers = TimerBank()
                    page_metrics = _empty_page_metrics(page_idx)
                    page_t0 = time.perf_counter()

                    try:
                        with timers.timed("stage0.pdfium_text"):
                            pdfium_text = _extract_pdfium_page_text(pdf_doc, page_idx)

                        stage2_dets = _stage2_page_elements(
                            model=page_elements,
                            page_tensor=page_tensor,
                            bitmap_shape_hw=bitmap_shape,
                            page_metrics=page_metrics,
                            timers=timers,
                        )

                        table_regions = [d for d in stage2_dets if int(d.get("label", -1)) == 0]
                        graphic_regions = [d for d in stage2_dets if int(d.get("label", -1)) in (1, 3)]
                        ocr_regions = [d for d in stage2_dets if int(d.get("label", -1)) in (0, 1, 3)]

                        if cfg.parallel_stages:
                            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                                fut_ge = ex.submit(
                                    _stage3_graphic_elements,
                                    model=graphic_elements,
                                    page_tensor=page_tensor,
                                    graphic_regions=graphic_regions,
                                    page_metrics=page_metrics,
                                    timers=timers,
                                    region_batch_size=int(cfg.region_batch_size),
                                )
                                fut_ts = ex.submit(
                                    _stage4_table_structure,
                                    model=table_structure,
                                    page_tensor=page_tensor,
                                    table_regions=table_regions,
                                    page_metrics=page_metrics,
                                    timers=timers,
                                    region_batch_size=int(cfg.region_batch_size),
                                )
                                fut_ocr = ex.submit(
                                    _stage5_ocr_streaming,
                                    ocr=ocr,
                                    page_tensor=page_tensor,
                                    ocr_regions=ocr_regions,
                                    page_metrics=page_metrics,
                                    timers=timers,
                                    ocr_batch_size=int(cfg.ocr_batch_size),
                                    resize_to_1024=bool(cfg.resize_to_1024),
                                )
                                fut_ge.result()
                                fut_ts.result()
                                ocr_regions_out = fut_ocr.result()
                        else:
                            _stage3_graphic_elements(
                                model=graphic_elements,
                                page_tensor=page_tensor,
                                graphic_regions=graphic_regions,
                                page_metrics=page_metrics,
                                timers=timers,
                                region_batch_size=int(cfg.region_batch_size),
                            )
                            _stage4_table_structure(
                                model=table_structure,
                                page_tensor=page_tensor,
                                table_regions=table_regions,
                                page_metrics=page_metrics,
                                timers=timers,
                                region_batch_size=int(cfg.region_batch_size),
                            )
                            ocr_regions_out = _stage5_ocr_streaming(
                                ocr=ocr,
                                page_tensor=page_tensor,
                                ocr_regions=ocr_regions,
                                page_metrics=page_metrics,
                                timers=timers,
                                ocr_batch_size=int(cfg.ocr_batch_size),
                                resize_to_1024=bool(cfg.resize_to_1024),
                            )

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
                        _stage6_embed_and_save_oom_safe(
                            embedder=embedder,
                            texts=texts,
                            text_kinds=kinds,
                            bboxes_xyxy_norm_in_page=bboxes,
                            out_path=emb_path,
                            page_metrics=page_metrics,
                            timers=timers,
                            embedding_batch_size=int(cfg.embedding_batch_size),
                            pdf_path=pdf_path,
                            page_index=page_idx,
                        )

                        page_metrics["timing_s"] = timers.as_dict()
                        page_metrics["timing_s"]["page_total"] = float(time.perf_counter() - page_t0)
                        page_metrics["outputs"]["embeddings_pt"] = str(emb_path)
                        pdf_metrics["pages"].append(page_metrics)

                    except Exception as e:
                        # Keep the worker resilient. On CUDA OOM we at least try to clean up,
                        # then continue to the next page.
                        if _is_cuda_oom(e):
                            page_metrics.setdefault("errors", []).append(f"cuda_oom:page_failed:{type(e).__name__}:{e}")
                            _cuda_oom_recover()
                            # Record that this page failed but keep moving.
                            page_metrics["timing_s"] = timers.as_dict()
                            page_metrics["timing_s"]["page_total"] = float(time.perf_counter() - page_t0)
                            pdf_metrics["pages"].append(page_metrics)
                            continue
                        raise
            finally:
                pdf_doc.close()

            pdf_metrics["wall_s"] = float(time.perf_counter() - pdf_t0)

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

            metrics_path = _pdf_metrics_out_path(metrics_dir, pdf_path)
            write_json(metrics_path, pdf_metrics)

            p_pdfs.update(1)
            elapsed = max(1e-9, time.perf_counter() - run_t0)
            p_pdfs.set_postfix(
                pdf=f"{pdf_idx}/{len(cfg.pdfs)}",
                pdf_s=f"{pdf_idx/elapsed:.2f}",
                last_pdf_wall=fmt_seconds_hms(pdf_metrics["wall_s"]),
            )
    finally:
        p_pdfs.close()


def _parse_gpu_list(gpus: str) -> List[int]:
    s = (gpus or "").strip().lower()
    n = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if s in {"", "all", "*"}:
        return list(range(n))
    out: List[int] = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        out.append(int(part))
    # Keep only valid ids
    out = [i for i in out if 0 <= i < n]
    # Dedup while preserving order
    seen: set[int] = set()
    uniq: List[int] = []
    for i in out:
        if i in seen:
            continue
        seen.add(i)
        uniq.append(i)
    return uniq


@app.command()
def run(
    input_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=True),
    output_dir: Path = typer.Option(Path("./outputs/multi_gpu"), "--output-dir", help="Directory to write embeddings + metrics."),
    dpi: float = typer.Option(150.0, "--dpi", help="PDF render DPI."),
    gpus: str = typer.Option("all", "--gpus", help="Comma-separated GPU ids to use (e.g. '0,1,2') or 'all'."),
    parallel_stages: bool = typer.Option(
        True,
        "--parallel-stages/--no-parallel-stages",
        help="Run graphic_elements, table_structure, and OCR concurrently per page (per worker).",
    ),
    region_batch_size: int = typer.Option(16, "--region-batch-size", min=1, help="Microbatch size for region crops."),
    ocr_batch_size: int = typer.Option(16, "--ocr-batch-size", min=1, help="OCR crops per batch."),
    resize_to_1024: bool = typer.Option(True, "--resize-to-1024/--no-resize-to-1024", help="Resize+pad OCR crops to 1024x1024."),
    embedding_batch_size: int = typer.Option(64, "--embedding-batch-size", min=1, help="Texts per embedding batch."),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip pages whose embeddings output already exists (resume-friendly).",
    ),
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
) -> None:
    if not torch.cuda.is_available():
        raise typer.BadParameter("CUDA is not available; `multi-gpu` requires at least one GPU.")

    pdfs = _list_pdfs(Path(input_path))
    if not pdfs:
        raise typer.BadParameter(f"No PDFs found under {input_path}")

    gpu_ids = _parse_gpu_list(gpus)
    if not gpu_ids:
        raise typer.BadParameter(f"No usable GPUs found/selected from --gpus={gpus!r}")

    output_dir = Path(output_dir)
    _ensure_dir(output_dir)

    # Assign PDFs to GPUs (round-robin) so workers can stream render pages efficiently.
    pdfs_by_worker: List[List[Path]] = [[] for _ in gpu_ids]
    for i, p in enumerate(pdfs):
        pdfs_by_worker[i % len(gpu_ids)].append(p)

    console.print(
        f"[bold cyan]multi-gpu[/bold cyan] pdfs={len(pdfs)} gpus={gpu_ids} "
        f"parallel_stages={parallel_stages} region_bs={region_batch_size} ocr_bs={ocr_batch_size} emb_bs={embedding_batch_size} "
        f"skip_existing={skip_existing} output_dir={output_dir}"
    )

    ctx = torch.multiprocessing.get_context("spawn")
    procs: List[Any] = []
    start_t0 = time.perf_counter()
    for worker_idx, (gpu_id, worker_pdfs) in enumerate(zip(gpu_ids, pdfs_by_worker)):
        cfg = _WorkerConfig(
            worker_idx=int(worker_idx),
            gpu_id=int(gpu_id),
            pdfs=list(worker_pdfs),
            output_dir=output_dir,
            dpi=float(dpi),
            parallel_stages=bool(parallel_stages),
            region_batch_size=int(region_batch_size),
            ocr_batch_size=int(ocr_batch_size),
            resize_to_1024=bool(resize_to_1024),
            embedding_batch_size=int(embedding_batch_size),
            page_elements_endpoint=page_elements_endpoint,
            table_structure_endpoint=table_structure_endpoint,
            graphic_elements_endpoint=graphic_elements_endpoint,
            ocr_endpoint=ocr_endpoint,
            embedding_endpoint=embedding_endpoint,
            embedding_model_name=embedding_model_name,
            ocr_model_dir=Path(ocr_model_dir),
            skip_existing=bool(skip_existing),
        )
        p = ctx.Process(target=_worker_main, args=(cfg,), daemon=False)
        p.start()
        procs.append(p)

    exit_codes: List[int] = []
    for p in procs:
        p.join()
        exit_codes.append(int(p.exitcode or 0))

    wall_s = float(time.perf_counter() - start_t0)
    bad: List[Tuple[int, int]] = []
    for (gpu_id, _), code in zip(zip(gpu_ids, pdfs_by_worker), exit_codes):
        if int(code) != 0:
            bad.append((int(gpu_id), int(code)))
    if bad:
        console.print(f"[red]Workers failed[/red] {bad} (gpu_id, exit_code)")
        raise typer.Exit(code=1)
    console.print(f"[green]Done[/green] gpus={gpu_ids} wall_s={wall_s:.2f} output_dir={output_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

