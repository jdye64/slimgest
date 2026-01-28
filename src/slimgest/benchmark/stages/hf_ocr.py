from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import typer
from rich.console import Console
from rich.traceback import install
from tqdm import tqdm

from slimgest.model.local.nemotron_ocr_v1 import NemotronOCRV1

from .._io import iter_images, load_image_rgb_chw_u8
from .._sysinfo import collect_system_info, format_system_info_human


install(show_locals=False)
console = Console()
app = typer.Typer(help="Benchmark: run Nemotron OCR (HF/local pipeline) over an image directory.")


def _fmt_secs(s: float) -> str:
    if s < 0 or not (s == s):  # NaN-safe
        return "unknown"
    ms = s * 1000.0
    if ms < 1000.0:
        return f"{ms:.2f} ms"
    return f"{s:.3f} s"


def _percentile(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs2 = sorted(xs)
    k = (len(xs2) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs2) - 1)
    if f == c:
        return xs2[f]
    return xs2[f] + (xs2[c] - xs2[f]) * (k - f)


def _resize_pad_tensor(img: torch.Tensor, target_hw: Tuple[int, int] = (1024, 1024), pad_value: float = 114.0) -> torch.Tensor:
    """
    Resize+pad CHW image tensor to fixed size (preserve aspect ratio).
    Returns uint8 CHW tensor.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(img.shape)}")
    H, W = int(img.shape[1]), int(img.shape[2])
    th, tw = int(target_hw[0]), int(target_hw[1])
    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid image shape: {tuple(img.shape)}")

    x = img.float()
    scale = min(th / H, tw / W)
    nh = max(1, int(H * scale))
    nw = max(1, int(W * scale))
    x = F.interpolate(x.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False).squeeze(0)
    x = torch.clamp(x, 0, 255)
    pad_b = th - nh
    pad_r = tw - nw
    x = F.pad(x, (0, pad_r, 0, pad_b), value=float(pad_value))
    return x.to(dtype=torch.uint8)


def _chunked(seq: List[Any], batch_size: int) -> List[List[Any]]:
    bs = max(1, int(batch_size))
    return [seq[i : i + bs] for i in range(0, len(seq), bs)]


@dataclass
class _RunStats:
    images_total: int = 0
    images_ok: int = 0
    images_failed: int = 0
    batches: int = 0
    warmup_batches: int = 0
    io_seconds: float = 0.0
    infer_seconds: float = 0.0
    total_seconds: float = 0.0
    infer_batch_seconds: List[float] = None  # type: ignore[assignment]
    total_batch_seconds: List[float] = None  # type: ignore[assignment]
    ocr_items_returned: int = 0

    def __post_init__(self) -> None:
        if self.infer_batch_seconds is None:
            self.infer_batch_seconds = []
        if self.total_batch_seconds is None:
            self.total_batch_seconds = []


def _pick_device(device: str, gpu: Optional[int]) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        if gpu is not None:
            torch.cuda.set_device(int(gpu))
            return torch.device(f"cuda:{int(gpu)}")
        return torch.device("cuda")
    return torch.device("cpu")


@app.command()
def run(
    input_dir: Path = typer.Option(..., "--input-dir", exists=True, file_okay=False, help="Directory of images to OCR."),
    ocr_model_dir: Optional[Path] = typer.Option(
        None,
        "--ocr-model-dir",
        help="Local nemotron-ocr-v1 checkpoints directory. If omitted, uses $SLIMGEST_NEMOTRON_OCR_MODEL_DIR.",
    ),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Torch device (e.g. cuda, cuda:0, cpu)."),
    gpu: Optional[int] = typer.Option(None, "--gpu", help="GPU index to use (best-effort)."),
    batch_size: int = typer.Option(8, "--batch-size", min=1, help="Images per invocation (single-threaded, sequential)."),
    warmup_batches: int = typer.Option(1, "--warmup-batches", min=0, help="Warmup batches (excluded from metrics)."),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Optionally cap number of images processed."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Recursively scan `input-dir` for images."),
    resize_to_1024: bool = typer.Option(True, "--resize-to-1024/--no-resize-to-1024", help="Resize+pad to 1024x1024 before OCR."),
    torch_threads: int = typer.Option(1, "--torch-threads", min=1, help="torch.set_num_threads() for consistent CPU-side work."),
    json_out: bool = typer.Option(False, "--json", help="Also print a JSON blob of system info + metrics."),
) -> None:
    """
    Single-process, single-threaded benchmark for Nemotron OCR.

    Notes:
    - This measures end-to-end time including image decode + H2D copy (unless device=cpu).
    - `batch-size` controls how many images are stacked and passed per `invoke()` call.
      (The underlying local pipeline may still process per-image internally.)
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(int(torch_threads))
    try:
        torch.set_num_interop_threads(int(torch_threads))
    except Exception:
        pass

    model_dir = ocr_model_dir or (
        Path(os.getenv("SLIMGEST_NEMOTRON_OCR_MODEL_DIR", "")).expanduser()
        if os.getenv("SLIMGEST_NEMOTRON_OCR_MODEL_DIR")
        else None
    )
    if model_dir is None:
        raise typer.BadParameter("Must provide --ocr-model-dir or set $SLIMGEST_NEMOTRON_OCR_MODEL_DIR")
    if not Path(model_dir).exists():
        raise typer.BadParameter(f"OCR model dir does not exist: {model_dir}")

    dev = _pick_device(device, gpu)
    selected_gpu = None
    if dev.type == "cuda" and torch.cuda.is_available():
        selected_gpu = int(torch.cuda.current_device())

    # System info (for cross-machine comparison)
    si = collect_system_info()
    console.print("[bold yellow]System[/bold yellow]", highlight=False)
    console.print(format_system_info_human(si, selected_gpu=selected_gpu), highlight=False)

    images = iter_images(input_dir, recursive=bool(recursive))
    if limit is not None:
        images = images[: int(limit)]
    if not images:
        console.print("[red]No images found[/red]", highlight=False)
        raise typer.Exit(code=2)

    console.print(
        f"[bold cyan]hf-ocr benchmark[/bold cyan] images={len(images)} batch_size={int(batch_size)} warmup_batches={int(warmup_batches)} device={dev} model_dir={model_dir}",
        highlight=False,
    )

    # Init OCR once (single process)
    ocr = NemotronOCRV1(model_dir=str(model_dir), endpoint=None, remote_batch_size=int(batch_size))

    if dev.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(dev)
        except Exception:
            pass

    stats = _RunStats(images_total=len(images))

    # Warmup
    warm_imgs = images[: min(len(images), int(batch_size))]
    if warm_imgs and int(warmup_batches) > 0:
        console.print(f"[cyan]Warmup[/cyan] batches={int(warmup_batches)}", highlight=False)
        for _ in range(int(warmup_batches)):
            batch_tensors: List[torch.Tensor] = []
            for p in warm_imgs:
                t, _ = load_image_rgb_chw_u8(p, dev)
                if resize_to_1024:
                    t = _resize_pad_tensor(t)
                batch_tensors.append(t)
            bt = torch.stack(batch_tensors, dim=0)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            with torch.inference_mode():
                _ = ocr.invoke(bt)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
        stats.warmup_batches = int(warmup_batches)

    # Main benchmark
    t_run0 = time.perf_counter()
    for batch_paths in tqdm(_chunked(images, int(batch_size)), desc="hf-ocr", unit="batch"):
        t_batch0 = time.perf_counter()
        io_t0 = time.perf_counter()
        batch_tensors2: List[torch.Tensor] = []
        ok_in_batch = 0
        for p in batch_paths:
            try:
                t, _ = load_image_rgb_chw_u8(p, dev)
                if resize_to_1024:
                    t = _resize_pad_tensor(t)
                batch_tensors2.append(t)
                ok_in_batch += 1
            except Exception:
                stats.images_failed += 1

        io_dt = time.perf_counter() - io_t0
        stats.io_seconds += float(io_dt)

        if not batch_tensors2:
            continue

        bt2 = torch.stack(batch_tensors2, dim=0)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        infer_t0 = time.perf_counter()
        with torch.inference_mode():
            out = ocr.invoke(bt2)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        infer_dt = time.perf_counter() - infer_t0
        stats.infer_seconds += float(infer_dt)

        stats.ocr_items_returned += int(len(out) if hasattr(out, "__len__") else 0)

        batch_dt = time.perf_counter() - t_batch0
        stats.total_batch_seconds.append(float(batch_dt))
        stats.infer_batch_seconds.append(float(infer_dt))

        stats.batches += 1
        stats.images_ok += int(ok_in_batch)

    stats.total_seconds = float(time.perf_counter() - t_run0)

    # Summary
    imgs = int(stats.images_ok)
    infer_s = float(stats.infer_seconds)
    total_s = float(stats.total_seconds)
    io_s = float(stats.io_seconds)
    throughput = (imgs / total_s) if total_s > 0 else 0.0

    infer_p50 = statistics.median(stats.infer_batch_seconds) if stats.infer_batch_seconds else 0.0
    infer_p90 = _percentile(stats.infer_batch_seconds, 90.0) or 0.0
    infer_p99 = _percentile(stats.infer_batch_seconds, 99.0) or 0.0

    console.print("\n[bold yellow]Results[/bold yellow]", highlight=False)
    console.print(
        f"images_ok={stats.images_ok} images_failed={stats.images_failed} batches={stats.batches} batch_size={int(batch_size)}",
        highlight=False,
    )
    console.print(
        f"time_total={_fmt_secs(total_s)} time_io={_fmt_secs(io_s)} time_infer={_fmt_secs(infer_s)}",
        highlight=False,
    )
    console.print(f"throughput={throughput:.3f} images/s", highlight=False)
    console.print(
        f"infer_batch p50={_fmt_secs(infer_p50)} p90={_fmt_secs(infer_p90)} p99={_fmt_secs(infer_p99)}",
        highlight=False,
    )
    console.print(f"ocr_items_returned={stats.ocr_items_returned}", highlight=False)

    if dev.type == "cuda" and torch.cuda.is_available():
        try:
            max_alloc = int(torch.cuda.max_memory_allocated(dev))
            max_reserved = int(torch.cuda.max_memory_reserved(dev))
            console.print(
                f"gpu_peak_mem_allocated={max_alloc / (1024**2):.1f} MiB gpu_peak_mem_reserved={max_reserved / (1024**2):.1f} MiB",
                highlight=False,
            )
        except Exception:
            pass

    if json_out:
        payload: Dict[str, Any] = {
            "system": si.to_dict(),
            "selected_device": str(dev),
            "selected_gpu": selected_gpu,
            "config": {
                "input_dir": str(input_dir),
                "batch_size": int(batch_size),
                "warmup_batches": int(warmup_batches),
                "limit": int(limit) if limit is not None else None,
                "recursive": bool(recursive),
                "resize_to_1024": bool(resize_to_1024),
                "torch_threads": int(torch_threads),
                "ocr_model_dir": str(model_dir),
            },
            "metrics": {
                "images_total": int(stats.images_total),
                "images_ok": int(stats.images_ok),
                "images_failed": int(stats.images_failed),
                "batches": int(stats.batches),
                "io_seconds": float(stats.io_seconds),
                "infer_seconds": float(stats.infer_seconds),
                "total_seconds": float(stats.total_seconds),
                "throughput_images_per_s": float(throughput),
                "infer_batch_seconds": list(stats.infer_batch_seconds),
                "total_batch_seconds": list(stats.total_batch_seconds),
                "ocr_items_returned": int(stats.ocr_items_returned),
            },
        }
        console.print("\n[bold yellow]JSON[/bold yellow]", highlight=False)
        console.print(json.dumps(payload, indent=2, ensure_ascii=False), highlight=False)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

