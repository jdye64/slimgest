from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.traceback import install
import torch
import typer
from tqdm import tqdm

from slimgest.model.local.nemotron_page_elements_v3 import NemotronPageElementsV3

from ._io import iter_images, load_image_rgb_chw_u8, to_bbox_list, to_scalar_float, to_scalar_int, write_json


install(show_locals=False)
console = Console()
app = typer.Typer(help="Stage 2: run page_elements_v3 over page images and save JSON alongside each image.")

# Configure your checkpoint directory here (can still override via CLI).
DEFAULT_INPUT_DIR = Path("./data/pages")


def _out_path_for_image(img_path: Path) -> Path:
    return img_path.with_name(img_path.name + ".page_elements_v3.json")


def _chunked(seq: Sequence[Path], batch_size: int) -> Iterable[List[Path]]:
    bs = max(1, int(batch_size))
    for i in range(0, len(seq), bs):
        yield list(seq[i : i + bs])


def _invoke_batched_page_elements(
    model: NemotronPageElementsV3,
    batch_tensor: torch.Tensor,
    orig_shapes: Sequence[Tuple[int, int]],
) -> List[Any]:
    """
    Best-effort batched inference.

    Underlying Nemotron implementations vary; some accept a list of shapes, some accept a single shape,
    and some do not support batching at all. This function normalizes successful outputs into a
    list of per-image prediction objects (length == batch size), or raises to allow fallback.
    """
    # If using remote endpoint, bypass local nn.Module batching and use remote batch API.
    if getattr(model, "_endpoint", None) is not None:
        out = model.invoke_remote(batch_tensor)
        if isinstance(out, list) and len(out) == int(batch_tensor.shape[0]):
            return out
        raise RuntimeError("Remote output was not per-image list; falling back to per-image inference.")

    m = model.model  # underlying callable
    # Try (BCHW, list[shape]) first, then (BCHW, shape)
    try:
        raw = m(batch_tensor, list(orig_shapes))
    except Exception:
        raw = m(batch_tensor, orig_shapes[0])

    # Many implementations return (preds, aux...), where preds is a per-image list.
    raw0 = raw[0] if isinstance(raw, (tuple, list)) and len(raw) > 0 else raw
    if isinstance(raw0, list) and len(raw0) == int(batch_tensor.shape[0]):
        return raw0
    raise RuntimeError("Batched model output was not per-image list; falling back to per-image inference.")


@app.command()
def run(
    input_dir: Path = typer.Option(DEFAULT_INPUT_DIR, "--input-dir", exists=True, file_okay=False),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device for model + tensors."),
    batch_size: int = typer.Option(32, "--batch-size", min=1, help="Best-effort inference batch size."),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        help="Optional page elements NIM endpoint URL. If set, runs remotely and local weights are not loaded.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing JSON outputs."),
    limit: Optional[int] = typer.Option(None, help="Optionally limit number of images processed."),
):
    """
    Reads images from input_dir (stage 1 outputs) and writes <image>.+page_elements_v3.json.
    """
    dev = torch.device(device)
    model = NemotronPageElementsV3(endpoint=endpoint, remote_batch_size=batch_size)

    images = iter_images(input_dir)
    if limit is not None:
        images = images[: int(limit)]

    console.print(f"[bold cyan]Stage2[/bold cyan] images={len(images)} input_dir={input_dir} device={dev}")

    processed = 0
    skipped = 0
    to_process: List[Path] = []
    for img_path in images:
        out_path = _out_path_for_image(img_path)
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        to_process.append(img_path)

    for batch_paths in tqdm(list(_chunked(to_process, batch_size)), desc="Stage2 images", unit="batch"):
        # Load + preprocess
        tensors: List[torch.Tensor] = []
        shapes: List[Tuple[int, int]] = []
        for p in batch_paths:
            t, (h, w) = load_image_rgb_chw_u8(p, dev)
            shapes.append((h, w))
            tensors.append(model.preprocess(t))

        t0 = time.perf_counter()
        with torch.inference_mode():
            # Attempt true batching; if unavailable, fall back to per-image invoke
            per_image_preds: Optional[List[Any]] = None
            try:
                batch_tensor = torch.stack(tensors, dim=0)
                per_image_preds = _invoke_batched_page_elements(model, batch_tensor, shapes)
            except Exception:
                per_image_preds = None

            for i, img_path in enumerate(batch_paths):
                orig_shape = shapes[i]
                if per_image_preds is None:
                    preds = model.invoke(tensors[i], orig_shape)
                else:
                    preds = per_image_preds[i]

                boxes, labels, scores = model.postprocess(preds)
                dets: List[Dict[str, Any]] = []
                for box, lab, score in zip(boxes, labels, scores):
                    dets.append(
                        {
                            "bbox_xyxy_norm": to_bbox_list(box),
                            "label": to_scalar_int(lab),
                            "score": to_scalar_float(score),
                        }
                    )

                payload: Dict[str, Any] = {
                    "schema_version": 1,
                    "stage": 2,
                    "model": "page_elements_v3",
                    "image": {
                        "path": str(img_path),
                        "height": int(orig_shape[0]),
                        "width": int(orig_shape[1]),
                    },
                    "detections": dets,
                    "timing": {"seconds": 0.0},  # filled below
                }
                out_path = _out_path_for_image(img_path)
                write_json(out_path, payload)
                processed += 1
        dt = time.perf_counter() - t0

        # Optional: update each file's timing without re-reading (best-effort).
        # We keep this coarse and avoid extra IO by not rewriting the JSON a second time.

    console.print(
        f"[green]Done[/green] processed={processed} skipped={skipped} wrote_json_suffix=.page_elements_v3.json"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

