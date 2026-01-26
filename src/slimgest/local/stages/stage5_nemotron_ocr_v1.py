from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.traceback import install
import torch
import torch.nn.functional as F
import typer
from tqdm import tqdm

from slimgest.model.local.nemotron_ocr_v1 import NemotronOCRV1

from ._io import crop_tensor_normalized_xyxy, iter_images, load_image_rgb_chw_u8, read_json, write_json


install(show_locals=False)
console = Console()
app = typer.Typer(
    help="Stage 5: run nemotron_ocr_v1 over detections from stages 3/4 and save JSON alongside each image."
)

DEFAULT_INPUT_DIR = Path("./data/pages")


def _stage3_json_for_image(img_path: Path) -> Path:
    return img_path.with_name(img_path.name + ".graphic_elements_v1.json")


def _stage4_json_for_image(img_path: Path) -> Path:
    return img_path.with_name(img_path.name + ".table_structure_v1.json")


def _out_path_for_image(img_path: Path) -> Path:
    return img_path.with_name(img_path.name + ".nemotron_ocr_v1.json")


def _resize_pad_tensor(img: torch.Tensor, target_hw: Tuple[int, int] = (1024, 1024), pad_value: float = 114.0) -> torch.Tensor:
    """
    Resize+pad CHW image tensor to fixed size (preserve aspect ratio).
    Returns uint8 CHW tensor.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(img.shape)}")
    C, H, W = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
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


def _collect_page_level_detections(s3: Any, s4: Any) -> List[Dict[str, Any]]:
    """
    Returns a flat list of items each with:
      - source: "graphic_elements_v1" | "table_structure_v1"
      - bbox_xyxy_norm_in_page: [x1,y1,x2,y2]
      - meta: other fields
    """
    out: List[Dict[str, Any]] = []
    for source, blob in (("graphic_elements_v1", s3), ("table_structure_v1", s4)):
        if not blob:
            continue
        for region in (blob.get("regions") or []):
            page_el = region.get("page_element") or {}
            for det in (region.get("detections") or []):
                bbox_page = det.get("bbox_xyxy_norm_in_page")
                if not bbox_page or len(bbox_page) != 4:
                    continue
                out.append(
                    {
                        "source_model": source,
                        "page_element": page_el,
                        "detection": det,
                        "bbox_xyxy_norm_in_page": bbox_page,
                    }
                )
    return out


@app.command()
def run(
    input_dir: Path = typer.Option(DEFAULT_INPUT_DIR, "--input-dir", exists=True, file_okay=False),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device for image tensors."),
    ocr_model_dir: Path = typer.Option(
        Path("/raid/jdyer/slimgest/models/nemotron-ocr-v1/checkpoints"),
        "--ocr-model-dir",
        help="Local nemotron-ocr-v1 checkpoints directory (ignored if --ocr-endpoint is set).",
    ),
    ocr_endpoint: Optional[str] = typer.Option(
        None,
        help="Optional OCR NIM endpoint URL. If set, OCR runs remotely and local weights are not loaded.",
    ),
    remote_batch_size: int = typer.Option(32, help="Remote OCR batch size when using --ocr-endpoint."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing JSON outputs."),
    limit: Optional[int] = typer.Option(None, help="Optionally limit number of images processed."),
):
    dev = torch.device(device)
    ocr = NemotronOCRV1(
        model_dir=str(ocr_model_dir),
        endpoint=ocr_endpoint,
        remote_batch_size=int(remote_batch_size),
    )

    images = iter_images(input_dir)
    if limit is not None:
        images = images[: int(limit)]

    console.print(f"[bold cyan]Stage5[/bold cyan] images={len(images)} input_dir={input_dir} device={dev}")

    processed = 0
    skipped = 0
    missing_prereq = 0
    bad_prereq = 0
    for img_path in tqdm(images, desc="Stage5 images", unit="img"):
        out_path = _out_path_for_image(img_path)
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        s3_path = _stage3_json_for_image(img_path)
        s4_path = _stage4_json_for_image(img_path)
        if not s3_path.exists() or not s4_path.exists():
            missing_prereq += 1
            continue

        try:
            s3 = read_json(s3_path)
            s4 = read_json(s4_path)
        except Exception:
            bad_prereq += 1
            continue
        items = _collect_page_level_detections(s3, s4)

        page_tensor, (h, w) = load_image_rgb_chw_u8(img_path, dev)

        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []
        with torch.inference_mode():
            for it in items:
                bbox = it["bbox_xyxy_norm_in_page"]
                crop = crop_tensor_normalized_xyxy(page_tensor, bbox)
                crop_in = _resize_pad_tensor(crop, target_hw=(1024, 1024))
                out = ocr.invoke(crop_in)
                results.append(
                    {
                        "source_model": it["source_model"],
                        "bbox_xyxy_norm_in_page": bbox,
                        "page_element": it.get("page_element"),
                        "detection": it.get("detection"),
                        "ocr_raw": out,
                        # Best-effort text flattening (works for both local and remote shapes)
                        "ocr_text": (
                            " ".join([str(p.get("text", "")) for p in out]).strip()
                            if isinstance(out, list)
                            else str(out)
                        ),
                    }
                )
        dt = time.perf_counter() - t0

        payload: Dict[str, Any] = {
            "schema_version": 1,
            "stage": 5,
            "model": "nemotron_ocr_v1",
            "image": {"path": str(img_path), "height": int(h), "width": int(w)},
            "stage3_json": str(s3_path),
            "stage4_json": str(s4_path),
            "regions": results,
            "timing": {"seconds": float(dt)},
        }
        write_json(out_path, payload)
        processed += 1

    console.print(
        f"[green]Done[/green] processed={processed} skipped={skipped} missing_prereq={missing_prereq} bad_prereq={bad_prereq} wrote_json_suffix=.nemotron_ocr_v1.json"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

