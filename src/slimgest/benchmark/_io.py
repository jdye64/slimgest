from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def iter_images(input_dir: Path, *, recursive: bool = True) -> List[Path]:
    """
    Return a stable-sorted list of image paths from a directory.
    """
    root = Path(input_dir)
    if not root.exists() or not root.is_dir():
        return []

    out: List[Path] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    out.append(p)
    else:
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p)

    # Stable, cross-platform ordering
    out.sort(key=lambda p: str(p))
    return out


def load_image_rgb_chw_u8(path: Path, device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Load an image as CHW uint8 tensor on device.
    Returns (tensor, (H, W)).
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im, dtype=np.uint8)  # HWC
    h, w = int(arr.shape[0]), int(arr.shape[1])
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
    t = t.to(device=device, dtype=torch.uint8, non_blocking=(device.type == "cuda"))
    return t, (h, w)

