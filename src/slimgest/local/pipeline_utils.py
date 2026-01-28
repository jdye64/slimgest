from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def extract_text_best_effort(raw: Any) -> str:
    """
    Normalize OCR wrapper outputs into a plain string.

    Nemotron OCR wrapper shapes vary:
    - remote: typically list[dict] where dict has "text"
    - local: often list[dict] per line/word; sometimes dict; sometimes string
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        for k in ("text", "output_text", "generated_text", "ocr_text"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        if "raw" in raw:
            return extract_text_best_effort(raw.get("raw"))
        return str(raw).strip()
    if isinstance(raw, list):
        parts = []
        for item in raw:
            t = extract_text_best_effort(item)
            if t:
                parts.append(t)
        return " ".join(parts).strip()
    return str(raw).strip()


def resize_pad_tensor(
    img: torch.Tensor,
    *,
    target_hw: Tuple[int, int] = (1024, 1024),
    pad_value: float = 114.0,
) -> torch.Tensor:
    """
    Resize+pad CHW image tensor to fixed size (preserve aspect ratio).
    Returns uint8 CHW tensor.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(img.shape)}")
    _, H, W = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
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


@dataclass
class TimerBank:
    """
    Small helper to accumulate timing for named sub-steps.
    """

    acc_s: Dict[str, float]

    def __init__(self) -> None:
        self.acc_s = {}

    def add(self, key: str, dt_s: float) -> None:
        self.acc_s[key] = float(self.acc_s.get(key, 0.0) + float(dt_s))

    def timed(self, key: str):
        bank = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner._t0 = time.perf_counter()
                return None

            def __exit__(self_inner, exc_type, exc, tb):
                bank.add(key, time.perf_counter() - self_inner._t0)
                return False

        return _Ctx()

    def as_dict(self) -> Dict[str, float]:
        return dict(sorted(self.acc_s.items(), key=lambda kv: kv[0]))


def fmt_seconds_hms(s: Optional[float]) -> str:
    if s is None:
        return "unknown"
    try:
        s_f = float(s)
    except Exception:
        return "unknown"
    if s_f < 0 or not (s_f == s_f):  # NaN check
        return "unknown"
    s_i = int(s_f)
    h = s_i // 3600
    m = (s_i % 3600) // 60
    sec = s_i % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"

