from __future__ import annotations

import os
import platform
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def _read_first_match(path: Path, pattern: str) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    return (m.group(1) or "").strip()


def _proc_meminfo_bytes() -> Dict[str, int]:
    out: Dict[str, int] = {}
    p = Path("/proc/meminfo")
    try:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            # e.g. "MemTotal:       131900020 kB"
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            k = parts[0].strip()
            rhs = parts[1].strip()
            m = re.match(r"^(\d+)\s+kB$", rhs)
            if not m:
                continue
            out[k] = int(m.group(1)) * 1024
    except Exception:
        return {}
    return out


def _cpu_model_name() -> Optional[str]:
    return _read_first_match(Path("/proc/cpuinfo"), r"^model name\s*:\s*(.+)$")


def _fmt_bytes(n: Optional[int]) -> str:
    if n is None:
        return "unknown"
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{n} B"


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_memory_bytes: int
    capability: str
    multi_processor_count: int


@dataclass(frozen=True)
class SystemInfo:
    platform: str
    python: str
    cpu_model: Optional[str]
    cpu_logical_cores: Optional[int]
    mem_total_bytes: Optional[int]
    mem_available_bytes: Optional[int]
    torch_version: str
    torch_cuda_version: Optional[str]
    cudnn_version: Optional[int]
    cuda_available: bool
    cuda_device_count: int
    cuda_visible_devices: Optional[str]
    gpus: List[GpuInfo]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def collect_system_info() -> SystemInfo:
    mem = _proc_meminfo_bytes()
    gpus: List[GpuInfo] = []
    if torch.cuda.is_available():
        for i in range(int(torch.cuda.device_count())):
            try:
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    GpuInfo(
                        index=int(i),
                        name=str(getattr(props, "name", "")),
                        total_memory_bytes=int(getattr(props, "total_memory", 0)),
                        capability=f"{int(getattr(props, 'major', 0))}.{int(getattr(props, 'minor', 0))}",
                        multi_processor_count=int(getattr(props, "multi_processor_count", 0)),
                    )
                )
            except Exception:
                continue

    cudnn_v: Optional[int]
    try:
        cudnn_v = int(torch.backends.cudnn.version() or 0) or None
    except Exception:
        cudnn_v = None

    return SystemInfo(
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python=sys.version.split()[0],
        cpu_model=_cpu_model_name(),
        cpu_logical_cores=os.cpu_count(),
        mem_total_bytes=mem.get("MemTotal"),
        mem_available_bytes=mem.get("MemAvailable"),
        torch_version=str(torch.__version__),
        torch_cuda_version=getattr(torch.version, "cuda", None),
        cudnn_version=cudnn_v,
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device_count=int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
        gpus=gpus,
    )


def format_system_info_human(si: SystemInfo, *, selected_gpu: Optional[int] = None) -> str:
    lines: List[str] = []
    lines.append(f"platform={si.platform}")
    lines.append(f"python={si.python}")
    if si.cpu_model:
        lines.append(f"cpu_model={si.cpu_model}")
    if si.cpu_logical_cores is not None:
        lines.append(f"cpu_logical_cores={si.cpu_logical_cores}")
    lines.append(f"mem_total={_fmt_bytes(si.mem_total_bytes)} mem_available={_fmt_bytes(si.mem_available_bytes)}")
    lines.append(
        f"torch={si.torch_version} cuda_available={si.cuda_available} torch_cuda={si.torch_cuda_version or 'none'} cudnn={si.cudnn_version or 'none'}"
    )
    if si.cuda_visible_devices is not None:
        lines.append(f"CUDA_VISIBLE_DEVICES={si.cuda_visible_devices}")

    if si.gpus:
        for g in si.gpus:
            sel = " (selected)" if selected_gpu is not None and int(g.index) == int(selected_gpu) else ""
            lines.append(
                f"gpu[{g.index}]{sel} name={g.name} total_mem={_fmt_bytes(g.total_memory_bytes)} cc={g.capability} sm={g.multi_processor_count}"
            )
    else:
        lines.append("gpu=none")

    return "\n".join(lines)

