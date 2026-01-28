"""
Compatibility shim.

The benchmark command is implemented in `slimgest.benchmark.stages.hf_ocr` to mirror
the `slimgest.local.stages.*` structure.
"""

from .stages.hf_ocr import app, main  # noqa: F401

