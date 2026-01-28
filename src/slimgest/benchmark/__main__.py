from __future__ import annotations

import typer

from .stages import hf_ocr

app = typer.Typer(help="Benchmark utilities for slimgest (repeatable perf runs across machines).")
app.add_typer(hf_ocr.app, name="hf-ocr")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

