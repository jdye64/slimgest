from __future__ import annotations

import typer

from . import benchmark_pdf_pipeline, world_class_pdf_with_crops

app = typer.Typer(help="slimgest world class entrypoint")
app.add_typer(world_class_pdf_with_crops.app, name="world-class-pdf-with-crops")


def main():
    app()


