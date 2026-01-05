import os
from pathlib import Path

import typer

import pikepdf
from PIL import Image

app = typer.Typer(help="Process PDFs locally using shared pipeline")

@app.command()
def run(
    input_pdf: Path = typer.Argument(..., exists=True, file_okay=True),
):
    print("TODO: Implement extract utilities")

