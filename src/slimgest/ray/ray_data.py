from pathlib import Path
from typing import List, Tuple, Optional

from rich.console import Console
from rich.traceback import install
import typer

app = typer.Typer(help="Ray Data Parallelism for Slim-Gest")
install(show_locals=False)
console = Console()

@app.command()
def run(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=True),
    raw_output_dir: Optional[Path] = typer.Option(None, help="Directory to save raw OCR results (optional)."),
):
    # boilerplate code then call the local run_pipeline function ....
    pass