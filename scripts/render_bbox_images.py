from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from slimgest.image.render import render_page_element_detections_for_dir

console = Console()
app = typer.Typer(help="Render stage2 (page_elements_v3) bounding boxes onto images.")


@app.command()
def main(
    input_dir: Path = typer.Option(..., "--input-dir", exists=True, file_okay=False, dir_okay=True),
    overwrite: bool = typer.Option(False, "--overwrite"),
    min_score: float = typer.Option(0.0, "--min-score"),
    line_width: int = typer.Option(3, "--line-width", min=1),
    draw_labels: bool = typer.Option(True, "--draw-labels/--no-draw-labels"),
    limit: Optional[int] = typer.Option(None, "--limit"),
) -> None:
    """
    For each image in input_dir, read adjacent stage2 JSON (<image>.page_elements_v3.json)
    and write <image>.page_element_detections.png with bounding boxes rendered.
    """
    outs = render_page_element_detections_for_dir(
        input_dir=input_dir,
        overwrite=overwrite,
        min_score=min_score,
        line_width=line_width,
        draw_labels=draw_labels,
        limit=limit,
    )
    console.print(f"[green]Done[/green] wrote={len(outs)}")


if __name__ == "__main__":
    app()

