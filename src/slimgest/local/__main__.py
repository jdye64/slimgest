from __future__ import annotations

import typer

from . import simple
from .stages import (
    stage2_page_elements_v3,
    stage3_graphic_elements_v1,
    stage4_table_structure_v1,
    stage5_nemotron_ocr_v1,
    stage6_embeddings,
    report_stage_outputs,
)

app = typer.Typer(help="Simplest pipeline with limited CPU parallelism while using maximum GPU possible")
app.add_typer(simple.app, name="simple")
app.add_typer(stage2_page_elements_v3.app, name="stage2")
app.add_typer(stage3_graphic_elements_v1.app, name="stage3")
app.add_typer(stage4_table_structure_v1.app, name="stage4")
app.add_typer(stage5_nemotron_ocr_v1.app, name="stage5")
app.add_typer(stage6_embeddings.app, name="stage6")
app.add_typer(report_stage_outputs.app, name="report")


def main():
    app()


