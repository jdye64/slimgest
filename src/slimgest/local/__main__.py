from __future__ import annotations

import typer

from . import simple_all_gpu

app = typer.Typer(help="Simplest pipeline with limited CPU parallelism while using maximum GPU possible")
app.add_typer(simple_all_gpu.app, name="simple-all-gpu")


def main():
    app()


