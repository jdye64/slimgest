from __future__ import annotations

import typer

from . import simple_all_gpu
from . import in_memory

app = typer.Typer(help="Simplest pipeline with limited CPU parallelism while using maximum GPU possible")
app.add_typer(simple_all_gpu.app, name="simple-all-gpu")
app.add_typer(in_memory.app, name="in-memory")


def main():
    app()


