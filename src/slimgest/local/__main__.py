from __future__ import annotations

import typer

from . import simple

app = typer.Typer(help="Simplest pipeline with limited CPU parallelism while using maximum GPU possible")
app.add_typer(simple.app, name="simple")


def main():
    app()


