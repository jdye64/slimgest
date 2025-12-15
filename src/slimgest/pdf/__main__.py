from __future__ import annotations

import typer

from . import convert, examine, extract

app = typer.Typer(help="Utilities for processing, rendering, examining, and converting PDFs")
app.add_typer(convert.app, name="convert")
app.add_typer(examine.app, name="examine")
app.add_typer(extract.app, name="extract")

def main():
    app()
