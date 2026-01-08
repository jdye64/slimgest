from __future__ import annotations

import typer

from .pdf import app as pdf_app
from .local import app as local_app

app = typer.Typer(help="Slimgest")
app.add_typer(pdf_app, name="pdf")
app.add_typer(local_app, name="local")

def main():
    app()
