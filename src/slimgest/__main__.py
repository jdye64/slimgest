from __future__ import annotations

import typer

from .cicd import app as cicd_app
from .pdf import app as pdf_app

app = typer.Typer(help="Slimgest")
app.add_typer(cicd_app, name="cicd")
app.add_typer(pdf_app, name="pdf")

def main():
    app()
