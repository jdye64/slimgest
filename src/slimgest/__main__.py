from __future__ import annotations

import typer

from .cicd import app as cicd_app
from .pdf import app as pdf_app
from .simple import app as simple_app

app = typer.Typer(help="Slimgest")
app.add_typer(cicd_app, name="cicd")
app.add_typer(pdf_app, name="pdf")
app.add_typer(simple_app, name="simple")

def main():
    app()
