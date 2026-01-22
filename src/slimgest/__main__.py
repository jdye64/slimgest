from __future__ import annotations

import typer

from .pdf import app as pdf_app
from .local import app as local_app
from .recall import app as recall_app
from .ray import app as ray_app

app = typer.Typer(help="Slimgest")
app.add_typer(pdf_app, name="pdf")
app.add_typer(local_app, name="local")
app.add_typer(recall_app, name="recall")
app.add_typer(ray_app, name="ray")

def main():
    app()
