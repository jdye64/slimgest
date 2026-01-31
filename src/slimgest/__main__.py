from __future__ import annotations

import typer

from .pdf import app as pdf_app
from .image import app as image_app
from .local import app as local_app
from .recall import app as recall_app
from .ray import app as ray_app
from .benchmark import app as benchmark_app
from .util import app as util_app

app = typer.Typer(help="Slimgest")
app.add_typer(pdf_app, name="pdf")
app.add_typer(image_app, name="image")
app.add_typer(local_app, name="local")
app.add_typer(recall_app, name="recall")
app.add_typer(ray_app, name="ray")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(util_app, name="util")

def main():
    app()
