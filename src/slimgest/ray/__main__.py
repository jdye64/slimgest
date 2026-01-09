from __future__ import annotations

import typer

from .ray_data import app as ray_data_app

app = typer.Typer(help="Ray Data Parallelism for Slim-Gest")
app.add_typer(ray_data_app, name="ray-data")


def main():
    app()


