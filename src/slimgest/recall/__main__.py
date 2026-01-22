from __future__ import annotations

import typer

from .topk import topk

app = typer.Typer(help="Recall utilities for SlimGest embeddings")
app.command(name="topk")(topk)


def main():
    app()

