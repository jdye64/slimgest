from __future__ import annotations

import typer

from . import deepseek_local, local, client

app = typer.Typer(help="slimgest CLI entrypoint")
app.add_typer(deepseek_local.app, name="deepseek-local")
app.add_typer(local.app, name="local")
app.add_typer(client.app, name="client")


def main():
    app()


