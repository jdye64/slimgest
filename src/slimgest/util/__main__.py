from __future__ import annotations

import typer

from . import inspect_pkl, cleanup_stages, gather_skipped, find_skipped_embeddings_in_query

app = typer.Typer(help="Utilities for inspecting and analyzing data files")
app.add_typer(inspect_pkl.app, name="inspect-pkl")
app.add_typer(cleanup_stages.app, name="cleanup-stages")
app.add_typer(gather_skipped.app, name="gather-skipped")
app.add_typer(find_skipped_embeddings_in_query.app, name="find-skipped-embeddings-in-query")

def main():
    app()
