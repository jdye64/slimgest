from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


app = typer.Typer(help="Client for slimgest REST API")
console = Console()


async def _submit(server: str, pdf_path: Path, dpi: int, model: str, max_pages: Optional[int]) -> str:
    async with httpx.AsyncClient() as client:
        files = {"file": (pdf_path.name, open(pdf_path, "rb"), "application/pdf")}
        data = {"dpi": str(dpi), "model": model}
        if max_pages is not None:
            data["max_pages"] = str(max_pages)
        r = await client.post(f"{server.rstrip('/')}/v1/jobs", files=files, data=data)
        r.raise_for_status()
        return r.json()["job_id"]


async def _follow(server: str, job_id: str):
    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), TimeElapsedColumn()) as progress:
        task = progress.add_task("Waiting", total=None)
        async with httpx.AsyncClient() as client:
            while True:
                r = await client.get(f"{server.rstrip('/')}/v1/jobs/{job_id}")
                if r.status_code >= 400:
                    console.print(f"[red]Error:[/red] {r.text}")
                    return 1
                data = r.json()
                status = data.get("status")
                phase = data.get("phase")
                prog = data.get("progress", 0.0)
                progress.update(task, description=f"{status} - {phase} ({prog*100:.1f}%)")
                if status in ("completed", "failed"):
                    break
                await asyncio.sleep(1.0)

            if status == "failed":
                console.print(f"[red]Job failed:[/red] {data.get('error')}")
                return 2

            # Fetch result
            r = await client.get(f"{server.rstrip('/')}/v1/jobs/{job_id}/result")
            r.raise_for_status()
            res = r.json()

            # Show metrics table
            metrics = res.get("metrics", {})
            table = Table(title="Metrics (ms)")
            table.add_column("Metric")
            table.add_column("Value")
            for k, v in sorted(metrics.items()):
                table.add_row(k, f"{v:.2f}")
            console.print(table)

            # Print short summary
            console.print(f"[green]Completed[/green]: {len(res.get('pages', []))} pages")
            return 0


@app.command()
def submit(
    pdf: Path = typer.Argument(..., exists=True, dir_okay=False),
    server: str = typer.Option("http://localhost:8000", help="Server URL"),
    dpi: int = typer.Option(220),
    model: str = typer.Option("deepseek-ocr"),
    max_pages: Optional[int] = typer.Option(None),
):
    job_id = asyncio.run(_submit(server, pdf, dpi, model, max_pages))
    console.print(f"Submitted job: [bold]{job_id}[/bold]")


@app.command()
def follow(job_id: str = typer.Argument(...), server: str = typer.Option("http://localhost:8000")):
    code = asyncio.run(_follow(server, job_id))
    raise typer.Exit(code)


