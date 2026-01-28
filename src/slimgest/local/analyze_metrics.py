from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import typer

from slimgest.local.pipeline_utils import fmt_seconds_hms

app = typer.Typer(help="Analyze per-PDF metrics JSON files produced by `slimgest local simple run`.")


def _iter_metrics_files(metrics_dir: Path) -> List[Path]:
    return sorted([p for p in metrics_dir.iterdir() if p.is_file() and p.name.endswith(".metrics.json")])


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summarize(metrics_blobs: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    model_totals: Dict[str, Dict[str, float]] = {}
    component_totals: Dict[str, float] = {}
    pdfs = len(metrics_blobs)
    pages = 0
    errors = 0

    for blob in metrics_blobs:
        for p in blob.get("pages", []) or []:
            pages += 1
            errs = p.get("errors") or []
            if isinstance(errs, list):
                errors += int(len(errs))
            for k, v in (p.get("timing_s") or {}).items():
                try:
                    component_totals[k] = float(component_totals.get(k, 0.0) + float(v))
                except Exception:
                    pass
            for name, m in (p.get("models") or {}).items():
                cur = model_totals.setdefault(name, {"seconds": 0.0, "calls": 0.0, "items": 0.0})
                cur["seconds"] += float(m.get("seconds", 0.0) or 0.0)
                cur["calls"] += float(m.get("calls", 0) or 0.0)
                cur["items"] += float(m.get("items", 0) or 0.0)

    lines: List[str] = []
    lines.append("slimgest metrics summary")
    lines.append(f"pdfs={pdfs} pages={pages} page_errors={errors}")
    lines.append("")
    lines.append("Top model hotspots (by total seconds):")
    for name, m in sorted(model_totals.items(), key=lambda kv: kv[1]["seconds"], reverse=True)[:15]:
        calls = int(m["calls"])
        items = int(m["items"])
        secs = float(m["seconds"])
        per_call = secs / calls if calls else 0.0
        per_item = secs / items if items else 0.0
        lines.append(f"- {name}: seconds={secs:.2f} calls={calls} items={items} per_call_s={per_call:.3f} per_item_s={per_item:.6f}")

    lines.append("")
    lines.append("Top timed components (by total seconds):")
    for k, v in sorted(component_totals.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        lines.append(f"- {k}: {float(v):.2f}s")

    report = {
        "pdfs": int(pdfs),
        "pages": int(pages),
        "page_errors": int(errors),
        "model_totals": model_totals,
        "component_totals": component_totals,
    }
    return "\n".join(lines).strip() + "\n", report


@app.command()
def run(
    metrics_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    output_txt: Path = typer.Option(Path("./results.txt"), "--output-txt", help="Where to write the text report."),
):
    """
    Reads `*.metrics.json` from a directory and writes a hotspot summary.
    """
    metrics_dir = Path(metrics_dir)
    files = _iter_metrics_files(metrics_dir)
    if not files:
        raise typer.BadParameter(f"No '*.metrics.json' found in {metrics_dir}")

    blobs: List[Dict[str, Any]] = []
    for p in files:
        blob = _read_json(p)
        if isinstance(blob, dict):
            blobs.append(blob)

    txt, report = _summarize(blobs)
    output_txt.write_text(txt, encoding="utf-8")
    typer.echo(f"Wrote {output_txt} (pdfs={report['pdfs']} pages={report['pages']} wall={fmt_seconds_hms(report['component_totals'].get('page_total', 0.0))})")

