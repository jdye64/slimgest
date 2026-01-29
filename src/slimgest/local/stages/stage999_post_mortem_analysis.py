from __future__ import annotations

import csv
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import typer
from rich.console import Console


console = Console()
app = typer.Typer(
    help=(
        "Stage 999: Post-mortem analysis viewer for per-page artifacts.\n\n"
        "Reads bo767_query_gt.csv (query -> pdf_page) and loads page images + adjacent stage sidecars:\n"
        "  - <image>.+page_elements_v3.json\n"
        "  - <image>.+graphic_elements_v1.json\n"
        "  - <image>.+table_structure_v1.json\n"
        "  - <image>.+nemotron_ocr_v1.json\n"
        "  - <image_stem>.pdfium_text.txt\n"
        "Optionally displays <image>.page_element_detections.png if present.\n\n"
        "Provides an interactive UI (Tk) for browsing and a PDF report export mode."
    )
)


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class QueryRow:
    query: str
    pdf: str
    page: Optional[int]  # CSV 'page' column (often 0-based in this dataset)
    modality: Optional[str]
    pdf_page: str  # e.g. "1102434_20"

    @property
    def pdf_page_pdf(self) -> str:
        # "1102434_20" -> "1102434"
        parts = self.pdf_page.split("_", 1)
        return parts[0] if parts else self.pdf

    @property
    def pdf_page_page1(self) -> Optional[int]:
        # "1102434_20" -> 20 (1-based)
        parts = self.pdf_page.split("_", 1)
        if len(parts) != 2:
            return None
        try:
            return int(parts[1])
        except Exception:
            return None


@dataclass(frozen=True)
class ResolvedExample:
    row: QueryRow
    image_path: Optional[Path]


def _read_json_best_effort(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _read_text_best_effort(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def _iter_images(input_dir: Path, recursive: bool) -> List[Path]:
    it = input_dir.rglob("*") if recursive else input_dir.iterdir()
    out: List[Path] = []
    for p in it:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return sorted(out)


def _index_images(images: Sequence[Path]) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    Returns:
      - by_name: filename -> path
      - by_stem: stem -> path
    If multiple paths collide, keeps the first in sorted order.
    """
    by_name: Dict[str, Path] = {}
    by_stem: Dict[str, Path] = {}
    for p in images:
        by_name.setdefault(p.name, p)
        by_stem.setdefault(p.stem, p)
    return by_name, by_stem


def _candidate_image_stems(row: QueryRow) -> List[str]:
    """
    Try to resolve dataset IDs to rendered image stems.

    Primary convention from `slimgest.pdf.convert`:
      <pdf_basename>_page<NNNN>.<ext> where NNNN is 1-based.

    The bo767 CSV includes `pdf_page` like "<pdf>_<page1based>".
    """
    cands: List[str] = []

    pdf = (row.pdf_page_pdf or row.pdf or "").strip()
    page1 = row.pdf_page_page1
    if pdf and page1 is not None and page1 > 0:
        cands.append(f"{pdf}_page{int(page1):04d}")
        cands.append(f"{pdf}_page{int(page1)}")

    # Fallback: if CSV 'page' looks 0-based, try +1.
    if pdf and row.page is not None:
        try:
            p0 = int(row.page)
            if p0 >= 0:
                cands.append(f"{pdf}_page{int(p0 + 1):04d}")
                cands.append(f"{pdf}_page{int(p0 + 1)}")
        except Exception:
            pass

    # Last resort: sometimes datasets store "<pdf>_<page>" images directly.
    if row.pdf_page:
        cands.append(row.pdf_page)

    # De-dupe while preserving order
    seen: set[str] = set()
    out: List[str] = []
    for s in cands:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _resolve_image_for_row(row: QueryRow, *, by_stem: Dict[str, Path], by_name: Dict[str, Path]) -> Optional[Path]:
    # Try exact stems first
    for stem in _candidate_image_stems(row):
        p = by_stem.get(stem)
        if p is not None:
            return p

    # Fallback: try filenames with extensions
    for stem in _candidate_image_stems(row):
        for ext in IMAGE_EXTS:
            p = by_name.get(stem + ext)
            if p is not None:
                return p

    return None


def _load_csv_rows(csv_path: Path) -> List[QueryRow]:
    rows: List[QueryRow] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for rec in r:
            query = (rec.get("query") or "").strip()
            pdf = (rec.get("pdf") or "").strip()
            modality = (rec.get("modality") or "").strip() or None
            pdf_page = (rec.get("pdf_page") or "").strip()
            page_raw = rec.get("page")
            page: Optional[int]
            try:
                page = int(page_raw) if page_raw is not None and str(page_raw).strip() != "" else None
            except Exception:
                page = None
            if not pdf_page:
                # if missing, attempt to synthesize
                if pdf and page is not None:
                    pdf_page = f"{pdf}_{int(page)}"
            if not query and not pdf_page:
                continue
            rows.append(QueryRow(query=query, pdf=pdf, page=page, modality=modality, pdf_page=pdf_page))
    return rows


def _paths_for_image(img_path: Path) -> Dict[str, Path]:
    # Note: stage2-5 scripts use img_path.name + ".suffix.json" (double-extension).
    return {
        "img": img_path,
        "img_overlay": img_path.with_name(img_path.name + ".page_element_detections.png"),
        "pdfium_text": img_path.with_suffix(".pdfium_text.txt"),
        "stage2": img_path.with_name(img_path.name + ".page_elements_v3.json"),
        "stage3": img_path.with_name(img_path.name + ".graphic_elements_v1.json"),
        "stage4": img_path.with_name(img_path.name + ".table_structure_v1.json"),
        "stage5": img_path.with_name(img_path.name + ".nemotron_ocr_v1.json"),
        # Stage6 writes this (see stage6_embeddings.py): <image>.+embedder-input.txt
        "embedder_input": img_path.with_name(img_path.name + ".embedder-input.txt"),
    }


def _count_stage2(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not d:
        return {"present": False}
    dets = d.get("detections") or []
    by_label: Dict[str, int] = {}
    for det in dets if isinstance(dets, list) else []:
        if not isinstance(det, dict):
            continue
        name = det.get("label_name")
        if not isinstance(name, str) or not name.strip():
            name = f"label_{det.get('label')}"
        k = str(name)
        by_label[k] = int(by_label.get(k, 0) + 1)
    timing_s = None
    t = d.get("timing")
    if isinstance(t, dict):
        try:
            timing_s = float(t.get("seconds"))
        except Exception:
            timing_s = None
    return {
        "present": True,
        "num_detections": int(len(dets)) if isinstance(dets, list) else 0,
        "by_label": dict(sorted(by_label.items(), key=lambda kv: (-kv[1], kv[0]))),
        "timing_s": timing_s,
    }


def _count_regions_model(d: Optional[Dict[str, Any]], *, regions_key: str = "regions") -> Dict[str, Any]:
    if not d:
        return {"present": False}
    regions = d.get(regions_key) or []
    n_regions = int(len(regions)) if isinstance(regions, list) else 0
    n_dets = 0
    for r in regions if isinstance(regions, list) else []:
        if not isinstance(r, dict):
            continue
        dets = r.get("detections") or []
        if isinstance(dets, list):
            n_dets += int(len(dets))
    timing_s = None
    t = d.get("timing")
    if isinstance(t, dict):
        try:
            timing_s = float(t.get("seconds"))
        except Exception:
            timing_s = None
    return {"present": True, "num_regions": n_regions, "num_detections": int(n_dets), "timing_s": timing_s}


def _count_stage5_ocr(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not d:
        return {"present": False}
    regions = d.get("regions") or []
    n_regions = int(len(regions)) if isinstance(regions, list) else 0
    nonempty = 0
    by_kind: Dict[str, int] = {}
    sample_texts: List[str] = []
    for r in regions if isinstance(regions, list) else []:
        if not isinstance(r, dict):
            continue
        txt = (r.get("ocr_text") or "").strip()
        if txt:
            nonempty += 1
            if len(sample_texts) < 8:
                sample_texts.append(txt.replace("\n", " ").strip())
        name = r.get("label_name")
        if not isinstance(name, str) or not name.strip():
            name = f"label_{r.get('label')}"
        k = str(name)
        by_kind[k] = int(by_kind.get(k, 0) + 1)
    return {
        "present": True,
        "num_regions": n_regions,
        "num_nonempty": int(nonempty),
        "by_label_name": dict(sorted(by_kind.items(), key=lambda kv: (-kv[1], kv[0]))),
        "sample_texts": sample_texts,
    }


def _format_summary_for_page(img_path: Path, row: QueryRow) -> Dict[str, Any]:
    paths = _paths_for_image(img_path)
    pdfium_text = _read_text_best_effort(paths["pdfium_text"]) if paths["pdfium_text"].exists() else ""

    s2_raw = _read_json_best_effort(paths["stage2"]) if paths["stage2"].exists() else None
    s3_raw = _read_json_best_effort(paths["stage3"]) if paths["stage3"].exists() else None
    s4_raw = _read_json_best_effort(paths["stage4"]) if paths["stage4"].exists() else None
    s5_raw = _read_json_best_effort(paths["stage5"]) if paths["stage5"].exists() else None

    return {
        "row": row,
        "paths": paths,
        "pdfium_text": pdfium_text,
        "stage2": _count_stage2(s2_raw),
        "stage3": _count_regions_model(s3_raw),
        "stage4": _count_regions_model(s4_raw),
        "stage5": _count_stage5_ocr(s5_raw),
        "raw": {
            "stage2": s2_raw,
            "stage3": s3_raw,
            "stage4": s4_raw,
            "stage5": s5_raw,
        },
    }


def _compute_global_metrics(examples: Sequence[ResolvedExample]) -> Dict[str, Any]:
    # Aggregate across unique images referenced by the CSV (best-effort).
    uniq_imgs: Dict[Path, QueryRow] = {}
    for ex in examples:
        if ex.image_path is not None and ex.image_path.exists():
            uniq_imgs.setdefault(ex.image_path, ex.row)

    totals = {
        "unique_pages": int(len(uniq_imgs)),
        "missing_images": int(sum(1 for ex in examples if ex.image_path is None)),
        "present": {"stage2": 0, "stage3": 0, "stage4": 0, "stage5": 0, "pdfium_text": 0, "overlay": 0},
        "counts": {"stage2_detections": 0, "stage3_detections": 0, "stage4_detections": 0, "stage5_regions": 0},
    }

    for img_path, row in uniq_imgs.items():
        paths = _paths_for_image(img_path)
        if paths["stage2"].exists():
            totals["present"]["stage2"] += 1
            s2 = _read_json_best_effort(paths["stage2"])
            dets = (s2 or {}).get("detections") if isinstance(s2, dict) else None
            if isinstance(dets, list):
                totals["counts"]["stage2_detections"] += int(len(dets))
        if paths["stage3"].exists():
            totals["present"]["stage3"] += 1
            s3 = _read_json_best_effort(paths["stage3"])
            regions = (s3 or {}).get("regions") if isinstance(s3, dict) else None
            if isinstance(regions, list):
                for r in regions:
                    if isinstance(r, dict) and isinstance(r.get("detections"), list):
                        totals["counts"]["stage3_detections"] += int(len(r["detections"]))
        if paths["stage4"].exists():
            totals["present"]["stage4"] += 1
            s4 = _read_json_best_effort(paths["stage4"])
            regions = (s4 or {}).get("regions") if isinstance(s4, dict) else None
            if isinstance(regions, list):
                for r in regions:
                    if isinstance(r, dict) and isinstance(r.get("detections"), list):
                        totals["counts"]["stage4_detections"] += int(len(r["detections"]))
        if paths["stage5"].exists():
            totals["present"]["stage5"] += 1
            s5 = _read_json_best_effort(paths["stage5"])
            regions = (s5 or {}).get("regions") if isinstance(s5, dict) else None
            if isinstance(regions, list):
                totals["counts"]["stage5_regions"] += int(len(regions))
        if paths["pdfium_text"].exists():
            totals["present"]["pdfium_text"] += 1
        if paths["img_overlay"].exists():
            totals["present"]["overlay"] += 1

    return totals


def _export_report_pdf(
    *,
    examples: Sequence[ResolvedExample],
    output_pdf: Path,
    recursive: bool,
    limit_unique_pages: Optional[int] = None,
    include_overlay_if_present: bool = True,
) -> None:
    """
    Create a PDF report where each PDF page is a rendered composite image.
    Uses only existing dependencies (Pillow + img2pdf).
    """
    from PIL import Image, ImageDraw, ImageFont  # pillow is a project dependency
    import img2pdf  # project dependency

    # Build unique pages in stable order (by image name) for reproducibility.
    uniq: Dict[Path, QueryRow] = {}
    for ex in examples:
        if ex.image_path is None or (not ex.image_path.exists()):
            continue
        uniq.setdefault(ex.image_path, ex.row)

    page_items = sorted(uniq.items(), key=lambda kv: kv[0].name)
    if limit_unique_pages is not None:
        page_items = page_items[: int(limit_unique_pages)]

    # Try a reasonable default font; fall back to PIL default.
    def _load_font(size: int) -> ImageFont.ImageFont:
        try:
            # common on many Linux distros
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

    font_title = _load_font(22)
    font_body = _load_font(14)
    font_small = _load_font(12)

    # Report page layout (pixel-based). Keep it modest for speed.
    W, H = 1654, 2339  # ~A4 @ 150 DPI
    margin = 30
    gutter = 20

    def _fit(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(max_w / max(1, w), max_h / max(1, h))
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        return im.resize((nw, nh), Image.BILINEAR)

    def _wrap(s: str, width: int) -> List[str]:
        s = (s or "").replace("\r", "").strip()
        if not s:
            return []
        return textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False)

    def _draw_block(draw: ImageDraw.ImageDraw, x: int, y: int, lines: Sequence[str], font, line_h: int) -> int:
        yy = y
        for ln in lines:
            draw.text((x, yy), ln, fill=(0, 0, 0), font=font)
            yy += line_h
        return yy

    rendered_pages: List[bytes] = []
    for img_path, row in page_items:
        summary = _format_summary_for_page(img_path, row)
        paths = summary["paths"]

        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Title
        y = margin
        title = f"{row.pdf_page}  |  {img_path.name}"
        draw.text((margin, y), title, fill=(0, 0, 0), font=font_title)
        y += 34

        # Query
        q_lines = _wrap(f"Query: {row.query}", width=110)
        y = _draw_block(draw, margin, y, q_lines[:6], font=font_body, line_h=18) + 10

        # Images region (two columns: original + overlay if present)
        img_area_h = 820
        col_w = (W - 2 * margin - gutter) // 2

        with Image.open(paths["img"]) as im:
            left = _fit(im, col_w, img_area_h)
        canvas.paste(left, (margin, y))

        if include_overlay_if_present and paths["img_overlay"].exists():
            with Image.open(paths["img_overlay"]) as im2:
                right = _fit(im2, col_w, img_area_h)
            canvas.paste(right, (margin + col_w + gutter, y))
            draw.text((margin + col_w + gutter, y + right.size[1] + 6), "page_element_detections", fill=(0, 0, 0), font=font_small)
        else:
            draw.text((margin + col_w + gutter, y), "(no overlay image found)", fill=(120, 120, 120), font=font_small)

        y = y + img_area_h + 25

        # Metrics summary
        s2 = summary["stage2"]
        s3 = summary["stage3"]
        s4 = summary["stage4"]
        s5 = summary["stage5"]
        metrics_lines = [
            "Model / stage presence & counts:",
            f"- pdfium_text: {'present' if bool(summary['pdfium_text']) else 'missing/empty'} (file: {'yes' if paths['pdfium_text'].exists() else 'no'})",
            f"- stage2 page_elements_v3: present={s2.get('present')} dets={s2.get('num_detections', 0)} timing_s={s2.get('timing_s')}",
            f"- stage3 graphic_elements_v1: present={s3.get('present')} regions={s3.get('num_regions', 0)} dets={s3.get('num_detections', 0)} timing_s={s3.get('timing_s')}",
            f"- stage4 table_structure_v1: present={s4.get('present')} regions={s4.get('num_regions', 0)} dets={s4.get('num_detections', 0)} timing_s={s4.get('timing_s')}",
            f"- stage5 nemotron_ocr_v1: present={s5.get('present')} regions={s5.get('num_regions', 0)} nonempty={s5.get('num_nonempty', 0)}",
            "",
        ]
        y = _draw_block(draw, margin, y, metrics_lines, font=font_small, line_h=16) + 8

        # PDFium + OCR samples (truncate)
        pdfium_lines = _wrap("PDFium text: " + (summary["pdfium_text"] or ""), width=120)[:18]
        ocr_samples = summary["stage5"].get("sample_texts") or []
        ocr_lines: List[str] = []
        if ocr_samples:
            ocr_lines.append("OCR sample texts:")
            for t in ocr_samples[:6]:
                for ln in _wrap("- " + t, width=120)[:2]:
                    ocr_lines.append(ln)
        else:
            ocr_lines = ["OCR sample texts: (none)"]

        y = _draw_block(draw, margin, y, pdfium_lines, font=font_small, line_h=16) + 8
        y = _draw_block(draw, margin, y, ocr_lines, font=font_small, line_h=16) + 8

        # Render to PNG bytes (then img2pdf will embed as a PDF page).
        import io

        buf = io.BytesIO()
        canvas.save(buf, format="PNG", optimize=True)
        rendered_pages.append(buf.getvalue())

    if not rendered_pages:
        raise typer.BadParameter("No pages could be resolved to images; nothing to export.")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = img2pdf.convert(rendered_pages)
    output_pdf.write_bytes(pdf_bytes)


def _gather_results_zip(
    *,
    examples: Sequence[ResolvedExample],
    input_dir: Path,
    csv_path: Path,
    zip_path: Path,
    recursive: bool,
    limit_unique_pages: Optional[int] = None,
) -> None:
    """
    Gather the artifacts used by the UI into a single shareable zip:
      - the CSV
      - resolved page images + adjacent artifacts (if present)
      - a manifest.json with pointers + basic stats
    """
    import time
    import zipfile

    input_dir = Path(input_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # Unique pages in stable order.
    uniq: Dict[Path, QueryRow] = {}
    for ex in examples:
        if ex.image_path is None or (not ex.image_path.exists()):
            continue
        uniq.setdefault(ex.image_path, ex.row)
    page_items = sorted(uniq.items(), key=lambda kv: kv[0].name)
    if limit_unique_pages is not None:
        page_items = page_items[: int(limit_unique_pages)]

    # Helper to store files under a stable namespace.
    def _arc_for(p: Path) -> str:
        try:
            rel = p.resolve().relative_to(input_dir.resolve())
            return str(Path("pages") / rel)
        except Exception:
            # Not under input_dir (or resolve fails); fall back to basename.
            return str(Path("pages") / p.name)

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "created_unix_s": int(time.time()),
        "input_dir": str(input_dir),
        "recursive": bool(recursive),
        "csv_path": str(csv_path),
        "counts": {
            "csv_rows": int(len(examples)),
            "unique_pages_resolved": int(len(page_items)),
            "unique_pages_limit": int(limit_unique_pages) if limit_unique_pages is not None else None,
        },
        "entries": [],
    }

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Include the CSV verbatim.
        try:
            zf.write(csv_path, arcname="bo767_query_gt.csv")
        except Exception:
            # still allow zip creation even if CSV isn't readable for some reason
            pass

        # Include a small readme to guide recipients.
        readme = (
            "slimgest stage999 gathered results\n"
            "\n"
            "Contents:\n"
            "- bo767_query_gt.csv: original query->pdf_page mapping\n"
            "- pages/: resolved page images and adjacent artifacts (sidecar JSON, pdfium text, overlay, embedder input)\n"
            "- manifest.json: what was included and how it maps to the CSV\n"
        )
        zf.writestr("README.txt", readme)

        # Add per-page artifacts
        for img_path, row in page_items:
            paths = _paths_for_image(img_path)
            files: List[Tuple[str, Path]] = []
            for k in ("img", "img_overlay", "pdfium_text", "stage2", "stage3", "stage4", "stage5", "embedder_input"):
                p = paths.get(k)
                if isinstance(p, Path) and p.exists():
                    files.append((k, p))

            # Write files into zip
            included: Dict[str, str] = {}
            for kind, p in files:
                arc = _arc_for(p)
                # Avoid overwriting if collisions happen; disambiguate with a prefix.
                if arc in zf.namelist():
                    arc = str(Path("pages") / f"{img_path.stem}__{p.name}")
                zf.write(p, arcname=arc)
                included[kind] = arc

            manifest["entries"].append(
                {
                    "pdf_page": row.pdf_page,
                    "query": row.query,
                    "modality": row.modality,
                    "image_name": img_path.name,
                    "included": included,
                }
            )

        # Always write manifest at end.
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))


def _run_ui(*, examples: Sequence[ResolvedExample], global_metrics: Dict[str, Any]) -> None:
    # Tk is stdlib; pillow is project dependency.
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk

    def _short(s: str, n: int = 90) -> str:
        s = (s or "").replace("\n", " ").strip()
        return (s[: n - 1] + "…") if len(s) > n else s

    # Precompute list entries text (stable indices).
    list_labels: List[str] = []
    for ex in examples:
        row = ex.row
        status = "OK" if ex.image_path is not None else "MISSING"
        list_labels.append(f"{row.pdf_page} [{status}]  |  {_short(row.query)}")

    # Cache for loaded page summaries to keep UI snappy.
    summary_cache: Dict[int, Dict[str, Any]] = {}

    root = tk.Tk()
    root.title("slimgest stage999 post-mortem analysis")

    # Layout: left search+list, right detail panel.
    root.columnconfigure(0, weight=0)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)

    left = ttk.Frame(root, padding=8)
    left.grid(row=0, column=0, sticky="nsw")
    left.rowconfigure(2, weight=1)

    right = ttk.Frame(root, padding=8)
    right.grid(row=0, column=1, sticky="nsew")
    right.columnconfigure(0, weight=1)
    right.rowconfigure(2, weight=1)

    # Search + global metrics
    ttk.Label(left, text="Search (query or pdf_page):").grid(row=0, column=0, sticky="w")
    search_var = tk.StringVar()
    search_entry = ttk.Entry(left, textvariable=search_var, width=42)
    search_entry.grid(row=1, column=0, sticky="ew", pady=(0, 8))

    gm = global_metrics
    gm_text = (
        f"unique_pages={gm.get('unique_pages')} missing_images={gm.get('missing_images')}\n"
        f"present: s2={gm.get('present', {}).get('stage2')} s3={gm.get('present', {}).get('stage3')} "
        f"s4={gm.get('present', {}).get('stage4')} s5={gm.get('present', {}).get('stage5')} "
        f"pdfium={gm.get('present', {}).get('pdfium_text')} overlay={gm.get('present', {}).get('overlay')}"
    )
    ttk.Label(left, text=gm_text, justify="left").grid(row=3, column=0, sticky="ew", pady=(8, 0))

    # Listbox with scrollbar
    list_frame = ttk.Frame(left)
    list_frame.grid(row=2, column=0, sticky="nsew")
    list_frame.rowconfigure(0, weight=1)
    list_frame.columnconfigure(0, weight=1)

    lb = tk.Listbox(list_frame, width=54, height=28, activestyle="dotbox")
    sb = ttk.Scrollbar(list_frame, orient="vertical", command=lb.yview)
    lb.configure(yscrollcommand=sb.set)
    lb.grid(row=0, column=0, sticky="nsew")
    sb.grid(row=0, column=1, sticky="ns")

    # Right panel widgets
    query_lbl = ttk.Label(right, text="", wraplength=820, justify="left")
    query_lbl.grid(row=0, column=0, sticky="ew")

    path_lbl = ttk.Label(right, text="", justify="left")
    path_lbl.grid(row=1, column=0, sticky="ew", pady=(2, 8))

    # Images row
    img_row = ttk.Frame(right)
    img_row.grid(row=2, column=0, sticky="nsew")
    img_row.columnconfigure(0, weight=1)
    img_row.columnconfigure(1, weight=1)
    img_row.rowconfigure(0, weight=1)

    img_lbl = ttk.Label(img_row)
    img_lbl.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    overlay_lbl = ttk.Label(img_row)
    overlay_lbl.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    # Text notebook
    nb = ttk.Notebook(right)
    nb.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
    right.rowconfigure(3, weight=1)

    def _make_text_tab(title: str) -> tk.Text:
        frame = ttk.Frame(nb)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        t = tk.Text(frame, wrap="word")
        s = ttk.Scrollbar(frame, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=s.set)
        t.grid(row=0, column=0, sticky="nsew")
        s.grid(row=0, column=1, sticky="ns")
        nb.add(frame, text=title)
        return t

    pdfium_text = _make_text_tab("PDFium text")
    det_text = _make_text_tab("Detections / OCR / metrics")
    embedder_input_text = _make_text_tab("Embedder input")
    raw_text = _make_text_tab("Raw JSON")

    # Keep references to images to prevent GC.
    img_refs: Dict[str, Any] = {"img": None, "overlay": None}

    def _set_text(widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content or "")
        widget.configure(state="disabled")

    def _pretty_json(obj: Any, *, limit_chars: int = 250_000) -> str:
        if obj is None:
            return ""
        try:
            s = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            s = str(obj)
        if len(s) > int(limit_chars):
            s = s[: int(limit_chars)] + "\n... (truncated) ...\n"
        return s

    def _render_image_to_label(label: ttk.Label, path: Optional[Path]) -> None:
        if path is None or (not path.exists()):
            label.configure(image="", text="(missing)", anchor="center")
            return
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                # Fit to a reasonable UI size
                max_w, max_h = 520, 520
                w, h = im.size
                scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                im = im.resize((nw, nh), Image.BILINEAR)
                photo = ImageTk.PhotoImage(im)
        except Exception:
            label.configure(image="", text="(failed to load image)", anchor="center")
            return
        label.configure(image=photo, text="")
        # Store reference
        if label is img_lbl:
            img_refs["img"] = photo
        else:
            img_refs["overlay"] = photo

    def _summary_text(summary: Dict[str, Any]) -> str:
        row: QueryRow = summary["row"]
        paths: Dict[str, Path] = summary["paths"]
        lines: List[str] = []
        lines.append(f"pdf_page: {row.pdf_page}")
        lines.append(f"image: {paths['img']}")
        lines.append(f"overlay present: {paths['img_overlay'].exists()}")
        lines.append(f"pdfium_text file: {paths['pdfium_text']} (exists={paths['pdfium_text'].exists()})")
        lines.append("")

        s2 = summary["stage2"]
        s3 = summary["stage3"]
        s4 = summary["stage4"]
        s5 = summary["stage5"]
        lines.append("Model / stage summary:")
        lines.append(f"- stage2 page_elements_v3: present={s2.get('present')} dets={s2.get('num_detections', 0)} timing_s={s2.get('timing_s')}")
        if s2.get("by_label"):
            lines.append("  label counts: " + ", ".join([f"{k}={v}" for k, v in list(s2["by_label"].items())[:12]]))
        lines.append(f"- stage3 graphic_elements_v1: present={s3.get('present')} regions={s3.get('num_regions', 0)} dets={s3.get('num_detections', 0)} timing_s={s3.get('timing_s')}")
        lines.append(f"- stage4 table_structure_v1: present={s4.get('present')} regions={s4.get('num_regions', 0)} dets={s4.get('num_detections', 0)} timing_s={s4.get('timing_s')}")
        lines.append(f"- stage5 nemotron_ocr_v1: present={s5.get('present')} regions={s5.get('num_regions', 0)} nonempty={s5.get('num_nonempty', 0)}")
        if s5.get("by_label_name"):
            lines.append("  region kinds: " + ", ".join([f"{k}={v}" for k, v in list(s5["by_label_name"].items())[:12]]))
        if s5.get("sample_texts"):
            lines.append("")
            lines.append("OCR sample texts:")
            for t in s5["sample_texts"]:
                lines.append(f"- {t}")
        return "\n".join(lines).strip() + "\n"

    def _get_summary_for_index(idx: int) -> Optional[Dict[str, Any]]:
        if idx in summary_cache:
            return summary_cache[idx]
        ex = examples[idx]
        if ex.image_path is None or (not ex.image_path.exists()):
            return None
        summary = _format_summary_for_page(ex.image_path, ex.row)
        summary_cache[idx] = summary
        return summary

    # Filtering state: listbox shows only indices in `visible`.
    visible: List[int] = list(range(len(examples)))

    def _apply_filter() -> None:
        nonlocal visible
        q = (search_var.get() or "").strip().lower()
        if not q:
            visible = list(range(len(examples)))
        else:
            out: List[int] = []
            for i, ex in enumerate(examples):
                hay = f"{ex.row.pdf_page} {ex.row.query}".lower()
                if q in hay:
                    out.append(i)
            visible = out

        lb.delete(0, "end")
        for i in visible:
            lb.insert("end", list_labels[i])
        if visible:
            lb.selection_clear(0, "end")
            lb.selection_set(0)
            lb.event_generate("<<ListboxSelect>>")

    def _on_select(_evt=None) -> None:
        sel = lb.curselection()
        if not sel:
            return
        vis_pos = int(sel[0])
        idx = visible[vis_pos]
        ex = examples[idx]
        row = ex.row
        query_lbl.configure(text=f"Query: {row.query}")
        if ex.image_path is None:
            path_lbl.configure(text=f"{row.pdf_page}  |  (image not found in input_dir)")
            _render_image_to_label(img_lbl, None)
            _render_image_to_label(overlay_lbl, None)
            _set_text(pdfium_text, "")
            _set_text(det_text, "")
            _set_text(embedder_input_text, "")
            _set_text(raw_text, "")
            return

        paths = _paths_for_image(ex.image_path)
        path_lbl.configure(text=f"{row.pdf_page}  |  {ex.image_path}")

        _render_image_to_label(img_lbl, paths["img"])
        _render_image_to_label(overlay_lbl, paths["img_overlay"] if paths["img_overlay"].exists() else None)

        s = _get_summary_for_index(idx)
        if s is None:
            _set_text(pdfium_text, "")
            _set_text(det_text, "")
            _set_text(embedder_input_text, "")
            _set_text(raw_text, "")
            return
        _set_text(pdfium_text, s.get("pdfium_text") or "")
        _set_text(det_text, _summary_text(s))
        embed_path = paths.get("embedder_input")
        if isinstance(embed_path, Path) and embed_path.exists():
            _set_text(embedder_input_text, _read_text_best_effort(embed_path) + "\n")
        else:
            _set_text(embedder_input_text, "")
        raw = s.get("raw") or {}
        raw_blob = "\n\n".join(
            [
                f"## stage2: {paths['stage2']} (exists={paths['stage2'].exists()})\n{_pretty_json(raw.get('stage2'))}",
                f"## stage3: {paths['stage3']} (exists={paths['stage3'].exists()})\n{_pretty_json(raw.get('stage3'))}",
                f"## stage4: {paths['stage4']} (exists={paths['stage4'].exists()})\n{_pretty_json(raw.get('stage4'))}",
                f"## stage5: {paths['stage5']} (exists={paths['stage5'].exists()})\n{_pretty_json(raw.get('stage5'))}",
            ]
        ).strip()
        _set_text(raw_text, raw_blob + ("\n" if raw_blob else ""))

    def _export_current_to_pdf() -> None:
        # Minimal UX: export the currently selected page only to a PDF adjacent to input_dir.
        sel = lb.curselection()
        if not sel:
            return
        idx = visible[int(sel[0])]
        ex = examples[idx]
        if ex.image_path is None:
            return
        out = ex.image_path.parent / f"{ex.row.pdf_page}.post_mortem.pdf"
        _export_report_pdf(examples=[ex], output_pdf=out, recursive=recursive)
        console.print(f"[green]Exported[/green] {out}")

    btns = ttk.Frame(left)
    btns.grid(row=4, column=0, sticky="ew", pady=(10, 0))
    ttk.Button(btns, text="Export selected PDF", command=_export_current_to_pdf).grid(row=0, column=0, sticky="ew")

    search_var.trace_add("write", lambda *_: _apply_filter())
    lb.bind("<<ListboxSelect>>", _on_select)

    _apply_filter()
    search_entry.focus_set()
    root.mainloop()


@app.command()
def run(
    input_dir: Path = typer.Option(..., "--input-dir", exists=True, file_okay=False, dir_okay=True),
    csv_path: Path = typer.Option(
        Path("./bo767_query_gt.csv"),
        "--csv-path",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="CSV containing query -> pdf_page mappings (expects columns like query,pdf,page,pdf_page).",
    ),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan subdirectories for images."),
    export_pdf: Optional[Path] = typer.Option(
        None,
        "--export-pdf",
        help="If set, exports a multi-page PDF report (one PDF page per unique resolved page) and exits unless --also-ui is set.",
    ),
    export_limit_unique_pages: Optional[int] = typer.Option(
        None,
        "--export-limit-unique-pages",
        min=1,
        help="Optional limit on number of unique pages included in the exported report.",
    ),
    gather_results: Optional[Path] = typer.Option(
        None,
        "--gather-results",
        help="If set, writes a .zip containing all artifacts used by the UI (images, sidecars, text, overlays, embedder-input) plus a manifest.",
    ),
    gather_limit_unique_pages: Optional[int] = typer.Option(
        None,
        "--gather-limit-unique-pages",
        min=1,
        help="Optional limit on number of unique pages included in the gathered .zip.",
    ),
    also_ui: bool = typer.Option(False, "--also-ui", help="If --export-pdf is set, still open the UI after exporting."),
):
    """
    Default behavior: open an interactive UI viewer over pages referenced in the CSV.

    If --export-pdf is provided, creates a shareable report PDF containing the same information.
    """
    input_dir = Path(input_dir)
    csv_path = Path(csv_path)

    rows = _load_csv_rows(csv_path)
    if not rows:
        raise typer.BadParameter(f"No rows parsed from csv_path={csv_path}")

    images = _iter_images(input_dir, recursive=recursive)
    if not images:
        console.print(f"[yellow]warning[/yellow] No images found under input_dir={input_dir} (recursive={recursive})")
    by_name, by_stem = _index_images(images)

    resolved: List[ResolvedExample] = []
    for row in rows:
        img = _resolve_image_for_row(row, by_stem=by_stem, by_name=by_name)
        resolved.append(ResolvedExample(row=row, image_path=img))

    global_metrics = _compute_global_metrics(resolved)
    console.print(
        f"[bold cyan]Stage999[/bold cyan] rows={len(rows)} unique_pages={global_metrics.get('unique_pages')} "
        f"missing_images={global_metrics.get('missing_images')} input_dir={input_dir} recursive={recursive}"
    )

    if export_pdf is not None:
        out = Path(export_pdf)
        console.print(f"[cyan]Exporting report[/cyan] output={out}")
        _export_report_pdf(
            examples=resolved,
            output_pdf=out,
            recursive=recursive,
            limit_unique_pages=export_limit_unique_pages,
        )
        console.print(f"[green]Export complete[/green] output={out}")
        if not also_ui:
            return

    if gather_results is not None:
        out_zip = Path(gather_results)
        if out_zip.suffix.lower() != ".zip":
            out_zip = out_zip.with_suffix(out_zip.suffix + ".zip") if out_zip.suffix else out_zip.with_suffix(".zip")
        console.print(f"[cyan]Gathering results[/cyan] output={out_zip}")
        _gather_results_zip(
            examples=resolved,
            input_dir=input_dir,
            csv_path=csv_path,
            zip_path=out_zip,
            recursive=recursive,
            limit_unique_pages=gather_limit_unique_pages,
        )
        console.print(f"[green]Gather complete[/green] output={out_zip}")

    _run_ui(examples=resolved, global_metrics=global_metrics)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

