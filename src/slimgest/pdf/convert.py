"""
pdf_utils.py

Utility for converting PDF pages to images, CLI included.

This script provides functions and a Typer CLI command to convert all PDF files in a given directory
into JPEG images. Each PDF page is rendered as a separate image, and all output images are written to a specified directory.

Dependencies:
    - pypdfium2: for fast and accurate PDF rendering
    - Pillow (PIL): for image processing and saving
    - tqdm: for progress bars
    - typer: for CLI interface

Example usage:
    python -m slimgest.pdf.convert convert ./input_pdfs ./output_images --dpi 300 --image-format png --target-size 1024 1024

The CLI is also intended for use with the slimgest-local CLI via subcommands.
"""

import os
import concurrent.futures
import typer
import pypdfium2
from PIL import Image
from tqdm import tqdm
from typing import Any, Dict, Literal, Optional, Tuple

app = typer.Typer(help="Convert all PDF files in a directory to images (one image per page).")

def save_pdf_pages_as_images(
    pdf_path: str, 
    out_dir: str, 
    *,
    dpi: float,
    image_format: Literal["png", "jpeg"],
    target_size: Tuple[int, int],
    overwrite: bool = False,
    pages_pbar=None,
):
    """
    Convert all pages in a single PDF to images and save as PNG/JPEG.

    Args:
        pdf_path (str): Path to the PDF file to process.
        out_dir (str): Directory to write output images.
        dpi (float): Render DPI (default from CLI is 300).
        image_format (str): "png" or "jpeg".
        target_size (Tuple[int, int]): Size (width, height) for output images.
        overwrite (bool): If True, overwrite existing page images.
        pages_pbar (tqdm.tqdm | None): Optional TQDM progress bar for pages.

    Images are saved as <pdf_basename>_page<NNNN>.<ext> in the output directory.
    Embedded PDF text (not OCR) is saved as <pdf_basename>_page<NNNN>.pdfium_text.txt.
    Pages are resized (with aspect preserved), centered on a white background of size target_size.
    Skips pages if corresponding output image already exists unless overwrite=True.
    """
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf = pypdfium2.PdfDocument(pdf_path)
    n_pages = len(pdf)
    written = 0
    skipped = 0
    text_written = 0
    for page_num in range(n_pages):
        out_ext = "png" if image_format == "png" else "jpg"
        out_name = f"{pdf_basename}_page{page_num+1:04d}.{out_ext}"
        out_path = os.path.join(out_dir, out_name)
        text_name = f"{pdf_basename}_page{page_num+1:04d}.pdfium_text.txt"
        text_path = os.path.join(out_dir, text_name)

        have_img = os.path.exists(out_path)
        have_txt = os.path.exists(text_path)
        if (not overwrite) and have_img and have_txt:
            skipped += 1
            if pages_pbar is not None:
                pages_pbar.update(1)
            continue  # Skip if file already exists for this page
        page = pdf.get_page(page_num)
        try:
            # Extract embedded text (not OCR) using PDFium APIs.
            if overwrite or (not have_txt):
                page_text = ""
                try:
                    textpage = page.get_textpage()
                    try:
                        page_text = textpage.get_text_range() or ""
                    finally:
                        close_fn = getattr(textpage, "close", None)
                        if callable(close_fn):
                            close_fn()
                except Exception:
                    page_text = ""

                # Always write a file (even if empty) so downstream stages can rely on it.
                with open(text_path, "w", encoding="utf-8", errors="replace") as f:
                    f.write(page_text)
                text_written += 1

            # PDF "native" DPI is 72. PDFium expects a scale factor.
            if overwrite or (not have_img):
                scale = float(dpi) / 72.0
                pil_image = page.render(scale=scale).to_pil().convert("RGB")

                # Resize to fit within target_size while preserving aspect ratio, then pad to exact target_size
                pil_image.thumbnail(target_size, Image.LANCZOS)
                background = Image.new("RGB", target_size, (255, 255, 255))
                x_offset = (target_size[0] - pil_image.width) // 2
                y_offset = (target_size[1] - pil_image.height) // 2
                background.paste(pil_image, (x_offset, y_offset))

                if image_format == "png":
                    background.save(out_path, "PNG", optimize=True)
                else:
                    background.save(out_path, "JPEG", quality=95, optimize=True)
                written += 1
        finally:
            page.close()
        if pages_pbar is not None:
            pages_pbar.update(1)
    pdf.close()
    return {
        "n_pages": int(n_pages),
        "written": int(written),
        "text_written": int(text_written),
        "skipped": int(skipped),
    }


def _process_one_pdf(args: Tuple[str, str, float, str, Tuple[int, int], bool]) -> Dict[str, Any]:
    """
    Worker entrypoint for multiprocessing / process pools.
    """
    pdf_path, out_dir, dpi, image_format, target_size, overwrite = args
    try:
        res = save_pdf_pages_as_images(
            pdf_path,
            out_dir,
            dpi=float(dpi),
            image_format=image_format,  # type: ignore[arg-type]
            target_size=target_size,
            overwrite=bool(overwrite),
            pages_pbar=None,
        )
        return {"pdf_path": pdf_path, "ok": True, **(res or {})}
    except Exception as e:
        return {"pdf_path": pdf_path, "ok": False, "error": str(e)}

@app.command()
def convert(
    pdf_dir: str = typer.Argument(..., help="Input directory containing PDF files."),
    out_dir: str = typer.Argument(..., help="Output directory for saved images."),
    dpi: float = typer.Option(300.0, "--dpi", min=1.0, help="Render DPI (e.g. 300)."),
    image_format: Literal["png", "jpeg"] = typer.Option(
        "png", "--image-format", help="Output image format: png or jpeg."
    ),
    target_size: Tuple[int, int] = typer.Option((1024, 1024), "--target-size", "-s", help="Size of output images, e.g. --target-size 1024 1024 [width height]."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing page images."),
    num_processes: int = typer.Option(
        8,
        "--num-processes",
        "-p",
        min=1,
        help="Number of worker processes. Each worker processes one PDF at a time.",
    ),
):
    """
    Convert all PDF files in a directory to images.

    This function finds all PDF files in the input directory, then for each PDF,
    renders each page as an image using save_pdf_pages_as_images() and saves them to the output directory.
    Progress is shown for both PDFs and total pages.

    Args:
        pdf_dir (str): Input directory containing PDF files.
        out_dir (str): Output directory for saved images.
        dpi (float): Render DPI for PDFium.
        image_format (str): Output image format ("png" or "jpeg").
        target_size (Tuple[int, int]): Output image size as (width, height).
        overwrite (bool): Overwrite existing page images.
        num_processes (int): Number of worker processes.
    """
    os.makedirs(out_dir, exist_ok=True)

    print("Converting PDFs to images...")
    # Gather all PDF filenames
    pdf_filenames = sorted([fname for fname in os.listdir(pdf_dir) if fname.lower().endswith(".pdf")])
    pdf_paths = [os.path.join(pdf_dir, fname) for fname in pdf_filenames]

    # Count total number of pages for all PDFs, for progress bar
    total_pages = 0
    pdf_page_counts = []
    for fname, pdf_path in zip(pdf_filenames, pdf_paths):
        try:
            pdf = pypdfium2.PdfDocument(pdf_path)
            n_pages = len(pdf)
            pdf_page_counts.append(n_pages)
            total_pages += n_pages
            pdf.close()
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            pdf_page_counts.append(0)

    with tqdm(total=len(pdf_filenames), desc="PDFs", unit="pdf") as pdf_pbar, \
         tqdm(total=total_pages, desc="Pages", unit="page") as pages_pbar:
        # Parallelize across PDFs. Each worker converts one PDF at a time, then takes the next.
        # Note: per-page progress isn't updated live from workers; we update Pages by the PDF's page count on completion.
        tasks: list[Tuple[str, str, float, str, Tuple[int, int], bool]] = [
            (pdf_path, out_dir, float(dpi), str(image_format), tuple(target_size), bool(overwrite))
            for pdf_path in pdf_paths
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=int(num_processes)) as ex:
            futures = {ex.submit(_process_one_pdf, t): t[0] for t in tasks}
            for fut in concurrent.futures.as_completed(futures):
                pdf_path = futures[fut]
                fname = os.path.basename(pdf_path)
                # update progress bars on completion
                try:
                    res = fut.result()
                except Exception as e:
                    pdf_pbar.write(f"Error processing {fname}: {e}")
                    res = {"ok": False}

                # advance "Pages" by the known page count (keeps it monotonic even when skipping/overwrite)
                try:
                    idx = pdf_paths.index(pdf_path)
                    pages_pbar.update(int(pdf_page_counts[idx] or 0))
                except Exception:
                    pass

                if not res.get("ok"):
                    pdf_pbar.write(f"Error processing {fname}: {res.get('error', 'unknown error')}")
                else:
                    pdf_pbar.write(
                        f"Completed {fname}: pages={res.get('n_pages', '?')} written={res.get('written', '?')} skipped={res.get('skipped', '?')}"
                    )
                pdf_pbar.update(1)

if __name__ == "__main__":
    app()
