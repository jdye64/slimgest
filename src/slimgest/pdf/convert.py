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
    python pdf_utils.py convert ./input_pdfs ./output_images --target-size 1024 1024

The CLI is also intended for use with the slimgest-local CLI via subcommands.
"""

import os
import typer
import pypdfium2
from PIL import Image
from tqdm import tqdm
from typing import Optional, Tuple

app = typer.Typer(help="Convert all PDF files in a directory to images (one image per page).")

def save_pdf_pages_as_images(
    pdf_path: str, 
    out_dir: str, 
    target_size: Tuple[int, int], 
    pages_pbar=None
):
    """
    Convert all pages in a single PDF to images and save as JPEGs.

    Args:
        pdf_path (str): Path to the PDF file to process.
        out_dir (str): Directory to write output images.
        target_size (Tuple[int, int]): Size (width, height) for output images.
        pages_pbar (tqdm.tqdm | None): Optional TQDM progress bar for pages.

    Images are saved as <pdf_basename>_page<N>.jpg in the output directory.
    Pages are resized (with aspect preserved), centered on a white background of size target_size.
    Skips pages if corresponding output image already exists.
    """
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf = pypdfium2.PdfDocument(pdf_path)
    n_pages = len(pdf)
    for page_num in range(n_pages):
        out_name = f"{pdf_basename}_page{page_num+1}.jpg"
        out_path = os.path.join(out_dir, out_name)
        if os.path.exists(out_path):
            if pages_pbar is not None:
                pages_pbar.update(1)
            continue  # Skip if file already exists for this page
        page = pdf.get_page(page_num)
        w, h = page.get_size()
        # Calculate zoom to fit the larger dimension in target_size, preserving aspect ratio
        zoom = target_size[0] / w if w < h else target_size[1] / h
        pil_image = page.render(scale=zoom).to_pil()
        pil_image = pil_image.convert("RGB")
        pil_image.thumbnail(target_size, Image.LANCZOS)
        background = Image.new("RGB", target_size, (255, 255, 255))
        x_offset = (target_size[0] - pil_image.width) // 2
        y_offset = (target_size[1] - pil_image.height) // 2
        background.paste(pil_image, (x_offset, y_offset))
        background.save(out_path, "JPEG", quality=95)
        page.close()
        if pages_pbar is not None:
            pages_pbar.update(1)
    pdf.close()

@app.command()
def convert(
    pdf_dir: str = typer.Argument(..., help="Input directory containing PDF files."),
    out_dir: str = typer.Argument(..., help="Output directory for saved images."),
    target_size: Tuple[int, int] = typer.Option((1024, 1024), "--target-size", "-s", help="Size of output images, e.g. --target-size 1024 1024 [width height]."),
):
    """
    Convert all PDF files in a directory to images.

    This function finds all PDF files in the input directory, then for each PDF,
    renders each page as an image using save_pdf_pages_as_images() and saves them to the output directory.
    Progress is shown for both PDFs and total pages.

    Args:
        pdf_dir (str): Input directory containing PDF files.
        out_dir (str): Output directory for saved images.
        target_size (Tuple[int, int]): Output image size as (width, height).
    """
    os.makedirs(out_dir, exist_ok=True)

    print("Converting PDFs to images...")
    # Gather all PDF filenames
    pdf_filenames = [fname for fname in os.listdir(pdf_dir) if fname.lower().endswith(".pdf")]

    # Count total number of pages for all PDFs, for progress bar
    total_pages = 0
    pdf_page_counts = []
    for fname in pdf_filenames:
        pdf_path = os.path.join(pdf_dir, fname)
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
        for fname, n_pages in zip(pdf_filenames, pdf_page_counts):
            pdf_path = os.path.join(pdf_dir, fname)
            pdf_pbar.write(f"Converting {fname}... ({n_pages} pages)")
            try:
                save_pdf_pages_as_images(pdf_path, out_dir, target_size, pages_pbar=pages_pbar)
            except Exception as e:
                pdf_pbar.write(f"Error processing {fname}: {e}")
            pdf_pbar.update(1)

if __name__ == "__main__":
    app()
