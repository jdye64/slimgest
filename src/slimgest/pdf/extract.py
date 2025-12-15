import os
from pathlib import Path

import typer

import pikepdf
from PIL import Image

app = typer.Typer(help="Process PDFs locally using shared pipeline")

@app.command()
def run(
    input_pdf: Path = typer.Argument(..., exists=True, file_okay=True),
):
    pdf = pikepdf.Pdf.open(str(input_pdf))

    print("\nPDF Layout Summary\n" + "=" * 80)

    # Document info
    print(f"Title: {pdf.docinfo.get('Title', 'N/A')}")
    print(f"Creator: {pdf.docinfo.get('Creator', 'N/A')}")
    print(f"Producer: {pdf.docinfo.get('Producer', 'N/A')}")
    print(f"Number of pages: {len(pdf.pages)}")
    print("-" * 80)

    output_dir = input_pdf.parent
    base_name = input_pdf.stem

    image_save_counter = 0  # For global image indexing

    for idx, page in enumerate(pdf.pages, start=1):
        # Fetch basic media box info
        mediabox = page.obj.get("/MediaBox", None)
        if mediabox:
            width = float(mediabox[2]) - float(mediabox[0])
            height = float(mediabox[3]) - float(mediabox[1])
        else:
            width, height = "?", "?"

        print(f"Page {idx}: Size: {width} x {height} points")

        # Look for common layout cues in /Resources (/Font, /XObject), /Contents
        resources = page.obj.get("/Resources", None)
        contents = page.obj.get("/Contents", None)

        fonts = None
        images = 0
        
        if resources:
            fonts = resources.get("/Font", None)
            xobj = resources.get("/XObject", None)
        else:
            xobj = None

        # For image extraction
        if xobj:
            try:
                for key in xobj.keys():
                    try:
                        x = xobj[key]
                        subtype = x.get("/Subtype", None)
                        if subtype and subtype == "/Image":
                            images += 1
                            
                            # Use pikepdf's PdfImage class to properly extract and decode images
                            try:
                                # Create a PdfImage object from the stream
                                pdf_image = pikepdf.PdfImage(x)
                                
                                # Construct save path (use PNG for consistency)
                                save_path = output_dir / f"{base_name}_page{idx}_image{image_save_counter}.png"
                                
                                # Extract the image as a PIL Image
                                # This handles all the decoding, color space conversion, etc.
                                pil_image = pdf_image.as_pil_image()
                                
                                # Save as PNG
                                pil_image.save(save_path, "PNG")
                                print(f"    Extracted image saved to {save_path}")
                                image_save_counter += 1
                                
                            except Exception as e:
                                print(f"    Failed to extract or save image on page {idx}: {e}")
                    except Exception:
                        pass
            except Exception:
                pass

        # Estimate number of text blocks/streams by counting /Contents streams
        num_content_streams = 0
        if contents is not None:
            if isinstance(contents, pikepdf.Array):
                num_content_streams = len(contents)
            else:
                num_content_streams = 1

        font_names = []
        if fonts:
            try:
                font_names = [str(k) for k in fonts.keys()]
            except Exception:
                pass

        print(f"  Content Streams: {num_content_streams}")
        print(f"  Fonts: {', '.join(font_names) if font_names else 'None found'}")
        print(f"  Images: {images if images else 'None found'}")
        annots = page.obj.get("/Annots", None)
        if annots is not None:
            try:
                annots_count = len(annots)
            except Exception:
                annots_count = "?"
            print(f"  Annotations: {annots_count}")
        links = 0
        if annots:
            try:
                for annot in annots:
                    subtype = annot.get("/Subtype", None)
                    if subtype == "/Link":
                        links += 1
            except Exception:
                pass
        if links:
            print(f"  Links: {links}")
        print("-" * 80)

