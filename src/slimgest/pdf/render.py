"""
PDF page rendering utilities.

Provides generator functions for efficiently rendering PDF pages to bitmaps and tensors.
Each page is yielded as it's processed, allowing for streaming/pipeline processing without
loading all pages into memory at once.
"""

from pathlib import Path
from typing import Generator, Tuple, Optional, Union
from dataclasses import dataclass
import pypdfium2 as pdfium
import numpy as np
import torch


@dataclass
class PageBitmap:
    """Metadata and bitmap for a single PDF page."""
    page_number: int  # 0-indexed
    bitmap: pdfium.PdfBitmap
    width: int
    height: int
    
    def to_numpy(self, rgb_only: bool = True) -> np.ndarray:
        """
        Convert the bitmap to a numpy array.
        
        Args:
            rgb_only: If True, returns only RGB channels (drops alpha if present).
                     If False, returns all channels as-is.
        
        Returns:
            numpy array of shape [H, W, 3] or [H, W, 4]
        """
        arr = self.bitmap.to_numpy()
        if rgb_only and arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr


@dataclass
class PageBitmapWithText(PageBitmap):
    """Rendered bitmap plus raw PDF-extracted text for a single page."""
    text: str


@dataclass
class PageTensor:
    """Metadata and tensor for a single PDF page."""
    page_number: int  # 0-indexed
    tensor: torch.Tensor  # Shape [C, H, W]
    original_width: int
    original_height: int
    device: torch.device


def iter_pdf_page_bitmaps(
    pdf_path: Union[str, Path],
    dpi: float = 150.0,
    rotation: int = 0,
    grayscale: bool = False,
) -> Generator[PageBitmapWithText, None, None]:
    """
    Generator that yields rendered bitmaps for each page in a PDF.
    
    This is a memory-efficient way to process PDF pages one at a time without
    loading all pages into memory at once.
    
    Args:
        pdf_path: Path to the PDF file.
        dpi: DPI for rendering (default 150). Standard PDF is 72 DPI,
             so scale = dpi/72.0
        rotation: Rotation angle in degrees (0, 90, 180, 270).
        grayscale: If True, render in grayscale instead of RGB.
    
    Yields:
        PageBitmapWithText objects containing the rendered bitmap, metadata, and extracted text.
    
    Example:
        >>> for page_bitmap in iter_pdf_page_bitmaps("document.pdf", dpi=150):
        ...     arr = page_bitmap.to_numpy()
        ...     print(f"Page {page_bitmap.page_number}: {arr.shape}, text_len={len(page_bitmap.text)}")
    """
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        num_pages = len(pdf)
        scale = dpi / 72.0
        
        for page_idx in range(num_pages):
            page = pdf.get_page(page_idx)
            try:
                # Extract raw embedded text (not OCR) using standard PDFium APIs
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

                bitmap = page.render(
                    scale=scale,
                    rotation=rotation,
                    grayscale=grayscale
                )
                
                yield PageBitmapWithText(
                    page_number=page_idx,
                    bitmap=bitmap,
                    width=bitmap.width,
                    height=bitmap.height,
                    text=page_text,
                )
            finally:
                page.close()
    finally:
        pdf.close()


def iter_pdf_page_tensors(
    pdf_path: Union[str, Path],
    dpi: float = 150.0,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.uint8,
    rotation: int = 0,
    grayscale: bool = False,
    non_blocking: bool = True,
) -> Generator[PageTensor, None, None]:
    """
    Generator that yields PyTorch tensors for each page in a PDF.
    
    This is a memory-efficient way to process PDF pages, yielding one tensor
    at a time. Each tensor is in CHW format (channels first).
    
    Args:
        pdf_path: Path to the PDF file.
        dpi: DPI for rendering (default 150).
        device: Target device for tensors ("cpu", "cuda", or torch.device object).
        dtype: Data type for tensors (default torch.uint8).
        rotation: Rotation angle in degrees (0, 90, 180, 270).
        grayscale: If True, render in grayscale (1 channel) instead of RGB (3 channels).
        non_blocking: If True and copying to CUDA, use non-blocking transfer.
    
    Yields:
        PageTensor objects containing the tensor and metadata.
    
    Example:
        >>> for page_tensor in iter_pdf_page_tensors("document.pdf", device="cuda"):
        ...     tensor = page_tensor.tensor  # Shape: [3, H, W]
        ...     print(f"Page {page_tensor.page_number}: {tensor.shape}, device={tensor.device}")
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    for page_bitmap in iter_pdf_page_bitmaps(pdf_path, dpi, rotation, grayscale):
        # Convert bitmap to numpy array (RGB only)
        arr = page_bitmap.to_numpy(rgb_only=True)
        
        # Convert to tensor: [H, W, C] -> [C, H, W]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        
        # Move to target device
        tensor = tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)
        
        yield PageTensor(
            page_number=page_bitmap.page_number,
            tensor=tensor,
            original_width=page_bitmap.width,
            original_height=page_bitmap.height,
            device=device,
        )


def load_pdf_page_tensors(
    pdf_path: Union[str, Path],
    dpi: float = 150.0,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.uint8,
    rotation: int = 0,
    grayscale: bool = False,
    non_blocking: bool = True,
) -> Tuple[list[torch.Tensor], list[Tuple[int, int]]]:
    """
    Load all pages from a PDF as PyTorch tensors (non-streaming version).
    
    This is a convenience function that loads all pages at once.
    For large PDFs or memory-constrained scenarios, prefer iter_pdf_page_tensors().
    
    Args:
        pdf_path: Path to the PDF file.
        dpi: DPI for rendering (default 150).
        device: Target device for tensors.
        dtype: Data type for tensors (default torch.uint8).
        rotation: Rotation angle in degrees.
        grayscale: If True, render in grayscale.
        non_blocking: If True and copying to CUDA, use non-blocking transfer.
    
    Returns:
        Tuple of (tensors, shapes) where:
            - tensors: List of tensors, each of shape [C, H, W]
            - shapes: List of (height, width) tuples for original dimensions
    """
    tensors = []
    shapes = []
    
    for page_tensor in iter_pdf_page_tensors(
        pdf_path, dpi, device, dtype, rotation, grayscale, non_blocking
    ):
        tensors.append(page_tensor.tensor)
        shapes.append((page_tensor.original_height, page_tensor.original_width))
    
    return tensors, shapes
