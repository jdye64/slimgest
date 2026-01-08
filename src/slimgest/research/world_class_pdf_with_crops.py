# world_class_pdf_with_crops.py
import os
from pathlib import Path
import time
from typing import Dict, Optional, List, Tuple, Generator

import torch
from nemotron_page_elements_v3.model import define_model
from nemotron_ocr.inference.pipeline import NemotronOCR
from nemotron_table_structure_v1.model import define_model as define_model_table_structure
from nemotron_graphic_elements_v1.model import define_model as define_model_graphic_elements

import torch
import torch.nn.functional as F
import pypdfium2 as pdfium
import typer

class DocumentEngine:

    def __init__(
        self,
        device: str = "cuda",
        target_det_size: Tuple[int, int] = (1024, 1024),
        target_crop_size: Tuple[int, int] = (224, 224),
        pdf_render_dpi: int = 300,
        verbose: bool = True,
        preload_models: bool = True,
    ):
        self.device = device
        self.target_det_size = target_det_size
        self.target_crop_size = target_crop_size
        self.pdf_render_dpi = pdf_render_dpi
        self.verbose = verbose

        if preload_models:
            self._load_models()

    def _load_models(self):
        self.page_elements_model = define_model("page_element_v3")
        self.table_structure_model = define_model_table_structure("table_structure_v1")
        self.graphic_elements_model = define_model_graphic_elements("graphic_elements_v1")
        self.ocr_model = NemotronOCR(model_dir="/home/jdyer/Development/slim-gest/models/nemotron-ocr-v1/checkpoints")

    def _device_resize_pad_image(self, img: torch.Tensor, size: tuple) -> torch.Tensor:
        """
        Resizes and pads an image to a given size.
        The goal is to preserve the aspect ratio of the image.

        This function operates on the device of the input `img` tensor, which can be either CPU or GPU.
        All operations (F.interpolate, F.pad) will be performed on the same device as `img`.

        Args:
            img (torch.Tensor[C x H x W]): The image to resize and pad.
            size (tuple[2]): The size to resize and pad the image to.

        Returns:
            torch.Tensor: The resized and padded image.
        """
        img = img.float()
        _, h, w = img.shape
        scale = min(size[0] / h, size[1] / w)
        nh = int(h * scale)
        nw = int(w * scale)
        img = F.interpolate(
            img.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False
        ).squeeze(0)
        img = torch.clamp(img, 0, 255)
        pad_b = size[0] - nh
        pad_r = size[1] - nw
        img = F.pad(img, (0, pad_r, 0, pad_b), value=114.0)
        return img

    def _crop_tensor_on_gpu(self, page_image: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """
        Crop a tensor on the GPU.
        
        Args:
            page_image: Tensor of shape (C, H, W) or (1, C, H, W)
            box: Box tensor of shape (4,) with coordinates in format [x1, y1, x2, y2]
                 Can be either normalized (0-1) or pixel coordinates
        """
        # Get image dimensions
        if page_image.ndim == 4:
            _, _, H, W = page_image.shape
            page_image = page_image.squeeze(0)  # Remove batch dimension for cropping
        else:
            _, H, W = page_image.shape
        
        # Check if box is normalized (values between 0 and 1)
        if box.max() <= 1.0:
            # Convert normalized coordinates to pixel coordinates
            x1 = int((box[0] * W).item())
            y1 = int((box[1] * H).item())
            x2 = int((box[2] * W).item())
            y2 = int((box[3] * H).item())
        else:
            # Already in pixel coordinates
            x1 = int(box[0].item())
            y1 = int(box[1].item())
            x2 = int(box[2].item())
            y2 = int(box[3].item())
        
        # Clamp coordinates to valid range
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        
        # Ensure valid crop (x2 > x1 and y2 > y1)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid crop coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        
        # Crop using integer indices
        cropped_tensor_view = page_image[:, y1:y2, x1:x2]
        return cropped_tensor_view

    def load_tensors_from_pdf(self, pdf_path: str) -> Generator[Tuple[torch.Tensor, Tuple[int, int]], None, None]:
        """
        Yields tensors from PDF pages along with their original bitmap dimensions.
        
        Yields:
            Tuple[torch.Tensor, Tuple[int, int]]: 
                - Tensor of shape (C, H, W) on the specified device
                - Original bitmap shape (height, width) before conversion to tensor
        """
        pdf = pdfium.PdfDocument(pdf_path)
        num_pages = len(pdf)
        try:
            for page_index in range(num_pages):
                # --- PDF Page to Torch Tensor Conversion ---
                # 1. Retrieve the page object from the PDF at the specified index.
                #    This uses pypdfium2 which allows high-quality, fast rasterization.
                page = pdf.get_page(page_index)

                # 2. Compute the scale factor so the rendered bitmap matches the requested DPI (default: 300).
                #    PDF "point" units are at 72 DPI by default.
                scale = self.pdf_render_dpi / 72.0

                # 3. Render the PDF page to a bitmap with the desired scale, no rotation, and as an RGB image (not grayscale).
                #    This turns the page into a numpy-backed pixel buffer.
                bitmap = page.render(scale=scale, rotation=0, grayscale=False)

                # 4. Convert the rendered bitmap to a numpy array.
                #    Resulting shape: (height, width, 3) for RGB or (height, width, 4) for RGBA.
                arr = bitmap.to_numpy()  # H x W x (3 or 4)

                # 5. Close the PDFium page to free resources as soon as the pixel buffer is extracted.
                page.close()

                # 6. Store the original bitmap shape (height, width) before any modifications.
                orig_shape = (arr.shape[0], arr.shape[1])

                # 7. If the array includes an alpha channel (RGBA), discard it and keep only RGB data.
                if arr.shape[-1] == 4:
                    arr = arr[..., :3]

                # 8. Convert the numpy image to a torch tensor and reorder it to (channels, height, width) format.
                #    The .contiguous() ensures tensor memory is packed, which is preferred for GPU transfer.
                t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

                # 9. Move the tensor to the designated device (GPU or CPU).
                #    The non_blocking=True flag can speed up transfer if pinning is enabled and device is CUDA.
                t_device = t.to(self.device, non_blocking=True)

                # NOTE: This code streamlines PDF page rasterization with on-the-fly transfer to a torch tensor,
                #       enabling efficient page-wise processing for tasks like vision models or optical character recognition.
                yield t_device, orig_shape
        finally:
            pdf.close()

    def page_element_detections(self, page_image: torch.Tensor, orig_shape: Tuple[int, int]) -> List[Dict[str, torch.Tensor]]:
        """
        Run page element detection on a page tensor.
        
        Args:
            page_image: Tensor of shape (C, H, W) or (1, C, H, W)
            orig_shape: Original bitmap shape (height, width) before any preprocessing
            
        Returns:
            List of detection dictionaries with 'boxes', 'labels', 'scores' keys
        """

        if not isinstance(page_image, torch.Tensor):
            page_image = torch.from_numpy(page_image)
            page_image = page_image.to(self.device, non_blocking=True)
        else:
            # Check if the tensor is already on the correct device, move if not
            if page_image.device != torch.device(self.device):
                page_image = page_image.to(self.device, non_blocking=True)

        # Resize and pad page_image to the target size (self.img_size) for the model.
        # The resize_pad function should resize the page_image so that its longest edge matches
        # the target dimension while maintaining aspect ratio, and then pad the remaining space.
        page_image = self._device_resize_pad_image(page_image, self.target_det_size)

        # Convert page_image data type to float, which is the standard type for neural network input tensors.
        # Optionally, this also ensures proper normalization/scaling as required by downstream models.
        page_image = page_image.float()
            
        # Format orig_shape as expected by the model: [[height, width]]
        orig_size = [list(orig_shape)]
        
        with torch.no_grad():
            det_out = self.page_elements_model(page_image, orig_size)
        return det_out

    # def table_structure_detection(self, table_image: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
    #     with torch.no_grad():
    #         det_out = self.table_structure_model(table_image, orig_size)
    #     return det_out

    # def graphic_elements_detection(self, graphic_image: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
    #     with torch.no_grad():
    #         det_out = self.graphic_elements_model(graphic_image, orig_size)
    #     return det_out

    def perform_ocr(self, page_image: torch.Tensor, merge_level: str = "paragraph") -> List[Dict[str, torch.Tensor]]:
        """
        Perform OCR on a page tensor.
        
        Args:
            page_image: Tensor of shape (C, H, W) or (1, C, H, W)
            merge_level: The granularity of text merging ('word', 'sentence', 'paragraph')
        
        Returns:
            OCR results from the NemotronOCR model
        """
        return self.ocr_model(page_image, merge_level=merge_level)

app = typer.Typer(help="Process PDFs locally using shared pipeline")

@app.command()
def run(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False),
):
    engine = DocumentEngine(device="cuda:0", pdf_render_dpi=300, preload_models=True)


    # TODO: Refactor page_elements_model.labels to be a class variable and expose this so as no need to redeclare this list here.
    page_element_labels: List[str] = [
        "table",
        "chart",
        "title",
        "infographic",
        "text",
        "header_footer",
    ]

    pdf_files = [
        str(f) for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".pdf"
    ]

    for pdf_path in pdf_files:
        for page_tensor, orig_shape in engine.load_tensors_from_pdf(pdf_path):
            print(f"Page shape: {orig_shape}")
            page_element_detections = engine.page_element_detections(page_tensor, orig_shape)

            combined_text_results = ""

            # Iterate over the detections and extract labels
            for detection in page_element_detections:
                labels = detection.get("labels", [])

                for i in range(len(labels)):
                    label = labels[i]

                    if label == 0:
                        print(f"Table detection")
                    elif label == 1:
                        print(f"Chart detection")
                    elif label == 2:
                        print(f"Title detection")
                    elif label == 3:
                        print(f"Infographic detection")
                    elif label == 4:
                        print(f"Text detection")
                        cropped = engine._crop_tensor_on_gpu(page_tensor, detection.get('boxes', [])[i])
                        ocr_results = engine.perform_ocr(cropped, merge_level="paragraph")
                        # Iterate through the OCR results, concatenate the 'text' field for each result,
                        # and append to the combined_text_results string.
                        if isinstance(ocr_results, list):
                            text_content = "".join([res.get("text", "") for res in ocr_results if "text" in res])
                            combined_text_results += text_content
                        elif isinstance(ocr_results, dict) and "text" in ocr_results:
                            combined_text_results += ocr_results["text"]
                    elif label == 5:
                        print(f"Header/footer detection")

            print(f"Combined text results: {combined_text_results}")

