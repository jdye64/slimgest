from typing import Any, Dict, List, Tuple

import torch
import os
from ..model import BaseModel, RunMode

from nemotron_ocr.inference.pipeline import NemotronOCR


class NemotronOCRV1(BaseModel):
    """
    Nemotron OCR v1 model for optical character recognition.
    
    End-to-end OCR model that integrates:
    - Text detector for region localization
    - Text recognizer for transcription
    - Relational model for layout and reading order analysis
    """
    
    def __init__(self, model_dir: str) -> None:
        super().__init__()
        self._model = NemotronOCR(model_dir=model_dir)
        # NemotronOCR is a high-level pipeline (not an nn.Module). We can optionally
        # TensorRT-compile individual submodules (e.g. the detector backbone) but
        # must keep post-processing (NMS, box decoding, etc.) in eager PyTorch/C++.
        self._enable_trt = os.getenv("SLIMGEST_ENABLE_TORCH_TRT", "").strip().lower() in {"1", "true", "yes", "on"}
        if self._enable_trt:
            self._maybe_compile_submodules()

    def _maybe_compile_submodules(self) -> None:
        """
        Best-effort TensorRT compilation of internal nn.Modules.
        Any failure falls back to eager PyTorch without breaking initialization.
        """
        try:
            import torch_tensorrt  # type: ignore
        except Exception:
            return

        # Detector is the safest candidate: input is a BCHW image tensor.
        detector = getattr(self._model, "detector", None)
        if not isinstance(detector, torch.nn.Module):
            return

        # NemotronOCR internally resizes/pads to 1024 and runs B=1 (see upstream FIXME);
        # keep the TRT input shape fixed to avoid accidental batching issues.
        try:
            trt_input = torch_tensorrt.Input((1, 3, 1024, 1024), dtype=torch.float16)
        except TypeError:
            # Older/newer API variants: fall back to named arg.
            trt_input = torch_tensorrt.Input(shape=(1, 3, 1024, 1024), dtype=torch.float16)

        # If any torchvision NMS makes it into a compiled graph elsewhere, forcing
        # that op to run in Torch avoids hard failures.
        compile_kwargs: Dict[str, Any] = {
            "inputs": [trt_input],
            "enabled_precisions": {torch.float16},
        }
        if hasattr(torch_tensorrt, "compile"):
            for k in ("torch_executed_ops", "torch_executed_modules"):
                if k == "torch_executed_ops":
                    compile_kwargs[k] = {"torchvision::nms"}
                elif k == "torch_executed_modules":
                    compile_kwargs[k] = set()
            try:
                self._model.detector = torch_tensorrt.compile(detector, **compile_kwargs)
            except Exception:
                # Leave detector as-is on any failure.
                return


    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess the input tensor."""
        # no-op for now
        return tensor
    
    def invoke(self, input_data: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        # NemotronOCR expects a single image tensor (CHW). If a batch (BCHW) is
        # provided, run per-image to keep behavior correct.
        if isinstance(input_data, torch.Tensor) and input_data.ndim == 4:
            out: List[Dict[str, torch.Tensor]] = []
            for i in range(int(input_data.shape[0])):
                out.extend(self._model(input_data[i]))
            return out

        results = self._model(input_data)
        return results

    def postprocess(self, preds: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes, labels, scores = postprocess_preds_page_element(
            preds, self._model.thresholds_per_class, self._model.labels
        )
        return boxes, labels, scores
    
    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return "Nemotron OCR v1"
    
    @property
    def model_type(self) -> str:
        """Model category/type."""
        return "ocr"
    
    @property
    def model_runmode(self) -> RunMode:
        """Execution mode: local, NIM, or build-endpoint."""
        return "local"
    
    @property
    def input(self) -> Any:
        """
        Input schema for the model.
        
        Returns:
            dict: Schema describing RGB image input with variable dimensions
        """
        return {
            "type": "image",
            "format": "RGB",
            "supported_formats": ["PNG", "JPEG"],
            "data_types": ["float32", "uint8"],
            "dimensions": "variable (H x W)",
            "batch_support": True,
            "value_range": {
                "float32": "[0, 1]",
                "uint8": "[0, 255] (auto-converted)"
            },
            "aggregation_levels": ["word", "sentence", "paragraph"],
            "description": "Document or scene image in RGB format with automatic multi-scale resizing"
        }
    
    @property
    def output(self) -> Any:
        """
        Output schema for the model.
        
        Returns:
            dict: Schema describing OCR output format
        """
        return {
            "type": "ocr_results",
            "format": "structured",
            "structure": {
                "boxes": "List[List[List[float]]] - quadrilateral bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]",
                "texts": "List[str] - recognized text strings",
                "confidences": "List[float] - confidence scores per detection"
            },
            "properties": {
                "reading_order": True,
                "layout_analysis": True,
                "multi_line_support": True,
                "multi_block_support": True
            },
            "description": "Structured OCR results with bounding boxes, recognized text, and confidence scores"
        }
    
    @property
    def input_batch_size(self) -> int:
        """Maximum or default input batch size."""
        return 8
