from typing import Any, Dict, List, Tuple

import torch
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
        self._ocr_input_shape = (1024, 1024)

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess the input tensor."""
        # no-op for now
        return tensor
    
    def invoke(self, input_data: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        # Conditionally check and make sure the input data is on the correct device and shape
        results = self._model(input_data)
        print(f"Type of results: {type(results)}")
        print(f"Length of results: {len(results)}")
        if isinstance(results, list) and len(results) > 0:
            return results[0]
        else:
            return []

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
