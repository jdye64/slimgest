"""
Tensor operations for image processing.

Provides utilities for cropping, transforming, and manipulating image tensors,
particularly for use with object detection bounding boxes.
"""

from typing import List, Tuple
import torch
import numpy as np


def crop_tensor_with_bbox(
    image_tensor: torch.Tensor,
    bbox: np.ndarray,
    original_shape: Tuple[int, int],
    resized_shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Crop a tensor image using a normalized bounding box.
    
    The bbox is in normalized coordinates (0-1) relative to the original image dimensions.
    The image_tensor has been resized/padded to resized_shape, so we transform the bbox
    coordinates to match the resized tensor space, accounting for any padding.
    
    This function assumes the resize operation used aspect-ratio-preserving scaling with
    centered padding (like resize_pad functions from nemotron models).
    
    Args:
        image_tensor: Resized image tensor of shape [C, H, W] on any device.
        bbox: Normalized bounding box [xmin, ymin, xmax, ymax] in range [0, 1],
              relative to the original image dimensions.
        original_shape: Original image shape (height, width) before resize.
        resized_shape: Target shape (height, width) the image was resized to.
    
    Returns:
        Cropped image tensor [C, cropped_H, cropped_W] on the same device.
    
    Example:
        >>> tensor = torch.randn(3, 1024, 1024).cuda()
        >>> bbox = np.array([0.1, 0.2, 0.5, 0.8])  # normalized coords
        >>> cropped = crop_tensor_with_bbox(tensor, bbox, (800, 600), (1024, 1024))
        >>> print(cropped.shape)  # Will be [3, <h>, <w>] based on bbox
    """
    # Get dimensions
    orig_h, orig_w = original_shape
    input_h, input_w = resized_shape
    
    # Calculate scale and padding (aspect-ratio-preserving resize with center padding)
    scale = min(input_h / orig_h, input_w / orig_w)
    scaled_h = int(orig_h * scale)
    scaled_w = int(orig_w * scale)
    pad_y = (input_h - scaled_h) / 2
    pad_x = (input_w - scaled_w) / 2
    
    # Convert normalized bbox to pixel coordinates in original image space
    boxes_plot = bbox.copy()
    boxes_plot[0] *= orig_w  # xmin
    boxes_plot[2] *= orig_w  # xmax
    boxes_plot[1] *= orig_h  # ymin
    boxes_plot[3] *= orig_h  # ymax
    
    # Scale to resized coordinates and add padding offset
    xmin = int(boxes_plot[0] * scale + pad_x)
    ymin = int(boxes_plot[1] * scale + pad_y)
    xmax = int(boxes_plot[2] * scale + pad_x)
    ymax = int(boxes_plot[3] * scale + pad_y)
    
    # Clamp the bounds to the actual tensor dimensions
    _, H, W = image_tensor.shape
    xmin = max(0, min(xmin, W - 1))
    ymin = max(0, min(ymin, H - 1))
    xmax = max(xmin + 1, min(xmax, W))  # Ensure xmax > xmin
    ymax = max(ymin + 1, min(ymax, H))  # Ensure ymax > ymin
    
    # Crop the tensor
    cropped = image_tensor[:, ymin:ymax, xmin:xmax]
    return cropped


def batch_crop_tensors(
    image_tensor: torch.Tensor,
    bboxes: List[np.ndarray],
    original_shape: Tuple[int, int],
    resized_shape: Tuple[int, int],
    clone: bool = True,
) -> List[torch.Tensor]:
    """
    Crop multiple regions from a single image tensor.
    
    Args:
        image_tensor: Resized image tensor of shape [C, H, W].
        bboxes: List of normalized bounding boxes, each [xmin, ymin, xmax, ymax].
        original_shape: Original image shape (height, width) before resize.
        resized_shape: Target shape (height, width) the image was resized to.
        clone: If True, clone each cropped tensor (useful to avoid memory aliasing).
    
    Returns:
        List of cropped tensors, each of shape [C, cropped_H, cropped_W].
    
    Example:
        >>> tensor = torch.randn(3, 1024, 1024).cuda()
        >>> bboxes = [np.array([0.1, 0.1, 0.5, 0.5]), np.array([0.6, 0.6, 0.9, 0.9])]
        >>> crops = batch_crop_tensors(tensor, bboxes, (800, 600), (1024, 1024))
        >>> print(len(crops))  # 2
    """
    crops = []
    for bbox in bboxes:
        cropped = crop_tensor_with_bbox(image_tensor, bbox, original_shape, resized_shape)
        if clone:
            cropped = cropped.clone()
        crops.append(cropped)
    return crops


def denormalize_bbox(
    bbox: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Convert normalized bounding box [0, 1] to pixel coordinates.
    
    Args:
        bbox: Normalized bounding box [xmin, ymin, xmax, ymax] in range [0, 1].
        image_shape: Image shape (height, width).
    
    Returns:
        Bounding box in pixel coordinates [xmin, ymin, xmax, ymax].
    
    Example:
        >>> bbox = np.array([0.1, 0.2, 0.5, 0.8])
        >>> pixel_bbox = denormalize_bbox(bbox, (1000, 800))
        >>> print(pixel_bbox)  # [80, 200, 400, 800]
    """
    h, w = image_shape
    result = bbox.copy()
    result[0] *= w  # xmin
    result[2] *= w  # xmax
    result[1] *= h  # ymin
    result[3] *= h  # ymax
    return result


def normalize_bbox(
    bbox: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Convert pixel bounding box to normalized coordinates [0, 1].
    
    Args:
        bbox: Bounding box in pixel coordinates [xmin, ymin, xmax, ymax].
        image_shape: Image shape (height, width).
    
    Returns:
        Normalized bounding box [xmin, ymin, xmax, ymax] in range [0, 1].
    
    Example:
        >>> bbox = np.array([80, 200, 400, 800])
        >>> norm_bbox = normalize_bbox(bbox, (1000, 800))
        >>> print(norm_bbox)  # [0.1, 0.2, 0.5, 0.8]
    """
    h, w = image_shape
    result = bbox.copy()
    result[0] /= w  # xmin
    result[2] /= w  # xmax
    result[1] /= h  # ymin
    result[3] /= h  # ymax
    return result
