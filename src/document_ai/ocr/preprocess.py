"""
Image preprocessing for OCR.
Accepts image paths, PIL Images, or OpenCV ndarrays and applies contrast
enhancement and normalization for multilingual OCR.
"""

from pathlib import Path
from typing import Union
import cv2
import numpy as np
from PIL import Image


def load_image(image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Load image from various input types into a BGR/RGB numpy array.
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at: {img_path.resolve()}")
        # Support utf-8 file paths on Windows via cv2.imdecode
        img_bytes = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image from: {img_path}")
        return img
    elif isinstance(image_input, Image.Image):
        # Convert PIL to BGR numpy array
        rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Input numpy image array is empty")
        return image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def preprocess_for_ocr(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    grayscale: bool = True,
    enhance_contrast: bool = True,
) -> np.ndarray:
    """
    Preprocess document image for OCR extraction.

    Args:
        image_input: Path, PIL Image, or numpy array.
        grayscale: Whether to convert image to single-channel grayscale.
        enhance_contrast: Whether to apply CLAHE contrast enhancement.

    Returns:
        Preprocessed image as a numpy array.
    """
    img = load_image(image_input)

    if grayscale:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
    else:
        gray = img

    if enhance_contrast and len(gray.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    return gray
