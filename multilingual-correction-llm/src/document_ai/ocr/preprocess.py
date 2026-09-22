"""
Image preprocessing for OCR.

Member 2, Responsibility 1: "Resizing, denoising, contrast enhancement,
deskewing, and preprocessing experiments."

Each step is exposed as its own function so preprocessing experiments (the
brief explicitly asks for these) are just a matter of re-ordering / dropping
steps in `preprocess_image`, not rewriting anything.
"""

import cv2
import numpy as np


def load_image(path):
    """Load an image from disk as a BGR numpy array (OpenCV's default)."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resize_image(image, max_dimension=2000):
    """
    Scale the image so its longer side is at most `max_dimension` pixels.

    Tesseract (and most OCR engines) work best in a moderate resolution
    band: too small and small text is unreadable, too large and processing
    is slow with no accuracy benefit. Upscaling very small scans a little
    also tends to help, so we scale in both directions.
    """
    height, width = image.shape[:2]
    longer_side = max(height, width)

    if longer_side == 0:
        return image

    scale = max_dimension / longer_side

    # Also upscale small images -- OCR engines generally do better with
    # text that's at least ~20-30px tall.
    if longer_side < 1000:
        scale = 1500 / longer_side

    if abs(scale - 1.0) < 1e-3:
        return image

    new_size = (int(width * scale), int(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

    return cv2.resize(image, new_size, interpolation=interpolation)


def to_grayscale(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray_image, strength=10):
    """Non-local means denoising -- removes scan/camera noise while keeping
    text edges relatively sharp (better for OCR than a plain blur)."""
    return cv2.fastNlMeansDenoising(gray_image, h=strength)


def enhance_contrast(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE: local contrast enhancement. Helps with unevenly lit photos of
    documents (a common case for phone-photographed pages) without blowing
    out already-bright regions the way global histogram equalization does."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)


def estimate_skew_angle(gray_image):
    """
    Estimate the document's rotation angle in degrees using the minimum-area
    bounding box of foreground (text) pixels.

    Returns a value in (-45, 45]; small scan skew is the common case this
    is meant to correct, not arbitrary rotation.
    """
    # Threshold first so we get clean foreground pixels to fit a box around.
    _, binary = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 20:
        # Not enough foreground pixels to estimate skew reliably.
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]

    # cv2.minAreaRect returns angle in [-90, 0); normalize to [-45, 45].
    if angle < -45:
        angle = 90 + angle

    return float(angle)


def deskew(gray_image, angle=None):
    """Rotate the image to correct skew. If `angle` isn't given, it's
    estimated with `estimate_skew_angle`."""
    if angle is None:
        angle = estimate_skew_angle(gray_image)

    # Skip tiny rotations -- not worth the interpolation cost/quality loss.
    if abs(angle) < 0.5:
        return gray_image

    height, width = gray_image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        gray_image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def adaptive_binarize(gray_image, block_size=31, c=15):
    """
    Adaptive thresholding to black-and-white. Local thresholding (vs. one
    global threshold) handles scans/photos with uneven lighting across the
    page, which is common for phone-captured documents.

    block_size must be odd; enforced here so callers can't pass a value
    that would make cv2.adaptiveThreshold raise.
    """
    if block_size % 2 == 0:
        block_size += 1

    return cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def preprocess_image(
    image_path,
    do_denoise=True,
    do_contrast=True,
    do_deskew=True,
    do_binarize=True,
):
    """
    Full preprocessing pipeline: load -> resize -> grayscale -> [denoise] ->
    [contrast] -> [deskew] -> [binarize].

    Each stage is optional so this doubles as the "preprocessing
    experiments" deliverable -- e.g. `preprocess_image(p, do_binarize=False)`
    to compare OCR accuracy with/without binarization on a given dataset.

    Returns a single-channel (grayscale or binary) numpy array ready for
    `ocr.extract`.
    """
    image = load_image(image_path)
    image = resize_image(image)
    gray = to_grayscale(image)

    if do_denoise:
        gray = denoise(gray)

    if do_contrast:
        gray = enhance_contrast(gray)

    if do_deskew:
        gray = deskew(gray)

    if do_binarize:
        gray = adaptive_binarize(gray)

    return gray
