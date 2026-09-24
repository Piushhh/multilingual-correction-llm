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


MAX_DESKEW_DEGREES = 15.0
MIN_DESKEW_DEGREES = 0.5
# Peak projection variance must beat the median by this ratio, or we skip.
_DESKEW_CONFIDENCE_RATIO = 1.08


def _rotate_same_size(image, angle_deg, border_value=255, interpolation=cv2.INTER_CUBIC):
    """Rotate around the image centre. Positive `angle_deg` is CCW (OpenCV)."""
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    ), matrix


def _ink_mask(gray_image):
    """Binary mask with ink = 255, background = 0."""
    if gray_image.ndim == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def _projection_variance(ink_mask, angle_deg):
    rotated, _ = _rotate_same_size(
        ink_mask, angle_deg, border_value=0, interpolation=cv2.INTER_NEAREST
    )
    projection = np.sum(rotated > 0, axis=1, dtype=np.float64)
    return float(np.var(projection))


def _estimate_skew_with_confidence(gray_image, search_range=MAX_DESKEW_DEGREES):
    """
    Projection-profile search for the CCW correction angle.

    Horizontal text produces a peaky row-sum histogram only when lines are
    aligned with the x-axis. We therefore search candidate correction
    angles and pick the one that maximises the variance of that histogram.

    This avoids `cv2.minAreaRect` whose angle convention changed in OpenCV
    4.5.1 ([-90, 0) historically, (0, 90] afterwards) and which, when the
    raw angle was applied as a rotation, *doubled* residual skew.

    Returns (correction_angle_degrees, confidence_ratio). Confidence is
    best_score / median_score; values near 1.0 mean the peak is not
    distinctive and the caller should skip correction.
    """
    ink = _ink_mask(gray_image)
    ink_pixels = int(np.count_nonzero(ink))
    if ink_pixels < 200:
        return 0.0, 0.0

    # Downsample so the coarse+fine search stays cheap on large pages.
    height, width = ink.shape[:2]
    longer = max(height, width)
    if longer > 800:
        scale = 800.0 / longer
        ink = cv2.resize(
            ink,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        _, ink = cv2.threshold(ink, 127, 255, cv2.THRESH_BINARY)

    def search(lo, hi, step):
        angles = np.arange(lo, hi + step * 0.5, step)
        scores = np.empty(len(angles), dtype=np.float64)
        for i, angle in enumerate(angles):
            scores[i] = _projection_variance(ink, float(angle))
        best_i = int(np.argmax(scores))
        median = float(np.median(scores))
        ratio = float(scores[best_i] / (median + 1e-9))
        return float(angles[best_i]), ratio, scores[best_i]

    coarse_angle, _, _ = search(-search_range, search_range, 1.0)
    window = 1.25
    fine_lo = max(-search_range, coarse_angle - window)
    fine_hi = min(search_range, coarse_angle + window)
    fine_angle, confidence, _ = search(fine_lo, fine_hi, 0.25)

    if abs(fine_angle) < MIN_DESKEW_DEGREES:
        return 0.0, confidence
    if abs(fine_angle) > search_range + 1e-6:
        return 0.0, 0.0
    return fine_angle, confidence


def estimate_skew_angle(gray_image, search_range=MAX_DESKEW_DEGREES):
    """
    Return the CCW rotation (degrees) that should be *applied* to deskew
    the page so text lines are horizontal.

    Range is approximately [-search_range, search_range]. Returns 0.0 when
    there is too little ink or the estimate is not distinctive.
    """
    angle, _confidence = _estimate_skew_with_confidence(
        gray_image, search_range=search_range
    )
    return float(angle)


def deskew(gray_image, angle=None, max_angle=MAX_DESKEW_DEGREES):
    """
    Rotate the image to correct skew.

    If `angle` is omitted it is estimated with the projection-profile
    method. The rotation applied is the *correction* angle (the opposite of
    the page's current tilt). Corrections larger than `max_angle` or with
    low confidence are skipped rather than guessing.
    """
    if angle is None:
        angle, confidence = _estimate_skew_with_confidence(
            gray_image, search_range=max_angle
        )
        if confidence < _DESKEW_CONFIDENCE_RATIO:
            return gray_image
    else:
        angle = float(angle)

    if abs(angle) < MIN_DESKEW_DEGREES:
        return gray_image
    if abs(angle) > max_angle:
        return gray_image

    rotated, _matrix = _rotate_same_size(
        gray_image,
        angle,
        border_value=255 if np.mean(gray_image) > 127 else 0,
        interpolation=cv2.INTER_CUBIC,
    )
    return rotated


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
