"""
Image preprocessing for OCR.

Member 2, Responsibility 1: "Resizing, denoising, contrast enhancement,
deskewing, and preprocessing experiments."

Each step is exposed as its own function so preprocessing experiments (the
brief explicitly asks for these) are just a matter of re-ordering / dropping
steps in `preprocess_image`, not rewriting anything.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


def load_image(path):
    """Load an image from disk as a BGR numpy array (OpenCV's default)."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def compute_resize_scale(width, height, max_dimension=2000, upscale_below=1000, upscale_to=1500):
    longer_side = max(height, width)
    if longer_side == 0:
        return 1.0
    scale = max_dimension / longer_side
    if longer_side < upscale_below:
        scale = upscale_to / longer_side
    if abs(scale - 1.0) < 1e-3:
        return 1.0
    return float(scale)


def resize_image(image, max_dimension=2000):
    """
    Scale the image so its longer side is at most `max_dimension` pixels.

    Tesseract (and most OCR engines) work best in a moderate resolution
    band: too small and small text is unreadable, too large and processing
    is slow with no accuracy benefit. Upscaling very small scans a little
    also tends to help, so we scale in both directions.
    """
    resized, _scale = resize_image_tracked(image, max_dimension=max_dimension)
    return resized


def resize_image_tracked(image, max_dimension=2000, upscale_below=1000, upscale_to=1500):
    height, width = image.shape[:2]
    scale = compute_resize_scale(
        width, height, max_dimension=max_dimension,
        upscale_below=upscale_below, upscale_to=upscale_to,
    )
    if abs(scale - 1.0) < 1e-3:
        return image, 1.0
    new_size = (int(round(width * scale)), int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, new_size, interpolation=interpolation), scale


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


def deskew_tracked(gray_image, max_angle=MAX_DESKEW_DEGREES):
    """Like `deskew`, but also returns the CCW correction actually applied."""
    angle, confidence = _estimate_skew_with_confidence(
        gray_image, search_range=max_angle
    )
    if confidence < _DESKEW_CONFIDENCE_RATIO:
        return gray_image, 0.0
    if abs(angle) < MIN_DESKEW_DEGREES or abs(angle) > max_angle:
        return gray_image, 0.0
    rotated, _matrix = _rotate_same_size(
        gray_image,
        angle,
        border_value=255 if np.mean(gray_image) > 127 else 0,
        interpolation=cv2.INTER_CUBIC,
    )
    return rotated, float(angle)


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


def otsu_binarize(gray_image):
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def sauvola_binarize(gray_image, window_size=25, k=0.2, r=128.0):
    """
    Sauvola local thresholding. `window_size` is forced odd. Implemented
    with box filters so we do not depend on scikit-image.
    """
    if window_size % 2 == 0:
        window_size += 1
    gray = gray_image.astype(np.float32)
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(window_size, window_size), normalize=True)
    mean_sq = cv2.boxFilter(gray * gray, ddepth=-1, ksize=(window_size, window_size), normalize=True)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    threshold = mean * (1.0 + k * (std / float(r) - 1.0))
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return binary


def page_quality_stats(gray_image, tile=32):
    """
    Cheap histogram / illumination diagnostics used to decide whether
    contrast enhancement or binarization would help or hurt.
    """
    gray = gray_image if gray_image.ndim == 2 else to_grayscale(gray_image)
    # 1st and 99th percentiles capture text ink vs background even on sparse pages
    p1, p99 = np.percentile(gray, [1, 99])
    dynamic_range = float(p99 - p1)
    height, width = gray.shape[:2]
    tile = max(8, min(tile, height, width))
    tile_backgrounds = []
    for y in range(0, height - tile + 1, tile):
        for x in range(0, width - tile + 1, tile):
            t = gray[y:y + tile, x:x + tile]
            tile_backgrounds.append(float(np.percentile(t, 95)))
    illumination_std = float(np.std(tile_backgrounds)) if tile_backgrounds else 0.0
    return {
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "dynamic_range": dynamic_range,
        "illumination_std": illumination_std,
        "low_contrast": dynamic_range < 80.0,
        "uneven_illumination": illumination_std > 22.0,
    }


@dataclass
class PreprocessConfig:
    """
    Explicit preprocessing knobs for ablation and production.

    Default (problem B): grayscale + light denoise only. Aggressive CLAHE,
    deskew, and binarization are applied only when `auto_quality` sees a
    low-contrast or unevenly lit page. Task 9's OCR ablation is the source
    of this default; do not re-enable full-page binarize globally without
    re-running that evaluation.
    """

    resize: bool = True
    max_dimension: int = 2000
    upscale_below: int = 1000
    upscale_to: int = 1500
    denoise: bool = True
    denoise_strength: int = 7
    contrast: bool = False
    contrast_clip_limit: float = 2.0
    deskew: bool = False
    binarize: str = "none"  # none | otsu | adaptive | sauvola
    adaptive_block_size: int = 31
    adaptive_c: int = 15
    sauvola_window: int = 25
    sauvola_k: float = 0.2
    auto_quality: bool = True
    low_contrast_range: float = 80.0
    uneven_illumination_std: float = 22.0


DEFAULT_PREPROCESS_CONFIG = PreprocessConfig()


@dataclass
class GeometryTransform:
    """
    Records scale then rotation applied to the original page so OCR boxes
    can be mapped back to original-image coordinates (problem C).
    """

    original_width: int
    original_height: int
    scale: float = 1.0
    rotation_deg: float = 0.0  # CCW, applied after scale, same-size warp
    processed_width: int = 0
    processed_height: int = 0

    def map_xy_to_original(self, x: float, y: float) -> Tuple[float, float]:
        width = self.processed_width or max(1, int(round(self.original_width * self.scale)))
        height = self.processed_height or max(1, int(round(self.original_height * self.scale)))
        cx, cy = width / 2.0, height / 2.0
        theta = -np.deg2rad(self.rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx, dy = x - cx, y - cy
        # Inverse of OpenCV getRotationMatrix2D(center, rotation_deg, 1.0)
        xr = cos_t * dx + sin_t * dy + cx
        yr = -sin_t * dx + cos_t * dy + cy
        scale = self.scale if self.scale else 1.0
        return xr / scale, yr / scale

    def map_bbox_to_original(self, bbox: Sequence[float]) -> List[int]:
        x_min, y_min, x_max, y_max = bbox
        corners = [
            self.map_xy_to_original(x_min, y_min),
            self.map_xy_to_original(x_max, y_min),
            self.map_xy_to_original(x_min, y_max),
            self.map_xy_to_original(x_max, y_max),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        ox1 = int(np.floor(min(xs)))
        oy1 = int(np.floor(min(ys)))
        ox2 = int(np.ceil(max(xs)))
        oy2 = int(np.ceil(max(ys)))
        ox1 = max(0, min(ox1, self.original_width))
        oy1 = max(0, min(oy1, self.original_height))
        ox2 = max(0, min(ox2, self.original_width))
        oy2 = max(0, min(oy2, self.original_height))
        if ox2 < ox1:
            ox1, ox2 = ox2, ox1
        if oy2 < oy1:
            oy1, oy2 = oy2, oy1
        return [ox1, oy1, ox2, oy2]


@dataclass
class PreprocessResult:
    image: np.ndarray
    geometry: GeometryTransform
    config: PreprocessConfig
    quality: dict
    applied: dict = field(default_factory=dict)


def _apply_binarize(gray, method, config: PreprocessConfig):
    method = (method or "none").lower()
    if method in ("none", "", "false"):
        return gray
    if method == "otsu":
        return otsu_binarize(gray)
    if method == "adaptive":
        return adaptive_binarize(
            gray, block_size=config.adaptive_block_size, c=config.adaptive_c
        )
    if method == "sauvola":
        return sauvola_binarize(
            gray, window_size=config.sauvola_window, k=config.sauvola_k
        )
    raise ValueError(f"Unknown binarize method: {method}")


def config_from_legacy_flags(do_denoise=True, do_contrast=False, do_deskew=False, do_binarize=False):
    """Map the original boolean kwargs onto a PreprocessConfig."""
    explicit = do_contrast or do_deskew or do_binarize
    return PreprocessConfig(
        denoise=do_denoise,
        denoise_strength=7 if do_denoise else 10,
        contrast=do_contrast,
        deskew=do_deskew,
        binarize="adaptive" if do_binarize else "none",
        # Honour explicit old-style True flags; otherwise keep quality gating.
        auto_quality=not explicit,
    )


def preprocess_array(image, config: Optional[PreprocessConfig] = None) -> PreprocessResult:
    """
    Preprocess a BGR or grayscale numpy image. Tracks geometry for later
    mapping of OCR boxes back to the original pixel space.
    """
    config = config or DEFAULT_PREPROCESS_CONFIG
    if image.ndim == 2:
        original = image
        original_height, original_width = original.shape[:2]
        working = original
    else:
        original_height, original_width = image.shape[:2]
        working = image

    scale = 1.0
    if config.resize:
        working, scale = resize_image_tracked(
            working,
            max_dimension=config.max_dimension,
            upscale_below=config.upscale_below,
            upscale_to=config.upscale_to,
        )

    gray = to_grayscale(working)
    quality = page_quality_stats(gray)
    applied = {
        "denoise": False,
        "contrast": False,
        "deskew": False,
        "binarize": "none",
        "deskew_angle": 0.0,
        "scale": scale,
    }

    if config.denoise:
        gray = denoise(gray, strength=config.denoise_strength)
        applied["denoise"] = True

    want_contrast = config.contrast
    want_binarize = config.binarize if config.binarize != "none" else "none"
    if config.auto_quality:
        if quality["dynamic_range"] < config.low_contrast_range:
            want_contrast = True
        if quality["illumination_std"] > config.uneven_illumination_std:
            if want_binarize == "none":
                want_binarize = "adaptive"
        else:
            # Do not binarise a well-lit page just because the method is set,
            # unless the caller disabled auto_quality.
            if not config.contrast and config.binarize == "none":
                want_binarize = "none"
            elif config.binarize != "none" and not quality["uneven_illumination"] and not quality["low_contrast"]:
                want_binarize = "none"
        if not quality["low_contrast"] and not config.contrast:
            want_contrast = False

    if want_contrast:
        gray = enhance_contrast(gray, clip_limit=config.contrast_clip_limit)
        applied["contrast"] = True

    rotation = 0.0
    if config.deskew:
        gray, rotation = deskew_tracked(gray)
        applied["deskew"] = abs(rotation) > 0
        applied["deskew_angle"] = rotation

    if want_binarize != "none":
        gray = _apply_binarize(gray, want_binarize, config)
        applied["binarize"] = want_binarize

    proc_h, proc_w = gray.shape[:2]
    geometry = GeometryTransform(
        original_width=original_width,
        original_height=original_height,
        scale=scale,
        rotation_deg=rotation,
        processed_width=proc_w,
        processed_height=proc_h,
    )
    return PreprocessResult(image=gray, geometry=geometry, config=config, quality=quality, applied=applied)


def preprocess_with_geometry(image_path, config: Optional[PreprocessConfig] = None) -> PreprocessResult:
    image = load_image(image_path)
    return preprocess_array(image, config=config)


def preprocess_image(
    image_path,
    do_denoise=True,
    do_contrast=False,
    do_deskew=False,
    do_binarize=False,
    config: Optional[PreprocessConfig] = None,
):
    """
    Full preprocessing pipeline: load -> resize -> grayscale -> [denoise] ->
    [contrast] -> [deskew] -> [binarize].

    Backward compatible with the original boolean kwargs. Defaults changed
    after the problem-B ablation: contrast/deskew/binarize are off unless
    the caller enables them or `auto_quality` detects a degraded page.

    Pass `config=PreprocessConfig(...)` to control methods (otsu/adaptive/
    sauvola) and parameters. Returns a single-channel numpy array.
    """
    if config is None:
        config = config_from_legacy_flags(
            do_denoise=do_denoise,
            do_contrast=do_contrast,
            do_deskew=do_deskew,
            do_binarize=do_binarize,
        )
    return preprocess_with_geometry(image_path, config=config).image
