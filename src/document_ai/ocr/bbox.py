"""
Bounding box conversion, validation, and reading-order sorting utilities.
Standard format: [x1, y1, x2, y2] (top-left, bottom-right).
"""

from typing import Any, List, Sequence, Union


def polygon_to_xyxy(polygon: Sequence[Sequence[Union[int, float]]]) -> List[int]:
    """
    Convert a 4-point polygon or quadrilateral from EasyOCR into [x1, y1, x2, y2].

    Args:
        polygon: Sequence of [x, y] coordinates, e.g., [[x1, y1], [x2, y1], [x2, y2], [x1, y2]].

    Returns:
        [x1, y1, x2, y2] bounding box with integer coordinates.
    """
    if not polygon or len(polygon) == 0:
        raise ValueError("Polygon sequence cannot be empty")

    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]

    min_x = max(0, int(round(min(xs))))
    min_y = max(0, int(round(min(ys))))
    max_x = max(0, int(round(max(xs))))
    max_y = max(0, int(round(max(ys))))

    if max_x < min_x:
        max_x = min_x
    if max_y < min_y:
        max_y = min_y

    return [min_x, min_y, max_x, max_y]


def validate_bbox(bbox: Sequence[int]) -> List[int]:
    """
    Validate that bbox is a list/tuple of 4 non-negative integers with x2 >= x1 and y2 >= y1.
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected 4 bounding box elements, got {len(bbox)}")

    x1, y1, x2, y2 = [int(v) for v in bbox]

    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        raise ValueError("Bounding box coordinates must be non-negative")

    if x2 < x1 or y2 < y1:
        raise ValueError(f"Invalid bounding box: [{x1}, {y1}, {x2}, {y2}]")

    return [x1, y1, x2, y2]


def sort_reading_order(blocks: List[dict], line_threshold: int = 15) -> List[dict]:
    """
    Sort blocks top-to-bottom, left-to-right (standard document reading order).

    Args:
        blocks: List of dicts, each containing a 'bbox' key [x1, y1, x2, y2].
        line_threshold: Pixel threshold to group blocks on approximately the same horizontal line.

    Returns:
        Sorted list of block dicts.
    """
    if not blocks:
        return []

    # Sort primarily by y1 quantized by line_threshold, secondarily by x1
    return sorted(
        blocks,
        key=lambda b: (b["bbox"][1] // max(1, line_threshold), b["bbox"][0])
    )
