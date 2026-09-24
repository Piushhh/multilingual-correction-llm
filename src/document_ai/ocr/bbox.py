"""
Bounding-box utilities shared by the OCR pipeline.

Member 2, Responsibility 1/2: "bounding boxes and document layout." Kept
separate from extract.py so bbox math can be unit-tested without needing
Tesseract installed.

Box format used everywhere in this module: [x_min, y_min, x_max, y_max]
(absolute pixel coordinates), matching the interface's `bbox` field.
"""


def to_xyxy(x, y, width, height):
    """Convert Tesseract-style (left, top, width, height) to
    [x_min, y_min, x_max, y_max]."""
    return [x, y, x + width, y + height]


def box_area(box):
    x_min, y_min, x_max, y_max = box
    return max(0, x_max - x_min) * max(0, y_max - y_min)


def union_box(box_a, box_b):
    """Smallest box containing both input boxes."""
    return [
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    ]


def intersection_over_union(box_a, box_b):
    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])

    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    if intersection == 0:
        return 0.0

    union = box_area(box_a) + box_area(box_b) - intersection
    if union <= 0:
        return 0.0

    return intersection / union


def vertical_overlap_ratio(box_a, box_b):
    """
    Fraction of the shorter box's height that overlaps vertically with the
    other box. Used to decide whether two word-boxes sit on the same text
    line, which is a much better signal than IOU for that purpose (two
    words on the same line rarely overlap in x, so their IOU is ~0).
    """
    y_min = max(box_a[1], box_b[1])
    y_max = min(box_a[3], box_b[3])
    overlap = max(0, y_max - y_min)

    height_a = box_a[3] - box_a[1]
    height_b = box_b[3] - box_b[1]
    shorter = max(1, min(height_a, height_b))

    return overlap / shorter


def group_words_into_lines(words, vertical_overlap_threshold=0.5):
    """
    Group word-level boxes into line-level regions.

    `words` is a list of dicts, each with at least `text`, `bbox`,
    `confidence`. Words are grouped greedily: process top-to-bottom,
    left-to-right, and attach a word to the most recent line it has
    sufficient vertical overlap with, else start a new line.

    Returns a list of line dicts: {text, bbox, confidence}, where `text` is
    the words joined with single spaces (in left-to-right order) and
    `confidence` is the mean of the word confidences in that line.
    """
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (w["bbox"][1], w["bbox"][0]))

    lines = []  # each: {"words": [...]}

    for word in ordered:
        placed = False

        for line in lines:
            last_word = line["words"][-1]
            if (
                vertical_overlap_ratio(word["bbox"], last_word["bbox"])
                >= vertical_overlap_threshold
            ):
                line["words"].append(word)
                placed = True
                break

        if not placed:
            lines.append({"words": [word]})

    regions = []
    for line in lines:
        line_words = sorted(line["words"], key=lambda w: w["bbox"][0])

        text = " ".join(w["text"] for w in line_words)

        box = line_words[0]["bbox"]
        for w in line_words[1:]:
            box = union_box(box, w["bbox"])

        confidences = [w["confidence"] for w in line_words if w["confidence"] is not None]
        mean_confidence = sum(confidences) / len(confidences) if confidences else None

        regions.append(
            {
                "text": text,
                "bbox": box,
                "confidence": mean_confidence,
            }
        )

    # Final reading-order sort: top-to-bottom by the grouped line boxes.
    regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

    return regions
