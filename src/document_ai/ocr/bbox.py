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


def order_regions_reading_order(regions):
    """
    Sort regions in human reading order: column left-to-right, then lines top-to-bottom.
    Detects vertical column gutters that separate disjoint column blocks.
    """
    if len(regions) <= 1:
        return regions

    # Check for vertical gutters that partition regions into left and right columns
    min_x = min(r["bbox"][0] for r in regions)
    max_x = max(r["bbox"][2] for r in regions)
    page_span = max_x - min_x
    if page_span <= 0:
        return sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    # Search for an x-gap that splits regions into two non-empty sets with no region crossing
    sorted_by_x = sorted(regions, key=lambda r: r["bbox"][0])
    best_split = None
    max_gutter_width = 0

    # Test candidate gutters between consecutive regions sorted by x
    for i in range(len(sorted_by_x) - 1):
        left_sub = sorted_by_x[:i + 1]
        right_sub = sorted_by_x[i + 1:]
        left_max_x = max(r["bbox"][2] for r in left_sub)
        right_min_x = min(r["bbox"][0] for r in right_sub)
        gutter = right_min_x - left_max_x
        # Require a clear gutter of at least 30px or 5% of span, with balanced content
        if gutter >= max(30, page_span * 0.04) and gutter > max_gutter_width:
            # Check neither subset is empty
            if len(left_sub) >= 1 and len(right_sub) >= 1:
                max_gutter_width = gutter
                best_split = (left_sub, right_sub)

    if best_split is not None:
        left_cols, right_cols = best_split
        # Recursively order left columns then right columns
        ordered_left = order_regions_reading_order(left_cols)
        ordered_right = order_regions_reading_order(right_cols)
        return ordered_left + ordered_right

    # Single column: top-to-bottom by y_min, tie-break by x_min
    return sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))


def group_words_into_lines(words, vertical_overlap_threshold=0.5, max_word_gap_ratio=2.5):
    """
    Group word-level boxes into line-level regions.

    Layout-aware (problems D and C):
    - Respects Tesseract block_num/par_num when available
    - Disallows merging across wide horizontal gutters (multi-column layouts)
    - Preserves word-level boxes inside each line dict (`words: [...]`)
    - Produces regions in multi-column reading order (left column first, top-to-bottom)

    Returns a list of line dicts: {text, bbox, confidence, words}.
    """
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (w["bbox"][1], w["bbox"][0]))

    lines = []  # each: {"words": [...]}

    for word in ordered:
        placed = False

        for line in lines:
            last_word = line["words"][-1]

            # 1. Vertical overlap check
            if vertical_overlap_ratio(word["bbox"], last_word["bbox"]) < vertical_overlap_threshold:
                continue

            # 2. Block/paragraph check (if Tesseract layout metadata is present)
            if "block_num" in word and "block_num" in last_word:
                if word["block_num"] != last_word["block_num"]:
                    continue

            # 3. Horizontal column gutter check: prevent joining words across column boundaries
            word_h = max(
                1,
                min(
                    word["bbox"][3] - word["bbox"][1],
                    last_word["bbox"][3] - last_word["bbox"][1],
                ),
            )
            h_gap = word["bbox"][0] - last_word["bbox"][2]
            max_gap = max(40, int(word_h * max_word_gap_ratio))
            if h_gap > max_gap:
                continue

            # 4. Do not attach if word starts well to the left of line start (wrap/disjoint)
            line_min_x = min(w["bbox"][0] for w in line["words"])
            if word["bbox"][2] < line_min_x - max_gap:
                continue

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

        confidences = [w["confidence"] for w in line_words if w.get("confidence") is not None]
        mean_confidence = sum(confidences) / len(confidences) if confidences else None

        word_dicts = [
            {
                "text": w["text"],
                "bbox": list(w["bbox"]),
                "confidence": w.get("confidence"),
            }
            for w in line_words
        ]

        regions.append(
            {
                "text": text,
                "bbox": box,
                "confidence": mean_confidence,
                "words": word_dicts,
            }
        )

    # Multi-column reading order: columns left-to-right, lines top-to-bottom
    return order_regions_reading_order(regions)
