"""
Tests for GeometryTransform, original-space coordinates mapping (Task 3 / Problem C),
and PipelineOutputError (Task 4 / Problem E).
"""

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from src.document_ai.ocr.preprocess import GeometryTransform
from src.document_ai.ocr.extract import extract_regions
from src.document_ai.pipeline import PipelineOutputError, process_document
from src.document_ai.schemas import DocumentAIOutput


def test_geometry_transform_scale_only():
    w, h = 1000, 800
    scale = 1.5
    geom = GeometryTransform(
        original_width=w,
        original_height=h,
        scale=scale,
        rotation_deg=0.0,
        processed_width=int(w * scale),
        processed_height=int(h * scale),
    )
    # Box in processed coordinates
    proc_box = [150, 300, 300, 450]
    orig_box = geom.map_bbox_to_original(proc_box)
    assert orig_box == [100, 200, 200, 300]


def test_geometry_transform_rotation_roundtrip():
    w, h = 800, 1000
    angle = 6.0
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # A point in original image coordinates
    orig_x, orig_y = 200.0, 300.0
    pt = np.array([[[orig_x, orig_y]]], dtype=np.float32)
    # Forward rotate
    rot_pt = cv2.transform(pt, M)[0][0]

    geom = GeometryTransform(
        original_width=w,
        original_height=h,
        scale=1.0,
        rotation_deg=angle,
        processed_width=w,
        processed_height=h,
    )
    # Inverse map
    back_x, back_y = geom.map_xy_to_original(rot_pt[0], rot_pt[1])
    assert abs(back_x - orig_x) < 1e-3
    assert abs(back_y - orig_y) < 1e-3

    # An upright box in deskewed/processed space
    proc_box = [200, 300, 450, 340]
    orig_box = geom.map_bbox_to_original(proc_box)
    # The mapped box must enclose all 4 inversely mapped corners
    c1 = geom.map_xy_to_original(200, 300)
    c2 = geom.map_xy_to_original(450, 340)
    assert orig_box[0] <= c1[0] <= orig_box[2]
    assert orig_box[1] <= c1[1] <= orig_box[3]
    assert orig_box[0] <= c2[0] <= orig_box[2]
    assert orig_box[1] <= c2[1] <= orig_box[3]


def test_geometry_transform_scale_and_rotation():
    w, h = 600, 900
    scale = 1.25
    angle = -4.5
    pw, ph = int(w * scale), int(h * scale)

    geom = GeometryTransform(
        original_width=w,
        original_height=h,
        scale=scale,
        rotation_deg=angle,
        processed_width=pw,
        processed_height=ph,
    )

    orig_pt = (250.0, 400.0)
    # Forward manually
    cx, cy = pw / 2.0, ph / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    scaled_pt = np.array([[[orig_pt[0] * scale, orig_pt[1] * scale]]], dtype=np.float32)
    proc_pt = cv2.transform(scaled_pt, M)[0][0]

    rx, ry = geom.map_xy_to_original(proc_pt[0], proc_pt[1])
    assert abs(rx - orig_pt[0]) < 1e-3
    assert abs(ry - orig_pt[1]) < 1e-3


def test_extract_regions_returns_metadata_and_original_dimensions(tmp_path):
    img = np.full((700, 900, 3), 255, dtype=np.uint8)
    img_path = tmp_path / "page.png"
    cv2.imwrite(str(img_path), img)

    regions, meta = extract_regions(str(img_path), return_metadata=True)
    assert meta["original_width"] == 900
    assert meta["original_height"] == 700
    assert meta["geometry"].original_width == 900
    assert meta["geometry"].original_height == 700


def test_pipeline_output_error_chains_validation_error(monkeypatch, tmp_path):
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img_path = tmp_path / "tiny.png"
    cv2.imwrite(str(img_path), img)

    # Force an invalid language value from detector to trigger validation failure
    monkeypatch.setattr(
        "src.document_ai.pipeline.detect_language",
        lambda *args, **kwargs: {"language": "invalid_lang_code", "confidence": 1.0, "scripts": {}},
    )

    with pytest.raises(PipelineOutputError) as exc_info:
        process_document(str(img_path), page_id="test_page")

    assert isinstance(exc_info.value.__cause__, ValidationError)
    assert "Output failed schema validation" in str(exc_info.value)
