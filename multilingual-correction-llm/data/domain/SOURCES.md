# Domain corpus sources

## Survey-derived addition (added by Member 1, this session)

`train.txt` and `val.txt` were extended with 14 + 5 sentences transcribed
and cleaned from a real scanned document: "Understanding in Deep Learning —
Anonymous Survey" (page 1 of 3; only page 1 was available).

Pipeline this went through:
1. Page photographed → `data_ai` cropping produced 206 word/line-level
   region images (`cropped/text_*.png`).
2. Each crop was OCR'd with EasyOCR (`ocr_results_easyocr.txt`). This raw
   output is **too noisy to train on directly** — e.g. `"DeeD"`,
   `"Learning7"`, `"itermediate"`, many blank detections — so it was not
   appended as-is.
3. The sentences actually added here were manually reconstructed from the
   source photo (ground truth), not from the raw OCR strings, then split
   into EN / HI / code-mixed lines matching this project's three target
   languages (`configs/base.yaml`).
4. Train/val were split by *statement*, not by language, so no sentence's
   translation pair leaks across the split.

This is the first Hindi content in `train.txt` (previously English-only)
and the first code-mixed content in either `train.txt` or `val.txt`.

**Not yet done:** pages 2–3 of the survey weren't provided, and the raw
EasyOCR output / cropped images are still sitting in the uploaded zips —
useful as a labeled OCR test case for Member 2's `evaluation/metrics.py`
(CER/WER) later, but that wiring is out of scope for this change.
