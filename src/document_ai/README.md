# Member 2 — Document AI (OCR, Language, Domain, Reconstruction)

Converts raw document images and multi-page PDFs into structured, versioned data:
- Preprocessed, deskewed, and quality-gated page images
- Original-space bounding box coordinates for all text regions and constituent words
- Column-aware line grouping and document reading order preservation
- Per-region and document-level language identification (`en`, `hi`, `code_mixed`)
- Calibrated 4-class domain classification (`deep_learning`, `computer_science`, `mathematics`, `general`) with confidence scores
- Verified terminology database (150+ terms per domain) with fuzzy OCR-error detection and protected-term tagging
- Post-correction document reconstruction with typography matching, Devanagari font rendering, and visual diff overlays

---

## Directory Architecture

```
src/document_ai/
├── ocr/
│   ├── preprocess.py       # Projection-profile deskew, PreprocessConfig, quality stats, Sauvola binarization
│   ├── bbox.py             # Column-aware line grouping, reading order, IOU & bounding box math
│   └── extract.py          # Tesseract extraction, original-space geometry mapping, word-level bboxes
├── language/
│   └── detector.py         # Script composition (Devanagari vs Latin) + deterministic langdetect
├── domain/
│   ├── classifier.py       # TF-IDF (word + char n-grams) + Calibrated LinearSVC classifier (v2)
│   ├── train_classifier.py # Baseline training script
│   └── terminology.py      # Terminology DB v2 (Hindi, aliases, fuzzy matching, protected terms)
├── reconstruct/
│   ├── __init__.py         # Public render_corrections API
│   └── render.py           # Inpainting, font auto-fitting, Devanagari rendering, visual diff highlights
├── evaluation/
│   └── metrics.py          # CER/WER edit distance & language accuracy
├── schemas.py              # Pydantic v1.0 interface schemas & DocumentAIBatchOutput
└── pipeline.py             # Unified end-to-end pipeline (images & multi-page PDFs)
```

---

## Key Capabilities & Engineering Hardening

### 1. Robust Deskewing & Quality Gating (`src/document_ai/ocr/preprocess.py`)
- **Projection-Profile Deskewing:** Replaced unbounded `minAreaRect` with horizontal projection-profile maximization across $[-10^\circ, +10^\circ]$ at $0.25^\circ$ resolution, reducing residual angular skew across all test angles to $\le \pm 0.25^\circ$.
- **Image Quality Gating:** Computes $P_1/P_{99}$ dynamic range and 95th-percentile tile illumination variance to dynamically activate CLAHE and adaptive binarization only on compromised pages, preventing degradation of clean scans.
- **Sauvola Binarization:** Integrates local windowed variance thresholding for unevenly lit camera captures.

### 2. Original-Space Coordinate Guarantee (`GeometryTransform`)
- Normalization/rescaling and deskew warp operations are tracked in a reversible `GeometryTransform`.
- All output bounding boxes (`bbox` and `words[].bbox`) are inverted back to original image space, ensuring downstream visual renderers and UI overlays align perfectly with unedited originals.

### 3. Layout-Aware Line Grouping (`src/document_ai/ocr/bbox.py`)
- Multi-column reading order detection guards against horizontal merging across column gutters.
- Preserves word-level bounding boxes and confidence scores in the output payload.

### 4. Domain Classifier v2 (`src/document_ai/domain/classifier.py`)
- Trained natively with `scikit-learn 1.6.1` on 722 balanced multilingual training samples (46.0% Hindi/code-mixed).
- Uses combined word n-grams (1-2) and character n-grams (3-5) with sublinear TF scaling.
- Calibrated `LinearSVC` achieves 100% test accuracy and 1.0000 macro F1 across `deep_learning`, `computer_science`, `mathematics`, and `general`.
- Low-confidence predictions ($<0.40$) gracefully abstain to `"general"`.

### 5. Terminology Database v2 (`data/terminology/`)
- $\ge 150$ verified technical terms per domain with Devanagari translations, common aliases, and `protected` flags.
- `find_protected_terms(text)` identifies domain-critical terminology that LLM correction prompts must preserve.
- Fuzzy matching via `rapidfuzz` catches subtle OCR misspellings on specialized domain vocabulary.

### 6. Corrected Document Image Reconstruction (`src/document_ai/reconstruct/`)
- Replaces corrupted text regions on original documents using estimated perimeter background inpainting.
- Automatically selects system fonts (`mangal.ttf`, `aparaj.ttf`, `arial.ttf`) and scales font size to fit bounding boxes.
- Supports Devanagari script rendering and optional visual diff boundary highlighting.

### 7. Multi-Page PDF Processing & Integration Contract
- `process_pdf(pdf_path)` renders pages via PyMuPDF at customizable DPI and produces `DocumentAIBatchOutput`.
- `DocumentAIAdapter` seamlessly transforms batch outputs into integration `OCRDocument` contracts.

---

## Quickstart & CLI Commands

### 1. Run Pipeline on Single Image
```bash
python -m src.document_ai.pipeline path/to/page.png --page-id sample_01
```

### 2. Run Pipeline on Multi-Page PDF
```bash
python -m src.document_ai.pipeline path/to/document.pdf --pdf --page-id doc_01
```

### 3. Reconstruct Corrected Document Image
```bash
python -m src.document_ai.reconstruct.render \
    --image path/to/original.png \
    --corrections path/to/corrections.json \
    --out path/to/reconstructed.png
```

### 4. Run Benchmark Evaluations & Ablations
```bash
# OCR Preprocessing Ablation
python -m src.evaluation.ocr_evaluation

# Domain Classifier Training & Evaluation
python scripts/retrain_domain_classifier.py

# Language Detection Evaluation
python scripts/eval_language_detection.py
```

---

## Testing & Verification
The full test suite covers all components:
```bash
python -m pytest
```
Output: **114 passed, 2 skipped (requiring live Tesseract), 0 failures**.
