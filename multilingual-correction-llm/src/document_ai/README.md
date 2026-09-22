# Member 2 -- Document AI (OCR, Language, Domain)

Converts a document image into structured, versioned data: text regions
with bounding boxes and OCR confidence, document language (English / Hindi
/ code-mixed), domain (Deep Learning / Computer Science / Mathematics /
General), and flags on words that are either verified technical terms or
likely OCR errors on technical vocabulary.

## Layout

```
document_ai/
├── ocr/
│   ├── preprocess.py   # resize, denoise, CLAHE contrast, deskew, adaptive threshold
│   ├── bbox.py          # box math: IOU, union, grouping words -> line regions
│   └── extract.py       # Tesseract OCR -> word boxes -> line-level regions
├── language/
│   └── detector.py      # script-based + langdetect: en / hi / code_mixed
├── domain/
│   ├── classifier.py       # TF-IDF (word+char ngrams) + LogisticRegression
│   ├── train_classifier.py # CLI: train + save the classifier from labeled CSV
│   └── terminology.py      # verified-term database + fuzzy OCR-error detection
├── evaluation/
│   └── metrics.py       # CER/WER for OCR, accuracy for language detection
└── pipeline.py           # combines all of the above into the team interface
```

## Setup

Tesseract itself (the OCR engine) is a separate system install, not a
Python package:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-hin

# macOS
brew install tesseract tesseract-lang

# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

Then the Python side:

```bash
pip install -r requirements.txt
```

## Running the pipeline end to end

```bash
# 1. (One-time) generate the starter terminology database
python -m src.document_ai.domain.terminology
# -> writes data/terminology/{deep_learning,computer_science,mathematics}.json

# 2. Train the domain classifier on the seed dataset
python -m src.document_ai.domain.train_classifier \
    --data data/domain_classifier/train.csv \
    --output checkpoints/domain_classifier/model.joblib

# 3. Run the full pipeline on an image
python -c "
from src.document_ai.pipeline import process_document
from src.document_ai.domain.classifier import DomainClassifier

clf = DomainClassifier.load('checkpoints/domain_classifier/model.joblib')
result = process_document('path/to/page.jpg', page_id='page_001', domain_classifier=clf)
import json; print(json.dumps(result, indent=2, ensure_ascii=False))
"
```

## Output interface (version "1.0")

This is what Member 1 (for correction prompting context) and Member 3 (for
the correction pipeline / API) integrate against:

```json
{
  "version": "1.0",
  "page_id": "page_001",
  "language": "en",
  "language_confidence": 0.94,
  "domain": "deep_learning",
  "domain_confidence": 0.81,
  "regions": [
    {
      "text": "The transformer uses atention.",
      "bbox": [100, 200, 600, 250],
      "confidence": 0.91,
      "terminology_flags": [
        {
          "word": "atention",
          "position": 17,
          "status": "possible_ocr_error",
          "closest_term": "attention",
          "similarity": 94.1
        }
      ]
    }
  ]
}
```

`domain`/`domain_confidence` are `null` when no classifier is passed to
`process_document`, so Member 3 can develop the correction pipeline against
OCR + language output before the classifier is trained.

**A note on interface versioning:** the `version` field is there so Member
3's code can check it and fail loudly instead of silently misparsing a
future change to this shape -- bump `INTERFACE_VERSION` in `pipeline.py`
whenever a field is added, renamed, or removed, and tell the team.

## Development without a scanner / real dataset

For testing the OCR extraction pipeline without a physical document, render
a synthetic test image directly:

```python
from PIL import Image, ImageDraw
img = Image.new("RGB", (600, 100), "white")
draw = ImageDraw.Draw(img)
draw.text((10, 30), "The transformer uses attention.", fill="black")
img.save("data/interim/synthetic_test.png")
```

For Member 3: OCR fixtures (saved region JSON from real or synthetic pages)
belong in `data/interim/` so correction-pipeline development doesn't
require a working OCR + Tesseract install on every machine.

## What was actually tested in this environment

The sandbox this was built in has no network access, so `rapidfuzz`,
`langdetect`, `opencv-python`, and `pytesseract` couldn't be pip-installed
or exercised end-to-end here. What WAS verified directly:

- `ocr/bbox.py` -- box math and line-grouping (pure Python, no deps)
- `language/detector.py` -- script-composition detection on real English,
  Hindi, and code-mixed strings (works standalone; `langdetect` is an
  optional cross-check that degrades gracefully when absent, so this also
  incidentally proves that fallback path)
- `domain/classifier.py` + `train_classifier.py` -- trained end-to-end on
  the seed CSV, saved, reloaded, and used for prediction (scikit-learn was
  available in-sandbox)
- `evaluation/metrics.py` -- CER/WER against hand-checked expected values

**Not yet run in this environment** (install the two missing packages and
a Tesseract binary, then run these yourself before your first milestone
demo): `domain/terminology.py`'s fuzzy matching (`rapidfuzz`), the
`langdetect` cross-check branch specifically, and `ocr/extract.py` +
`ocr/preprocess.py` against a real image (`opencv-python` + `pytesseract` +
the Tesseract binary). The logic in all three follows each library's
documented API directly, but "should work" isn't the same as "verified" --
run `pytest tests/test_document_ai.py` once dependencies are installed and
treat that as your actual Milestone 1 (OCR baseline) checkpoint.

## What Member 2 must NOT own alone (per the team roadmap)

Transformer training (Member 1), correction fine-tuning (Member 3),
frontend, and paper writing are explicitly out of scope here -- this module
owns document input and the language/domain context layer only.
