# Member 2: Document AI Layer

## 1. Overview & Responsibilities

Member 2 implements the Document AI layer responsible for extracting, analyzing, and structuring text content from document images before correction:

- **OCR Preprocessing & Extraction**: Grayscale conversion, contrast adjustment (CLAHE), and EasyOCR extraction for English (`en`) and Hindi (`hi`).
- **Bounding Box Standardization**: Polygon-to-axis-aligned bounding box conversion in `[x1, y1, x2, y2]` format with reading-order sorting.
- **Language Detection**: Deterministic Unicode script-based detection for English, Hindi, and code-mixed text.
- **Domain Classification & Terminology**: Technical terminology database for the Deep Learning domain with keyword-density heuristic classification.
- **OCR Evaluation**: CER (Character Error Rate) and WER (Word Error Rate) computation utilities.
- **Structured Representation**: Versioned schema (`schema_version: 1.0`) strictly conforming to Member 3 integration contracts.

```
Document Image
     ↓
Preprocessing (CLAHE contrast normalization)
     ↓
OCR Extraction (EasyOCR English & Hindi)
     ↓
Text + Bounding Boxes [x1, y1, x2, y2] + Confidence
     ↓
Language Detection (en / hi / code-mixed)
     ↓
Domain Classification & Terminology
     ↓
Structured DocumentAIDocument (Schema 1.0)
     ↓
Member 3 Integration Pipeline
```

---

## 2. Modules & Structure

```text
src/document_ai/
├── __init__.py
├── schemas.py              # Versioned Pydantic schemas (DocumentAIDocument, Page, Block)
├── pipeline.py             # DocumentAIPipeline end-to-end coordinator
├── ocr/
│   ├── __init__.py
│   ├── preprocess.py       # Grayscale, contrast enhancement, image loaders
│   ├── bbox.py             # Polygon-to-[x1, y1, x2, y2] conversion and reading-order sort
│   └── extract.py          # EasyOCR extractor for English and Hindi
├── language/
│   ├── __init__.py
│   └── detector.py         # Script-based detector (en, hi, code-mixed)
├── domain/
│   ├── __init__.py
│   ├── terminology.py      # Deep Learning technical terminology seed database
│   ├── classifier.py       # Terminology-based domain classifier
│   └── train_classifier.py # Keyword compilation utility
└── evaluation/
    ├── __init__.py
    └── ocr_metrics.py      # CER and WER evaluation metrics
```

---

## 3. Specifications

### Bounding Box Format
- Format: `[x1, y1, x2, y2]` (top-left x, top-left y, bottom-right x, bottom-right y).
- Coordinates are integer pixel offsets, with validation enforcing `x1 >= 0`, `y1 >= 0`, `x2 >= x1`, `y2 >= y1`.

### Language Detection
- Uses Unicode character block distributions:
  - Dominant Latin: `en`
  - Dominant Devanagari: `hi`
  - Co-occurring Latin + Devanagari (>=10% representation): `code-mixed`
  - Whitespace/punctuation only: `unknown` (confidence 0.0)

### Domain Classification
- Keyword matching against documented seed glossary for Deep Learning (terms such as `transformer`, `attention`, `neural network`, `अटेंशन`, `बैकप्रोपेगेशन`, etc.).
- Returns primary domain (`deep_learning` or fallback to `general`) with a heuristic density confidence score.

### Integration Compatibility
`DocumentAIDocument` provides `.to_integration_dict()` which maps directly to Member 3's `OCRDocument` contract (`document_id`, `pages`, `blocks` with `block_id`, `text`, `bbox`, `confidence`).

---

## 4. Limitations

1. **OCR Baseline**: Relies on EasyOCR without fine-tuning on noisy or damaged documents.
2. **Language Detection**: Uses script analysis rather than morphosyntactic linguistic modeling; Romanized Hindi (Hinglish written exclusively in Latin script) is detected as `en` by script analysis.
3. **Domain Classification**: Based on keyword density rather than statistical text classification.
