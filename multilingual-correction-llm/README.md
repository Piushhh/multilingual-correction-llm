# Multilingual Domain-Aware LLM

## Member 1 — LLM Core & Model Training

### Completed Work

Member 1 is responsible for building the LLM core and training pipeline.

### Work Completed

* Transformer model architecture
* Token embeddings
* Positional embeddings
* Multi-head self-attention
* Feed-forward network
* Layer normalization
* Residual connections
* Language-model head
* Tokenizer integration
* Dataset loading and batching
* Causal language-model training
* Training and validation pipeline
* Checkpoint saving/loading
* Base-model training pipeline
* Domain-adaptive pretraining pipeline
* Text generation / inference
* Pytest test setup

### Main Commands

**Base training**

```bash
python -m src.core_llm.training.pretrain --train-file data/raw/train.txt --val-file data/raw/val.txt --tokenizer data/tokenizer/tokenizer.json --checkpoint-dir checkpoints/base
```

**Domain training**

```bash
python -m src.core_llm.training.pretrain --train-file data/domain/train.txt --val-file data/domain/val.txt --tokenizer data/tokenizer/tokenizer.json --init-from checkpoints/base/best.pt --checkpoint-dir checkpoints/domain
```

**Inference**

```bash
python -m src.core_llm.inference.generate --checkpoint checkpoints/domain/best.pt --tokenizer data/tokenizer/tokenizer.json --prompt "Deep learning" --max-new-tokens 40
```

**Testing**

```bash
python -m pytest
```

### Expected Checkpoints

```text
checkpoints/
├── base/
│   └── best.pt
└── domain/
    └── best.pt
```

---

# Member 2 — OCR & Document Intelligence

## Completed Work

Member 2 is responsible for converting document images into structured text and information.

### Work Completed

* OCR text extraction
* OCR bounding-box extraction
* OCR confidence information
* OCR ground-truth correction
* Corrected OCR dataset
* OCR vs corrected-text comparison
* CER evaluation
* WER evaluation
* Structured spatial parsing using bounding boxes
* Text-region classification/handling
* Extraction of structured fields from document regions
* Checkbox / selection interpretation
* Likert-scale spatial interpretation

### OCR Dataset

The corrected OCR dataset contains fields such as:

```text
page_id
text_region_id
bbox_x1
bbox_y1
bbox_x2
bbox_y2
text_type
language
corrected_text
ocr_text
changed
confidence
```

### Current OCR Workflow

```text
Document Image
      ↓
OCR
      ↓
Text + Bounding Boxes + Confidence
      ↓
OCR Correction / Ground Truth
      ↓
CER / WER Evaluation
      ↓
Spatial / Bounding-Box Parsing
      ↓
Structured Document Data
```

### Remaining Member 2 Work

The following roadmap items have not yet been confirmed as completed:

* Language detection module
* Domain classification
* Domain terminology database
* Complete structured document-understanding interface

Therefore, Member 2 is currently partially completed, with the OCR, evaluation, and spatial-parsing components completed.
