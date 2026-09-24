# Multilingual Domain-Aware Document Correction System

An end-to-end research and engineering pipeline for **OCR text extraction, language detection, domain classification, terminology verification, and contextual error correction**, served through a FastAPI application.

Target inputs: **English**, **Hindi**, and **English-Hindi code-mixed** text or document images. Initial domains: **Deep Learning**, **Computer Science**, **Mathematics**, and **General**.

---

## 1. Team and Ownership

Roles below follow the git history of this repository.

| Role | Member | Owns | Main areas |
|:---|:---|:---|:---|
| **Member 1: LLM Core** | Khushi Kumari Singh | `src/core_llm/` | Tokenizer, causal Transformer, pretraining, domain-adaptive pretraining, correction fine-tuning, inference, LLM tests |
| **Member 2: Document AI** | Aliah Jamil | `src/document_ai/` | OCR (preprocess, extract, bounding boxes), language detection, domain classifier, terminology DB, Pydantic schemas, OCR/domain evaluation |
| **Member 3: Correction and API** | Piush | `src/correction/`, `app/` | Correction dataset, prompts, model adapters, change detection, training script, FastAPI service |
| **Shared** | All | `src/integration/`, `src/evaluation/`, `tests/`, `docs/` | Contracts, adapters, pipeline, metrics, test suite |

Evidence for the split:
- **Khushi:** her first commit (`c30fa1a`) is entirely `src/core_llm/`: model, SentencePiece BPE tokenizer, `pretrain.py`, `domain_adapt.py`, `correction_finetune.py`, `generate.py`, `correct.py`, and four test files.
- **Aliah:** her first commit (`80d8f66`) contains the full `src/document_ai/` package; a later commit (`dbcbf9e`) adds the Document AI schema, trained classifier, Tesseract setup guide, and end-to-end tests.
- **Piush:** every early commit is tagged `feat(member3)`: dataset, correction inference, evaluation metrics, API, adapters.

---

## 2. System Architecture

```text
Document image (.png / .jpg)  or  raw text
        │
        ▼
[ Member 2: Document AI ]
  ├── Preprocessing (grayscale, denoise, deskew)
  ├── OCR (Tesseract, eng+hin) -> regions: text, bbox, confidence
  ├── Language detection -> en / hi / code_mixed / unknown
  ├── Domain classification (TF-IDF + Logistic Regression)
  └── Terminology verification (fuzzy match against domain lexicons)
        │   structured JSON (schema v1.0)
        ▼
[ Shared: Integration layer ]
  ├── DocumentAIAdapter: Document AI output -> OCRDocument contract
  └── IntegrationPipeline: forwards language, domain, terminology flags
        │
        ▼
[ Member 3: Correction engine ]
  ├── PromptFormatter: domain- and language-aware prompt
  ├── ModelAdapter: CustomLLMAdapter (Member 1) / GemmaBaselineAdapter / MockAdapter
  └── Change detection: word-level diff + category tagging
        │
        ▼
[ Member 3: FastAPI ]
  ├── GET  /health
  ├── POST /correct
  ├── POST /correct/batch
  ├── GET  /model/info
  └── POST /generate      (direct generation from the custom LLM)
```

The core language model is a decoder-only Transformer implemented in PyTorch in this repository (no pretrained LLM weights), trained in three phases: general pretraining, domain-adaptive pretraining, then correction fine-tuning.

---

## 3. Component Status

### Member 1: LLM Core (`src/core_llm/`)
- **Architecture:** decoder-only causal Transformer LM. Default config: 4 layers, 4 heads, `d_model=256`, context length 256 (`configs/model.yaml`).
- **Tokenizer:** byte-level BPE, 1,739-token vocabulary (`data/tokenizer/tokenizer.json`).
- **Merged on the unified branch:** model files, tokenizer training, dataset loader, `pretrain.py`, `train_utils.py`, `generate.py`, and `CustomLLMAdapter`.
- **Still to be merged from `feature/khushi-work`:** `domain_adapt.py`, `correction_finetune.py`, `inference/correct.py`, `model/lm_head.py`, `model/config.py`, the SentencePiece tokenizer (`bpe_tokenizer.py`, `tokenizer.model`), corpus and tokenizer scripts, and `src/core_llm/tests/`.
- **Checkpoints:** `*.pt` files and `checkpoints/*` are git-ignored. Train locally or share them outside Git.

### Member 2: Document AI (`src/document_ai/`)
- **OCR:** multi-region extraction with coordinates and per-block confidence. Reads the `TESSERACT_CMD` environment variable. OCR tests need the system Tesseract binary; see [docs/WINDOWS_TESSERACT.md](docs/WINDOWS_TESSERACT.md).
- **Language detector:** Unicode script analysis (Devanagari vs Latin) plus `langdetect`, with code-mixed handling.
- **Domain classifier:** word and character n-gram TF-IDF with balanced Logistic Regression. Trained on a small hand-built dataset (about 100 labelled samples in `data/domain_classifier/`). Treat it as a baseline; accuracy on so few samples is not a reliable research result.
- **Terminology DB:** `deep_learning.json`, `computer_science.json`, `mathematics.json` with fuzzy matching (`rapidfuzz`).
- **Schema:** Pydantic v1.0 contract in `src/document_ai/schemas.py`, exported to `data/schemas/document_ai_schema.json`.

### Member 3: Correction Engine and API (`src/correction/`, `app/`)
- **Correction engine** with real-model and zero-dependency mock modes.
- **Dataset tooling:** schema, error taxonomy, validation, splitting, prompt formatting (`data/`, `src/correction/`).
- **FastAPI:** lifespan singleton model loading, dependency injection, request validation.
- **Evaluation:** correction metrics, error analysis, terminology-preservation checks (`src/evaluation/`).

---

## 4. Setup

**Prerequisites:** Python 3.10+; optionally Tesseract OCR with English and Hindi data.

```bash
git clone https://github.com/Piushhh/multilingual-correction-llm.git
cd multilingual-correction-llm

python -m venv .venv
# Windows:  .\.venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements-dev.txt
```

### Run the tests
```bash
python -m pytest
```
The suite has about 78 test functions across all modules. OCR-dependent tests need Tesseract installed.

### Start the API
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive docs: `http://localhost:8000/docs`. The service falls back to mock mode when no model weights are present.

---

## 5. Usage Examples

**Document AI CLI**
```bash
python -m src.document_ai.pipeline --image sample.png --out output.json
```

**Correct text**
```bash
curl -X POST http://localhost:8000/correct \
  -H "Content-Type: application/json" \
  -d '{"text": "The transformer architechture uses self-attenshun.", "language": "en", "domain": "deep_learning"}'
```

**Generate with the custom LLM**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Deep learning models", "max_new_tokens": 30, "temperature": 0.8}'
```

---

## 6. Project Structure

```text
multilingual-correction-llm/
├── app/
│   ├── api/                  # main.py, dependencies.py, routes/ (correction, generate, model)
│   └── schemas/              # API Pydantic schemas
├── configs/                  # model / training YAML
├── data/
│   ├── correction/           # correction pairs
│   ├── domain_classifier/    # train.csv, val.csv, classifier.joblib
│   ├── schemas/              # correction + Document AI JSON schemas
│   ├── terminology/          # domain lexicons
│   └── tokenizer/            # BPE tokenizer.json
├── docs/                     # team-ownership, Member 1 notes, Tesseract guide
├── src/
│   ├── core_llm/             # Member 1
│   ├── document_ai/          # Member 2
│   ├── correction/           # Member 3
│   ├── integration/          # Shared: contracts, adapters, pipeline
│   └── evaluation/           # Shared: metrics, error analysis
├── tests/
├── requirements.txt
└── requirements-dev.txt
```

---

## 7. Git Workflow

Branches: `main`, `feature/khushi-work` (Member 1), `feature/aliah-work` (Member 2), `feature/unified-integration`.

1. `git pull origin main` before starting work.
2. Work only in your own module; discuss before editing another member's files.
3. Open a pull request into `main` and get one review.
4. Keep datasets and model weights out of Git.

## 8. Open Items

- Merge Khushi's remaining `core_llm` files (Section 3) into the unified branch.
- Update `docs/team-ownership.md`, which still lists Aliah as Member 1, to match the table above.
- Expand the domain classifier dataset and report proper train/validation/test metrics.
- Add Hindi correction pairs and evaluate on code-mixed text.
