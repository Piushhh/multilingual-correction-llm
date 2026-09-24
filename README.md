# Multilingual Domain-Aware Document Correction System

An end-to-end research and engineering pipeline for **multilingual OCR text extraction, language detection, domain classification, domain terminology verification, and contextual error correction** with FastAPI serving.

Supports **English**, **Hindi**, and **English-Hindi code-mixed** document images and texts across domains (**Deep Learning**, **Computer Science**, **Mathematics**, and **General**).

---

## 1. System Architecture

```text
Document Image (.png / .jpg)
       │
       ▼
[ Member 2: DocumentAI Pipeline ]
  ├── 1. Tesseract OCR (bilingual eng+hin) -> regions: text, bbox, confidence
  ├── 2. Language Detection (Unicode script analysis + langdetect) -> en / hi / code_mixed / unknown
  ├── 3. Domain Classification (TF-IDF + Logistic Regression) -> domain + confidence
  └── 4. Terminology Verification (fuzzy matching against verified domain lexicons)
       │
       ▼
[ Stage 4 & 5: Integration Layer ]
  ├── DocumentAIAdapter: v1.0 DocumentAI output -> OCRDocument contract
  └── IntegrationPipeline: forwards full metadata (language, domain, terminology flags)
       │
       ▼
[ Member 3: Correction Engine ]
  ├── PromptFormatter: constructs domain/language-aware correction prompt
  ├── ModelAdapter: CustomLLMAdapter (Member 1) / GemmaBaselineAdapter / MockAdapter
  └── ChangeTracker: word-level diff analysis + category tagging (spelling, terminology, script)
       │
       ▼
[ REST API: FastAPI Application ]
  ├── GET  /health         — Service status + model readiness
  ├── POST /correct        — Correct single OCR text block
  ├── POST /correct/batch  — Batch correction of multiple text blocks
  ├── GET  /model/info     — Model metadata & capabilities
  └── POST /generate       — Direct text generation from Member 1 custom LLM
```

---

## 2. Team Ownership & Responsibilities

| Role | Member | Components & Ownership |
|:---|:---|:---|
| **Member 1** | Khushi Kumari Singh | `src/core_llm/`: Tokenizer, Causal Transformer, training pipelines, checkpoints, inference, `CustomLLMAdapter`, `app/api/routes/generate.py` |
| **Member 2** | Aliah Jamil | `src/document_ai/`: OCR extraction, Language detector, Domain classifier, Terminology DB, Pydantic schemas v1.0, pipeline CLI, OCR evaluation |
| **Member 3** | Piush | `src/correction/`, `app/`: Correction engine, dataset preparation, training scripts, FastAPI lifespan singleton, correction API routes, schemas |
| **Shared** | All | `src/integration/`: Contracts, adapters, pipeline; `src/evaluation/`: Metrics & error analysis; `tests/`: Test suite; `docs/` |

---

## 3. Component Details & Status

### Member 1: Core LLM (`src/core_llm/`)
- **Status:** **IMPLEMENTED & VERIFIED**
- **Architecture:** 4-layer, 4-head causal Transformer LM ($d_{model}=256, d_{ff}=1024, max\_seq\_len=256$, vocabulary size 1,739 tokens).
- **Checkpoints:** `checkpoints/base/best.pt` (~44 MB) and `checkpoints/domain/best.pt` (~44 MB).
- **CustomLLMAdapter:** Wraps the trained model through the `ModelAdapter` interface with left-truncation, prompt-prefix stripping, and EOS token handling.

### Member 2: DocumentAI (`src/document_ai/`)
- **Status:** **IMPLEMENTED & VERIFIED** (OCR marked *NOT VERIFIED* in environments without system Tesseract binary).
- **OCR:** Multi-region extraction with coordinates and per-block confidence (`src/document_ai/ocr/extract.py`). Auto-detects `TESSERACT_CMD` environment variable.
- **Language Detector:** Script composition analyzer supporting Devanagari, Latin, and code-mixed texts with thresholded confidence.
- **Domain Classifier:** Word + character n-gram TF-IDF vectorizer + balanced Logistic Regression. Trained on 80 samples across 4 domains; achieves 90% validation accuracy on held-out test split.
- **Terminology Verification:** Multi-domain dictionaries (`deep_learning.json`, `computer_science.json`, `mathematics.json`) with fuzzy string matching.
- **Schemas:** Pydantic v1.0 interface contract (`src/document_ai/schemas.py`) with JSON Schema export (`data/schemas/document_ai_schema.json`).

### Member 3: Correction Engine & API (`src/correction/`, `app/`)
- **Status:** **IMPLEMENTED & VERIFIED**
- **Inference:** `CorrectionEngine` supporting both real models and zero-dependency mock mode for test environments.
- **FastAPI:** Lifespan management for singleton model loading, thread-safe dependency injection, and complete request validation.

---

## 4. Setup & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.13.5 on Windows)
- (Optional) Tesseract OCR with English and Hindi language data (see [docs/WINDOWS_TESSERACT.md](docs/WINDOWS_TESSERACT.md))

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd multilingual-correction-llm

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate

# Install runtime and dev dependencies
pip install -r requirements-dev.txt
```

### Running Tests
The project features a comprehensive test suite across all three members and integration layers:
```bash
python -m pytest
```
**Test Results:** **78 passed, 0 failed** across all modules.

---

## 5. Usage & Examples

### DocumentAI CLI
Process a document image and output structured JSON:
```bash
python -m src.document_ai.pipeline --image sample.png --out output.json
```

### Start the REST API
```bash
# Run server with uvicorn (defaults to mock mode if weights/GPUs are absent)
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation available at `http://localhost:8000/docs`.

### API Endpoints
- **Health Check:** `GET /health`
  ```bash
  curl http://localhost:8000/health
  ```
- **Text Correction:** `POST /correct`
  ```bash
  curl -X POST http://localhost:8000/correct \
    -H "Content-Type: application/json" \
    -d '{"text": "The transformer architechture uses self-attenshun.", "language": "en", "domain": "deep_learning"}'
  ```
- **Custom LLM Direct Generation:** `POST /generate`
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
│   ├── api/
│   │   ├── dependencies.py       # Thread-safe CorrectionEngine singleton
│   │   ├── main.py               # Merged FastAPI entry point (lifespan + routers)
│   │   └── routes/
│   │       ├── correction.py     # /correct and /correct/batch
│   │       ├── generate.py       # /generate (Member 1 LLM endpoint)
│   │       └── model.py          # /model/info
│   └── schemas/correction.py     # API Pydantic schemas
├── checkpoints/
│   ├── base/best.pt              # Base pretrained LLM (~44MB)
│   └── domain/best.pt            # Domain-adapted LLM (~44MB)
├── data/
│   ├── domain_classifier/        # train.csv, val.csv, classifier.joblib
│   ├── raw/                      # train.txt, val.txt, starter_dataset.jsonl
│   ├── schemas/                  # document_ai_schema.json (v1.0 exported)
│   ├── terminology/              # deep_learning.json, computer_science.json, mathematics.json
│   └── tokenizer/                # tokenizer.json (BPE)
├── docs/
│   ├── team-ownership.md         # Detailed ownership & boundaries
│   └── WINDOWS_TESSERACT.md      # Tesseract OCR install guide
├── src/
│   ├── core_llm/                 # Member 1: Model, Tokenizer, Training, Inference
│   ├── document_ai/              # Member 2: OCR, Language, Domain, Schemas, Pipeline
│   ├── correction/               # Member 3: Engine, Prompts, Adapters, Training
│   ├── integration/              # Shared: Contracts, Adapters, Integration Pipeline
│   └── evaluation/               # Shared: OCR evaluation, correction metrics
├── tests/                        # 78 automated pytest tests
├── requirements.txt              # Core runtime dependencies
└── requirements-dev.txt          # Development, testing, and linting tools
```
