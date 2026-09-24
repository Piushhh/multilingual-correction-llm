# Team Ownership

This document defines file ownership for the "Multilingual Domain-Aware
Document Correction" project. All contributors must respect these boundaries
when creating, editing, or reviewing code.

---

## Member 1 — Aliah Jamil (Custom LLM)

**Primary ownership:** `src/core_llm/`

| Sub-area | Path |
|:---|:---|
| Tokenizer (BPE, training, serialization) | `src/core_llm/tokenizer/` |
| Model architecture (Transformer, Config) | `src/core_llm/model/` |
| Pretraining loop + checkpointing | `src/core_llm/training/` |
| Inference + LLM.generate() | `src/core_llm/inference/` |
| Domain-adaptive pretraining | `src/core_llm/training/pretrain.py` |
| Trained checkpoints | `checkpoints/` |
| LM corpus | `data/raw/train.txt`, `data/raw/val.txt` |
| Tokenizer artifacts | `data/tokenizer/` |

**Adapters (shared, primary author: M1):**
- `src/integration/adapters/custom_llm_adapter.py`
- `app/api/routes/generate.py`
- `tests/test_checkpoint.py`

---

## Member 2 — Aliah Jamil (DocumentAI)

**Primary ownership:** `src/document_ai/`

| Sub-area | Path |
|:---|:---|
| OCR extraction (Tesseract wrapper) | `src/document_ai/ocr/` |
| Language detection (langdetect + script) | `src/document_ai/language/` |
| Domain classifier (TF-IDF + LogReg) | `src/document_ai/domain/` |
| Terminology database + checking | `src/document_ai/terminology/` → `data/terminology/` |
| Interface schema (Pydantic v1.0) | `src/document_ai/schemas.py` |
| End-to-end pipeline + CLI | `src/document_ai/pipeline.py` |
| OCR evaluation | `src/evaluation/ocr_evaluation.py` |
| Domain classifier training data | `data/domain_classifier/` |
| Exported JSON schema | `data/schemas/document_ai_schema.json` |

**Adapters (shared, primary author: M2):**
- `src/integration/adapters/document_ai_adapter.py`
- `tests/test_document_ai.py`

---

## Member 3 — Piush (Correction Engine + API)

**Primary ownership:** `src/correction/`, `app/`

| Sub-area | Path |
|:---|:---|
| Correction engine inference | `src/correction/inference.py` |
| Model adapters (Gemma, Mock) | `src/correction/model_adapter.py` |
| Prompt formatting | `src/correction/prompts.py` |
| Training pipeline | `src/correction/train_correction.py` |
| Dataset preparation | `src/correction/prepare_dataset.py` |
| Correction dataset | `correction data/` |
| FastAPI app (lifespan, health) | `app/api/main.py` |
| Correction routes | `app/api/routes/correction.py` |
| Model info route | `app/api/routes/model.py` |
| Engine singleton | `app/api/dependencies.py` |
| App schemas | `app/schemas/` |
| Correction model config | `config/training_config.yaml` |

**Tests (primary: M3):**
- `tests/test_api.py`, `tests/test_correction.py`, etc.

---

## Shared Ownership

| Area | Path | Notes |
|:---|:---|:---|
| Integration contracts | `src/integration/contracts.py` | Any change requires team approval |
| Integration pipeline | `src/integration/pipeline.py` | Any change requires team approval |
| OCR adapter | `src/integration/adapters/ocr_adapter.py` | Any change requires team approval |
| Evaluation | `src/evaluation/` | Open collaboration |
| Tests | `tests/` | Each member owns tests for their code; shared tests by agreement |
| Documentation | `docs/`, `paper/` | Open collaboration |
| README | `README.md` | Open collaboration |
| LM + correction dataset | `data/raw/README.md` | Documents dual purpose |

---

## Ground Rules

1. **Do not move code** across ownership boundaries without a PR discussion.
2. **Do not duplicate** logic owned by another member — use adapters/contracts.
3. Before modifying a shared file, document: **FILE / WHY / WHAT / WHY ADAPTER NOT ENOUGH**.
4. All checkpoint and tokenizer artifacts stay under `checkpoints/` and `data/` (gitignored except for listed exceptions).
5. OCR results marked **NOT VERIFIED** until Tesseract is installed (see `docs/WINDOWS_TESSERACT.md`).
