# Member 3: Correction Pipeline, Evaluation & API Integration

## Status

**Core implementation: COMPLETE.**
External dependencies (Member 1's trained model, Member 2's OCR and domain classifier) are not yet delivered. The system is fully functional with the mock adapter and ready for integration when those components are available.

---

## Architecture

```
OCR Output (Member 1) ──► OCR Adapter ──┐
                                         ├──► IntegrationPipeline ──► CorrectedDocument
Domain Classification (Member 2) ──► Domain Adapter ──┘         │
                                                                 │
                                          CorrectionEngine ◄─────┘
                                               │
                                          ModelAdapter (ABC)
                                         ╱       │        ╲
                              MockAdapter  GemmaBaseline  CustomLLMAdapter
                              (testing)    (baseline)     (Member 1 — stub)
```

## Completed Components

| Component | File(s) | Status |
|---|---|---|
| Dataset preparation & validation | `src/correction/prepare_dataset.py` | ✅ Complete |
| Dataset formatting | `src/correction/format_dataset.py` | ✅ Complete |
| Prompt engineering | `src/correction/prompts.py` | ✅ Complete |
| Correction inference engine | `src/correction/inference.py` | ✅ Complete |
| Change detection | `src/correction/change_detection.py` | ✅ Complete |
| Training pipeline | `src/correction/train_correction.py` | ✅ Complete |
| Evaluation metrics | `src/evaluation/correction_metrics.py` | ✅ Complete |
| Model adapter interface | `src/correction/model_adapter.py` | ✅ Complete |
| Gemma baseline adapter | `src/correction/model_adapter.py` | ✅ Baseline only |
| Custom LLM adapter stub | `src/correction/model_adapter.py` | ⏳ Stub — awaiting Member 1 |
| Integration contracts | `src/integration/contracts.py` | ✅ Complete |
| OCR & Domain adapters | `src/integration/adapters/ocr_adapter.py` | ✅ Complete |
| End-to-end pipeline | `src/integration/pipeline.py` | ✅ Complete |
| FastAPI application | `app/api/main.py` | ✅ Complete |
| Correction API routes | `app/api/routes/correction.py` | ✅ Complete |
| Model info API routes | `app/api/routes/model.py` | ✅ Complete |
| API dependency layer | `app/api/dependencies.py` | ✅ Complete |
| API Pydantic schemas | `app/schemas/correction.py` | ✅ Complete |
| Starter dataset | `data/raw/starter_dataset.jsonl` | ✅ Complete |
| Dataset card | `data/dataset_card.md` | ✅ Complete |
| Error taxonomy | `data/error_taxonomy.json` | ✅ Complete |
| Training config | `config/training_config.yaml` | ✅ Complete |

## Current Baseline Model

The system uses **google/gemma-2b-it** as the documented baseline via `GemmaBaselineAdapter`. This is explicitly labeled as a baseline in:
- `config/training_config.yaml` (`is_baseline: true`)
- `/model/info` API response (`"baseline": true`)
- Adapter code (`is_baseline` property returns `True`)

**No actual model training has been performed.** Gemma is used as-is from HuggingFace for baseline evaluation.

## Model Adapter Interface (Custom LLM Contract)

Any model backend must implement `ModelAdapter` (defined in `src/correction/model_adapter.py`):

```python
class ModelAdapter(ABC):
    def load(self) -> None: ...
    def generate(self, prompt: str, **kwargs) -> str: ...
    def is_loaded(self) -> bool: ...

    @property
    def model_id(self) -> str: ...

    @property
    def is_baseline(self) -> bool: ...
```

To integrate Member 1's model:
1. Implement `CustomLLMAdapter.load()` using their model class from `src/core_llm/`
2. Implement `CustomLLMAdapter.generate()` using their inference interface
3. Set `is_baseline` to `False`
4. Update `config/training_config.yaml` with the model path/ID

## Integration Contracts

Defined in `src/integration/contracts.py`:

- **`OCRDocument`** → `OCRPage` → `OCRBlock` (block_id, text, bbox, confidence)
- **`DomainClassification`** (domain, confidence, alternatives)
- **`CorrectedDocument`** → `CorrectedPage` → `CorrectedBlock`

Every `CorrectedBlock` preserves:
- `block_id`, `original_text`, `corrected_text`, `changes`
- `bbox`, `ocr_confidence`, `language`, `domain`, `correction_metadata`

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Application health + model status |
| `POST` | `/correct` | Correct a single text |
| `POST` | `/correct/batch` | Correct up to 50 texts |
| `GET` | `/model/info` | Model ID, baseline flag, load status |

### Running the API

```bash
# Mock mode (no model required)
CORRECTION_MOCK_MODE=true uvicorn app.api.main:app --reload

# Production mode (requires model weights)
uvicorn app.api.main:app --reload
```

### Configuration

| Environment Variable | Description | Default |
|---|---|---|
| `CORRECTION_MOCK_MODE` | `true` / `1` / `yes` to use mock adapter | `false` |
| `CORRECTION_MODEL_CONFIG` | Path to YAML config file | `config/training_config.yaml` |

## Testing

```bash
python3 -m pytest tests/ -v
```

**53 tests** covering:
- Dataset validation and splitting (5 tests)
- Prompt formatting (3 tests)
- Change detection (4 tests)
- Inference engine: mock, empty input, no-model error (3 tests)
- API endpoints: correct, batch, health, model info, validation (8 tests)
- Health endpoint (2 tests)
- Dependency singleton, thread safety, mock mode, config, failure propagation (18 tests)
- Integration pipeline: single block, field preservation, multi-page, adapters, adapter contracts (10 tests)
- Training pipeline (1 test)

## Dependencies on Other Members

### Member 1 (Core LLM)
- **Needed:** Trained model checkpoint or HuggingFace model ID
- **Needed:** Model class in `src/core_llm/model/`
- **Needed:** Tokenizer in `src/core_llm/tokenizer/`
- **Needed:** Inference interface in `src/core_llm/inference/`
- **Currently:** `src/core_llm/` directories exist but contain no Python files
- **Impact:** `CustomLLMAdapter` raises `NotImplementedError` until delivered

### Member 2 (Document AI)
- **Needed:** OCR engine producing `OCRDocument`-compatible output
- **Needed:** Domain classifier producing `DomainClassification`-compatible output
- **Currently:** `src/document_ai/` directories exist but contain no Python files
- **Impact:** `OCRAdapter` and `DomainAdapter` use passthrough parsing; real adapter logic will be added when Member 2's output format is known

## What Cannot Be Completed Without External Dependencies

1. **Real model inference** — requires Member 1's trained model
2. **Real OCR integration** — requires Member 2's OCR pipeline output format
3. **Real domain classification** — requires Member 2's classifier output format
4. **End-to-end evaluation on production data** — requires both of the above
5. **Production deployment** — requires a real model, not the baseline
