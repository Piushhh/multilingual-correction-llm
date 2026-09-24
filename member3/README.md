# Member 3: Correction Pipeline, Evaluation & API Integration

## Status

**Core implementation: COMPLETE.**
All integration contracts, correction engine, adapter architecture, FastAPI routes, and evaluation pipelines are fully wired. Member 1's custom model is connected via `CustomLLMAdapter`, and Member 2's Document AI components are connected via `OCRAdapter` and `DomainAdapter`.

---

## Architecture

```
OCR Output (Member 2) ────────► OCRAdapter ──────┐
                                                  ├──► IntegrationPipeline ──► CorrectedDocument
Domain Classification (Member 2) ► DomainAdapter ─┘         │
                                                            │
                                                     CorrectionEngine ◄─────┘
                                                          │
                                                     ModelAdapter (ABC)
                                                    ╱       │        ╲
                                         MockAdapter  GemmaBaseline  CustomLLMAdapter
                                         (testing)    (reference)    (Member 1 Custom PyTorch)
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
| Error analysis | `src/evaluation/error_analysis.py` | ✅ Complete |
| Model adapter interface | `src/correction/model_adapter.py` | ✅ Complete |
| Custom LLM adapter | `src/correction/model_adapter.py` | ✅ Connected to Member 1 |
| Integration contracts | `src/integration/contracts.py` | ✅ Complete |
| OCR & Domain adapters | `src/integration/adapters/ocr_adapter.py` | ✅ Complete |
| End-to-end pipeline | `src/integration/pipeline.py` | ✅ Complete |
| FastAPI application | `app/api/main.py` | ✅ Complete |
| Correction API routes | `app/api/routes/correction.py` | ✅ Complete |
| Model info API routes | `app/api/routes/model.py` | ✅ Complete |
| API dependency layer | `app/api/dependencies.py` | ✅ Complete |
| API Pydantic schemas | `app/schemas/correction.py` | ✅ Complete |
| Starter dataset | `data/raw/starter_dataset.jsonl` | ✅ Complete (synthetic) |
| Dataset card | `data/dataset_card.md` | ✅ Complete |
| Error taxonomy | `data/error_taxonomy.json` | ✅ Complete |
| Training config | `config/training_config.yaml` | ✅ Complete |

## Model Adapter Interface

Any model backend must implement `ModelAdapter` (defined in `src/correction/model_adapter.py`):
- `MockAdapter`: Fast deterministic output for testing without model weights.
- `CustomLLMAdapter`: Uses Member 1's custom PyTorch checkpoint (`src/core_llm/checkpoints/correction_model.pt`) and SentencePiece tokenizer, extracting core text to strictly respect the 64-token context window.
- `GemmaBaselineAdapter`: Hugging Face reference baseline (`google/gemma-2b-it`).

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Application health + model readiness |
| `POST` | `/correct` | Correct single text |
| `POST` | `/correct/batch` | Correct up to 50 texts |
| `GET` | `/model/info` | Model ID, baseline flag, load status |

## Known Limitations & Synthetic Dataset Disclaimer

> [!NOTE]
> The starter dataset in `data/raw/starter_dataset.jsonl` contains 5 synthetic samples used exclusively for smoke-testing and pipeline verification. It does not represent an empirical research benchmark.
