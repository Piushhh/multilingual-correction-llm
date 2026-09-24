# Multilingual Domain-Aware Correction LLM

A three-member research project for building a compact multilingual LLM from scratch that supports English, Hindi, code-mixed text, domain-aware correction, OCR, and document correction.

## Team Ownership

- `src/core_llm/` → Member 1: Tokenizer, Transformer, Training, Inference
- `src/document_ai/` → Member 2: OCR, Language Detection, Domain Intelligence
- `src/correction/` → Member 3: Correction Pipeline
- `app/` → API Integration
- `src/integration/`, `src/evaluation/`, `tests/` → Shared

---

# Member 1 — Core LLM

Member 1 implements the language model from scratch using PyTorch.

## Components

### Tokenizer

A SentencePiece BPE tokenizer is used for multilingual tokenization.

Location:

```text
src/core_llm/tokenizer/