# Multilingual Domain-Aware Correction LLM

A three-member research project for building a compact multilingual LLM from scratch that supports English, Hindi, code-mixed text, domain-aware correction, OCR, and document correction.

## Team ownership

- `src/core_llm/`: Member 1 - tokenizer, Transformer, training, inference
- `src/document_ai/`: Member 2 - OCR, language detection, domain intelligence
- `src/correction/` and `app/`: Member 3 - correction pipeline and API integration
- `src/integration/`, `src/evaluation/`, `tests/`: Shared ownership

## Initial scope

- Languages: English, Hindi, and code-mixed text
- Initial domain: Deep Learning
- Input: Text and images
- Output: Corrected text, structured changes, and later corrected images

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run tests

```bash
python -m pytest
```
