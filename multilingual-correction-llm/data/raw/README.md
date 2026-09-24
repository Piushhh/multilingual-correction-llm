# Raw Datasets Documentation

This directory contains the original input datasets used across different stages of the project.
They serve distinct purposes and are owned by different team members:

---

## 1. Language Model Pretraining Corpus (Member 1)
- **`train.txt`**: Unsupervised multilingual text corpus used for causal language model base pretraining.
- **`val.txt`**: Held-out validation text corpus for evaluating pretraining perplexity and cross-entropy loss.
- **Languages**: English, Hindi, and English-Hindi code-mixed sentences.
- **Usage**:
  ```bash
  python -m src.core_llm.training.pretrain --train-file data/raw/train.txt --val-file data/raw/val.txt --tokenizer data/tokenizer/tokenizer.json --checkpoint-dir checkpoints/base
  ```

---

## 2. Text & OCR Correction Dataset (Member 3)
- **`starter_dataset.jsonl`**: Supervised correction examples mapping noisy/corrupted text (including OCR-induced errors) to ground truth text.
- **Format**: JSON Lines matching `data/schemas/correction_schema.json`, annotated with error categories defined in `data/error_taxonomy.json`.
- **Fields**: `id`, `original_text`, `corrected_text`, `language`, `domain`, `error_types`, `source`.
- **Usage**: Used by `src/correction/prepare_dataset.py` and `src/correction/format_dataset.py` to create train/val/test splits for correction model fine-tuning.
