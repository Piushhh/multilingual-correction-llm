# Correction Dataset Card

## Dataset Summary
This dataset contains examples for training and evaluating the multilingual correction LLM. The initial release is a small starter dataset covering English, Hindi, code-mixed text, and OCR-specific errors.

## Languages & Domains
- **Languages:** English (`en`), Hindi (`hi`), Code-mixed (`code-mixed`)
- **Domains:** General text, Deep Learning (`deep_learning`)

## Structure & Categories
The dataset follows the `CorrectionDatasetSchema` defined in `data/schemas/correction_schema.json`.

### Error Categories Present
- `grammar` (e.g., subject-verb agreement)
- `spelling`
- `matra_error` (Hindi specific)
- `language_boundary` (Code-mixed specific)
- `character_confusion` (OCR specific)
- `technical_term` (Domain specific)

## Data Provenance & Verification
- **Source:** The current starter dataset (`starter_dataset.jsonl`) is synthetically generated for demonstration and pipeline testing purposes.
- **Verification:** These samples are NOT human-verified. Their `verified` flag is set to `false` and `source_type` is `synthetic`.

## Preprocessing & Splits
To generate processed splits, run:
```bash
python -m src.correction.prepare_dataset --input data/raw/starter_dataset.jsonl --output-dir data/processed
```
This script handles validation against the schema, deduplication, and creates train/validation/test splits deterministically based on a configured random seed.

## Known Limitations
- The starter dataset is extremely small and intended only for validating the data pipelines, training scripts, and evaluation metrics.
- As a synthetic dataset, it may not perfectly represent the long-tail distribution of real-world OCR or code-mixed errors.
- Real human-verified data needs to be collected and appended to train a production-quality model.
