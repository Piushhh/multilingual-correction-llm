# Correction Dataset Card

## Dataset Summary
This dataset contains examples for demonstrating, testing, and smoke-validating the multilingual correction pipeline. The initial release is a small 5-sample starter dataset covering English, Hindi, code-mixed text, and OCR-specific errors.

> [!WARNING]
> This is a synthetic starter / smoke-test dataset intended exclusively for pipeline validation. It is **NOT** a real-world benchmark and must not be cited or interpreted as measuring real-world model accuracy.

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
- **Source:** The starter dataset (`starter_dataset.jsonl`) is synthetically generated for demonstration and pipeline smoke testing.
- **Verification:** These samples are NOT human-verified. Their `verified` flag is set to `false` and `source_type` is `synthetic`.

## Preprocessing & Splits
To generate processed splits, run:
```bash
python -m src.correction.prepare_dataset --input data/raw/starter_dataset.jsonl --output-dir data/processed
```
This script handles validation against the schema, deduplication, and creates train/validation/test splits deterministically based on a configured random seed.

## Known Limitations
- The starter dataset is extremely small (5 examples) and intended only for validating the data pipelines, training scripts, and evaluation metrics.
- As synthetic example data, it does not represent the long-tail distribution of real-world OCR or code-mixed errors.
- Real human-verified and authenticated real-world evaluation datasets must be collected for genuine benchmark reporting.
