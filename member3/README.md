# Member 3: Correction Pipeline & API Integration

## Role
Member 3 is responsible for the complete correction pipeline, evaluation, inference, backend API, and integration layer.

## Correction Architecture
The correction pipeline consists of:
- **Preprocessing:** Cleaning and formatting input text.
- **Language & Domain Processing:** Utilizing language and domain information to contextualize corrections.
- **Correction Model:** A configurable LLM-based model to perform grammatical, spelling, and OCR-related corrections.
- **Change Detection:** Comparing the original and corrected text to produce structured change information mapping back to error categories.
- **Integration Layer:** Adapting outputs from the OCR component and Domain Classifier to the correction pipeline.

## Expected Inputs
- **Text:** Raw text to be corrected.
- **Language:** e.g., 'en', 'hi'.
- **Domain:** e.g., 'deep_learning'.
- **OCR Outputs:** Text with bounding boxes, confidences, page and block identifiers.
- **Domain Classifier Outputs:** Domain predictions and confidences.

## Expected Outputs
- **Corrected Text:** The final text output.
- **Structured Changes:** Line-by-line or word-level changes with error categories.
- **Metadata:** Language, domain, and original OCR location data to reconstruct the document.
- **Reconstruction Output:** Structured JSON formatted for the frontend.

## Dependencies
- `fastapi` and `uvicorn` for the API.
- `torch` and `transformers` (expected) for the model.
- OCR outputs from Member 1.
- Domain outputs from Member 2.

## How to Run Each Component
*(To be updated as components are implemented)*
- **Data Preparation:** `python -m member3.correction.prepare_dataset ...`
- **Training:** `python -m member3.correction.train_correction ...`
- **Evaluation:** `python -m member3.evaluation.run_evaluation ...`

## How the API is Started
*(To be updated when the API is implemented)*
```bash
uvicorn member3.api.main:app --reload
```

## Testing Commands
Run all Member 3 tests via:
```bash
python -m pytest member3/tests/
```
