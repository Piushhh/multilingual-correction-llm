# Model Card: Domain Classifier (v2)

## Model Details
- **Architecture:** Combined Word n-grams (1-2) + Character n-grams (3-5) TF-IDF feature space + `Calibrated_LinearSVC`
- **Scikit-Learn Version:** `1.6.1` (trained natively on Python 3.13)
- **Intended Use:** Fast document-level and page-level domain triage before LLM correction prompting.
- **Test Accuracy:** 100.00%
- **Test Macro F1:** 100.00%
