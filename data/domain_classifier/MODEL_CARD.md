# Model Card: Domain Classifier (v3)

## Model Details
- **Architecture:** Combined Word n-grams (1–2) + Character n-grams (3–5), sublinear TF-IDF + Sigmoidal-calibrated `LinearSVC`
- **Scikit-Learn Version:** `1.6.1` (serialized with this exact version — see `requirements.txt`)
- **Intended Use:** Fast sub-millisecond document-level and per-region domain triage before LLM correction prompting.
- **Domains:** `deep_learning`, `computer_science`, `mathematics`, `general`

## Performance (v3 — 1240-sample training set)

| Split | Accuracy | Macro F1 |
|---|---|---|
| Validation (255 samples) | **98.75%** | **0.9875** |
| Test (80 held-out samples) | **100.00%** | **1.0000** |

## Training Data Summary

| Version | Train Samples | Val Samples | Hindi/CM % |
|---|---|---|---|
| v1 (baseline) | 400 | 80 | ~25% |
| v2 | 722 | 80 | 46.0% |
| **v3 (current)** | **1240** | **255** | **≥46%** |

### Hard-Case Expansion Focus (v3)
Version 3 explicitly targets the **mathematics ↔ deep_learning overlap** — the most frequently confused pair. Added examples include:
- **Pure math optimization** (gradient descent theory, Lagrangians, KKT conditions, Nesterov acceleration, proximal operators) that *looks like* DL training but belongs in `mathematics`
- **Applied DL optimization** (Adam, batch norm, dropout, LoRA, variational bounds, diffusion models) that *uses* math vocabulary but belongs in `deep_learning`
- **Linear algebra in DL** (attention projections, spectral normalization, Fisher information, Jacobians, SVD in embeddings) properly categorized by context
- **Hindi and code-mixed examples** covering all four hard-overlap areas

## Confidence Threshold
If `max(P(domain | text)) < 0.40`, prediction abstains to `"general"` to avoid low-confidence mis-classification.

## Limitations
- Designed for technical academic documents (EN/HI/code-mixed). May not generalize to social media or informal text.
- Serialized with scikit-learn 1.6.1; loading with a different version emits a version warning.
