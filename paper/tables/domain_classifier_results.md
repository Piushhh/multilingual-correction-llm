# Domain Classifier v2 Evaluation Results

- **Algorithm Selected:** `Calibrated_LinearSVC`
- **Test Accuracy:** 100.00%
- **Test Macro F1:** 100.00%
- **Dataset Split:** 722 train / 80 val / 80 test
- **Multilingual Composition:** 46.0% Hindi & Code-Mixed samples in train set

### Per-Domain Performance on Held-Out Test Set

| Domain | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `deep_learning` | 100.0% | 100.0% | 100.0% | 20 |
| `computer_science` | 100.0% | 100.0% | 100.0% | 20 |
| `mathematics` | 100.0% | 100.0% | 100.0% | 20 |
| `general` | 100.0% | 100.0% | 100.0% | 20 |

### Candidate Model Comparison

| Candidate Model | Val Acc (%) | Val Macro F1 (%) | Test Acc (%) | Test Macro F1 (%) |
|---|---|---|---|---|
| `LogisticRegression` | 96.2% | 96.2% | 97.5% | 97.5% |
| `Calibrated_LinearSVC` | 98.8% | 98.8% | 100.0% | 100.0% |
| `MultinomialNB` | 97.5% | 97.5% | 95.0% | 94.9% |
