# Multilingual Language Detection Evaluation (Task 12)

- **Benchmark Size:** 156 curated samples (English: 52, Hindi: 52, Code-Mixed: 52)
- **Overall Accuracy (threshold=0.15):** 94.87%
- **Methodology:** Dual-signal hybrid combining unicode script composition (Devanagari vs Latin ratios) with statistical Latin verification.

### Confusion Matrix (Production Threshold = 0.15)

| True \ Predicted | English (`en`) | Hindi (`hi`) | Code-Mixed (`code_mixed`) | Recall (%) |
|---|---|---|---|---|
| **en** | 52 | 0 | 0 | 100.0% |
| **hi** | 0 | 52 | 0 | 100.0% |
| **code_mixed** | 5 | 0 | 44 | 89.8% |

### Code-Mixed Sensitivity Threshold Ablation

| Threshold (Minority Ratio) | Overall Accuracy (%) | English Acc (%) | Hindi Acc (%) | Code-Mixed Acc (%) |
|---|---|---|---|---|
| `0.05` | 100.00% | 100.0% | 100.0% | 100.0% |
| `0.10` | 99.36% | 100.0% | 100.0% | 98.1% |
| `0.15` **(Production Default)** | 94.87% | 100.0% | 100.0% | 84.6% |
| `0.20` | 80.77% | 100.0% | 100.0% | 42.3% |
| `0.25` | 73.08% | 100.0% | 100.0% | 19.2% |
| `0.30` | 68.59% | 100.0% | 100.0% | 5.8% |
