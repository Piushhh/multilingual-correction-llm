# OCR Evaluation & Preprocessing Ablation Results

Evaluation across 27 benchmark pages (English, Hindi, Code-mixed) and 9 degradation regimes.

| Configuration | Mean CER (%) | Mean WER (%) | Clean CER (%) | Skew CER (%) | Uneven Light CER (%) |
|---|---|---|---|---|---|
| `raw` | 14.40% | 32.68% | 3.77% | 15.67% | 20.27% |
| `default_adaptive` | 10.80% | 24.76% | 3.77% | 13.88% | 8.72% |
| `clahe_only` | 11.49% | 26.27% | 3.77% | 15.67% | 9.54% |
| `deskew_only` | 11.03% | 25.26% | 3.77% | 5.55% | 20.27% |
| `sauvola_only` | 12.41% | 28.31% | 3.77% | 15.67% | 10.37% |
| `full_aggressive` | 9.34% | 21.55% | 7.27% | 5.55% | 9.54% |


### Language Breakdown (Default Adaptive Pipeline)

| Language | Mean CER (%) | Mean WER (%) |
|---|---|---|
| English (`en`) | 9.43% | 21.75% |
| Hindi (`hi`) | 11.83% | 27.03% |
| Code-Mixed (`code_mixed`) | 11.13% | 25.49% |
