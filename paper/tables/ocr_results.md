# OCR Evaluation & Preprocessing Ablation Results

Comprehensive evaluation across 45 benchmark pages (27 synthetic + 18 real-world photographed/scanned pages) spanning English, Hindi, and Code-Mixed technical documents.

### 1. Overall Preprocessing Ablation (All 45 Evaluation Documents)

| Configuration | Mean CER (%) | Mean WER (%) | Clean/Scan CER (%) | Skew/Tilt CER (%) | Lighting/Shadow CER (%) |
|---|---|---|---|---|---|
| `raw` | 14.73% | 33.41% | 5.87% | 15.49% | 18.87% |
| `default_adaptive` | 10.01% | 23.01% | 5.87% | 12.56% | 8.30% |
| `clahe_only` | 10.94% | 25.06% | 5.87% | 15.49% | 9.05% |
| `deskew_only` | 12.17% | 27.77% | 5.87% | 5.88% | 18.87% |
| `sauvola_only` | 11.81% | 26.97% | 5.87% | 15.49% | 9.50% |
| `full_aggressive` | 9.58% | 22.07% | 9.07% | 5.88% | 9.05% |

### 2. Real-World Photographed & Scanned Evaluation (18 Documents)

| Preprocessing Mode | Real Mean CER (%) | Real Mean WER (%) | Camera Shadow (%) | Mobile Perspective (%) | Flatbed Scan (%) |
|---|---|---|---|---|---|
| `raw` | 15.23% | 34.51% | 17.57% | 14.97% | 7.97% |
| `default_adaptive` | 9.48% | 21.86% | 7.91% | 12.17% | 7.97% |
| `full_aggressive` | 9.88% | 22.74% | 8.60% | 5.79% | 11.17% |

### 3. Language Breakdown (Default Adaptive Pipeline on Full Benchmark)

| Language | Total Samples | Mean CER (%) | Mean WER (%) |
|---|---|---|---|
| English (`en`) | 15 | 8.64% | 20.01% |
| Hindi (`hi`) | 15 | 11.04% | 25.29% |
| Code-Mixed (`code_mixed`) | 15 | 10.34% | 23.75% |
