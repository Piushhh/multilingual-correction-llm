# OCR Evaluation Dataset (Member 2 - Document AI)

This dataset benchmarks OCR accuracy across English (`en`), Hindi (`hi`), and code-mixed (`code_mixed`) technical documents under various real-world and synthetic degradations.

## Directory Structure

```
data/ocr_eval/
├── manifest.csv           # Index of evaluation samples
├── images/                # Document images (.png / .jpg)
│   ├── syn_en_clean.png
│   ├── syn_hi_skew_pos2.png
│   └── ...
├── ground_truth/          # UTF-8 ground truth transcriptions (.txt)
│   ├── syn_en_clean.txt
│   ├── syn_hi_skew_pos2.txt
│   └── ...
└── README.md
```

## Manifest Format (`manifest.csv`)

A CSV file containing the following columns:

| Column | Description | Example |
|---|---|---|
| `image_path` | Relative path to image file | `data/ocr_eval/images/syn_en_clean.png` |
| `ground_truth_path` | Relative path to transcription | `data/ocr_eval/ground_truth/syn_en_clean.txt` |
| `language` | Language code (`en`, `hi`, `code_mixed`) | `en` |
| `source` | `synthetic` or `real` | `synthetic` |
| `degradation` | Applied degradation type or condition | `skew_pos2`, `clean`, `blur`, `uneven_lighting` |

## Degradation Types in Synthetic Set
- `clean`: Untouched rendered text baseline
- `skew_pos2`: +2.0° affine rotation
- `skew_neg2`: -2.0° affine rotation
- `skew_pos5`: +5.0° affine rotation
- `gaussian_noise`: Additive Gaussian noise ($\sigma = 18$)
- `blur`: Gaussian blur ($5 \times 5$, $\sigma = 1.2$)
- `low_res`: Downsampled 2x area and upsampled with bicubic interpolation
- `jpeg_compression`: Heavy lossy JPEG compression ($Q = 25$)
- `uneven_lighting`: Horizontal non-linear lighting gradient shadow ($0.45 \to 1.0$)

## Real-World Photographed and Scanned Set (18 Samples)
- `camera_shadow`: Handheld mobile phone shadow with angle tilt (samples: `real_*_01`)
- `mobile_perspective`: Perspective homography and camera defocus (samples: `real_*_02`)
- `flatbed_scan`: CCD scanner sensor line artifacts and contrast compression (samples: `real_*_03`)
- `desk_lamp_glare`: Strong radial overhead illumination hotspot (samples: `real_*_04`)
- `xerox_bleed`: Ink bleed and toner threshold noise from multi-generation photocopy (samples: `real_*_05`)
- `spine_curve`: Cylindrical book spine curvature warp (samples: `real_*_06`)

## Adding Further Real Camera Photos / Scanned Pages
To incorporate additional real-world camera captures or physical document scans:
1. Save the document image in `data/ocr_eval/images/` (e.g. `real_hi_lecture_notes_07.jpg`).
2. Transcribe the reference text carefully in UTF-8 and save to `data/ocr_eval/ground_truth/real_hi_lecture_notes_07.txt`.
3. Append a new row to `data/ocr_eval/manifest.csv`:
   ```csv
   data/ocr_eval/images/real_hi_lecture_notes_07.jpg,data/ocr_eval/ground_truth/real_hi_lecture_notes_07.txt,hi,real,camera_shadow
   ```

## Running Evaluation
Run the automated ablation and benchmarking suite:
```bash
python -m src.evaluation.ocr_evaluation
```
Or directly:
```bash
python src/evaluation/ocr_evaluation.py --manifest data/ocr_eval/manifest.csv --out-dir paper/tables
```
Results, CER/WER ablation metrics, and visualizations will be generated in `paper/tables/` and `paper/figures/`.
