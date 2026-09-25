# Windows Tesseract OCR Installation Guide

> **Status for this project:** Tesseract is NOT required for core development.
> The Member 2 pipeline gracefully handles absence of Tesseract: `is_tesseract_available()`
> returns `False`, `extract_regions()` returns `[]`, and all OCR-dependent tests
> are marked **NOT VERIFIED** until Tesseract is installed.

---

## 1. Install Tesseract

### Option A — UB Mannheim Installer (recommended for Windows)
1. Download the latest **64-bit installer** from:
   https://github.com/UB-Mannheim/tesseract/wiki
2. Run the `.exe` installer. Recommended install path:
   ```
   C:\Program Files\Tesseract-OCR\
   ```
3. During install, select the language packs you need:
   - **English** (eng) — included by default
   - **Hindi** (hin) — tick "Additional language data → hin"

### Option B — winget
```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```
Then separately download `hin.traineddata`:
```powershell
# Download Hindi traineddata into Tesseract tessdata folder
Invoke-WebRequest `
  -Uri "https://github.com/tesseract-ocr/tessdata/raw/main/hin.traineddata" `
  -OutFile "C:\Program Files\Tesseract-OCR\tessdata\hin.traineddata"
```

---

## 1b. Linux (Ubuntu / Debian) Installation

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-hin fonts-noto-core
```
Verify binary:
```bash
tesseract --version
tesseract --list-langs  # Should show eng, hin, osd
```

---

## 1c. macOS Installation

Using Homebrew:
```bash
brew install tesseract tesseract-lang
brew install font-noto-sans-devanagari
```
Verify binary:
```bash
tesseract --version
tesseract --list-langs  # Should show eng, hin, osd
```

---

## 2. Configure Path

After installation, tell Python where Tesseract lives. You can do this
in **one** of two ways:

### 2a. Environment Variable (preferred — no code changes)
Set the `TESSERACT_CMD` environment variable:
```powershell
# For the current session:
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"

# Permanently (User):
[System.Environment]::SetEnvironmentVariable("TESSERACT_CMD", "C:\Program Files\Tesseract-OCR\tesseract.exe", "User")
```

The `src/document_ai/ocr/extract.py` auto-reads `TESSERACT_CMD` on import
and calls `pytesseract.pytesseract.tesseract_cmd` accordingly.

### 2b. `.env` file
```
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
```

---

## 3. Verify Installation

```powershell
# Check binary exists
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version

# Check Python can find it
.\.venv\Scripts\python.exe -c "
from src.document_ai.ocr.extract import is_tesseract_available
print('Tesseract available:', is_tesseract_available())
"
```

---

## 4. Hindi Fonts (for test image generation)

Hindi Devanagari fonts are already installed on this machine:
- `C:\Windows\Fonts\mangal.ttf`
- `C:\Windows\Fonts\aparaj.ttf`

Use these in `PIL.ImageFont` when generating test images.

---

## 5. Troubleshooting

| Error | Fix |
|:---|:---|
| `TesseractNotFoundError` | Set `TESSERACT_CMD` to the full path of `tesseract.exe` |
| Hindi text detected as garbage | Add `hin` language pack; use `languages="eng+hin"` in `extract_regions()` |
| Low confidence on scanned PDFs | Pre-process with Pillow: increase DPI, threshold, denoise |
| `pytesseract` not installed | `.\.venv\Scripts\python.exe -m pip install pytesseract` |
