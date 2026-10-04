"""
PDF to PNG Dataset Preparation & Verification Script
===================================================
Automates organizing, converting, and verifying the Bengali & Hindi PDF dataset.
1. Organizes/copies SCAN PDFs from current directory into Forms_Dataset/{Language}/PDFs
2. Converts every page of every PDF into high-res 300 DPI PNGs:
   - Bengali1_001.png, Bengali1_002.png, ...
   - Hindi1_001.png, Hindi1_002.png, ...
3. Runs an automated verification audit to confirm zero page loss.
"""

import sys
import re
import shutil
from pathlib import Path

# Import PyMuPDF safely
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        print("[ERROR] PyMuPDF is not installed!")
        print("Please install it using: pip install pymupdf")
        sys.exit(1)

# =====================================================================
# CONFIGURATION
# =====================================================================
WORKSPACE_DIR = Path(__file__).resolve().parent
ROOT_DIR = WORKSPACE_DIR / "Forms_Dataset"
DPI = 300
LANGUAGES = ["Bengali", "Hindi"]
# =====================================================================


def natural_sort_key(file_path: Path):
    """
    Sorts filenames naturally (e.g., SCAN_1, SCAN_2, ..., SCAN_10, SCAN_100).
    """
    return [
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", file_path.name)
    ]


def auto_populate_dataset_folders():
    """
    If the Forms_Dataset/Bengali/PDFs and Hindi/PDFs folders are empty,
    safely copies the SCAN_*.pdf files from the workspace directory:
      - SCAN_1.pdf to SCAN_42.pdf   -> Hindi/PDFs
      - SCAN_43.pdf to SCAN_133.pdf -> Bengali/PDFs
    Never deletes or alters the original files.
    """
    bengali_pdf_dir = ROOT_DIR / "Bengali" / "PDFs"
    hindi_pdf_dir = ROOT_DIR / "Hindi" / "PDFs"
    bengali_pdf_dir.mkdir(parents=True, exist_ok=True)
    hindi_pdf_dir.mkdir(parents=True, exist_ok=True)

    bengali_existing = list(bengali_pdf_dir.glob("*.pdf"))
    hindi_existing = list(hindi_pdf_dir.glob("*.pdf"))

    # Only populate if both folders are currently empty
    if len(bengali_existing) == 0 and len(hindi_existing) == 0:
        scan_files = sorted(list(WORKSPACE_DIR.glob("SCAN_*.pdf")), key=natural_sort_key)
        if scan_files:
            print("Organizing root SCAN_*.pdf files into Bengali and Hindi folders...")
            hindi_copied = 0
            bengali_copied = 0

            for pdf_file in scan_files:
                match = re.search(r"SCAN_(\d+)", pdf_file.stem, re.IGNORECASE)
                if match:
                    scan_num = int(match.group(1))
                    if scan_num <= 42:
                        dest = hindi_pdf_dir / pdf_file.name
                        if not dest.exists():
                            shutil.copy2(pdf_file, dest)
                            hindi_copied += 1
                    else:
                        dest = bengali_pdf_dir / pdf_file.name
                        if not dest.exists():
                            shutil.copy2(pdf_file, dest)
                            bengali_copied += 1

            print(f"  -> Safely copied {hindi_copied} Hindi PDFs to Forms_Dataset/Hindi/PDFs/")
            print(f"  -> Safely copied {bengali_copied} Bengali PDFs to Forms_Dataset/Bengali/PDFs/")
            print("  -> Original files were NOT modified or deleted.\n")


def convert_and_verify_dataset(root_path: Path, dpi: int = 300):
    print("=" * 60)
    print("   PDF TO PNG DATASET CONVERTER & VERIFIER (300 DPI)   ")
    print("=" * 60)
    print(f"Dataset Root Directory: {root_path.resolve()}")
    print(f"Target Resolution:      {dpi} DPI\n")

    # Step 1: Ensure folders are populated
    auto_populate_dataset_folders()

    dataset_stats = {}
    total_failed_pdfs = []
    total_skipped_pngs = 0

    for lang in LANGUAGES:
        lang_dir = root_path / lang
        pdf_dir = lang_dir / "PDFs"
        png_dir = lang_dir / "PNG"

        print(f"[{lang.upper()}] Starting processing...")

        if not pdf_dir.exists():
            print(f"  [WARNING] Folder not found: {pdf_dir}")
            dataset_stats[lang] = {
                "pdf_count": 0, "total_pages": 0, "created_pngs": 0,
                "skipped_pngs": 0, "verification_pass": False,
                "issues": [f"Folder missing: {pdf_dir}"]
            }
            continue

        png_dir.mkdir(parents=True, exist_ok=True)

        # Natural sort guarantees deterministic numbering (Bengali1, Bengali2, ...)
        pdf_files = [
            f for f in pdf_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".pdf"
        ]
        pdf_files.sort(key=natural_sort_key)

        print(f"  Found {len(pdf_files)} PDF file(s) in {lang}/PDFs/")

        created_png_count = 0
        skipped_png_count = 0
        total_page_count = 0
        lang_failures = []
        pdf_records = []

        for pdf_idx, pdf_path in enumerate(pdf_files, start=1):
            pdf_num = pdf_idx
            try:
                with fitz.open(pdf_path) as doc:
                    num_pages = len(doc)
                    total_page_count += num_pages
                    pdf_records.append({
                        "pdf_num": pdf_num,
                        "pdf_name": pdf_path.name,
                        "num_pages": num_pages
                    })

                    for page_idx in range(num_pages):
                        page_num = page_idx + 1
                        png_filename = f"{lang}{pdf_num}_{page_num:03d}.png"
                        png_path = png_dir / png_filename

                        # Skip if already rendered
                        if png_path.exists() and png_path.stat().st_size > 0:
                            skipped_png_count += 1
                            continue

                        page = doc[page_idx]
                        pixmap = page.get_pixmap(dpi=dpi)
                        pixmap.save(str(png_path))
                        created_png_count += 1

            except Exception as exc:
                err_msg = f"{pdf_path.name}: {str(exc)}"
                print(f"     [ERROR] Failed to process {err_msg}")
                lang_failures.append(err_msg)
                total_failed_pdfs.append((lang, pdf_path.name, str(exc)))

        total_skipped_pngs += skipped_png_count

        # =================================================================
        # VERIFICATION PHASE FOR THIS LANGUAGE
        # =================================================================
        print(f"  Verifying generated {lang} PNG files against PDF page counts...")
        verification_issues = []

        for record in pdf_records:
            pdf_num = record["pdf_num"]
            pdf_name = record["pdf_name"]
            expected_pages = record["num_pages"]

            missing_pages = []
            for p in range(1, expected_pages + 1):
                expected_filename = f"{lang}{pdf_num}_{p:03d}.png"
                expected_file_path = png_dir / expected_filename

                if not expected_file_path.exists() or expected_file_path.stat().st_size == 0:
                    missing_pages.append(expected_filename)

            if missing_pages:
                issue_str = (
                    f"{pdf_name} (assigned as {lang}{pdf_num}): "
                    f"Expected {expected_pages} pages, missing {len(missing_pages)} PNG(s): "
                    f"{', '.join(missing_pages)}"
                )
                verification_issues.append(issue_str)

        if len(pdf_files) == 0:
            verification_issues.append(f"No PDF files found in {pdf_dir}")

        verif_passed = (
            len(lang_failures) == 0 and
            len(verification_issues) == 0 and
            len(pdf_files) > 0
        )

        dataset_stats[lang] = {
            "pdf_count": len(pdf_files),
            "total_pages": total_page_count,
            "created_pngs": created_png_count,
            "skipped_pngs": skipped_png_count,
            "verification_pass": verif_passed,
            "issues": verification_issues + lang_failures
        }
        print(f"  [{lang.upper()}] Done: {created_png_count} PNGs created ({skipped_png_count} skipped).\n")

    # =====================================================================
    # FINAL DATASET SUMMARY
    # =====================================================================
    print("=" * 10 + " DATASET SUMMARY " + "=" * 10 + "\n")

    for lang in LANGUAGES:
        stats = dataset_stats.get(lang, {
            "pdf_count": 0, "total_pages": 0, "created_pngs": 0, "verification_pass": False
        })
        print(f"{lang} PDFs found: {stats['pdf_count']}")
        print(f"{lang} pages: {stats['total_pages']}")
        print(f"{lang} PNGs created: {stats['created_pngs']}")
        if stats.get("skipped_pngs", 0) > 0:
            print(f"{lang} PNGs already existing (skipped): {stats['skipped_pngs']}")
        print()

    print(f"Failed PDFs: {len(total_failed_pdfs)}")
    print(f"Skipped existing PNGs: {total_skipped_pngs}\n")

    print("Verification:")
    all_passed = True
    for lang in LANGUAGES:
        passed = dataset_stats.get(lang, {}).get("verification_pass", False)
        status_text = "PASS" if passed else "FAIL"
        print(f"{lang}: {status_text}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 38)

    if not all_passed or total_failed_pdfs:
        print("\nDETAILED ISSUES REPORT:")
        for lang in LANGUAGES:
            issues = dataset_stats.get(lang, {}).get("issues", [])
            if issues:
                print(f"\n[{lang}] Issues:")
                for issue in issues:
                    print(f"  - {issue}")
    else:
        print("\nSUCCESS: All PDF pages have been rendered and verified with 100% accuracy!")


if __name__ == "__main__":
    convert_and_verify_dataset(ROOT_DIR, DPI)
