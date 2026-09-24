from pathlib import Path

import easyocr


ROOT_DIR = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT_DIR / "data" / "raw" / "cropped"
OUTPUT_FILE = ROOT_DIR / "data" / "raw" / "ocr_results_easyocr_clean.txt"


def main():
    reader = easyocr.Reader(
        ["en", "hi"],
        gpu=False,
    )

    images = sorted(
        IMAGE_DIR.glob("*.png"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    print(f"Images found: {len(images)}")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for i, image_path in enumerate(images, start=1):
            print(f"[{i}/{len(images)}] {image_path.name}")

            results = reader.readtext(
                str(image_path),
                detail=0,
                paragraph=True,
            )

            text = " ".join(
                result.strip()
                for result in results
                if result and result.strip()
            )

            f.write(f"File : {image_path.name}\n")
            f.write(f"Text : {text}\n")
            f.write("-" * 50 + "\n")

    print()
    print("OCR completed successfully.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()