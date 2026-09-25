"""
Synthetic OCR evaluation dataset generator (Task 9 / Milestone 6).

Generates synthetic pages across English, Hindi, and code-mixed domain text,
applies realistic document degradations (skew, noise, blur, low resolution,
compression, uneven illumination), and writes data/ocr_eval/manifest.csv.
"""

import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Set fixed seed for full reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EVAL_DIR = Path("data/ocr_eval")
IMAGES_DIR = EVAL_DIR / "images"
GT_DIR = EVAL_DIR / "ground_truth"


def find_fonts():
    """Locate Latin and Devanagari fonts on the system."""
    windir = os.environ.get("WINDIR", r"C:\Windows")
    latin_candidates = [
        Path(windir) / "Fonts" / "arial.ttf",
        Path(windir) / "Fonts" / "calibri.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    devanagari_candidates = [
        Path(windir) / "Fonts" / "mangal.ttf",
        Path(windir) / "Fonts" / "aparaj.ttf",
        Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
        Path("/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"),
    ]

    latin_font = next((str(p) for p in latin_candidates if p.is_file()), None)
    devanagari_font = next((str(p) for p in devanagari_candidates if p.is_file()), None)

    if not devanagari_font:
        print(
            "WARNING: Devanagari font (mangal.ttf/NotoSansDevanagari) not found.\n"
            "To install on Ubuntu: sudo apt-get install fonts-noto-core\n"
            "To install on macOS: brew install font-noto-sans-devanagari",
            file=sys.stderr,
        )
    return latin_font, devanagari_font


SAMPLE_CORPUS = {
    "en": [
        "Convolutional neural networks extract hierarchical spatial representations from raw input images.",
        "The transformer architecture relies on multi-head self-attention mechanisms for sequence modeling.",
        "Gradient descent with momentum accelerates optimization along directions of low curvature.",
        "Backpropagation computes partial derivatives of the loss function with respect to weights.",
        "Regularization techniques such as weight decay and dropout prevent catastrophic overfitting.",
        "Eigenvalues and eigenvectors characterize invariant linear transformations in vector spaces.",
        "The computational complexity of matrix multiplication scales cubically with standard algorithms.",
    ],
    "hi": [
        "गहन शिक्षण में न्यूरल नेटवर्क जटिल पैटर्न को सीखने के लिए कई परतों का उपयोग करते हैं।",
        "ग्रेडिएंट डिसेंट एल्गोरिदम हानि फलन को न्यूनतम करने के लिए भारों को अपडेट करता है।",
        "कंप्यूटर विज्ञान में डेटा संरचनाएं और एल्गोरिदम समस्याओं के कुशल समाधान का आधार हैं।",
        "बैकप्रोपेगेशन प्रत्येक परत के लिए ढाल की गणना करने की एक प्रभावी विधि है।",
        "मैट्रिक्स गुणन और रैखिक बीजगणित मशीन लर्निंग के गणितीय आधार का निर्माण करते हैं।",
        "अति-अनुकूलन को रोकने के लिए नियमितीकरण और ड्रॉपआउट का व्यापक उपयोग किया जाता है।",
    ],
    "code_mixed": [
        "यह transformer model बहु-शीर्ष self-attention mechanism का प्रभावी रूप से उपयोग करता है।",
        "Deep learning में backpropagation की मदद से weights और biases को optimize किया जाता है।",
        "Convolutional layers स्थानीय spatial features को extract करने के लिए filter kernels apply करती हैं।",
        "Overfitting से बचने के लिए dropout layer और L2 regularization का प्रयोग आवश्यक है।",
        "Adam optimizer सीखने की दर को adaptive step sizes के साथ dynamically adjust करता है।",
        "Neural network training के दौरान cross-entropy loss gradients compute किए जाते हैं।",
    ],
}


def render_page(lines, font_path, font_size=28, width=800, height=600):
    """Render a text page on a clean white background."""
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    y = 50
    for line in lines:
        draw.text((40, y), line, fill=(0, 0, 0), font=font)
        y += int(font_size * 1.8)

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def apply_degradation(image: np.ndarray, degradation_type: str) -> np.ndarray:
    """Apply realistic synthetic scanning and capture degradations."""
    h, w = image.shape[:2]

    if degradation_type == "clean":
        return image.copy()

    elif degradation_type == "skew_pos2":
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 2.0, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))

    elif degradation_type == "skew_neg2":
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -2.0, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))

    elif degradation_type == "skew_pos5":
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 5.0, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))

    elif degradation_type == "gaussian_noise":
        noise = np.random.normal(0, 18, image.shape).astype(np.float32)
        degraded = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return degraded

    elif degradation_type == "blur":
        return cv2.GaussianBlur(image, (5, 5), 1.2)

    elif degradation_type == "low_res":
        small = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    elif degradation_type == "jpeg_compression":
        _, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    elif degradation_type == "uneven_lighting":
        # Create horizontal shadow gradient across the page
        gradient = np.linspace(0.45, 1.0, w, dtype=np.float32).reshape(1, w, 1)
        degraded = np.clip(image.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
        return degraded

    else:
        raise ValueError(f"Unknown degradation type: {degradation_type}")


def generate_dataset():
    latin_font, devanagari_font = find_fonts()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)

    degradations = [
        "clean",
        "skew_pos2",
        "skew_neg2",
        "skew_pos5",
        "gaussian_noise",
        "blur",
        "low_res",
        "jpeg_compression",
        "uneven_lighting",
    ]

    manifest_rows = ["image_path,ground_truth_path,language,source,degradation"]
    item_idx = 0

    for lang, corpus_lines in SAMPLE_CORPUS.items():
        font = devanagari_font if lang in ("hi", "code_mixed") and devanagari_font else latin_font
        # Render clean base page
        base_image = render_page(corpus_lines[:5], font_path=font)
        gt_text = "\n".join(corpus_lines[:5])

        for deg in degradations:
            deg_image = apply_degradation(base_image, deg)
            img_filename = f"syn_{lang}_{deg}.png"
            gt_filename = f"syn_{lang}_{deg}.txt"

            img_path = IMAGES_DIR / img_filename
            gt_path = GT_DIR / gt_filename

            cv2.imwrite(str(img_path), deg_image)
            gt_path.write_text(gt_text, encoding="utf-8")

            # Store paths relative to repo root
            rel_img = str(img_path).replace("\\", "/")
            rel_gt = str(gt_path).replace("\\", "/")
            manifest_rows.append(f"{rel_img},{rel_gt},{lang},synthetic,{deg}")
            item_idx += 1

    manifest_path = EVAL_DIR / "manifest.csv"
    manifest_path.write_text("\n".join(manifest_rows), encoding="utf-8")
    print(f"Generated {item_idx} synthetic evaluation pages in {EVAL_DIR}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    generate_dataset()
