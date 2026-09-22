"""
CLI to train the domain classifier and save it to disk.

Member 2 Milestone 4: "Domain classification baseline."

    python -m src.document_ai.domain.train_classifier \
        --data data/domain_classifier/train.csv \
        --output checkpoints/domain_classifier/model.joblib

CSV must have `text` and `domain` columns; `domain` values must be one of
src.document_ai.domain.classifier.DOMAINS.
"""

import argparse
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.document_ai.domain.classifier import DomainClassifier, load_labeled_csv


def train(args):
    texts, labels = load_labeled_csv(args.data, text_column="text", label_column="domain")

    print(f"Loaded {len(texts)} labeled examples from {args.data}")

    label_counts = {label: labels.count(label) for label in set(labels)}
    print("Class distribution:", label_counts)

    if len(texts) < 10:
        print(
            "WARNING: fewer than 10 training examples. This baseline will "
            "overfit badly -- treat any accuracy number as meaningless "
            "until the dataset grows (see docs on scaling in README)."
        )

    # Stratify keeps the (small, imbalanced) class proportions similar in
    # train/val; skip stratification gracefully if any class has too few
    # examples for it to be possible.
    try:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=args.val_split,
            random_state=args.seed,
            stratify=labels,
        )
    except ValueError:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=args.val_split, random_state=args.seed
        )

    classifier = DomainClassifier()
    classifier.fit(train_texts, train_labels)

    if val_texts:
        predictions = [classifier.predict(t)["domain"] for t in val_texts]
        print("\nValidation report:")
        print(classification_report(val_labels, predictions, zero_division=0))

    output_path = Path(args.output)
    classifier.save(output_path)
    print(f"Saved classifier to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the domain classifier.")
    parser.add_argument("--data", required=True, help="Path to labeled CSV (text, domain).")
    parser.add_argument(
        "--output",
        default="checkpoints/domain_classifier/model.joblib",
        help="Where to save the trained classifier.",
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
