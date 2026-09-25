"""
Domain classification: Deep Learning, Computer Science, Mathematics, or
General text.

Member 2, Responsibility 4: "Build a classifier for Deep Learning, Computer
Science, Mathematics, and general text."

Approach: TF-IDF character/word n-grams + Logistic Regression, via
scikit-learn. This is the right complexity level for a small, fast-to-train
baseline that Member 1's LLM inference doesn't have to run just to answer
"what domain is this page" -- a linear classifier over TF-IDF features
trains in seconds on a few hundred labeled examples and is easy for the
whole team to inspect (top weighted terms per class are directly
interpretable), unlike a fine-tuned transformer for the same job.

Word-level AND character n-gram (3-5) TF-IDF are combined: word n-grams
capture domain vocabulary ("gradient", "eigenvalue"), character n-grams
give some robustness to OCR noise (a misspelled "grad1ent" from a bad OCR
read still shares character n-grams with "gradient").
"""

import json
from pathlib import Path

import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


DOMAINS = ["deep_learning", "computer_science", "mathematics", "general"]


class DomainClassifier:
    """
    Wraps a word-ngram TF-IDF vectorizer + char-ngram TF-IDF vectorizer +
    LogisticRegression into one object with a simple predict() interface,
    and handles saving/loading so training (train_classifier.py) and
    inference (this class, used by pipeline.py) stay decoupled.
    """

    def __init__(self):
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )
        self._fitted = False

    def _features(self, texts, fit=False):
        if fit:
            word_features = self.word_vectorizer.fit_transform(texts)
            char_features = self.char_vectorizer.fit_transform(texts)
        else:
            word_features = self.word_vectorizer.transform(texts)
            char_features = self.char_vectorizer.transform(texts)
        return hstack([word_features, char_features])

    def fit(self, texts, labels):
        """
        texts: list[str], labels: list[str] (each in DOMAINS).

        Raises ValueError early if a label outside DOMAINS shows up --
        better than silently training a classifier that can never predict
        the class you actually care about.
        """
        unknown = set(labels) - set(DOMAINS)
        if unknown:
            raise ValueError(
                f"Labels {unknown} are not in the agreed DOMAINS list {DOMAINS}. "
                "Fix the training data or extend DOMAINS (and tell the team, "
                "since Member 1/3 depend on this exact label set)."
            )

        features = self._features(texts, fit=True)
        self.classifier.fit(features, labels)
        self._fitted = True
        return self

    def predict(self, text):
        """
        Returns {"domain": str, "confidence": float, "scores": {domain: prob}}.

        `confidence` is the top class's predicted probability, so a
        near-uniform distribution across domains (i.e. the model genuinely
        isn't sure) shows up as low confidence rather than a falsely
        authoritative single label.
        """
        if not self._fitted:
            raise RuntimeError(
                "DomainClassifier has not been fit or loaded. Call .fit(...) "
                "or DomainClassifier.load(path) first."
            )

        features = self._features([text], fit=False)
        probabilities = self.classifier.predict_proba(features)[0]

        scores = {
            str(label): round(float(prob), 4)
            for label, prob in zip(self.classifier.classes_, probabilities)
        }

        best_domain = max(scores, key=scores.get)

        return {
            "domain": best_domain,
            "confidence": scores[best_domain],
            "scores": scores,
        }

    def save(self, path, metadata=None):
        import sklearn

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "word_vectorizer": self.word_vectorizer,
            "char_vectorizer": self.char_vectorizer,
            "classifier": self.classifier,
            "fitted": self._fitted,
            "metadata": {
                "sklearn_version": sklearn.__version__,
                "domains": DOMAINS,
                **(metadata or {}),
            },
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path):
        state = joblib.load(Path(path))
        instance = cls()
        instance.word_vectorizer = state["word_vectorizer"]
        instance.char_vectorizer = state["char_vectorizer"]
        instance.classifier = state["classifier"]
        instance._fitted = state["fitted"]
        instance.metadata = state.get("metadata", {})
        return instance


def load_labeled_csv(csv_path, text_column="text", label_column="domain"):
    """
    Load a training set from CSV with at least `text_column` and
    `label_column`. Kept as a plain function (not baked into
    DomainClassifier) so train_classifier.py can support other data
    sources later without touching this class.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    for col in (text_column, label_column):
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' in {csv_path}, found: {list(df.columns)}"
            )

    df = df.dropna(subset=[text_column, label_column])

    return df[text_column].tolist(), df[label_column].tolist()


def load_labeled_jsonl(jsonl_path, text_field="text", label_field="domain"):
    """Alternative loader for JSONL: one {"text": ..., "domain": ...} per line."""
    texts, labels = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            texts.append(record[text_field])
            labels.append(record[label_field])
    return texts, labels
