"""
Tests for language detector determinism and repeatability (Task 5 / Problem H).
"""

from langdetect import DetectorFactory
from src.document_ai.language.detector import detect_language


def test_detector_factory_seed_is_zero():
    assert DetectorFactory.seed == 0


def test_repeated_calls_return_identical_results():
    test_samples = [
        "This is a sample sentence used to test deterministic detection.",
        "Short text maybe ambiguous.",
        "यह एक परीक्षण वाक्य है।",
        "Transformer model attention mechanism बहुत उपयोगी है।",
        "1234567890 !@#$%^&*()",
        "Machine learning models need clean data and consistent results.",
    ]

    for sample in test_samples:
        first_run = detect_language(sample)
        for _ in range(5):
            repeated_run = detect_language(sample)
            assert repeated_run == first_run, f"Non-deterministic result on: {sample}"
