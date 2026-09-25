"""
Correction inference tests.

Tests:
    - CorrectionInference loads from a checkpoint
    - correct() returns a string
    - Empty input returns ''
    - Missing checkpoint raises FileNotFoundError with actionable message
    - Interface is callable without CLI arguments

NOTE: These tests validate the INTERFACE structure only.
A randomly initialized model will NOT produce meaningful corrections.
Actual correction quality requires proper training and evaluation.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.core_llm.model.config import ModelConfig
from src.core_llm.model.transformer import CausalTransformerLM
from src.core_llm.inference.correct import CorrectionInference


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def smoke_correction_env(tmp_path_factory):
    """
    Creates a tiny checkpoint for correction interface testing.
    Model is randomly initialized — output will be meaningless.
    """
    import sentencepiece as spm

    tmp = tmp_path_factory.mktemp("correction")

    corpus = tmp / "corpus.txt"
    corpus.write_text(
        "correct deep learning neural network gradient\n" * 20
        + "answer backpropagation weight update\n" * 20
        + "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e\n" * 10,
        encoding="utf-8",
    )
    prefix = str(tmp / "tok")
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=prefix,
        vocab_size=128,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
    )
    tokenizer_path = prefix + ".model"

    config = ModelConfig(
        vocab_size=128,
        context_length=64,
        embedding_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model = CausalTransformerLM(config)
    model.eval()

    ckpt_path = tmp / "smoke_correction.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "tokenizer_path": tokenizer_path,
            "stage": "correction_finetune",
        },
        ckpt_path,
    )

    return {
        "checkpoint_path": str(ckpt_path),
        "tokenizer_path": tokenizer_path,
    }


@pytest.fixture(scope="module")
def corrector(smoke_correction_env):
    return CorrectionInference.from_checkpoint(
        checkpoint_path=smoke_correction_env["checkpoint_path"],
        tokenizer_path=smoke_correction_env["tokenizer_path"],
    )


# ── Loading ───────────────────────────────────────────────────────────────────

def test_correction_inference_loads(smoke_correction_env):
    """CorrectionInference.from_checkpoint() succeeds."""
    corrector = CorrectionInference.from_checkpoint(
        checkpoint_path=smoke_correction_env["checkpoint_path"],
        tokenizer_path=smoke_correction_env["tokenizer_path"],
    )
    assert corrector is not None


def test_correction_no_checkpoint_raises(tmp_path):
    """Missing checkpoint raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        CorrectionInference.from_checkpoint(
            checkpoint_path=str(tmp_path / "nonexistent.pt"),
            tokenizer_path="any_path",
        )


# ── Correct interface ─────────────────────────────────────────────────────────

def test_correct_returns_string(corrector):
    """correct() returns a str regardless of content."""
    result = corrector.correct("Deep learnig is a subfeld.", max_new_tokens=10)
    assert isinstance(result, str)


def test_correct_empty_input(corrector):
    """correct() on empty string returns ''."""
    result = corrector.correct("", max_new_tokens=10)
    assert result == ""


def test_correct_english_text(corrector):
    """correct() accepts English input without error."""
    result = corrector.correct("The model use attention.", max_new_tokens=10)
    assert isinstance(result, str)


def test_correct_hindi_text(corrector):
    """correct() accepts Hindi input without error."""
    text = "\u0921\u0940\u092a \u0932\u0930\u094d\u0928\u093f\u0902\u0917"
    result = corrector.correct(text, max_new_tokens=10)
    assert isinstance(result, str)


def test_correct_greedy_deterministic(corrector):
    """temperature=0.0 gives deterministic output."""
    text = "neural netwroks"
    r1 = corrector.correct(text, max_new_tokens=5, temperature=0.0)
    r2 = corrector.correct(text, max_new_tokens=5, temperature=0.0)
    assert r1 == r2


def test_correct_accesses_llm(corrector):
    """corrector.llm exposes the underlying LLMInference."""
    from src.core_llm.inference.generate import LLMInference
    assert isinstance(corrector.llm, LLMInference)
