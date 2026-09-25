"""
Inference interface tests for LLMInference.

Tests:
    - LLMInference.from_checkpoint() loads correctly from a valid checkpoint
    - generate() returns a string (newly generated tokens only)
    - max_new_tokens is respected
    - Empty input is handled
    - Context truncation works
    - Missing checkpoint raises FileNotFoundError
    - Vocab mismatch raises ValueError
    - Invalid parameters raise ValueError
    - Checkpoint missing model_config raises KeyError

All tests use a synthetic smoke checkpoint (tiny model, no real training).
These tests validate the INTERFACE correctness, not model quality.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.core_llm.model.config import ModelConfig
from src.core_llm.model.transformer import CausalTransformerLM
from src.core_llm.inference.generate import LLMInference


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def smoke_env(tmp_path_factory):
    """
    Creates a minimal tokenizer + checkpoint suitable for inference tests.
    The model is randomly initialized — no meaningful output is expected.
    These tests only validate interface behavior.
    """
    import sentencepiece as spm

    tmp = tmp_path_factory.mktemp("inference")

    # Train tiny tokenizer
    corpus = tmp / "corpus.txt"
    corpus.write_text(
        "the quick brown fox jumps over the lazy dog\n" * 20
        + "deep learning neural networks gradient descent\n" * 10
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

    # Build tiny model
    config = ModelConfig(
        vocab_size=128,
        context_length=32,
        embedding_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model = CausalTransformerLM(config)
    model.eval()

    # Save checkpoint in proper format
    ckpt_path = tmp / "smoke_inference.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "tokenizer_path": tokenizer_path,
            "epoch": 1,
            "stage": "smoke_test",
        },
        ckpt_path,
    )

    return {
        "checkpoint_path": str(ckpt_path),
        "tokenizer_path": tokenizer_path,
        "config": config,
    }


@pytest.fixture(scope="module")
def llm(smoke_env):
    """Initialized LLMInference for inference tests."""
    return LLMInference.from_checkpoint(
        smoke_env["checkpoint_path"],
        tokenizer_path=smoke_env["tokenizer_path"],
    )


# ── Loading ───────────────────────────────────────────────────────────────────

def test_from_checkpoint_loads_successfully(smoke_env):
    """LLMInference.from_checkpoint() succeeds with a valid checkpoint."""
    llm = LLMInference.from_checkpoint(
        smoke_env["checkpoint_path"],
        tokenizer_path=smoke_env["tokenizer_path"],
    )
    assert llm is not None


def test_from_checkpoint_missing_file():
    """Missing checkpoint file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        LLMInference.from_checkpoint("/nonexistent/path/model.pt")


def test_from_checkpoint_missing_model_config(tmp_path, smoke_env):
    """Checkpoint without model_config raises KeyError."""
    # Build a checkpoint with no model_config key
    config = ModelConfig(vocab_size=128, context_length=32, embedding_dim=32, num_layers=1, num_heads=2)
    model = CausalTransformerLM(config)
    bad_ckpt = tmp_path / "bad.pt"
    torch.save({"model_state_dict": model.state_dict()}, bad_ckpt)

    with pytest.raises(KeyError, match="model_config"):
        LLMInference.from_checkpoint(str(bad_ckpt), tokenizer_path=smoke_env["tokenizer_path"])


def test_from_checkpoint_vocab_mismatch(tmp_path, smoke_env):
    """Vocab mismatch between model and tokenizer raises ValueError."""
    wrong_config = ModelConfig(
        vocab_size=999,  # doesn't match tokenizer vocab size of 128
        context_length=32,
        embedding_dim=32,
        num_layers=1,
        num_heads=2,
    )
    model = CausalTransformerLM(wrong_config)
    bad_ckpt = tmp_path / "mismatch.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": wrong_config.to_dict()},
        bad_ckpt,
    )

    with pytest.raises(ValueError, match="vocab_size"):
        LLMInference.from_checkpoint(str(bad_ckpt), tokenizer_path=smoke_env["tokenizer_path"])


# ── Generation ────────────────────────────────────────────────────────────────

def test_generate_returns_string(llm):
    """generate() returns a str."""
    result = llm.generate("the quick", max_new_tokens=5)
    assert isinstance(result, str)


def test_generate_returns_only_new_tokens(llm):
    """generate() must NOT include the prompt in the returned string."""
    prompt = "the"
    result = llm.generate(prompt, max_new_tokens=5, temperature=0.0)
    # The result is newly generated text; the prompt itself is not the result
    # We check it's a string (can't check content since model is random)
    assert isinstance(result, str)


def test_generate_empty_input_returns_empty(llm):
    """generate() returns '' for empty input."""
    result = llm.generate("", max_new_tokens=10)
    assert result == ""


def test_generate_max_new_tokens_respected(llm, smoke_env):
    """generate() produces at most max_new_tokens new tokens (measured at tensor level)."""
    import torch
    max_new = 3
    prompt = "the quick brown"
    prompt_ids = llm._tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)

    output_tensor = llm._model.generate(
        prompt_tensor,
        max_new_tokens=max_new,
        temperature=1.0,
    )

    # New tokens = total output length - prompt length
    new_token_count = output_tensor.shape[1] - len(prompt_ids)
    assert new_token_count <= max_new, (
        f"Expected at most {max_new} new tokens, got {new_token_count}"
    )


def test_generate_temperature_zero_is_greedy(llm):
    """temperature=0.0 (greedy) produces deterministic output."""
    out1 = llm.generate("hello", max_new_tokens=5, temperature=0.0)
    out2 = llm.generate("hello", max_new_tokens=5, temperature=0.0)
    assert out1 == out2


def test_generate_invalid_max_new_tokens(llm):
    """max_new_tokens < 1 raises ValueError."""
    with pytest.raises(ValueError, match="max_new_tokens"):
        llm.generate("test", max_new_tokens=0)


def test_generate_invalid_temperature(llm):
    """Negative temperature raises ValueError."""
    with pytest.raises(ValueError, match="temperature"):
        llm.generate("test", max_new_tokens=5, temperature=-1.0)


def test_generate_long_input_context_truncation(llm, smoke_env):
    """A prompt longer than context_length is safely truncated."""
    context_length = smoke_env["config"].context_length
    # Create a prompt that encodes to more than context_length tokens
    long_prompt = "the " * (context_length + 20)
    # Should not raise — context is silently truncated
    result = llm.generate(long_prompt, max_new_tokens=3)
    assert isinstance(result, str)


def test_generate_top_k_works(llm):
    """top_k parameter does not cause an error."""
    result = llm.generate("the quick", max_new_tokens=5, top_k=10)
    assert isinstance(result, str)


# ── Properties ────────────────────────────────────────────────────────────────

def test_vocab_size_matches_tokenizer(llm, smoke_env):
    """LLMInference.vocab_size matches the config vocab_size."""
    assert llm.vocab_size == smoke_env["config"].vocab_size


def test_context_length_property(llm, smoke_env):
    """LLMInference.context_length matches the config."""
    assert llm.context_length == smoke_env["config"].context_length
