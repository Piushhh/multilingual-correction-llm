"""
Integration tests for CustomLLMAdapter and Member 1 model pipeline.

Tests:
    - CustomLLMAdapter satisfies ModelAdapter interface
    - is_baseline is False
    - is_loaded() returns False before load()
    - load() with missing checkpoint raises FileNotFoundError
    - load() with no custom_model_path raises RuntimeError
    - CorrectionEngine can be instantiated with CustomLLMAdapter
    - After successful load, generate() works and returns a string

These tests validate interface correctness, not model quality.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.core_llm.model.config import ModelConfig
from src.core_llm.model.transformer import CausalTransformerLM
from src.correction.model_adapter import CustomLLMAdapter, ModelAdapter, MockAdapter
from src.correction.inference import CorrectionEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def smoke_custom_env(tmp_path_factory):
    """
    Creates a tiny checkpoint and tokenizer for CustomLLMAdapter testing.
    """
    import sentencepiece as spm

    tmp = tmp_path_factory.mktemp("custom_adapter")

    corpus = tmp / "corpus.txt"
    corpus.write_text(
        "the quick brown fox jumps deep learning neural\n" * 20
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
        context_length=32,
        embedding_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model = CausalTransformerLM(config)
    model.eval()

    ckpt_path = tmp / "custom_test.pt"
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


# ── Interface compliance ──────────────────────────────────────────────────────

def test_custom_adapter_is_model_adapter():
    """CustomLLMAdapter must implement ModelAdapter interface."""
    adapter = CustomLLMAdapter({})
    assert isinstance(adapter, ModelAdapter)


def test_custom_adapter_is_not_baseline():
    """CustomLLMAdapter.is_baseline must be False."""
    adapter = CustomLLMAdapter({"custom_model_path": "some/path"})
    assert adapter.is_baseline is False


def test_custom_adapter_not_loaded_initially():
    """is_loaded() returns False before load() is called."""
    adapter = CustomLLMAdapter({"custom_model_path": "some/path"})
    assert adapter.is_loaded() is False


def test_custom_adapter_model_id_uses_path():
    """model_id reflects the configured path."""
    adapter = CustomLLMAdapter({"custom_model_path": "path/to/model.pt"})
    assert "path/to/model.pt" in adapter.model_id


def test_custom_adapter_model_id_default():
    """model_id has a fallback when no path is configured."""
    adapter = CustomLLMAdapter({})
    assert isinstance(adapter.model_id, str)
    assert len(adapter.model_id) > 0


# ── Load error cases ──────────────────────────────────────────────────────────

def test_custom_adapter_load_no_path_raises():
    """load() without custom_model_path raises RuntimeError."""
    adapter = CustomLLMAdapter({})
    with pytest.raises(RuntimeError, match="custom_model_path"):
        adapter.load()


def test_custom_adapter_load_missing_checkpoint():
    """load() with nonexistent checkpoint raises FileNotFoundError."""
    adapter = CustomLLMAdapter({"custom_model_path": "/nonexistent/model.pt"})
    with pytest.raises(FileNotFoundError):
        adapter.load()


def test_custom_adapter_generate_before_load_raises():
    """generate() before load() raises RuntimeError."""
    adapter = CustomLLMAdapter({"custom_model_path": "some/path"})
    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.generate("test prompt")


# ── Successful load and generate ──────────────────────────────────────────────

def test_custom_adapter_loads_successfully(smoke_custom_env):
    """CustomLLMAdapter.load() succeeds with a valid checkpoint."""
    adapter = CustomLLMAdapter({
        "custom_model_path": smoke_custom_env["checkpoint_path"],
        "tokenizer_path": smoke_custom_env["tokenizer_path"],
    })
    adapter.load()
    assert adapter.is_loaded() is True


def test_custom_adapter_generate_returns_string(smoke_custom_env):
    """generate() returns a str after successful load()."""
    adapter = CustomLLMAdapter({
        "custom_model_path": smoke_custom_env["checkpoint_path"],
        "tokenizer_path": smoke_custom_env["tokenizer_path"],
        "max_new_tokens": 5,
        "temperature": 0.0,
    })
    adapter.load()
    result = adapter.generate("the quick brown")
    assert isinstance(result, str)


def test_custom_adapter_generate_empty_prompt(smoke_custom_env):
    """generate() with empty prompt returns empty string."""
    adapter = CustomLLMAdapter({
        "custom_model_path": smoke_custom_env["checkpoint_path"],
        "tokenizer_path": smoke_custom_env["tokenizer_path"],
        "max_new_tokens": 5,
    })
    adapter.load()
    result = adapter.generate("")
    assert result == ""


# ── CorrectionEngine integration ──────────────────────────────────────────────

def test_correction_engine_with_custom_adapter(smoke_custom_env):
    """CorrectionEngine can be used with CustomLLMAdapter."""
    adapter = CustomLLMAdapter({
        "custom_model_path": smoke_custom_env["checkpoint_path"],
        "tokenizer_path": smoke_custom_env["tokenizer_path"],
        "max_new_tokens": 5,
        "temperature": 0.0,
    })
    adapter.load()

    engine = CorrectionEngine({}, adapter=adapter)
    engine.load_model()

    assert engine.is_baseline is False
    assert engine.adapter is adapter

    result = engine.correct({
        "text": "Deep learnig is good.",
        "language": "en",
        "domain": "deep_learning",
    })

    assert "corrected_text" in result
    assert "changes" in result
    assert "metadata" in result
    assert isinstance(result["corrected_text"], str)


def test_mock_adapter_is_still_baseline():
    """MockAdapter.is_baseline remains True (baseline mode preserved)."""
    adapter = MockAdapter()
    assert adapter.is_baseline is True
    assert adapter.is_loaded() is True
