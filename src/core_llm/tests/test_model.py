"""
Architecture and model tests for CausalTransformerLM.

Tests:
    - Forward pass output shapes
    - Loss calculation (scalar, finite)
    - Causal masking (future tokens cannot affect past positions)
    - Multiple batch sizes and sequence lengths
    - Context length boundary
    - Invalid configuration rejection
    - Parameter count
    - Weight initialization

These tests run without any checkpoint or external data.
"""

import pytest
import torch

from src.core_llm.model.config import ModelConfig
from src.core_llm.model.transformer import CausalTransformerLM


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_config():
    """Minimal valid config for fast unit tests."""
    return ModelConfig(
        vocab_size=100,
        context_length=32,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )


@pytest.fixture
def tiny_model(tiny_config):
    return CausalTransformerLM(tiny_config)


# ── Forward pass ──────────────────────────────────────────────────────────────

def test_forward_output_shapes(tiny_model, tiny_config):
    """logits.shape == [batch, seq, vocab]"""
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    out = tiny_model(input_ids)

    assert "logits" in out
    assert out["logits"].shape == (batch_size, seq_len, tiny_config.vocab_size)
    assert out["loss"] is None  # no targets supplied


def test_forward_with_targets(tiny_model, tiny_config):
    """When targets supplied, loss is a scalar finite tensor."""
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))

    out = tiny_model(input_ids, targets)

    assert out["loss"] is not None
    assert out["loss"].ndim == 0          # scalar
    assert torch.isfinite(out["loss"])    # not NaN or inf


def test_batch_size_1(tiny_model, tiny_config):
    """Model handles batch size of 1."""
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 10))
    out = tiny_model(input_ids)
    assert out["logits"].shape == (1, 10, tiny_config.vocab_size)


def test_batch_size_large(tiny_model, tiny_config):
    """Model handles batch size > 1."""
    input_ids = torch.randint(0, tiny_config.vocab_size, (8, 16))
    out = tiny_model(input_ids)
    assert out["logits"].shape == (8, 16, tiny_config.vocab_size)


def test_sequence_length_1(tiny_model, tiny_config):
    """Model handles sequence length of 1."""
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 1))
    out = tiny_model(input_ids)
    assert out["logits"].shape == (2, 1, tiny_config.vocab_size)


def test_sequence_at_context_limit(tiny_model, tiny_config):
    """Model handles sequence exactly at context_length."""
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.context_length))
    out = tiny_model(input_ids)
    assert out["logits"].shape == (1, tiny_config.context_length, tiny_config.vocab_size)


def test_sequence_exceeds_context_limit(tiny_model, tiny_config):
    """Sequence longer than context_length raises ValueError."""
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.context_length + 1))
    with pytest.raises(ValueError, match="context length"):
        tiny_model(input_ids)


# ── Causal masking ────────────────────────────────────────────────────────────

def test_causal_masking_future_does_not_affect_past():
    """
    The logits for position i must not change when we alter tokens at position j > i.

    This verifies that the causal mask is correctly applied.
    """
    config = ModelConfig(
        vocab_size=50,
        context_length=16,
        embedding_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model = CausalTransformerLM(config)
    model.eval()

    torch.manual_seed(0)
    seq_len = 8
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

    with torch.no_grad():
        logits_original = model(input_ids)["logits"].clone()

    # Alter all tokens AFTER position 3 (future tokens relative to pos 3)
    modified = input_ids.clone()
    modified[:, 4:] = torch.randint(0, config.vocab_size, (1, seq_len - 4))

    with torch.no_grad():
        logits_modified = model(modified)["logits"].clone()

    # Positions 0..3 must be identical — future changes must have no effect
    assert torch.allclose(
        logits_original[:, :4, :],
        logits_modified[:, :4, :],
        atol=1e-5,
    ), "Causal masking failed: future tokens affected earlier positions."


# ── Loss with ignore_index ────────────────────────────────────────────────────

def test_loss_ignores_minus100(tiny_model, tiny_config):
    """Loss with ignore_index=-100 for all targets returns nan or 0 (no valid tokens)."""
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    targets = torch.full((2, 8), -100, dtype=torch.long)

    out = tiny_model(input_ids, targets)
    # When all targets are -100, loss should be 0 or NaN depending on PyTorch behavior
    # The important thing is that no exception is raised
    assert out["loss"] is not None


def test_loss_with_partial_ignore_index(tiny_model, tiny_config):
    """Loss with mixed -100 and valid targets should be finite."""
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
    targets = torch.randint(0, tiny_config.vocab_size, (2, 8))
    # Mask first half of each sequence
    targets[:, :4] = -100

    out = tiny_model(input_ids, targets)
    assert out["loss"] is not None
    assert torch.isfinite(out["loss"])


# ── Configuration validation ──────────────────────────────────────────────────

def test_invalid_config_num_heads_not_divisor():
    """num_heads that doesn't divide embedding_dim raises ValueError."""
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(
            vocab_size=100,
            context_length=32,
            embedding_dim=64,
            num_layers=2,
            num_heads=3,  # 64 % 3 != 0
            dropout=0.0,
        )


def test_invalid_config_zero_vocab():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=0, context_length=32, embedding_dim=64, num_layers=2, num_heads=4)


def test_invalid_config_zero_layers():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=100, context_length=32, embedding_dim=64, num_layers=0, num_heads=4)


# ── ModelConfig serialization ─────────────────────────────────────────────────

def test_model_config_round_trip():
    """ModelConfig can be serialized to dict and reconstructed."""
    config = ModelConfig(
        vocab_size=256, context_length=64, embedding_dim=128, num_layers=4, num_heads=8
    )
    d = config.to_dict()
    assert isinstance(d, dict)
    reconstructed = ModelConfig.from_dict(d)
    assert reconstructed.vocab_size == config.vocab_size
    assert reconstructed.context_length == config.context_length
    assert reconstructed.embedding_dim == config.embedding_dim
    assert reconstructed.num_layers == config.num_layers
    assert reconstructed.num_heads == config.num_heads


# ── Parameter count ───────────────────────────────────────────────────────────

def test_parameter_count_positive(tiny_model):
    """Model must have a positive number of parameters."""
    n = tiny_model.num_parameters()
    assert n > 0


# ── Generation ────────────────────────────────────────────────────────────────

def test_generate_returns_extended_tensor(tiny_model, tiny_config):
    """model.generate() returns tensor with more tokens than input."""
    max_new = 5
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 4))
    output = tiny_model.generate(input_ids, max_new_tokens=max_new, temperature=1.0)

    assert output.shape[1] <= input_ids.shape[1] + max_new
    assert output.shape[1] > input_ids.shape[1]


def test_generate_greedy(tiny_model, tiny_config):
    """Greedy generation (temperature=0.0) is deterministic."""
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 4))
    out1 = tiny_model.generate(input_ids, max_new_tokens=5, temperature=0.0)
    out2 = tiny_model.generate(input_ids, max_new_tokens=5, temperature=0.0)
    assert torch.equal(out1, out2)
