"""
Stage 7: Checkpoint verification tests.

Tests:
  1. save_checkpoint -> LLM.from_checkpoint round-trip (tiny model)
  2. LLM.from_checkpoint raises a clear KeyError when 'config' key is missing
  3. CustomLLMAdapter raises FileNotFoundError for missing checkpoint
  4. CustomLLMAdapter raises FileNotFoundError for missing tokenizer
  5. CustomLLMAdapter raises KeyError for checkpoint missing 'config' key
  6. CustomLLMAdapter.load() succeeds on real checkpoints (integration)

Member 1 ownership — tests for src/core_llm/inference/generate.py
and src/integration/adapters/custom_llm_adapter.py.
"""

import os
import pytest
import torch
from pathlib import Path


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

REPO_ROOT = Path(__file__).parent.parent
REAL_CHECKPOINT = REPO_ROOT / "checkpoints" / "domain" / "best.pt"
REAL_TOKENIZER = REPO_ROOT / "data" / "tokenizer" / "tokenizer.json"


def _make_tiny_model():
    """Create the smallest valid TransformerConfig + LLM model for unit tests."""
    from src.core_llm.model.transformer import CausalTransformerLM, TransformerConfig

    cfg = TransformerConfig(
        vocab_size=100,
        max_seq_len=16,
        d_model=32,
        num_heads=2,
        num_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    model = CausalTransformerLM(cfg)
    return model, cfg


# ------------------------------------------------------------------ #
# Test 1: save_checkpoint -> from_checkpoint round-trip               #
# ------------------------------------------------------------------ #

def test_checkpoint_roundtrip(tmp_path):
    """
    Verify that save_checkpoint + LLM.from_checkpoint correctly restores
    the model configuration and state dict.
    """
    from src.core_llm.training.train_utils import save_checkpoint
    from src.core_llm.inference.generate import LLM

    model, cfg = _make_tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ckpt_dir = tmp_path / "checkpoint_roundtrip"
    ckpt_dir.mkdir()
    ckpt_path = str(ckpt_dir / "best.pt")
    tok_path = str(REAL_TOKENIZER)

    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        step=10,
        loss=1.234,
    )

    assert Path(ckpt_path).exists(), "Checkpoint file should have been created"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "config" in ckpt, "Checkpoint must contain 'config' key"
    assert ckpt["config"]["vocab_size"] == cfg.vocab_size
    assert ckpt["config"]["d_model"] == cfg.d_model
    assert ckpt["epoch"] == 1
    assert ckpt["step"] == 10
    assert abs(ckpt["loss"] - 1.234) < 1e-4

    # from_checkpoint reads 'config' key — this would fail if config were
    # saved wrong (the bug we are guarding against)
    reloaded = LLM.from_checkpoint(
        checkpoint_path=ckpt_path,
        tokenizer_path=tok_path,
    )
    assert reloaded.model.config.vocab_size == cfg.vocab_size
    assert reloaded.model.config.d_model == cfg.d_model
    assert reloaded.model.config.max_seq_len == cfg.max_seq_len


# ------------------------------------------------------------------ #
# Test 2: from_checkpoint raises KeyError on missing 'config' key     #
# ------------------------------------------------------------------ #

def test_from_checkpoint_missing_config_key_raises_clearly(tmp_path):
    """
    LLM.from_checkpoint must raise a KeyError (not AttributeError, not
    a cryptic CUDA error) when the checkpoint file is missing 'config'.
    """
    from src.core_llm.inference.generate import LLM

    # Create a checkpoint without 'config' (e.g., saved by old code)
    bad_ckpt_path = tmp_path / "bad.pt"
    torch.save({"epoch": 0, "step": 0, "loss": 9.9}, str(bad_ckpt_path))

    with pytest.raises((KeyError, RuntimeError)):
        LLM.from_checkpoint(
            checkpoint_path=str(bad_ckpt_path),
            tokenizer_path=str(REAL_TOKENIZER),
        )


# ------------------------------------------------------------------ #
# Test 3: CustomLLMAdapter raises FileNotFoundError for missing ckpt  #
# ------------------------------------------------------------------ #

def test_custom_llm_adapter_missing_checkpoint():
    from src.integration.adapters.custom_llm_adapter import CustomLLMAdapter

    adapter = CustomLLMAdapter(
        checkpoint_path="/nonexistent/path/checkpoint.pt",
        tokenizer_path=str(REAL_TOKENIZER),
    )
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        adapter.load()


# ------------------------------------------------------------------ #
# Test 4: CustomLLMAdapter raises FileNotFoundError for missing tok   #
# ------------------------------------------------------------------ #

def test_custom_llm_adapter_missing_tokenizer(tmp_path):
    from src.integration.adapters.custom_llm_adapter import CustomLLMAdapter

    # checkpoint exists but tokenizer doesn't
    fake_ckpt = tmp_path / "model.pt"
    torch.save({"config": {}, "model_state_dict": {}}, str(fake_ckpt))

    adapter = CustomLLMAdapter(
        checkpoint_path=str(fake_ckpt),
        tokenizer_path="/nonexistent/tokenizer.json",
    )
    with pytest.raises(FileNotFoundError, match="tokenizer not found"):
        adapter.load()


# ------------------------------------------------------------------ #
# Test 5: CustomLLMAdapter raises KeyError for missing 'config' key   #
# ------------------------------------------------------------------ #

def test_custom_llm_adapter_missing_config_in_checkpoint(tmp_path):
    from src.integration.adapters.custom_llm_adapter import CustomLLMAdapter

    bad_ckpt = tmp_path / "no_config.pt"
    torch.save({"epoch": 0}, str(bad_ckpt))

    adapter = CustomLLMAdapter(
        checkpoint_path=str(bad_ckpt),
        tokenizer_path=str(REAL_TOKENIZER),
    )
    with pytest.raises(KeyError, match="config"):
        adapter.load()


# ------------------------------------------------------------------ #
# Test 6: CustomLLMAdapter integration test with real checkpoint       #
# ------------------------------------------------------------------ #

@pytest.mark.skipif(
    not REAL_CHECKPOINT.exists() or not REAL_TOKENIZER.exists(),
    reason="Real checkpoint or tokenizer not present — skipped in CI",
)
def test_custom_llm_adapter_loads_real_checkpoint():
    from src.integration.adapters.custom_llm_adapter import CustomLLMAdapter

    adapter = CustomLLMAdapter(
        checkpoint_path=str(REAL_CHECKPOINT),
        tokenizer_path=str(REAL_TOKENIZER),
    )
    assert not adapter.is_loaded()
    adapter.load()
    assert adapter.is_loaded()
    assert "custom-llm" in adapter.model_id
    assert adapter.is_baseline is False


@pytest.mark.skipif(
    not REAL_CHECKPOINT.exists() or not REAL_TOKENIZER.exists(),
    reason="Real checkpoint or tokenizer not present — skipped in CI",
)
def test_custom_llm_adapter_generate_returns_continuation():
    """
    Generate with the real checkpoint. Verifies continuation-only output
    (i.e., does not start with the prompt text verbatim).
    """
    from src.integration.adapters.custom_llm_adapter import CustomLLMAdapter

    adapter = CustomLLMAdapter(
        checkpoint_path=str(REAL_CHECKPOINT),
        tokenizer_path=str(REAL_TOKENIZER),
    )
    adapter.load()

    prompt = "The model"
    output = adapter.generate(prompt, max_new_tokens=20, temperature=0.8)

    # Output should be a string (possibly empty if EOS triggered immediately)
    assert isinstance(output, str)
    # Output should NOT start with the full prompt (continuation only)
    assert not output.startswith(prompt), (
        f"Expected continuation only, got output starting with prompt: {repr(output[:50])}"
    )
