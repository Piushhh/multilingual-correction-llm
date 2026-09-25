"""
Training pipeline tests.

Tests:
    - LanguageModelDataset: next-token shift, length, batching
    - Dataset rejects corpus too small for context_length
    - Dataset reproducibility (deterministic split)
    - Smoke training: one tiny end-to-end train loop produces finite loss
      and saves a valid checkpoint
    - Checkpoint reload: saved checkpoint can be loaded and run inference

These tests run entirely without external data or checkpoints.
The smoke-training corpus is a TEST-ONLY synthetic dataset — it is NOT
a multilingual pretraining corpus and produces no meaningful model quality.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, random_split

from src.core_llm.model.config import ModelConfig
from src.core_llm.model.transformer import CausalTransformerLM
from src.core_llm.tokenizer.bpe_tokenizer import BPETokenizer
from src.core_llm.training.dataset import (
    LanguageModelDataset,
    load_corpus_tokens,
    create_dataloader,
)
from src.core_llm.training.train_utils import (
    train_one_epoch,
    evaluate,
    perplexity,
    count_parameters,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def smoke_tokenizer(tmp_path_factory):
    """
    Minimal BPETokenizer trained on a small synthetic corpus.
    SMOKE TEST ONLY — not representative of actual multilingual data.
    """
    import sentencepiece as spm

    tmp = tmp_path_factory.mktemp("tokenizer")
    corpus = tmp / "corpus.txt"
    corpus.write_text(
        "the quick brown fox jumps over the lazy dog\n" * 20
        + "neural network deep learning gradient\n" * 20
        + "नमस्ते दुनिया मशीन लर्निंग\n" * 10,
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
    return BPETokenizer(prefix + ".model")


@pytest.fixture(scope="module")
def smoke_corpus_path(tmp_path_factory):
    """
    Tiny synthetic corpus file for smoke training.
    TEST/SMOKE DATASET ONLY.
    """
    tmp = tmp_path_factory.mktemp("data")
    corpus = tmp / "corpus.txt"
    # Repeat enough to get > context_length tokens
    corpus.write_text(
        "the quick brown fox jumps over the lazy dog " * 50,
        encoding="utf-8",
    )
    return corpus


# ── LanguageModelDataset ──────────────────────────────────────────────────────

def test_dataset_next_token_shift(smoke_tokenizer):
    """input_ids[i] and target_ids[i] are shifted by 1."""
    token_ids = list(range(50))
    ctx = 10
    dataset = LanguageModelDataset(token_ids, ctx)

    inp, tgt = dataset[0]
    assert inp.tolist() == list(range(0, 10))
    assert tgt.tolist() == list(range(1, 11))


def test_dataset_length(smoke_tokenizer):
    """Dataset length is len(tokens) - context_length."""
    token_ids = list(range(100))
    ctx = 20
    dataset = LanguageModelDataset(token_ids, ctx)
    assert len(dataset) == 80


def test_dataset_batching(smoke_tokenizer):
    """DataLoader produces correctly shaped batches."""
    token_ids = list(range(200))
    ctx = 16
    dataset = LanguageModelDataset(token_ids, ctx)
    loader = create_dataloader(dataset, batch_size=4, shuffle=False)

    batch = next(iter(loader))
    inp, tgt = batch
    assert inp.shape == (4, 16)
    assert tgt.shape == (4, 16)


def test_dataset_rejects_too_small_corpus():
    """Corpus smaller than context_length + 1 raises ValueError."""
    with pytest.raises(ValueError, match="tokens"):
        LanguageModelDataset(list(range(10)), context_length=10)


def test_load_corpus_tokens_missing_file(tmp_path, smoke_tokenizer):
    """Missing corpus file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_corpus_tokens(str(tmp_path / "nonexistent.txt"), smoke_tokenizer)


def test_load_corpus_tokens_empty_file(tmp_path, smoke_tokenizer):
    """Empty corpus file raises ValueError."""
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_corpus_tokens(str(empty), smoke_tokenizer)


def test_deterministic_split():
    """random_split with the same seed produces the same split every time."""
    token_ids = list(range(500))
    dataset = LanguageModelDataset(token_ids, context_length=10)
    n = len(dataset)
    train_n = int(0.9 * n)
    val_n = n - train_n

    split1 = random_split(
        dataset, [train_n, val_n], generator=torch.Generator().manual_seed(42)
    )
    split2 = random_split(
        dataset, [train_n, val_n], generator=torch.Generator().manual_seed(42)
    )

    assert split1[0].indices == split2[0].indices


# ── Smoke training ────────────────────────────────────────────────────────────

def test_smoke_training_produces_finite_loss_and_checkpoint(
    smoke_tokenizer, smoke_corpus_path, tmp_path
):
    """
    SMOKE TEST (TEST/SMOKE DATASET ONLY):
    One tiny training run should produce a finite loss and save a valid checkpoint.
    This does NOT validate model quality.
    """
    token_ids = load_corpus_tokens(str(smoke_corpus_path), smoke_tokenizer)
    assert len(token_ids) > 32, "Smoke corpus must have more than 32 tokens"

    context_length = 16
    dataset = LanguageModelDataset(token_ids, context_length)

    train_n = max(1, int(0.8 * len(dataset)))
    val_n = len(dataset) - train_n
    if val_n == 0:
        train_n -= 1
        val_n = 1

    train_ds, val_ds = random_split(
        dataset, [train_n, val_n], generator=torch.Generator().manual_seed(42)
    )

    train_loader = create_dataloader(train_ds, batch_size=2, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=2, shuffle=False)

    config = ModelConfig(
        vocab_size=smoke_tokenizer.vocab_size,
        context_length=context_length,
        embedding_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )

    device = torch.device("cpu")
    model = CausalTransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss = evaluate(model, val_loader, device)
    ppl = perplexity(val_loss)

    assert math.isfinite(train_loss), f"Train loss is not finite: {train_loss}"
    assert math.isfinite(val_loss), f"Val loss is not finite: {val_loss}"
    assert math.isfinite(ppl), f"Perplexity is not finite: {ppl}"

    # Save checkpoint
    ckpt_path = tmp_path / "smoke_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": config.to_dict(),
            "tokenizer_path": "smoke_test",
            "epoch": 1,
            "global_step": len(train_loader),
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_perplexity": ppl,
            "seed": 42,
            "stage": "smoke_test",
        },
        ckpt_path,
    )

    assert ckpt_path.is_file(), "Checkpoint file was not created"

    # Reload checkpoint
    loaded = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    assert "model_state_dict" in loaded
    assert "model_config" in loaded
    assert isinstance(loaded["model_config"], dict)

    # Reconstruct and verify inference works
    reloaded_config = ModelConfig.from_dict(loaded["model_config"])
    reloaded_model = CausalTransformerLM(reloaded_config).to(device)
    reloaded_model.load_state_dict(loaded["model_state_dict"])
    reloaded_model.eval()

    dummy_input = torch.randint(0, config.vocab_size, (1, 8))
    out = reloaded_model(dummy_input)
    assert out["logits"].shape == (1, 8, config.vocab_size)


# ── CorrectionDataset ─────────────────────────────────────────────────────────

def test_correction_dataset_ignore_index(smoke_tokenizer, tmp_path):
    """
    CorrectionDataset must use -100 on prompt positions.
    """
    from src.core_llm.training.correction_finetune import CorrectionDataset

    examples = [
        {"incorrect": "hello wrold", "correct": "hello world", "language": "en"},
    ]
    dataset = CorrectionDataset(examples, smoke_tokenizer, context_length=32)
    inp, labels = dataset[0]

    assert inp.dtype == torch.long
    assert labels.dtype == torch.long
    # At least some positions must be -100 (prompt tokens)
    assert (labels == -100).any(), "No prompt tokens were masked with -100"
    # At least some positions must be valid (correction tokens)
    assert (labels != -100).any(), "All tokens were masked — correction tokens lost"
