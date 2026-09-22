import torch

from src.core_llm.model import (
    CausalTransformerLM,
    ModelConfig,
)


def test_model_forward():
    config = ModelConfig(
        vocab_size=100,
        context_length=32,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )

    model = CausalTransformerLM(config)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    output = model(
        input_ids,
        targets,
    )

    assert output["logits"].shape == (
        2,
        16,
        100,
    )

    assert output["loss"] is not None
    assert output["loss"].ndim == 0


def test_model_vocab_size():
    config = ModelConfig(
        vocab_size=256,
        context_length=32,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )

    model = CausalTransformerLM(config)

    input_ids = torch.randint(
        0,
        256,
        (2, 10),
    )

    output = model(input_ids)

    assert output["logits"].shape == (
        2,
        10,
        256,
    )