"""
Training utilities: one-epoch training loop, evaluation, perplexity.
"""

import math

import torch
import torch.nn.functional as F


def calculate_loss(model, input_ids, target_ids, ignore_index: int = -100):
    """
    Compute cross-entropy loss for a batch.

    Args:
        model:        CausalTransformerLM
        input_ids:    [batch, seq]
        target_ids:   [batch, seq] — use -100 at positions to ignore
        ignore_index: Index to ignore in loss (default -100)

    Returns:
        Scalar loss tensor.
    """
    output = model(input_ids)
    logits = output["logits"]

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=ignore_index,
    )


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model:         CausalTransformerLM
        dataloader:    DataLoader yielding (input_ids, target_ids) batches
        optimizer:     PyTorch optimizer
        device:        torch.device
        max_grad_norm: Gradient clipping norm (default 1.0)

    Returns:
        Average training loss over the epoch.
    """
    model.train()

    total_loss = 0.0
    total_batches = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        loss = calculate_loss(model, input_ids, target_ids)

        if torch.isnan(loss):
            raise RuntimeError(
                "Training loss is NaN. Check your data pipeline and learning rate."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        return float("inf")

    return total_loss / total_batches


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    ignore_index: int = -100,
) -> float:
    """
    Evaluate the model on a validation DataLoader.

    Args:
        model:        CausalTransformerLM
        dataloader:   DataLoader yielding (input_ids, target_ids) batches
        device:       torch.device
        ignore_index: Loss ignore index (default -100)

    Returns:
        Average validation loss. Returns inf if dataloader is empty.
    """
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        loss = calculate_loss(model, input_ids, target_ids, ignore_index=ignore_index)
        total_loss += loss.item()
        total_batches += 1

    model.train()

    if total_batches == 0:
        return float("inf")

    return total_loss / total_batches


def perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Clamps loss to 20 to prevent numerical overflow.
    """
    return math.exp(min(loss, 20.0))


def count_parameters(model) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
