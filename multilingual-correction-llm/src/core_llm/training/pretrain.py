import argparse
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core_llm.model.transformer import (
    TransformerConfig,
    CausalTransformerLM,
)

from src.core_llm.training.dataset import TextDataset

from src.core_llm.training.train_utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
)


def evaluate(model, loader, device):
    """
    Evaluate the model on validation data.

    Returns:
        Average validation loss.
    """

    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():

        for input_ids, targets in loader:

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            _, loss = model(
                input_ids,
                targets
            )

            total_loss += loss.item()
            total_batches += 1

    model.train()

    return total_loss / max(total_batches, 1)


def calculate_perplexity(loss):
    """
    Calculate perplexity from cross-entropy loss.

    Perplexity = exp(loss)
    """

    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def train(args):

    # --------------------------------------------------
    # 1. Set random seed
    # --------------------------------------------------

    set_seed(args.seed)

    # --------------------------------------------------
    # 2. Select device
    # --------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    # --------------------------------------------------
    # 3. Load training dataset
    # --------------------------------------------------

    train_dataset = TextDataset(
        text_file=args.train_file,
        tokenizer_file=args.tokenizer,
        seq_len=args.seq_len,
    )

    # --------------------------------------------------
    # 4. Load validation dataset
    # --------------------------------------------------

    val_dataset = TextDataset(
        text_file=args.val_file,
        tokenizer_file=args.tokenizer,
        seq_len=args.seq_len,
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    # --------------------------------------------------
    # 5. Create DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------
    # 6. Create Transformer configuration
    # --------------------------------------------------

    config = TransformerConfig(
        vocab_size=train_dataset.vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )

    # --------------------------------------------------
    # 7. Create model
    # --------------------------------------------------

    model = CausalTransformerLM(
        config
    ).to(device)

    # --------------------------------------------------
    # 7b. Optionally warm-start from an existing checkpoint.
    #     This is how domain-adaptive pretraining works: train the base
    #     model on general data first, then continue training ("fine-tune")
    #     that same architecture on domain-specific text by pointing
    #     --init-from at the base run's checkpoint.
    # --------------------------------------------------

    if args.init_from:
        print(f"Initializing weights from checkpoint: {args.init_from}")

        checkpoint = load_checkpoint(
            args.init_from,
            model,
            optimizer=None,
            device=device,
        )

        loaded_vocab = checkpoint["config"]["vocab_size"]

        if loaded_vocab != config.vocab_size:
            raise ValueError(
                f"Checkpoint vocab_size ({loaded_vocab}) does not match "
                f"the current tokenizer's vocab_size ({config.vocab_size}). "
                "Domain-adaptive pretraining must reuse the exact same "
                "tokenizer.json that trained the base checkpoint."
            )

    # --------------------------------------------------
    # 8. Create optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------
    # 9. Count parameters
    # --------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Total parameters:",
        total_parameters
    )

    print(
        "Trainable parameters:",
        trainable_parameters
    )

    # --------------------------------------------------
    # 10. Best validation loss
    # --------------------------------------------------

    best_val_loss = float("inf")

    # --------------------------------------------------
    # 11. Training loop
    # --------------------------------------------------

    for epoch in range(args.epochs):

        model.train()

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}"
        )

        running_loss = 0.0

        # --------------------------------------------------
        # 12. Process training batches
        # --------------------------------------------------

        for step, (input_ids, targets) in enumerate(
            progress
        ):

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Clear previous gradients
            optimizer.zero_grad(
                set_to_none=True
            )

            # Forward pass
            _, loss = model(
                input_ids,
                targets
            )

            # Backpropagation
            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            # Update model parameters
            optimizer.step()

            # Add current loss
            running_loss += loss.item()

            # Update progress bar
            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        # --------------------------------------------------
        # 13. Calculate average training loss
        # --------------------------------------------------

        train_loss = (
            running_loss / max(
                len(train_loader),
                1
            )
        )

        # --------------------------------------------------
        # 14. Evaluate on validation dataset
        # --------------------------------------------------

        val_loss = evaluate(
            model,
            val_loader,
            device,
        )

        # --------------------------------------------------
        # 15. Calculate validation perplexity
        # --------------------------------------------------

        val_perplexity = calculate_perplexity(
            val_loss
        )

        # --------------------------------------------------
        # 16. Print evaluation results
        # --------------------------------------------------

        print()

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_ppl={val_perplexity:.2f}"
        )

        # --------------------------------------------------
        # 17. Create checkpoint directory
        # --------------------------------------------------

        checkpoint_dir = Path(
            args.checkpoint_dir
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------------
        # 18. Save latest checkpoint
        # --------------------------------------------------

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model,
            optimizer,
            epoch,
            0,
            val_loss,
        )

        print(
            f"Saved latest checkpoint: "
            f"{checkpoint_dir / 'latest.pt'}"
        )

        # --------------------------------------------------
        # 19. Save best checkpoint
        # --------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                0,
                val_loss,
            )

            print(
                "Saved new best checkpoint."
            )

        print()


def load_yaml_defaults(model_config_path, training_config_path):
    """
    Read configs/model.yaml and configs/training.yaml (if present) and turn
    them into a dict of argparse defaults. CLI flags always override these,
    so `--d-model 512` still wins even if model.yaml says 256. This is what
    actually makes those two YAML files do something, instead of sitting in
    configs/ unused while pretrain.py hardcodes its own argparse defaults.
    """

    defaults = {}

    key_map = {
        # yaml key -> argparse dest
        "context_length": "seq_len",
        "embedding_dim": "d_model",
        "num_heads": "num_heads",
        "num_layers": "num_layers",
        "dropout": "dropout",
        "batch_size": "batch_size",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
    }

    for path in (model_config_path, training_config_path):
        path = Path(path)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for yaml_key, arg_dest in key_map.items():
            if yaml_key in raw:
                defaults[arg_dest] = raw[yaml_key]

    return defaults


def parse_args():

    # configs/model.yaml and configs/training.yaml, relative to project root
    root = Path(__file__).resolve().parents[3]
    yaml_defaults = load_yaml_defaults(
        root / "configs" / "model.yaml",
        root / "configs" / "training.yaml",
    )

    parser = argparse.ArgumentParser(
        description="Train a causal Transformer language model. "
        "Defaults come from configs/model.yaml and configs/training.yaml "
        "when present; any flag passed on the command line overrides them."
    )

    # --------------------------------------------------
    # Dataset arguments
    # --------------------------------------------------

    parser.add_argument(
        "--train-file",
        required=True,
        help="Path to training text file.",
    )

    parser.add_argument(
        "--val-file",
        required=True,
        help="Path to validation text file.",
    )

    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to tokenizer.json.",
    )

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------

    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/base",
        help="Directory for model checkpoints.",
    )

    parser.add_argument(
        "--init-from",
        default=None,
        help=(
            "Path to an existing checkpoint (.pt) to warm-start the model "
            "weights from before training. Used for domain-adaptive "
            "pretraining: pass the base model's best.pt here together with "
            "--train-file pointing at domain data."
        ),
    )

    # --------------------------------------------------
    # Model configuration
    # --------------------------------------------------

    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Maximum sequence length.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size.",
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Transformer embedding dimension.",
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads.",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of Transformer blocks.",
    )

    parser.add_argument(
        "--d-ff",
        type=int,
        default=1024,
        help="Feed-forward hidden dimension.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability.",
    )

    # --------------------------------------------------
    # Training configuration
    # --------------------------------------------------

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    # Apply configs/*.yaml values as defaults (CLI flags above still win,
    # since argparse only falls back to set_defaults() when a flag is absent
    # from sys.argv).
    parser.set_defaults(**yaml_defaults)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    train(args)
