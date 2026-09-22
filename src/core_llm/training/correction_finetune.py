import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from src.core_llm.model.config import ModelConfig
from src.core_llm.model.transformer import CausalTransformerLM
from src.core_llm.tokenizer.bpe_tokenizer import BPETokenizer


CORRECTION_DATA_PATH = "data/correction/correction_pairs.jsonl"

BASE_CHECKPOINT_PATH = (
    "src/core_llm/checkpoints/domain_adapted_model.pt"
)

CORRECTION_CHECKPOINT_PATH = (
    "src/core_llm/checkpoints/correction_model.pt"
)

TOKENIZER_PATH = (
    "src/core_llm/tokenizer/tokenizer.model"
)


# Correction model gets a larger context window.
CONTEXT_LENGTH = 64

BATCH_SIZE = 2
LEARNING_RATE = 1e-5
EPOCHS = 15

VALIDATION_RATIO = 0.2
PATIENCE = 4


class CorrectionDataset(Dataset):

    def __init__(self, examples, tokenizer, context_length):

        self.examples = examples
        self.tokenizer = tokenizer
        self.context_length = context_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        example = self.examples[index]

        # Keep the prompt compact so more space is available
        # for the corrected answer.
        prompt = (
            "Correct:\n"
            + example["incorrect"]
            + "\nAnswer:\n"
        )

        correct_text = example["correct"]

        prompt_ids = self.tokenizer.encode(prompt)
        correct_ids = self.tokenizer.encode(correct_text)

        # Reserve one token for the first prediction.
        max_total_tokens = self.context_length

        # Keep as much of the correction as possible.
        available_for_prompt = (
            max_total_tokens - len(correct_ids)
        )

        if available_for_prompt < 2:

            correct_ids = correct_ids[
                : max_total_tokens // 2
            ]

            available_for_prompt = (
                max_total_tokens - len(correct_ids)
            )

        prompt_ids = prompt_ids[:available_for_prompt]

        token_ids = prompt_ids + correct_ids

        token_ids = token_ids[:max_total_tokens]

        input_ids = token_ids[:-1]
        labels = token_ids[1:]

        # Ignore the prompt tokens when calculating loss.
        #
        # The model should learn:
        #
        # incorrect text -> corrected text
        #
        # rather than simply learning to reproduce
        # the prompt.

        prompt_length = max(
            0,
            min(
                len(prompt_ids),
                len(labels),
            ),
        )

        labels = (
            [-100] * prompt_length
            + labels[prompt_length:]
        )

        # Pad input and labels.
        input_padding = (
            self.context_length - 1 - len(input_ids)
        )

        label_padding = (
            self.context_length - 1 - len(labels)
        )

        input_ids = (
            input_ids
            + [0] * input_padding
        )

        labels = (
            labels
            + [-100] * label_padding
        )

        return (
            torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            torch.tensor(
                labels,
                dtype=torch.long,
            ),
        )


def load_examples(path):

    examples = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if line.strip():

                examples.append(
                    json.loads(line)
                )

    return examples


def load_domain_model(
    tokenizer,
    device,
):

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LENGTH,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
    )

    model = CausalTransformerLM(config)

    checkpoint = torch.load(
        BASE_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint["model_state_dict"]

    # The domain model was trained with context length 32.
    # The correction model uses 64.
    #
    # Therefore we load all compatible parameters and
    # initialize the additional positional embeddings.

    current_state = model.state_dict()

    compatible_state = {}

    for name, value in state_dict.items():

        if (
            name in current_state
            and current_state[name].shape
            == value.shape
        ):

            compatible_state[name] = value

    model.load_state_dict(
        compatible_state,
        strict=False,
    )

    # Copy the existing 32 positional embeddings into
    # the first 32 positions of the 64-position model.
    position_name = "position_embedding"

    if position_name in state_dict:

        old_position = state_dict[
            position_name
        ]

        new_position = model.state_dict()[
            position_name
        ]

        copy_length = min(
            old_position.shape[0],
            new_position.shape[0],
        )

        with torch.no_grad():

            new_position[
                :copy_length
            ].copy_(
                old_position[
                    :copy_length
                ]
            )

            if new_position.shape[0] > copy_length:

                new_position[
                    copy_length:
                ].copy_(
                    old_position[
                        :1
                    ].expand(
                        new_position.shape[0]
                        - copy_length,
                        -1,
                    )
                )

    model.to(device)

    return model, config


def calculate_loss(
    model,
    input_ids,
    labels,
):

    output = model(input_ids)

    logits = output["logits"]

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )

    return loss


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
):

    model.train()

    total_loss = 0.0

    for input_ids, labels in dataloader:

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        loss = calculate_loss(
            model,
            input_ids,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
):

    model.eval()

    total_loss = 0.0

    for input_ids, labels in dataloader:

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        loss = calculate_loss(
            model,
            input_ids,
            labels,
        )

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    tokenizer = BPETokenizer(
        TOKENIZER_PATH
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size,
    )

    examples = load_examples(
        CORRECTION_DATA_PATH
    )

    print(
        "Total correction examples:",
        len(examples),
    )

    english_count = sum(
        1
        for x in examples
        if x["language"] == "en"
    )

    hindi_count = sum(
        1
        for x in examples
        if x["language"] == "hi"
    )

    print(
        "English examples:",
        english_count,
    )

    print(
        "Hindi examples:",
        hindi_count,
    )

    full_dataset = CorrectionDataset(
        examples,
        tokenizer,
        CONTEXT_LENGTH,
    )

    validation_size = max(
        1,
        int(
            len(full_dataset)
            * VALIDATION_RATIO
        ),
    )

    training_size = (
        len(full_dataset)
        - validation_size
    )

    train_dataset, validation_dataset = (
        random_split(
            full_dataset,
            [
                training_size,
                validation_size,
            ],
            generator=torch.Generator().manual_seed(42),
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(validation_dataset),
    )

    model, config = load_domain_model(
        tokenizer,
        device,
    )

    print(
        "Loaded domain-adapted checkpoint:",
        BASE_CHECKPOINT_PATH,
    )

    print(
        "Correction context length:",
        CONTEXT_LENGTH,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    best_validation_loss = float("inf")

    best_epoch = 0

    patience_counter = 0

    for epoch in range(EPOCHS):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        validation_loss = evaluate(
            model,
            validation_loader,
            device,
        )

        validation_perplexity = math.exp(
            min(
                validation_loss,
                20,
            )
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Validation Loss: "
            f"{validation_loss:.4f} "
            f"- Validation Perplexity: "
            f"{validation_perplexity:.4f}"
        )

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch + 1

            patience_counter = 0

            Path(
                CORRECTION_CHECKPOINT_PATH
            ).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "config": config,

                    "tokenizer_path":
                        TOKENIZER_PATH,

                    "base_checkpoint":
                        BASE_CHECKPOINT_PATH,

                    "context_length":
                        CONTEXT_LENGTH,

                    "validation_loss":
                        validation_loss,

                    "validation_perplexity":
                        validation_perplexity,

                    "epoch":
                        best_epoch,
                },
                CORRECTION_CHECKPOINT_PATH,
            )

            print(
                "  ✓ Best correction model saved "
                f"(validation loss: "
                f"{best_validation_loss:.4f})"
            )

        else:

            patience_counter += 1

            print(
                f"  No improvement "
                f"({patience_counter}/"
                f"{PATIENCE})"
            )

        if (
            patience_counter
            >= PATIENCE
        ):

            print()

            print(
                "Early stopping triggered."
            )

            break

    print()

    print(
        "Correction fine-tuning complete."
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation loss:",
        f"{best_validation_loss:.4f}",
    )

    print(
        "Best validation perplexity:",
        f"{math.exp(min(best_validation_loss, 20)):.4f}",
    )

    print()

    print(
        "Correction checkpoint saved:"
    )

    print(
        CORRECTION_CHECKPOINT_PATH
    )


if __name__ == "__main__":
    main()