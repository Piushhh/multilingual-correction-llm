import torch


def calculate_loss(model, input_ids, target_ids):
    output = model(
        input_ids,
        target_ids,
    )

    return output["loss"]


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        loss = calculate_loss(
            model,
            input_ids,
            target_ids,
        )

        loss.backward()

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

    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        loss = calculate_loss(
            model,
            input_ids,
            target_ids,
        )

        total_loss += loss.item()

    return total_loss / len(dataloader)