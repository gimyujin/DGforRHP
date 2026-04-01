from __future__ import annotations

from typing import Dict
from tqdm import tqdm
import torch
import torch.nn as nn


def train_one_epoch_amp(
    model,
    loader,
    optimizer,
    device,
    scaler_amp: torch.cuda.amp.GradScaler,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    total = 0
    running_loss = 0.0
    correct = 0

    use_amp = scaler_amp is not None and scaler_amp.is_enabled()

    for batch in tqdm(loader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ✅ PyTorch 2.5+ 표준 autocast (경고 제거)
        if device.type == "cuda":
            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)
        else:
            autocast_ctx = torch.amp.autocast("cpu", enabled=False)

        with autocast_ctx:
            if "num" in batch:
                num = batch["num"].to(device, non_blocking=True)
                logits = model(input_ids=input_ids, attention_mask=attention_mask, num=num)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = loss_fn(logits, labels)

        if use_amp:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        bs = labels.size(0)
        total += bs
        running_loss += loss.item() * bs
        preds = torch.argmax(logits.detach(), dim=1)
        correct += (preds == labels).sum().item()

    return {
        "loss": running_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }
