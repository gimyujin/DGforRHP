# src/groupdro/train.py
from __future__ import annotations

from typing import Dict
from tqdm import tqdm
import torch
import torch.nn as nn


def _q_entropy(q: torch.Tensor, eps: float = 1e-12) -> float:
    # Shannon entropy (higher = more uniform, lower = more concentrated)
    q = torch.clamp(q, min=eps)
    return float(-(q * torch.log(q)).sum().item())


def train_one_epoch_amp(
    model,
    loader,
    optimizer,
    device,
    scaler_amp: torch.cuda.amp.GradScaler,
    max_grad_norm: float = 1.0,
    eta: float = 0.1,
    q_eps: float = 1e-8,
) -> Dict[str, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    # infer num_groups
    if hasattr(loader, "dataset") and hasattr(loader.dataset, "df") and "domain_id" in loader.dataset.df.columns:
        num_groups = int(loader.dataset.df["domain_id"].nunique())
    else:
        first = next(iter(loader))
        num_groups = int(first["domain_id"].max().item()) + 1

    q = torch.ones(num_groups, device=device, dtype=torch.float32)
    q = q / q.sum()

    total = 0
    running_loss = 0.0
    correct = 0

    use_amp = scaler_amp is not None and scaler_amp.is_enabled()

    # ---- q logging accumulators (epoch-level)
    q_sum = torch.zeros(num_groups, device=device, dtype=torch.float32)
    q_steps = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        domain_ids = batch["domain_id"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

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

            losses = loss_fn(logits, labels)  # (B,)

            group_loss = torch.zeros(num_groups, device=device, dtype=torch.float32)
            present = torch.zeros(num_groups, device=device, dtype=torch.bool)

            for g in range(num_groups):
                mask = (domain_ids == g)
                if mask.any():
                    group_loss[g] = losses[mask].mean().float()
                    present[g] = True

            # update q (only present groups)
            with torch.no_grad():
                q = q * torch.exp(eta * group_loss * present.float())
                q = torch.clamp(q, min=q_eps)
                q = q / q.sum()

                # log q after update
                q_sum += q
                q_steps += 1

            loss = (q * group_loss).sum()

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
        running_loss += float(loss.item()) * bs

        preds = torch.argmax(logits.detach(), dim=1)
        correct += (preds == labels).sum().item()

    # ---- epoch-level q stats
    if q_steps > 0:
        q_mean = (q_sum / q_steps).detach().cpu()
    else:
        q_mean = q.detach().cpu()

    q_mean_max = float(q_mean.max().item())
    q_mean_min = float(q_mean.min().item())
    q_mean_entropy = _q_entropy(q_mean)

    return {
        "loss": running_loss / max(total, 1),
        "acc": correct / max(total, 1),

        # q logging
        "q_max": q_mean_max,
        "q_min": q_mean_min,
        "q_entropy": q_mean_entropy,
        # if you want the full vector later in main_train:
        "q_mean_vec": q_mean.tolist(),
    }
