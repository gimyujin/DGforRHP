# src/dann/train.py
from __future__ import annotations

from typing import Dict, Any, Optional
import math

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


def grl_lambda_schedule(step: int, total_steps: int, max_lambda: float = 1.0, gamma: float = 10.0) -> float:
    """
    DANN commonly used schedule:
      lambda(p) = max_lambda * (2/(1+exp(-gamma*p)) - 1)
      p in [0,1]
    """
    if total_steps <= 0:
        return float(max_lambda)
    p = min(max(step / total_steps, 0.0), 1.0)
    return float(max_lambda * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0))


def train_one_epoch_dann(
    model,
    dataloader,
    optimizer,
    device: torch.device,
    use_amp: bool,
    domain_loss_weight: float,
    global_step: int,
    total_steps: int,
    max_grl_lambda: float,
    grl_gamma: float,
    max_grad_norm: float,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, Any]:
    model.train()

    ce_y = nn.CrossEntropyLoss()
    ce_d = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_y = 0.0
    total_d = 0.0
    n = 0

    for batch in dataloader:
        n += 1

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        y = batch["labels"].to(device)            # 0/1
        d = batch["domain_id"].to(device)         # 0..K-1

        num = batch.get("num", None)
        if num is not None:
            num = num.to(device)

        grl_lambda = grl_lambda_schedule(
            step=global_step,
            total_steps=total_steps,
            max_lambda=max_grl_lambda,
            gamma=grl_gamma,
        )

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num=num,
                    grl_lambda=grl_lambda,
                )
                loss_y = ce_y(out.y_logits, y)
                loss_d = ce_d(out.d_logits, d)
                loss = loss_y + domain_loss_weight * loss_d

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num=num,
                grl_lambda=grl_lambda,
            )
            loss_y = ce_y(out.y_logits, y)
            loss_d = ce_d(out.d_logits, d)
            loss = loss_y + domain_loss_weight * loss_d

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_y += float(loss_y.detach().cpu())
        total_d += float(loss_d.detach().cpu())
        global_step += 1

    return {
        "loss": total_loss / max(n, 1),
        "loss_y": total_y / max(n, 1),
        "loss_d": total_d / max(n, 1),
        "global_step": global_step,
    }


@torch.no_grad()
def eval_binary_metrics(model, dataloader, device: torch.device, threshold: float = 0.5) -> Dict[str, float]:
    """
    Binary classification with CE logits (B,2).
    Returns: acc, f1_macro
    """
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    ys, ps = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        num = batch.get("num", None)
        if num is not None:
            num = num.to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, num=num, grl_lambda=0.0)
        prob = torch.softmax(out.y_logits, dim=1)[:, 1]
        pred = (prob >= threshold).long()

        ys.append(y.detach().cpu())
        ps.append(pred.detach().cpu())

    ys = torch.cat(ys).numpy()
    ps = torch.cat(ps).numpy()

    return {
        "acc": float(accuracy_score(ys, ps)),
        "f1_macro": float(f1_score(ys, ps, average="macro")),
    }
