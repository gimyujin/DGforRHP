# src/coral/train.py
from __future__ import annotations

from typing import Dict, List, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn


# -------------------------
# metrics (binary macro-F1)
# -------------------------
def macro_f1_binary(labels: List[int], preds: List[int]) -> float:
    cm00 = cm01 = cm10 = cm11 = 0
    for t, p in zip(labels, preds):
        t = int(t)
        p = int(p)
        if t == 0 and p == 0:
            cm00 += 1
        elif t == 0 and p == 1:
            cm01 += 1
        elif t == 1 and p == 0:
            cm10 += 1
        else:
            cm11 += 1

    def f1_for(pos_label: int) -> float:
        if pos_label == 1:
            tp, fp, fn = cm11, cm01, cm10
        else:
            tp, fp, fn = cm00, cm10, cm01
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return 2 * precision * recall / max(precision + recall, 1e-12)

    return (f1_for(0) + f1_for(1)) / 2.0


def acc(labels: List[int], preds: List[int]) -> float:
    if len(labels) == 0:
        return 0.0
    return float(np.mean([int(t == p) for t, p in zip(labels, preds)]))


# -------------------------
# CORAL loss
# -------------------------
def covariance(feat: torch.Tensor) -> torch.Tensor:
    # feat: (n, d)
    n = feat.size(0)
    d = feat.size(1)
    if n <= 1:
        return feat.new_zeros((d, d))
    feat = feat - feat.mean(dim=0, keepdim=True)
    return (feat.t() @ feat) / (n - 1)


def coral_loss_pair(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    # feat_a, feat_b: (n, d)
    ca = covariance(feat_a)
    cb = covariance(feat_b)
    d = ca.size(0)
    return ((ca - cb) ** 2).sum() / (4.0 * (d ** 2))


def coral_loss_batch(feat: torch.Tensor, domain: torch.Tensor, min_per_domain: int = 2) -> torch.Tensor:
    """
    배치 안에 섞여 있는 도메인들에 대해 pairwise CORAL 평균을 계산.
    - feat: (bs, d)
    - domain: (bs,)
    """
    uniq = torch.unique(domain)

    feats = []
    for d in uniq:
        m = (domain == d)
        f = feat[m]
        if f.size(0) >= min_per_domain:
            feats.append(f)

    if len(feats) < 2:
        return feat.new_zeros(())

    loss = feat.new_zeros(())
    cnt = 0
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            loss = loss + coral_loss_pair(feats[i], feats[j])
            cnt += 1

    return loss / max(cnt, 1)


# -------------------------
# train (AMP) - CORAL
# -------------------------
def train_one_epoch_amp_coral(
    model,
    loader,
    optimizer,
    device,
    scaler_amp: torch.cuda.amp.GradScaler,
    coral_lambda: float,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """
    네 baseline 루프 구조를 유지하면서,
    loss = CE + coral_lambda * CORAL(feat, domain) 를 최적화한다.

    batch에 반드시 포함:
      - input_ids
      - attention_mask
      - labels
      - domain
      - (옵션) num
    model은 return_features=True일 때 (logits, feat)를 반환해야 한다.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    total = 0
    running_loss = 0.0
    running_cls = 0.0
    running_coral = 0.0
    correct = 0

    use_amp = scaler_amp is not None and scaler_amp.is_enabled()

    for batch in tqdm(loader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if "domain" not in batch:
            raise KeyError("CORAL 학습은 batch에 'domain'이 필요합니다.")
        domain = batch["domain"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)
        else:
            autocast_ctx = torch.amp.autocast("cpu", enabled=False)

        with autocast_ctx:
            if "num" in batch:
                num = batch["num"].to(device, non_blocking=True)
                logits, feat = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num=num,
                    return_features=True,
                )
            else:
                logits, feat = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_features=True,
                )

            cls_loss = loss_fn(logits, labels)
            c_loss = coral_loss_batch(feat, domain)
            loss = cls_loss + coral_lambda * c_loss

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
        running_loss += float(loss.detach().item()) * bs
        running_cls += float(cls_loss.detach().item()) * bs
        running_coral += float(c_loss.detach().item()) * bs

        preds = torch.argmax(logits.detach(), dim=1)
        correct += (preds == labels).sum().item()

    return {
        "loss": running_loss / max(total, 1),
        "loss_cls": running_cls / max(total, 1),
        "loss_coral": running_coral / max(total, 1),
        "acc": correct / max(total, 1),
    }


# -------------------------
# eval (baseline-format report)
# -------------------------
@torch.no_grad()
def evaluate_domain_unweighted_mean(
    model,
    loader,
    device,
    id2domain: Optional[Dict[int, str]] = None,
    metric: str = "f1_macro",
) -> Dict[str, object]:
    """
    baseline과 동일한 리포트 포맷:
    {
      "metric": "f1_macro",
      "domain_scores": {"movie": 0.71, ...},
      "domain_unweighted_mean": 0.68,
      "num_domains": 4
    }
    """
    model.eval()

    labels_by: Dict[int, List[int]] = {}
    preds_by: Dict[int, List[int]] = {}

    for batch in tqdm(loader, desc="Val", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        if "domain" not in batch:
            raise KeyError("evaluate_domain_unweighted_mean: batch에 'domain'이 필요합니다.")
        domain_ids = batch["domain"].cpu().tolist()

        labels = batch["labels"].cpu().tolist()

        if "num" in batch:
            num = batch["num"].to(device, non_blocking=True)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num=num,
                return_features=False,
            )
        else:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_features=False,
            )

        preds = torch.argmax(logits, dim=1).cpu().tolist()

        for d, t, p in zip(domain_ids, labels, preds):
            d = int(d)
            labels_by.setdefault(d, []).append(int(t))
            preds_by.setdefault(d, []).append(int(p))

    if metric != "f1_macro":
        raise ValueError(f"지원 metric: f1_macro (요청: {metric})")

    domain_scores: Dict[str, float] = {}
    scores: List[float] = []

    for d in sorted(labels_by.keys()):
        y = labels_by[d]
        pr = preds_by[d]
        s = float(macro_f1_binary(y, pr))
        name = id2domain.get(d, str(d)) if id2domain is not None else str(d)
        domain_scores[name] = s
        scores.append(s)

    return {
        "metric": metric,
        "domain_scores": domain_scores,
        "domain_unweighted_mean": float(np.mean(scores)) if scores else 0.0,
        "num_domains": int(len(scores)),
    }
