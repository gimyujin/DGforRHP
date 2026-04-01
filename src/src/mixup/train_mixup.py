# train_mixup.py
from __future__ import annotations

from typing import Dict
from tqdm import tqdm

import torch
import torch.nn.functional as F


def soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """
    Soft-label cross entropy.
    logits: [B, C]
    soft_targets: [B, C] (rows sum to 1)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def sample_mixup_params(
    batch_size: int,
    alpha: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    논문의 설정(fixed lambda = 0.5)을 따르기 위한 수정
    """
    # 기존처럼 베타 분포를 쓰지 않고 0.5로 고정
    lam = torch.tensor(0.5, device=device) 
    idx = torch.randperm(batch_size, device=device)
    return lam, idx

def apply_mix(x: torch.Tensor, lam: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return lam * x + (1.0 - lam) * x[idx]


def train_one_epoch_mixup_amp(
    model,
    loader,
    optimizer,
    device,
    scaler_amp: torch.cuda.amp.GradScaler,
    mixup_alpha: float = 0.4,
    num_labels: int = 2,
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """
    Representation-level Mixup for BERT classifiers.
    Requires model to expose:
      - encode(input_ids, attention_mask) -> cls [B, H]
      - classify(cls) or classify(cls, num=...) -> logits [B, C]
      - forward(...) still works for eval
    """
    model.train()

    total = 0
    running_loss = 0.0
    correct = 0

    use_amp = scaler_amp is not None and scaler_amp.is_enabled()

    for batch in tqdm(loader, desc="Train(Mixup)", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)
        else:
            autocast_ctx = torch.amp.autocast("cpu", enabled=False)

        with autocast_ctx:
            bs = labels.size(0)
            y = F.one_hot(labels, num_classes=num_labels).float()
            lam, idx = sample_mixup_params(bs, mixup_alpha, device)

            # 텍스트 믹스
            cls = model.encode(input_ids=input_ids, attention_mask=attention_mask)
            cls_mix = apply_mix(cls, lam, idx)

            # 라벨 믹스
            y_mix = apply_mix(y, lam, idx)
 
            # 수치혀 믹스
            if "num" in batch:
                num = batch["num"].to(device, non_blocking=True)
                # Raw num이 아니라 프로젝션된 num_feat를 뽑아서 Mixup 진행
                num_feat = model.encode_num(num) 
                num_feat_mix = apply_mix(num_feat, lam, idx)
                
                logits = model.classify(cls_mix, num_feat=num_feat_mix)
            else:
                logits = model.classify(cls_mix)

            loss = soft_ce_loss(logits, y_mix)

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

        # logging acc against original hard labels (convention)
        total += bs
        running_loss += loss.item() * bs
        preds = torch.argmax(logits.detach(), dim=1)
        correct += (preds == labels).sum().item()

    return {
        "loss": running_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }
