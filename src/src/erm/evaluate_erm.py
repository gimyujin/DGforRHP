# evaluate_erm.py
from __future__ import annotations
from typing import Dict, Any
import torch


def _macro_f1_binary(labels, preds) -> float:
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

    def f1_for_class(pos_label: int) -> float:
        if pos_label == 1:
            tp, fp, fn = cm11, cm01, cm10
        else:
            tp, fp, fn = cm00, cm10, cm01

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return 2 * precision * recall / max(precision + recall, 1e-12)

    return (f1_for_class(0) + f1_for_class(1)) / 2


def _acc(labels, preds) -> float:
    correct = sum(int(t == p) for t, p in zip(labels, preds))
    return correct / max(len(labels), 1)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    metric: str = "f1_macro",
) -> Dict[str, Any]:
    """
    Standard ERM evaluation:
    - 평가셋 전체 샘플을 대상으로 metric 계산
    """
    model.eval()

    all_labels = []
    all_preds = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if "num" in batch:
            num = batch["num"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, num=num)
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        preds = torch.argmax(logits, dim=1)

        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    if metric == "acc":
        score = _acc(all_labels, all_preds)
    elif metric == "f1_macro":
        score = _macro_f1_binary(all_labels, all_preds)
    else:
        raise ValueError("metric must be 'acc' or 'f1_macro'")

    return {
        "metric": metric,
        "score": score,
        "num_samples": len(all_labels),
    }