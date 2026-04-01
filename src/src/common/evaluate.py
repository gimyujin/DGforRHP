# evaluate.py
from __future__ import annotations
from typing import Dict, Any
import torch


def _macro_f1_binary(labels, preds) -> float:
    cm00 = cm01 = cm10 = cm11 = 0
    for t, p in zip(labels, preds):
        t = int(t); p = int(p)
        if t == 0 and p == 0: cm00 += 1
        elif t == 0 and p == 1: cm01 += 1
        elif t == 1 and p == 0: cm10 += 1
        else: cm11 += 1

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
def evaluate_by_domain(
    model,
    loader,
    device,
    id2domain: Dict[int, str],
    metric: str = "f1_macro",
) -> Dict[str, Any]:
    """
    DG validation score:
    1) 도메인별 metric 계산
    2) 도메인 균등 평균(unweighted mean) = 최종 val score
    """
    model.eval()

    # domain_id -> list of (labels, preds)
    buckets = {}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        domain_ids = batch["domain_id"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask, num=batch["num"].to(device)) \
            if "num" in batch else model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=1)

        labels_cpu = labels.detach().cpu().tolist()
        preds_cpu = preds.detach().cpu().tolist()
        dom_cpu = domain_ids.detach().cpu().tolist()

        for d, t, p in zip(dom_cpu, labels_cpu, preds_cpu):
            if d not in buckets:
                buckets[d] = {"labels": [], "preds": []}
            buckets[d]["labels"].append(t)
            buckets[d]["preds"].append(p)

    domain_scores = {}
    for d, pack in buckets.items():
        labels_d = pack["labels"]
        preds_d = pack["preds"]

        if metric == "acc":
            score = _acc(labels_d, preds_d)
        elif metric == "f1_macro":
            score = _macro_f1_binary(labels_d, preds_d)
        else:
            raise ValueError("metric must be 'acc' or 'f1_macro'")

        name = id2domain.get(int(d), str(d))
        domain_scores[name] = score

    domain_mean = sum(domain_scores.values()) / max(len(domain_scores), 1)

    return {
        "metric": metric,
        "domain_scores": domain_scores,
        "domain_unweighted_mean": domain_mean,
        "num_domains": len(domain_scores),
    }
