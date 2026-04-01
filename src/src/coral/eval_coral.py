# src/coral/eval_coral.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

try:
    import joblib
except Exception:
    joblib = None

from src.coral.model import BertTextCoralClassifier, BertTextNumCoralClassifier


# -------------------------
# metrics (binary macro-F1)``
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


# -------------------------
# dataset (with domain)
# -------------------------
class ReviewDatasetCoralEval(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str,
        label_col: str,
        domain_col: str,
        num_cols: Optional[List[str]],
        max_length: int,
        domain2id: Dict[str, int],
        scaler=None,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.domain_col = domain_col
        self.num_cols = num_cols or []
        self.max_length = max_length
        self.domain2id = domain2id
        self.scaler = scaler

        need = [text_col, label_col, domain_col] + self.num_cols
        self.df = self.df.dropna(subset=need).copy()

        self.df[text_col] = self.df[text_col].astype(str)
        self.df[label_col] = self.df[label_col].astype(int)
        self.df[domain_col] = self.df[domain_col].astype(str)

        self.df["_domain_id"] = self.df[domain_col].map(self.domain2id)
        self.df = self.df.dropna(subset=["_domain_id"]).copy()
        self.df["_domain_id"] = self.df["_domain_id"].astype(int)

        if self.num_cols:
            for c in self.num_cols:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
            self.df = self.df.dropna(subset=self.num_cols).copy()

            if self.scaler is not None:
                x = self.df[self.num_cols].to_numpy(dtype=np.float32)
                x = self.scaler.transform(x)
                self.df.loc[:, self.num_cols] = x

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row[self.text_col],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item: Dict[str, Any] = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row[self.label_col]), dtype=torch.long),
            "domain": torch.tensor(int(row["_domain_id"]), dtype=torch.long),
        }

        if self.num_cols:
            num = row[self.num_cols].to_numpy(dtype=np.float32)
            item["num"] = torch.tensor(num, dtype=torch.float32)

        return item


@torch.no_grad()
def evaluate_domain_unweighted_mean(
    model,
    loader,
    device,
    id2domain: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    model.eval()

    labels_by: Dict[int, List[int]] = {}
    preds_by: Dict[int, List[int]] = {}

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        domain_ids = batch["domain"].cpu().tolist()
        labels = batch["labels"].cpu().tolist()

        if "num" in batch:
            num = batch["num"].to(device, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, num=num, return_features=False)
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask, return_features=False)

        preds = torch.argmax(logits, dim=1).cpu().tolist()

        for d, t, p in zip(domain_ids, labels, preds):
            d = int(d)
            labels_by.setdefault(d, []).append(int(t))
            preds_by.setdefault(d, []).append(int(p))

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
        "metric": "f1_macro",
        "domain_scores": domain_scores,
        "domain_unweighted_mean": float(np.mean(scores)) if scores else 0.0,
        "num_domains": int(len(scores)),
    }


def safe_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--file", type=str, required=True)  # parquet

    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--domain_col", type=str, default="domain")

    p.add_argument("--num_cols", nargs="*", default=None)  # 없으면 text-only
    p.add_argument("--scaler", type=str, default=None)     # text+num이면 보통 필요
    p.add_argument("--domain2id", type=str, required=True) # train에서 저장한 domain2id.json

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=320)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--out_json", type=str, default=None)   # 저장 경로 옵션
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.ckpt)
    file_path = Path(args.file)
    domain2id_path = Path(args.domain2id)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")
    if not domain2id_path.exists():
        raise FileNotFoundError(f"domain2id not found: {domain2id_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", "bert-base-uncased")
    hidden_num = int(ckpt.get("hidden_num", ckpt.get("num_hidden", 128)))


    # num_cols 우선순위: CLI > ckpt > text-only
    ckpt_num_cols = ckpt.get("num_cols", None)
    if args.num_cols is not None:
        num_cols = args.num_cols
    elif ckpt_num_cols is not None:
        num_cols = ckpt_num_cols
    else:
        num_cols = []

    use_num = len(num_cols) > 0

    scaler = None
    if use_num:
        if args.scaler is None:
            raise ValueError("text+num 평가면 --scaler (학습 때 저장한 scaler.pkl)가 필요합니다.")
        if joblib is None:
            raise ImportError("joblib이 없어서 scaler 로드가 불가합니다. pip install joblib")
        scaler = joblib.load(args.scaler)

    with open(domain2id_path, "r", encoding="utf-8") as f:
        domain2id = json.load(f)
    # id2domain 생성
    id2domain = {int(v): str(k) for k, v in domain2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    
    if use_num:
        model = BertTextNumCoralClassifier(
            model_name=model_name,
            num_features=len(num_cols),
            num_labels=2,
            dropout=0.1,
            hidden_num=hidden_num,
        ).to(device)
    else:
        model = BertTextCoralClassifier(
            model_name=model_name,
            num_labels=2,
            dropout=0.1,
        ).to(device)

    if "model_state" not in ckpt:
        raise KeyError("Checkpoint에 'model_state'가 없습니다.")
    model.load_state_dict(ckpt["model_state"], strict=True)

    df = pd.read_parquet(file_path)

    ds = ReviewDatasetCoralEval(
        df=df,
        tokenizer=tokenizer,
        text_col=args.text_col,
        label_col=args.label_col,
        domain_col=args.domain_col,
        num_cols=num_cols if use_num else [],
        max_length=args.max_length,
        domain2id=domain2id,
        scaler=scaler,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory = (device.type == "cuda"),
    )

    rep = evaluate_domain_unweighted_mean(model, loader, device, id2domain=id2domain)

    # 저장 경로
    if args.out_json is not None:
        out_json = Path(args.out_json)
    else:
        out_json = ckpt_path.parent / "test_domain_report.json"


    safe_json_dump(rep, out_json)

    print(f"[CORAL/{file_path.stem}] domain_unweighted_mean({rep['metric']})={rep['domain_unweighted_mean']:.4f}")
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
