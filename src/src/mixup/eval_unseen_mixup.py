# src/common/eval_unseen_mixup.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

try:
    import joblib
except Exception:
    joblib = None


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
# dataset (no domain column)
# -------------------------
class ReviewDatasetNoDomain(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str,
        label_col: str,
        num_cols: Optional[List[str]],
        max_length: int,
        scaler=None,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.num_cols = num_cols or []
        self.max_length = max_length
        self.scaler = scaler

        need = [text_col, label_col] + self.num_cols
        self.df = self.df.dropna(subset=need).copy()
        self.df[text_col] = self.df[text_col].astype(str)
        self.df[label_col] = self.df[label_col].astype(int)

        if self.num_cols:
            for c in self.num_cols:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
            self.df = self.df.dropna(subset=self.num_cols).copy()

            if self.scaler is not None:
                x = self.df[self.num_cols].to_numpy(dtype=np.float32)
                x = self.scaler.transform(x)
                self.df.loc[:, self.num_cols] = x

    def __len__(self):
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

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row[self.label_col]), dtype=torch.long),
        }

        if self.num_cols:
            num = row[self.num_cols].to_numpy(dtype=np.float32)
            item["num"] = torch.tensor(num, dtype=torch.float32)

        return item


@torch.no_grad()
def evaluate_file(model, loader, device) -> Dict[str, float]:
    model.eval()
    labels_all: List[int] = []
    preds_all: List[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        if "num" in batch:
            num = batch["num"].to(device, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, num=num)
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
        labels = batch["labels"].detach().cpu().tolist()

        preds_all.extend(preds)
        labels_all.extend(labels)

    return {
        "n": int(len(labels_all)),
        "acc": float(acc(labels_all, preds_all)),
        "f1_macro": float(macro_f1_binary(labels_all, preds_all)),
    }


def safe_json_dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--file", type=str, required=True)

    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--label_col", type=str, default="label")

    p.add_argument("--num_cols", nargs="*", default=None)  # 없으면 text-only
    p.add_argument("--scaler", type=str, default=None)     # text+num이면 보통 필요

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=320)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.ckpt)
    file_path = Path(args.file)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", "bert-base-uncased")
    ckpt_num_cols = ckpt.get("num_cols", None)

    # num_cols 우선순위: CLI > ckpt > text-only
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
            raise ImportError("joblib이 없어서 scaler를 로드할 수 없습니다. pip install joblib")
        scaler = joblib.load(args.scaler)

    # model import: 구조가 baseline과 동일하면 baseline model로 로드해도 OK
    from model_mixup import BertTextClassifierMixup, BertTextNumClassifierMixup

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if use_num:
        model = BertTextNumClassifierMixup(
            model_name=model_name,
            num_features=len(num_cols),
            num_labels=2,
            dropout=0.1,
        ).to(device)
    else:
        model = BertTextClassifierMixup(
            model_name=model_name,
            num_labels=2,
            dropout=0.1,
        ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)

    df = pd.read_parquet(file_path)
    dname = file_path.stem

    ds = ReviewDatasetNoDomain(
        df=df,
        tokenizer=tokenizer,
        text_col=args.text_col,
        label_col=args.label_col,
        num_cols=num_cols if use_num else [],
        max_length=args.max_length,
        scaler=scaler,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    rep = evaluate_file(model, loader, device)
    rep["file"] = str(file_path)
    rep["domain_name"] = dname
    rep["algo"] = "mixup"
    rep["ckpt"] = str(ckpt_path)

    out_dir = Path("outputs") / "unseen_eval" / "mixup" / dname
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"{dname}_report.json"
    safe_json_dump(rep, out_json)

    print(f"[mixup/{dname}] n={rep['n']:,} acc={rep['acc']:.4f} f1_macro={rep['f1_macro']:.4f}")
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
