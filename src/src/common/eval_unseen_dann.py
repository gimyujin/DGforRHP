# src/common/eval_unseen_dann.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

try:
    import joblib
except Exception:
    joblib = None

# IMPORTANT: 학습에 사용한 모델을 그대로 import (구조 불일치 방지)
from dann.model import DANNModel  # requires: PYTHONPATH=src


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

        item: Dict[str, Any] = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row[self.label_col]), dtype=torch.long),
        }

        if self.num_cols:
            num = row[self.num_cols].to_numpy(dtype=np.float32)
            item["num"] = torch.tensor(num, dtype=torch.float32)

        return item


# -------------------------
# checkpoint helpers
# -------------------------
def _looks_like_state_dict(obj: Any) -> bool:
    if not isinstance(obj, dict) or len(obj) == 0:
        return False
    # quick check
    sample_items = list(obj.items())[:10]
    for k, v in sample_items:
        if not isinstance(k, str):
            return False
        if not torch.is_tensor(v):
            return False
    return True


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """
    지원 포맷:
      - ckpt_obj 자체가 state_dict
      - ckpt_obj["model_state"] / ["model"] / ["state_dict"] / ["model_state_dict"] / ["state_dict_model"]
    """
    if isinstance(ckpt_obj, dict):
        for key in ["model_state", "model", "state_dict", "model_state_dict", "state_dict_model"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict) and _looks_like_state_dict(ckpt_obj[key]):
                return ckpt_obj[key]
        if _looks_like_state_dict(ckpt_obj):
            return ckpt_obj
    raise KeyError(
        "Checkpoint에서 state_dict를 찾지 못했습니다. "
        "지원 키: model_state / model / state_dict / model_state_dict / state_dict_model "
        "또는 ckpt 자체가 state_dict여야 합니다."
    )


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _infer_model_name_numcols(
    ckpt_path: Path, args_model_name: Optional[str], args_num_cols: Optional[List[str]]
) -> Tuple[str, List[str], dict]:
    """
    우선순위:
      model_name: CLI > config_used.json > default
      num_cols:   CLI > config_used.json > []
    """
    parent = ckpt_path.parent
    cfg = (
        _load_json(parent / "config_used.json")
        or _load_json(parent / "config.json")
        or _load_json(parent / "hparams.json")
        or {}
    )

    model_name = args_model_name or cfg.get("encoder_name") or cfg.get("model_name") or "bert-base-uncased"

    if args_num_cols is not None:
        num_cols = args_num_cols
    else:
        nc = cfg.get("num_cols", [])
        if isinstance(nc, str):
            num_cols = [nc]
        elif isinstance(nc, list):
            num_cols = nc
        else:
            num_cols = []

    num_cols = [str(x) for x in num_cols] if num_cols else []
    return model_name, num_cols, cfg


def _infer_num_domains(ckpt_path: Path, state_dict: Dict[str, torch.Tensor], cfg: dict) -> int:
    """
    우선순위:
      1) domain_vocab.json 길이
      2) cfg의 num_domains 힌트
      3) state_dict에서 domain_head 마지막 레이어 weight shape 추정
      4) fallback=2
    """
    parent = ckpt_path.parent

    dom_vocab = _load_json(parent / "domain_vocab.json")
    if isinstance(dom_vocab, dict) and len(dom_vocab) > 0:
        return int(len(dom_vocab))

    for k in ["num_domains", "n_domains"]:
        if k in cfg:
            try:
                return int(cfg[k])
            except Exception:
                pass

    # for src/dann/model.py version you posted:
    # domain_head = MLP(...), 마지막 weight key는 domain_head.net.3.weight
    cand = [k for k in state_dict.keys() if k.endswith("domain_head.net.3.weight")]
    if cand:
        w = state_dict[cand[0]]
        if torch.is_tensor(w) and w.ndim == 2:
            return int(w.shape[0])

    return 2


# -------------------------
# eval
# -------------------------
@torch.no_grad()
def evaluate_file(model, loader, device) -> Dict[str, float]:
    model.eval()
    labels_all: List[int] = []
    preds_all: List[int] = []

    # sanity
    try:
        print("Model device:", next(model.parameters()).device)
    except Exception:
        pass

    for batch in loader:
        # ✅ 배치 전체를 한 번에 device로 이동 (cpu/cuda mismatch 방지)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        if "num" in batch:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                num=batch["num"],
                grl_lambda=0.0,  # inference: adversarial off
            )
        else:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                num=None,
                grl_lambda=0.0,
            )

        logits = out.y_logits
        preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
        labels = batch["labels"].detach().cpu().tolist()

        preds_all.extend(preds)
        labels_all.extend(labels)

    return {
        "n": int(len(labels_all)),
        "acc": float(acc(labels_all, preds_all)),
        "f1_macro": float(macro_f1_binary(labels_all, preds_all)),
    }


def _safe_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, required=False, default="dann", choices=["dann"])
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--file", type=str, required=True)

    p.add_argument("--text_col", type=str, default="text")
    p.add_argument("--label_col", type=str, default="label")

    p.add_argument("--num_cols", nargs="*", default=None)
    p.add_argument("--scaler", type=str, default=None)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=320)
    p.add_argument("--num_workers", type=int, default=2)

    # config에 model_name이 없을 때 강제 지정
    p.add_argument("--model_name", type=str, default=None)

    # 결과 저장 경로 커스텀 (기본: outputs/unseen_eval/dann/<domain>/)
    p.add_argument("--outdir", type=str, default=None)

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

    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt_obj)
    state_dict = _strip_module_prefix(state_dict)

    model_name, num_cols, cfg = _infer_model_name_numcols(ckpt_path, args.model_name, args.num_cols)
    use_num = len(num_cols) > 0

    # scaler
    scaler = None
    if use_num:
        if args.scaler is None:
            raise ValueError("text+num 평가면 --scaler (학습 때 저장한 scaler.pkl)가 필요합니다.")
        if joblib is None:
            raise ImportError("joblib이 없어서 scaler를 로드할 수 없습니다. pip install joblib")
        scaler = joblib.load(args.scaler)

    # num_domains
    num_domains = _infer_num_domains(ckpt_path, state_dict, cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # ✅ 생성자 인자 최소화: 시그니처 mismatch 방지
    model = DANNModel(
        encoder_name=model_name,
        num_labels=2,
        num_domains=num_domains,
        num_numeric=len(num_cols),
        use_numeric=use_num,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # 안전장치: BERT/encoder 제외 missing이 너무 많으면 구조 불일치 가능
    bad_missing = [k for k in missing if not k.startswith("encoder.") and not k.startswith("bert.")]
    if len(bad_missing) > 50:
        raise RuntimeError(
            "Checkpoint 로딩 시 누락 키가 너무 많습니다(구조 불일치 가능). "
            f"예: {bad_missing[:20]}"
        )

    if missing:
        print("[INFO] Missing keys (first 20):", missing[:20], ("..." if len(missing) > 20 else ""))
    if unexpected:
        print("[INFO] Unexpected keys (first 20):", unexpected[:20], ("..." if len(unexpected) > 20 else ""))

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
    rep["algo"] = args.algo
    rep["ckpt"] = str(ckpt_path)
    rep["model_name"] = model_name
    rep["num_cols"] = num_cols
    rep["use_num"] = bool(use_num)

    # 저장 경로
    if args.outdir is not None:
        out_dir = Path(args.outdir)
    else:
        out_dir = Path("outputs") / "unseen_eval" / args.algo / dname

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{dname}_report.json"
    _safe_json_dump(rep, out_json)

    # stdout: JSON 그대로 출력 (복붙/로그용)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print("Saved:", str(out_json))


if __name__ == "__main__":
    main()
