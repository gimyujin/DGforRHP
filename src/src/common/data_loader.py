# data_loader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import joblib

import torch
from torch.utils.data import Dataset, DataLoader


# =========================
# Paths
# =========================
@dataclass
class SplitPaths:
    train_path: str
    val_path: str
    test_path: str


# =========================
# Basic utils
# =========================
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def basic_clean(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    domain_col: str,
    num_cols: Optional[List[str]],
) -> pd.DataFrame:
    out = df.copy()

    need = [text_col, label_col, domain_col] + (num_cols if num_cols else [])
    out = out.dropna(subset=need)

    out[text_col] = out[text_col].astype(str)
    out[label_col] = out[label_col].astype(int)
    out[domain_col] = out[domain_col].astype(str)

    if num_cols:
        for c in num_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(subset=num_cols)

    return out


# =========================
# Domain handling
# =========================
def build_domain_vocab(df_train: pd.DataFrame, domain_col: str) -> Dict[str, int]:
    domains = sorted(df_train[domain_col].unique().tolist())
    return {d: i for i, d in enumerate(domains)}


def encode_domain(
    df: pd.DataFrame,
    domain_col: str,
    domain2id: Dict[str, int],
) -> pd.DataFrame:
    out = df.copy()
    out["domain_id"] = out[domain_col].map(domain2id)

    if out["domain_id"].isna().any():
        unseen = out.loc[out["domain_id"].isna(), domain_col].unique().tolist()
        raise ValueError(f"Unseen domains in split (not in train): {unseen}")

    out["domain_id"] = out["domain_id"].astype(int)
    return out


# =========================
# Numeric scaling (train-only fit)
# =========================
def fit_num_scaler(df_train: pd.DataFrame, num_cols: List[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df_train[num_cols].to_numpy(dtype=np.float32))
    return scaler


def apply_num_scaler(
    df: pd.DataFrame,
    num_cols: List[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    out = df.copy()
    scaled = scaler.transform(out[num_cols].to_numpy(dtype=np.float32))
    out[num_cols] = scaled
    return out


def save_scaler(scaler: StandardScaler, path: str) -> None:
    joblib.dump(scaler, path)


# =========================
# Load splits + domain + scaler
# =========================
def load_splits_with_domain(
    paths: SplitPaths,
    text_col: str = "text",
    label_col: str = "label",
    domain_col: str = "domain",
    num_cols: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, int],
    Dict[int, str],
    Optional[StandardScaler],
]:
    # ---- load + clean
    df_train = basic_clean(
        load_parquet(paths.train_path),
        text_col,
        label_col,
        domain_col,
        num_cols,
    )
    df_val = basic_clean(
        load_parquet(paths.val_path),
        text_col,
        label_col,
        domain_col,
        num_cols,
    )
    df_test = basic_clean(
        load_parquet(paths.test_path),
        text_col,
        label_col,
        domain_col,
        num_cols,
    )

    # ---- numeric scaling (fit on train only)
    fitted_scaler = None
    if num_cols and len(num_cols) > 0:
        fitted_scaler = scaler if scaler is not None else fit_num_scaler(df_train, num_cols)
        df_train = apply_num_scaler(df_train, num_cols, fitted_scaler)
        df_val = apply_num_scaler(df_val, num_cols, fitted_scaler)
        df_test = apply_num_scaler(df_test, num_cols, fitted_scaler)

    # ---- domain vocab (train only)
    domain2id = build_domain_vocab(df_train, domain_col)
    id2domain = {v: k for k, v in domain2id.items()}

    df_train = encode_domain(df_train, domain_col, domain2id)
    df_val = encode_domain(df_val, domain_col, domain2id)
    df_test = encode_domain(df_test, domain_col, domain2id)

    splits = {
        "train": df_train,
        "val": df_val,
        "test": df_test,
    }

    return splits, domain2id, id2domain, fitted_scaler


# =========================
# Dataset / DataLoader
# =========================
class ReviewDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str = "text",
        label_col: str = "label",
        num_cols: Optional[List[str]] = None,
        max_length: int = 256,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.num_cols = num_cols or []
        self.max_length = max_length

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
            "domain_id": torch.tensor(int(row["domain_id"]), dtype=torch.long),
        }

        if self.num_cols:
            num = row[self.num_cols].to_numpy(dtype=np.float32)
            item["num"] = torch.tensor(num, dtype=torch.float32)

        return item


def build_dataloader(
    df: pd.DataFrame,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    text_col: str = "text",
    label_col: str = "label",
    num_cols: Optional[List[str]] = None,
    max_length: int = 256,
    num_workers: int = 2,
) -> DataLoader:
    ds = ReviewDataset(
        df,
        tokenizer,
        text_col=text_col,
        label_col=label_col,
        num_cols=num_cols,
        max_length=max_length,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
