# src/coral/data_loader_coral.py
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class DomainMapping:
    domain2id: Dict[str, int]
    id2domain: Dict[int, str]


def build_domain_mapping(domains: List[str]) -> DomainMapping:
    uniq = sorted(set([str(d) for d in domains]))
    domain2id = {d: i for i, d in enumerate(uniq)}
    id2domain = {i: d for d, i in domain2id.items()}
    return DomainMapping(domain2id=domain2id, id2domain=id2domain)


class ReviewDatasetCoral(Dataset):
    """
    CORAL 학습용 Dataset
    - 반드시 domain을 반환한다 (batch["domain"]).
    - text-only / text+num 둘 다 지원.
    - num_scaler는 train에서 fit해서 val/test에 재사용하는 방식 권장.
    """

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
        num_scaler=None,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.domain_col = domain_col
        self.num_cols = num_cols or []
        self.max_length = max_length
        self.domain2id = domain2id
        self.num_scaler = num_scaler

        need = [text_col, label_col, domain_col] + self.num_cols
        self.df = self.df.dropna(subset=need).copy()

        self.df[text_col] = self.df[text_col].astype(str)
        self.df[label_col] = self.df[label_col].astype(int)
        self.df[domain_col] = self.df[domain_col].astype(str)

        # domain을 id로 변환 (학습/평가 모두 동일 mapping 사용)
        self.df["_domain_id"] = self.df[domain_col].map(self.domain2id)
        self.df = self.df.dropna(subset=["_domain_id"]).copy()
        self.df["_domain_id"] = self.df["_domain_id"].astype(int)

        # num 처리
        if self.num_cols:
            for c in self.num_cols:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
            self.df = self.df.dropna(subset=self.num_cols).copy()

            if self.num_scaler is not None:
                x = self.df[self.num_cols].to_numpy(dtype=np.float32)
                x = self.num_scaler.transform(x)
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


def read_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return pd.read_parquet(path)


def make_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def fit_num_scaler(df_train: pd.DataFrame, num_cols: List[str]):
    """
    text+num에서만 사용.
    sklearn StandardScaler를 쓰는 패턴을 가정.
    """
    from sklearn.preprocessing import StandardScaler

    x = df_train[num_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler


def save_scaler(scaler, path: str | Path) -> None:
    if joblib is None:
        raise ImportError("joblib이 없어서 scaler 저장이 불가합니다. pip install joblib")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: str | Path):
    if joblib is None:
        raise ImportError("joblib이 없어서 scaler 로드가 불가합니다. pip install joblib")
    return joblib.load(path)


def make_loaders_coral(
    cfg: Dict[str, Any],
    *,
    split: str = "train",
    domain_mapping: Optional[DomainMapping] = None,
    num_scaler=None,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], DomainMapping, Any]:
    """
    cfg(yaml) 기반으로 train/val/test loader를 만든다.
    - train 호출 시: domain_mapping/num_scaler를 None으로 두면 내부에서 생성/fit
    - val/test 호출 시: train에서 만든 domain_mapping/num_scaler를 그대로 넘기면 됨

    반환:
      train_loader, val_loader, test_loader, domain_mapping, num_scaler
    """
    data_dir = Path(cfg["data"]["data_dir"])
    train_path = data_dir / cfg["data"]["train_file"]
    val_path = data_dir / cfg["data"]["val_file"]
    test_path = data_dir / cfg["data"]["test_file"]

    text_col = cfg["columns"]["text"]
    label_col = cfg["columns"]["label"]
    domain_col = cfg["columns"]["domain"]
    num_cols = cfg["columns"].get("num", []) if cfg["exp"]["mode"] == "text_num" else []

    model_name = cfg["model"]["name"]
    max_length = int(cfg["model"]["max_length"])

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 2))

    tokenizer = make_tokenizer(model_name)

    df_train = read_parquet(train_path)
    df_val = read_parquet(val_path) if val_path.exists() else None
    df_test = read_parquet(test_path) if test_path.exists() else None

    # domain mapping (train 기준으로 고정)
    if domain_mapping is None:
        domain_mapping = build_domain_mapping(df_train[domain_col].astype(str).tolist())

    # num scaler (train에서 fit)
    if num_cols and num_scaler is None:
        # 결측 제거/숫자 변환 후 fit
        tmp = df_train.dropna(subset=num_cols).copy()
        for c in num_cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        tmp = tmp.dropna(subset=num_cols).copy()
        num_scaler = fit_num_scaler(tmp, num_cols)

    ds_train = ReviewDatasetCoral(
        df=df_train,
        tokenizer=tokenizer,
        text_col=text_col,
        label_col=label_col,
        domain_col=domain_col,
        num_cols=num_cols,
        max_length=max_length,
        domain2id=domain_mapping.domain2id,
        num_scaler=num_scaler,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if df_val is not None:
        ds_val = ReviewDatasetCoral(
            df=df_val,
            tokenizer=tokenizer,
            text_col=text_col,
            label_col=label_col,
            domain_col=domain_col,
            num_cols=num_cols,
            max_length=max_length,
            domain2id=domain_mapping.domain2id,
            num_scaler=num_scaler,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    test_loader = None
    if df_test is not None:
        ds_test = ReviewDatasetCoral(
            df=df_test,
            tokenizer=tokenizer,
            text_col=text_col,
            label_col=label_col,
            domain_col=domain_col,
            num_cols=num_cols,
            max_length=max_length,
            domain2id=domain_mapping.domain2id,
            num_scaler=num_scaler,
        )
        test_loader = DataLoader(
            ds_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader, domain_mapping, num_scaler
