# utils.py
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_adamw_optimizer(model, lr: float, eps: float, weight_decay: float):
    # BERT 관례: bias / LayerNorm에는 weight decay 적용 X
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)

    print(
        "Optimizer groups:",
        f"decay={len(decay_params)}",
        f"no_decay={len(no_decay_params)}",
        f"weight_decay={weight_decay}",
    )
    return optimizer


def get_val_cfg(cfg: dict) -> dict:
    # validation / val 둘 다 허용
    return cfg.get("validation", cfg.get("val", {}))


def safe_json_dump(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def make_outdir(base_dir: str, algo: str, model_tag: str, run_id: str) -> Path:
    outdir = Path(base_dir) / algo / model_tag / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir
