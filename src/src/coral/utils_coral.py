# src/coral/utils_coral.py
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    import yaml
except Exception:
    yaml = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML이 필요합니다. pip install pyyaml")
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_json_dump(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_amp_scaler(enabled: bool) -> torch.cuda.amp.GradScaler:
    # PyTorch 2.5+ 권장 방식
    return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt: Dict[str, Any] = {
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        ckpt["optim_state"] = optimizer.state_dict()
    if epoch is not None:
        ckpt["epoch"] = int(epoch)
    if extra:
        ckpt.update(extra)

    torch.save(ckpt, path)


def maybe_set_cuda_visible_devices(gpu: Optional[str]) -> None:
    """
    main_train.py에서 config로 GPU를 지정하고 싶을 때 사용.
    gpu="0" 또는 "0,1" 같은 문자열.
    """
    if gpu is None:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
