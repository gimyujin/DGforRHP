# src/coral/main_train.py
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW

from src.coral.utils_coral import (
    set_seed,
    load_yaml,
    ensure_dir,
    safe_json_dump,
    get_device,
    make_amp_scaler,
    save_checkpoint,
)
from src.coral.data_loader_coral import (
    make_loaders_coral,
    save_scaler,
)
from src.coral.model import (
    BertTextCoralClassifier,
    BertTextNumCoralClassifier,
)
from src.coral.train import (
    train_one_epoch_amp_coral,
    evaluate_domain_unweighted_mean,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def build_output_dir(cfg: Dict[str, Any]) -> Path:
    out_root = Path(cfg["output"]["dir"])
    algo = cfg["exp"]["algo"]          # coral
    mode = cfg["exp"]["mode"]          # text_only / text_num
    run_id = cfg["exp"]["run_id"]      # v1
    return out_root / algo / f"bert_{mode}" / run_id


def append_train_log_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    # 고정 컬럼(필요하면 추가 가능)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_loss_cls",
        "train_loss_coral",
        "train_acc",
        "val_domain_unweighted_mean",
        "best_score_so_far",
    ]

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    out_dir = ensure_dir(build_output_dir(cfg))

    # baseline/dann 스타일로 파일명 맞춤
    safe_json_dump(cfg, out_dir / "config_used.json")

    device = get_device()
    print("Device:", device)

    # -------------------------
    # loaders
    # -------------------------
    train_loader, val_loader, test_loader, dom_map, num_scaler = make_loaders_coral(cfg)

    # baseline/dann 파일명에 맞춰 저장
    # (내용은 domain2id지만 파일명은 domain_vocab.json으로 맞춤)
    safe_json_dump(dom_map.domain2id, out_dir / "domain_vocab.json")

    if cfg["exp"]["mode"] == "text_num":
        save_scaler(num_scaler, out_dir / "scaler.pkl")

    # -------------------------
    # model
    # -------------------------
    model_name = cfg["model"]["name"]
    dropout = float(cfg["model"].get("dropout", 0.1))
    hidden_num = int(cfg["model"].get("num_hidden", 128))  # YAML 키는 num_hidden 유지
    feature_mode = cfg.get("coral", {}).get("feature", "cls")  # model.py는 cls/pooled 지원
    coral_lambda = float(cfg.get("coral", {}).get("lambda", 0.1))

    use_num = cfg["exp"]["mode"] == "text_num"
    num_cols = cfg["columns"].get("num", []) if use_num else []

    if use_num:
        model = BertTextNumCoralClassifier(
            model_name=model_name,
            num_features=len(num_cols),
            num_labels=2,
            dropout=dropout,
            hidden_num=hidden_num,
        ).to(device)
    else:
        model = BertTextCoralClassifier(
            model_name=model_name,
            num_labels=2,
            dropout=dropout,
        ).to(device)


    # -------------------------
    # optimizer / amp
    # -------------------------
    lr = float(cfg["train"]["lr"])
    eps = float(cfg["train"].get("eps", 1e-8))
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))
    epochs = int(cfg["train"]["epochs"])

    use_amp = bool(cfg["train"].get("mixed_precision", False)) and device.type == "cuda"
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
    scaler_amp = make_amp_scaler(use_amp)

    # -------------------------
    # train loop
    # -------------------------
    best_score = -math.inf
    best_epoch: Optional[int] = None

    train_log_csv = out_dir / "train_log.csv"

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch_amp_coral(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler_amp=scaler_amp,
            coral_lambda=coral_lambda,
            max_grad_norm=max_grad_norm,
        )

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"loss={tr['loss']:.4f} cls={tr['loss_cls']:.4f} coral={tr['loss_coral']:.4f} acc={tr['acc']:.4f}"
        )

        val_score = ""
        if val_loader is not None:
            rep = evaluate_domain_unweighted_mean(
                model=model,
                loader=val_loader,
                device=device,
                id2domain=dom_map.id2domain,
                metric=cfg["val"].get("metric", "f1_macro"),
            )

            # baseline/dann 스타일: val_epoch{e}.json
            safe_json_dump(rep, out_dir / f"val_epoch{epoch}.json")

            score = float(rep["domain_unweighted_mean"])
            val_score = score

            print(
                f"[Val] metric={rep['metric']} "
                f"domain_unweighted_mean={score:.4f} num_domains={rep['num_domains']}"
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch

                save_checkpoint(
                    out_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    extra={
                        "model_name": model_name,
                        "num_cols": num_cols if use_num else [],
                        "hidden_num": hidden_num,
                        "feature_mode": feature_mode,
                        "coral_lambda": coral_lambda,
                        "seed": seed,
                        "algo": cfg["exp"]["algo"],
                        "mode": cfg["exp"]["mode"],
                        "run_id": cfg["exp"]["run_id"],
                        "best_metric": rep["metric"],
                        "best_score": float(best_score),
                        "best_epoch": int(best_epoch),
                    },
                )
                print(f"Saved best: {out_dir / 'best.pt'} (epoch={best_epoch}, score={best_score:.4f})")

        # train_log.csv (매 epoch append)
        append_train_log_csv(
            train_log_csv,
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_loss_cls": tr["loss_cls"],
                "train_loss_coral": tr["loss_coral"],
                "train_acc": tr["acc"],
                "val_domain_unweighted_mean": val_score,
                "best_score_so_far": best_score if best_score > -math.inf else "",
            },
        )

    # best 요약 저장 (baseline과 동일한 느낌)
    if best_epoch is not None:
        safe_json_dump({"best_score": float(best_score), "best_epoch": int(best_epoch)}, out_dir / "best.json")

    print("Done.")
    if best_epoch is not None:
        print(f"Best epoch={best_epoch}, best_score={best_score:.4f}")
        print(f"Checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
