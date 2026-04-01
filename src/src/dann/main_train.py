# src/dann/main_train.py
from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
import yaml
from transformers import AutoTokenizer

from common.data_loader import SplitPaths, load_splits_with_domain, build_dataloader, save_scaler
from common.evaluate_dann import evaluate_by_domain_dann
from dann.model import DANNModel
from dann.train import train_one_epoch_dann


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # baseline과 동일하게 config_used.json 저장
    _write_json(outdir / "config_used.json", cfg)

    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load splits (domain_id + scaler)
    # -------------------------
    paths = SplitPaths(
        train_path=cfg["paths"]["train"],
        val_path=cfg["paths"]["val"],
        test_path=cfg["paths"]["test"],
    )

    text_col = cfg.get("text_col", "text")
    label_col = cfg.get("label_col", "label")
    domain_col = cfg.get("domain_col", "domain")

    use_numeric = bool(cfg.get("use_numeric", False))
    num_cols = cfg.get("num_cols", []) if use_numeric else None

    splits, domain2id, id2domain, fitted_scaler = load_splits_with_domain(
        paths=paths,
        text_col=text_col,
        label_col=label_col,
        domain_col=domain_col,
        num_cols=num_cols,
        scaler=None,
    )

    # baseline과 동일하게 domain_vocab.json 저장 (domain2id)
    _write_json(outdir / "domain_vocab.json", domain2id)

    # scaler 저장 (text_num이면 생성됨)
    if fitted_scaler is not None:
        save_scaler(fitted_scaler, str(outdir / "scaler.pkl"))

    # -------------------------
    # Tokenizer / DataLoaders
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder_name"])

    train_loader = build_dataloader(
        df=splits["train"],
        tokenizer=tokenizer,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        text_col=text_col,
        label_col=label_col,
        num_cols=num_cols,
        max_length=int(cfg["max_length"]),
        num_workers=int(cfg.get("num_workers", 2)),
    )
    val_loader = build_dataloader(
        df=splits["val"],
        tokenizer=tokenizer,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        text_col=text_col,
        label_col=label_col,
        num_cols=num_cols,
        max_length=int(cfg["max_length"]),
        num_workers=int(cfg.get("num_workers", 2)),
    )
    test_loader = build_dataloader(
        df=splits["test"],
        tokenizer=tokenizer,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        text_col=text_col,
        label_col=label_col,
        num_cols=num_cols,
        max_length=int(cfg["max_length"]),
        num_workers=int(cfg.get("num_workers", 2)),
    )

    # -------------------------
    # Model / Optim
    # -------------------------
    model = DANNModel(
        encoder_name=cfg["encoder_name"],
        num_labels=int(cfg.get("num_labels", 2)),
        num_domains=len(domain2id),
        use_numeric=use_numeric,
        num_numeric=(len(num_cols) if num_cols else 0),
        dropout=float(cfg.get("dropout", 0.1)),
        hidden_num=int(cfg.get("hidden_num", 128)),
        head_hidden=int(cfg.get("head_hidden", 256)),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )

    use_amp = bool(cfg.get("use_amp", True))
    scaler = GradScaler(enabled=use_amp)

    epochs = int(cfg["epochs"])
    total_steps = epochs * len(train_loader)
    global_step = 0

    best_score = -1.0
    best_path = outdir / "best.pt"

    # -------------------------
    # train_log.csv (baseline 스타일)
    # -------------------------
    log_path = outdir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "loss",
            "loss_y",
            "loss_d",
            "val_metric",
            "val_domain_unweighted_mean",
            "num_domains",
            "global_step",
        ])

    val_metric = cfg.get("val_metric", "f1_macro")

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch_dann(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            domain_loss_weight=float(cfg["domain_loss_weight"]),
            global_step=global_step,
            total_steps=total_steps,
            max_grl_lambda=float(cfg.get("max_grl_lambda", 1.0)),
            grl_gamma=float(cfg.get("grl_gamma", 10.0)),
            max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
            scaler=scaler,
        )
        global_step = int(tr["global_step"])

        # -------------------------
        # DG val (domain별 + 균등 평균)
        # -------------------------
        val_report = evaluate_by_domain_dann(
            model=model,
            loader=val_loader,
            device=device,
            id2domain=id2domain,
            metric=val_metric,
        )

        # baseline처럼 val_epoch{n}.json 저장
        _write_json(outdir / f"val_epoch{epoch}.json", val_report)

        val_score = float(val_report["domain_unweighted_mean"])

        # train_log.csv append
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{tr['loss']:.6f}",
                f"{tr['loss_y']:.6f}",
                f"{tr['loss_d']:.6f}",
                val_report["metric"],
                f"{val_score:.6f}",
                int(val_report["num_domains"]),
                global_step,
            ])

        # best selection: domain_unweighted_mean
        if val_score > best_score:
            best_score = val_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_domain_unweighted_mean": best_score,
                    "epoch": epoch,
                    "config": cfg,
                },
                best_path,
            )

    # -------------------------
    # Final test with best.pt
    # -------------------------
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    test_report = evaluate_by_domain_dann(
        model=model,
        loader=test_loader,
        device=device,
        id2domain=id2domain,
        metric=cfg.get("test_metric", val_metric),
    )

    # baseline처럼 test_domain_report.json 저장
    _write_json(outdir / "test_domain_report.json", test_report)


if __name__ == "__main__":
    main()
