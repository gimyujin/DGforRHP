# erm/main_train.py
import argparse
from pathlib import Path
import csv
import yaml
import torch
from transformers import AutoTokenizer

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

from utils import set_seed, build_adamw_optimizer, get_val_cfg, safe_json_dump, make_outdir
from data_loader import SplitPaths, load_splits_with_domain, validate_columns, build_dataloader, save_scaler
from evaluate_erm import evaluate
from train import train_one_epoch_amp


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    exp = cfg.get("exp", {})
    algo = exp.get("algo", "erm")
    run_id = exp.get("run_id", "v1")

    if algo != "erm":
        raise ValueError("This entrypoint is for ERM only. exp.algo must be 'erm'.")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_dir = Path(cfg["data"]["data_dir"])
    text_col = cfg["columns"]["text"]
    label_col = cfg["columns"]["label"]
    domain_col = cfg["columns"]["domain"]
    num_cols = cfg["columns"].get("num", []) or []

    model_name = cfg["model"]["name"]
    max_length = int(cfg["model"]["max_length"])
    dropout = float(cfg["model"]["dropout"])

    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])
    eps = float(cfg["train"]["eps"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    num_workers = int(cfg["train"].get("num_workers", 2))
    use_amp = bool(cfg["train"].get("mixed_precision", False))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))

    val_cfg = get_val_cfg(cfg)
    metric = val_cfg.get("metric", "f1_macro")

    model_tag = "bert_text_num" if num_cols else "bert_text_only"
    outdir = make_outdir(cfg["output"]["dir"], algo=algo, model_tag=model_tag, run_id=run_id)
    print("Outputs:", outdir)

    safe_json_dump(cfg, outdir / "config_used.json")

    log_path = outdir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_score"])

    paths = SplitPaths(
        train_path=str(data_dir / cfg["data"]["train_file"]),
        val_path=str(data_dir / cfg["data"]["val_file"]),
        test_path=str(data_dir / cfg["data"]["test_file"]),
    )

    splits, domain2id, id2domain, scaler = load_splits_with_domain(
        paths,
        text_col=text_col,
        label_col=label_col,
        domain_col=domain_col,
        num_cols=num_cols if num_cols else None,
        scaler=None,
    )

    if scaler is not None:
        save_scaler(scaler, str(outdir / "scaler.pkl"))

    safe_json_dump(domain2id, outdir / "domain_vocab.json")

    required = [text_col, label_col, domain_col, "domain_id"] + (num_cols if num_cols else [])
    for split_name, df in splits.items():
        validate_columns(df, required)
        print(f"{split_name}: n={len(df):,}, domains={df[domain_col].nunique()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    train_loader = build_dataloader(
        splits["train"], tokenizer,
        batch_size=batch_size, shuffle=True,
        text_col=text_col, label_col=label_col,
        num_cols=num_cols if num_cols else None,
        max_length=max_length, num_workers=num_workers
    )
    val_loader = build_dataloader(
        splits["val"], tokenizer,
        batch_size=batch_size, shuffle=False,
        text_col=text_col, label_col=label_col,
        num_cols=num_cols if num_cols else None,
        max_length=max_length, num_workers=num_workers
    )
    test_loader = build_dataloader(
        splits["test"], tokenizer,
        batch_size=batch_size, shuffle=False,
        text_col=text_col, label_col=label_col,
        num_cols=num_cols if num_cols else None,
        max_length=max_length, num_workers=num_workers
    )

    if num_cols:
        from model import BertTextNumClassifier
        model = BertTextNumClassifier(
            model_name=model_name,
            num_features=len(num_cols),
            num_labels=2,
            dropout=dropout
        ).to(device)
    else:
        from model import BertTextClassifier
        model = BertTextClassifier(
            model_name=model_name,
            num_labels=2,
            dropout=dropout
        ).to(device)

    optimizer = build_adamw_optimizer(model, lr=lr, eps=eps, weight_decay=weight_decay)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    print("AMP enabled:", scaler_amp.is_enabled())

    best_val = -1.0
    best_path = outdir / "best.pt"

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch_amp(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler_amp=scaler_amp,
            max_grad_norm=max_grad_norm,
        )
        train_loss = float(tr["loss"])
        train_acc = float(tr["acc"])

        va = evaluate(model, val_loader, device, metric=metric)
        val_score = float(va["score"])

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 6), round(train_acc, 6), round(val_score, 6)])

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_score={val_score:.4f}")

        safe_json_dump(va, outdir / f"val_epoch{epoch}.json")

        if val_score > best_val:
            best_val = val_score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_score": best_val,
                    "num_cols": num_cols,
                    "algo": algo,
                    "model_tag": model_tag,
                },
                best_path,
            )
            print("Saved best:", best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    te = evaluate(model, test_loader, device, metric=metric)
    print(f"[Test] score={float(te['score']):.4f}")
    safe_json_dump(te, outdir / "test_report.json")


if __name__ == "__main__":
    main()