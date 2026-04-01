# baseline/main_train_mixup.py
import argparse
from pathlib import Path
import csv
import yaml
import torch
from transformers import AutoTokenizer

# common import (패키지화 안 하고도 동작하도록)
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

from utils import set_seed, build_adamw_optimizer, get_val_cfg, safe_json_dump, make_outdir
from data_loader import SplitPaths, load_splits_with_domain, validate_columns, build_dataloader, save_scaler
from evaluate import evaluate_by_domain
from train_mixup import train_one_epoch_mixup_amp


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
    run_id = exp.get("run_id", "v1")

    # mixup 전용: outdir / ckpt 메타에 algo를 mixup으로 고정
    algo = "mixup"

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----- config values
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

    # mixup hyperparam
    num_labels = int(cfg["model"].get("num_labels", 2))  # 기본 2

    val_cfg = get_val_cfg(cfg)
    metric = val_cfg.get("metric", "f1_macro")

    model_tag = "bert_text_num" if num_cols else "bert_text_only"
    outdir = make_outdir(cfg["output"]["dir"], algo=algo, model_tag=model_tag, run_id=run_id)
    print("Outputs:", outdir)

    safe_json_dump(cfg, outdir / "config_used.json")

    # ----- init log
    log_path = outdir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_domain_mean"])

    # ----- load data
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

    # ----- tokenizer & loaders
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

    # ----- model (encode/classify 가 있는 버전이어야 함)
    if num_cols:
        from model_mixup import BertTextNumClassifierMixup
        model = BertTextNumClassifierMixup(
            model_name=model_name,
            num_features=len(num_cols),
            num_labels=num_labels,
            dropout=dropout,
            # hidden_num 쓰면 여기 넣기
        ).to(device)

    else:
        from model_mixup import BertTextClassifierMixup
        model = BertTextClassifierMixup(
            model_name=model_name,
            num_labels=num_labels,
            dropout=dropout,
        ).to(device)

    optimizer = build_adamw_optimizer(model, lr=lr, eps=eps, weight_decay=weight_decay)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    print("AMP enabled:", scaler_amp.is_enabled())

    best_val = -1.0
    best_path = outdir / "best.pt"

    # ----- train loop (mixup only)
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch_mixup_amp(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler_amp=scaler_amp,
            num_labels=num_labels,
            max_grad_norm=max_grad_norm,
        )

        train_loss = float(tr["loss"])
        train_acc = float(tr["acc"])

        va = evaluate_by_domain(model, val_loader, device, id2domain=id2domain, metric=metric)
        val_score = float(va["domain_unweighted_mean"])

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 6), round(train_acc, 6), round(val_score, 6)])

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_domain_mean={val_score:.4f}")
        safe_json_dump(va, outdir / f"val_epoch{epoch}.json")

        if val_score > best_val:
            best_val = val_score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_domain_unweighted_mean": best_val,
                    "num_cols": num_cols,
                    "algo": algo,
                    "model_tag": model_tag,
                },
                best_path,
            )
            print("Saved best:", best_path)

    # ----- test best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    te = evaluate_by_domain(model, test_loader, device, id2domain=id2domain, metric=metric)
    print(f"[Test] domain_mean={float(te['domain_unweighted_mean']):.4f}")
    safe_json_dump(te, outdir / "test_domain_report.json")


if __name__ == "__main__":
    main()
