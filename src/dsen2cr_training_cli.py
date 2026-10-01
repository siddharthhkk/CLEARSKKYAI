from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dsen2cr import DSen2CR, load_dsen2cr_weights
from dsen2cr_training import SEN12MSCRDataset, read_training_samples, carl_loss


def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    loss_sum = mae_sum = mse_sum = 0.0
    batches = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            cloudy = batch["cloudy"].to(device, non_blocking=True)
            clear = batch["clear"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            prediction = model(inputs)
            loss_sum += float(carl_loss(prediction, cloudy, clear, mask).item())
            estimate = prediction.clamp(0.0, 5.0) / 5.0
            target = clear.clamp(0.0, 5.0) / 5.0
            difference = estimate - target
            mae_sum += float(difference.abs().mean().item())
            mse_sum += float(difference.square().mean().item())
            batches += 1
    if not batches:
        raise ValueError("Validation split produced no batches.")
    rmse = (mse_sum / batches) ** 0.5
    psnr = float("inf") if rmse == 0 else 20.0 * np.log10(1.0 / rmse)
    return {
        "carl_loss": loss_sum / batches,
        "mae_0_1": mae_sum / batches,
        "rmse_0_1": rmse,
        "psnr_db": float(psnr),
    }


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    best_val_loss: float,
    config: dict,
    training_config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
            "training_config": training_config,
            "trained_with": "ClearSky-AI PyTorch training entry point",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": "New checkpoint produced by this run; see training_config for data and run settings.",
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train DSen2-CR in PyTorch. Accepts upstream datasetfilelist.csv or a "
            "manifest with cloudy_s2, sar_s1, clear_target, and split columns."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Dataset file list / manifest CSV")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Root for paths in upstream file lists")
    parser.add_argument("--output", type=Path, default=Path("weights/dsen2cr_sar_carl_trained.pth"))
    parser.add_argument("--resume", type=Path, help="Resume model and optimizer from a training checkpoint")
    parser.add_argument("--init-checkpoint", type=Path, help="Initialize from a compatible pretrained state dict")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--features", type=int, default=256, help="Architecture width; 256 matches standard DSen2-CR")
    parser.add_argument("--blocks", type=int, default=16, help="Residual block count; 16 matches standard DSen2-CR")
    parser.add_argument("--learning-rate", type=float, default=7e-5)
    parser.add_argument("--cloud-threshold", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is robust on Windows)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.crop_size < 16:
        parser.error("epochs and batch-size must be positive; crop-size must be at least 16")
    if args.features < 1 or args.blocks < 1 or args.learning_rate <= 0:
        parser.error("features, blocks, and learning-rate must be positive")
    if args.resume and args.init_checkpoint:
        parser.error("choose --resume or --init-checkpoint, not both")
    if args.output.exists() and not args.resume:
        parser.error(f"output already exists; choose another --output or resume explicitly: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but PyTorch cannot access a CUDA device")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    samples = read_training_samples(args.manifest, args.data_root)
    train_set = SEN12MSCRDataset(samples, "train", args.crop_size, args.cloud_threshold, augment=True)
    val_set = SEN12MSCRDataset(samples, "val", args.crop_size, args.cloud_threshold, augment=False)
    if args.workers < 0:
        parser.error("workers cannot be negative")
    for sample in train_set.samples + val_set.samples:
        for path in (sample.cloudy_path, sample.sar_path, sample.clear_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {sample.split} sample file: {path}")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model_config = {"features": args.features, "blocks": args.blocks, "res_scale": 0.1}
    training_config = {
        "manifest": str(args.manifest.resolve()),
        "data_root": str(args.data_root.resolve()),
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "crop_size": args.crop_size,
        "learning_rate": args.learning_rate,
        "cloud_threshold": args.cloud_threshold,
        "seed": args.seed,
        "device": str(device),
        "initial_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
    }
    model = DSen2CR(**model_config).to(device)
    if args.init_checkpoint:
        load_dsen2cr_weights(model, str(args.init_checkpoint), map_location=device)
    optimizer = torch.optim.NAdam(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8
    )
    start_epoch, best_val_loss = 1, float("inf")
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=True)
        if not isinstance(resume, dict) or "state_dict" not in resume:
            raise ValueError("--resume expects a checkpoint written by this PyTorch train.py")
        if resume.get("config") != model_config:
            raise ValueError(f"Checkpoint architecture {resume.get('config')} does not match {model_config}.")
        model.load_state_dict(resume["state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_val_loss = float(resume.get("best_val_loss", float("inf")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    history_path = args.output.with_name(args.output.stem + "_history.csv")
    latest_path = args.output.with_name(args.output.stem + "_latest.pth")
    print(
        f"PyTorch DSen2-CR training: {len(train_set)} train, {len(val_set)} validation "
        f"triplets; {device}; architecture={model_config}; output={args.output}",
        flush=True,
    )
    if not args.resume:
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                ("epoch", "train_carl_loss", "val_carl_loss", "val_mae_0_1", "val_rmse_0_1", "val_psnr_db")
            )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        batches = 0
        for batch in train_loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            cloudy = batch["cloudy"].to(device, non_blocking=True)
            clear = batch["clear"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = carl_loss(prediction, cloudy, clear, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite CARL loss at epoch {epoch}, batch {batches + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().item())
            batches += 1
        if not batches:
            raise ValueError("Training split produced no batches.")

        train_loss = train_loss_sum / batches
        validation = evaluate(model, val_loader, device)
        if validation["carl_loss"] < best_val_loss:
            best_val_loss = validation["carl_loss"]
            save_checkpoint(
                args.output, model, optimizer, epoch, best_val_loss, model_config, training_config
            )
        save_checkpoint(
            latest_path, model, optimizer, epoch, best_val_loss, model_config, training_config
        )
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                (epoch, train_loss, validation["carl_loss"], validation["mae_0_1"], validation["rmse_0_1"], validation["psnr_db"])
            )
        print(
            f"epoch {epoch}/{args.epochs} train CARL={train_loss:.6f} "
            f"val CARL={validation['carl_loss']:.6f} MAE={validation['mae_0_1']:.5f} "
            f"PSNR={validation['psnr_db']:.2f} dB; best={best_val_loss:.6f}",
            flush=True,
        )
    print(f"Best validation checkpoint: {args.output}")
    print(f"Latest/resume checkpoint: {latest_path}")
    print(f"Training history: {history_path}")

if __name__ == "__main__":
    main()
