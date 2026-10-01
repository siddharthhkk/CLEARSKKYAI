"""Train the LISS-IV reconstruction model on paired cloudy/clear GeoTIFFs."""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from liss4_dsen2cr import LISS4DSen2CR
from liss4_diffcr_transfer import LISS4DiffCRTransfer, resolve_diffcr_root
from liss4_window_dataset import LISS4WindowPairDataset


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(name)


def masked_l1(prediction, target, valid, cloud_mask, has_mask):
    valid = valid.expand_as(prediction)
    errors = torch.abs(prediction.float() - target.float())
    base = torch.sum(errors * valid) / valid.sum().clamp_min(1.0)

    labeled_cloud = (
        cloud_mask.float()
        * valid[:, :1]
        * has_mask.float().view(-1, 1, 1, 1)
    ).expand_as(prediction)
    cloud_loss = torch.sum(errors * labeled_cloud) / labeled_cloud.sum().clamp_min(1.0)
    return base, cloud_loss


def validate(model, loader, device):
    model.eval()
    abs_error = 0.0
    squared_error = 0.0
    valid_values = 0.0
    cloud_abs_error = 0.0
    cloud_values = 0.0

    with torch.no_grad():
        for batch in loader:
            cloudy = batch["cloudy"].to(device, non_blocking=True)
            clear = batch["clear"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            has_mask = batch["has_mask"].to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
            ):
                prediction = torch.clamp(model(cloudy), 0.0, 1.0)

            error = prediction.float() - clear.float()
            expanded_valid = valid.expand_as(error)
            abs_error += torch.sum(torch.abs(error) * expanded_valid).item()
            squared_error += torch.sum(error.square() * expanded_valid).item()
            valid_values += expanded_valid.sum().item()

            cloud_valid = (
                mask
                * valid
                * has_mask.float().view(-1, 1, 1, 1)
            ).expand_as(error)
            cloud_abs_error += (
                torch.sum(torch.abs(error) * cloud_valid).item()
            )
            cloud_values += cloud_valid.sum().item()

    mae = abs_error / max(1.0, valid_values)
    mse = squared_error / max(1.0, valid_values)
    psnr = 20.0 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else float("inf")
    cloud_mae = (
        cloud_abs_error / cloud_values if cloud_values > 0 else float("nan")
    )
    return {
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "cloud_mae": cloud_mae,
    }


def write_history(path, row):
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the project LISS-IV model from co-registered cloudy/clear "
            "LISS-IV GeoTIFF pairs."
        )
    )
    parser.add_argument("--manifest", required=True, help="CSV with train/val rows")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--dn-max", type=float, default=1023.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Default: 2e-4 from scratch, 1e-5 for DiffCR fine-tuning.",
    )
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument(
        "--model-type",
        choices=("scratch", "diffcr-transfer"),
        default="scratch",
        help="Use the local residual CNN or fine-tune a pretrained DiffCR denoiser.",
    )
    parser.add_argument(
        "--pretrained-diffcr",
        default=None,
        help="DiffCR checkpoint, e.g. external/DiffCR/pretrained/diffcr_new.pth.",
    )
    parser.add_argument(
        "--diffcr-root",
        default=None,
        help="DiffCR source repository root; auto-detected in common locations.",
    )
    parser.add_argument(
        "--cloud-weight",
        type=float,
        default=2.0,
        help="Extra L1 loss weight over pixels marked cloudy in an optional mask.",
    )
    parser.add_argument("--output", default="weights/liss4_paired.pth")
    parser.add_argument("--latest-output", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--allow-aoi-overlap",
        action="store_true",
        help="Allow an AOI/source scene to appear in both train and validation.",
    )
    args = parser.parse_args()

    for name, value in (
        ("epochs", args.epochs),
        ("batch-size", args.batch_size),
        ("patch-size", args.patch_size),
        ("features", args.features),
        ("blocks", args.blocks),
    ):
        if value < 1:
            raise ValueError(f"--{name} must be at least 1")
    if args.stride < 1 or args.stride > args.patch_size:
        raise ValueError("--stride must be between 1 and --patch-size")
    if not math.isfinite(args.dn_max) or args.dn_max <= 0:
        raise ValueError("--dn-max must be finite and positive")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError("--lr must be finite and positive")
    if args.model_type == "scratch" and args.pretrained_diffcr:
        raise ValueError("Use --model-type diffcr-transfer with --pretrained-diffcr")
    if args.model_type == "diffcr-transfer" and not args.pretrained_diffcr and not args.resume:
        raise ValueError(
            "A new DiffCR transfer run needs --pretrained-diffcr. For a resumed "
            "run, pass --resume with the existing DiffCR checkpoint."
        )
    if not math.isfinite(args.cloud_weight) or args.cloud_weight < 0:
        raise ValueError("--cloud-weight must be finite and non-negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Device         : {device}")

    train_data = LISS4WindowPairDataset(
        args.manifest,
        split="train",
        patch_size=args.patch_size,
        stride=args.stride,
        dn_max=args.dn_max,
        augment=True,
    )
    val_data = LISS4WindowPairDataset(
        args.manifest,
        split="val",
        patch_size=args.patch_size,
        stride=args.stride,
        dn_max=args.dn_max,
        augment=False,
    )

    train_aois = {row["aoi"] for row in train_data.rows}
    val_aois = {row["aoi"] for row in val_data.rows}
    shared_aois = train_aois & val_aois
    train_paths = {
        row[field]
        for row in train_data.rows
        for field in ("cloudy", "clear")
    }
    val_paths = {
        row[field]
        for row in val_data.rows
        for field in ("cloudy", "clear")
    }
    shared_paths = train_paths & val_paths
    if (shared_aois or shared_paths) and not args.allow_aoi_overlap:
        details = []
        if shared_aois:
            details.append("AOIs: " + ", ".join(sorted(shared_aois)))
        if shared_paths:
            details.append("source raster paths: " + ", ".join(sorted(shared_paths)))
        raise ValueError(
            "Train and validation overlap (" + "; ".join(details) + "). "
            + "Split by independent scenes/AOIs or pass --allow-aoi-overlap "
            "if this is intentional."
        )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)

    diffcr_root = None
    pretrained_report = None
    if args.model_type == "diffcr-transfer":
        diffcr_root = resolve_diffcr_root(args.diffcr_root)
        model = LISS4DiffCRTransfer(
            diffcr_root=diffcr_root,
            pretrained_checkpoint=args.pretrained_diffcr,
        )
        pretrained_report = model.pretrained_report
        learning_rate = args.lr if args.lr is not None else 1e-5
    else:
        model = LISS4DSen2CR(
            features=args.features,
            blocks=args.blocks,
            res_scale=0.1,
            use_sar=False,
        )
        learning_rate = args.lr if args.lr is not None else 2e-4
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters     : {parameter_count:,}")
    print(f"Model type     : {args.model_type}")
    print(f"Learning rate  : {learning_rate:g}")
    if pretrained_report:
        print(
            "DiffCR init    : "
            f"{pretrained_report['matched_tensors']}/"
            f"{pretrained_report['total_tensors']} tensors, "
            f"{pretrained_report['parameter_coverage']:.1%} parameter coverage"
        )
    print(
        f"Training chips : {len(train_data)} | Validation chips: {len(val_data)} | "
        f"Patch: {args.patch_size} | Batch: {args.batch_size}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if args.latest_output:
        latest_path = os.path.abspath(args.latest_output)
    else:
        output_root, extension = os.path.splitext(output_path)
        latest_path = output_root + "_latest" + extension
    os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    history_path = os.path.splitext(output_path)[0] + "_history.csv"

    start_epoch = 1
    best_psnr = -float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        saved_model_type = checkpoint.get("model_type", "liss4_paired_supervised")
        expected_model_type = (
            "liss4_diffcr_transfer"
            if args.model_type == "diffcr-transfer"
            else "liss4_paired_supervised"
        )
        if saved_model_type != expected_model_type:
            raise ValueError(
                f"Resume checkpoint model_type={saved_model_type!r} does not match "
                f"--model-type {args.model_type!r}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_psnr = float(checkpoint.get("best_val_psnr", best_psnr))
        print(f"Resuming from epoch {start_epoch}")

    train_aois_sorted = sorted(train_aois)
    val_aois_sorted = sorted(val_aois)
    metadata = {
        "features": args.features if args.model_type == "scratch" else None,
        "blocks": args.blocks if args.model_type == "scratch" else None,
        "dn_max": args.dn_max,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "train_aois": train_aois_sorted,
        "val_aois": val_aois_sorted,
        "train_chips": len(train_data),
        "val_chips": len(val_data),
        "cloud_weight": args.cloud_weight,
        "input_bands": ["Green", "Red", "NIR"],
        "uses_sar": False,
        "model_type": (
            "liss4_diffcr_transfer"
            if args.model_type == "diffcr-transfer"
            else "liss4_paired_supervised"
        ),
        "diffcr_root": diffcr_root,
        "pretrained_diffcr_checkpoint": (
            os.path.abspath(args.pretrained_diffcr)
            if args.pretrained_diffcr
            else None
        ),
        "learning_rate": learning_rate,
    }

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        samples_seen = 0
        started = time.time()

        for batch in train_loader:
            cloudy = batch["cloudy"].to(device, non_blocking=True)
            clear = batch["clear"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            cloud_mask = batch["mask"].to(device, non_blocking=True)
            has_mask = batch["has_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
            ):
                prediction = model(cloudy)
                base_loss, cloud_loss = masked_l1(
                    prediction, clear, valid, cloud_mask, has_mask
                )
                loss = base_loss + args.cloud_weight * cloud_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * cloudy.shape[0]
            samples_seen += cloudy.shape[0]

        metrics = validate(model, val_loader, device)
        train_loss = total_loss / max(1, samples_seen)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_L1 {train_loss:.6f} | "
            f"val_MAE {metrics['mae']:.6f} | "
            f"val_PSNR {metrics['psnr']:.3f} dB | "
            f"val_cloud_MAE {metrics['cloud_mae']:.6f} | "
            f"time {time.time() - started:.1f}s"
        )
        write_history(
            history_path,
            {
                "epoch": epoch,
                "train_l1": train_loss,
                "val_mae": metrics["mae"],
                "val_mse": metrics["mse"],
                "val_psnr_db": metrics["psnr"],
                "val_cloud_mae": metrics["cloud_mae"],
                "seconds": time.time() - started,
            },
        )

        model_state = model.state_dict()
        if metrics["psnr"] > best_psnr:
            best_psnr = metrics["psnr"]
            torch.save(
                {
                    "model_state_dict": model_state,
                    "epoch": epoch,
                    "best_val_psnr": best_psnr,
                    **metadata,
                },
                output_path,
            )

        torch.save(
            {
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "best_val_psnr": best_psnr,
                **metadata,
            },
            latest_path,
        )

    print(f"Best checkpoint: {output_path}")
    print(f"Resume state   : {latest_path}")
    print(f"Epoch history  : {history_path}")
    print(
        "This run uses paired cloudy/clear LISS-IV targets. Its real-scene "
        "validity depends on the quality and independence of those pairs."
    )


if __name__ == "__main__":
    main()
