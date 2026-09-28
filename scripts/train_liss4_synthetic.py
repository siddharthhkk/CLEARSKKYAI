import argparse
import csv
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from liss4 import normalize_liss4
from liss4_dsen2cr import LISS4DSen2CR


class NPZLISS4Dataset(Dataset):
    def __init__(self, manifest, allowed_scenes):
        self.manifest = os.path.abspath(manifest)
        self.root = os.path.dirname(self.manifest)
        with open(self.manifest, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if allowed_scenes:
            self.rows = [
                r for r in rows
                if os.path.abspath(r["scene"]) in allowed_scenes
            ]
        else:
            self.rows = rows

        if not self.rows:
            raise ValueError("No samples remain after scene filtering.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        p = os.path.join(self.root, f"sample_{int(r['idx']):06d}.npz")
        with np.load(p) as d:
            cloudy = d["cloudy"].astype(np.float32)
            clear = d["clear"].astype(np.float32)
            mask = d["mask"].astype(np.float32)
        return {
            "cloudy": torch.from_numpy(cloudy),
            "clear": torch.from_numpy(clear),
            "mask": torch.from_numpy(mask),
        }


def datasets_from_manifests(manifests):
    paths = [os.path.abspath(path) for path in manifests]
    if len(set(paths)) != len(paths):
        raise ValueError("A training or validation manifest was provided more than once.")

    datasets = [NPZLISS4Dataset(path, set()) for path in paths]
    combined = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    source = paths[0] if len(paths) == 1 else paths
    return combined, source, paths


def masked_l1(pred, target, mask):
    mask = mask.expand_as(pred)
    return torch.sum(torch.abs(pred - target) * mask) / mask.sum().clamp_min(1.0)


def main():
    ap = argparse.ArgumentParser(
        description="Train DSen2-CR-style LISS-IV model on synthetic native-LISS-IV clouds."
    )
    ap.add_argument("--manifest", default="data/synthetic_pretrain_v3r/manifest.csv")
    ap.add_argument(
        "--train-manifest",
        action="append",
        default=None,
        help="Training manifest; repeat to combine independent datasets.",
    )
    ap.add_argument(
        "--val-manifest",
        action="append",
        default=None,
        help="Validation manifest; repeat to combine independent datasets.",
    )
    ap.add_argument("--train-scene", action="append")
    ap.add_argument("--val-scene", action="append")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--features", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--lambda-cloud", type=float, default=2.0)
    ap.add_argument("--output", default="weights/liss4_dsen2cr_synthetic.pth")
    ap.add_argument("--latest-output", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be at least 1.")
    if args.features < 1 or args.blocks < 1:
        raise ValueError("--features and --blocks must be at least 1.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive.")
    if not math.isfinite(args.lambda_cloud) or args.lambda_cloud < 0:
        raise ValueError("--lambda-cloud must be finite and non-negative.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    train_source = None
    val_source = None

    if args.train_manifest or args.val_manifest:
        if not (args.train_manifest and args.val_manifest):
            raise ValueError("Provide both --train-manifest and --val-manifest.")
        train_ds, train_source, train_paths = datasets_from_manifests(
            args.train_manifest
        )
        val_ds, val_source, val_paths = datasets_from_manifests(args.val_manifest)
        if set(train_paths) & set(val_paths):
            raise ValueError("A manifest cannot appear in both train and validation.")
    else:
        if args.train_scene or args.val_scene:
            if not args.train_scene or not args.val_scene:
                raise ValueError(
                    "Provide both --train-scene and --val-scene when using scene filters."
                )

            train_scenes = {os.path.abspath(p) for p in args.train_scene}
            val_scenes = {os.path.abspath(p) for p in args.val_scene}
            if train_scenes & val_scenes:
                raise ValueError("A scene cannot appear in both train and validation.")

            manifest = os.path.abspath(args.manifest)
            train_ds = NPZLISS4Dataset(manifest, train_scenes)
            val_ds = NPZLISS4Dataset(manifest, val_scenes)
            train_source = sorted(train_scenes)
            val_source = sorted(val_scenes)
        else:
            manifest = os.path.abspath(args.manifest)
            manifest_dir = os.path.dirname(manifest)
            train_manifest = os.path.join(manifest_dir, "train_manifest.csv")
            val_manifest = os.path.join(manifest_dir, "val_manifest.csv")
            missing = [p for p in (train_manifest, val_manifest) if not os.path.isfile(p)]
            if missing:
                raise FileNotFoundError(
                    "Default training needs sibling train_manifest.csv and "
                    "val_manifest.csv files. Missing: " + ", ".join(missing)
                )
            train_ds = NPZLISS4Dataset(train_manifest, set())
            val_ds = NPZLISS4Dataset(val_manifest, set())
            train_source = train_manifest
            val_source = val_manifest

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = LISS4DSen2CR(
        features=args.features,
        blocks=args.blocks,
        res_scale=0.1,
        use_sar=False,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters   : {params:,}")
    print(
        f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | "
        f"Batch: {args.batch_size} | Accum: {args.grad_accum}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_l1 = nn.L1Loss()
    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=device.type == "cuda",
        )
    except (AttributeError, TypeError):
        # Keep compatibility with early PyTorch 2.x versions supported by
        # requirements.txt, where the CUDA scaler lived under torch.cuda.amp.
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best = -float("inf")
    start_epoch = 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    latest_output = args.latest_output
    if latest_output is None:
        base, ext = os.path.splitext(args.output)
        latest_output = base + "_latest" + ext

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if ckpt.get("val_source") == val_source:
            best = float(ckpt.get("best_val_psnr", best))
        else:
            print(
                "Validation source changed; resetting best validation PSNR "
                "for the new split."
            )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resuming from epoch {start_epoch}")

    checkpoint_metadata = {
        "train_source": train_source,
        "val_source": val_source,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "lambda_cloud": args.lambda_cloud,
        "initial_checkpoint": os.path.abspath(args.resume) if args.resume else None,
    }

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        seen = 0
        t0 = time.time()

        for step, batch in enumerate(train_dl, start=1):
            cloudy = normalize_liss4(batch["cloudy"], 1023.0).to(
                device, non_blocking=True
            )
            clear = normalize_liss4(batch["clear"], 1023.0).to(
                device, non_blocking=True
            )
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
            ):
                # Do not clamp during training. Because the model has a
                # cloudy-image long skip, cloud pixels start near 1.0.
                # Clamping there can create zero gradient and prevent
                # the residual branch from learning to remove clouds.
                pred = model(cloudy)

                base = loss_l1(pred, clear)
                cloud = masked_l1(pred, clear, mask)
                loss = (base + args.lambda_cloud * cloud) / args.grad_accum

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0 or step == len(train_dl):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total += float(loss.item()) * args.grad_accum * cloudy.shape[0]
            seen += cloudy.shape[0]

        model.eval()
        sse = 0.0
        n = 0

        with torch.no_grad():
            for batch in val_dl:
                cloudy = normalize_liss4(batch["cloudy"], 1023.0).to(device)
                clear = normalize_liss4(batch["clear"], 1023.0).to(device)

                with torch.autocast(
                    device_type="cuda",
                    enabled=device.type == "cuda",
                ):
                    pred = torch.clamp(model(cloudy), 0.0, 1.0)

                d = pred.float() - clear.float()
                sse += torch.sum(d * d).item()
                n += d.numel()

        mse = sse / max(1, n)
        val_psnr = (
            20.0 * np.log10(1.0 / np.sqrt(mse))
            if mse > 0
            else float("inf")
        )

        avg = total / max(1, seen)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss {avg:.6f} | "
            f"val_PSNR {val_psnr:.3f} dB | "
            f"time {time.time()-t0:.1f}s"
        )

        if val_psnr > best:
            best = val_psnr
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "features": args.features,
                    "blocks": args.blocks,
                    "dn_max": 1023.0,
                    "best_val_psnr": best,
                    "epoch": epoch,
                    **checkpoint_metadata,
                },
                args.output,
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "features": args.features,
                "blocks": args.blocks,
                "dn_max": 1023.0,
                "best_val_psnr": best,
                **checkpoint_metadata,
            },
            latest_output,
        )

    print()
    print(f"Best validation PSNR: {best:.3f} dB")
    print(f"Best checkpoint     : {os.path.abspath(args.output)}")
    print(f"Latest checkpoint   : {os.path.abspath(latest_output)}")
    print(
        "This is synthetic-cloud pretraining. It is not the final real-cloud benchmark."
    )


if __name__ == "__main__":
    main()
