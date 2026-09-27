import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from liss4 import normalize_liss4
from liss4_dsen2cr import LISS4DSen2CR


class NPZLISS4V21Dataset(Dataset):
    """
    Load Synthetic Clouds V2 samples and build a corruption-aware mask.

    Corruption region includes:
      - V2 cloud mask
      - low-opacity cloud fringes
      - translated cloud-shadow pixels
    """

    def __init__(self, manifest):
        self.manifest = os.path.abspath(manifest)
        self.root = os.path.dirname(self.manifest)

        with open(self.manifest, "r", newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        if not self.rows:
            raise ValueError("Manifest is empty.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        p = os.path.join(self.root, f"sample_{int(r['idx']):06d}.npz")

        with np.load(p) as d:
            cloudy = d["cloudy"].astype(np.float32)
            clear = d["clear"].astype(np.float32)

            if "mask" not in d or "alpha" not in d or "shadow" not in d:
                raise ValueError(
                    f"{p} is missing one of mask/alpha/shadow required for V2.1."
                )

            mask = d["mask"].astype(np.float32)
            alpha = d["alpha"].astype(np.float32)
            shadow = d["shadow"].astype(np.float32)

        if cloudy.shape != (3, 256, 256):
            raise ValueError(f"{p}: cloudy shape={cloudy.shape}")
        if clear.shape != (3, 256, 256):
            raise ValueError(f"{p}: clear shape={clear.shape}")
        if mask.shape != (1, 256, 256):
            raise ValueError(f"{p}: mask shape={mask.shape}")
        if alpha.shape != (1, 256, 256):
            raise ValueError(f"{p}: alpha shape={alpha.shape}")
        if shadow.shape != (1, 256, 256):
            raise ValueError(f"{p}: shadow shape={shadow.shape}")

        # Include thin cloud fringes and visible shadow contamination.
        corr = np.maximum(mask, (alpha > 0.05).astype(np.float32))
        corr = np.maximum(corr, (shadow > 0.03).astype(np.float32))

        return {
            "cloudy": torch.from_numpy(cloudy),
            "clear": torch.from_numpy(clear),
            "corr": torch.from_numpy(corr.astype(np.float32)),
        }


def masked_l1(pred, target, mask):
    mask = mask.expand_as(pred)
    return torch.sum(torch.abs(pred - target) * mask) / mask.sum().clamp_min(1.0)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Train LISS4DSen2CR with cloud/shadow-aware reconstruction "
            "and clean-region preservation on Synthetic Clouds V2."
        )
    )
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--features", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--lambda-cloud", type=float, default=2.0)
    ap.add_argument("--lambda-clean", type=float, default=2.0)
    ap.add_argument(
        "--output",
        default="weights/liss4_dsen2cr_synthetic_v21.pth",
    )
    ap.add_argument("--latest-output", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    train_ds = NPZLISS4V21Dataset(args.train_manifest)
    val_ds = NPZLISS4V21Dataset(args.val_manifest)

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
    print(
        f"Loss weights : cloud={args.lambda_cloud:.2f} | "
        f"clean={args.lambda_clean:.2f}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_l1 = nn.L1Loss()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    best = -float("inf")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    latest_output = args.latest_output
    if latest_output is None:
        base, ext = os.path.splitext(args.output)
        latest_output = base + "_latest" + ext

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total = 0.0
        total_cloud = 0.0
        total_clean = 0.0
        seen = 0
        t0 = time.time()

        for step, batch in enumerate(train_dl, start=1):
            cloudy = normalize_liss4(batch["cloudy"], 1023.0).to(
                device,
                non_blocking=True,
            )
            clear = normalize_liss4(batch["clear"], 1023.0).to(
                device,
                non_blocking=True,
            )
            corr = batch["corr"].to(device, non_blocking=True)

            clean_mask = 1.0 - corr

            with torch.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
            ):
                # Do not clamp during training: the long skip can start
                # cloud pixels close to 1.0, and clamping can kill gradients.
                pred = model(cloudy)

                base = loss_l1(pred, clear)
                cloud = masked_l1(pred, clear, corr)
                clean = masked_l1(pred, clear, clean_mask)

                # Base loss + extra emphasis on both corrupted and clean regions.
                loss = (
                    base
                    + args.lambda_cloud * cloud
                    + args.lambda_clean * clean
                ) / args.grad_accum

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0 or step == len(train_dl):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total += float(loss.item()) * args.grad_accum * cloudy.shape[0]
            total_cloud += float(cloud.item()) * cloudy.shape[0]
            total_clean += float(clean.item()) * cloudy.shape[0]
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
        avg_cloud = total_cloud / max(1, seen)
        avg_clean = total_clean / max(1, seen)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss {avg:.6f} | "
            f"cloud_l1 {avg_cloud:.6f} | "
            f"clean_l1 {avg_clean:.6f} | "
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
                    "lambda_cloud": args.lambda_cloud,
                    "lambda_clean": args.lambda_clean,
                    "train_manifest": os.path.abspath(args.train_manifest),
                    "val_manifest": os.path.abspath(args.val_manifest),
                    "version": "v2.1",
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
                "lambda_cloud": args.lambda_cloud,
                "lambda_clean": args.lambda_clean,
                "train_manifest": os.path.abspath(args.train_manifest),
                "val_manifest": os.path.abspath(args.val_manifest),
                "version": "v2.1",
            },
            latest_output,
        )

    print()
    print(f"Best validation PSNR: {best:.3f} dB")
    print(f"Best checkpoint     : {os.path.abspath(args.output)}")
    print(f"Latest checkpoint   : {os.path.abspath(latest_output)}")
    print("V2.1 uses a corruption-aware mask including cloud fringes and shadows.")
    print("This remains synthetic-cloud training, not a real-cloud benchmark.")


if __name__ == "__main__":
    main()
