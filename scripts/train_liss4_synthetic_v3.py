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


class NPZLISS4V3Dataset(Dataset):
    """Load the exact Synthetic Clouds V2 samples used by V2/V2.1/V2.2."""

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
            mask = d["mask"].astype(np.float32)

        if cloudy.shape != (3, 256, 256):
            raise ValueError(f"{p}: cloudy shape={cloudy.shape}")
        if clear.shape != (3, 256, 256):
            raise ValueError(f"{p}: clear shape={clear.shape}")
        if mask.shape != (1, 256, 256):
            raise ValueError(f"{p}: mask shape={mask.shape}")

        return {
            "cloudy": torch.from_numpy(cloudy),
            "clear": torch.from_numpy(clear),
            "mask": torch.from_numpy(mask),
        }


def masked_l1(pred, target, mask):
    mask = mask.expand_as(pred)
    return torch.sum(torch.abs(pred - target) * mask) / mask.sum().clamp_min(1.0)


def load_pretrained_blocks(model, path, device):
    """
    Transfer only the 16 residual blocks from the pretrained DSen2-CR model.

    The DSen2-CR input layer is incompatible (15 channels vs 3 LISS-IV
    channels), and its output layer is incompatible (13 channels vs 3).
    The residual blocks are shape-compatible and are therefore copied exactly.
    """
    state = torch.load(path, map_location=device, weights_only=False)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(state)}")

    clean = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        clean[k] = v

    dst = model.state_dict()
    copied = []
    skipped = []

    for k, v in clean.items():
        if not k.startswith("blocks."):
            continue

        if k not in dst:
            skipped.append(k)
            continue

        if tuple(v.shape) != tuple(dst[k].shape):
            raise ValueError(
                f"Shape mismatch for {k}: checkpoint={tuple(v.shape)} "
                f"model={tuple(dst[k].shape)}"
            )

        dst[k] = v
        copied.append(k)

    expected = 16 * 4
    if len(copied) != expected:
        raise RuntimeError(
            f"Expected {expected} residual-block tensors, copied {len(copied)}. "
            "The pretrained checkpoint may not match the expected DSen2-CR architecture."
        )

    model.load_state_dict(dst, strict=True)
    print(f"Pretrained checkpoint: {os.path.abspath(path)}")
    print(f"Transferred tensors  : {len(copied)} ({len(copied) // 4} residual blocks)")
    print("Input/output layers  : randomly initialized for 3-band LISS-IV")
    return model


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Train LISS-IV DSen2-CR using pretrained DSen2-CR residual-block "
            "weights and Synthetic Clouds V2."
        )
    )
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument(
        "--pretrained",
        default="weights/dsen2cr_sar_carl.pth",
        help="Converted public DSen2-CR PyTorch state-dict.",
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--features", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--lambda-cloud", type=float, default=2.0)
    ap.add_argument(
        "--output",
        default="weights/liss4_dsen2cr_synthetic_v3.pth",
    )
    ap.add_argument("--latest-output", default=None)
    args = ap.parse_args()

    if args.features != 256 or args.blocks != 16:
        raise ValueError(
            "V3 pretrained transfer requires the compatible "
            "features=256 and blocks=16 architecture."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")

    train_ds = NPZLISS4V3Dataset(args.train_manifest)
    val_ds = NPZLISS4V3Dataset(args.val_manifest)

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
        features=256,
        blocks=16,
        res_scale=0.1,
        use_sar=False,
    ).to(device)

    model = load_pretrained_blocks(model, args.pretrained, device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters   : {params:,}")
    print(
        f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | "
        f"Batch: {args.batch_size} | Accum: {args.grad_accum}"
    )
    print(f"Loss weights : cloud={args.lambda_cloud:.2f} | clean=0.00")
    print("Dataset      : Synthetic Clouds V2")
    print()

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
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
            ):
                # Same objective as V2. Keep the experiment controlled:
                # only initialization differs.
                pred = model(cloudy)

                base = loss_l1(pred, clear)
                cloud = masked_l1(pred, clear, mask)
                loss = (
                    base
                    + args.lambda_cloud * cloud
                ) / args.grad_accum

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0 or step == len(train_dl):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total += float(loss.item()) * args.grad_accum * cloudy.shape[0]
            total_cloud += float(cloud.item()) * cloudy.shape[0]
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

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss {avg:.6f} | "
            f"cloud_l1 {avg_cloud:.6f} | "
            f"val_PSNR {val_psnr:.3f} dB | "
            f"time {time.time()-t0:.1f}s"
        )

        if val_psnr > best:
            best = val_psnr
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "features": 256,
                    "blocks": 16,
                    "dn_max": 1023.0,
                    "best_val_psnr": best,
                    "epoch": epoch,
                    "lambda_cloud": args.lambda_cloud,
                    "lambda_clean": 0.0,
                    "train_manifest": os.path.abspath(args.train_manifest),
                    "val_manifest": os.path.abspath(args.val_manifest),
                    "pretrained": os.path.abspath(args.pretrained),
                    "version": "v3_pretrained_blocks",
                },
                args.output,
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "features": 256,
                "blocks": 16,
                "dn_max": 1023.0,
                "best_val_psnr": best,
                "lambda_cloud": args.lambda_cloud,
                "lambda_clean": 0.0,
                "train_manifest": os.path.abspath(args.train_manifest),
                "val_manifest": os.path.abspath(args.val_manifest),
                "pretrained": os.path.abspath(args.pretrained),
                "version": "v3_pretrained_blocks",
            },
            latest_output,
        )

    print()
    print(f"Best validation PSNR: {best:.3f} dB")
    print(f"Best checkpoint     : {os.path.abspath(args.output)}")
    print(f"Latest checkpoint   : {os.path.abspath(latest_output)}")
    print("V3 transfers only the compatible DSen2-CR residual blocks.")
    print("This remains synthetic-cloud training, not a real-cloud benchmark.")


if __name__ == "__main__":
    main()
