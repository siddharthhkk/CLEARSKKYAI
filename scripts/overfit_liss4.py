import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from liss4 import normalize_liss4
from liss4_patch_dataset import LISS4PatchDataset
from liss4_dsen2cr import LISS4DSen2CR


def psnr(a, b, peak=1.0):
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 0.0:
        return float("inf")
    return 20.0 * np.log10(peak / np.sqrt(mse))


def main():
    ap = argparse.ArgumentParser(
        description="Tiny LISS-IV overfit test: verify the model can learn real cloudy->clear patches."
    )
    ap.add_argument(
        "--manifest",
        default="data/patches/guwahati/patch_manifest.csv",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--features", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--output", default="weights/liss4_overfit_smoke.pth")
    args = ap.parse_args()

    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    ds = LISS4PatchDataset(args.manifest, split="smoke")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = LISS4DSen2CR(
        features=args.features,
        blocks=args.blocks,
        res_scale=0.1,
        use_sar=False,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    print(
        f"Samples: {len(ds)} | Params: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    best = -float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        t0 = time.time()

        for batch in dl:
            cloudy = normalize_liss4(batch["cloudy"].numpy(), 1023.0).to(
                device
            )
            clear = normalize_liss4(batch["clear"].numpy(), 1023.0).to(
                device
            )

            # The input .npz is still stored as DN. Convert to [0,1].
            opt.zero_grad(set_to_none=True)
            pred = torch.clamp(model(cloudy), 0.0, 1.0)
            loss = loss_fn(pred, clear)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * cloudy.shape[0]
            seen += cloudy.shape[0]

        model.eval()
        sse = 0.0
        n = 0
        with torch.no_grad():
            for batch in dl:
                cloudy = normalize_liss4(batch["cloudy"].numpy(), 1023.0).to(
                    device
                )
                clear = normalize_liss4(batch["clear"].numpy(), 1023.0).to(
                    device
                )
                pred = torch.clamp(model(cloudy), 0.0, 1.0)
                d = pred - clear
                sse += torch.sum(d * d).item()
                n += d.numel()

        mse = sse / n
        score = float(20.0 * np.log10(1.0 / np.sqrt(mse))) if mse > 0 else float("inf")
        avg_loss = loss_sum / max(1, seen)

        if score > best:
            best = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "features": args.features,
                    "blocks": args.blocks,
                    "dn_max": 1023.0,
                    "score": score,
                },
                args.output,
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"L1 {avg_loss:.6f} | "
            f"PSNR {score:.3f} dB | "
            f"Best {best:.3f} dB | "
            f"{time.time() - t0:.1f}s"
        )

    print()
    print("=== LISS-IV OVERFIT GATE ===")
    print(f"Best PSNR : {best:.3f} dB")
    print(f"Checkpoint: {os.path.abspath(args.output)}")
    if best < 32.0:
        print(
            "FAIL/INVESTIGATE: the tiny model did not substantially learn "
            "the training patches. Check normalization, target alignment, "
            "or data quality before full training."
        )
        raise SystemExit(1)

    print(
        "PASS: the tiny model learned the real cloudy->clear training mapping. "
        "This is a software/data sanity test, NOT a generalization result."
    )


if __name__ == "__main__":
    main()
