import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from liss4 import normalize_liss4
from liss4_patch_dataset import LISS4PatchDataset
from liss4_dsen2cr import LISS4DSen2CR


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a saved LISS-IV overfit smoke-test checkpoint."
    )
    ap.add_argument(
        "--manifest",
        default="data/patches/guwahati/patch_manifest.csv",
    )
    ap.add_argument("--checkpoint", default="weights/liss4_overfit_smoke.pth")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = LISS4DSen2CR(
        features=int(ckpt["features"]),
        blocks=int(ckpt["blocks"]),
        res_scale=0.1,
        use_sar=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = LISS4PatchDataset(args.manifest, split="smoke")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    sse_in = 0.0
    sse_out = 0.0
    n = 0
    mae_in = 0.0
    mae_out = 0.0

    with torch.no_grad():
        for batch in dl:
            cloudy = normalize_liss4(batch["cloudy"].numpy(), ckpt["dn_max"]).to(device)
            clear = normalize_liss4(batch["clear"].numpy(), ckpt["dn_max"]).to(device)
            pred = torch.clamp(model(cloudy), 0.0, 1.0)

            di = cloudy - clear
            do = pred - clear

            sse_in += torch.sum(di * di).item()
            sse_out += torch.sum(do * do).item()
            mae_in += torch.sum(torch.abs(di)).item()
            mae_out += torch.sum(torch.abs(do)).item()
            n += di.numel()

    mse_in = sse_in / n
    mse_out = sse_out / n

    def score(mse):
        if mse <= 0:
            return float("inf")
        return 20.0 * np.log10(1.0 / np.sqrt(mse))

    psnr_in = score(mse_in)
    psnr_out = score(mse_out)

    print("=== LISS-IV OVERFIT EVALUATION ===")
    print(f"Input  MAE : {mae_in / n:.6f}")
    print(f"Output MAE : {mae_out / n:.6f}")
    print(f"Input  PSNR: {psnr_in:.3f} dB")
    print(f"Output PSNR: {psnr_out:.3f} dB")
    print(f"Gain       : {psnr_out - psnr_in:+.3f} dB")
    print()
    print(
        "This compares a model trained on the same 16 development patches "
        "against their historical clear targets. It does not measure "
        "generalization to unseen scenes."
    )


if __name__ == "__main__":
    main()
