import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dsen2cr import DSen2CR
from models import ClearSkyUNet
from preprocess import normalize_optical, normalize_sar


def psnr(pred, target):
    mse = float(np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2))
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def clear_run(model, sar, cloudy, target, device):
    s = normalize_sar(sar)
    c = normalize_optical(cloudy[[3, 2, 1, 7]])
    t = normalize_optical(target[[3, 2, 1, 7]])

    with torch.no_grad():
        out = model(
            s.unsqueeze(0).to(device),
            c.unsqueeze(0).to(device),
        ).squeeze(0).cpu().numpy()

    out = np.clip((out + 1.0) / 2.0, 0.0, 1.0)
    t = np.clip((t.numpy() + 1.0) / 2.0, 0.0, 1.0)

    return out[:3], t[:3]


def dsen_run(model, sar, cloudy, target, device):
    c = np.clip(cloudy, 0.0, 10000.0) / 2000.0
    t = np.clip(target, 0.0, 10000.0) / 2000.0

    s = sar.copy()
    s[0] = 2.0 * (np.clip(s[0], -25.0, 0.0) + 25.0) / 25.0
    s[1] = 2.0 * (np.clip(s[1], -35.0, 0.0) + 35.0) / 35.0

    x = torch.from_numpy(np.concatenate([c, s], axis=0))
    with torch.no_grad():
        out = model(
            x.unsqueeze(0).to(device=device, dtype=torch.float32)
        ).squeeze(0).cpu().numpy()

    ridx = [3, 2, 1]
    out = np.clip(out[ridx] / 5.0, 0.0, 1.0)
    t = np.clip(t[ridx] / 5.0, 0.0, 1.0)

    return out, t


def evaluate_folder(folder, clear_model, dsen_model, device):
    samples = sorted(Path(folder).glob("sample_*.npz"))
    if not samples:
        raise RuntimeError(f"No sample_*.npz files found in {folder}")

    rows = []

    for path in samples:
        d = np.load(path, allow_pickle=True)
        sar = d["sar"].astype(np.float32)
        cloudy = d["cloudy"].astype(np.float32)
        target = d["target"].astype(np.float32)

        input_rgb = np.clip(cloudy[[3, 2, 1]], 0.0, 10000.0) / 10000.0
        target_rgb = np.clip(target[[3, 2, 1]], 0.0, 10000.0) / 10000.0

        cin = float(np.mean(np.abs(input_rgb - target_rgb)))
        c_psnr = psnr(input_rgb, target_rgb)

        clear_out, clear_t = clear_run(
            clear_model, sar, cloudy, target, device
        )
        dsen_out, dsen_t = dsen_run(
            dsen_model, sar, cloudy, target, device
        )

        clear_mae = float(np.mean(np.abs(clear_out - clear_t)))
        clear_psnr = psnr(clear_out, clear_t)

        dsen_mae = float(np.mean(np.abs(dsen_out - dsen_t)))
        dsen_psnr = psnr(dsen_out, dsen_t)

        rows.append(
            {
                "name": path.name,
                "season": str(d["season"].item()) if d["season"].shape == () else str(d["season"]),
                "scene": str(d["scene"].item()) if d["scene"].shape == () else str(d["scene"]),
                "patch": str(d["patch"].item()) if d["patch"].shape == () else str(d["patch"]),
                "input_mae": cin,
                "input_psnr": c_psnr,
                "clear_mae": clear_mae,
                "clear_psnr": clear_psnr,
                "clear_gain": clear_psnr - c_psnr,
                "dsen_mae": dsen_mae,
                "dsen_psnr": dsen_psnr,
                "dsen_gain": dsen_psnr - c_psnr,
            }
        )

        print(
            f"{path.name}: "
            f"ClearSky {clear_psnr:.2f} dB | "
            f"DSen2-CR {dsen_psnr:.2f} dB | "
            f"Input {c_psnr:.2f} dB"
        )

    print()
    print("=" * 78)
    print("REAL-CLOUD BATCH SUMMARY")
    print("=" * 78)

    def avg(key):
        return float(np.mean([r[key] for r in rows]))

    print(f"Samples evaluated: {len(rows)}")
    print(
        f"Unique season/scene pairs: "
        f"{len({(r['season'], r['scene']) for r in rows})}"
    )

    print()
    print(
        f"{'Metric':<22}"
        f"{'Cloudy Input':>16}"
        f"{'ClearSkyUNet':>16}"
        f"{'DSen2-CR':>16}"
    )
    print("-" * 70)
    print(
        f"{'RGB MAE':<22}"
        f"{avg('input_mae'):>16.6f}"
        f"{avg('clear_mae'):>16.6f}"
        f"{avg('dsen_mae'):>16.6f}"
    )
    print(
        f"{'RGB PSNR':<22}"
        f"{avg('input_psnr'):>16.2f}"
        f"{avg('clear_psnr'):>16.2f}"
        f"{avg('dsen_psnr'):>16.2f}"
    )
    print(
        f"{'PSNR gain':<22}"
        f"{'-':>16}"
        f"{avg('clear_gain'):>16.2f}"
        f"{avg('dsen_gain'):>16.2f}"
    )

    print()
    print("Per-sample results:")
    for r in rows:
        print(
            f"  {r['name']} | "
            f"{r['season']} scene={r['scene']} patch={r['patch']} | "
            f"ClearSky={r['clear_psnr']:.2f} dB "
            f"(gain {r['clear_gain']:+.2f}) | "
            f"DSen2-CR={r['dsen_psnr']:.2f} dB "
            f"(gain {r['dsen_gain']:+.2f})"
        )


def main():
    p = argparse.ArgumentParser(
        description="Compare ClearSkyUNet and pretrained DSen2-CR on real-cloud samples."
    )
    p.add_argument("--dir", default="real_cloud_diverse")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    if device.type == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

    clear_model = ClearSkyUNet().to(device)
    clear_model.load_state_dict(
        torch.load(
            PROJECT_ROOT / "weights" / "best_model.pth",
            map_location=device,
        )
    )
    clear_model.eval()

    dsen_model = DSen2CR().to(device=device, dtype=torch.float32)
    state = torch.load(
        PROJECT_ROOT / "weights" / "dsen2cr_sar_carl.pth",
        map_location="cpu",
    )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    dsen_model.load_state_dict(state, strict=True)
    dsen_model.eval()

    evaluate_folder(args.dir, clear_model, dsen_model, device)


if __name__ == "__main__":
    main()
