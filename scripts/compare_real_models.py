import argparse
import math
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from dsen2cr import DSen2CR
from models import ClearSkyUNet
from preprocess import normalize_optical, normalize_sar


def psnr(pred, target):
    mse = float(np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2))
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def run_clear(model, sar, cloudy, target, device):
    # ClearSkyUNet uses B04,B03,B02,B08 = indices 3,2,1,7.
    s = normalize_sar(sar)
    c = normalize_optical(cloudy[[3, 2, 1, 7]])
    t = normalize_optical(target[[3, 2, 1, 7]])

    with torch.no_grad():
        out = model(
            s.unsqueeze(0).to(device),
            c.unsqueeze(0).to(device),
        ).squeeze(0).cpu().numpy()

    # [-1,1] -> [0,1]
    out = np.clip((out + 1.0) / 2.0, 0.0, 1.0)
    t = np.clip((t.numpy() + 1.0) / 2.0, 0.0, 1.0)
    return out, t


def run_dsen(model, sar, cloudy, target, device):
    def opt(x):
        return np.clip(x, 0.0, 10000.0) / 2000.0

    s = sar.copy()
    s[0] = 2.0 * (np.clip(s[0], -25.0, 0.0) + 25.0) / 25.0
    s[1] = 2.0 * (np.clip(s[1], -35.0, 0.0) + 35.0) / 35.0

    c = opt(cloudy)
    t = opt(target)

    x = torch.from_numpy(np.concatenate([c, s], axis=0))
    with torch.no_grad():
        out = model(
            x.unsqueeze(0).to(device=device, dtype=torch.float32)
        ).squeeze(0).cpu().numpy()

    # RGB = B04,B03,B02 = indices 3,2,1
    ridx = [3, 2, 1]
    out = np.clip(out[ridx] / 5.0, 0.0, 1.0)
    t = np.clip(t[ridx] / 5.0, 0.0, 1.0)
    return out, t


def metric(name, out, target):
    e = np.abs(out - target)
    return (
        f"  {name:<12} MAE={e.mean():.6f}  PSNR={psnr(out, target):.2f} dB"
    )


def main():
    p = argparse.ArgumentParser(
        description="Compare ClearSkyUNet and pretrained DSen2-CR on real cloudy SEN12MS-CR samples."
    )
    p.add_argument("--sample", required=True)
    args = p.parse_args()

    d = np.load(args.sample, allow_pickle=True)
    sar = d["sar"].astype(np.float32)
    cloudy = d["cloudy"].astype(np.float32)
    target = d["target"].astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    if device.type == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

    clear = ClearSkyUNet().to(device)
    clear.load_state_dict(
        torch.load(
            os.path.join(PROJECT_ROOT, "weights", "best_model.pth"),
            map_location=device,
        )
    )
    clear.eval()

    dsen = DSen2CR().to(device=device, dtype=torch.float32)
    state = torch.load(
        os.path.join(PROJECT_ROOT, "weights", "dsen2cr_sar_carl.pth"),
        map_location="cpu",
    )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    dsen.load_state_dict(state, strict=True)
    dsen.eval()

    cin = np.clip(cloudy[[3, 2, 1], :], 0, 10000) / 10000.0
    tin = np.clip(target[[3, 2, 1], :], 0, 10000) / 10000.0

    c_out, c_t = run_clear(clear, sar, cloudy, target, device)
    d_out, d_t = run_dsen(dsen, sar, cloudy, target, device)

    print(f"\n📂 Sample: {os.path.basename(args.sample)}")
    for key in ("season", "scene", "patch"):
        if key in d:
            v = d[key].item() if d[key].shape == () else d[key]
            print(f"   {key}: {v}")

    print("\nREAL CLOUDY INPUT")
    print(f"  MAE={np.abs(cin - tin).mean():.6f}  PSNR={psnr(cin, tin):.2f} dB")

    print("\nMODEL OUTPUTS")
    print(metric("ClearSkyUNet", c_out[:3], c_t[:3]))
    print(metric("DSen2-CR", d_out, d_t))

    c_psnr = psnr(c_out[:3], c_t[:3])
    d_psnr = psnr(d_out, d_t)
    b_psnr = psnr(cin, tin)

    print("\nPSNR CHANGE FROM CLOUDY INPUT")
    print(f"  ClearSkyUNet : {c_psnr - b_psnr:+.2f} dB")
    print(f"  DSen2-CR     : {d_psnr - b_psnr:+.2f} dB")


if __name__ == "__main__":
    main()
