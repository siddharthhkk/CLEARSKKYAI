import argparse
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from liss4 import normalize_liss4
from liss4_dsen2cr import LISS4DSen2CR
from train_liss4_synthetic import NPZLISS4Dataset


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available.")
    return torch.device(requested)


def evaluate_manifest(model, manifest, batch_size, device, dn_max):
    dataset = NPZLISS4Dataset(manifest, set())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    sse = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            cloudy = normalize_liss4(batch["cloudy"], dn_max).to(
                device, non_blocking=True
            )
            clear = normalize_liss4(batch["clear"], dn_max).to(
                device, non_blocking=True
            )
            with torch.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
            ):
                pred = torch.clamp(model(cloudy), 0.0, 1.0)

            difference = pred.float() - clear.float()
            sse += torch.sum(difference * difference).item()
            n += difference.numel()

    mse = sse / max(1, n)
    psnr = 20.0 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else float("inf")
    return len(dataset), sse, n, psnr


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved LISS-IV synthetic checkpoint on one or more "
            "manifest datasets using the trainer's clamped normalized-DN PSNR."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Validation manifest; repeat to evaluate a combined validation set.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint must contain a model_state_dict: {checkpoint_path}"
        )

    device = select_device(args.device)
    model = LISS4DSen2CR(
        features=int(checkpoint.get("features", 256)),
        blocks=int(checkpoint.get("blocks", 16)),
        res_scale=0.1,
        use_sar=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dn_max = float(checkpoint.get("dn_max", 1023.0))

    total_sse = 0.0
    total_values = 0
    total_samples = 0
    print("=== SYNTHETIC CHECKPOINT EVALUATION ===")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Epoch      : {checkpoint.get('epoch', 'unknown')}")
    print(f"Device     : {device}")
    print(f"DN max     : {dn_max:g}")

    for manifest in args.manifest:
        manifest_path = os.path.abspath(manifest)
        samples, sse, values, psnr = evaluate_manifest(
            model,
            manifest_path,
            args.batch_size,
            device,
            dn_max,
        )
        total_samples += samples
        total_sse += sse
        total_values += values
        print(f"{manifest_path} | samples={samples} | PSNR={psnr:.3f} dB")

    total_mse = total_sse / max(1, total_values)
    total_psnr = (
        20.0 * math.log10(1.0 / math.sqrt(total_mse))
        if total_mse > 0
        else float("inf")
    )
    print(f"Combined validation | samples={total_samples} | PSNR={total_psnr:.3f} dB")
    print("NOTE: this is synthetic-cloud validation, not real-cloud accuracy.")


if __name__ == "__main__":
    main()
