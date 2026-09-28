"""Run and score DSen2-CR against all locally prepared SEN12MS-CR demo samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import statistics

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from infer_sentinel_dsen2cr import choose_device, run_inference
from dsen2cr import DSen2CR, load_dsen2cr_weights


def score(prediction_path: Path, reference_path: Path) -> dict:
    with rasterio.open(prediction_path) as prediction, rasterio.open(reference_path) as reference:
        if (
            prediction.count != 13
            or reference.count != 13
            or prediction.width != reference.width
            or prediction.height != reference.height
            or prediction.crs != reference.crs
            or prediction.transform != reference.transform
        ):
            raise ValueError("Prediction and clear reference are not on the same 13-band grid.")
        estimate = prediction.read().astype(np.float32)
        target = reference.read().astype(np.float32)
        valid = (prediction.dataset_mask() > 0) & (reference.dataset_mask() > 0)
    if not valid.any():
        raise ValueError("No valid reference pixels are available for scoring.")
    difference = (estimate - target)[:, valid]
    mae = float(np.abs(difference).mean()) / 10000.0
    rmse = float(np.sqrt(np.square(difference).mean())) / 10000.0
    psnr = float("inf") if rmse == 0 else float(20.0 * np.log10(1.0 / rmse))
    return {"MAE_0_1": mae, "RMSE_0_1": rmse, "PSNR_dB": psnr, "valid_pixels": int(valid.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "data" / "sentinel_demo" / "manifest.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "eval" / "sentinel_demo"
    )
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "weights" / "dsen2cr_sar_carl.pth")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(
            f"Gallery manifest not found: {args.manifest}. Prepare samples first with "
            "scripts/prepare_sen12mscr_demo.py."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        cases = list(csv.DictReader(stream))
    if not cases:
        raise ValueError("Gallery manifest has no cases.")

    device = choose_device(args.device)
    model = DSen2CR().to(device)
    load_dsen2cr_weights(model, str(args.checkpoint), map_location=device)
    model.eval()

    for index, case in enumerate(cases, start=1):
        sample_root = args.manifest.parent
        cloudy = sample_root / case["cloudy_s2"]
        sar = sample_root / case["sar_s1"]
        target = sample_root / case["clear_target"]
        output = args.output_dir / f"{case['sample_id']}_estimate.tif"
        print(f"[{index}/{len(cases)}] Inferring {case['sample_id']} ({case['split']} split)", flush=True)
        run_inference(
            cloudy,
            sar,
            args.checkpoint,
            output,
            tile=256,
            overlap=32,
            device_name=args.device,
            model=model,
        )
        row = {
            key: case[key]
            for key in ("sample_id", "split", "season", "scene", "patch", "note")
        }
        input_metrics = score(cloudy, target)
        output_metrics = score(output, target)
        row.update(output_metrics)
        row.update({f"input_{key}": value for key, value in input_metrics.items()})
        row["PSNR_gain_dB"] = output_metrics["PSNR_dB"] - input_metrics["PSNR_dB"]
        row["MAE_gain_0_1"] = input_metrics["MAE_0_1"] - output_metrics["MAE_0_1"]
        row["prediction"] = str(output)
        results.append(row)
        print(
            f"  input PSNR={input_metrics['PSNR_dB']:.2f} dB -> "
            f"output PSNR={output_metrics['PSNR_dB']:.2f} dB "
            f"({row['PSNR_gain_dB']:+.2f} dB); "
            f"output MAE={output_metrics['MAE_0_1']:.5f}; reference-based patch metrics",
            flush=True,
        )

    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote per-case results to {csv_path}")
    test_results = [row for row in results if row["split"].lower() == "test"]
    if test_results:
        input_psnr = statistics.mean(row["input_PSNR_dB"] for row in test_results)
        output_psnr = statistics.mean(row["PSNR_dB"] for row in test_results)
        psnr_gain = statistics.mean(row["PSNR_gain_dB"] for row in test_results)
        improved = sum(row["PSNR_gain_dB"] > 0 for row in test_results)
        print(
            f"Held-out per-patch mean PSNR: {input_psnr:.2f} dB cloudy -> "
            f"{output_psnr:.2f} dB output; mean gain {psnr_gain:+.2f} dB "
            f"({improved}/{len(test_results)} patches improved)."
        )
    print("Train-split scores are demonstrations, not held-out performance.")
    print("Test-split examples are from SEN12MS-CR, not an independent external dataset.")


if __name__ == "__main__":
    main()
