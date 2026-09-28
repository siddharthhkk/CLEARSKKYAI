import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_split(directory, manifest_name, idx, seed):
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(seed)
    clear = rng.uniform(0, 700, size=(3, 32, 32)).astype(np.float32)
    cloudy = np.clip(clear * 0.75 + 100, 0, 1023).astype(np.float32)
    mask = np.zeros((1, 32, 32), dtype=np.float32)
    mask[:, 8:24, 8:24] = 1
    np.savez_compressed(
        os.path.join(directory, f"sample_{idx:06d}.npz"),
        cloudy=cloudy,
        clear=clear,
        mask=mask,
    )

    manifest_path = os.path.join(directory, manifest_name)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=("idx", "scene", "row", "col", "cloud_seed")
        )
        writer.writeheader()
        writer.writerow(
            {
                "idx": idx,
                "scene": "synthetic-smoke-scene",
                "row": 0,
                "col": idx * 32,
                "cloud_seed": seed,
            }
        )
    return manifest_path


def main():
    with tempfile.TemporaryDirectory(prefix="clearsky_liss4_training_") as temp:
        data_dir = os.path.join(temp, "data")
        os.makedirs(data_dir)
        with open(os.path.join(data_dir, "manifest.csv"), "w", encoding="utf-8") as f:
            f.write("idx,scene,row,col,cloud_seed\n")

        train_manifest = write_split(data_dir, "train_manifest.csv", 0, 4)
        val_manifest = write_split(data_dir, "val_manifest.csv", 1, 9)
        # Both split manifests and their referenced NPZ files share a directory.
        best_path = os.path.join(temp, "weights", "best.pth")
        latest_path = os.path.join(temp, "weights", "latest.pth")
        command = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "train_liss4_synthetic.py"),
            "--manifest",
            os.path.join(data_dir, "manifest.csv"),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--grad-accum",
            "1",
            "--features",
            "8",
            "--blocks",
            "1",
            "--output",
            best_path,
            "--latest-output",
            latest_path,
        ]
        for option, value, expected_error in (
            ("--lambda-cloud", "nan", "--lambda-cloud must be finite and non-negative"),
            ("--lr", "nan", "--lr must be finite and positive"),
        ):
            invalid = subprocess.run(
                command + [option, value],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            assert invalid.returncode != 0
            assert expected_error in invalid.stderr
        assert not os.path.exists(best_path)

        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert os.path.isfile(best_path)
        assert os.path.isfile(latest_path)
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 1
        assert checkpoint["features"] == 8
        assert checkpoint["blocks"] == 1
        assert checkpoint["train_source"] == train_manifest
        assert checkpoint["val_source"] == val_manifest
        assert "val_PSNR" in result.stdout

    print("PASS one-epoch training smoke test with automatic sibling train/val manifests")
    print("PASS non-finite learning-rate and cloud-loss-weight validation")
    print("PASS best/latest checkpoint creation and source metadata")


if __name__ == "__main__":
    main()
