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
        assert checkpoint["train_samples"] == 1
        assert checkpoint["val_samples"] == 1
        assert checkpoint["batch_size"] == 1
        assert checkpoint["grad_accum"] == 1
        assert checkpoint["initial_checkpoint"] is None
        assert "val_PSNR" in result.stdout

        second_data_dir = os.path.join(temp, "second_data")
        second_train_manifest = write_split(
            second_data_dir, "train_manifest.csv", 0, 14
        )
        second_val_manifest = write_split(
            second_data_dir, "val_manifest.csv", 1, 19
        )
        combined_best = os.path.join(temp, "weights", "combined_best.pth")
        combined_latest = os.path.join(temp, "weights", "combined_latest.pth")
        combined_command = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "train_liss4_synthetic.py"),
            "--train-manifest",
            train_manifest,
            "--train-manifest",
            second_train_manifest,
            "--val-manifest",
            val_manifest,
            "--val-manifest",
            second_val_manifest,
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
            combined_best,
            "--latest-output",
            combined_latest,
        ]
        combined = subprocess.run(
            combined_command,
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        combined_checkpoint = torch.load(
            combined_best, map_location="cpu", weights_only=False
        )
        assert "Train samples: 2 | Val samples: 2" in combined.stdout
        assert combined_checkpoint["train_source"] == [
            os.path.abspath(train_manifest),
            os.path.abspath(second_train_manifest),
        ]
        assert combined_checkpoint["val_source"] == [
            os.path.abspath(val_manifest),
            os.path.abspath(second_val_manifest),
        ]
        assert combined_checkpoint["train_samples"] == 2
        assert combined_checkpoint["val_samples"] == 2

        alternate_val_manifest = write_split(
            data_dir, "val_manifest_alternate.csv", 1, 9
        )
        resumed_best = os.path.join(temp, "weights", "resumed_best.pth")
        resumed_latest = os.path.join(temp, "weights", "resumed_latest.pth")
        resumed = subprocess.run(
            [
                sys.executable,
                os.path.join(PROJECT_ROOT, "scripts", "train_liss4_synthetic.py"),
                "--train-manifest",
                train_manifest,
                "--val-manifest",
                alternate_val_manifest,
                "--epochs",
                "2",
                "--batch-size",
                "1",
                "--grad-accum",
                "1",
                "--features",
                "8",
                "--blocks",
                "1",
                "--output",
                resumed_best,
                "--latest-output",
                resumed_latest,
                "--resume",
                latest_path,
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert "Validation source changed; resetting best validation PSNR" in resumed.stdout
        assert "Resuming from epoch 2" in resumed.stdout
        resumed_checkpoint = torch.load(
            resumed_best, map_location="cpu", weights_only=False
        )
        assert resumed_checkpoint["epoch"] == 2
        assert resumed_checkpoint["val_source"] == alternate_val_manifest
        assert resumed_checkpoint["initial_checkpoint"] == os.path.abspath(latest_path)

        evaluation = subprocess.run(
            [
                sys.executable,
                os.path.join(
                    PROJECT_ROOT, "scripts", "evaluate_liss4_synthetic_checkpoint.py"
                ),
                "--checkpoint",
                combined_best,
                "--manifest",
                val_manifest,
                "--manifest",
                second_val_manifest,
                "--device",
                "cpu",
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert "Combined validation | samples=2 | PSNR=" in evaluation.stdout

    print("PASS one-epoch training smoke test with automatic sibling train/val manifests")
    print("PASS training from paired lists of independent train/validation manifests")
    print("PASS resume resets best validation score when validation source changes")
    print("PASS synthetic checkpoint evaluation combines manifest datasets")
    print("PASS non-finite learning-rate and cloud-loss-weight validation")
    print("PASS best/latest checkpoint creation and source metadata")


if __name__ == "__main__":
    main()
