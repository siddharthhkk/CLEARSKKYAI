"""End-to-end smoke test for the PyTorch Sentinel training entry point."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dsen2cr import DSen2CR, load_dsen2cr_weights
from dsen2cr_training import cloud_cloudshadow_mask, carl_loss, read_training_samples


def write_tiff(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[-1],
        height=data.shape[-2],
        count=data.shape[0],
        dtype=str(data.dtype),
        transform=from_origin(500000, 5100000, 10, 10),
    ) as destination:
        destination.write(data)


def main() -> None:
    optical = np.full((13, 16, 16), 1800, dtype=np.uint16)
    cloudy = optical.copy()
    cloudy[1, 4:12, 4:12] = 6000
    sar = np.full((2, 16, 16), -18.0, dtype=np.float32)
    mask = cloud_cloudshadow_mask(cloudy, threshold=0.2)
    assert mask.shape == (1, 16, 16) and set(np.unique(mask)).issubset({0.0, 1.0})

    prediction = torch.zeros((1, 13, 16, 16), requires_grad=True)
    cloudy_t = torch.ones_like(prediction)
    clear_t = torch.full_like(prediction, 0.5)
    mask_t = torch.from_numpy(mask[None])
    loss = carl_loss(prediction, cloudy_t, clear_t, mask_t)
    loss.backward()
    assert torch.isfinite(loss) and prediction.grad is not None

    with tempfile.TemporaryDirectory(prefix="validate_dsen2cr_training_") as directory:
        root = Path(directory)
        upstream_index = root / "datasetfilelist.csv"
        with upstream_index.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(("1, S1, S2cloudFree, S2cloudy, train.tif",))
            writer.writerow(("2, S1, S2cloudFree, S2cloudy, val.tif",))
            writer.writerow(("3, S1, S2cloudFree, S2cloudy, test.tif",))
        parsed = read_training_samples(upstream_index, root)
        assert [sample.split for sample in parsed] == ["train", "val", "test"]
        assert parsed[0].cloudy_path == (root / "S2cloudy" / "train.tif").resolve()

        rows = []
        for split in ("train", "val"):
            sample_dir = root / split
            cloudy_path = sample_dir / "cloudy.tif"
            sar_path = sample_dir / "sar.tif"
            clear_path = sample_dir / "clear.tif"
            write_tiff(cloudy_path, cloudy)
            write_tiff(sar_path, sar)
            write_tiff(clear_path, optical)
            rows.append(
                {
                    "sample_id": split,
                    "split": split,
                    "cloudy_s2": cloudy_path.relative_to(root).as_posix(),
                    "sar_s1": sar_path.relative_to(root).as_posix(),
                    "clear_target": clear_path.relative_to(root).as_posix(),
                }
            )
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        output = root / "smoke.pth"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--manifest", str(manifest),
            "--data-root", str(root),
            "--output", str(output),
            "--epochs", "1",
            "--batch-size", "1",
            "--crop-size", "16",
            "--features", "4",
            "--blocks", "1",
            "--workers", "0",
            "--device", "cpu",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        latest = output.with_name("smoke_latest.pth")
        assert output.is_file() and latest.is_file()
        resume_command = command.copy()
        resume_command[resume_command.index("--epochs") + 1] = "2"
        resume_command.extend(("--resume", str(latest)))
        resumed = subprocess.run(resume_command, capture_output=True, text=True, check=True)
        latest_checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        assert latest_checkpoint["epoch"] == 2
        checkpoint = torch.load(output, map_location="cpu", weights_only=True)
        assert checkpoint["config"] == {"features": 4, "blocks": 1, "res_scale": 0.1}
        assert checkpoint["training_config"]["train_samples"] == 1
        model = DSen2CR(features=4, blocks=1)
        load_dsen2cr_weights(model, str(output))
        with torch.inference_mode():
            result = model(torch.zeros((1, 15, 16, 16)))
        assert result.shape == (1, 13, 16, 16) and torch.isfinite(result).all()
        print(completed.stdout.strip())
        print(resumed.stdout.strip())

    print("PyTorch DSen2-CR training smoke test passed (mask, CARL, train/val, save/load).")


if __name__ == "__main__":
    main()
