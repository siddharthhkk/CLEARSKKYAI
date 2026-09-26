import argparse
import csv
import os
import sys

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from liss4 import normalize_liss4


class SyntheticLISS4Dataset(Dataset):
    """Generate deterministic synthetic cloud corruption over native LISS-IV clear scenes."""

    def __init__(self, scene_paths, samples_per_scene=256, seed=42, dn_max=1023.0):
        self.dn_max = float(dn_max)
        self.scenes = []

        for scene in scene_paths:
            scene = os.path.abspath(scene)
            if not os.path.isfile(scene):
                raise FileNotFoundError(scene)

            with rasterio.open(scene) as src:
                if src.count != 3:
                    raise ValueError(
                        f"Expected 3 bands in {scene}, got {src.count}"
                    )
                if src.height < 256 or src.width < 256:
                    raise ValueError(
                        f"Scene too small for 256x256 patches: {scene}"
                    )
                h, w = src.height, src.width
                self.scenes.append((scene, h, w))

        if not self.scenes:
            raise ValueError("No LISS-IV scenes were supplied.")

        rng = np.random.default_rng(seed)
        self.samples = []

        for scene, h, w in self.scenes:
            for _ in range(samples_per_scene):
                accepted = False

                for _attempt in range(1000):
                    row = int(rng.integers(0, h - 255))
                    col = int(rng.integers(0, w - 255))

                    with rasterio.open(scene) as src:
                        patch = src.read(
                            window=Window(col, row, 256, 256)
                        ).astype(np.float32)

                    if float(np.mean(patch == 0)) <= 0.30:
                        cloud_seed = int(rng.integers(0, 2**31 - 1))
                        self.samples.append(
                            (scene, row, col, cloud_seed)
                        )
                        accepted = True
                        break

                if not accepted:
                    raise RuntimeError(
                        f"Could not find a sufficiently valid 256x256 patch in {scene}."
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene, row, col, cloud_seed = self.samples[idx]

        with rasterio.open(scene) as src:
            clear = src.read(
                window=Window(col, row, 256, 256)
            ).astype(np.float32)

        g = np.random.default_rng(cloud_seed)
        noise = g.random((16, 16), dtype=np.float32)

        import torch.nn.functional as F

        mask = torch.from_numpy(noise)[None, None]
        mask = F.interpolate(
            mask,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        mask = (mask > 0.65).float().numpy()

        cloudy = (
            clear * (1.0 - mask[None])
            + self.dn_max * mask[None]
        )

        return {
            "cloudy": torch.from_numpy(cloudy.astype(np.float32)),
            "clear": torch.from_numpy(clear.astype(np.float32)),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
            "scene": scene,
            "row": row,
            "col": col,
        }


def main():
    ap = argparse.ArgumentParser(
        description="Create a synthetic-cloud LISS-IV pretraining set from native clear scenes."
    )
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--output", default="data/synthetic_pretrain")
    ap.add_argument("--samples-per-scene", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dn-max", type=float, default=1023.0)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    ds = SyntheticLISS4Dataset(
        args.scenes,
        samples_per_scene=args.samples_per_scene,
        seed=args.seed,
        dn_max=args.dn_max,
    )

    manifest = os.path.join(args.output, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "scene", "row", "col", "cloud_seed"])
        for i, (scene, row, col, cloud_seed) in enumerate(ds.samples):
            w.writerow([i, scene, row, col, cloud_seed])

    for i in range(len(ds)):
        item = ds[i]
        np.savez_compressed(
            os.path.join(args.output, f"sample_{i:06d}.npz"),
            cloudy=item["cloudy"].numpy().astype(np.float32),
            clear=item["clear"].numpy().astype(np.float32),
            mask=item["mask"].numpy().astype(np.float32),
            scene=item["scene"],
            row=np.int32(item["row"]),
            col=np.int32(item["col"]),
        )

    print("=== SYNTHETIC LISS-IV PRETRAIN DATA ===")
    print(f"Scenes : {len(args.scenes)}")
    print(f"Samples: {len(ds)}")
    print(f"Output : {os.path.abspath(args.output)}")
    print(f"Manifest: {os.path.abspath(manifest)}")
    print(
        "IMPORTANT: native LISS-IV targets with synthetic cloud corruption. "
        "Use for pretraining/augmentation, not real-cloud benchmark reporting."
    )


if __name__ == "__main__":
    main()
