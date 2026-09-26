import argparse
import csv
import os
import sys

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window
from torch.utils.data import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from liss4 import normalize_liss4


class SyntheticLISS4DatasetV2(Dataset):
    """Generate deterministic Synthetic Clouds V2 over native clear LISS-IV scenes."""

    def __init__(self, scene_paths, samples_per_scene=256, seed=42, dn_max=1023.0):
        self.dn_max = float(dn_max)
        self.scenes = []

        for scene in scene_paths:
            scene = os.path.abspath(scene)
            if not os.path.isfile(scene):
                raise FileNotFoundError(scene)

            with rasterio.open(scene) as src:
                if src.count != 3:
                    raise ValueError(f"Expected 3 bands in {scene}, got {src.count}")
                if src.height < 256 or src.width < 256:
                    raise ValueError(f"Scene too small for 256x256 patches: {scene}")
                self.scenes.append((scene, src.height, src.width))

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
                        self.samples.append((scene, row, col, cloud_seed))
                        accepted = True
                        break

                if not accepted:
                    raise RuntimeError(
                        f"Could not find a sufficiently valid 256x256 patch in {scene}."
                    )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _field(g, scale, size=256):
        n = max(4, int(np.ceil(size / scale)))
        x = g.random((1, 1, n, n), dtype=np.float32)
        x = torch.from_numpy(x)

        if n >= 3:
            x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        x = F.interpolate(
            x,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()

        x -= x.min()
        x /= max(float(x.max()), 1e-6)
        return x

    @classmethod
    def _cloud_maps(cls, seed, size=256):
        g = np.random.default_rng(seed)

        # Cloud geometry.
        f0 = cls._field(g, 64, size)
        f1 = cls._field(g, 28, size)
        f2 = cls._field(g, 10, size)

        shape = 0.55 * f0 + 0.32 * f1 + 0.13 * f2
        shape -= shape.min()
        shape /= max(float(shape.max()), 1e-6)

        # Vary cloud coverage by sample instead of centering most samples near 50%.
        mode = int(g.choice(3, p=[0.35, 0.45, 0.20]))
        ranges = [(0.15, 0.28), (0.24, 0.40), (0.36, 0.55)]
        cov = float(g.uniform(*ranges[mode]))

        # Percentile threshold gives a controlled initial coverage.
        thr = float(np.quantile(shape, 1.0 - cov))
        binary = (shape >= thr).astype(np.float32)

        # Soft cloud boundary.
        support = torch.from_numpy(binary)[None, None]
        support = F.avg_pool2d(
            support,
            kernel_size=9,
            stride=1,
            padding=4,
        )[0, 0].numpy()

        # Internal optical-density structure: coarse + medium + fine variations.
        d0 = cls._field(g, 64, size)
        d1 = cls._field(g, 20, size)
        d2 = cls._field(g, 6, size)
        density = 0.50 * d0 + 0.32 * d1 + 0.18 * d2
        density -= density.min()
        density /= max(float(density.max()), 1e-6)

        # Thin edges and dense cores, but zero outside actual cloud support.
        alpha = support * (0.08 + 0.87 * density)
        alpha = np.clip(alpha, 0.0, 0.95).astype(np.float32)
        mask = (alpha > 0.12).astype(np.float32)

        # Translate + blur cloud opacity to form a soft cloud shadow.
        angle = float(g.uniform(0.0, 2.0 * np.pi))
        dist = int(g.integers(18, 70))
        dy = int(round(np.sin(angle) * dist))
        dx = int(round(np.cos(angle) * dist))

        shadow = np.zeros_like(alpha, dtype=np.float32)

        ys = max(0, dy)
        ye = min(size, size + dy)
        xs = max(0, dx)
        xe = min(size, size + dx)
        sy = max(0, -dy)
        ey = sy + max(0, ye - ys)
        sx = max(0, -dx)
        ex = sx + max(0, xe - xs)

        if ye > ys and xe > xs:
            shadow[ys:ye, xs:xe] = alpha[sy:ey, sx:ex]

        shadow = torch.from_numpy(shadow)[None, None]
        shadow = F.avg_pool2d(
            shadow,
            kernel_size=15,
            stride=1,
            padding=7,
        )[0, 0].numpy()
        shadow *= float(g.uniform(0.18, 0.45))
        shadow = np.clip(shadow, 0.0, 0.45).astype(np.float32)

        return mask, alpha, shadow

    def __getitem__(self, idx):
        scene, row, col, cloud_seed = self.samples[idx]

        with rasterio.open(scene) as src:
            clear = src.read(
                window=Window(col, row, 256, 256)
            ).astype(np.float32)

        clear = np.clip(clear, 0.0, self.dn_max)
        mask, alpha, shadow = self._cloud_maps(cloud_seed)

        g = np.random.default_rng(cloud_seed + 17)

        cloud_base = np.array(
            [
                g.uniform(0.82, 0.98),
                g.uniform(0.84, 1.00),
                g.uniform(0.88, 1.00),
            ],
            dtype=np.float32,
        )
        spectral = np.array(
            [
                g.uniform(0.97, 1.03),
                g.uniform(0.98, 1.03),
                g.uniform(0.95, 1.02),
            ],
            dtype=np.float32,
        )

        # Slowly varying cloud radiance adds internal brightness structure.
        cloud_var = self._field(g, 24, 256)
        cloud_var = 0.88 + 0.20 * cloud_var

        cloud = np.clip(
            cloud_base[:, None, None]
            * spectral[:, None, None]
            * cloud_var[None],
            0.0,
            1.0,
        )

        clear_n = np.clip(clear / self.dn_max, 0.0, 1.0)

        cloudy_n = clear_n * (1.0 - alpha[None]) + cloud * alpha[None]
        cloudy_n *= 1.0 - shadow[None]

        noise = g.normal(
            0.0,
            0.003,
            size=cloudy_n.shape,
        ).astype(np.float32)
        cloudy_n += noise * (0.35 + 0.65 * alpha[None])
        cloudy_n = np.clip(cloudy_n, 0.0, 1.0)

        cloudy = (cloudy_n * self.dn_max).astype(np.float32)

        return {
            "cloudy": torch.from_numpy(cloudy),
            "clear": torch.from_numpy(clear.astype(np.float32)),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
            "alpha": torch.from_numpy(alpha[None].astype(np.float32)),
            "shadow": torch.from_numpy(shadow[None].astype(np.float32)),
            "scene": scene,
            "row": row,
            "col": col,
        }


def main():
    ap = argparse.ArgumentParser(
        description="Create Synthetic Clouds V2 for native LISS-IV pretraining."
    )
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--output", default="data/synthetic_pretrain_v2")
    ap.add_argument("--samples-per-scene", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dn-max", type=float, default=1023.0)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    ds = SyntheticLISS4DatasetV2(
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
            alpha=item["alpha"].numpy().astype(np.float32),
            shadow=item["shadow"].numpy().astype(np.float32),
            scene=item["scene"],
            row=np.int32(item["row"]),
            col=np.int32(item["col"]),
        )

    print("=== SYNTHETIC LISS-IV V2 ===")
    print(f"Scenes : {len(args.scenes)}")
    print(f"Samples: {len(ds)}")
    print(f"Output : {os.path.abspath(args.output)}")
    print(f"Manifest: {os.path.abspath(manifest)}")
    print()
    print("V2 corruption:")
    print("  - controlled 15-55% cloud coverage")
    print("  - multi-scale irregular cloud geometry")
    print("  - internal thin/medium/thick opacity variation")
    print("  - band-dependent + spatially varying cloud radiance")
    print("  - translated cloud shadows")
    print("  - small atmospheric/sensor noise")
    print()
    print(
        "IMPORTANT: native LISS-IV targets with synthetic cloud corruption. "
        "Use for pretraining/augmentation, not real-cloud benchmark reporting."
    )


if __name__ == "__main__":
    main()
