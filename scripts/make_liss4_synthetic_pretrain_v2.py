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


class SyntheticLISS4V2Dataset(Dataset):
    """
    Generate deterministic synthetic cloudy/clear LISS-IV pairs.

    V2 corruption models:
      - multi-scale irregular cloud geometry
      - continuous cloud opacity/transmission
      - slightly different cloud radiance by spectral band
      - translated cloud shadows
      - native LISS-IV clear targets are kept unchanged

    Output samples contain:
      cloudy : [3, 256, 256] float32 DN
      clear  : [3, 256, 256] float32 DN
      mask   : [1, 256, 256] binary cloud-support mask
      alpha  : [1, 256, 256] continuous cloud opacity
      shadow : [1, 256, 256] shadow attenuation map
    """

    def __init__(
        self,
        scene_paths,
        samples_per_scene=256,
        seed=42,
        dn_max=1023.0,
    ):
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

                    # Avoid patches dominated by NoData/background.
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

    @staticmethod
    def _field(g, scale, size=256):
        """Create a smooth random field from a low-resolution noise grid."""
        n = max(4, int(np.ceil(size / scale)))
        x = g.random((1, 1, n, n), dtype=np.float32)
        x = torch.from_numpy(x)

        k = 3 if n >= 3 else 1
        if k > 1:
            x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

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

        # Multi-scale structure: broad cloud masses + medium structure + fine variation.
        f0 = cls._field(g, 64, size)
        f1 = cls._field(g, 28, size)
        f2 = cls._field(g, 10, size)

        field = 0.55 * f0 + 0.32 * f1 + 0.13 * f2
        field -= field.min()
        field /= max(float(field.max()), 1e-6)

        # Keep coverage varied instead of producing huge solid rectangles.
        thr = float(g.uniform(0.58, 0.72))
        mask = (field > thr).astype(np.float32)

        # Smooth the binary support so edges are not unnaturally hard.
        mt = torch.from_numpy(mask)[None, None]
        mt = F.avg_pool2d(mt, kernel_size=9, stride=1, padding=4)
        support = mt[0, 0].numpy()

        # Continuous opacity: thin at edges, thicker in cloud interiors.
        fine = cls._field(g, 7, size)
        a0 = 0.18 + 0.72 * support
        a = a0 * (0.72 + 0.28 * fine)
        a = np.clip(a, 0.0, 0.95).astype(np.float32)

        # Binary cloud-support mask follows the continuous alpha map.
        mask = (a > 0.12).astype(np.float32)

        # Cloud shadow is a translated, blurred version of the cloud opacity.
        angle = float(g.uniform(0.0, 2.0 * np.pi))
        dist = int(g.integers(18, 70))
        dy = int(round(np.sin(angle) * dist))
        dx = int(round(np.cos(angle) * dist))

        shadow = np.zeros_like(a, dtype=np.float32)
        ys = max(0, dy)
        ye = min(size, size + dy)
        xs = max(0, dx)
        xe = min(size, size + dx)
        sy = max(0, -dy)
        ey = sy + (ye - ys)
        sx = max(0, -dx)
        ex = sx + (xe - xs)

        if ye > ys and xe > xs:
            shadow[ys:ye, xs:xe] = a[sy:ey, sx:ex]

        shadow = torch.from_numpy(shadow)[None, None]
        shadow = F.avg_pool2d(shadow, kernel_size=15, stride=1, padding=7)
        shadow = shadow[0, 0].numpy()
        shadow *= float(g.uniform(0.25, 0.55))
        shadow = np.clip(shadow, 0.0, 0.65).astype(np.float32)

        return mask, a, shadow

    def __getitem__(self, idx):
        scene, row, col, cloud_seed = self.samples[idx]

        with rasterio.open(scene) as src:
            clear = src.read(
                window=Window(col, row, 256, 256)
            ).astype(np.float32)

        clear = np.clip(clear, 0.0, self.dn_max)

        mask, alpha, shadow = self._cloud_maps(cloud_seed)

        # Small per-band radiance differences for white-ish clouds.
        # Values are intentionally below 1 so clouds are not forced to saturation.
        g = np.random.default_rng(cloud_seed + 17)
        cloud_base = np.array(
            [
                g.uniform(0.82, 0.98),  # Green
                g.uniform(0.84, 1.00),  # Red
                g.uniform(0.88, 1.00),  # NIR
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
        cloud = cloud_base[:, None, None] * spectral[:, None, None]
        cloud = np.clip(cloud, 0.0, 1.0)

        clear_n = np.clip(clear / self.dn_max, 0.0, 1.0)

        # Atmospheric cloud mixing.
        cloudy_n = (
            clear_n * (1.0 - alpha[None])
            + cloud * alpha[None]
        )

        # Apply cloud shadows separately from the cloud radiance.
        cloudy_n *= (1.0 - shadow[None])

        # A little sensor/atmospheric variation avoids perfectly clean synthetic math.
        noise = g.normal(0.0, 0.004, size=cloudy_n.shape).astype(np.float32)
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
        description=(
            "Create Synthetic Clouds V2 for native LISS-IV pretraining: "
            "multi-scale clouds, variable opacity, spectral response, and shadows."
        )
    )
    ap.add_argument("scenes", nargs="+")
    ap.add_argument(
        "--output",
        default="data/synthetic_pretrain_v2",
    )
    ap.add_argument("--samples-per-scene", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dn-max", type=float, default=1023.0)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    ds = SyntheticLISS4V2Dataset(
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
    print("  - multi-scale irregular cloud geometry")
    print("  - continuous cloud opacity")
    print("  - band-dependent cloud radiance")
    print("  - translated cloud shadows")
    print("  - small atmospheric/sensor noise")
    print()
    print(
        "IMPORTANT: native LISS-IV targets with synthetic cloud corruption. "
        "Use for pretraining/augmentation, not real-cloud benchmark reporting."
    )


if __name__ == "__main__":
    main()
