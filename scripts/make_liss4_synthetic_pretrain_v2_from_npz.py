import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class V2FromNPZDataset(Dataset):
    """Generate deterministic Synthetic Clouds V2 over native clear LISS-IV scenes."""

    def __init__(self, manifest, dn_max=1023.0):
        self.manifest = os.path.abspath(manifest)
        self.root = os.path.dirname(self.manifest)
        self.dn_max = float(dn_max)

        with open(self.manifest, "r", newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        if not self.rows:
            raise ValueError("Manifest is empty.")

        if "idx" not in self.rows[0]:
            raise ValueError("Manifest missing required column: idx")

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _field(g, scale, size=256):
        n = max(4, int(np.ceil(size / scale)))
        x = g.random((1, 1, n, n), dtype=np.float32)
        x = torch.from_numpy(x)

        if n >= 3:
            x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        x = F.interpolate(
            x, size=(size, size), mode="bilinear", align_corners=False
        )[0, 0].numpy()

        x -= x.min()
        x /= max(float(x.max()), 1e-6)
        return x

    @classmethod
    def _cloud_maps(cls, seed, size=256):
        g = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

        # Build several compact cloud masses instead of one giant low-frequency blob.
        n_blobs = int(g.integers(2, 6))
        field = np.zeros((size, size), dtype=np.float32)

        for _ in range(n_blobs):
            cx = float(g.uniform(0.12, 0.88) * size)
            cy = float(g.uniform(0.12, 0.88) * size)
            rx = float(g.uniform(24, 58))
            ry = float(g.uniform(18, 52))
            ang = float(g.uniform(0.0, 2.0 * np.pi))

            ca = np.cos(ang)
            sa = np.sin(ang)
            dx = xx - cx
            dy = yy - cy
            xr = ca * dx + sa * dy
            yr = -sa * dx + ca * dy

            d = (xr / rx) ** 2 + (yr / ry) ** 2
            blob = np.clip(1.0 - d, 0.0, 1.0)
            blob = blob ** float(g.uniform(0.8, 1.8))

            local = cls._field(g, float(g.uniform(10, 24)), size)
            blob *= 0.78 + 0.30 * local
            field = np.maximum(field, blob.astype(np.float32))

        fine = cls._field(g, 9, size)
        field = 0.94 * field + 0.06 * fine

        # Target moderate cloud coverage. Some patches can still be heavier.
        coverage = float(g.uniform(0.12, 0.32))
        thr = float(np.quantile(field, 1.0 - coverage))
        binary = (field >= thr).astype(np.float32)

        # Soft edges for partial transmission, while keeping alpha exactly zero
        # outside the binary cloud support.
        support = torch.from_numpy(binary)[None, None]
        support = F.avg_pool2d(
            support,
            kernel_size=11,
            stride=1,
            padding=5,
        )[0, 0].numpy()

        core = np.clip(
            (field - thr) / max(float(field.max() - thr), 1e-6),
            0.0,
            1.0,
        )

        alpha = binary * (
            0.12
            + 0.72 * (0.45 * support + 0.55 * core)
        )
        alpha *= 0.78 + 0.22 * fine
        alpha = np.clip(alpha, 0.0, 0.88).astype(np.float32)

        # Cloud shadows: translated, blurred opacity in the opposite direction
        # of the synthetic illumination vector.
        angle = float(g.uniform(0.0, 2.0 * np.pi))
        dist = int(g.integers(25, 90))
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
            kernel_size=21,
            stride=1,
            padding=10,
        )[0, 0].numpy()
        shadow *= float(g.uniform(0.18, 0.35))
        shadow = np.clip(shadow, 0.0, 0.45).astype(np.float32)

        return binary, alpha, shadow

    def __getitem__(self, idx):
        row = self.rows[idx]
        p = os.path.join(self.root, f"sample_{int(row['idx']):06d}.npz")

        with np.load(p) as d:
            if "clear" not in d:
                raise ValueError(f"{p} does not contain a clear target.")
            clear = d["clear"].astype(np.float32)
            old_row = int(d["row"]) if "row" in d else int(row.get("row", 0))
            old_col = int(d["col"]) if "col" in d else int(row.get("col", 0))
            scene = str(d["scene"]) if "scene" in d else row.get("scene", "")

        if clear.shape != (3, 256, 256):
            raise ValueError(
                f"Expected clear shape (3,256,256) in {p}, got {clear.shape}"
            )

        clear = np.clip(clear, 0.0, self.dn_max)

        base_seed = int(row.get("cloud_seed", 0))
        seed = (base_seed ^ (old_row * 1000003 + old_col)) & 0x7FFFFFFF
        mask, alpha, shadow = self._cloud_maps(seed)

        g = np.random.default_rng(cloud_seed + 17)

        # Cloud radiance is bright but not forced to saturation.
        cloud_base = np.array(
            [
                g.uniform(0.74, 0.92),
                g.uniform(0.78, 0.95),
                g.uniform(0.80, 0.96),
            ],
            dtype=np.float32,
        )
        spectral = np.array(
            [
                g.uniform(0.98, 1.03),
                g.uniform(0.98, 1.03),
                g.uniform(0.96, 1.02),
            ],
            dtype=np.float32,
        )

        cloud = np.clip(
            cloud_base[:, None, None] * spectral[:, None, None],
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
        cloudy_n += noise * (0.25 + 0.75 * alpha[None])
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
    print("  - multiple cloud masses with irregular shape")
    print("  - moderate 12%-32% cloud coverage")
    print("  - variable cloud opacity")
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
