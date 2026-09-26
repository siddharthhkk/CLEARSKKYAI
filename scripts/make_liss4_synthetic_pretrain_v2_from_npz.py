import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class V2FromNPZDataset(Dataset):
    """Apply Synthetic Clouds V2 to existing clear LISS-IV NPZ patches."""

    def __init__(self, manifest, dn_max=1023.0):
        self.manifest = os.path.abspath(manifest)
        self.root = os.path.dirname(self.manifest)
        self.dn_max = float(dn_max)

        with open(self.manifest, "r", newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        if not self.rows:
            raise ValueError("Manifest is empty.")

        required = {"idx"}
        missing = required - set(self.rows[0].keys())
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")

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

        f0 = cls._field(g, 64, size)
        f1 = cls._field(g, 28, size)
        f2 = cls._field(g, 10, size)

        field = 0.55 * f0 + 0.32 * f1 + 0.13 * f2
        field -= field.min()
        field /= max(float(field.max()), 1e-6)

        thr = float(g.uniform(0.58, 0.72))
        support = (field > thr).astype(np.float32)

        st = torch.from_numpy(support)[None, None]
        st = F.avg_pool2d(st, kernel_size=9, stride=1, padding=4)
        support = st[0, 0].numpy()

        fine = cls._field(g, 7, size)
        alpha = 0.18 + 0.72 * support
        alpha *= 0.72 + 0.28 * fine
        alpha = np.clip(alpha, 0.0, 0.95).astype(np.float32)

        mask = (alpha > 0.12).astype(np.float32)

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
        shadow = F.avg_pool2d(shadow, kernel_size=15, stride=1, padding=7)
        shadow = shadow[0, 0].numpy()
        shadow *= float(g.uniform(0.25, 0.55))
        shadow = np.clip(shadow, 0.0, 0.65).astype(np.float32)

        return mask, alpha, shadow

    def __getitem__(self, idx):
        row = self.rows[idx]
        p = os.path.join(self.root, f"sample_{int(row['idx']):06d}.npz")

        with np.load(p) as d:
            if "clear" not in d:
                raise ValueError(f"{p} does not contain a clear target.")
            clear = d["clear"].astype(np.float32)
            old_seed = int(d["row"]) * 1000003 + int(d["col"])
            if "scene" in d:
                scene = str(d["scene"])
            else:
                scene = row.get("scene", "")

        if clear.shape != (3, 256, 256):
            raise ValueError(f"Expected clear shape (3,256,256) in {p}, got {clear.shape}")

        # Derive a deterministic new seed from the original patch identity.
        # This keeps V2 reproducible without requiring the missing source TIFF.
        base_seed = int(row.get("cloud_seed", 0))
        seed = (base_seed ^ old_seed) & 0x7FFFFFFF

        mask, alpha, shadow = self._cloud_maps(seed)

        g = np.random.default_rng(seed + 17)

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

        cloud = cloud_base[:, None, None] * spectral[:, None, None]
        cloud = np.clip(cloud, 0.0, 1.0)

        clear_n = np.clip(clear / self.dn_max, 0.0, 1.0)

        cloudy_n = (
            clear_n * (1.0 - alpha[None])
            + cloud * alpha[None]
        )
        cloudy_n *= (1.0 - shadow[None])

        noise = g.normal(0.0, 0.004, size=cloudy_n.shape).astype(np.float32)
        cloudy_n += noise * (0.35 + 0.65 * alpha[None])
        cloudy_n = np.clip(cloudy_n, 0.0, 1.0)

        cloudy = (cloudy_n * self.dn_max).astype(np.float32)

        return {
            "cloudy": cloudy,
            "clear": clear,
            "mask": mask[None].astype(np.float32),
            "alpha": alpha[None].astype(np.float32),
            "shadow": shadow[None].astype(np.float32),
            "scene": scene,
            "row": np.int32(row.get("row", -1)),
            "col": np.int32(row.get("col", -1)),
            "cloud_seed": np.int64(seed),
        }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate Synthetic Clouds V2 from existing native LISS-IV NPZ "
            "patches, so the original clear TIFF is not required."
        )
    )
    ap.add_argument(
        "--manifest",
        default="data/synthetic_pretrain_prod/manifest.csv",
    )
    ap.add_argument(
        "--output",
        default="data/synthetic_pretrain_v2",
    )
    ap.add_argument("--dn-max", type=float, default=1023.0)
    args = ap.parse_args()

    manifest = os.path.abspath(args.manifest)
    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok=True)

    ds = V2FromNPZDataset(manifest, dn_max=args.dn_max)

    out_manifest = os.path.join(output, "manifest.csv")
    fields = ["idx", "scene", "row", "col", "cloud_seed"]

    with open(out_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for i in range(len(ds)):
            item = ds[i]

            np.savez_compressed(
                os.path.join(output, f"sample_{i:06d}.npz"),
                cloudy=item["cloudy"],
                clear=item["clear"],
                mask=item["mask"],
                alpha=item["alpha"],
                shadow=item["shadow"],
                scene=item["scene"],
                row=item["row"],
                col=item["col"],
            )

            w.writerow(
                {
                    "idx": i,
                    "scene": item["scene"],
                    "row": int(item["row"]),
                    "col": int(item["col"]),
                    "cloud_seed": int(item["cloud_seed"]),
                }
            )

    print("=== SYNTHETIC LISS-IV V2 FROM EXISTING NPZ ===")
    print(f"Source manifest: {manifest}")
    print(f"Samples        : {len(ds)}")
    print(f"Output         : {output}")
    print(f"Manifest       : {out_manifest}")
    print()
    print("Uses the existing clear LISS-IV patches already present in the dataset.")
    print("The original native clear TIFF is not required.")
    print("V2 includes variable-opacity clouds, spectral cloud radiance, shadows, and noise.")
    print("This remains synthetic pretraining data, not a real-cloud benchmark.")


if __name__ == "__main__":
    main()
