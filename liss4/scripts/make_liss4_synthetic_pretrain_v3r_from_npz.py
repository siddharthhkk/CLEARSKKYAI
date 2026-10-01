import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class V3RealFromNPZDataset(Dataset):
    """
    Candidate Synthetic Clouds V3R:
    keep V2 geometry, but make cloud corruption substantially milder and
    closer to the real Guwahati cloudy-clear difference distribution.

    IMPORTANT:
      - This is a candidate generator for diagnosis.
      - It changes corruption amplitude only; clear-scene radiometry is
        intentionally left unchanged for a controlled next experiment.
    """

    def __init__(self, manifest, dn_max=1023.0):
        self.manifest = os.path.abspath(manifest)
        self.root = os.path.dirname(self.manifest)
        self.dn_max = float(dn_max)

        with open(self.manifest, "r", newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        if not self.rows:
            raise ValueError("Manifest is empty.")

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _field(g, scale, size=256):
        n = max(4, int(np.ceil(size / scale)))
        x = torch.from_numpy(g.random((1, 1, n, n), dtype=np.float32))

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

        shape = 0.55 * f0 + 0.32 * f1 + 0.13 * f2
        shape -= shape.min()
        shape /= max(float(shape.max()), 1e-6)

        mode = int(g.choice(3, p=[0.35, 0.45, 0.20]))
        # Keep V2 cloud-support geometry unchanged for this controlled test.
        ranges = [(0.15, 0.28), (0.24, 0.40), (0.36, 0.55)]
        cov = float(g.uniform(*ranges[mode]))

        thr = float(np.quantile(shape, 1.0 - cov))
        binary = (shape >= thr).astype(np.float32)

        support = torch.from_numpy(binary)[None, None]
        support = F.avg_pool2d(
            support,
            kernel_size=9,
            stride=1,
            padding=4,
        )[0, 0].numpy()

        d0 = cls._field(g, 64, size)
        d1 = cls._field(g, 20, size)
        d2 = cls._field(g, 6, size)
        density = 0.50 * d0 + 0.32 * d1 + 0.18 * d2
        density -= density.min()
        density /= max(float(density.max()), 1e-6)

        # V2 max alpha was 0.95 and mean alpha was 0.216 over the full patch.
        # Candidate V3R reduces opacity to roughly 0.45 max / much lower mean.
        alpha = support * (0.02 + 0.43 * density)
        alpha = np.clip(alpha, 0.0, 0.45).astype(np.float32)

        # Keep translated shadows, but reduce their strength substantially.
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
        shadow *= float(g.uniform(0.10, 0.25))
        shadow = np.clip(shadow, 0.0, 0.12).astype(np.float32)

        return (alpha > 0.12).astype(np.float32), alpha, shadow

    def __getitem__(self, idx):
        row = self.rows[idx]
        p = os.path.join(self.root, f"sample_{int(row['idx']):06d}.npz")

        with np.load(p) as d:
            clear = d["clear"].astype(np.float32)

            old_row = int(d["row"]) if "row" in d else int(row.get("row", 0))
            old_col = int(d["col"]) if "col" in d else int(row.get("col", 0))

        if clear.shape != (3, 256, 256):
            raise ValueError(f"Expected clear shape (3,256,256) in {p}")

        old_seed = old_row * 1000003 + old_col
        base_seed = int(row.get("cloud_seed", 0))
        seed = (base_seed ^ old_seed) & 0x7FFFFFFF

        mask, alpha, shadow = self._cloud_maps(seed)
        g = np.random.default_rng(seed + 17)

        # Lower cloud radiance; this is the main intentional change from V2.
        cloud_base = np.array(
            [
                g.uniform(0.22, 0.42),
                g.uniform(0.24, 0.46),
                g.uniform(0.28, 0.52),
            ],
            dtype=np.float32,
        )
        spectral = np.array(
            [
                g.uniform(0.96, 1.03),
                g.uniform(0.97, 1.04),
                g.uniform(0.95, 1.02),
            ],
            dtype=np.float32,
        )

        cloud_var = self._field(g, 24, 256)
        cloud_var = 0.90 + 0.16 * cloud_var

        cloud = np.clip(
            cloud_base[:, None, None]
            * spectral[:, None, None]
            * cloud_var[None],
            0.0,
            0.65,
        )

        clear_n = np.clip(clear / self.dn_max, 0.0, 1.0)

        cloudy_n = clear_n * (1.0 - alpha[None]) + cloud * alpha[None]
        cloudy_n *= 1.0 - shadow[None]

        noise = g.normal(0.0, 0.0015, size=cloudy_n.shape).astype(np.float32)
        cloudy_n += noise * (0.25 + 0.75 * alpha[None])
        cloudy_n = np.clip(cloudy_n, 0.0, 1.0)

        cloudy = (cloudy_n * self.dn_max).astype(np.float32)

        return {
            "cloudy": cloudy,
            "clear": clear,
            "mask": mask[None].astype(np.float32),
            "alpha": alpha[None].astype(np.float32),
            "shadow": shadow[None].astype(np.float32),
            "scene": str(row.get("scene", "")),
            "row": np.int32(old_row),
            "col": np.int32(old_col),
            "cloud_seed": np.int64(seed),
        }


def main():
    ap = argparse.ArgumentParser(
        description="Generate candidate Synthetic Clouds V3R from existing clear LISS-IV NPZ patches."
    )
    ap.add_argument(
        "--manifest",
        default="data/synthetic_pretrain_prod/manifest.csv",
    )
    ap.add_argument(
        "--output",
        default="data/synthetic_pretrain_v3r",
    )
    ap.add_argument("--dn-max", type=float, default=1023.0)
    args = ap.parse_args()

    manifest = os.path.abspath(args.manifest)
    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok=True)

    ds = V3RealFromNPZDataset(manifest, dn_max=args.dn_max)

    out_manifest = os.path.join(output, "manifest.csv")
    fields = list(ds.rows[0].keys())
    for field in ("idx", "scene", "row", "col", "cloud_seed"):
        if field not in fields:
            fields.append(field)

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

            row = dict(ds.rows[i])
            row.update(
                {
                    "idx": i,
                    "scene": item["scene"],
                    "row": int(item["row"]),
                    "col": int(item["col"]),
                    "cloud_seed": int(item["cloud_seed"]),
                }
            )
            w.writerow(row)

    print("=== SYNTHETIC LISS-IV V3R ===")
    print(f"Source manifest: {manifest}")
    print(f"Samples        : {len(ds)}")
    print(f"Output         : {output}")
    print(f"Manifest       : {out_manifest}")
    print()
    print("Controlled change from V2:")
    print("- lower cloud radiance")
    print("- lower cloud opacity")
    print("- lower shadow strength")
    print("- unchanged V2 cloud-support coverage/geometry")
    print("- slightly lower noise")
    print("- same basic multi-scale cloud geometry")
    print()
    print("This is a candidate domain-matching generator; inspect statistics")
    print("before training a new model.")


if __name__ == "__main__":
    main()
