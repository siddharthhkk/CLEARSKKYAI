import argparse
import csv
import os

import numpy as np
import rasterio


def pct(x, q):
    return float(np.percentile(x, q))


def stats(x):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p50": pct(x, 50),
        "p90": pct(x, 90),
        "p95": pct(x, 95),
        "p99": pct(x, 99),
        "max": float(np.max(x)),
    }


def print_stats(name, x):
    s = stats(x)
    print(
        f"{name:<24} mean={s['mean']:.5f} std={s['std']:.5f} "
        f"p50={s['p50']:.5f} p90={s['p90']:.5f} "
        f"p95={s['p95']:.5f} p99={s['p99']:.5f} max={s['max']:.5f}"
    )


def load_tif(path, dn_max):
    with rasterio.open(path) as src:
        x = src.read().astype(np.float32)

    if x.shape[0] != 3:
        raise ValueError(f"{path}: expected 3 bands, got {x.shape}")

    return np.clip(x, 0.0, dn_max) / dn_max


def load_synthetic(manifest, dn_max, max_samples):
    manifest = os.path.abspath(manifest)
    root = os.path.dirname(manifest)

    with open(manifest, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if max_samples is not None:
        rows = rows[:max_samples]

    if not rows:
        raise ValueError("Synthetic manifest is empty.")

    cloudy = []
    clear = []
    masks = []
    alphas = []
    shadows = []
    deltas = []

    for r in rows:
        p = os.path.join(root, f"sample_{int(r['idx']):06d}.npz")
        with np.load(p) as d:
            cloudy.append(np.clip(d["cloudy"].astype(np.float32), 0.0, dn_max) / dn_max)
            clear.append(np.clip(d["clear"].astype(np.float32), 0.0, dn_max) / dn_max)
            masks.append(d["mask"].astype(np.float32))

            if "alpha" in d:
                alphas.append(d["alpha"].astype(np.float32))
            if "shadow" in d:
                shadows.append(d["shadow"].astype(np.float32))

    c = np.stack(cloudy)
    t = np.stack(clear)
    delta = c - t
    m = np.stack(masks)

    out = {
        "cloudy": c,
        "clear": t,
        "delta": delta,
        "mask": m,
        "rows": rows,
    }

    if alphas:
        out["alpha"] = np.stack(alphas)
    if shadows:
        out["shadow"] = np.stack(shadows)

    return out


def band_line(prefix, x, bands=("G", "R", "NIR")):
    for i, b in enumerate(bands):
        print_stats(f"{prefix} {b}", x[:, i].ravel())


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compare Synthetic Clouds V2 and real Guwahati LISS-IV "
            "corruption statistics before changing the generator."
        )
    )
    ap.add_argument(
        "--synthetic-manifest",
        default="data/synthetic_pretrain_v2/manifest.csv",
    )
    ap.add_argument(
        "--real-cloudy",
        default="data/raw/cloudy/guwahati_cloudy_test.tif",
    )
    ap.add_argument(
        "--real-clear",
        default="data/raw/clear/guwahati_clear.tif",
    )
    ap.add_argument("--dn-max", type=float, default=1023.0)
    ap.add_argument(
        "--max-synthetic-samples",
        type=int,
        default=None,
        help="Optional cap for synthetic samples.",
    )
    ap.add_argument(
        "--delta-thresholds",
        type=float,
        nargs="+",
        default=[5, 10, 25, 50, 100],
        help="Absolute cloudy-clear thresholds in native DN units.",
    )
    args = ap.parse_args()

    syn = load_synthetic(
        args.synthetic_manifest,
        args.dn_max,
        args.max_synthetic_samples,
    )
    rc = load_tif(args.real_cloudy, args.dn_max)
    rt = load_tif(args.real_clear, args.dn_max)

    if rc.shape != rt.shape:
        raise ValueError("Real cloudy and clear images have different shapes.")

    if syn["cloudy"].shape[1:] != rc.shape:
        print(
            "NOTE: Synthetic patches are 256x256 while the real scene is "
            f"{rc.shape[1]}x{rc.shape[2]}. This is expected."
        )

    print("=== SYNTHETIC V2 vs REAL GUWAHATI DOMAIN ANALYSIS ===")
    print(f"Synthetic samples analyzed: {syn['cloudy'].shape[0]}")
    print("Bands: [G, R, NIR]")
    print("All radiometry normalized by DN max.")
    print()

    # Basic radiometry.
    print("--- 1. Absolute radiometry ---")
    band_line("Synthetic cloudy", syn["cloudy"])
    band_line("Synthetic clear ", syn["clear"])
    band_line("Real cloudy     ", rc[None])
    band_line("Real clear      ", rt[None])
    print()

    # Signed and absolute corruption deltas.
    print("--- 2. Corruption delta statistics ---")
    band_line("Synthetic delta ", syn["delta"])
    real_delta = rc - rt
    band_line("Real delta      ", real_delta[None])
    print()

    print("--- 3. Synthetic V2 cloud/shadow geometry ---")
    m = syn["mask"][:, 0]
    print(f"Synthetic cloud-mask coverage mean: {m.mean() * 100:.2f}%")
    print(
        f"Synthetic cloud-mask coverage p50/p90/p99: "
        f"{pct(m.mean(axis=(1, 2)), 50) * 100:.2f}% / "
        f"{pct(m.mean(axis=(1, 2)), 90) * 100:.2f}% / "
        f"{pct(m.mean(axis=(1, 2)), 99) * 100:.2f}%"
    )

    if "alpha" in syn:
        a = syn["alpha"][:, 0]
        print(
            f"Alpha mean={a.mean():.5f} p90={pct(a,90):.5f} "
            f"p99={pct(a,99):.5f} max={a.max():.5f}"
        )

    if "shadow" in syn:
        s = syn["shadow"][:, 0]
        print(
            f"Shadow mean={s.mean():.5f} p90={pct(s,90):.5f} "
            f"p99={pct(s,99):.5f} max={s.max():.5f}"
        )
    print()

    # Compare magnitude distributions in DN.
    print("--- 4. Absolute cloudy-clear difference in native DN ---")
    syn_abs_dn = np.abs(syn["delta"]) * args.dn_max
    real_abs_dn = np.abs(real_delta) * args.dn_max

    syn_mag = np.max(syn_abs_dn, axis=1)
    real_mag = np.max(real_abs_dn, axis=0)

    print_stats("Synthetic max |delta| DN", syn_mag.ravel())
    print_stats("Real max |delta| DN", real_mag.ravel())
    print()

    print("--- 5. Coverage above cloudy-clear difference thresholds ---")
    print("Synthetic = corruption generated by V2; Real = diagnostic proxy only.")
    for thr in args.delta_thresholds:
        syn_cov = float(np.mean(syn_mag >= thr) * 100)
        real_cov = float(np.mean(real_mag >= thr) * 100)
        print(
            f">= {thr:6.1f} DN : synthetic={syn_cov:7.3f}% "
            f"real={real_cov:7.3f}%"
        )
    print()

    # Signed direction matters because V2 adds cloud radiance and shadow.
    print("--- 6. Signed delta direction ---")
    for name, d in [("Synthetic", syn["delta"]), ("Real", real_delta[None])]:
        flat = d.ravel()
        print(
            f"{name:<10} mean signed delta={np.mean(flat):+.6f} "
            f"| positive={np.mean(flat > 0) * 100:.2f}% "
            f"| negative={np.mean(flat < 0) * 100:.2f}%"
        )
    print()

    # Synthetic-only inside/outside cloud corruption.
    print("--- 7. Synthetic V2 delta by generated mask ---")
    for i, b in enumerate(("G", "R", "NIR")):
        d = syn_abs_dn[:, i]
        inside = m > 0.5
        outside = ~inside

        print(
            f"{b:<3} inside-cloud MAE={d[inside].mean():.3f} DN "
            f"| outside-cloud MAE={d[outside].mean():.3f} DN"
        )
    print()

    print(
        "Interpretation: the real cloudy-clear difference is NOT a true cloud "
        "mask and can contain temporal, illumination, registration, and land-change "
        "effects. Use these distributions to guide generator design, not as a "
        "ground-truth cloud map."
    )


if __name__ == "__main__":
    main()
