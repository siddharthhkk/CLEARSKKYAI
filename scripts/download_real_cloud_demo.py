import argparse
import os
from pathlib import Path

import numpy as np


def decode_row(row):
    sar = np.frombuffer(
        row["sar"],
        dtype=np.float32,
    ).reshape(row["sar_shape"]).copy()

    cloudy = np.frombuffer(
        row["cloudy"],
        dtype=np.int16,
    ).reshape(row["opt_shape"]).astype(np.float32)

    target = np.frombuffer(
        row["target"],
        dtype=np.int16,
    ).reshape(row["opt_shape"]).astype(np.float32)

    # Dataset stores optical data as HWC.
    cloudy = np.transpose(cloudy, (2, 0, 1))
    target = np.transpose(target, (2, 0, 1))

    return sar, cloudy, target


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stream a small number of real cloudy/cloud-free/SAR triplets "
            "from the public SEN12MS-CR Hugging Face mirror."
        )
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--buffer",
        type=int,
        default=100,
        help="Shuffle buffer used by Hugging Face streaming. Smaller values start faster.",
    )
    parser.add_argument(
        "--out",
        default="real_cloud_demo",
        help="Output directory.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "Missing dependency: datasets. Install it with: pip install datasets"
        )

    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🌍 Loading SEN12MS-CR in streaming mode...")
    print("   Dataset: Hermanni/sen12mscr")
    print("   No full 389 GB download is required.")

    ds = load_dataset(
        "Hermanni/sen12mscr",
        split="train",
        streaming=True,
    )

    if args.seed is not None:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.buffer)

    saved = 0

    for row in ds:
        sar, cloudy, target = decode_row(row)

        if sar.shape != (2, 256, 256):
            continue
        if cloudy.shape != (13, 256, 256):
            continue
        if target.shape != (13, 256, 256):
            continue

        name = (
            f"{row.get('season', 'unknown')}_"
            f"scene_{row.get('scene', 'unknown')}_"
            f"patch_{row.get('patch', 'unknown')}"
        )

        np.savez_compressed(
            out_dir / f"sample_{saved + 1:03d}.npz",
            sar=sar.astype(np.float32),
            cloudy=cloudy.astype(np.float32),
            target=target.astype(np.float32),
            season=str(row.get("season", "")),
            scene=str(row.get("scene", "")),
            patch=str(row.get("patch", "")),
            source="Hermanni/sen12mscr",
        )

        saved += 1

        print(
            f"✅ [{saved}/{args.samples}] "
            f"{name} | cloudy={cloudy.shape} | SAR={sar.shape}"
        )

        if saved >= args.samples:
            break

    if saved < args.samples:
        raise RuntimeError(
            f"Stream ended after {saved} valid samples; requested {args.samples}."
        )

    print()
    print(f"🎉 Saved {saved} real SEN12MS-CR triplets to: {out_dir.resolve()}")
    print()
    print("Each .npz contains:")
    print("  sar    : [2,256,256] float32")
    print("  cloudy : [13,256,256] float32")
    print("  target : [13,256,256] float32")
    print("  season / scene / patch metadata")


if __name__ == "__main__":
    main()
