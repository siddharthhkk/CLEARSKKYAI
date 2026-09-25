import argparse
import base64
from pathlib import Path

import numpy as np
import requests

API_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "Hermanni/sen12mscr"
CONFIG = "default"
SPLIT = "train"


def decode_binary(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return base64.b64decode(value)
    if isinstance(value, dict):
        for key in ("bytes", "base64", "data", "encoded"):
            if key in value and isinstance(value[key], str):
                return base64.b64decode(value[key])
    raise TypeError(f"Unsupported binary cell type: {type(value)}")


def decode_row(row):
    sar_shape = tuple(row["sar_shape"])
    opt_shape = tuple(row["opt_shape"])

    sar = np.frombuffer(
        decode_binary(row["sar"]),
        dtype=np.float32,
    ).reshape(sar_shape).copy()

    cloudy = np.frombuffer(
        decode_binary(row["cloudy"]),
        dtype=np.int16,
    ).reshape(opt_shape).astype(np.float32)

    target = np.frombuffer(
        decode_binary(row["target"]),
        dtype=np.int16,
    ).reshape(opt_shape).astype(np.float32)

    if sar.ndim == 3 and sar.shape[-1] == 2:
        sar = np.transpose(sar, (2, 0, 1)).copy()

    if cloudy.ndim == 3 and cloudy.shape[-1] == 13:
        cloudy = np.transpose(cloudy, (2, 0, 1)).copy()
        target = np.transpose(target, (2, 0, 1)).copy()

    return sar, cloudy, target


def get_rows(offset, length):
    response = requests.get(
        API_URL,
        params={
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def save_row(item, out_dir, number):
    row = item["row"]
    sar, cloudy, target = decode_row(row)

    if sar.shape != (2, 256, 256):
        raise ValueError(f"SAR shape {sar.shape}")
    if cloudy.shape != (13, 256, 256):
        raise ValueError(f"Cloudy shape {cloudy.shape}")
    if target.shape != (13, 256, 256):
        raise ValueError(f"Target shape {target.shape}")

    row_idx = int(item.get("row_idx", -1))

    np.savez_compressed(
        out_dir / f"sample_{number:03d}.npz",
        sar=sar.astype(np.float32),
        cloudy=cloudy.astype(np.float32),
        target=target.astype(np.float32),
        season=str(row.get("season", "")),
        scene=str(row.get("scene", "")),
        patch=str(row.get("patch", "")),
        row_idx=row_idx,
        source=DATASET,
    )

    return {
        "row_idx": row_idx,
        "season": str(row.get("season", "")),
        "scene": str(row.get("scene", "")),
        "patch": str(row.get("patch", "")),
    }


def main():
    p = argparse.ArgumentParser(
        description=(
            "Download real SEN12MS-CR samples from widely separated row "
            "positions for a more diverse inference check."
        )
    )
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="real_cloud_diverse")
    args = p.parse_args()

    if not 1 <= args.samples <= 50:
        raise SystemExit("--samples must be between 1 and 50")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🌍 Querying SEN12MS-CR Dataset Viewer...")
    first = get_rows(0, 1)
    total = int(first.get("num_rows_total", 0))
    if total <= 0:
        raise RuntimeError("Dataset Viewer did not return a valid total row count.")

    print(f"📦 Total rows available: {total}")

    rng = np.random.default_rng(args.seed)
    offsets = sorted(
        rng.choice(total, size=min(args.samples, total), replace=False).tolist()
    )

    print(
        f"🎯 Selecting {len(offsets)} widely separated random row positions "
        f"(seed={args.seed})..."
    )

    saved = 0
    metadata = []

    for offset in offsets:
        payload = get_rows(offset, 1)
        rows = payload.get("rows", [])
        if not rows:
            print(f"⚠️ No row returned at offset={offset}; skipping.")
            continue

        try:
            info = save_row(rows[0], out_dir, saved + 1)
        except (KeyError, TypeError, ValueError) as exc:
            print(f"⚠️ Invalid row at offset={offset}: {exc}; skipping.")
            continue

        saved += 1
        metadata.append(info)

        print(
            f"✅ [{saved}/{len(offsets)}] "
            f"row={info['row_idx']} "
            f"season={info['season']} "
            f"scene={info['scene']} "
            f"patch={info['patch']}"
        )

    unique_scenes = len(
        {(m["season"], m["scene"]) for m in metadata}
    )

    print()
    print(f"🎉 Saved {saved} real-cloud samples to: {out_dir.resolve()}")
    print(f"🌐 Unique season/scene combinations: {unique_scenes}")
    print("   The downloaded samples are ignored by Git.")
    print()
    print("Run the batch comparison with:")
    print(
        "  python scripts/compare_real_models_batch.py "
        f'--dir "{out_dir}"'
    )


if __name__ == "__main__":
    main()
