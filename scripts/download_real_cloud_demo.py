import argparse
import base64
import os
from pathlib import Path

import numpy as np


API_URL = "https://datasets-server.huggingface.co/rows"


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

    # The Dataset Viewer currently exposes SAR and optical arrays in HWC
    # shapes: SAR [H,W,2], optical [H,W,13].
    if sar.ndim == 3 and sar.shape[-1] == 2:
        sar = np.transpose(sar, (2, 0, 1)).copy()

    if cloudy.ndim == 3 and cloudy.shape[-1] == 13:
        cloudy = np.transpose(cloudy, (2, 0, 1)).copy()
        target = np.transpose(target, (2, 0, 1)).copy()

    return sar, cloudy, target


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download a few real cloudy/cloud-free/SAR SEN12MS-CR samples "
            "through the Hugging Face Dataset Viewer API."
        )
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", default="real_cloud_demo")
    args = parser.parse_args()

    if args.samples < 1 or args.samples > 100:
        raise SystemExit("--samples must be between 1 and 100")

    try:
        import requests
    except ImportError:
        raise SystemExit("Missing dependency: requests. Run: pip install requests")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "dataset": "Hermanni/sen12mscr",
        "config": "default",
        "split": "train",
        "offset": args.offset,
        "length": args.samples,
    }

    print("🌍 Requesting real SEN12MS-CR rows from Hugging Face Dataset Viewer...")
    print("   This reads only the requested rows; it does not download the 389 GB dataset.")
    print(f"   offset={args.offset} length={args.samples}")

    try:
        response = requests.get(API_URL, params=params, timeout=180)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise SystemExit(
            "\n❌ Hugging Face Dataset Viewer request failed.\n"
            f"   {exc}\n"
            "   Try again in a few minutes; the API can temporarily return errors for large datasets."
        )

    rows = payload.get("rows", [])
    if not rows:
        raise RuntimeError(
            f"No rows returned. Server response keys: {list(payload.keys())}"
        )

    total = payload.get("num_rows_total", "?")
    print(f"📦 Server returned {len(rows)} rows out of {total} total.")

    saved = 0

    for item in rows:
        row = item["row"]
        sar, cloudy, target = decode_row(row)

        if sar.shape != (2, 256, 256):
            print(f"⚠️ Skipping row {item.get('row_idx', '?')}: SAR shape {sar.shape}")
            continue
        if cloudy.shape != (13, 256, 256):
            print(f"⚠️ Skipping row {item.get('row_idx', '?')}: cloudy shape {cloudy.shape}")
            continue
        if target.shape != (13, 256, 256):
            print(f"⚠️ Skipping row {item.get('row_idx', '?')}: target shape {target.shape}")
            continue

        sample_id = item.get("row_idx", args.offset + saved)

        np.savez_compressed(
            out_dir / f"sample_{saved + 1:03d}.npz",
            sar=sar.astype(np.float32),
            cloudy=cloudy.astype(np.float32),
            target=target.astype(np.float32),
            season=str(row.get("season", "")),
            scene=str(row.get("scene", "")),
            patch=str(row.get("patch", "")),
            row_idx=int(sample_id),
            source="Hermanni/sen12mscr",
        )

        saved += 1

        print(
            f"✅ [{saved}/{args.samples}] "
            f"row={sample_id} "
            f"season={row.get('season', '?')} "
            f"scene={row.get('scene', '?')} "
            f"patch={row.get('patch', '?')}"
        )

        if saved >= args.samples:
            break

    if saved < args.samples:
        raise RuntimeError(
            f"Only saved {saved} valid samples from the requested {len(rows)} rows."
        )

    print()
    print(f"🎉 Saved {saved} real SEN12MS-CR triplets to: {out_dir.resolve()}")
    print("Each .npz contains:")
    print("  sar    : [2,256,256] float32")
    print("  cloudy : [13,256,256] float32")
    print("  target : [13,256,256] float32")


if __name__ == "__main__":
    main()
