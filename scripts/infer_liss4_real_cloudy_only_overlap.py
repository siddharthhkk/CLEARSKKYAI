import argparse
import os
import sys
import tempfile

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from liss4 import normalize_liss4
from liss4_dsen2cr import LISS4DSen2CR


def make_weight(h, w):
    """Smooth positive center-weighted window for overlap blending."""
    y = np.clip(np.sin(np.pi * np.linspace(0.0, 1.0, h, dtype=np.float32)), 0.0, 1.0)
    x = np.clip(np.sin(np.pi * np.linspace(0.0, 1.0, w, dtype=np.float32)), 0.0, 1.0)

    wy = 0.15 + 0.85 * np.sqrt(y)
    wx = 0.15 + 0.85 * np.sqrt(x)

    return (wy[:, None] * wx[None, :]).astype(np.float32)


def tile_starts(length, tile, stride):
    """Return starts that cover the entire axis, including the far edge."""
    if length <= tile:
        return [0]

    starts = list(range(0, length - tile + 1, stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)

    return starts


def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(name)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint must contain a 'model_state_dict': {checkpoint_path}"
        )

    model = LISS4DSen2CR(
        features=int(checkpoint.get("features", 256)),
        blocks=int(checkpoint.get("blocks", 16)),
        res_scale=0.1,
        use_sar=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def infer_to_dataset(src, dst, model, device, dn_max, tile, overlap):
    """Infer by tile rows so full-scene float arrays are never held in memory."""
    if src.count != 3:
        raise ValueError(f"Expected 3-band LISS-IV GeoTIFF, got {src.count} bands.")
    if src.width < 1 or src.height < 1:
        raise ValueError(f"Input GeoTIFF is empty: {src.name}")

    stride = tile - overlap
    row_starts = tile_starts(src.height, tile, stride)
    col_starts = tile_starts(src.width, tile, stride)
    accum = None
    weights = None
    buffer_start = row_starts[0]
    tile_count = 0

    for row_index, row in enumerate(row_starts):
        row_end = min(row + tile, src.height)

        if accum is None:
            buffer_start = row
            accum = np.zeros((3, row_end - buffer_start, src.width), dtype=np.float32)
            weights = np.zeros((row_end - buffer_start, src.width), dtype=np.float32)
        else:
            current_end = buffer_start + weights.shape[0]
            if current_end < row_end:
                extra = row_end - current_end
                accum = np.concatenate(
                    [accum, np.zeros((3, extra, src.width), dtype=np.float32)], axis=1
                )
                weights = np.concatenate(
                    [weights, np.zeros((extra, src.width), dtype=np.float32)], axis=0
                )

        for col in col_starts:
            row2 = min(row + tile, src.height)
            col2 = min(col + tile, src.width)
            hh = row2 - row
            ww = col2 - col

            x = src.read(
                indexes=(1, 2, 3),
                window=Window(col, row, ww, hh),
            ).astype(np.float32, copy=False)
            if hh != tile or ww != tile:
                x = np.pad(
                    x,
                    ((0, 0), (0, tile - hh), (0, tile - ww)),
                    mode="edge",
                )

            xt = normalize_liss4(x, dn_max)[None].to(device, non_blocking=True)
            with torch.inference_mode():
                prediction = torch.clamp(model(xt), 0.0, 1.0)

            pred = prediction[0].cpu().numpy()[:, :hh, :ww]
            weight = make_weight(hh, ww)
            top = row - buffer_start
            bottom = top + hh
            accum[:, top:bottom, col:col2] += pred * weight[None]
            weights[top:bottom, col:col2] += weight
            tile_count += 1

        flush_end = (
            row_starts[row_index + 1]
            if row_index + 1 < len(row_starts)
            else src.height
        )
        flush_count = flush_end - buffer_start
        if flush_count <= 0 or flush_count > weights.shape[0]:
            raise RuntimeError("Internal tile-row buffer does not cover the flush range.")
        active_weights = weights[:flush_count]
        if not np.all(active_weights > 0):
            raise RuntimeError(
                "Overlap blending left uncovered pixels. Check tile/overlap settings."
            )

        out = accum[:, :flush_count] / active_weights[None]
        if not np.isfinite(out).all():
            raise RuntimeError("Non-finite values detected in blended output.")

        out_dn = np.clip(np.rint(out * dn_max), 0, dn_max).astype(np.uint16)
        dst.write(out_dn, window=Window(0, buffer_start, src.width, flush_count))

        accum = accum[:, flush_count:]
        weights = weights[flush_count:]
        buffer_start = flush_end

    if buffer_start != src.height:
        raise RuntimeError("Inference ended before all output rows were written.")
    return tile_count, len(row_starts), len(col_starts)


def main():
    ap = argparse.ArgumentParser(
        description="Cloudy-only LISS-IV inference with overlap-tiled GeoTIFF output."
    )
    ap.add_argument(
        "--checkpoint",
        default="weights/liss4_dsen2cr_synthetic_v3r.pth",
    )
    ap.add_argument("--cloudy", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dn-max", type=float, default=1023.0)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap between neighboring tiles in pixels.",
    )
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device; auto selects CUDA when available (default).",
    )
    args = ap.parse_args()

    if args.tile <= 0:
        raise ValueError("--tile must be positive.")
    if not 0 <= args.overlap < args.tile:
        raise ValueError("--overlap must satisfy 0 <= overlap < tile.")
    if args.dn_max <= 0:
        raise ValueError("--dn-max must be positive.")
    if args.dn_max > np.iinfo(np.uint16).max:
        raise ValueError("--dn-max must fit the uint16 output range (<= 65535).")

    input_path = os.path.abspath(args.cloudy)
    output_path = os.path.abspath(args.output)
    if os.path.normcase(input_path) == os.path.normcase(output_path):
        raise ValueError("--output must not overwrite the cloudy input GeoTIFF.")
    if os.path.isdir(output_path):
        raise ValueError("--output must name a file, not a directory.")

    device = select_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(
        prefix=".clearsky-inference-",
        suffix=".tif",
        dir=output_dir,
    )
    os.close(temp_fd)

    try:
        with rasterio.open(input_path) as src:
            if src.count != 3:
                raise ValueError(
                    f"Expected 3-band LISS-IV GeoTIFF, got {src.count} bands."
                )
            if src.width < 1 or src.height < 1:
                raise ValueError(f"Input GeoTIFF is empty: {src.name}")

            profile = src.profile.copy()
            descriptions = src.descriptions
            units = src.units
            dataset_tags = src.tags()
            band_tags = [src.tags(i) for i in range(1, 4)]
            profile.update(
                driver="GTiff",
                dtype="uint16",
                count=3,
                compress="deflate",
                predictor=2,
                BIGTIFF="IF_SAFER",
            )

            with rasterio.open(temp_path, "w", **profile) as dst:
                tile_count, row_count, col_count = infer_to_dataset(
                    src=src,
                    dst=dst,
                    model=model,
                    device=device,
                    dn_max=args.dn_max,
                    tile=args.tile,
                    overlap=args.overlap,
                )
                dst.update_tags(**dataset_tags)
                for band_index in range(1, 4):
                    if descriptions[band_index - 1]:
                        dst.set_band_description(band_index, descriptions[band_index - 1])
                    if units[band_index - 1]:
                        dst.set_band_unit(band_index, units[band_index - 1])
                    if band_tags[band_index - 1]:
                        dst.update_tags(band_index, **band_tags[band_index - 1])

            width, height = src.width, src.height

        os.replace(temp_path, output_path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    print("=== REAL LISS-IV OVERLAP-TILED INFERENCE ===")
    print(f"Device     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(device)}")
    print(f"Checkpoint : {os.path.abspath(args.checkpoint)}")
    if "epoch" in checkpoint:
        print(f"Epoch      : {checkpoint['epoch']}")
    print(f"Input      : {input_path}")
    print(f"Output     : {output_path}")
    print(f"Size       : {width} x {height}")
    print(f"Tile       : {args.tile} px")
    print(f"Overlap    : {args.overlap} px")
    print(f"Tiles      : {tile_count} ({row_count} rows x {col_count} cols)")
    print()
    print(
        "No clear reference was used. Overlap blending reduces tile-boundary "
        "artifacts for qualitative cross-scene evaluation."
    )


if __name__ == "__main__":
    main()
