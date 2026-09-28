"""Run the published SAR-fusion DSen2-CR checkpoint on co-registered rasters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dsen2cr import DSen2CR, load_dsen2cr_weights


S2_BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
)
SAR_LIMITS = np.array([[-25.0, 0.0], [-32.5, 0.0]], dtype=np.float32)
OPTICAL_SCALE = 2000.0


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def validate_rasters(s2_path: Path, sar_path: Path) -> dict:
    with rasterio.open(s2_path) as optical, rasterio.open(sar_path) as radar:
        if optical.count != 13:
            raise ValueError(
                f"Sentinel-2 input must have 13 bands in {','.join(S2_BANDS)} order; "
                f"received {optical.count}."
            )
        if radar.count != 2:
            raise ValueError(
                f"Sentinel-1 input must have 2 bands in VV,VH order; received {radar.count}."
            )
        if (optical.width, optical.height) != (radar.width, radar.height):
            raise ValueError("Sentinel-1 and Sentinel-2 dimensions do not match.")
        if optical.crs != radar.crs:
            raise ValueError("Sentinel-1 and Sentinel-2 coordinate reference systems do not match.")
        if not np.allclose(tuple(optical.transform), tuple(radar.transform), rtol=0, atol=1e-8):
            raise ValueError("Sentinel-1 and Sentinel-2 affine transforms do not match.")
        if optical.width < 1 or optical.height < 1:
            raise ValueError("Input rasters are empty.")
        return {
            "width": optical.width,
            "height": optical.height,
            "crs": optical.crs.to_string() if optical.crs else "Not set (pixel patch)",
            "transform": optical.transform,
        }


def normalize_inputs(optical_dn: np.ndarray, sar_db: np.ndarray) -> np.ndarray:
    """Apply the released DSen2-CR preprocessing: S2/2000 and clipped SAR to [0,2]."""
    optical = np.clip(optical_dn, 0.0, 10000.0) / OPTICAL_SCALE
    lower = SAR_LIMITS[:, 0, None, None]
    upper = SAR_LIMITS[:, 1, None, None]
    sar = (np.clip(sar_db, lower, upper) - lower) / (upper - lower) * 2.0
    return np.concatenate((optical, sar), axis=0).astype(np.float32, copy=False)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_inference(
    s2_path: Path,
    sar_path: Path,
    checkpoint: Path,
    output_path: Path,
    tile: int = 256,
    overlap: int = 32,
    device_name: str = "auto",
) -> dict:
    if tile < 32:
        raise ValueError("Tile size must be at least 32 pixels.")
    if overlap < 0 or overlap >= tile:
        raise ValueError("Overlap must be between 0 and tile size - 1.")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    resolved_output = output_path.resolve()
    if resolved_output in {s2_path.resolve(), sar_path.resolve()}:
        raise ValueError("Output path must be different from both input raster paths.")

    info = validate_rasters(s2_path, sar_path)
    device = choose_device(device_name)
    model = DSen2CR().to(device)
    load_dsen2cr_weights(model, str(checkpoint), map_location=device)
    model.eval()

    height, width = info["height"], info["width"]
    prediction_sum = np.zeros((13, height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    valid_map = np.zeros((height, width), dtype=np.uint8)
    y_starts = tile_starts(height, tile, overlap)
    x_starts = tile_starts(width, tile, overlap)

    with rasterio.open(s2_path) as optical, rasterio.open(sar_path) as radar:
        for y in y_starts:
            for x in x_starts:
                h = min(tile, height - y)
                w = min(tile, width - x)
                window = Window(x, y, w, h)
                optical_ma = optical.read(window=window, masked=True).astype(np.float32)
                sar_ma = radar.read(window=window, masked=True).astype(np.float32)
                optical_arr = np.asarray(optical_ma.filled(0), dtype=np.float32)
                sar_arr = np.asarray(sar_ma.filled(0), dtype=np.float32)
                valid = (
                    ~np.ma.getmaskarray(optical_ma).any(axis=0)
                    & ~np.ma.getmaskarray(sar_ma).any(axis=0)
                    & np.isfinite(optical_arr).all(axis=0)
                    & np.isfinite(sar_arr).all(axis=0)
                )
                valid_map[y : y + h, x : x + w] = np.maximum(
                    valid_map[y : y + h, x : x + w], valid.astype(np.uint8)
                )

                model_input = normalize_inputs(optical_arr, sar_arr)
                tensor = torch.from_numpy(model_input[None]).to(device)
                with torch.inference_mode():
                    estimate = model(tensor).squeeze(0).cpu().numpy()
                estimate = np.clip(estimate, 0.0, 5.0)

                # A positive Hann window blends adjacent tiles without zero-weight borders.
                wy = np.maximum(np.hanning(h + 2)[1:-1], 1e-3)
                wx = np.maximum(np.hanning(w + 2)[1:-1], 1e-3)
                weights = np.outer(wy, wx).astype(np.float32)
                prediction_sum[:, y : y + h, x : x + w] += estimate * weights[None]
                weight_sum[y : y + h, x : x + w] += weights

    prediction = prediction_sum / np.maximum(weight_sum[None], 1e-8)
    output_dn = np.rint(np.clip(prediction, 0.0, 5.0) * OPTICAL_SCALE).astype(np.uint16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(s2_path) as optical:
        profile = optical.profile.copy()
    profile.update(
        driver="GTiff",
        count=13,
        dtype="uint16",
        nodata=None,
        compress="deflate",
        predictor=2,
    )
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    if width >= 256 and height >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    else:
        profile.update(tiled=False)
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(output_dn)
        destination.write_mask(valid_map * 255)
        destination.descriptions = S2_BANDS
        destination.update_tags(
            model="published DSen2-CR SAR+CARL checkpoint",
            optical_normalization="clip reflectance DN to 0..10000; divide by 2000",
            sar_normalization="clip VV to -25..0 dB, VH to -32.5..0 dB; scale to 0..2",
            band_order=",".join(S2_BANDS),
        )
    info.update({"device": str(device), "output": str(output_path)})
    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct cloudy Sentinel-2 using co-registered Sentinel-1 SAR."
    )
    parser.add_argument("--s2-cloudy", type=Path, required=True, help="13-band S2 GeoTIFF")
    parser.add_argument("--s1-vv-vh", type=Path, required=True, help="2-band S1 VV,VH GeoTIFF")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "weights" / "dsen2cr_sar_carl.pth",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    info = run_inference(
        args.s2_cloudy,
        args.s1_vv_vh,
        args.checkpoint,
        args.output,
        tile=args.tile,
        overlap=args.overlap,
        device_name=args.device,
    )
    print(
        f"Wrote {args.output} ({info['width']}x{info['height']}, 13 bands, "
        f"{info['crs']}, {info['device']})."
    )


if __name__ == "__main__":
    main()
