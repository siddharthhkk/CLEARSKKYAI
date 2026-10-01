"""Create a window-written [Green, Red, NIR] stack from aligned LISS-IV bands."""

import argparse
import os

import numpy as np
import rasterio
from rasterio.windows import Window


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stack co-registered single-band LISS-IV Green, Red, and NIR "
            "GeoTIFFs without loading whole scenes into memory."
        )
    )
    parser.add_argument("--green", required=True, help="BAND2 input")
    parser.add_argument("--red", required=True, help="BAND3 input")
    parser.add_argument("--nir", required=True, help="BAND4 input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = [os.path.abspath(path) for path in (args.green, args.red, args.nir)]
    output = os.path.abspath(args.output)
    if output in inputs:
        raise ValueError("Output path must not overwrite one of the input bands")
    if os.path.exists(output) and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {output}")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    with rasterio.open(inputs[0]) as green, rasterio.open(inputs[1]) as red, rasterio.open(
        inputs[2]
    ) as nir:
        sources = [green, red, nir]
        reference = green
        for label, source in zip(("Green", "Red", "NIR"), sources):
            if source.count != 1:
                raise ValueError(f"{label} input must contain exactly one band")
            same_grid = (
                (source.height, source.width) == (reference.height, reference.width)
                and source.crs == reference.crs
                and source.transform == reference.transform
                and np.allclose(source.res, reference.res, atol=1e-6)
            )
            if not same_grid:
                raise ValueError(
                    f"{label} input is not on the same grid as Green/BAND2"
                )
            if source.nodata != reference.nodata:
                raise ValueError(
                    f"{label} nodata value differs from Green/BAND2; "
                    "prepare a consistent triplet before stacking"
                )

        output_dtype = np.result_type(*(np.dtype(source.dtypes[0]) for source in sources))
        profile = reference.profile.copy()
        profile.update(
            driver="GTiff",
            count=3,
            dtype=output_dtype.name,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        if output_dtype.kind in "iu":
            profile["predictor"] = 2
        elif output_dtype.kind == "f":
            profile["predictor"] = 3

        with rasterio.open(output, "w", **profile) as destination:
            for band_index, source in enumerate(sources, start=1):
                destination.set_band_description(
                    band_index, ("Green", "Red", "NIR")[band_index - 1]
                )
                for top in range(0, reference.height, 512):
                    for left in range(0, reference.width, 512):
                        height = min(512, reference.height - top)
                        width = min(512, reference.width - left)
                        window = Window(left, top, width, height)
                        destination.write(
                            source.read(1, window=window).astype(output_dtype, copy=False),
                            band_index,
                            window=window,
                        )
                        if band_index == 1:
                            valid = np.ones((height, width), dtype=bool)
                            for mask_source in sources:
                                valid &= mask_source.read_masks(1, window=window) > 0
                            destination.write_mask(
                                valid.astype(np.uint8) * 255,
                                window=window,
                            )

    print(f"Wrote [Green, Red, NIR] stack: {output}")


if __name__ == "__main__":
    main()
