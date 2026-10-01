"""Fetch a small, split-labelled SEN12MS-CR patch gallery for the Streamlit demo.

Only Parquet row groups containing selected sample rows are read. The script does
not download or unpack the complete SEN12MS-CR archive.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OPTICAL_BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
)

# Scene assignments follow the split lists published in the SEN12MS-CR mirror
# card, which points to the official TUM dataset. The default gallery samples two
# patches from each of the ten held-out test scenes, plus one train-split example.
TRAIN_SCENE = ("train", "spring", 101, (0,))
TEST_SCENES = [
    ("test", "spring", 31),
    ("test", "spring", 44),
    ("test", "spring", 106),
    ("test", "spring", 123),
    ("test", "spring", 140),
    ("test", "summer", 73),
    ("test", "summer", 119),
    ("test", "fall", 139),
    ("test", "winter", 63),
    ("test", "winter", 108),
]
DEFAULT_TEST_ROWS = (0, 32)
EXPANDED_TEST_ROWS = (0, 16, 32, 48)
MIRROR_ROOT = "https://huggingface.co/datasets/Hermanni/sen12mscr/resolve/main"
# Curated patches from the public SEN12MS-CR-derived cloud-coverage annotation.
# Cloud percentages are from https://zenodo.org/records/17114706; IGBP class 13
# is urban/built-up according to the SEN12MS labels repository.
CLOUD_CHALLENGE_PATCHES = (
    {"season": "spring", "scene": 31, "patch": "p518", "cloud_coverage_pct": 97.25},
    {"season": "spring", "scene": 44, "patch": "p573", "cloud_coverage_pct": 86.79},
    {"season": "summer", "scene": 73, "patch": "p462", "cloud_coverage_pct": 89.70},
    {
        "season": "winter",
        "scene": 108,
        "patch": "p437",
        "cloud_coverage_pct": 63.01,
        "land_cover": "Urban / built-up (IGBP class 13)",
    },
)


def write_raster(path: Path, array: np.ndarray, descriptions: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": array.shape[-2],
        "width": array.shape[-1],
        "count": array.shape[0],
        "dtype": str(array.dtype),
        "transform": Affine.identity(),
        "compress": "deflate",
        "predictor": 2 if np.issubdtype(array.dtype, np.integer) else 3,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
        dst.descriptions = descriptions


def fetch_rows(season: str, scene: int, row_indices: tuple[int, ...]):
    try:
        import fsspec
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Preparing the gallery requires fsspec and pyarrow. Install them with "
            "`pip install fsspec pyarrow`."
        ) from exc

    relative = f"{season}/scene_{scene}.parquet"
    url = f"{MIRROR_ROOT}/{relative}?download=true"
    print(f"Reading requested compressed Parquet row groups: {relative}", flush=True)
    filesystem = fsspec.filesystem("https", block_size=16 * 1024 * 1024)
    with filesystem.open(url, "rb") as remote_file:
        parquet = pq.ParquetFile(remote_file)
        if parquet.metadata.num_row_groups < 1:
            raise RuntimeError(f"No rows found in {relative}.")
        ranges = []
        group_start = 0
        for group_index in range(parquet.metadata.num_row_groups):
            group_stop = group_start + parquet.metadata.row_group(group_index).num_rows
            ranges.append((group_start, group_stop))
            group_start = group_stop
        if max(row_indices) >= group_start:
            raise RuntimeError(
                f"{relative} contains {group_start} rows, "
                f"but patch index {max(row_indices)} was requested."
            )
        selected = {}
        for group_index, (start, stop) in enumerate(ranges):
            in_group = [index for index in row_indices if start <= index < stop]
            if not in_group:
                continue
            table = parquet.read_row_group(group_index)
            selected.update(
                {
                    index: table.slice(index - start, 1).to_pylist()[0]
                    for index in in_group
                }
            )
    return selected


def fetch_patches(season: str, scene: int, patch_ids: tuple[str, ...]) -> dict[str, tuple[int, dict]]:
    """Find exact patch IDs, then read only the Parquet row groups containing them."""
    try:
        import fsspec
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Preparing the gallery requires fsspec and pyarrow. Install them with "
            "`pip install fsspec pyarrow`."
        ) from exc

    relative = f"{season}/scene_{scene}.parquet"
    url = f"{MIRROR_ROOT}/{relative}?download=true"
    print(f"Locating curated patches in {relative}: {', '.join(patch_ids)}", flush=True)
    filesystem = fsspec.filesystem("https", block_size=1024 * 1024)
    with filesystem.open(url, "rb") as remote_file:
        parquet = pq.ParquetFile(remote_file)
        all_patches = parquet.read(columns=["patch"])["patch"].to_pylist()
        wanted = set(patch_ids)
        positions = {
            patch: index for index, patch in enumerate(all_patches) if patch in wanted
        }
        missing = wanted - positions.keys()
        if missing:
            raise ValueError(f"Could not find {sorted(missing)} in {relative}.")

        group_ranges = []
        group_start = 0
        for group_index in range(parquet.metadata.num_row_groups):
            group_size = parquet.metadata.row_group(group_index).num_rows
            group_ranges.append((group_start, group_start + group_size))
            group_start += group_size

        selected: dict[str, tuple[int, dict]] = {}
        for group_index, (start, stop) in enumerate(group_ranges):
            in_group = {
                patch: index
                for patch, index in positions.items()
                if start <= index < stop
            }
            if not in_group:
                continue
            print(f"Reading row group {group_index} for {sorted(in_group)}", flush=True)
            table = parquet.read_row_group(group_index)
            for patch, index in in_group.items():
                selected[patch] = (index, table.slice(index - start, 1).to_pylist()[0])
    return selected


def sample_id_for(split: str, season: str, scene: int, row_index: int) -> str:
    base = f"{'training_source' if split == 'train' else 'heldout'}_{season}_scene{scene}"
    if row_index == 0:
        return base
    return f"{base}_row{row_index}"


def write_case(
    split: str,
    season: str,
    scene: int,
    row_index: int,
    row: dict,
    output_dir: Path,
    overwrite: bool,
    sample_id_override: str | None = None,
    cloud_coverage_pct: float | None = None,
    land_cover: str = "",
) -> dict:
    sample_id = sample_id_override or sample_id_for(split, season, scene, row_index)
    sample_dir = output_dir / sample_id
    paths = {
        "cloudy_s2": sample_dir / "cloudy_s2.tif",
        "sar_s1": sample_dir / "sar_s1_vv_vh.tif",
        "clear_target": sample_dir / "clear_s2_reference.tif",
    }
    if all(path.is_file() for path in paths.values()) and not overwrite:
        print(f"Keeping existing prepared case: {sample_id}", flush=True)
    else:
        if any(path.exists() for path in paths.values()) and not overwrite:
            raise FileExistsError(
                f"Partially prepared case {sample_id}; rerun with --overwrite after checking it."
            )
        optical_shape = tuple(row["opt_shape"])
        sar_shape = tuple(row["sar_shape"])
        cloudy = np.frombuffer(row["cloudy"], dtype=np.int16).reshape(optical_shape)
        clear = np.frombuffer(row["target"], dtype=np.int16).reshape(optical_shape)
        sar = np.frombuffer(row["sar"], dtype=np.float32).reshape(sar_shape)
        cloudy = np.transpose(cloudy, (2, 0, 1)).copy()
        clear = np.transpose(clear, (2, 0, 1)).copy()
        if sar.shape[0] != 2 and sar.shape[-1] == 2:
            sar = np.transpose(sar, (2, 0, 1)).copy()
        if cloudy.shape[0] != 13 or sar.shape[0] != 2 or clear.shape != cloudy.shape:
            raise ValueError(
                f"Unexpected array shapes in {season}/scene_{scene}: "
                f"cloudy={cloudy.shape}, SAR={sar.shape}, target={clear.shape}."
            )
        sample_patch = str(row.get("patch", "first row"))
        write_raster(paths["cloudy_s2"], cloudy, OPTICAL_BANDS)
        write_raster(paths["sar_s1"], sar, ("VV", "VH"))
        write_raster(paths["clear_target"], clear, OPTICAL_BANDS)

    note = (
        "Source dataset's train split; treat this as a training-split illustration, not "
        "an independent held-out evaluation sample."
        if split == "train"
        else "Official source dataset test split, in a scene absent from its listed train scenes."
    )
    if cloud_coverage_pct is not None:
        note += (
            f" Published SEN12MS-CR-derived cloud annotation: "
            f"{cloud_coverage_pct:.2f}% coverage."
        )
    if land_cover:
        note += f" Land-cover label: {land_cover}; this is not a named city or geolocation."
    return {
        "sample_id": sample_id,
        "split": split,
        "season": season,
        "scene": f"scene_{scene}",
        "patch": str(row.get("patch", f"row_{row_index}")),
        "patch_row_index": row_index,
        "cloud_coverage_pct": "" if cloud_coverage_pct is None else f"{cloud_coverage_pct:.2f}",
        "land_cover": land_cover,
        "note": note,
        **{key: str(path.relative_to(output_dir)).replace("\\", "/") for key, path in paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "sentinel_demo"
    )
    parser.add_argument(
        "--all-test",
        action="store_true",
        help="Prepare four patches per test scene instead of two (45 cases total including curated cases).",
    )
    parser.add_argument(
        "--volume",
        action="store_true",
        help=(
            "Prepare the first 96 patches from each of the ten held-out test scenes plus "
            "96 train-split examples (about 1,060 gallery cases including curated patches)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing selected sample files."
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    old_manifest_path = args.output_dir / "manifest.csv"
    old_manifest = {}
    if old_manifest_path.is_file():
        with old_manifest_path.open(newline="", encoding="utf-8") as stream:
            old_manifest = {row["sample_id"]: row for row in csv.DictReader(stream)}

    if args.volume and args.all_test:
        parser.error("Use either --volume or --all-test, not both.")
    train_rows = tuple(range(96)) if args.volume else TRAIN_SCENE[3]
    scene_requests = [(TRAIN_SCENE[0], TRAIN_SCENE[1], TRAIN_SCENE[2], train_rows)]
    test_rows = tuple(range(96)) if args.volume else (
        EXPANDED_TEST_ROWS if args.all_test else DEFAULT_TEST_ROWS
    )
    scene_requests.extend(
        (split, season, scene, test_rows) for split, season, scene in TEST_SCENES
    )
    total_cases = sum(len(row_indices) for _, _, _, row_indices in scene_requests)
    total_cases += len(CLOUD_CHALLENGE_PATCHES)
    manifest_by_id = {}
    completed = 0

    for split, season, scene, row_indices in scene_requests:
        requested = []
        for row_index in row_indices:
            sample_id = sample_id_for(split, season, scene, row_index)
            previous = old_manifest.get(sample_id)
            sample_dir = args.output_dir / sample_id
            files_exist = all(
                (sample_dir / filename).is_file()
                for filename in ("cloudy_s2.tif", "sar_s1_vv_vh.tif", "clear_s2_reference.tif")
            )
            if previous and files_exist and not args.overwrite:
                previous["patch_row_index"] = str(row_index)
                manifest_by_id[sample_id] = previous
            else:
                requested.append(row_index)

        rows = {}
        if requested:
            print(
                f"Reading {season}/scene_{scene} for patch rows {requested}", flush=True
            )
            rows = fetch_rows(season, scene, tuple(requested))

        for row_index in row_indices:
            sample_id = sample_id_for(split, season, scene, row_index)
            completed += 1
            if sample_id in manifest_by_id:
                print(f"[{completed}/{total_cases}] Keeping {sample_id}", flush=True)
                continue
            print(f"[{completed}/{total_cases}] Writing {sample_id}", flush=True)
            manifest_by_id[sample_id] = write_case(
                split, season, scene, row_index, rows[row_index], args.output_dir, args.overwrite
            )

    pending_challenges = []
    for challenge in CLOUD_CHALLENGE_PATCHES:
        sample_id = (
            f"heldout_challenge_{challenge['season']}_scene{challenge['scene']}_"
            f"{challenge['patch']}"
        )
        sample_dir = args.output_dir / sample_id
        files_exist = all(
            (sample_dir / filename).is_file()
            for filename in ("cloudy_s2.tif", "sar_s1_vv_vh.tif", "clear_s2_reference.tif")
        )
        previous = old_manifest.get(sample_id)
        if previous and files_exist and not args.overwrite:
            previous.update(
                {
                    "cloud_coverage_pct": f"{challenge['cloud_coverage_pct']:.2f}",
                    "land_cover": challenge.get("land_cover", ""),
                }
            )
            manifest_by_id[sample_id] = previous
            print(f"Keeping curated challenge case: {sample_id}", flush=True)
        else:
            pending_challenges.append((sample_id, challenge))

    challenges_by_scene: dict[tuple[str, int], list[tuple[str, dict]]] = {}
    for sample_id, challenge in pending_challenges:
        key = (challenge["season"], challenge["scene"])
        challenges_by_scene.setdefault(key, []).append((sample_id, challenge))

    for (season, scene), challenges in challenges_by_scene.items():
        rows_by_patch = fetch_patches(
            season, scene, tuple(challenge["patch"] for _, challenge in challenges)
        )
        for sample_id, challenge in challenges:
            row_index, row = rows_by_patch[challenge["patch"]]
            manifest_by_id[sample_id] = write_case(
                "test",
                season,
                scene,
                row_index,
                row,
                args.output_dir,
                args.overwrite,
                sample_id_override=sample_id,
                cloud_coverage_pct=challenge["cloud_coverage_pct"],
                land_cover=challenge.get("land_cover", ""),
            )

    manifest = list(manifest_by_id.values())

    manifest_path = args.output_dir / "manifest.csv"
    fieldnames = (
        "sample_id", "split", "season", "scene", "patch", "patch_row_index",
        "cloud_coverage_pct", "land_cover", "note", "cloudy_s2", "sar_s1", "clear_target",
    )
    for row in manifest:
        for fieldname in fieldnames:
            row.setdefault(fieldname, "")
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Prepared {len(manifest)} cases. Gallery manifest: {manifest_path}")


if __name__ == "__main__":
    main()
