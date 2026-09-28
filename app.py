"""Four-panel Streamlit demo for published Sentinel DSen2-CR."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import rasterio
import streamlit as st
from rasterio.enums import Resampling


PROJECT_ROOT = Path(__file__).resolve().parent
INFERENCE_SCRIPT = PROJECT_ROOT / "scripts" / "infer_sentinel_dsen2cr.py"
CHECKPOINT = PROJECT_ROOT / "weights" / "dsen2cr_sar_carl.pth"
GALLERY_MANIFEST = PROJECT_ROOT / "data" / "sentinel_demo" / "manifest.csv"
MAX_PREVIEW_EDGE = 900
REFLECTANCE_MAX = 10000.0


def read_gallery() -> list[dict]:
    if not GALLERY_MANIFEST.is_file():
        return []
    with GALLERY_MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("cloudy_s2", "sar_s1", "clear_target"):
            row[key] = (GALLERY_MANIFEST.parent / row[key]).resolve()
    return rows


def stage_upload(uploaded_file, destination: Path) -> Path:
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def _preview_shape(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, MAX_PREVIEW_EDGE / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def read_true_color(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        if source.count < 4:
            raise ValueError(f"Expected at least four optical bands in {path.name}.")
        width, height = _preview_shape(source.width, source.height)
        # True-color order: B04 red, B03 green, B02 blue.
        return source.read(
            indexes=(4, 3, 2),
            out_shape=(3, height, width),
            resampling=Resampling.average,
            masked=True,
        ).filled(0).astype(np.float32)


def read_sar_preview(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        if source.count != 2:
            raise ValueError("Sentinel-1 input must contain VV then VH.")
        width, height = _preview_shape(source.width, source.height)
        vv = source.read(1, out_shape=(height, width), resampling=Resampling.average).astype(np.float32)
    gray = np.clip((vv + 25.0) / 25.0, 0.0, 1.0)
    return np.rint(gray[..., None] * 255).astype(np.uint8)


def make_shared_previews(*images: np.ndarray) -> list[np.ndarray]:
    finite_images = [np.nan_to_num(x, nan=0.0, posinf=REFLECTANCE_MAX, neginf=0.0) for x in images]
    combined = np.concatenate([image.reshape(3, -1) for image in finite_images], axis=1)
    low, high = np.percentile(combined, (2, 98), axis=1)
    high = np.maximum(high, low + 1.0)
    output = []
    for image in finite_images:
        stretched = np.clip(
            (image - low[:, None, None]) / (high - low)[:, None, None], 0.0, 1.0
        )
        output.append(np.rint(np.moveaxis(stretched, 0, -1) * 255).astype(np.uint8))
    return output


def difference_preview(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Turn the absolute true-color-band difference into a compact error heatmap."""
    error = np.abs(left - right)
    magnitude = np.mean(error, axis=0)
    scale = max(float(np.percentile(magnitude, 99)), 1.0)
    value = np.clip(magnitude / scale, 0.0, 1.0)
    red = np.clip(2.5 * value, 0.0, 1.0)
    green = np.clip(2.5 * value - 0.8, 0.0, 1.0)
    blue = np.clip(3.0 * value - 2.2, 0.0, 1.0)
    return np.rint(np.stack((red, green, blue), axis=-1) * 255).astype(np.uint8)


def _score_pair(reference: np.ndarray, estimate: np.ndarray, valid: np.ndarray) -> dict:
    difference = (estimate - reference)[:, valid]
    mae = float(np.abs(difference).mean()) / REFLECTANCE_MAX
    rmse = float(np.sqrt(np.square(difference).mean())) / REFLECTANCE_MAX
    psnr = float("inf") if rmse == 0 else float(20.0 * np.log10(1.0 / rmse))
    return {"MAE": mae, "RMSE": rmse, "PSNR": psnr}


def compare_reference(cloudy_path: Path, output_path: Path, target_path: Path) -> dict | None:
    with (
        rasterio.open(cloudy_path) as cloudy,
        rasterio.open(output_path) as output,
        rasterio.open(target_path) as target,
    ):
        datasets = (cloudy, output, target)
        grid = (cloudy.count, cloudy.width, cloudy.height, cloudy.crs, cloudy.transform)
        if any(
            (ds.count, ds.width, ds.height, ds.crs, ds.transform) != grid for ds in datasets[1:]
        ):
            return None
        cloudy_values = cloudy.read().astype(np.float32)
        output_values = output.read().astype(np.float32)
        target_values = target.read().astype(np.float32)
        valid = np.logical_and.reduce([ds.dataset_mask() > 0 for ds in datasets])
    if not valid.any():
        return None
    return {
        "cloudy": _score_pair(target_values, cloudy_values, valid),
        "output": _score_pair(target_values, output_values, valid),
    }


def run_model(s2_path, sar_path, output_path, tile, overlap, device):
    command = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--s2-cloudy", str(s2_path),
        "--s1-vv-vh", str(sar_path),
        "--checkpoint", str(CHECKPOINT),
        "--output", str(output_path),
        "--tile", str(tile),
        "--overlap", str(overlap),
        "--device", device,
    ]
    return subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )


def readable_case(row: dict) -> str:
    split = "TRAIN DEMO" if row["split"].lower() == "train" else "HELD-OUT TEST"
    tags = []
    if row.get("cloud_coverage_pct"):
        coverage = float(row["cloud_coverage_pct"])
        if coverage >= 80:
            tags.append(f"HEAVY CLOUD {coverage:.0f}%")
        elif row.get("land_cover"):
            tags.append(f"URBAN / BUILT-UP · {coverage:.0f}% CLOUD")
    if row.get("land_cover") and not tags:
        tags.append("URBAN / BUILT-UP")
    prefix = " · ".join(tags + [split])
    return f"{prefix} · {row['season'].title()} · {row['scene']} · patch {row['patch']}"


def render_sidebar(cases: list[dict]):
    st.sidebar.title("🕹️ Control Panel")
    gallery_mode = "Prepared SEN12MS-CR gallery"
    mode = st.sidebar.radio(
        "Input mode",
        (gallery_mode, "Upload raster pair", "Use local file paths"),
        index=0 if cases else 1,
        key="input_mode",
    )
    if CHECKPOINT.is_file():
        st.sidebar.success("Loaded published DSen2-CR SAR + CARL weights")
        st.sidebar.info("Expected input: cloudy S2 + aligned S1 VV/VH → 13-band estimate")
    else:
        st.sidebar.error("Checkpoint missing: weights/dsen2cr_sar_carl.pth")

    cloudy_path = sar_path = target_path = None
    uploads = []
    input_name = ""
    sample_note = ""
    selected = None

    if mode == gallery_mode:
        if cases:
            train_count = sum(row["split"].lower() == "train" for row in cases)
            test_count = len(cases) - train_count
            st.sidebar.success(
                f"Loaded {len(cases)} paired samples · {test_count} test · {train_count} train demo"
            )
            selected_id = st.sidebar.selectbox(
                "Select real-cloud sample",
                [row["sample_id"] for row in cases],
                format_func=lambda sample_id: readable_case(
                    next(row for row in cases if row["sample_id"] == sample_id)
                ),
            )
            selected = next(row for row in cases if row["sample_id"] == selected_id)
            cloudy_path = selected["cloudy_s2"]
            sar_path = selected["sar_s1"]
            target_path = selected["clear_target"]
            sample_note = selected["note"]
            input_name = selected_id
            if selected["split"].lower() == "train":
                st.sidebar.warning("Train-split example: do not present its score as held-out performance.")
            if selected.get("cloud_coverage_pct"):
                coverage = float(selected["cloud_coverage_pct"])
                st.sidebar.info(f"Published cloud-coverage annotation: {coverage:.2f}%")
            if selected.get("land_cover"):
                st.sidebar.info(
                    "Urban/built-up class label; this mirror does not identify a named city or preserve georeferencing."
                )
        else:
            st.sidebar.warning("No local gallery yet. Run `python scripts/prepare_sen12mscr_demo.py`.")
    elif mode == "Upload raster pair":
        s2_upload = st.sidebar.file_uploader(
            "Cloudy Sentinel-2 · 13 bands", type=("tif", "tiff"), key="s2_upload"
        )
        sar_upload = st.sidebar.file_uploader(
            "Sentinel-1 · VV then VH in dB", type=("tif", "tiff"), key="sar_upload"
        )
        uploads = [s2_upload, sar_upload]
        input_name = ", ".join(upload.name for upload in uploads if upload is not None)
    else:
        cloudy_text = st.sidebar.text_input(
            "Cloudy Sentinel-2 path", placeholder=r"C:\data\s2_cloudy_13band.tif"
        )
        sar_text = st.sidebar.text_input(
            "Sentinel-1 VV,VH path", placeholder=r"C:\data\s1_vv_vh.tif"
        )
        if cloudy_text.strip() and sar_text.strip():
            cloudy_path = Path(cloudy_text.strip()).expanduser().resolve()
            sar_path = Path(sar_text.strip()).expanduser().resolve()
            input_name = f"{cloudy_path.name}, {sar_path.name}"

    with st.sidebar.expander("Model input contract", expanded=False):
        st.markdown(
            "**Sentinel-2:** 13 bands in B01, B02, B03, B04, B05, B06, B07, "
            "B08, B8A, B09, B10, B11, B12 order; reflectance DN.\n\n"
            "**Sentinel-1:** two bands in VV, VH order; dB. Both GeoTIFFs must already "
            "share dimensions, CRS, and affine transform. Arbitrary RGB/JPEG images or "
            "unaligned inputs are not compatible."
        )
        st.caption(
            "Meeting the input contract is necessary, not a quality guarantee: the model can "
            "leave cloud artifacts or alter surface detail on unfamiliar scenes."
        )
    with st.sidebar.expander("Inference settings"):
        tile = st.select_slider("Tile size", options=(128, 256, 384, 512), value=256)
        overlap = st.slider("Tile overlap (pixels)", 0, tile - 1, min(32, tile - 1))
        device = st.selectbox("Device", ("auto", "cpu", "cuda"), index=0)
    run_clicked = st.sidebar.button("☁️ Run Real Cloud Removal", type="primary", width="stretch")
    return {
        "mode": mode,
        "cloudy_path": cloudy_path,
        "sar_path": sar_path,
        "target_path": target_path,
        "uploads": uploads,
        "input_name": input_name,
        "sample_note": sample_note,
        "selected": selected,
        "tile": tile,
        "overlap": overlap,
        "device": device,
        "run_clicked": run_clicked,
    }


def run_selected_input(selection: dict) -> None:
    if not CHECKPOINT.is_file():
        st.error("Download and convert the published checkpoint first; see `weights/README.md`.")
        return
    if selection["mode"] == "Upload raster pair":
        if any(upload is None for upload in selection["uploads"]):
            st.sidebar.error("Upload both the cloudy S2 and aligned VV/VH SAR GeoTIFFs.")
            return
    elif selection["cloudy_path"] is None or selection["sar_path"] is None:
        st.sidebar.error("Select a gallery sample or supply both required raster paths.")
        return

    run_directory = tempfile.TemporaryDirectory(prefix="clearsky_sentinel_")
    try:
        work = Path(run_directory.name)
        cloudy_path = selection["cloudy_path"]
        sar_path = selection["sar_path"]
        if selection["mode"] == "Upload raster pair":
            cloudy_path = stage_upload(selection["uploads"][0], work / "cloudy_s2.tif")
            sar_path = stage_upload(selection["uploads"][1], work / "sar_s1.tif")
        output_path = work / "dsen2cr_reconstruction.tif"

        with st.spinner("Checking the grids and running the published 13-band model…"):
            completed = run_model(
                cloudy_path,
                sar_path,
                output_path,
                selection["tile"],
                selection["overlap"],
                selection["device"],
            )
            cloudy_raw = read_true_color(cloudy_path)
            output_raw = read_true_color(output_path)
            target_raw = read_true_color(selection["target_path"]) if selection["target_path"] else None
            inputs = [cloudy_raw, output_raw]
            if target_raw is not None:
                inputs.append(target_raw)
            previews = make_shared_previews(*inputs)
            sar_preview = read_sar_preview(sar_path)
            if target_raw is not None:
                difference = difference_preview(previews[1], previews[2])
                metrics = compare_reference(cloudy_path, output_path, selection["target_path"])
                difference_caption = "Absolute error vs paired clear reference (relative color scale)"
            else:
                difference = difference_preview(previews[0], previews[1])
                metrics = None
                difference_caption = "Change from cloudy input (not a reference-based error map)"
            with rasterio.open(output_path) as result_ds:
                info = {
                    "width": result_ds.width,
                    "height": result_ds.height,
                    "crs": result_ds.crs.to_string() if result_ds.crs else "No CRS (pixel patch)",
                    "dtype": result_ds.dtypes[0],
                }

        previous = st.session_state.get("result_tempdir")
        if previous is not None:
            previous.cleanup()
        st.session_state["result_tempdir"] = run_directory
        st.session_state["last_result"] = {
            "output": str(output_path),
            "input_name": selection["input_name"],
            "sample_note": selection["sample_note"],
            "sample": selection["selected"],
            "cloudy": previews[0],
            "estimate": previews[1],
            "target": previews[2] if target_raw is not None else None,
            "sar": sar_preview,
            "difference": difference,
            "difference_caption": difference_caption,
            "metrics": metrics,
            "stdout": completed.stdout,
            **info,
        }
        run_directory = None
    except subprocess.CalledProcessError as exc:
        lines = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()
        st.error("Reconstruction failed. Check the band order, units, checkpoint, and co-registration.")
        st.code("\n".join(lines[-18:]))
    except Exception as exc:
        st.error(f"Could not run DSen2-CR: {exc}")
    finally:
        if run_directory is not None:
            run_directory.cleanup()


def display_results(result: dict | None, current_input: str) -> None:
    st.markdown(
        "<style>"
        ".block-container{padding-top:1.5rem;padding-bottom:2rem;}"
        "[data-testid='stSidebar']{background:#20212b;}"
        "[data-testid='stMetric']{background:#171922;padding:14px;border-radius:10px;}"
        "[data-testid='stImage'] img{border-radius:8px;}"
        "</style>",
        unsafe_allow_html=True,
    )
    st.title("🛰️ ClearSky-AI: SAR–Optical Cloud Removal")
    st.caption(
        "Multimodal reconstruction · Published DSen2-CR checkpoint · "
        "SEN12MS-CR offline demo gallery"
    )

    if result is None or result["input_name"] != current_input:
        st.info("Choose a paired sample or provide both inputs in the control panel, then run reconstruction.")
        return

    st.caption(
        f"Scene: {result['width']:,} × {result['height']:,} px · {result['crs']} · "
        f"checkpoint `dsen2cr_sar_carl.pth`"
    )
    columns = st.columns(4, gap="small")
    headings = ("1. SAR (Radar)", "2. Real Cloudy S2", "3. DSen2-CR Output", "4. Cloud-free S2")
    images = (result["sar"], result["cloudy"], result["estimate"], result["target"])
    descriptions = (
        "Sentinel-1 VV backscatter",
        "Genuine cloudy Sentinel-2 observation",
        "Published SAR-guided reconstruction",
        "Paired clear reference" if result["target"] is not None else "No reference supplied",
    )
    for column, heading, image, caption in zip(columns, headings, images, descriptions):
        column.markdown(f"### {heading}")
        if image is None:
            column.info("A clear target is available only for paired gallery samples.")
        else:
            column.image(image, caption=caption, width="stretch")

    st.divider()
    scene_col, difference_col, metrics_col = st.columns([1.0, 1.25, 2.1], gap="large")
    scene_col.markdown("### 🛰️ Scene")
    sample = result["sample"]
    if sample:
        scene_col.write(f"**Split:** {sample['split'].upper()}")
        scene_col.write(f"**Season:** {sample['season'].title()}")
        scene_col.write(f"**Scene:** {sample['scene']}")
        scene_col.write(f"**Patch:** {sample['patch']}")
        if sample.get("cloud_coverage_pct"):
            scene_col.write(f"**Annotated cloud coverage:** {float(sample['cloud_coverage_pct']):.2f}%")
        if sample.get("land_cover"):
            scene_col.write(f"**Land cover:** {sample['land_cover']}")
        scene_col.caption(sample["note"])
    else:
        scene_col.write("**Source:** User-provided raster pair")
        scene_col.write(f"**Input files:** {result['input_name']}")

    difference_col.markdown("### 🔎 Reconstruction Difference")
    difference_col.image(
        result["difference"],
        caption=result["difference_caption"],
        width="stretch",
    )

    metrics_col.markdown("### 📊 Reference Comparison")
    metrics = result["metrics"]
    if metrics is None:
        metrics_col.info("No paired clear target: quality metrics are not available for this input.")
    else:
        top = metrics_col.columns(3)
        top[0].metric("Cloudy Input PSNR", f"{metrics['cloudy']['PSNR']:.2f} dB")
        top[1].metric("DSen2-CR PSNR", f"{metrics['output']['PSNR']:.2f} dB")
        gain = metrics["output"]["PSNR"] - metrics["cloudy"]["PSNR"]
        top[2].metric("PSNR Gain", f"{gain:+.2f} dB")
        bottom = metrics_col.columns(2)
        bottom[0].metric("Cloudy Input MAE", f"{metrics['cloudy']['MAE']:.4f}")
        bottom[1].metric("DSen2-CR MAE", f"{metrics['output']['MAE']:.4f}")
        st.caption(
            "Metrics use all 13 bands, reflectance normalized to 0–1. The clear reference is "
            "co-registered but may be from another date; treat these as paired-reference scores, "
            "not perfect ground truth. Train-split scores are not held-out performance."
        )

    st.download_button(
        "Download reconstructed 13-band GeoTIFF",
        data=lambda: Path(result["output"]).read_bytes(),
        file_name="dsen2cr_sentinel2_reconstruction.tif",
        mime="image/tiff",
        type="primary",
    )
    with st.expander("Run details"):
        st.write(f"Input: {result['input_name']}")
        st.write(f"Output pixel type: {result['dtype']}")
        st.code(result["stdout"].strip())


def main() -> None:
    st.set_page_config(page_title="ClearSky-AI · DSen2-CR", page_icon="🛰️", layout="wide")
    cases = read_gallery()
    selection = render_sidebar(cases)
    if selection["run_clicked"]:
        run_selected_input(selection)

    current_input = selection["input_name"]
    result = st.session_state.get("last_result")
    display_results(result, current_input)
    st.caption(
        "Cloud removal is reconstruction, not recovery of guaranteed ground truth. "
        "For the preserved LISS-IV prototype, run `streamlit run app_liss4.py`."
    )


main()
