"""Legacy LISS-IV Streamlit interface retained alongside the Sentinel demo."""

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
INFERENCE_SCRIPT = PROJECT_ROOT / "liss4" / "scripts" / "infer_liss4_real_cloudy_only_overlap.py"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DEMO_MANIFEST = PROJECT_ROOT / "data" / "liss4_demo" / "manifest.csv"
MAX_PREVIEW_EDGE = 1200
DEMO_LFS_PENDING = False


def is_lfs_pointer(path):
    """Return whether a demo asset is present only as a Git LFS pointer."""
    if not path.is_file():
        return False
    try:
        with path.open("rb") as stream:
            return stream.read(128).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            )
    except OSError:
        return False


def read_demo_gallery():
    """Load available, bundled LISS-IV demo samples from the manifest."""
    global DEMO_LFS_PENDING
    DEMO_LFS_PENDING = False
    if not DEMO_MANIFEST.is_file():
        return []

    with DEMO_MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    available = []
    for row in rows:
        path = (DEMO_MANIFEST.parent / row["cloudy_tif"]).resolve()
        if is_lfs_pointer(path):
            DEMO_LFS_PENDING = True
            continue
        if not path.is_file():
            continue
        row["cloudy_path"] = path
        available.append(row)
    return available


def checkpoint_paths():
    """Return available LISS-IV checkpoints, excluding resumable latest states."""
    return sorted(
        (
            path
            for path in WEIGHTS_DIR.glob("liss4_*.pth")
            if "latest" not in path.stem
        ),
        key=lambda path: (path.name != "liss4_dsen2cr_synthetic_v3r.pth", path.name),
    )


def stage_upload(uploaded_file, destination):
    """Copy an uploaded TIFF to the per-run temporary directory."""
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def validate_sources(paths, separate_bands):
    """Validate the selected GeoTIFF grid before starting a potentially long run."""
    if len(paths) != (3 if separate_bands else 1):
        raise ValueError("Select one three-band TIFF or all three separate band TIFFs.")
    if separate_bands and len({path.resolve() for path in paths}) != 3:
        raise ValueError("Select three distinct TIFF files for Green, Red, and NIR.")
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError("Input file not found: " + ", ".join(missing))

    with rasterio.open(paths[0]) as reference:
        if separate_bands:
            if reference.count != 1:
                raise ValueError("The Green/BAND2 file must contain exactly one band.")
            for path, label in zip(paths[1:], ("Red/BAND3", "NIR/BAND4")):
                with rasterio.open(path) as band:
                    if band.count != 1:
                        raise ValueError(f"The {label} file must contain exactly one band.")
                    same_shape = (band.width, band.height) == (reference.width, reference.height)
                    same_grid = (
                        band.crs == reference.crs
                        and band.transform == reference.transform
                        and np.allclose(band.res, reference.res, atol=1e-6)
                    )
                    if not same_shape or not same_grid:
                        raise ValueError(
                            f"The {label} file is not on the same grid as Green/BAND2."
                        )
        elif reference.count != 3:
            raise ValueError(
                f"Expected a three-band GeoTIFF in Green, Red, NIR order; "
                f"this file has {reference.count} bands."
            )

        if reference.width < 1 or reference.height < 1:
            raise ValueError("The selected raster is empty.")
        return {
            "width": reference.width,
            "height": reference.height,
            "crs": reference.crs.to_string() if reference.crs else "Not set",
            "resolution": reference.res,
        }


def read_preview(path):
    """Read a downsampled preview so the UI doesn't load a full scene for display."""
    with rasterio.open(path) as source:
        scale = min(1.0, MAX_PREVIEW_EDGE / max(source.width, source.height))
        out_width = max(1, round(source.width * scale))
        out_height = max(1, round(source.height * scale))
        return source.read(
            indexes=(1, 2, 3),
            out_shape=(3, out_height, out_width),
            resampling=Resampling.average,
        ).astype(np.float32)


def read_separate_band_previews(paths):
    """Read three aligned single-band inputs at a common preview size."""
    with rasterio.open(paths[0]) as source:
        scale = min(1.0, MAX_PREVIEW_EDGE / max(source.width, source.height))
        out_width = max(1, round(source.width * scale))
        out_height = max(1, round(source.height * scale))
    band_previews = []
    for path in paths:
        with rasterio.open(path) as source:
            band_previews.append(
                source.read(
                    1,
                    out_shape=(out_height, out_width),
                    resampling=Resampling.average,
                )
            )
    return np.stack(band_previews).astype(np.float32)


def false_color_previews(input_preview, output_preview):
    """Create side-by-side comparable NIR/Red/Green false-color previews."""
    input_preview = np.nan_to_num(input_preview, nan=0.0, posinf=65535.0, neginf=0.0)
    output_preview = np.nan_to_num(output_preview, nan=0.0, posinf=65535.0, neginf=0.0)
    combined = np.concatenate(
        (input_preview.reshape(3, -1), output_preview.reshape(3, -1)), axis=1
    )
    low, high = np.percentile(combined, (2, 98), axis=1)
    high = np.where(high > low, high, low + 1.0)

    def to_rgb(image):
        stretched = np.clip(
            (image - low[:, None, None]) / (high - low)[:, None, None], 0.0, 1.0
        )
        # LISS-IV has no blue band: map NIR, Red, Green for a familiar false-color view.
        rgb = np.stack((stretched[2], stretched[1], stretched[0]), axis=-1)
        return np.rint(rgb * 255).astype(np.uint8)

    return to_rgb(input_preview), to_rgb(output_preview)


def run_inference(
    input_paths,
    separate_bands,
    checkpoint,
    output_path,
    tile,
    overlap,
    device,
    diffcr_root=None,
):
    command = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output_path),
        "--tile",
        str(tile),
        "--overlap",
        str(overlap),
        "--device",
        device,
    ]
    if diffcr_root:
        command.extend(("--diffcr-root", diffcr_root))
    if separate_bands:
        command.extend(("--cloudy-bands", *(str(path) for path in input_paths)))
    else:
        command.extend(("--cloudy", str(input_paths[0])))

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def read_download(path):
    """Read the result only when the user requests the download."""
    return Path(path).read_bytes()


st.set_page_config(page_title="ClearSky-AI", page_icon="🌤️", layout="wide")
st.title("ClearSky-AI")
st.caption("LISS-IV cloud reconstruction · DSen2-CR-inspired research prototype")

st.info(
    "Provide cloudy LISS-IV Green, Red, and NIR imagery. The model estimates a "
    "reconstruction; it does not produce a cloud mask or guarantee recovery of "
    "hidden surface detail. Results are experimental—inspect them before use."
)

available_checkpoints = checkpoint_paths()
if not available_checkpoints:
    st.error(
        "No LISS-IV checkpoint was found. Run `git lfs pull` to fetch the included "
        "project-trained V3R weights, or place another compatible checkpoint in "
        "the `weights` folder and follow the README instructions."
    )
    st.stop()

checkpoint_labels = {path.name: path for path in available_checkpoints}
preferred_checkpoint = "liss4_dsen2cr_synthetic_v3r.pth"
default_index = (
    list(checkpoint_labels).index(preferred_checkpoint)
    if preferred_checkpoint in checkpoint_labels
    else 0
)
checkpoint_name = st.sidebar.selectbox(
    "Model checkpoint",
    options=list(checkpoint_labels),
    index=default_index,
    help="The single-scene V3R checkpoint is the recommended demo baseline.",
)
checkpoint = checkpoint_labels[checkpoint_name]
diffcr_root = st.sidebar.text_input(
    "DiffCR source folder (optional)",
    value="",
    help=(
        "Only needed for a transferred DiffCR checkpoint when its saved source "
        "path is unavailable on this computer."
    ),
)

demo_mode = "Prepared LISS-IV demo"
input_source = st.sidebar.selectbox(
    "Image source",
    (demo_mode, "Upload TIFF(s)", "Use local file path(s)"),
)
input_paths = []
input_names = []
separate_bands = False

if input_source == demo_mode:
    demo_samples = read_demo_gallery()
    if DEMO_LFS_PENDING:
        st.sidebar.warning("The demo image needs Git LFS. Run `git lfs pull`.")
    if demo_samples:
        st.sidebar.caption(f"{len(demo_samples)} prepared demo samples")
        sample_by_label = {sample["label"]: sample for sample in demo_samples}
        selected_label = st.sidebar.selectbox(
            "Demo sample", options=list(sample_by_label), key="liss4_demo_sample"
        )
        selected_demo = sample_by_label[selected_label]
        input_paths = [selected_demo["cloudy_path"]]
        input_names = [selected_demo["cloudy_path"].name]
        st.sidebar.caption(selected_demo["description"])
    else:
        st.sidebar.info(
            "No bundled demo is available. Fetch it with `git lfs pull`, or choose "
            "Upload TIFF(s) / Use local file path(s)."
        )
elif input_source == "Upload TIFF(s)":
    input_format = st.radio(
        "Input format",
        ("One stacked 3-band GeoTIFF", "Three separate single-band TIFFs"),
        horizontal=True,
    )
    separate_bands = input_format.startswith("Three separate")
    st.caption("Large scenes are easier to run by local path; browser uploads have a size limit.")
    if separate_bands:
        upload_columns = st.columns(3)
        band_specs = (
            ("Green / BAND2", "green.tif"),
            ("Red / BAND3", "red.tif"),
            ("NIR / BAND4", "nir.tif"),
        )
        for column, (label, filename) in zip(upload_columns, band_specs):
            uploaded = column.file_uploader(
                label, type=("tif", "tiff"), key=f"upload_{filename}"
            )
            if uploaded is not None:
                input_paths.append(uploaded)
                input_names.append(uploaded.name)
    else:
        uploaded = st.file_uploader(
            "Cloudy three-band GeoTIFF (Green, Red, NIR)",
            type=("tif", "tiff"),
            key="upload_stacked",
        )
        if uploaded is not None:
            input_paths = [uploaded]
            input_names = [uploaded.name]
else:
    input_format = st.radio(
        "Input format",
        ("One stacked 3-band GeoTIFF", "Three separate single-band TIFFs"),
        horizontal=True,
    )
    separate_bands = input_format.startswith("Three separate")
    if separate_bands:
        path_columns = st.columns(3)
        band_specs = (
            ("Green / BAND2 path", "green_path"),
            ("Red / BAND3 path", "red_path"),
            ("NIR / BAND4 path", "nir_path"),
        )
        for column, (label, key) in zip(path_columns, band_specs):
            value = column.text_input(label, key=key, placeholder=r"C:\data\BAND2.tif")
            if value.strip():
                input_paths.append(Path(value.strip()).expanduser().resolve())
                input_names.append(Path(value.strip()).name)
    else:
        value = st.text_input(
            "Cloudy three-band GeoTIFF path",
            placeholder=r"C:\data\cloudy_liss4.tif",
            key="stacked_path",
        )
        if value.strip():
            input_paths = [Path(value.strip()).expanduser().resolve()]
            input_names = [Path(value.strip()).name]

band_order_confirmed = st.checkbox(
    "I confirm the optical bands are ordered Green, Red, NIR (BAND2, BAND3, BAND4)."
)

with st.expander("Inference settings", expanded=False):
    setting_columns = st.columns(3)
    tile = setting_columns[0].select_slider("Tile size", options=(128, 256, 384, 512), value=256)
    overlap = setting_columns[1].slider(
        "Tile overlap (pixels)", min_value=0, max_value=tile - 1, value=min(64, tile - 1)
    )
    device = setting_columns[2].selectbox("Device", ("auto", "cpu", "cuda"), index=0)

run_clicked = st.button("Reconstruct cloudy image", type="primary", width="stretch")
if run_clicked:
    if not band_order_confirmed:
        st.error("Confirm the Green, Red, NIR band order before running inference.")
    elif len(input_paths) != (3 if separate_bands else 1):
        st.error("Select all required TIFF inputs before running inference.")
    else:
        run_directory = tempfile.TemporaryDirectory(prefix="clearsky_streamlit_")
        try:
            temp_dir = Path(run_directory.name)
            if input_source == "Upload TIFF(s)":
                staged_paths = [
                    stage_upload(upload, temp_dir / f"input_{index}.tif")
                    for index, upload in enumerate(input_paths)
                ]
            else:
                staged_paths = input_paths

            with st.spinner("Checking raster grids and running tiled inference…"):
                raster_info = validate_sources(staged_paths, separate_bands)
                output_path = temp_dir / "reconstructed.tif"
                completed = run_inference(
                    staged_paths,
                    separate_bands,
                    checkpoint,
                    output_path,
                    tile,
                    overlap,
                    device,
                    diffcr_root.strip() or None,
                )
                if separate_bands:
                    input_preview = read_separate_band_previews(staged_paths)
                else:
                    input_preview = read_preview(staged_paths[0])
                output_preview = read_preview(output_path)
                input_rgb, output_rgb = false_color_previews(input_preview, output_preview)
                with rasterio.open(output_path) as output_dataset:
                    output_crs = (
                        output_dataset.crs.to_string() if output_dataset.crs else "Not set"
                    )
                    output_dtype = output_dataset.dtypes[0]

            if input_source == "Upload TIFF(s)":
                for staged_path in staged_paths:
                    staged_path.unlink(missing_ok=True)
            result = {
                "input_name": ", ".join(input_names),
                "checkpoint": checkpoint.name,
                "input_rgb": input_rgb,
                "output_rgb": output_rgb,
                "output_path": str(output_path),
                "width": raster_info["width"],
                "height": raster_info["height"],
                "crs": output_crs,
                "resolution": raster_info["resolution"],
                "dtype": output_dtype,
                "stdout": completed.stdout,
            }
            previous_run_directory = st.session_state.get("result_tempdir")
            if previous_run_directory is not None:
                previous_run_directory.cleanup()
            st.session_state["last_result"] = result
            st.session_state["result_tempdir"] = run_directory
            run_directory = None
        except subprocess.CalledProcessError as exc:
            run_directory.cleanup()
            error_text = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()
            st.error("Inference failed. Review the message below and check the TIFFs/settings.")
            st.code("\n".join(error_text[-14:]))
        except Exception as exc:  # Show user-actionable validation and I/O errors in the UI.
            run_directory.cleanup()
            st.error(f"Could not run reconstruction: {exc}")

result = st.session_state.get("last_result")
if result:
    st.subheader("Reconstruction preview")
    st.caption(
        f"False color (NIR, Red, Green) · {result['width']:,} × {result['height']:,} px · "
        f"{result['crs']} · {abs(result['resolution'][0]):g} × "
        f"{abs(result['resolution'][1]):g} map units/px · checkpoint `{result['checkpoint']}`"
    )
    preview_columns = st.columns(2)
    preview_columns[0].image(result["input_rgb"], caption="Cloudy input", width="stretch")
    preview_columns[1].image(
        result["output_rgb"], caption="Model reconstruction", width="stretch"
    )
    st.download_button(
        "Download reconstructed GeoTIFF",
        data=lambda: read_download(result["output_path"]),
        file_name="clearsky_reconstruction.tif",
        mime="image/tiff",
        type="primary",
    )
    with st.expander("Run details"):
        st.write(f"Input: {result['input_name']}")
        st.write(f"Checkpoint: {result['checkpoint']}")
        st.write(f"Output pixel type: {result['dtype']}")
        st.code(result["stdout"].strip())
