import os
import sys
import glob
import math

import torch
import torch.nn.functional as F
import rasterio
import numpy as np
import streamlit as st

# ==========================================
# UNIVERSAL PATH RESOLUTION
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from models import ClearSkyUNet  # type: ignore
from dsen2cr import DSen2CR  # type: ignore
from preprocess import normalize_sar, normalize_optical  # type: ignore

st.set_page_config(page_title="ClearSky-AI | Cloud Removal Demo", layout="wide")

FULL_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEMO_DATA_DIR = os.path.join(PROJECT_ROOT, "demo_samples")
REAL_DIVERSE_DIR = os.path.join(PROJECT_ROOT, "real_cloud_diverse")
REAL_DEMO_DIR = os.path.join(PROJECT_ROOT, "real_cloud_demo")

if os.path.exists(FULL_DATA_DIR) and glob.glob(
    f"{FULL_DATA_DIR}/**/*.tif", recursive=True
):
    DATA_PATH = FULL_DATA_DIR
    MODE_TEXT = "Full Dataset Mode (data/)"
else:
    DATA_PATH = DEMO_DATA_DIR
    MODE_TEXT = "Offline Demo Mode (demo_samples/)"

st.title("🛰️ ClearSky-AI: SAR-Optical Cloud Removal")
st.caption(
    f"Multi-Modal Cloud Removal Pipeline — Active Dataset: {MODE_TEXT}"
)


@st.cache_resource
def load_model(model_name):
    """Load the selected model backend."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "ClearSkyUNet":
        model = ClearSkyUNet().to(device)
        weights_path = os.path.join(PROJECT_ROOT, "weights", "best_model.pth")

        if os.path.exists(weights_path):
            model.load_state_dict(
                torch.load(weights_path, map_location=device)
            )
            st.sidebar.success("Loaded ClearSkyUNet: best_model.pth")
        else:
            st.sidebar.warning("ClearSkyUNet weights not found.")

    elif model_name == "DSen2-CR":
        model = DSen2CR().to(device=device, dtype=torch.float32)
        weights_path = os.path.join(
            PROJECT_ROOT,
            "weights",
            "dsen2cr_sar_carl.pth",
        )

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                "DSen2-CR weights not found. Convert the public checkpoint with "
                "scripts/convert_dsen2cr_weights.py."
            )

        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        model.load_state_dict(state, strict=True)
        model = model.to(device=device, dtype=torch.float32)
        st.sidebar.success("Loaded DSen2-CR pretrained CARL weights")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    return model, device


def preprocess_dsen2_sar(arr):
    """Map Sentinel-1 VV/VH from dB to DSen2-CR's [0,2] convention."""
    x = np.asarray(arr, dtype=np.float32).copy()
    mins = np.array([-25.0, -35.0], dtype=np.float32)

    for i in range(2):
        x[i] = np.clip(x[i], mins[i], 0.0)
        x[i] = 2.0 * (x[i] - mins[i]) / (0.0 - mins[i])

    return torch.from_numpy(x).float()


def preprocess_dsen2_optical(arr):
    """Map Sentinel-2 optical values to DSen2-CR's [0,5] convention."""
    x = np.asarray(arr, dtype=np.float32)
    x = np.clip(x, 0.0, 10000.0)
    return torch.from_numpy(x / 2000.0).float()


def make_synthetic_cloud(
    opt_tensor: torch.Tensor,
    seed: int,
    fill_value: float,
):
    """Create the synthetic cloud corruption used for ClearSkyUNet."""
    _, h, w = opt_tensor.shape
    nh = max(1, h // 16)
    nw = max(1, w // 16)

    g = torch.Generator()
    g.manual_seed(seed)
    noise = torch.rand(1, 1, nh, nw, generator=g)

    cloud_mask = F.interpolate(
        noise,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    cloud_mask = (cloud_mask > 0.65).float()
    cloudy = opt_tensor * (1.0 - cloud_mask) + cloud_mask * fill_value

    return cloudy, cloud_mask


def psnr_unit(img1, img2):
    """PSNR for values normalized to [0,1]."""
    mse = torch.mean((img1.float() - img2.float()) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def psnr_model_space(img1, img2, peak=2.0):
    """PSNR in the supplied model-space range."""
    mse = torch.mean((img1.float() - img2.float()) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(peak / math.sqrt(mse))


def normalize_for_display(img_array):
    """Visualization-only percentile scaling."""
    img_array = img_array.astype(np.float32)

    if img_array.ndim == 2:
        p2, p98 = np.percentile(img_array, (2, 98))
        if p98 > p2:
            return np.clip((img_array - p2) / (p98 - p2), 0.0, 1.0)
        return np.zeros_like(img_array)

    if img_array.ndim == 3:
        p2, p98 = np.percentile(img_array, (2, 98))
        if p98 > p2:
            norm = np.clip((img_array - p2) / (p98 - p2), 0.0, 1.0)
        else:
            norm = np.zeros_like(img_array)
        return np.power(norm, 0.85)

    return img_array


def normalize_rgb_pair_for_display(img1, img2):
    """Use one shared RGB display scale for comparable images."""
    if img1.shape != img2.shape or img1.ndim != 3 or img1.shape[0] != 3:
        raise ValueError(
            f"Expected RGB arrays [3,H,W], got {img1.shape} and {img2.shape}"
        )

    a = img1.astype(np.float32)
    b = img2.astype(np.float32)

    a_out = np.empty_like(a)
    b_out = np.empty_like(b)

    for ch in range(3):
        vals = np.concatenate((a[ch].ravel(), b[ch].ravel()))
        p2, p98 = np.percentile(vals, (2, 98))

        if p98 > p2:
            a_out[ch] = np.clip(
                (a[ch] - p2) / (p98 - p2), 0.0, 1.0
            )
            b_out[ch] = np.clip(
                (b[ch] - p2) / (p98 - p2), 0.0, 1.0
            )
        else:
            a_out[ch] = 0.0
            b_out[ch] = 0.0

    return np.power(a_out, 0.85), np.power(b_out, 0.85)


def make_difference_map(rec_rgb, target_rgb):
    """Mean absolute RGB error visualization."""
    diff = np.mean(
        np.abs(rec_rgb.astype(np.float32) - target_rgb.astype(np.float32)),
        axis=0,
    )

    p98 = np.percentile(diff, 98)
    if p98 > 0:
        return np.clip(diff / p98, 0.0, 1.0)

    return np.zeros_like(diff)


@st.cache_data
def get_valid_dataset_pairs(data_path):
    """Index local S1/S2 pairs."""
    all_tifs = (
        glob.glob(os.path.join(data_path, "**/*.tif"), recursive=True)
        + glob.glob(os.path.join(data_path, "**/*.TIF"), recursive=True)
    )

    sar_paths = []
    opt_dict = {}

    for path in all_tifs:
        filename = os.path.basename(path)

        base_key = (
            filename.lower()
            .replace("s1_", "")
            .replace("s2_", "")
            .replace(".tif", "")
        )

        if filename.lower().startswith("s1_"):
            sar_paths.append((base_key, path))
        elif filename.lower().startswith("s2_"):
            opt_dict[base_key] = path

    valid_pairs = {}

    for base_key, sar_path in sar_paths:
        if base_key in opt_dict:
            valid_pairs[sar_path] = opt_dict[base_key]

    return valid_pairs


@st.cache_data
def get_real_cloud_samples():
    """Find locally downloaded real-cloud SEN12MS-CR samples."""
    for directory in (REAL_DIVERSE_DIR, REAL_DEMO_DIR):
        if os.path.isdir(directory):
            samples = sorted(
                glob.glob(os.path.join(directory, "sample_*.npz"))
            )
            if samples:
                return samples, directory

    return [], None


# ==========================================
# CONTROL PANEL
# ==========================================
st.sidebar.header("🕹️ Control Panel")

input_mode = st.sidebar.radio(
    "Input Mode",
    [
        "Synthetic Cloud / ClearSkyUNet",
        "Real Cloud / DSen2-CR",
    ],
)

if input_mode.startswith("Synthetic"):
    model_name = "ClearSkyUNet"
else:
    model_name = "DSen2-CR"

model, device = load_model(model_name)

# ==========================================
# SYNTHETIC CLOUD MODE
# ==========================================
if input_mode.startswith("Synthetic"):
    st.sidebar.info(
        "Custom baseline: synthetic cloud corruption + ClearSkyUNet."
    )

    valid_pairs = get_valid_dataset_pairs(DATA_PATH)

    if not valid_pairs:
        st.error(f"No valid SAR-Optical image pairs found in {DATA_PATH}.")
        st.stop()

    sar_file_list = list(valid_pairs.keys())

    selected_sar = st.sidebar.selectbox(
        "Select Satellite Scene Tile",
        sar_file_list,
        format_func=lambda x: os.path.basename(x),
    )

    selected_opt = valid_pairs[selected_sar]

    run = st.sidebar.button(
        "✨ Run Synthetic Cloud Reconstruction",
        type="primary",
    )

    if run:
        with st.spinner("Generating synthetic cloud + running ClearSkyUNet..."):
            try:
                with rasterio.open(selected_sar) as src_sar, rasterio.open(
                    selected_opt
                ) as src_opt:
                    sar_raw = src_sar.read().astype(np.float32)
                    opt_raw = src_opt.read([4, 3, 2, 8]).astype(np.float32)

                sar_tensor = normalize_sar(sar_raw)
                opt_target = normalize_optical(opt_raw)

                demo_seed = (
                    sum(ord(ch) for ch in os.path.basename(selected_sar))
                    % (2**31 - 1)
                )

                opt_cloudy, cloud_mask = make_synthetic_cloud(
                    opt_target,
                    demo_seed,
                    fill_value=1.0,
                )

                with torch.no_grad():
                    reconstructed_tensor = model(
                        sar_tensor.unsqueeze(0).to(device),
                        opt_cloudy.unsqueeze(0).to(device),
                    ).squeeze(0).cpu()

                metric_out = reconstructed_tensor
                metric_tgt = opt_target.cpu()

                rec_img = torch.clamp(
                    (metric_out + 1.0) / 2.0,
                    0.0,
                    1.0,
                ).numpy()

                cloudy_img = torch.clamp(
                    (opt_cloudy + 1.0) / 2.0,
                    0.0,
                    1.0,
                ).numpy()

                target_img = torch.clamp(
                    (opt_target + 1.0) / 2.0,
                    0.0,
                    1.0,
                ).numpy()

                rgb_idx = [0, 1, 2]

                rec_rgb_display, target_rgb_display = (
                    normalize_rgb_pair_for_display(
                        rec_img[rgb_idx],
                        target_img[rgb_idx],
                    )
                )

                rec_rgb = rec_rgb_display.transpose(1, 2, 0)
                target_rgb = target_rgb_display.transpose(1, 2, 0)
                cloudy_rgb = normalize_for_display(
                    cloudy_img[rgb_idx]
                ).transpose(1, 2, 0)

                diff_map = make_difference_map(
                    rec_img[rgb_idx],
                    target_img[rgb_idx],
                )

                output_psnr = psnr_model_space(
                    metric_out,
                    metric_tgt,
                    peak=2.0,
                )

                rec_rgb_arr = rec_img[rgb_idx].astype(np.float32)
                target_rgb_arr = target_img[rgb_idx].astype(np.float32)
                rgb_abs_err = np.abs(rec_rgb_arr - target_rgb_arr)

                rgb_mae = float(rgb_abs_err.mean())

                cloud_px = cloud_mask.squeeze(0).numpy().astype(bool)
                visible_px = ~cloud_px

                if np.any(cloud_px):
                    cloud_err = rgb_abs_err[:, cloud_px]
                    cloud_mae = float(cloud_err.mean())
                    cloud_psnr = psnr_unit(
                        torch.from_numpy(rec_rgb_arr[:, cloud_px]),
                        torch.from_numpy(target_rgb_arr[:, cloud_px]),
                    )
                else:
                    cloud_mae = float("nan")
                    cloud_psnr = float("nan")

                visible_mae = (
                    float(rgb_abs_err[:, visible_px].mean())
                    if np.any(visible_px)
                    else float("nan")
                )

                cloud_mask_disp = cloud_px.astype(np.float32)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.subheader("1. SAR (Radar)")
                    st.image(
                        normalize_for_display(sar_raw[0]),
                        caption="Cloud-Penetrating Structural Radar",
                        use_container_width=True,
                    )

                with col2:
                    st.subheader("2. Synthetic Cloudy S2")
                    st.image(
                        cloudy_rgb,
                        caption="Training-style synthetic corruption",
                        channels="RGB",
                        use_container_width=True,
                    )

                with col3:
                    st.subheader("3. ClearSky Output")
                    st.image(
                        rec_rgb,
                        caption="Reconstructed RGB",
                        channels="RGB",
                        use_container_width=True,
                    )

                with col4:
                    st.subheader("4. Ground Truth S2")
                    st.image(
                        target_rgb,
                        caption="Original clean optical target",
                        channels="RGB",
                        use_container_width=True,
                    )

                st.divider()

                d1, d2, d3, d4 = st.columns(4)

                with d1:
                    st.subheader("☁️ Cloud Mask")
                    st.image(
                        cloud_mask_disp,
                        caption="1 = hidden/reconstructed region",
                        use_container_width=True,
                    )

                with d2:
                    st.subheader("🔎 Reconstruction Difference")
                    st.image(
                        diff_map,
                        caption="Brighter = larger RGB error",
                        use_container_width=True,
                    )

                with d3:
                    st.metric(
                        "Overall PSNR",
                        "∞ dB"
                        if not math.isfinite(output_psnr)
                        else f"{output_psnr:.2f} dB",
                    )
                    st.metric("Overall RGB MAE", f"{rgb_mae:.4f}")

                with d4:
                    st.metric(
                        "Cloud PSNR",
                        "∞ dB"
                        if not math.isfinite(cloud_psnr)
                        else f"{cloud_psnr:.2f} dB",
                    )
                    st.metric("Cloud MAE", f"{cloud_mae:.4f}")
                    st.metric("Visible MAE", f"{visible_mae:.4f}")

                st.success("✅ Synthetic-cloud reconstruction completed!")

                st.caption(
                    "ClearSkyUNet is evaluated here on the same synthetic "
                    "cloud corruption design used during its training."
                )

            except Exception as e:
                st.error(f"Error during reconstruction: {str(e)}")

# ==========================================
# REAL CLOUD MODE
# ==========================================
else:
    st.sidebar.info(
        "Published pretrained baseline: real cloudy S2 + SAR → DSen2-CR."
    )

    real_samples, real_dir = get_real_cloud_samples()

    if not real_samples:
        st.warning(
            "No real-cloud samples found. Run "
            "'python scripts/download_real_cloud_diverse.py --samples 20 --seed 42' "
            "first."
        )
        st.stop()

    st.sidebar.success(
        f"Loaded {len(real_samples)} real-cloud samples from "
        f"{os.path.basename(real_dir)}"
    )

    selected_sample = st.sidebar.selectbox(
        "Select Real Cloud Sample",
        real_samples,
        format_func=lambda x: os.path.basename(x),
    )

    run = st.sidebar.button(
        "☁️ Run Real Cloud Removal",
        type="primary",
    )

    if run:
        with st.spinner("Running pretrained DSen2-CR on real cloudy S2 + SAR..."):
            try:
                d = np.load(selected_sample, allow_pickle=True)

                sar_raw = d["sar"].astype(np.float32)
                cloudy_raw = d["cloudy"].astype(np.float32)
                target_raw = d["target"].astype(np.float32)

                if sar_raw.shape != (2, 256, 256):
                    raise ValueError(
                        f"Expected SAR [2,256,256], got {sar_raw.shape}"
                    )
                if cloudy_raw.shape != (13, 256, 256):
                    raise ValueError(
                        f"Expected cloudy S2 [13,256,256], got {cloudy_raw.shape}"
                    )
                if target_raw.shape != (13, 256, 256):
                    raise ValueError(
                        f"Expected target S2 [13,256,256], got {target_raw.shape}"
                    )

                sar_tensor = preprocess_dsen2_sar(sar_raw)
                cloudy_tensor = preprocess_dsen2_optical(cloudy_raw)

                inp = torch.cat(
                    [cloudy_tensor, sar_tensor],
                    dim=0,
                ).unsqueeze(0).to(
                    device=device,
                    dtype=torch.float32,
                )

                with torch.no_grad():
                    output_tensor = model(inp).squeeze(0).cpu()

                # DSen2-CR optical/model-space values are [0,5].
                output_img = torch.clamp(
                    output_tensor / 5.0,
                    0.0,
                    1.0,
                ).numpy()

                cloudy_img = np.clip(
                    cloudy_raw / 10000.0,
                    0.0,
                    1.0,
                )

                target_img = np.clip(
                    target_raw / 10000.0,
                    0.0,
                    1.0,
                )

                # B04,B03,B02 = indices 3,2,1.
                rgb_idx = [3, 2, 1]

                out_rgb = output_img[rgb_idx].astype(np.float32)
                cloudy_rgb_raw = cloudy_img[rgb_idx].astype(np.float32)
                target_rgb = target_img[rgb_idx].astype(np.float32)

                out_display, target_display = normalize_rgb_pair_for_display(
                    out_rgb,
                    target_rgb,
                )

                out_rgb_display = out_display.transpose(1, 2, 0)
                target_rgb_display = target_display.transpose(1, 2, 0)
                cloudy_rgb_display = normalize_for_display(
                    cloudy_rgb_raw
                ).transpose(1, 2, 0)

                diff_map = make_difference_map(
                    out_rgb,
                    target_rgb,
                )

                input_mae = float(
                    np.mean(np.abs(cloudy_rgb_raw - target_rgb))
                )
                output_mae = float(
                    np.mean(np.abs(out_rgb - target_rgb))
                )

                input_psnr = psnr_unit(
                    torch.from_numpy(cloudy_rgb_raw),
                    torch.from_numpy(target_rgb),
                )

                output_psnr = psnr_unit(
                    torch.from_numpy(out_rgb),
                    torch.from_numpy(target_rgb),
                )

                gain = output_psnr - input_psnr

                season = d["season"].item() if d["season"].shape == () else str(d["season"])
                scene = d["scene"].item() if d["scene"].shape == () else str(d["scene"])
                patch = d["patch"].item() if d["patch"].shape == () else str(d["patch"])

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.subheader("1. SAR (Radar)")
                    st.image(
                        normalize_for_display(sar_raw[0]),
                        caption="Sentinel-1 radar input",
                        use_container_width=True,
                    )

                with col2:
                    st.subheader("2. Real Cloudy S2")
                    st.image(
                        cloudy_rgb_display,
                        caption="Genuine cloudy Sentinel-2 observation",
                        channels="RGB",
                        use_container_width=True,
                    )

                with col3:
                    st.subheader("3. DSen2-CR Output")
                    st.image(
                        out_rgb_display,
                        caption="Pretrained SAR-guided reconstruction",
                        channels="RGB",
                        use_container_width=True,
                    )

                with col4:
                    st.subheader("4. Cloud-free S2")
                    st.image(
                        target_rgb_display,
                        caption="Paired cloud-free target",
                        channels="RGB",
                        use_container_width=True,
                    )

                st.divider()

                d1, d2, d3, d4 = st.columns(4)

                with d1:
                    st.subheader("🛰️ Scene")
                    st.write(f"Season: **{season}**")
                    st.write(f"Scene: **{scene}**")
                    st.write(f"Patch: **{patch}**")

                with d2:
                    st.subheader("🔎 Reconstruction Difference")
                    st.image(
                        diff_map,
                        caption="Brighter = larger RGB error",
                        use_container_width=True,
                    )

                with d3:
                    st.metric("Cloudy Input PSNR", f"{input_psnr:.2f} dB")
                    st.metric("Cloudy Input MAE", f"{input_mae:.4f}")

                with d4:
                    st.metric("DSen2-CR PSNR", f"{output_psnr:.2f} dB")
                    st.metric("DSen2-CR MAE", f"{output_mae:.4f}")
                    st.metric("PSNR Gain", f"{gain:+.2f} dB")

                st.success("✅ Real-cloud DSen2-CR reconstruction completed!")

                st.caption(
                    "This mode uses genuine cloudy/cloud-free SEN12MS-CR pairs. "
                    "The current public mirror used by the demo does not expose a "
                    "native pixel cloud mask, so cloud-only metrics are intentionally "
                    "not reported here."
                )

            except Exception as e:
                st.error(f"Error during real-cloud reconstruction: {str(e)}")
