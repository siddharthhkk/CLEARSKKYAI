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

# Dynamically add the 'src' folder to Python's path
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from models import ClearSkyUNet  # type: ignore
from preprocess import normalize_sar, normalize_optical  # type: ignore

st.set_page_config(page_title="ClearSky-AI | Cloud Removal Demo", layout="wide")

# Determine Dataset Root dynamically
FULL_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEMO_DATA_DIR = os.path.join(PROJECT_ROOT, "demo_samples")

if os.path.exists(FULL_DATA_DIR) and len(glob.glob(f"{FULL_DATA_DIR}/**/*.tif", recursive=True)) > 0:
    DATA_PATH = FULL_DATA_DIR
    MODE_TEXT = "Full Dataset Mode (data/)"
else:
    DATA_PATH = DEMO_DATA_DIR
    MODE_TEXT = "Offline Demo Mode (demo_samples/)"

st.title("🛰️ ClearSky-AI: SAR-Optical Data Fusion")
st.caption(f"Multi-Modal Cloud Removal Pipeline — Active Mode: {MODE_TEXT}")


@st.cache_resource
def load_model():
    """Load the trained generator."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClearSkyUNet().to(device)

    weights_path = os.path.join(PROJECT_ROOT, "weights", "best_model.pth")

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        st.sidebar.success("Loaded trained weights (best_model.pth)")
    else:
        st.sidebar.warning("Weights not found. Running with initialized architecture.")

    model.eval()
    return model, device


def scale_tensor_for_model(arr: np.ndarray, modality: str) -> torch.Tensor:
    """
    Apply the same fixed-range preprocessing used during training.

    SAR:
      VV -> clip [-25, 0] dB
      VH -> clip [-32.5, 0] dB

    Optical:
      B04, B03, B02, B08 -> clip [0, 10000]
    """
    if modality == "sar":
        return normalize_sar(arr)

    if modality == "optical":
        return normalize_optical(arr)

    raise ValueError(f"Unknown modality: {modality}")


def make_synthetic_cloud(opt_tensor: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Create the same synthetic-cloud corruption used by training.

    A low-resolution random field is upsampled, thresholded at 0.65,
    and clouded pixels are replaced by the normalized cloud value +1.
    """
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

    return opt_tensor * (1.0 - cloud_mask) + cloud_mask * 1.0


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """PSNR for tensors normalized to [-1, 1]."""
    mse = torch.mean((img1.float() - img2.float()) ** 2).item()
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(2.0 / math.sqrt(mse))


def normalize_for_display(img_array: np.ndarray) -> np.ndarray:
    """
    Visualization-only scaling for a single image.
    """
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


def normalize_rgb_pair_for_display(
    img1: np.ndarray,
    img2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply one shared RGB display scale to two channel-first RGB images.

    This keeps output and ground-truth brightness/color relationships
    comparable instead of contrast-stretching them independently.
    """
    if img1.shape != img2.shape or img1.ndim != 3 or img1.shape[0] != 3:
        raise ValueError(
            f"Expected two RGB arrays with shape [3,H,W], got {img1.shape} and {img2.shape}"
        )

    a = img1.astype(np.float32)
    b = img2.astype(np.float32)

    a_out = np.empty_like(a)
    b_out = np.empty_like(b)

    for ch in range(3):
        vals = np.concatenate((a[ch].ravel(), b[ch].ravel()))
        p2, p98 = np.percentile(vals, (2, 98))

        if p98 > p2:
            a_out[ch] = np.clip((a[ch] - p2) / (p98 - p2), 0.0, 1.0)
            b_out[ch] = np.clip((b[ch] - p2) / (p98 - p2), 0.0, 1.0)
        else:
            a_out[ch] = np.zeros_like(a[ch])
            b_out[ch] = np.zeros_like(b[ch])

    return np.power(a_out, 0.85), np.power(b_out, 0.85)


def make_difference_map(
    rec_rgb: np.ndarray,
    target_rgb: np.ndarray,
) -> np.ndarray:
    """
    Create an RGB mean-absolute-error map for visualization.
    Bright pixels indicate larger reconstruction differences.
    """
    diff = np.mean(np.abs(rec_rgb.astype(np.float32) - target_rgb.astype(np.float32)), axis=0)

    p98 = np.percentile(diff, 98)
    if p98 > 0:
        return np.clip(diff / p98, 0.0, 1.0)

    return np.zeros_like(diff)


@st.cache_data
def get_valid_dataset_pairs(data_path):
    """Index S1/S2 files in O(N) time using filename keys."""
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


# ==========================================
# CONTROL PANEL
# ==========================================
st.sidebar.header("🕹️ Control Panel")

model, device = load_model()
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

if st.sidebar.button("✨ Run Cloud Removal Reconstruction", type="primary"):
    with st.spinner("Generating synthetic cloud + running SAR-Optical reconstruction..."):
        try:
            # Load SAR and the exact four Sentinel-2 model bands.
            with rasterio.open(selected_sar) as src_sar, rasterio.open(selected_opt) as src_opt:
                sar_raw = src_sar.read().astype(np.float32)

                # [B04, B03, B02, B08] = [R, G, B, NIR]
                opt_raw = src_opt.read([4, 3, 2, 8]).astype(np.float32)

            # Exact same preprocessing as training.
            sar_tensor = scale_tensor_for_model(sar_raw, "sar")
            opt_target = scale_tensor_for_model(opt_raw, "optical")

            # Use a stable seed derived from the selected tile so the demo
            # shows the same synthetic cloud pattern when rerun.
            demo_seed = sum(ord(ch) for ch in os.path.basename(selected_sar)) % (2**31 - 1)

            opt_cloudy = make_synthetic_cloud(opt_target, demo_seed)

            sar_in = sar_tensor.unsqueeze(0).to(device)
            opt_in = opt_cloudy.unsqueeze(0).to(device)
            target_in = opt_target.unsqueeze(0).to(device)

            with torch.no_grad():
                reconstructed_tensor = model(sar_in, opt_in)

            # Map model output and ground truth from [-1, 1] to [0, 1].
            rec_img = (
                torch.clamp(
                    (reconstructed_tensor.squeeze(0) + 1.0) / 2.0,
                    0.0,
                    1.0,
                )
                .cpu()
                .numpy()
            )

            cloudy_img = (
                torch.clamp((opt_cloudy + 1.0) / 2.0, 0.0, 1.0)
                .cpu()
                .numpy()
            )

            target_img = (
                torch.clamp((opt_target + 1.0) / 2.0, 0.0, 1.0)
                .cpu()
                .numpy()
            )

            # Model band order is [R, G, B, NIR].
            rec_rgb_display, target_rgb_display = normalize_rgb_pair_for_display(
                rec_img[:3],
                target_img[:3],
            )

            # Cloudy image gets its own display transform because it is a
            # visibly corrupted input rather than an evaluation counterpart.
            cloudy_rgb = normalize_for_display(cloudy_img[:3]).transpose(1, 2, 0)

            rec_rgb = rec_rgb_display.transpose(1, 2, 0)
            target_rgb = target_rgb_display.transpose(1, 2, 0)

            # Difference visualization uses the same underlying model-space
            # RGB values, before display normalization.
            diff_map = make_difference_map(rec_img[:3], target_img[:3])

            sar_disp = normalize_for_display(sar_raw[0])

            output_psnr = psnr(
                reconstructed_tensor.squeeze(0).cpu(),
                opt_target.cpu(),
            )

            rgb_mae = float(
                np.mean(np.abs(rec_img[:3].astype(np.float32) - target_img[:3].astype(np.float32)))
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.subheader("1. SAR (Radar)")
                st.image(
                    sar_disp,
                    caption="Cloud-Penetrating Structural Radar",
                    use_container_width=True,
                )

            with col2:
                st.subheader("2. Synthetic Cloudy S2")
                st.image(
                    cloudy_rgb,
                    caption="Training-style cloud corruption",
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

            d1, d2, d3 = st.columns(3)

            with d1:
                st.subheader("🔎 Reconstruction Difference")
                st.image(
                    diff_map,
                    caption="Mean absolute RGB difference (brighter = larger error)",
                    use_container_width=True,
                )

            with d2:
                st.metric(
                    "Demo PSNR",
                    "∞ dB" if not math.isfinite(output_psnr) else f"{output_psnr:.2f} dB",
                )

            with d3:
                st.metric("RGB MAE", f"{rgb_mae:.4f}")

            st.success("✅ Feature fusion and reconstruction completed!")

            st.caption(
                "Comparison note: ClearSky Output and Ground Truth use one shared "
                "RGB display scale so their colors and brightness can be compared "
                "directly. The difference map shows where the reconstruction differs "
                "most from the clean target."
            )

            st.caption(
                "Demo note: the cloud is synthetically generated using the same "
                "cloud-corruption procedure used during training. The ground-truth "
                "image is available because the demo samples originate from paired "
                "clean Sentinel-2 tiles."
            )

            del sar_in, opt_in, target_in, reconstructed_tensor
            del sar_raw, opt_raw, sar_tensor, opt_cloudy, opt_target

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            st.error(f"Error during reconstruction: {str(e)}")
