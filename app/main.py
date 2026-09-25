import os
import sys
import glob
import torch
import rasterio
import numpy as np
import streamlit as st

# ==========================================
# UNIVERSAL PATH RESOLUTION
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# Dynamically add the 'src' folder to Python's path
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
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
    """Loads trained weights or initializes model architecture."""
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


def normalize_for_display(img_array: np.ndarray) -> np.ndarray:
    """
    Applies joint multi-channel percentile scaling (2%-98%) and gamma
    correction (0.85) for visualization only.

    This display transform is intentionally separate from the model
    preprocessing transform.
    """
    img_array = img_array.astype(np.float32)

    # 1. Single Channel (SAR)
    if img_array.ndim == 2:
        p2, p98 = np.percentile(img_array, (2, 98))
        if p98 > p2:
            return np.clip((img_array - p2) / (p98 - p2), 0.0, 1.0)
        return np.zeros_like(img_array)

    # 2. Multi-Channel RGB (Channels-First: 3, H, W)
    if img_array.ndim == 3:
        p2, p98 = np.percentile(img_array, (2, 98))
        if p98 > p2:
            norm = np.clip((img_array - p2) / (p98 - p2), 0.0, 1.0)
        else:
            norm = np.zeros_like(img_array)

        return np.power(norm, 0.85)

    return img_array


@st.cache_data
def get_valid_dataset_pairs(data_path):
    """Indexes dataset in O(N) time using a hash map for instant lookup."""
    all_tifs = (
        glob.glob(os.path.join(data_path, "**/*.tif"), recursive=True)
        + glob.glob(os.path.join(data_path, "**/*.TIF"), recursive=True)
    )

    sar_paths = []
    opt_dict = {}

    for path in all_tifs:
        filename = os.path.basename(path)

        # Universal matching key: s1_<key>.tif <-> s2_<key>.tif
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


# Sidebar - Control Panel
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
    with st.spinner("Processing multi-modal feature fusion..."):
        try:
            # Load SAR and the exact four Sentinel-2 model bands.
            with rasterio.open(selected_sar) as src_sar, rasterio.open(selected_opt) as src_opt:
                sar_raw = src_sar.read().astype(np.float32)

                # [B04, B03, B02, B08] = [R, G, B, NIR]
                opt_raw = src_opt.read([4, 3, 2, 8]).astype(np.float32)

            # Exact same preprocessing as training.
            sar_tensor = scale_tensor_for_model(sar_raw, "sar").unsqueeze(0).to(device)
            opt_tensor = scale_tensor_for_model(opt_raw, "optical").unsqueeze(0).to(device)

            with torch.no_grad():
                reconstructed_tensor = model(sar_tensor, opt_tensor)

            # Map model output from [-1, 1] to [0, 1] for display.
            rec_img = (
                torch.clamp(
                    (reconstructed_tensor.squeeze(0) + 1.0) / 2.0,
                    0.0,
                    1.0,
                )
                .cpu()
                .numpy()
            )

            # Model band order is [R, G, B, NIR].
            opt_rgb_raw = opt_raw[:3]
            rec_rgb_model = rec_img[:3]

            # Display normalization is visualization-only.
            sar_disp = normalize_for_display(sar_raw[0])
            opt_rgb = normalize_for_display(opt_rgb_raw).transpose(1, 2, 0)
            rec_rgb = normalize_for_display(rec_rgb_model).transpose(1, 2, 0)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("1. SAR (Radar)")
                st.image(
                    sar_disp,
                    caption="Cloud-Penetrating Structural Radar",
                    use_container_width=True,
                )

            with col2:
                st.subheader("2. Sentinel-2 (Cloudy)")
                st.image(
                    opt_rgb,
                    caption="Corrupted Optical Imagery",
                    channels="RGB",
                    use_container_width=True,
                )

            with col3:
                st.subheader("3. ClearSky Output")
                st.image(
                    rec_rgb,
                    caption="Reconstructed Target",
                    channels="RGB",
                    use_container_width=True,
                )

            st.divider()
            st.success("Feature fusion completed successfully!")

            del sar_tensor, opt_tensor, reconstructed_tensor, sar_raw, opt_raw

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            st.error(f"Error during reconstruction: {str(e)}")
