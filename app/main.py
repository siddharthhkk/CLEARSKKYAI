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
from models import ClearSkyUNet    # type: ignore

st.set_page_config(page_title="ClearSky-AI | Cloud Removal Demo", layout="wide")

# Determine Dataset Root dynamically
FULL_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEMO_DATA_DIR = os.path.join(PROJECT_ROOT, "demo_samples")

if os.path.exists(FULL_DATA_DIR) and len(glob.glob(f"{FULL_DATA_DIR}/**/*.tif", recursive=True)) > 0:
    DATA_PATH = FULL_DATA_DIR
    MODE_TEXT = "Full Dataset Mode (`data/`)"
else:
    DATA_PATH = DEMO_DATA_DIR
    MODE_TEXT = "Offline Demo Mode (`demo_samples/`)"

st.title("🛰️ ClearSky-AI: SAR-Optical Data Fusion")
st.caption(f"Multi-Modal Cloud Removal Pipeline — **Active Mode:** {MODE_TEXT}")

@st.cache_resource
def load_model():
    """Loads trained weights or initializes model architecture."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClearSkyUNet().to(device)
    
    weights_path = os.path.join(PROJECT_ROOT, "weights", "best_model.pth")
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        st.sidebar.success("✅ Loaded trained weights (`best_model.pth`)")
    else:
        st.sidebar.warning("⚠️ Weights not found. Running with initialized architecture.")
    
    model.eval()
    return model, device

def scale_tensor_for_model(arr: np.ndarray) -> torch.Tensor:
    """
    Min-Max scales raw physical reflectance arrays to [-1, 1] for PyTorch.
    Maintains strict scientific fidelity for the neural network.
    """
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        norm = (arr - arr_min) / (arr_max - arr_min)
    else:
        norm = np.zeros_like(arr, dtype=np.float32)
        
    return torch.from_numpy(norm * 2.0 - 1.0).to(torch.float32)

def normalize_for_display(img_array: np.ndarray) -> np.ndarray:
    """
    Applies joint multi-channel percentile scaling (2%-98%) and gamma correction (0.85).
    Preserves true color ratios without blowing out aquatic or cloudy tiles.
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

def get_rgb_indices(num_bands: int) -> list:
    """
    Dynamically determines the correct Red, Green, Blue band indices.
    Prevents the 'Blue Tint' (Coastal Aerosol) and 'Neon' channel swapping artifacts.
    """
    if num_bands == 4:
        # Standard SEN12MS 4-band: [Blue(2), Green(3), Red(4), NIR(8)]
        # True Color RGB Mapping: Red=2, Green=1, Blue=0
        return [2, 1, 0]
    elif num_bands >= 5:
        # 13-band Sentinel-2: [Coastal(1), Blue(2), Green(3), Red(4), ...]
        # True Color RGB Mapping: Red=3, Green=2, Blue=1
        return [3, 2, 1]
    
    # Fallback to direct indexing if pre-processed exactly as RGB
    return [0, 1, 2]

@st.cache_data
def get_valid_dataset_pairs(data_path):
    """Indexes dataset in O(N) time using a hash map for instant lookup."""
    # 1. Single pass to grab ALL tif files at once (Fast)
    all_tifs = glob.glob(os.path.join(data_path, "**/*.tif"), recursive=True) + \
               glob.glob(os.path.join(data_path, "**/*.TIF"), recursive=True)
    
    sar_paths = []
    opt_dict = {}
    
    # 2. Separate and hash the files
    for path in all_tifs:
        filename = os.path.basename(path)
        
        # Create a universal matching key by stripping prefixes and extensions
        # Example: 's1_ROIs2017_22...tif' -> 'rois2017_22...'
        base_key = filename.lower().replace("s1_", "").replace("s2_", "").replace(".tif", "")
        
        if filename.lower().startswith("s1_"):
            sar_paths.append((base_key, path))
        elif filename.lower().startswith("s2_"):
            opt_dict[base_key] = path
            
    # 3. Instantly pair them using O(1) dictionary lookups
    valid_pairs = {}
    for base_key, sar_path in sar_paths:
        if base_key in opt_dict:
            valid_pairs[sar_path] = opt_dict[base_key]
            
    return valid_pairs

# Sidebar - Control Panel
st.sidebar.header("🕹️ Control Panel")
model, device = load_model()

# Load only valid pairs into the dropdown menu
valid_pairs = get_valid_dataset_pairs(DATA_PATH)

if not valid_pairs:
    st.error(f"No valid SAR-Optical image pairs found in `{DATA_PATH}`.")
    st.stop()

sar_file_list = list(valid_pairs.keys())
selected_sar = st.sidebar.selectbox(
    "Select Satellite Scene Tile", 
    sar_file_list, 
    format_func=lambda x: os.path.basename(x)
)

selected_opt = valid_pairs[selected_sar]

if st.sidebar.button("✨ Run Cloud Removal Reconstruction", type="primary"):
    with st.spinner("Processing multi-modal feature fusion..."):
        try:
            # 1. Context Managers for safe file handling (prevents memory leaks)
            with rasterio.open(selected_sar) as src_sar, rasterio.open(selected_opt) as src_opt:
                sar_raw = src_sar.read().astype(np.float32)
                # Slice first 4 bands for model input consistency
                opt_raw = src_opt.read()[:4, :, :].astype(np.float32)
                
            # 2. Tensor Preparation
            sar_tensor = scale_tensor_for_model(sar_raw).unsqueeze(0).to(device)
            opt_tensor = scale_tensor_for_model(opt_raw).unsqueeze(0).to(device)
            
            # 3. Model Inference (Strictly no_grad to save VRAM)
            with torch.no_grad():
                reconstructed_tensor = model(sar_tensor, opt_tensor)
                
            # 4. Map Model Output back to [0, 1]
            rec_img = torch.clamp((reconstructed_tensor.squeeze(0) + 1.0) / 2.0, 0.0, 1.0).cpu().numpy()
            
            # 5. Extract RGB channels safely based on dataset shape
            rgb_idx = get_rgb_indices(opt_raw.shape[0])
            opt_rgb_raw = opt_raw[rgb_idx]
            rec_rgb_raw = rec_img[rgb_idx]

            # 6. Apply Display Normalization and Transpose to Channels-Last (H, W, C)
            sar_disp = normalize_for_display(sar_raw[0])
            opt_rgb = normalize_for_display(opt_rgb_raw).transpose(1, 2, 0)
            rec_rgb = normalize_for_display(rec_rgb_raw).transpose(1, 2, 0)

            # 7. Render UI Columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("1. SAR (Radar)")
                st.image(sar_disp, caption="Cloud-Penetrating Structural Radar", use_container_width=True)
                
            with col2:
                st.subheader("2. Sentinel-2 (Cloudy)")
                st.image(opt_rgb, caption="Corrupted Optical Imagery", channels="RGB", use_container_width=True)
                
            with col3:
                st.subheader("3. ClearSky Output")
                st.image(rec_rgb, caption="Reconstructed Target", channels="RGB", use_container_width=True)

            st.divider()
            st.success("🎉 Feature fusion completed successfully!")
            
            # 8. Explicit Memory Cleanup (Critical for Streamlit hot-reloading)
            del sar_tensor, opt_tensor, reconstructed_tensor, sar_raw, opt_raw
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            st.error(f"Error during reconstruction: {str(e)}")