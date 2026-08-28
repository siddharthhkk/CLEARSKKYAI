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

def normalize_for_display(img_array):
    """Converts raw multi-band image array to [0, 1] RGB visualization."""
    img_min, img_max = np.min(img_array), np.max(img_array)
    if img_max > img_min:
        img_norm = (img_array - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img_array)
    return np.clip(img_norm, 0, 1)

def find_optical_pair(sar_path, root_dir):
    """Locates the corresponding optical tile handling casing & path variations."""
    sar_dir = os.path.dirname(sar_path)
    sar_file = os.path.basename(sar_path)
    
    # Target optical filenames
    target_names = [
        sar_file.replace("s1_", "s2_"),
        sar_file.replace("s1_", "S2_"),
        sar_file.replace("S1_", "s2_"),
        sar_file.replace("S1_", "S2_")
    ]
    
    filenames_to_try = []
    for name in target_names:
        filenames_to_try.append(name)
        if name.endswith('.tif'):
            filenames_to_try.append(name[:-4] + '.TIF')
        elif name.endswith('.TIF'):
            filenames_to_try.append(name[:-4] + '.tif')
            
    filenames_to_try = list(set(filenames_to_try))
    
    # 1. Direct path replacement (e.g. replacing /s1/ with /s2/)
    for fname in filenames_to_try:
        sar_dir_opt = sar_dir.replace("/s1/", "/s2/").replace("/S1/", "/S2/").replace("/s1/", "/S2/").replace("/S1/", "/s2/")
        candidate = os.path.join(sar_dir_opt, fname)
        if os.path.exists(candidate):
            return candidate

    # 2. Local search inside parent ROI folder
    parent_dir = os.path.dirname(sar_dir)
    for fname in filenames_to_try:
        matches = glob.glob(os.path.join(parent_dir, "**", fname), recursive=True)
        if matches:
            return matches[0]
            
    # 3. Global search fallback
    for fname in filenames_to_try:
        matches = glob.glob(os.path.join(root_dir, "**", fname), recursive=True)
        if matches:
            return matches[0]
            
    return None

@st.cache_data
def get_valid_dataset_pairs(data_path):
    """Indexes dataset and filters only valid SAR + Optical image pairs."""
    all_sar = sorted(
        glob.glob(os.path.join(data_path, "**/s1_*.tif"), recursive=True) +
        glob.glob(os.path.join(data_path, "**/s1_*.TIF"), recursive=True) +
        glob.glob(os.path.join(data_path, "**/S1_*.tif"), recursive=True) +
        glob.glob(os.path.join(data_path, "**/S1_*.TIF"), recursive=True)
    )
    
    valid_pairs = {}
    for sar in all_sar:
        opt = find_optical_pair(sar, data_path)
        if opt:
            valid_pairs[sar] = opt
            
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
            # Read TIF files
            with rasterio.open(selected_sar) as src:
                sar_raw = src.read().astype(np.float32)
                
            with rasterio.open(selected_opt) as src:
                # Slice first 4 bands (RGB + NIR) to match training shape
                opt_raw = src.read()[:4, :, :].astype(np.float32)
                
            # Normalize to [-1, 1] for Model Input
            sar_tensor = torch.from_numpy(normalize_for_display(sar_raw) * 2 - 1).unsqueeze(0).to(device)
            opt_tensor = torch.from_numpy(normalize_for_display(opt_raw) * 2 - 1).unsqueeze(0).to(device)
            
            # Model Inference
            with torch.no_grad():
                reconstructed_tensor = model(sar_tensor, opt_tensor)
                
            # Convert output back to display format [0, 1]
            rec_img = (reconstructed_tensor.squeeze(0).cpu().numpy() + 1) / 2.0
            
            # Display Results Side-by-Side
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("1. SAR (Radar)")
                sar_disp = normalize_for_display(sar_raw[0])
                st.image(sar_disp, caption="Cloud-Penetrating Structural Radar", use_container_width=True)
                
            with col2:
                st.subheader("2. Sentinel-2 (Cloudy)")
                opt_rgb = normalize_for_display(opt_raw[:3].transpose(1, 2, 0))
                st.image(opt_rgb, caption="Corrupted Optical Imagery", use_container_width=True)
                
            with col3:
                st.subheader("3. ClearSky Output")
                rec_rgb = normalize_for_display(rec_img[:3].transpose(1, 2, 0))
                st.image(rec_rgb, caption="Reconstructed Target", use_container_width=True)

            st.divider()
            st.success("🎉 Feature fusion completed successfully!")
            
        except Exception as e:
            st.error(f"Error during reconstruction: {str(e)}")