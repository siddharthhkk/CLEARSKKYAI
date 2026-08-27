# app/main.py
import os
import sys
import glob
import torch
import rasterio
import numpy as np
import streamlit as st

# ==========================================
# UNIVERSAL PATH RESOLUTION
# Ensures the app runs on ANY machine, regardless
# of where the user runs the 'streamlit run' command from.
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# Dynamically add the 'src' folder to Python's path so it can import models
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
from models import ClearSkyUNet

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
    
    # Safely target the weights folder
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

# Sidebar - Sample Selection
st.sidebar.header("🕹️ Control Panel")

# Catch both lowercase and uppercase .tif extensions across OS environments (Windows/Mac/Linux)
sar_files = glob.glob(os.path.join(DATA_PATH, "**/s1_*.tif"), recursive=True) + \
            glob.glob(os.path.join(DATA_PATH, "**/s1_*.TIF"), recursive=True)

if not sar_files:
    st.error(f"No SAR sample tiles found in `{DATA_PATH}`. Ensure demo samples are tracked in Git.")
    st.stop()

# Show just the filename in the dropdown, not the ugly absolute path
selected_sar = st.sidebar.selectbox(
    "Select Satellite Scene Tile", 
    sar_files, 
    format_func=lambda x: os.path.basename(x)
)

# Resolve the optical partner pair
selected_opt = selected_sar.replace("/S1/", "/S2/").replace("\\S1\\", "\\S2\\").replace("s1_", "s2_")
if not os.path.exists(selected_opt) and selected_opt.endswith('.tif'):
    selected_opt = selected_opt.replace('.tif', '.TIF')

# Load Model
model, device = load_model()

if st.button("✨ Run Cloud Removal Reconstruction", type="primary"):
    with st.spinner("Processing multi-modal feature fusion..."):
        try:
            # Read TIF files
            with rasterio.open(selected_sar) as src:
                sar_raw = src.read().astype(np.float32)
            with rasterio.open(selected_opt) as src:
                opt_raw = src.read().astype(np.float32)
                
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
                st.image(sar_disp, caption="Cloud-Penetrating Structural Radar", use_column_width=True)
                
            with col2:
                st.subheader("2. Sentinel-2 (Cloudy)")
                opt_rgb = normalize_for_display(opt_raw[:3].transpose(1, 2, 0))
                st.image(opt_rgb, caption="Corrupted Optical Imagery", use_column_width=True)
                
            with col3:
                st.subheader("3. ClearSky Output")
                rec_rgb = normalize_for_display(rec_img[:3].transpose(1, 2, 0))
                st.image(rec_rgb, caption="Reconstructed Target", use_column_width=True)

            st.divider()
            st.success("🎉 Feature fusion completed successfully!")
            
        except Exception as e:
            st.error(f"Error during reconstruction: {str(e)}")