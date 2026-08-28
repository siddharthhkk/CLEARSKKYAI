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

# Resolve the optical partner pair dynamically
selected_opt = selected_sar.replace("s1_", "s2_").replace("S1_", "S2_")

# Swap directory names to locate the corresponding optical folder (handling case variations)
for old_dir, new_dir in [
    ("\\sar\\", "\\optical\\"), ("/sar/", "/optical/"),
    ("\\sar\\", "\\s2\\"), ("/sar/", "/s2/"),
    ("\\S1\\", "\\S2\\"), ("/S1/", "/S2/"),
    ("\\S1\\", "\\s2\\"), ("/S1/", "/s2/"),
    ("\\s1\\", "\\s2\\"), ("/s1/", "/s2/"),
    ("\\s1_\\", "\\s2_\\"), ("/s1_/", "/s2_/"),
    ("\\S1_\\", "\\S2_\\"), ("/S1_/", "/S2_/"),
]:
    if old_dir in selected_opt:
        selected_opt = selected_opt.replace(old_dir, new_dir)

# Robust Fallback Search: If direct path doesn't exist, search dynamically in DATA_PATH
if not os.path.exists(selected_opt):
    # Try alternate extension case (.tif <-> .TIF)
    alt_ext = (
        selected_opt.replace(".tif", ".TIF")
        if selected_opt.endswith(".tif")
        else selected_opt.replace(".TIF", ".tif")
    )

    if os.path.exists(alt_ext):
        selected_opt = alt_ext
    else:
        # Search anywhere in the dataset directory for the matching target optical filename
        opt_filename = os.path.basename(selected_opt)
        search_matches = glob.glob(
            os.path.join(DATA_PATH, f"**/{opt_filename}"), recursive=True
        )
        if not search_matches:
            alt_filename = os.path.basename(alt_ext)
            search_matches = glob.glob(
                os.path.join(DATA_PATH, f"**/{alt_filename}"), recursive=True
            )

        if search_matches:
            selected_opt = search_matches[0]
        else:
            st.error(
                f"Could not find matching optical file for `{os.path.basename(selected_sar)}`"
            )
            st.stop()

# Handle upper/lower case extension mismatches (.tif vs .TIF)
if not os.path.exists(selected_opt):
    if selected_opt.endswith('.tif') and os.path.exists(selected_opt.replace('.tif', '.TIF')):
        selected_opt = selected_opt.replace('.tif', '.TIF')
    elif selected_opt.endswith('.TIF') and os.path.exists(selected_opt.replace('.TIF', '.tif')):
        selected_opt = selected_opt.replace('.TIF', '.tif')
    
# Load Model
model, device = load_model()

if st.button("✨ Run Cloud Removal Reconstruction", type="primary"):
    with st.spinner("Processing multi-modal feature fusion..."):
        try:
            # Read TIF files
            with rasterio.open(selected_sar) as src:
                sar_raw = src.read().astype(np.float32)
                
            with rasterio.open(selected_opt) as src:
                # FIX: Slice the first 4 bands (RGB + NIR) from the 13-band Sentinel-2 image
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