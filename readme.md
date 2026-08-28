# 🛰️ ClearSky-AI: SAR-Optical Data Fusion for Cloud Removal

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)

**ClearSky-AI** is a deep learning pipeline designed to reconstruct cloud-free optical satellite imagery by fusing cloudy Sentinel-2 (Optical) data with cloud-penetrating Sentinel-1 Synthetic Aperture Radar (SAR) data[cite: 2]. This project was developed as an academic minor project, focusing on Generative Computer Vision and Earth Observation (EO)[cite: 2].

---

## 🧠 Architecture & Model Performance

Optical satellite imagery is frequently corrupted by cloud cover[cite: 2]. ClearSky-AI solves this by leveraging SAR signals (VV/VH polarizations) which penetrate clouds unhindered[cite: 2]. The architecture consists of:

*   **Encoder A (SAR)**: Extracts structural and textural features from radar[cite: 2].
*   **Encoder B (Optical)**: Extracts color and spectral information from cloudy optical bands[cite: 2].
*   **Spatial-Channel Attention (SCA) Bottleneck**: Dynamically weighs the importance of radar structural cues versus non-clouded optical features before fusion[cite: 2].
*   **PatchGAN Discriminator**: A $70 \times 70$ local patch discriminator that enforces high-frequency visual sharpness and realistic surface textures via an Adversarial Loss ($L_{GAN}$), combined with an $L_1$ Reconstruction Loss[cite: 2].

**Training Milestone:** The U-Net model completed a highly stable 25-epoch training run on a Lightning.ai cloud environment, achieving a peak **Peak Signal-to-Noise Ratio (PSNR) of 31.50 dB** at Epoch 10, demonstrating highly accurate structural-to-spectral feature alignment.

---

## 🚀 Optimization & System Engineering

Deploying the complex dual-stream pipeline to a responsive user dashboard required critical system and visualization fixes:

*   **$O(N)$ Hash Map Data Caching:** Bypassed a severe Streamlit UI rendering bottleneck by replacing an $O(N^2)$ recursive directory search with an in-memory hash map and `@st.cache_data`. This dropped file-pairing times for the 20,000+ files from several minutes to milliseconds.
*   **Sentinel-2 RGB True-Color Mapping:** Fixed a persistent "blue tint" rendering bug by remapping the multi-spectral array slicing from `[:3]` (Coastal Aerosol, Blue, Green) to `[[3, 2, 1]]` (Red, Green, Blue), ensuring natural true-color terrain display.
*   **Dynamic Path Resolution:** Developed an OS-agnostic regex string replacement protocol to seamlessly map disparate `\sar\` folders to `\s2\` directories across different cloud and local environments.

---

## 💾 The 90 GB Dataset & `demo_samples` Strategy

This model is trained on the **SEN12MS-CR-TS** benchmark dataset, which consists of over **90 GB** of multi-temporal Sentinel-1 and Sentinel-2 imagery[cite: 2]. 

Handling a 90 GB dataset is highly impractical for GitHub storage, peer review, and local evaluation[cite: 2]. To solve this, the repository implements a dual-mode data strategy[cite: 2]:

*   **The `data/` Directory (Ignored via Git):** Used exclusively for cloud-based model training[cite: 2]. Scripts (`download_sample.py` and `extract_sample.py`) handle the FTP download and extraction with auto-cleanup to prevent disk overflow[cite: 2].
*   **The `demo_samples/` Directory (Tracked in Git):** A custom sampling script (`create_demo_subset.py`) extracted a highly diverse, **~150 MB** subset of co-registered SAR/Optical tile pairs[cite: 2]. 

**Smart Fallback:** The Streamlit application dynamically probes your local file system[cite: 2]. If it cannot find the 90 GB `data/` directory, it automatically falls back to `demo_samples/`[cite: 2]. This allows evaluators to clone the repository and run live inferences instantly without downloading massive archives[cite: 2].

---

## ⚙️ Requirements & Environment Setup

To ensure the Streamlit dashboard and PyTorch inference run smoothly, please verify your local environment meets the following specifications.

### Hardware Requirements
*   **For Local Demo (Inference):** A standard multi-core CPU is sufficient. Minimum **8 GB RAM** is recommended to handle the in-memory hash maps and multi-spectral array processing.
*   **For Model Training:** An NVIDIA GPU with CUDA support (minimum 16 GB VRAM) is strictly required to handle the 91 GB dataset and U-Net backpropagation. (Note: The official training run was executed on Lightning.ai cloud instances).

### Software Dependencies
The project is built and tested on **Python 3.10+**. The core dependencies include:

*   **PyTorch (2.0+):** For loading the `.pth` weights and executing the dual-stream U-Net inference.
*   **Streamlit (1.30+):** For rendering the interactive web dashboard and caching the dataset.
*   **Rasterio & NumPy:** For reading multi-band `.tif` satellite files and processing $N$-dimensional arrays.
*   **Torchvision & Pillow:** For image tensor normalization and display rendering.

### Environment Installation
It is highly recommended to use a virtual environment to avoid version conflicts:

```bash
# Create and activate a virtual environment
python -m venv clearsky_env
source clearsky_env/bin/activate  # On Windows use: clearsky_env\Scripts\activate

# Install the required packages
pip install -r requirements.txt

---

## 💻 Quick Start: Run the Local Demo (Evaluation Mode)

You can evaluate the ClearSky-AI inference UI on your local machine using the pre-packaged demo samples[cite: 2].

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/ClearSky-AI.git](https://github.com/yourusername/ClearSky-AI.git)
    cd ClearSky-AI
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch the Dashboard:**
    ```bash
    streamlit run app/main.py
    ```