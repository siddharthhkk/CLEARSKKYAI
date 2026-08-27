# 🛰️ ClearSky-AI: SAR-Optical Data Fusion for Cloud Removal

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)

**ClearSky-AI** is a deep learning pipeline designed to reconstruct cloud-free optical satellite imagery by fusing cloudy Sentinel-2 (Optical) data with cloud-penetrating Sentinel-1 Synthetic Aperture Radar (SAR) data. 

This project was developed as an academic minor project, focusing on Generative Computer Vision and Earth Observation (EO).

---

## 🧠 Architecture & Methodology

Optical satellite imagery is frequently corrupted by cloud cover. ClearSky-AI solves this by leveraging SAR signals (VV/VH polarizations) which penetrate clouds unhindered. The architecture consists of:

1. **Dual-Encoder U-Net**: 
   - **Encoder A (SAR)**: Extracts structural and textural features from radar.
   - **Encoder B (Optical)**: Extracts color and spectral information from cloudy optical bands.
2. **Spatial-Channel Attention (SCA) Bottleneck**: Dynamically weighs the importance of radar structural cues versus non-clouded optical features before fusion.
3. **PatchGAN Discriminator**: A $70 \times 70$ local patch discriminator that enforces high-frequency visual sharpness and realistic surface textures via an Adversarial Loss ($L_{GAN}$), combined with an $L_1$ Reconstruction Loss.

---

## 💾 The 90 GB Dataset & `demo_samples` Strategy

This model is trained on the **SEN12MS-CR-TS** benchmark dataset, which consists of over **90 GB** of multi-temporal Sentinel-1 and Sentinel-2 imagery. 

Handling a 90 GB dataset is highly impractical for GitHub storage, peer review, and local evaluation. To solve this, the repository implements a dual-mode data strategy:

* **The `data/` Directory (Ignored via Git):** Used exclusively for cloud-based model training. Scripts (`download_sample.py` and `extract_sample.py`) handle the FTP download and extraction with auto-cleanup to prevent disk overflow.
* **The `demo_samples/` Directory (Tracked in Git):** A custom sampling script (`create_demo_subset.py`) extracted a highly diverse, **~150 MB** subset of co-registered SAR/Optical tile pairs. 

**Smart Fallback:** The Streamlit application dynamically probes your local file system. If it cannot find the 90 GB `data/` directory, it automatically falls back to `demo_samples/`. This allows evaluators to clone the repository and run live inferences instantly without downloading massive archives.

---

## 🚀 Quick Start: Run the Local Demo (Evaluation Mode)

You can evaluate the ClearSky-AI inference UI on your local machine using the pre-packaged demo samples.

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/ClearSky-AI.git](https://github.com/yourusername/ClearSky-AI.git)
cd ClearSky-AI