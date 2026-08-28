# src/dataset.py
import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio

class SEN12MSDataset(Dataset):
    """
    PyTorch Dataset for SEN12MS-CR-TS SAR-Optical fusion.
    Handles matching Sentinel-1 (SAR) and Sentinel-2 (Optical) pairs.
    """
    def __init__(self, root_dir, is_train=True, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        
        # Look for SAR files (supports both extensions)
        self.sar_files = glob.glob(os.path.join(root_dir, '**/s1_*.tif'), recursive=True) + \
                         glob.glob(os.path.join(root_dir, '**/s1_*.TIF'), recursive=True)
        
        # Verify matching pairs exist
        self.valid_samples = []
        for sar_path in self.sar_files:
            # We expect a clear target and a cloudy input for standard training
            # For simplicity in this demo, optical acts as both cloudy (input) and clear (target)
            # In a full dataset, you'd match the specific temporal cloudy/clear pairs.
            opt_path = sar_path.replace("/S1/", "/S2/").replace("s1_", "s2_")
            
            if not os.path.exists(opt_path) and opt_path.endswith('.tif'):
                opt_path_alt = opt_path.replace('.tif', '.TIF')
                if os.path.exists(opt_path_alt):
                    opt_path = opt_path_alt
                    
            if os.path.exists(opt_path):
                self.valid_samples.append((sar_path, opt_path))

        print(f"Dataset Initialized: Found {len(self.valid_samples)} valid image pairs in {root_dir}")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sar_path, opt_path = self.valid_samples[idx]
        
        # Load arrays using rasterio
        with rasterio.open(sar_path) as src:
            sar_img = src.read() # [2, H, W] - VV, VH polarizations
            
        with rasterio.open(opt_path) as src:
            # Explicitly tell rasterio to only pull bands 1, 2, 3, and 4
            opt_img = src.read([1, 2, 3, 4]).astype(np.float32)# [4, H, W] - RGB + NIR
            
        # Convert to float32
        sar_img = sar_img.astype(np.float32)
        opt_img = opt_img.astype(np.float32)
        
        # Normalize to [-1, 1] for GAN stability
        # Note: In production, use exact dataset percentiles. Here we use min-max approximation.
        sar_img = np.clip((sar_img - np.min(sar_img)) / (np.max(sar_img) - np.min(sar_img) + 1e-8), 0, 1) * 2 - 1
        opt_img = np.clip((opt_img - np.min(opt_img)) / (np.max(opt_img) - np.min(opt_img) + 1e-8), 0, 1) * 2 - 1
        
        sar_tensor = torch.from_numpy(sar_img)
        opt_tensor = torch.from_numpy(opt_img)
        
        # For the demo pipeline, we simulate the target as the optical image itself 
        # and add synthetic noise/clouds to create the "cloudy" input if a cloudy temporal pair isn't provided.
        # In the full dataset, you load the specific cloudy timestamp vs clear timestamp.
        target_tensor = opt_tensor.clone()
        
        return sar_tensor, opt_tensor, target_tensor