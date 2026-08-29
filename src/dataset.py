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
        import torch.nn.functional as F
        
        sar_path, opt_path = self.valid_samples[idx]
        
        # Load arrays using rasterio
        with rasterio.open(sar_path) as src:
            sar_img = src.read().astype(np.float32)  # [2, H, W]
            
        with rasterio.open(opt_path) as src:
            opt_img = src.read([1, 2, 3, 4]).astype(np.float32)  # [4, H, W]
            
        # 1. Per-Channel Normalization (CRITICAL FIX)
        for i in range(sar_img.shape[0]):
            c_min, c_max = np.min(sar_img[i]), np.max(sar_img[i])
            sar_img[i] = (sar_img[i] - c_min) / (c_max - c_min + 1e-8)
            
        for i in range(opt_img.shape[0]):
            c_min, c_max = np.min(opt_img[i]), np.max(opt_img[i])
            opt_img[i] = (opt_img[i] - c_min) / (c_max - c_min + 1e-8)
            
        # Scale to [-1, 1]
        sar_img = sar_img * 2.0 - 1.0
        opt_img = opt_img * 2.0 - 1.0
        
        sar_tensor = torch.from_numpy(sar_img)
        opt_tensor = torch.from_numpy(opt_img)
        
        # The clear optical image is our ground truth target
        target_tensor = opt_tensor.clone()
        
        # 2. Synthetic Cloud Generation (THE IDENTITY TRAP FIX)
        _, H, W = opt_tensor.shape
        # Create low-res noise and upscale it to create natural-looking "blobs"
        noise = torch.rand(1, 1, H // 16, W // 16)
        cloud_mask = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
        
        # Threshold the blobs to create opaque cloud regions (tweak 0.65 for more/less clouds)
        cloud_mask = (cloud_mask > 0.65).float()
        
        # Apply clouds (value 1.0 in [-1, 1] scale is pure white)
        cloudy_tensor = opt_tensor.clone()
        cloudy_tensor = cloudy_tensor * (1.0 - cloud_mask) + cloud_mask * 1.0
        
        return sar_tensor, cloudy_tensor, target_tensor