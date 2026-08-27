# src/train.py
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SEN12MSDataset
from models import ClearSkyUNet, PatchGANDiscriminator

def calculate_psnr(img1, img2):
    """Calculates Peak Signal-to-Noise Ratio (PSNR) for normalized [-1, 1] tensors."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    # Range is 2 for tensors normalized to [-1, 1]
    max_pixel = 2.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse.item()))
    return psnr

def train_model(data_dir="data/", epochs=25, batch_size=8, lr=0.0002, lambda_l1=100.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")
    
    os.makedirs("weights", exist_ok=True)
    
    # Initialize Dataset & DataLoader
    dataset = SEN12MSDataset(root_dir=data_dir, is_train=True)
    if len(dataset) == 0:
        print("❌ Dataset is empty! Ensure data is extracted or use demo_samples/.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Initialize Models
    net_G = ClearSkyUNet().to(device)
    net_D = PatchGANDiscriminator().to(device)
    
    # Loss Functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    
    # Optimizers (Adam with typical GAN parameters)
    optimizer_G = optim.Adam(net_G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(net_D.parameters(), lr=lr, betas=(0.5, 0.999))
    
    best_psnr = 0.0
    
    for epoch in range(1, epochs + 1):
        net_G.train()
        net_D.train()
        
        running_loss_G = 0.0
        running_loss_D = 0.0
        running_psnr = 0.0
        
        for i, (sar, opt_cloudy, opt_target) in enumerate(dataloader):
            sar = sar.to(device)
            opt_cloudy = opt_cloudy.to(device)
            opt_target = opt_target.to(device)
            
            # ----------------------------------------
            # 1. Train Discriminator
            # ----------------------------------------
            optimizer_D.zero_grad()
            
            # Real Pair
            pred_real = net_D(sar, opt_cloudy, opt_target)
            target_real = torch.ones_like(pred_real).to(device)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake Pair
            fake_opt = net_G(sar, opt_cloudy)
            pred_fake = net_D(sar, opt_cloudy, fake_opt.detach())
            target_fake = torch.zeros_like(pred_fake).to(device)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            
            # Combined Discriminator Loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # ----------------------------------------
            # 2. Train Generator
            # ----------------------------------------
            optimizer_G.zero_grad()
            
            # Adversarial Loss (Trick Discriminator)
            pred_fake_g = net_D(sar, opt_cloudy, fake_opt)
            loss_G_GAN = criterion_GAN(pred_fake_g, target_real)
            
            # L1 Reconstruction Loss
            loss_G_L1 = criterion_L1(fake_opt, opt_target)
            
            # Total Generator Loss
            loss_G = loss_G_GAN + (lambda_l1 * loss_G_L1)
            loss_G.backward()
            optimizer_G.step()
            
            # Track Metrics
            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
            batch_psnr = calculate_psnr(fake_opt, opt_target)
            running_psnr += batch_psnr
            
        epoch_loss_G = running_loss_G / len(dataloader)
        epoch_loss_D = running_loss_D / len(dataloader)
        epoch_psnr = running_psnr / len(dataloader)
        
        print(f"Epoch [{epoch}/{epochs}] | Loss G: {epoch_loss_G:.4f} | Loss D: {epoch_loss_D:.4f} | PSNR: {epoch_psnr:.2f} dB")
        
        # Save Best Model Checkpoint
        if epoch_psnr > best_psnr:
            best_psnr = epoch_psnr
            torch.save(net_G.state_dict(), "weights/best_model.pth")
            print(f"  🏆 New Best Model Saved! (PSNR: {best_psnr:.2f} dB)")

if __name__ == "__main__":
    train_model()