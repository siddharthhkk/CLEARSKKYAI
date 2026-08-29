import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models

from dataset import SEN12MSDataset
from models import ClearSkyUNet, PatchGANDiscriminator


def calculate_psnr(img1, img2):
    """Calculates Peak Signal-to-Noise Ratio (PSNR) for normalized [-1, 1] tensors."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 2.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse.item()))
    return psnr


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load pre-trained VGG19 features and freeze weights
        vgg = models.vgg19(weights='DEFAULT').features[:36].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        # VGG expects 3 channels (RGB). Slice out the NIR band if your tensors have 4 channels.
        x_rgb = x[:, :3, :, :]
        y_rgb = y[:, :3, :, :]
        
        # Extract features and compare
        features_x = self.vgg(x_rgb)
        features_y = self.vgg(y_rgb)
        return self.criterion(features_x, features_y)

def train_model(
    data_dir="data/",
    epochs=25,
    batch_size=8,
    lr=0.0002,
    lambda_l1=100.0,
    lambda_vgg=10.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    os.makedirs("weights", exist_ok=True)

    # 1. Initialize Dataset & Optimized DataLoader
    dataset = SEN12MSDataset(root_dir=data_dir, is_train=True)
    if len(dataset) == 0:
        print(
            "❌ Dataset is empty! Ensure data is extracted or use demo_samples/."
        )
        return

    # Increased num_workers & pin_memory for faster T4 GPU disk reading
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 2. Initialize Models
    net_G = ClearSkyUNet().to(device)
    net_D = PatchGANDiscriminator().to(device)

    # 3. Loss Functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    criterion_VGG = VGGPerceptualLoss(device)

    # 4. Optimizers
    optimizer_G = optim.Adam(net_G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(net_D.parameters(), lr=lr, betas=(0.5, 0.999))

    # 5. Mixed Precision Scaler (Speeds up T4 training significantly)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_psnr = 0.0

    for epoch in range(1, epochs + 1):
        net_G.train()
        net_D.train()

        running_loss_G = 0.0
        running_loss_D = 0.0
        running_psnr = 0.0

        # Wrap dataloader with tqdm for live progress tracking
        pbar = tqdm(
            dataloader,
            desc=f"Epoch [{epoch}/{epochs}]",
            unit="batch",
            leave=True,
        )

        for sar, opt_cloudy, opt_target in pbar:
            sar = sar.to(device, non_blocking=True)
            opt_cloudy = opt_cloudy.to(device, non_blocking=True)
            opt_target = opt_target.to(device, non_blocking=True)

            # ----------------------------------------
            # 1. Train Discriminator
            # ----------------------------------------
            optimizer_D.zero_grad()

            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.autocast(device_type=device_type, enabled=torch.cuda.is_available()):
                # Real Pair
                pred_real = net_D(sar, opt_cloudy, opt_target)
                target_real = torch.ones_like(pred_real).to(device)
                loss_D_real = criterion_GAN(pred_real, target_real)

                # Fake Pair
                fake_opt = net_G(sar, opt_cloudy)
                pred_fake = net_D(sar, opt_cloudy, fake_opt.detach())
                target_fake = torch.zeros_like(pred_fake).to(device)
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                loss_D = (loss_D_real + loss_D_fake) * 0.5

            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)

            # ----------------------------------------
            # 2. Train Generator
            # ----------------------------------------
            optimizer_G.zero_grad()

            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.autocast(device_type=device_type, enabled=torch.cuda.is_available()):
                pred_fake_g = net_D(sar, opt_cloudy, fake_opt)
                loss_G_GAN = criterion_GAN(pred_fake_g, target_real)
                loss_G_L1 = criterion_L1(fake_opt, opt_target)
                
                # NEW: Calculate Perceptual Loss
                loss_G_VGG = criterion_VGG(fake_opt, opt_target)
                
                # Balance the weights: Dial back L1 slightly, let VGG handle textures
                loss_G = loss_G_GAN + (50.0 * loss_G_L1) + (10.0 * loss_G_VGG)

            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)

            # Update scaler after step
            scaler.update()

            # Track Metrics
            batch_psnr = calculate_psnr(fake_opt, opt_target)
            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
            running_psnr += batch_psnr

            # Live terminal update inside the tqdm progress bar
            pbar.set_postfix(
                Loss_G=f"{loss_G.item():.3f}",
                Loss_D=f"{loss_D.item():.3f}",
                PSNR=f"{batch_psnr:.2f}dB",
            )

        epoch_loss_G = running_loss_G / len(dataloader)
        epoch_loss_D = running_loss_D / len(dataloader)
        epoch_psnr = running_psnr / len(dataloader)

        print(
            f"📊 Epoch [{epoch}/{epochs}] Summary | Loss G: {epoch_loss_G:.4f} | Loss D: {epoch_loss_D:.4f} | Avg PSNR: {epoch_psnr:.2f} dB"
        )

        # Always save latest checkpoint as fallback
        checkpoint = {
            'epoch': epoch,
            'model_G_state_dict': net_G.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }
        torch.save(checkpoint, "weights/latest_model.pth")

        # Save best model checkpoint
        if epoch_psnr > best_psnr:
            best_psnr = epoch_psnr
            torch.save(net_G.state_dict(), "weights/best_model.pth")
            print(f"  🏆 New Best Model Saved! (PSNR: {best_psnr:.2f} dB)")


if __name__ == "__main__":
    train_model()