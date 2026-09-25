import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models
from torchvision.transforms import Normalize

from dataset import SEN12MSDataset
from models import ClearSkyUNet, PatchGANDiscriminator


def calculate_psnr(img1, img2):
    """Calculate PSNR for tensors normalized to [-1, 1]."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")

    return 20 * math.log10(2.0 / math.sqrt(mse))


def psnr_from_sse(sse, n):
    """Calculate PSNR from aggregate squared error and element count."""
    if n == 0 or sse == 0:
        return float("inf")

    mse = sse / n
    return 20 * math.log10(2.0 / math.sqrt(mse))


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()

        vgg = models.vgg19(weights="DEFAULT").features[:36].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, x, y):
        x_rgb = (x[:, :3] + 1.0) / 2.0
        y_rgb = (y[:, :3] + 1.0) / 2.0

        x_rgb = self.normalize(x_rgb)
        y_rgb = self.normalize(y_rgb)

        features_x = self.vgg(x_rgb)
        features_y = self.vgg(y_rgb)

        return self.criterion(features_x, features_y)


def _set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def evaluate_psnr(net_G, loader, device):
    """Evaluate aggregate PSNR over a complete loader."""
    net_G.eval()

    sse = 0.0
    n = 0

    with torch.no_grad():
        for sar, opt_cloudy, opt_target in loader:
            sar = sar.to(device, non_blocking=True)
            opt_cloudy = opt_cloudy.to(device, non_blocking=True)
            opt_target = opt_target.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    fake_opt = net_G(sar, opt_cloudy)
            else:
                fake_opt = net_G(sar, opt_cloudy)

            diff = fake_opt.float() - opt_target.float()
            sse += torch.sum(diff * diff).item()
            n += diff.numel()

    return psnr_from_sse(sse, n)


def train_model(
    data_dir="data/",
    epochs=25,
    batch_size=8,
    lr=0.0002,
    lambda_l1=50.0,
    lambda_vgg=10.0,
    seed=42,
    resume=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    os.makedirs("weights", exist_ok=True)

    # -------------------------------------------------
    # 1. Official ROI-level train / validation / test
    # -------------------------------------------------
    train_dataset = SEN12MSDataset(root_dir=data_dir, split="train")
    val_dataset = SEN12MSDataset(root_dir=data_dir, split="val")
    test_dataset = SEN12MSDataset(root_dir=data_dir, split="test")

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training split is empty. Check the SEN12MS-CR-TS data structure."
        )
    if len(val_dataset) == 0:
        raise RuntimeError(
            "Validation split is empty. Expected validation ROIs may be missing."
        )
    if len(test_dataset) == 0:
        raise RuntimeError(
            "Test split is empty. Expected official hold-out ROIs may be missing."
        )

    print(
        f"📚 Dataset | Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin,
    )

    # -------------------------------------------------
    # 2. Initialize models and losses
    # -------------------------------------------------
    net_G = ClearSkyUNet().to(device)
    net_D = PatchGANDiscriminator().to(device)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    criterion_VGG = VGGPerceptualLoss(device)

    optimizer_G = optim.Adam(net_G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(net_D.parameters(), lr=lr, betas=(0.5, 0.999))

    scaler_G = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    scaler_D = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    start_epoch = 1
    best_val_psnr = -float("inf")

    # -------------------------------------------------
    # 3. Resume support
    # -------------------------------------------------
    if resume:
        checkpoint_path = "weights/latest_model.pth"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Resume requested but {checkpoint_path} was not found."
            )

        checkpoint = torch.load(checkpoint_path, map_location=device)

        net_G.load_state_dict(checkpoint["model_G_state_dict"])
        net_D.load_state_dict(checkpoint["model_D_state_dict"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        scaler_G.load_state_dict(checkpoint["scaler_G_state_dict"])
        scaler_D.load_state_dict(checkpoint["scaler_D_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_psnr = checkpoint.get("best_val_psnr", -float("inf"))

        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])

        print(
            f"🔄 Resumed from epoch {checkpoint['epoch']} | "
            f"Best validation PSNR: {best_val_psnr:.2f} dB"
        )

    # -------------------------------------------------
    # 4. Training loop
    # -------------------------------------------------
    for epoch in range(start_epoch, epochs + 1):
        net_G.train()
        net_D.train()

        running_loss_G = 0.0
        running_loss_D = 0.0
        train_sse = 0.0
        train_n = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch}/{epochs}]",
            unit="batch",
            leave=True,
        )

        for sar, opt_cloudy, opt_target in pbar:
            sar = sar.to(device, non_blocking=True)
            opt_cloudy = opt_cloudy.to(device, non_blocking=True)
            opt_target = opt_target.to(device, non_blocking=True)

            # ----------------------------------------
            # A. Train Discriminator
            # ----------------------------------------
            _set_requires_grad(net_D, True)
            optimizer_D.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    fake_opt = net_G(sar, opt_cloudy)
                    pred_real = net_D(sar, opt_cloudy, opt_target)
                    pred_fake = net_D(sar, opt_cloudy, fake_opt.detach())

                    loss_D_real = criterion_GAN(
                        pred_real, torch.ones_like(pred_real)
                    )
                    loss_D_fake = criterion_GAN(
                        pred_fake, torch.zeros_like(pred_fake)
                    )
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)

                scaler_D.scale(loss_D).backward()
                scaler_D.step(optimizer_D)
                scaler_D.update()
            else:
                fake_opt = net_G(sar, opt_cloudy)
                pred_real = net_D(sar, opt_cloudy, opt_target)
                pred_fake = net_D(sar, opt_cloudy, fake_opt.detach())

                loss_D_real = criterion_GAN(
                    pred_real, torch.ones_like(pred_real)
                )
                loss_D_fake = criterion_GAN(
                    pred_fake, torch.zeros_like(pred_fake)
                )
                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                loss_D.backward()
                optimizer_D.step()

            # ----------------------------------------
            # B. Train Generator
            # ----------------------------------------
            _set_requires_grad(net_D, False)
            optimizer_G.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    pred_fake_g = net_D(sar, opt_cloudy, fake_opt)

                    loss_G_GAN = criterion_GAN(
                        pred_fake_g, torch.ones_like(pred_fake_g)
                    )
                    loss_G_L1 = criterion_L1(fake_opt, opt_target)
                    loss_G_VGG = criterion_VGG(fake_opt, opt_target)

                    loss_G = (
                        loss_G_GAN
                        + lambda_l1 * loss_G_L1
                        + lambda_vgg * loss_G_VGG
                    )

                scaler_G.scale(loss_G).backward()
                scaler_G.step(optimizer_G)
                scaler_G.update()
            else:
                pred_fake_g = net_D(sar, opt_cloudy, fake_opt)
                loss_G_GAN = criterion_GAN(
                    pred_fake_g, torch.ones_like(pred_fake_g)
                )
                loss_G_L1 = criterion_L1(fake_opt, opt_target)
                loss_G_VGG = criterion_VGG(fake_opt, opt_target)

                loss_G = (
                    loss_G_GAN
                    + lambda_l1 * loss_G_L1
                    + lambda_vgg * loss_G_VGG
                )

                loss_G.backward()
                optimizer_G.step()

            _set_requires_grad(net_D, True)

            diff = fake_opt.detach().float() - opt_target.float()
            train_sse += torch.sum(diff * diff).item()
            train_n += diff.numel()

            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()

            batch_psnr = calculate_psnr(fake_opt.detach(), opt_target)

            pbar.set_postfix(
                Loss_G=f"{loss_G.item():.3f}",
                Loss_D=f"{loss_D.item():.3f}",
                PSNR=f"{batch_psnr:.2f}dB",
            )

        epoch_loss_G = running_loss_G / len(train_loader)
        epoch_loss_D = running_loss_D / len(train_loader)
        train_psnr = psnr_from_sse(train_sse, train_n)

        # ----------------------------------------
        # C. Validation
        # ----------------------------------------
        val_psnr = evaluate_psnr(net_G, val_loader, device)

        print(
            f"📊 Epoch [{epoch}/{epochs}] | "
            f"Loss G: {epoch_loss_G:.4f} | "
            f"Loss D: {epoch_loss_D:.4f} | "
            f"Train PSNR: {train_psnr:.2f} dB | "
            f"Val PSNR: {val_psnr:.2f} dB"
        )

        # Model selection uses validation only.
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(net_G.state_dict(), "weights/best_model.pth")
            print(
                f"  🏆 New Best Model Saved! "
                f"(Validation PSNR: {best_val_psnr:.2f} dB)"
            )

        # Fully resumable checkpoint.
        checkpoint = {
            "epoch": epoch,
            "model_G_state_dict": net_G.state_dict(),
            "model_D_state_dict": net_D.state_dict(),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "optimizer_D_state_dict": optimizer_D.state_dict(),
            "scaler_G_state_dict": scaler_G.state_dict(),
            "scaler_D_state_dict": scaler_D.state_dict(),
            "best_val_psnr": best_val_psnr,
            "seed": seed,
            "lambda_l1": lambda_l1,
            "lambda_vgg": lambda_vgg,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        torch.save(checkpoint, "weights/latest_model.pth")

    # ----------------------------------------
    # 5. Final hold-out test
    # ----------------------------------------
    best_state = torch.load("weights/best_model.pth", map_location=device)
    net_G.load_state_dict(best_state)

    test_psnr = evaluate_psnr(net_G, test_loader, device)

    print(
        f"🧪 Final Hold-out Test PSNR: {test_psnr:.2f} dB "
        f"(best model selected only by validation PSNR)"
    )


if __name__ == "__main__":
    train_model()
