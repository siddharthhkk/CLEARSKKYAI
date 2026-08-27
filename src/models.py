# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialChannelAttention(nn.Module):
    """SCA Bottleneck Module to weight SAR vs Optical features."""
    def __init__(self, in_channels):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False)
        )
        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention Branch
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_weight = torch.sigmoid(avg_out + max_out)
        x_c = x * channel_weight
        
        # Spatial Attention Branch
        avg_out_s = torch.mean(x_c, dim=1, keepdim=True)
        max_out_s, _ = torch.max(x_c, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_out_s, max_out_s], dim=1)
        spatial_weight = torch.sigmoid(self.conv_spatial(spatial_cat))
        
        return x_c * spatial_weight

class ConvBlock(nn.Module):
    """Standard Convolutional Block (Conv -> BatchNorm -> LeakyReLU)."""
    def __init__(self, in_c, out_c, downsample=True):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ClearSkyUNet(nn.Module):
    """Dual-Encoder U-Net with SCA Fusion for SAR-Optical Cloud Removal."""
    def __init__(self):
        super().__init__()
        # ENCODER A (SAR: 2 Channels)
        self.encA1 = ConvBlock(2, 64, downsample=True)
        self.encA2 = ConvBlock(64, 128, downsample=True)
        self.encA3 = ConvBlock(128, 256, downsample=True)
        self.encA4 = ConvBlock(256, 512, downsample=True)
        
        # ENCODER B (Optical: 4 Channels)
        self.encB1 = ConvBlock(4, 64, downsample=True)
        self.encB2 = ConvBlock(64, 128, downsample=True)
        self.encB3 = ConvBlock(128, 256, downsample=True)
        self.encB4 = ConvBlock(256, 512, downsample=True)
        
        # FUSION BOTTLENECK + SCA
        self.sca = SpatialChannelAttention(in_channels=1024)
        self.bottleneck_conv = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        
        # DECODER
        self.dec1 = self._up_block(512, 256)
        self.dec2 = self._up_block(256 + 512 + 512, 128) # + skip connections
        self.dec3 = self._up_block(128 + 256 + 256, 64)
        self.dec4 = self._up_block(64 + 128 + 128, 64)
        
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(64 + 64 + 64, 4, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Output bounded to [-1, 1]
        )

    def _up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, sar, opt):
        # Encode SAR
        a1 = self.encA1(sar)
        a2 = self.encA2(a1)
        a3 = self.encA3(a2)
        a4 = self.encA4(a3)
        
        # Encode Optical
        b1 = self.encB1(opt)
        b2 = self.encB2(b1)
        b3 = self.encB3(b2)
        b4 = self.encB4(b3)
        
        # Fusion Bottleneck
        fused = torch.cat([a4, b4], dim=1) # [B, 1024, H/16, W/16]
        fused = self.sca(fused)
        fused = self.bottleneck_conv(fused)
        
        # Decode with dual skip connections
        d1 = self.dec1(fused)
        d1 = torch.cat([d1, a3, b3], dim=1) # Add skip A3, B3
        
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, a2, b2], dim=1) # Add skip A2, B2
        
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, a1, b1], dim=1) # Add skip A1, B1
        
        d4 = self.dec4(d3)
        
        out = self.final_conv(d4) # [B, 4, H, W]
        return out

class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN Discriminator for high-frequency structural fidelity."""
    def __init__(self):
        super().__init__()
        # Inputs: SAR (2) + Cloudy Opt (4) + Target/Gen Opt (4) = 10
        self.model = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, sar, opt_cloudy, opt_target):
        # Concatenate multi-modal inputs along channel dimension
        x = torch.cat([sar, opt_cloudy, opt_target], dim=1)
        return self.model(x)