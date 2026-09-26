import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.c1 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.c2 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.c1(x))
        y = self.c2(y)
        return x + self.res_scale * y


class LISS4DSen2CR(nn.Module):
    """
    DSen2-CR-style residual reconstruction model adapted to LISS-IV.

    Optical input:
      [G, R, NIR]

    Optional SAR:
      [VV, VH]
    """
    def __init__(self, features=256, blocks=16, res_scale=0.1, use_sar=False):
        super().__init__()

        self.use_sar = use_sar
        self.opt_channels = 3
        self.sar_channels = 2 if use_sar else 0
        self.in_channels = self.opt_channels + self.sar_channels

        self.inp = nn.Sequential(
            nn.Conv2d(self.in_channels, features, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList(
            [ResBlock(features, res_scale=res_scale) for _ in range(blocks)]
        )

        self.out = nn.Conv2d(features, 3, 3, padding=1, bias=True)

    def forward(self, cloudy, sar=None):
        if cloudy.ndim != 4 or cloudy.shape[1] != 3:
            raise ValueError(
                f"cloudy must have shape [B,3,H,W], got {tuple(cloudy.shape)}"
            )

        if self.use_sar:
            if sar is None:
                raise ValueError(
                    "This model was created with use_sar=True but sar is missing."
                )
            if sar.ndim != 4 or sar.shape[1] != 2:
                raise ValueError(
                    f"sar must have shape [B,2,H,W], got {tuple(sar.shape)}"
                )
            if sar.shape[-2:] != cloudy.shape[-2:]:
                raise ValueError(
                    "SAR and LISS-IV spatial sizes must match after co-registration."
                )
            x = torch.cat([cloudy, sar], dim=1)
        else:
            if sar is not None:
                raise ValueError(
                    "sar was supplied to a model created with use_sar=False."
                )
            x = cloudy

        z = self.inp(x)
        for block in self.blocks:
            z = block(z)

        return cloudy + self.out(z)
