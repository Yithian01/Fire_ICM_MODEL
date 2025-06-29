import torch
import torch.nn as nn
import torch.nn.functional as F
from .recurrent_block import RecurrentResidualBlock

class DownR2(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            RecurrentResidualBlock(in_channels, out_channels, t)
        )

    def forward(self, x):
        return self.block(x)

class UpR2(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, t=2):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = RecurrentResidualBlock(in_channels, out_channels, t)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
