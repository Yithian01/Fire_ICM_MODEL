import torch
import torch.nn as nn

class RecurrentConvBlock(nn.Module):
    def __init__(self, channels, t=2):
        super().__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            else:
                x1 = self.conv(x + x1)
        return x1


class RecurrentResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.rcb = RecurrentConvBlock(out_channels, t=t)

    def forward(self, x):
        x = self.conv1x1(x)
        return x + self.rcb(x)
