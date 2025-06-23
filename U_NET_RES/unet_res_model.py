import torch.nn as nn
from .unet_res_parts import ResidualBlock, DownRes, UpRes
from U_NET.unet_parts import OutConv


class UNetResidual(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ResidualBlock(n_channels, 64)
        self.down1 = DownRes(64, 128)
        self.down2 = DownRes(128, 256)
        self.down3 = DownRes(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownRes(512, 1024 // factor)
        self.up1 = UpRes(1024, 512 // factor, bilinear)
        self.up2 = UpRes(512, 256 // factor, bilinear)
        self.up3 = UpRes(256, 128 // factor, bilinear)
        self.up4 = UpRes(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
