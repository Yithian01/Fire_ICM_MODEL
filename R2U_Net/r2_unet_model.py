import torch.nn as nn
from .r2_unet_parts import DownR2, UpR2
from .recurrent_block import RecurrentResidualBlock
from U_NET.unet_parts import OutConv  # 그대로 사용

class R2UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, t=2):
        super().__init__()
        self.inc = RecurrentResidualBlock(n_channels, 64, t)
        self.down1 = DownR2(64, 128, t)
        self.down2 = DownR2(128, 256, t)
        self.down3 = DownR2(256, 512, t)
        factor = 2 if bilinear else 1
        self.down4 = DownR2(512, 1024 // factor, t)
        self.up1 = UpR2(1024, 512 // factor, bilinear, t)
        self.up2 = UpR2(512, 256 // factor, bilinear, t)
        self.up3 = UpR2(256, 128 // factor, bilinear, t)
        self.up4 = UpR2(128, 64, bilinear, t)
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
