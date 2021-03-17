""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch.nn as nn
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64 * factor, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()

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






class Up_(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x):
        x = self.up(x)
        return self.conv(x)



class UNetHalf(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        """Only the up part of the unet

        Args:
            n_channels ([type]): [how many input channels=n_sources]
            n_classes ([type]): [how many output classes=n_sources]
            bilinear (bool, optional): [use interpolation or deconv]. Defaults to False(use deconv).
        """
        super(UNetHalf, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 1024)
        self.up1 = Up_(1024, 512, bilinear)
        self.up2 = Up_(512, 256, bilinear)
        self.up3 = Up_(256, 128, bilinear)
        self.up4 = Up_(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # self.classify = nn.Linear(4096, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # logits = self.outc(x)
        return x



