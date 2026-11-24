"""Exact architecture extracted from the existing U-Net notebook; no training code."""
import torch
from torch import nn
from torch.nn import functional as F

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1,bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1,bias=False),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
    def forward(self,x):
        return self.layers(x)

class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_channels,out_channels))
    def forward(self,x):
        return self.layers(x)

class Up(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
        self.conv = DoubleConv(in_channels,out_channels)
    def forward(self,x,skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
        return self.conv(torch.cat([skip,x],dim=1))

class UNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=1,base_channels=32):
        super().__init__()
        b = base_channels
        self.input = DoubleConv(in_channels,b)
        self.down1 = Down(b,b*2);self.down2 = Down(b*2,b*4)
        self.down3 = Down(b*4,b*8);self.bottleneck = Down(b*8,b*16)
        self.up1 = Up(b*16,b*8);self.up2 = Up(b*8,b*4)
        self.up3 = Up(b*4,b*2);self.up4 = Up(b*2,b)
        self.output = nn.Conv2d(b,out_channels,kernel_size=1)
    def forward(self,x):
        a = self.input(x);b = self.down1(a);c = self.down2(b);d = self.down3(c)
        x = self.bottleneck(d)
        x = self.up1(x,d);x = self.up2(x,c);x = self.up3(x,b);x = self.up4(x,a)
        return self.output(x)
