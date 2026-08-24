import torch
from torch import nn

class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ker_size):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ker_size = ker_size
        self.conv1 = nn.Conv2d(in_dim, out_dim, ker_size)
        self.conv2 = nn.Conv2d(out_dim, out_dim, ker_size)
        self.norm1 = nn.BatchNorm2d(out_dim)
        self.norm2 = nn.BatchNorm2d(out_dim)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.norm1(self.conv1(x)))
        x = self.ReLU(self.norm2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = ConvLayer(1, 128, 3)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.l1 = ConvLayer(128, 256, 3)
        self.l2 = ConvLayer(256, 512, 1)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.r1 = ConvLayer(512 + 256, 256, 1)
        self.up2 = nn.Upsample(scale_factor=3, mode='bilinear')
        self.r0 = ConvLayer(256 + 128, 128, 3)
        self.conv_end1 = nn.Conv2d(128, 64, 6)
        self.conv_end2 = nn.Conv2d(64, 4, 6)
        self.fc = nn.Linear(4 * 10 * 10, 10)
        self.ReLU = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(4)

    def forward(self, x):
        x = self.l0(x)
        x1 = self.max_pool(x)
        x2 = self.l1(x1)
        x3 = self.max_pool(x2)
        x3 = self.l2(x3)
        x3 = self.up1(x3)
        x3 = torch.cat([x3, x2], 1)
        x3 = self.r1(x3)
        x3 = self.up2(x3)
        x3 = torch.cat([x3, x], 1)
        x3 = self.r0(x3)
        x3 = self.ReLU(self.norm1(self.conv_end1(x3)))
        x3 = self.ReLU(self.norm2(self.conv_end2(x3)))
        x3 = x3.view(-1, 4* 10 * 10)
        x3 = self.fc(x3)
        return x3

model = UNet()

name = "UNet"

