import torch
import torch.nn.functional as F
import torch.nn as nn

class LMnet(nn.Module):
    def __init__(self, device, bilinear=True):
        super(LMnet, self).__init__()
        self.bilinear = bilinear
        self.linpara = nn.Parameter(torch.rand([1,49,128],requires_grad=True).to(device=device))
        self.up1 = nn.ConvTranspose2d(64,32,3,stride=2)
        self.up2 = nn.ConvTranspose2d(32,16,3,stride=2,padding=2)
        self.up3 = nn.ConvTranspose2d(16,8,4,stride=2,dilation=3, output_padding = 2)
        self.up4 = nn.ConvTranspose2d(8,4,4,stride=2,padding=1,dilation=1)
        self.conv1 = DoubleConv(32, 32)
        self.conv2 = DoubleConv(16, 16)
        self.conv3 = DoubleConv(8, 8)
        self.doublecov = DoubleConv(4,2)
        self.device = device
    def forward(self, x):
        
        linear = torch.mul(x, self.linpara)
        linear_transpose = torch.transpose(linear,1,2)
        N = linear_transpose.shape[0]
        split = linear_transpose.reshape(N,64,2,7,7)
        sum_split = torch.sum(split, dim=2)
        x1 = self.up1(sum_split)
        x1 = self.conv1(x1)
        x2 = self.up2(x1)
        x2 = self.conv2(x2)
        x3 = self.up3(x2)
        print(x3.shape)
        x3 = self.conv3(x3)
        x4 = self.up4(x3)
        x5 = self.doublecov(x4)
        norm = torch.norm(x5, dim=1)
        cos = x5[:, 0] / (norm + 1e-6)
        sin = x5[:, 1] / (norm + 1e-6)
        tanh_norm = torch.tanh(norm)
        real = tanh_norm * cos
        imag = tanh_norm * sin
        mu = torch.stack((real, imag), dim=1)

        return mu



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

