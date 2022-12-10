import torch
import torch.nn.functional as F
import torch.nn as nn

class AutoLM(nn.Module):
    def __init__(self, device):
        super(AutoLM, self).__init__()
        self.device = device
        
        self.inc1 = DoubleConv(2, 4)
        self.inc2 = DoubleConv(4, 2)

        self.down1 = Down(2, 4)
        self.down2 = Down(4, 4)
        self.down3 = Down(4, 2)
        self.down4 = Down(2, 1)

        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        

    def forward(self, x):
        N = x.shape[0]

        x_2 = self.inc1(x)
        x_3 = self.inc2(x_2)

        x_4 = self.down1(x_3)
        x_5 = self.down2(x_4)
        x_6 = self.down3(x_5) 
        x_7 = self.down4(x_6)

        lin1 = x_7.reshape(N, -1)
        lin2 = self.linear1(lin1)
        lin3 = self.linear2(lin2)
        lin4 = self.linear3(lin3)

        lin5 = torch.unsqueeze(lin4, 1)
        lin5 = lin5.reshape(N, 8, 2)

        return lin5

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, padding=True):
        super().__init__()
        pad = 1 if padding else 0
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=pad, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=True):
        super().__init__()
        pad = 1 if padding else 0
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
