
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

class BSNet(nn.Module):
    def __init__(self, size, device, vertex, num_bound_points, n_channels = 2, n_classes = 2, bilinear=True):
        super(BSNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        vertex = torch.from_numpy(vertex[num_bound_points:, :].astype(np.float32))
        vx, vy = vertex[:, 0], vertex[:, 1]
        vertex_rect = torch.zeros_like(vertex)
        vr = torch.sqrt(vx**2 + vy**2)
        cos = torch.abs(vx / vr)
        sin = torch.abs(vy / vr)
        ratio = torch.maximum(cos, sin)
        vertex_rect[:, 0] = vx / ratio
        vertex_rect[:, 1] = -vy / ratio 
        self.complement_grid = vertex_rect.unsqueeze(0).unsqueeze(0).to(device=device)
        self.complement_grid.requires_grad = False

        h, w = size

        assert h%8==0
        ws = int(h/8)

        self.fft = FFT(ws)
        self.bound = Boundary(device, ws*ws*2, num_bound_points)
        self.dtl = DTL(ws, ws)

        self.down = Down(16, 16)
        self.inc = DoubleConv(2, 64)
        self.inc2 = Conv(2, 16)
        self.up1 = Up(64, 48, bilinear)
        self.up2 = Up(48+16, 48, bilinear)
        self.up3 = Up(48, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        N = x.shape[0]
        x1 = self.fft(x)
        bound, theta = self.bound(x1)

        N = x1.shape[0]
        x12 = self.dtl(x1)
        x2 = self.inc2(x)
        x2 = self.down(x2)
        x = self.inc(x12)
        x = self.up1(x)
        x = self.up2(x, x2)
        x = self.up3(x)
        mapping = self.outc(x)

        inner_coord = F.grid_sample(mapping, self.complement_grid.repeat(N, 1, 1, 1))[:, :, 0, :]
        coord = torch.cat((bound, inner_coord), dim=2)

        return coord, theta

class Boundary(nn.Module):

    def __init__(self, device, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 400)
        self.linear2 = nn.Linear(400, 400)
        self.linear3 = nn.Linear(400,500)
        self.linear4 = nn.Linear(500,500)
        self.linear5 = nn.Linear(500,400)
        self.linear6 = nn.Linear(400,400)
        self.linear7 = nn.Linear(400, output_dim) 
        self.softmax = nn.Softmax(dim=1)
        squre = torch.triu(torch.ones(output_dim-1, output_dim-1))
        left = torch.zeros(output_dim-1,1)
        low = torch.zeros(1,output_dim)
        a = torch.cat((left,squre), dim = 1)
        self.matrix = torch.cat((a,low),dim = 0).to(device=device)
        self.matrix.requires_grad = False
        self.second_fixed_point = math.floor(output_dim / 2)

    def forward(self, x):
        N = x.shape[0]
        x0 = torch.reshape(x, (N,-1,))
        x1 = F.relu(self.linear1(x0))
        x2 = F.relu(self.linear2(x1))
        x3 = F.relu(self.linear3(x2))
        x4 = F.relu(self.linear4(x3))
        x5 = F.relu(self.linear5(x4))
        x6 = F.relu(self.linear6(x5))
        x7 = self.linear7(x6)

        N_outer_vertex = x7.shape[1]

        x7_upper = x7

        sigmoid_x7_upper = torch.sigmoid(x7_upper)
        sigmoid_x7_upper = sigmoid_x7_upper / torch.sum(sigmoid_x7_upper, dim=1).unsqueeze(1)
        theta_upper = sigmoid_x7_upper * math.pi * 2

        theta = theta_upper


        p_theta = torch.mm(theta, self.matrix)

        p_x = torch.cos(p_theta)
        p_y = torch.sin(p_theta)
        bound = torch.stack((p_x, p_y), dim=1)

        return bound, theta 



class FFT(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.r = width//2

    def fft(self, img):
        r = self.r
        rows, cols, c = img.shape
        crow, ccol = rows//2, cols//2 
        f_t = torch.fft(img, 2)

        roll1 = torch.roll(f_t, rows//2, 0)
        roll2 = torch.roll(roll1, cols//2, 1)
        img_back = roll2[crow-r:crow+r, ccol-r:ccol+r, :]
        return img_back

    def forward(self, x):
        N, c, rows, cols = x.shape
        assert c==2
        r = self.r
        x = x.permute((0, 2, 3, 1))
        out = torch.zeros((N, 2*r, 2*r, c), device = x.device)
        for n in range(N):
            out[n] = self.fft(x[n])

        return out.permute((0, 3, 1, 2))

class iFFT(nn.Module):
    def __init__(self):
        super().__init__()

    def fft(self, img):
        rows, cols, c = img.shape
        crow, ccol = rows//2, cols//2
        img_back = torch.zeros_like(img)
        for i in range(c//2):
            freq = img[:, :, i*2:i*2+2]
            roll1 = torch.roll(freq, rows//2, 0)
            roll2 = torch.roll(roll1, cols//2, 1)
            img_back[:, :, i*2:i*2+2] = torch.ifft(roll2, 2)
        return img_back

    def forward(self, x):
        N, c, rows, cols = x.shape
        x = x.permute((0, 2, 3, 1))
        out = torch.zeros_like(x)
        for n in range(N):
            out[n] = self.fft(x[n])

        return out.permute((0, 3, 1, 2))

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

class DTL(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, padding=True):
        super().__init__()
        self.in_channels = in_channels
        assert in_channels == out_channels
        self.conv1r = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias = False)
        self.conv1i = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias = False)
        self.conv2r = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias = False)
        self.conv2i = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias = False)

    def forward(self, x):
        N = x.shape[0]
        x_pmt = x.permute((0, 2, 3, 1))
        x1r = self.conv1r(x_pmt)
        x1i = self.conv1i(x_pmt)
        x1ac = x1r[:, :, :, 0]
        x1bc = x1r[:, :, :, 1]
        x1ad = x1i[:, :, :, 0]
        x1bd = x1i[:, :, :, 1]
        real = x1ac - x1bd
        imag = x1ad + x1bc
        x2 = torch.stack((real, imag), dim=1)
        x3 = x2.permute((0, 3, 2, 1))
        x4r = self.conv2r(x3)
        x4i = self.conv2i(x3)
        x4ac = x4r[:, :, :, 0]
        x4bc = x4r[:, :, :, 1]
        x4ad = x4i[:, :, :, 0]
        x4bd = x4i[:, :, :, 1]
        real = x4ac - x4bd
        imag = x4ad + x4bc
        x5 = torch.stack((real, imag), dim=1)
        x6 = x5.permute((0, 1, 3, 2))
        return x6

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            #DoubleConv(in_channels, out_channels)
            Conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
