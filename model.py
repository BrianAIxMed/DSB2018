import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DilatedConvBlock(nn.Module):
    ''' no dilation applied if dilation equals to 1 '''
    def __init__(self, in_size, out_size, kernel_size=3, dropout_rate=0.1, activation=F.relu, dilation=1):
        super().__init__()
        # to keep same width output, assign padding equal to dilation
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_size)
        self.activation = activation
        if dropout_rate > 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        else:
            self.drop = lambda x: x # no-op

    def forward(self, x):
        # CAB: conv -> activation -> batch normal
        x = self.norm(self.activation(self.conv(x)))
        x = self.drop(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout_rate=0.2, dilation=1):
        super().__init__()
        self.block1 = DilatedConvBlock(in_size, out_size, dropout_rate=0)
        self.block2 = DilatedConvBlock(out_size, out_size, dropout_rate=dropout_rate, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        out, indices = self.pool(x)
        return out, x, indices

class ConvUpBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout_rate=0.2, dilation=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_size, in_size//2, 2, stride=2)
        self.block1 = DilatedConvBlock(in_size//2 + out_size, out_size, dropout_rate=0)
        self.block2 = DilatedConvBlock(out_size, out_size, dropout_rate=dropout_rate, dilation=dilation)

    def forward(self, x, bridge):
        x = self.up(x)
        # align concat size by adding pad
        diffY = x.shape[2] - bridge.shape[2]
        diffX = x.shape[3] - bridge.shape[3]
        bridge = F.pad(bridge, (0, diffX, 0, diffY), mode='reflect')
        x = torch.cat([x, bridge], 1)
        # CAB: conv -> activation -> batch normal
        x = self.block1(x)
        x = self.block2(x)
        return x
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # down conv
        self.c1 = ConvBlock(3, 16)
        self.c2 = ConvBlock(16, 32)
        self.c3 = ConvBlock(32, 64)
        self.c4 = ConvBlock(64, 128)
        # bottom conv tunnel
        self.cu = ConvBlock(128, 256)
        # up conv
        self.u5 = ConvUpBlock(256, 128)
        self.u6 = ConvUpBlock(128, 64)
        self.u7 = ConvUpBlock(64, 32)
        self.u8 = ConvUpBlock(32, 16)
        # final conv tunnel
        self.ce = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        _, x = self.cu(x) # no maxpool for U bottom
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.ce(x)
        x = F.sigmoid(x)
        return x

# Dilated UNet
class DUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # down conv
        self.c1 = ConvBlock(3, 16, dilation=2)
        self.c2 = ConvBlock(16, 32, dilation=2)
        self.c3 = ConvBlock(32, 64, dilation=2)
        self.c4 = ConvBlock(64, 128, dilation=2)
        # bottom dilated conv tunnel
        self.d1 = DilatedConvBlock(128, 256)
        self.d2 = DilatedConvBlock(256, 256, dilation=2)
        self.d3 = DilatedConvBlock(256, 256, dilation=4)
        self.d4 = DilatedConvBlock(256, 256, dilation=8)
        # up conv
        self.u5 = ConvUpBlock(256, 128)
        self.u6 = ConvUpBlock(128, 64)
        self.u7 = ConvUpBlock(64, 32)
        self.u8 = ConvUpBlock(32, 16)
        # final conv tunnel
        self.ce = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.ce(x)
        x = F.sigmoid(x)
        return x

# Contour aware UNet
class CaUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 16)
        self.c2 = ConvBlock(16, 32)
        self.c3 = ConvBlock(32, 64)
        self.c4 = ConvBlock(64, 128)
        # bottom conv tunnel
        self.cu = ConvBlock(128, 256)
        # segmentation up conv branch
        self.u5s = ConvUpBlock(256, 128)
        self.u6s = ConvUpBlock(128, 64)
        self.u7s = ConvUpBlock(64, 32)
        self.u8s = ConvUpBlock(32, 16)
        self.ces = nn.Conv2d(16, 1, 1)
        # contour up conv branch
        self.u5c = ConvUpBlock(256, 128)
        self.u6c = ConvUpBlock(128, 64)
        self.u7c = ConvUpBlock(64, 32)
        self.u8c = ConvUpBlock(32, 16)
        self.cec = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        _, x = self.cu(x) # no maxpool for U bottom
        xs = self.u5s(x, c4)
        xs = self.u6s(xs, c3)
        xs = self.u7s(xs, c2)
        xs = self.u8s(xs, c1)
        xs = self.ces(xs)
        xs = F.sigmoid(xs)
        xc = self.u5c(x, c4)
        xc = self.u6c(xc, c3)
        xc = self.u7c(xc, c2)
        xc = self.u8c(xc, c1)
        xc = self.cec(xc)
        xc = F.sigmoid(xc)
        return xs, xc

# Contour aware Marker Unet
class CamUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 16)
        self.c2 = ConvBlock(16, 32)
        self.c3 = ConvBlock(32, 64)
        self.c4 = ConvBlock(64, 128)
        # bottom conv tunnel
        self.cu = ConvBlock(128, 256)
        # segmentation up conv branch
        self.u5s = ConvUpBlock(256, 128)
        self.u6s = ConvUpBlock(128, 64)
        self.u7s = ConvUpBlock(64, 32)
        self.u8s = ConvUpBlock(32, 16)
        self.ces = nn.Conv2d(16, 1, 1)
        # contour up conv branch
        self.u5c = ConvUpBlock(256, 128)
        self.u6c = ConvUpBlock(128, 64)
        self.u7c = ConvUpBlock(64, 32)
        self.u8c = ConvUpBlock(32, 16)
        self.cec = nn.Conv2d(16, 1, 1)
        # marker up conv branch
        self.u5m = ConvUpBlock(256, 128)
        self.u6m = ConvUpBlock(128, 64)
        self.u7m = ConvUpBlock(64, 32)
        self.u8m = ConvUpBlock(32, 16)
        self.cem = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        _, x = self.cu(x) # no maxpool for U bottom
        xs = self.u5s(x, c4)
        xs = self.u6s(xs, c3)
        xs = self.u7s(xs, c2)
        xs = self.u8s(xs, c1)
        xs = self.ces(xs)
        xs = F.sigmoid(xs)
        xc = self.u5c(x, c4)
        xc = self.u6c(xc, c3)
        xc = self.u7c(xc, c2)
        xc = self.u8c(xc, c1)
        xc = self.cec(xc)
        xc = F.sigmoid(xc)
        xm = self.u5m(x, c4)
        xm = self.u6m(xm, c3)
        xm = self.u7m(xm, c2)
        xm = self.u8m(xm, c1)
        xm = self.cem(xm)
        xm = F.sigmoid(xm)
        return xs, xc, xm

# Contour aware marker Dilated Unet
class CamDUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 16, dilation=2)
        self.c2 = ConvBlock(16, 32, dilation=2)
        self.c3 = ConvBlock(32, 64, dilation=2)
        self.c4 = ConvBlock(64, 128, dilation=2)
        # bottom dilated conv tunnel
        # bottom conv tunnel
        self.cu = ConvBlock(128, 256, dilation=2)
        # segmentation up conv branch
        self.u5s = ConvUpBlock(256, 128)
        self.u6s = ConvUpBlock(128, 64)
        self.u7s = ConvUpBlock(64, 32)
        self.u8s = ConvUpBlock(32, 16)
        self.ces = nn.Conv2d(16, 1, 1)
        # contour up conv branch
        self.u5c = ConvUpBlock(256, 128)
        self.u6c = ConvUpBlock(128, 64)
        self.u7c = ConvUpBlock(64, 32)
        self.u8c = ConvUpBlock(32, 16)
        self.cec = nn.Conv2d(16, 1, 1)
        # marker up conv branch
        self.u5m = ConvUpBlock(256, 128)
        self.u6m = ConvUpBlock(128, 64)
        self.u7m = ConvUpBlock(64, 32)
        self.u8m = ConvUpBlock(32, 16)
        self.cem = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        _, x = self.cu(x) # no maxpool for U bottom
        xs = self.u5s(x, c4)
        xs = self.u6s(xs, c3)
        xs = self.u7s(xs, c2)
        xs = self.u8s(xs, c1)
        xs = self.ces(xs)
        xs = F.sigmoid(xs)
        xc = self.u5c(x, c4)
        xc = self.u6c(xc, c3)
        xc = self.u7c(xc, c2)
        xc = self.u8c(xc, c1)
        xc = self.cec(xc)
        xc = F.sigmoid(xc)
        xm = self.u5m(x, c4)
        xm = self.u6m(xm, c3)
        xm = self.u7m(xm, c2)
        xm = self.u8m(xm, c1)
        xm = self.cem(xm)
        xm = F.sigmoid(xm)
        return xs, xc, xm


# Shared Contour aware Marker Unet
class SCamUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 16)
        self.c2 = ConvBlock(16, 32)
        self.c3 = ConvBlock(32, 64)
        self.c4 = ConvBlock(64, 128)
        # bottom conv tunnel
        self.cu = ConvBlock(128, 256)
        self.u5 = ConvUpBlock(256, 128)
        self.u6 = ConvUpBlock(128, 64)
        self.u7 = ConvUpBlock(64, 32)
        self.u8 = ConvUpBlock(32, 16)
        self.ce = nn.Conv2d(16, 3, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        _, x = self.cu(x) # no maxpool for U bottom
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.ce(x)
        x = torch.split(x, split_size=1, dim=1) # split 3 channels
        s = F.sigmoid(x[0])
        c = F.sigmoid(x[1])
        m = F.sigmoid(x[2])
        return s, c, m


# Shared Contour aware marker Dilated Unet
class SCamDUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 16, dilation=2)
        self.c2 = ConvBlock(16, 32, dilation=2)
        self.c3 = ConvBlock(32, 64, dilation=2)
        self.c4 = ConvBlock(64, 128, dilation=2)
        # bottom dilated conv tunnel
        # bottom conv tunnel
        self.cu = ConvBlock(128, 256, dilation=2)
        self.u5 = ConvUpBlock(256, 128)
        self.u6 = ConvUpBlock(128, 64)
        self.u7 = ConvUpBlock(64, 32)
        self.u8 = ConvUpBlock(32, 16)
        self.ce = nn.Conv2d(16, 3, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        _, x = self.cu(x) # no maxpool for U bottom
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.ce(x)
        x = torch.split(x, split_size=1, dim=1) # split 3 channels
        s = F.sigmoid(x[0])
        c = F.sigmoid(x[1])
        m = F.sigmoid(x[2])
        return s, c, m


# Transfer Learning VGG16_BatchNorm as Encoder part of UNet
class Vgg_UNet(nn.Module):
    def __init__(self, layers=16, fixed_feature=True):
        super().__init__()
        # load weight of pre-trained resnet
        self.vggnet = models.vgg16_bn(pretrained=True)
        # remove unused classifier submodule
        del self.vggnet.classifier
        self.vggnet.classifier = None
        # fine-tune or extract feature
        if fixed_feature:
            for param in self.vggnet.parameters():
                param.requires_grad = False
        # up conv
        self.u5 = ConvUpBlock(512, 512)
        self.u6 = ConvUpBlock(512, 256)
        self.u7 = ConvUpBlock(256, 128)
        self.u8 = ConvUpBlock(128, 64)
        # final conv tunnel
        self.ce = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # refer https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
        c = []
        for f in self.vggnet.features:
            if isinstance(f, nn.MaxPool2d):
                c.append(x)
            x = f(x)
        assert len(c) == 5
        x = self.u5(c[4], c[3])
        x = self.u6(x, c[2])
        x = self.u7(x, c[1])
        x = self.u8(x, c[0])
        x = self.ce(x)
        x = F.sigmoid(x)
        return x

# Transfer Learning ResNet as Encoder part of UNet
class Res_UNet(nn.Module):
    def __init__(self, layers=34, fixed_feature=True):
        super().__init__()
        # define pre-train model parameters
        if layers == 101:
            builder = models.resnet101
            l = [64, 256, 512, 1024, 2048]
        else:
            builder = models.resnet34
            l = [64, 64, 128, 256, 512]
        # load weight of pre-trained resnet
        self.resnet = builder(pretrained=True)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # up conv
        self.u5 = ConvUpBlock(l[4], l[3])
        self.u6 = ConvUpBlock(l[3], l[2])
        self.u7 = ConvUpBlock(l[2], l[1])
        self.u8 = ConvUpBlock(l[1], l[0])
        # final conv tunnel
        self.ce = nn.ConvTranspose2d(l[0], 1, 2, stride=2)

    def forward(self, x):
        # refer https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = c1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = c2 = self.resnet.layer1(x)
        x = c3 = self.resnet.layer2(x)
        x = c4 = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.ce(x)
        x = F.sigmoid(x)
        return x

# Transfer Learning ResNet as Encoder part of Contour aware Marker Unet
class Res_CamUNet(nn.Module):
    def __init__(self, layers=34, fixed_feature=True):
        super().__init__()
        # define pre-train model parameters
        if layers == 101:
            builder = models.resnet101
            l = [64, 256, 512, 1024, 2048]
        else:
            builder = models.resnet34
            l = [64, 64, 128, 256, 512]
        # load weight of pre-trained resnet
        self.resnet = builder(pretrained=True)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # segmentation up conv branch
        self.u5s = ConvUpBlock(l[4], l[3])
        self.u6s = ConvUpBlock(l[3], l[2])
        self.u7s = ConvUpBlock(l[2], l[1])
        self.u8s = ConvUpBlock(l[1], l[0])
        self.ces = nn.ConvTranspose2d(l[0], 1, 2, stride=2)
        # contour up conv branch
        self.u5c = ConvUpBlock(l[4], l[3])
        self.u6c = ConvUpBlock(l[3], l[2])
        self.u7c = ConvUpBlock(l[2], l[1])
        self.u8c = ConvUpBlock(l[1], l[0])
        self.cec = nn.ConvTranspose2d(l[0], 1, 2, stride=2)
        # marker up conv branch
        self.u5m = ConvUpBlock(l[4], l[3])
        self.u6m = ConvUpBlock(l[3], l[2])
        self.u7m = ConvUpBlock(l[2], l[1])
        self.u8m = ConvUpBlock(l[1], l[0])
        self.cem = nn.ConvTranspose2d(l[0], 1, 2, stride=2)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = c1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = c2 = self.resnet.layer1(x)
        x = c3 = self.resnet.layer2(x)
        x = c4 = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        # segmentation up conv branch
        xs = self.u5s(x, c4)
        xs = self.u6s(xs, c3)
        xs = self.u7s(xs, c2)
        xs = self.u8s(xs, c1)
        xs = self.ces(xs)
        xs = F.sigmoid(xs)
        # contour up conv branch
        xc = self.u5c(x, c4)
        xc = self.u6c(xc, c3)
        xc = self.u7c(xc, c2)
        xc = self.u8c(xc, c1)
        xc = self.cec(xc)
        xc = F.sigmoid(xc)
        # marker up conv branch
        xm = self.u5m(x, c4)
        xm = self.u6m(xm, c3)
        xm = self.u7m(xm, c2)
        xm = self.u8m(xm, c1)
        xm = self.cem(xm)
        xm = F.sigmoid(xm)
        return xs, xc, xm

# Transfer Learning ResNet as Encoder part of Contour aware Marker Unet
class Res_SamUNet(nn.Module):
    def __init__(self, layers=34, fixed_feature=True):
        super().__init__()
        # define pre-train model parameters
        if layers == 101:
            builder = models.resnet101
            l = [64, 256, 512, 1024, 2048]
        else:
            builder = models.resnet34
            l = [64, 64, 128, 256, 512]
        # load weight of pre-trained resnet
        self.resnet = builder(pretrained=True)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # segmentation up conv branch
        self.u5 = ConvUpBlock(l[4], l[3])
        self.u6 = ConvUpBlock(l[3], l[2])
        self.u7 = ConvUpBlock(l[2], l[1])
        self.u8 = ConvUpBlock(l[1], l[0])
        # final conv tunnel
        self.ces = nn.ConvTranspose2d(l[0], 1, 2, stride=2)
        self.cec = nn.ConvTranspose2d(l[0], 1, 2, stride=2)
        self.cem = nn.ConvTranspose2d(l[0], 1, 2, stride=2)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = c1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = c2 = self.resnet.layer1(x)
        x = c3 = self.resnet.layer2(x)
        x = c4 = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        xs = self.ces(x)
        xs = F.sigmoid(xs)
        xc = self.cec(x)
        xc = F.sigmoid(xc)
        xm = self.cem(x)
        xm = F.sigmoid(xm)
        return xs, xc, xm

# Transfer Learning DenseNet as Encoder part of UNet
class Dense_UNet(nn.Module):
    def __init__(self, layers=121, fixed_feature=True):
        super().__init__()
        # define pre-train model parameters
        if layers == 201:
            builder = models.densenet201
            l = [64, 256, 512, 1792, 1920]
        else:
            builder = models.densenet121
            l = [64, 256, 512, 1024, 1024]
        # load weight of pre-trained resnet
        self.densenet = builder(pretrained=True)
        if fixed_feature:
            for param in self.densenet.parameters():
                param.requires_grad = False
        # remove unused classifier submodule
        del self.densenet.classifier
        self.densenet.classifier = None
        # up conv
        self.u5 = ConvUpBlock(l[4], l[3])
        self.u6 = ConvUpBlock(l[3], l[2])
        self.u7 = ConvUpBlock(l[2], l[1])
        self.u8 = ConvUpBlock(l[1], l[0])
        # final conv tunnel
        self.ce = nn.ConvTranspose2d(l[0], 1, 2, stride=2)

    def forward(self, x):
        # refer https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
        c = []
        for f in self.densenet.features:
            if f.__class__.__name__ in ['MaxPool2d', '_Transition']:
                c.append(x)
            x = f(x)
        assert len(c) == 4
        x = self.u5(x, c[3])
        x = self.u6(x, c[2])
        x = self.u7(x, c[1])
        x = self.u8(x, c[0])
        x = self.ce(x)
        x = F.sigmoid(x)
        return x


# Deep Contour Aware Network (DCAN)
class dcanConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_ratio=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Dropout2d(p=dropout_ratio),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class dcanDeConv(nn.Module):
    def __init__(self, in_ch, out_ch, upscale_factor, dropout_ratio=0.2):
        super().__init__()
        self.upscaling = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=upscale_factor, stride=upscale_factor)
        self.conv = dcanConv(out_ch, out_ch, dropout_ratio)

    def forward(self, x):
        x = self.upscaling(x)
        x = self.conv(x)
        return x

class DCAN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = dcanConv(n_channels, 64)
        self.conv2 = dcanConv(64, 128)
        self.conv3 = dcanConv(128, 256)
        self.conv4 = dcanConv(256, 512)
        self.conv5 = dcanConv(512, 512)
        self.conv6 = dcanConv(512, 1024)
        self.deconv3s = dcanDeConv(512, n_classes, 8) # 8 = 2^3 (3 maxpooling)
        self.deconv3c = dcanDeConv(512, n_classes, 8)
        self.deconv2s = dcanDeConv(512, n_classes, 16) # 16 = 2^4 (4 maxpooling)
        self.deconv2c = dcanDeConv(512, n_classes, 16)
        self.deconv1s = dcanDeConv(1024, n_classes, 32) # 32 = 2^5 (5 maxpooling)
        self.deconv1c = dcanDeConv(1024, n_classes, 32)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.maxpool(c1))
        c3 = self.conv3(self.maxpool(c2))
        c4 = self.conv4(self.maxpool(c3))
        # s for segment branch, c for contour branch
        u3s = self.deconv3s(c4)
        u3c = self.deconv3c(c4)
        c5 = self.conv5(self.maxpool(c4))
        u2s = self.deconv2s(c5)
        u2c = self.deconv2c(c5)
        c6 = self.conv6(self.maxpool(c5))
        u1s = self.deconv1s(c6)
        u1c = self.deconv1c(c6)
        outs = F.sigmoid(u1s + u2s + u3s)
        outc = F.sigmoid(u1c + u2c + u3c)
        return outs, outc

class DA_Unet(nn.Module):
    def __init__(self):
        super().__init__()
        # source down conv
        self.c1s = ConvBlock(3, 16)
        self.c2s = ConvBlock(16, 32)
        self.c3s = ConvBlock(32, 64)
        self.c4s = ConvBlock(64, 128)
        # target down conv
        self.c1t = ConvBlock(3, 16)
        self.c2t = ConvBlock(16, 32)
        self.c3t = ConvBlock(32, 64)
        self.c4t = ConvBlock(64, 128)
        # bottom conv tunnel
        self.cu = ConvBlock(128, 256)
        # up conv
        self.u5 = ConvUpBlock(256, 128)
        self.u6 = ConvUpBlock(128, 64)
        self.u7 = ConvUpBlock(64, 32)
        self.u8 = ConvUpBlock(32, 16)
        # final conv tunnel
        self.ce = nn.Conv2d(16, 1, 1)

    def forward_train(self, s, t):
        #source
        s, c1s, _ = self.c1s(s)
        s, c2s, _ = self.c2s(s)
        s, c3s, _ = self.c3s(s)
        s, c4s, _ = self.c4s(s)
        _, s, _ = self.cu(s) # no maxpool for U bottom
        s = self.u5(s, c4s)
        s = self.u6(s, c3s)
        s = self.u7(s, c2s)
        s_f = self.u8(s, c1s)
        s = self.ce(s_f)
        s = F.sigmoid(s)
        #target
        t, c1t, _ = self.c1t(t)
        t, c2t, _ = self.c2t(t)
        t, c3t, _ = self.c3t(t)
        t, c4t, _ = self.c4t(t)
        _, t, _ = self.cu(t) # no maxpool for U bottom
        t = self.u5(t, c4t)
        t = self.u6(t, c3t)
        t = self.u7(t, c2t)
        t_f = self.u8(t, c1t)
        t = self.ce(t_f)
        t = F.sigmoid(t)
        return s, t, s_f, t_f

    def forward_test(self, t):
        #target
        t, c1t, _ = self.c1t(t)
        t, c2t, _ = self.c2t(t)
        t, c3t, _ = self.c3t(t)
        t, c4t, _ = self.c4t(t)
        _, t, _ = self.cu(t) # no maxpool for U bottom
        t = self.u5(t, c4t)
        t = self.u6(t, c3t)
        t = self.u7(t, c2t)
        t = self.u8(t, c1t)
        t = self.ce(t)
        t = F.sigmoid(t)
        return t

    def forward(self, data, mode):
        if mode == 'train':
            return self.forward_train(data[0], data[1])
        else:
            return self.forward_test(data)

class Ynet(nn.Module):
    def __init__(self):
        super().__init__()
        # down conv
        self.c1 = ConvBlock(3, 16)
        self.c2 = ConvBlock(16, 32)
        self.c3 = ConvBlock(32, 64)
        self.c4 = ConvBlock(64, 128)
        # bottom conv tunnel
        self.cu = ConvBlock(128, 256)
        # up conv
        self.u5 = ConvUpBlock(256, 128)
        self.u6 = ConvUpBlock(128, 64)
        self.u7 = ConvUpBlock(64, 32)
        self.u8 = ConvUpBlock(32, 16)
        # final conv tunnel
        self.ce = nn.Conv2d(16, 1, 1)
        # autoencoder up conv
        self.u5d = nn.ConvTranspose2d(256, 128, 3, padding=1)
        self.up5 = nn.MaxUnpool2d(2, stride=2)
        self.u6d = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.up6 = nn.MaxUnpool2d(2, stride=2)
        self.u7d = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.up7 = nn.MaxUnpool2d(2, stride=2)
        self.u8d = nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.up8 = nn.MaxUnpool2d(2, stride=2)
        #self.up9 = nn.MaxUnpool2d(2, stride=1)
        # final output
        self.cea = nn.ConvTranspose2d(16, 3, 3, padding=1)

    def forward_train_step1(self, s, t):
        # source unet forward
        s, c1s, _ = self.c1(s)
        s, c2s, _ = self.c2(s)
        s, c3s, _ = self.c3(s)
        s, c4s, _ = self.c4(s)
        _, s, _ = self.cu(s) # no maxpool for U bottom
        s_u1 = self.u5(s, c4s)
        s_u2 = self.u6(s_u1, c3s)
        s_u3 = self.u7(s_u2, c2s)
        s_u4 = self.u8(s_u3, c1s)
        s_u = self.ce(s_u4)
        s_u = F.sigmoid(s_u)
        # target unet forward
        t, c1t, _ = self.c1(t)
        t, c2t, _ = self.c2(t)
        t, c3t, _ = self.c3(t)
        t, c4t, _ = self.c4(t)
        _, t, _ = self.cu(t) # no maxpool for U bottom
        t_u1 = self.u5(t, c4t)
        t_u2 = self.u6(t_u1, c3t)
        t_u3 = self.u7(t_u2, c2t)
        t_u4 = self.u8(t_u3, c1t)
        t_u = self.ce(t_u4)
        t_u = F.sigmoid(t_u)
        return s_u, [c1s, c2s, c3s, c4s, s, s_u1, s_u2, s_u3, s_u4], [c1t, c2t, c3t, c4t, t, t_u1, t_u2, t_u3, t_u4]

    def forward_train_step2(self, s, t):
        # source unet forward
        s, c1s, inds1 = self.c1(s)
        s, c2s, inds2 = self.c2(s)
        s, c3s, inds3 = self.c3(s)
        s, c4s, inds4 = self.c4(s)
        _, s, _ = self.cu(s) # no maxpool for U bottom
        s_u = self.u5(s, c4s)
        s_u = self.u6(s_u, c3s)
        s_u = self.u7(s_u, c2s)
        s_u = self.u8(s_u, c1s)
        s_u = self.ce(s_u)
        s_u = F.sigmoid(s_u)
        # source reconstruct image
        s_d = self.u5d(s)
        s_d = self.up5(s_d, inds4)
        s_d = self.u6d(s_d)
        s_d = self.up6(s_d, inds3)
        s_d = self.u7d(s_d)
        s_d = self.up7(s_d, inds2)
        s_d = self.u8d(s_d)
        s_d = self.up8(s_d, inds1)
        #s_d = self.up9(s_d)
        s_d = self.cea(s_d)
        s_d = F.sigmoid(s_d)

        # target forward
        t, c1t, indt1 = self.c1(t)
        t, c2t, indt2 = self.c2(t)
        t, c3t, indt3 = self.c3(t)
        t, c4t, indt4 = self.c4(t)
        _, t, _ = self.cu(t) # no maxpool for U bottom
        # target reconstruct image
        t_d = self.u5d(t)
        t_d = self.up5(t_d, indt4)
        t_d = self.u6d(t_d)
        t_d = self.up6(t_d, indt3)
        t_d = self.u7d(t_d)
        t_d = self.up7(t_d, indt2)
        t_d = self.u8d(t_d)
        t_d = self.up8(t_d, indt1)
        #t_d = self.up9(t_d)
        t_d = self.cea(t_d)
        t_d = F.sigmoid(t_d)
        return s_u, s_d, t_d
        
    def forward_train_combine(self, s, t):
        # source unet forward
        s, c1s, inds1 = self.c1(s)
        s, c2s, inds2 = self.c2(s)
        s, c3s, inds3 = self.c3(s)
        s, c4s, inds4 = self.c4(s)
        _, s, _ = self.cu(s) # no maxpool for U bottom
        s_u1 = self.u5(s, c4s)
        s_u2 = self.u6(s_u1, c3s)
        s_u3 = self.u7(s_u2, c2s)
        s_u4 = self.u8(s_u3, c1s)
        s_u = self.ce(s_u4)
        s_u = F.sigmoid(s_u)

        # source reconstruct image
        s_d = self.u5d(s)
        s_d = self.up5(s_d, inds4)
        s_d = self.u6d(s_d)
        s_d = self.up6(s_d, inds3)
        s_d = self.u7d(s_d)
        s_d = self.up7(s_d, inds2)
        s_d = self.u8d(s_d)
        s_d = self.up8(s_d, inds1)
        #s_d = self.up9(s_d)
        s_d = self.cea(s_d)
        s_d = F.sigmoid(s_d)

        # target unet forward
        t, c1t, indt1 = self.c1(t)
        t, c2t, indt2 = self.c2(t)
        t, c3t, indt3 = self.c3(t)
        t, c4t, indt4 = self.c4(t)
        _, t, _ = self.cu(t) # no maxpool for U bottom
        t_u1 = self.u5(t, c4t)
        t_u2 = self.u6(t_u1, c3t)
        t_u3 = self.u7(t_u2, c2t)
        t_u4 = self.u8(t_u3, c1t)
        t_u = self.ce(t_u4)
        t_u = F.sigmoid(t_u)

        # target reconstruct image
        t_d = self.u5d(t)
        t_d = self.up5(t_d, indt4)
        t_d = self.u6d(t_d)
        t_d = self.up6(t_d, indt3)
        t_d = self.u7d(t_d)
        t_d = self.up7(t_d, indt2)
        t_d = self.u8d(t_d)
        t_d = self.up8(t_d, indt1)
        #t_d = self.up9(t_d)
        t_d = self.cea(t_d)
        t_d = F.sigmoid(t_d)
        return s_u, s_d, t_d, [c1s, c2s, c3s, c4s, s, s_u1, s_u2, s_u3, s_u4], [c1t, c2t, c3t, c4t, t, t_u1, t_u2, t_u3, t_u4]

    def forward_test(self, x):
        x, c1, _ = self.c1(x)
        x, c2, _ = self.c2(x)
        x, c3, _ = self.c3(x)
        x, c4, _ = self.c4(x)
        _, x, _ = self.cu(x) # no maxpool for U bottom
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.ce(x)
        x = F.sigmoid(x)
        return x

    def forward(self, data, mode):
        if mode == 'pretrain':
            return self.forward_train_step1(data[0],data[1])
        elif mode == 'train':
            return self.forward_train_step2(data[0],data[1])
        elif mode == 'combine':
            return self.forward_train_combine(data[0],data[1])
        elif mode == 'test' or mode == 'valid':
            return self.forward_test(data)
        else:
            raise NotImplementedError()
            return None

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(model_name='unet'):
    # initialize model
    if model_name == 'unet':
        model = UNet()  # line57
    elif model_name == 'dcan':
        model = DCAN(3, 1)
    elif model_name == 'caunet':
        model = CaUNet()
    elif model_name == 'camunet':
        model = CamUNet()
    elif model_name == 'camdunet':
        model = CamDUNet()
    elif model_name == 'scamunet':
        model = SCamUNet()
    elif model_name == 'scamdunet':
        model = SCamDUNet()
    elif model_name == 'vgg_unet':
        model = Vgg_UNet(16, fixed_feature=True)
    elif model_name == 'res_unet':
        model = Res_UNet(34, fixed_feature=True)
    elif model_name == 'dense_unet':
        model = Dense_UNet(121, fixed_feature=True)
    elif model_name == 'res_camunet':
        model = Res_CamUNet(34, fixed_feature=True)
    elif model_name == 'res_samunet':
        model = Res_SamUNet(34, fixed_feature=True)
    elif model_name == 'da_unet':
        model = DA_Unet()
    elif model_name == 'ynet':
        model = Ynet()
    else:
        raise NotImplementedError()
    return model


if __name__ == '__main__':
    print('Network parameters -')
    for n in ['unet', 'camunet', 'scamunet', 'res_unet', 'res_camunet', 'res_samunet']:
        net = build_model(n)
        #print(net)
        print('\t model {}: {}'.format(n, count_parameters(net)))
        del net

    print("Forward pass sanity check - ")
    for n in ['camunet', 'res_camunet', 'res_samunet']:
        t = time.time()
        net = build_model(n)
        x = torch.randn(1, 3, 256, 256)
        y = net(x)
        #print(x.shape, y.shape)
        del net
        print('\t model {0}: {1:.3f} seconds'.format(n, time.time() - t))

    # x = torch.randn(10, 3, 256, 256)
    # b = ConvBlock(3, 16)
    # p, y = b(x)
    # print(p.shape, y.shape)