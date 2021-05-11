import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class SimpleBlock(nn.Module):

    def __init__(self, depth=3, n_channels=64, in_nc=3, out_nc=64, kernel_size=3, padding=1, bias=True):
        super(SimpleBlock, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_nc, n_channels, kernel_size=(kernel_size, kernel_size), padding=(padding, padding), bias=bias)
        )
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(n_channels, n_channels, kernel_size=(kernel_size, kernel_size),
                          padding=(padding, padding), bias=bias)
            )
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        layers.append(
            nn.Conv2d(n_channels, out_nc, kernel_size=(kernel_size, kernel_size),
                      padding=(padding, padding), bias=bias)
        )
        self.simple_block = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.simple_block(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


@ARCH_REGISTRY.register()
class UNet(nn.Module):

    def __init__(self, depth=2, nf=64, in_nc=1, out_nc=1):
        super(UNet, self).__init__()

        # encoder
        self.conv_block_s1 = SimpleBlock(depth=depth, n_channels=1 * nf, in_nc=in_nc,
                                         out_nc=nf, kernel_size=3)
        self.pool1 = nn.Conv2d(nf, 2 * nf, (3, 3), (2, 2), (1, 1), bias=True)

        self.conv_block_s2 = SimpleBlock(depth=depth, n_channels=2 * nf, in_nc=2 * nf,
                                         out_nc=2 * nf, kernel_size=3)
        self.pool2 = nn.Conv2d(2 * nf, 4 * nf, (3, 3), (2, 2), (1, 1), bias=True)

        self.conv_block_s3 = SimpleBlock(depth=depth, n_channels=4 * nf, in_nc=4 * nf,
                                         out_nc=4 * nf, kernel_size=3)

        # decoder
        self.up1 = nn.ConvTranspose2d(4 * nf, 2 * nf, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True)
        # cat with conv_block_s4 (256, H/2, W/2)
        self.conv_block_s4 = SimpleBlock(depth=depth, n_channels=2 * nf, in_nc=4 * nf,
                                         out_nc=2 * nf, kernel_size=3)

        self.up2 = nn.ConvTranspose2d(2 * nf, 1 * nf, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True)
        # cat with conv_block_s3 (128, H/1, W/1)
        self.conv_block_s5 = SimpleBlock(depth=depth, n_channels=nf, in_nc=2 * nf,
                                         out_nc=out_nc, kernel_size=3)
        '''
        # encoder
        self.conv_block_s1 = SimpleBlock(depth=5, n_channels=nf, in_nc=in_nc, \
            out_nc=nf, kernel_size=3) # 32, H, W
        self.pool1 = nn.Conv2d(nf, 2*nf, 3, 2, 1, bias=True) # 64 

        self.conv_block_s2 = SimpleBlock(depth=5, n_channels=2*nf, in_nc=2*nf, \
            out_nc=2*nf, kernel_size=3) # 64, H//2, W//2
        self.pool2 = nn.Conv2d(2*nf, 4*nf, 3, 2, 1, bias=True) # 128

        self.conv_block_s3 = SimpleBlock(depth=5, n_channels=4*nf, in_nc=4*nf, \
            out_nc=4*nf, kernel_size=3) # 128, H//4, W//4
        self.pool3 = nn.Conv2d(4*nf, 8*nf, 3, 2, 1, bias=True) # 256

        self.conv_block_s4 = SimpleBlock(depth=2, n_channels=8*nf, in_nc=8*nf, \
            out_nc=8*nf, kernel_size=3) # 256, H//8, W//8
        self.pool4 = nn.Conv2d(8*nf, 16*nf, 3, 2, 1, bias=True) # 512

        self.conv_block_s5 = SimpleBlock(depth=2, n_channels=16*nf, in_nc=16*nf, \
            out_nc=16*nf, kernel_size=3) # 512, H//16, W//16
        # decoder
        self.up1 = nn.ConvTranspose2d(in_nc=16*nf, out_nc=8*nf,\
            kernel_size=2, stride=2, padding=0, bias=True) # 256, H//8, W//8
        # cat with conv_block_s4 # 512, H//8, W//8
        self.conv_block_s6 = SimpleBlock(depth=2, n_channels=8*nf, in_nc=16*nf, \
            out_nc=8*nf, kernel_size=3) # 256, H//8, W//8

        self.up2 = nn.ConvTranspose2d(in_nc=8*nf, out_nc=4*nf,\
            kernel_size=2, stride=2, padding=0, bias=True) # 128, H//4, W//4
        # cat with conv_block_s3 # 256, H//4, W//4
        self.conv_block_s7 = SimpleBlock(depth=2, n_channels=4*nf, in_nc=8*nf, \
            out_nc=4*nf, kernel_size=3) # 128, H//4, W//4

        self.up3 = nn.ConvTranspose2d(in_nc=4*nf, out_nc=2*nf,\
            kernel_size=2, stride=2, padding=0, bias=True) # 64, H//2, W//2
        # cat with conv_block_s2 # 128, H//2, W//2
        self.conv_block_s8 = SimpleBlock(depth=2, n_channels=2*nf, in_nc=4*nf, \
            out_nc=2*nf, kernel_size=3) # 64, H//2, W//2

        self.up4 = nn.ConvTranspose2d(in_nc=2*nf, out_nc=nf,\
            kernel_size=2, stride=2, padding=0, bias=True) # 32, H, W
        # cat with conv_block_s1 # 64, H, W
        self.conv_block_s9 = SimpleBlock(depth=2, n_channels=nf, in_nc=2*nf, \
            out_nc=out_nc, kernel_size=3) # 64, H//2, W//2
        '''

    def forward(self, x):

        # encoder
        x_s1 = self.conv_block_s1(x)     # 064, H/1, W/1
        x_s2 = self.pool1(x_s1)          # 128, H/2, W/2
        x_s2 = self.conv_block_s2(x_s2)  # 128, H/2, W/2
        x_s3 = self.pool2(x_s2)          # 256, H/4, W/4
        x_s3 = self.conv_block_s3(x_s3)  # 256, H/4, W/4

        # decoder
        out = self.up1(x_s3)             # 128, H/2, W/2
        out = torch.cat((out, x_s2), 1)  # 256, H/2, W/2
        out = self.conv_block_s4(out)    # 128, H/2, W/2
        out = self.up2(out)              # 064, H/1, W/1
        out = torch.cat((out, x_s1), 1)  # 128, H/1, W/1
        out = self.conv_block_s5(out)    # out, H/1, W/1

        '''
        # encoder
        x_s1 = self.conv_block_s1(x)          # 032, H, W
        x_s2 = self.pool1(x_s1)               # 064, H//2, W//2
        x_s2 = self.conv_block_s2(x_s2)       # 064, H//2, W//2
        x_s3 = self.pool2(x_s2)               # 128, H//4, W//4
        x_s3 = self.conv_block_s3(x_s3)       # 128, H//4, W//4
        x_s4 = self.pool3(x_s3)               # 256, H//8, W//8
        x_s4 = self.conv_block_s4(x_s4)       # 256, H//8, W//8
        x_s5 = self.pool4(x_s4)               # 512, H//16, W//16
        x_s5 = self.conv_block_s5(x_s5)       # 512, H//16, W//16

        # decoder
        out = self.up1(x_s5)                 # 256, H//8, W//8
        out = torch.cat((out, x_s4), 1)      # 512, H//8, W//8
        out = self.conv_block_s6(out)        # 256, H//8, W//8
        out = self.up2(out)                  # 128, H//4, W//4
        out = torch.cat((out, x_s3), 1)      # 256, H//4, W//4
        out = self.conv_block_s7(out)        # 128, H//4, W//4
        out = self.up3(out)                  # 064, H//2, W//2
        out = torch.cat((out, x_s2), 1)      # 128, H//2, W//2
        out = self.conv_block_s8(out)        # 064, H//2, W//2
        out = self.up4(out)                  # 032, H, W
        out = torch.cat((out, x_s1), 1)      # 064, H, W
        out = self.conv_block_s9(out)        # out, H, W
        '''

        return out