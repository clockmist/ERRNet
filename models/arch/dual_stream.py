import torch
from torch import nn
import torch.nn.functional as F

from .default import ConvLayer, ResidualBlock, PyramidPooling, SELayer


class DualStreamNet(torch.nn.Module):
    """Dual-stream variant of DRNet that outputs both transmission T and reflection R.

    Shares the same encoder architecture as DRNet (conv1-conv2-conv3-res_module-
    deconv1-deconv2-pyramid_module) but replaces the single output head with two
    parallel 1x1 conv heads.  Encoder parameter names are kept identical to DRNet
    so pretrained DRNet weights can be loaded directly.
    """

    def __init__(self, in_channels, out_channels, n_feats, n_resblocks,
                 norm=nn.BatchNorm2d, se_reduction=None, res_scale=1,
                 bottom_kernel_size=3, pyramid=False):
        super(DualStreamNet, self).__init__()

        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)

        self.pyramid_module = None

        # Shared encoder — identical to DRNet
        self.conv1 = ConvLayer(conv, in_channels, n_feats,
                               kernel_size=bottom_kernel_size, stride=1,
                               norm=None, act=act)
        self.conv2 = ConvLayer(conv, n_feats, n_feats,
                               kernel_size=3, stride=1, norm=norm, act=act)
        self.conv3 = ConvLayer(conv, n_feats, n_feats,
                               kernel_size=3, stride=2, norm=norm, act=act)

        dilation_config = [1] * n_resblocks
        self.res_module = nn.Sequential(*[
            ResidualBlock(n_feats, dilation=dilation_config[i],
                          norm=norm, act=act,
                          se_reduction=se_reduction, res_scale=res_scale)
            for i in range(n_resblocks)
        ])

        self.deconv1 = ConvLayer(deconv, n_feats, n_feats,
                                 kernel_size=4, stride=2, padding=1,
                                 norm=norm, act=act)

        self.deconv2 = ConvLayer(conv, n_feats, n_feats,
                                 kernel_size=3, stride=1, norm=norm, act=act)

        if pyramid:
            self.pyramid_module = PyramidPooling(n_feats, n_feats,
                                                 scales=(4, 8, 16, 32),
                                                 ct_channels=n_feats // 4)

        # Dual output heads — 1x1 conv, no norm, no activation
        self.head_T = ConvLayer(conv, n_feats, out_channels,
                                kernel_size=1, stride=1, norm=None, act=None)
        self.head_R = ConvLayer(conv, n_feats, out_channels,
                                kernel_size=1, stride=1, norm=None, act=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_module(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        if self.pyramid_module is not None:
            x = self.pyramid_module(x)

        t = self.head_T(x)
        r = self.head_R(x)

        return t, r


class ResidualModule(nn.Module):
    """Learnable residual Phi(T,R) for modeling non-linear blending.

    Used only during training to relax the strict I = T + R linear assumption.
    Discarded at inference time.
    """

    def __init__(self, in_channels=6, out_channels=3, n_feats=64):
        super(ResidualModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_feats, kernel_size=3,
                               stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, out_channels, kernel_size=3,
                               stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, t, r):
        x = torch.cat([t, r], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return self.tanh(x)
