# Add your custom network here
from .default import DRNet
import torch.nn as nn


def basenet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, bottom_kernel_size=1, **kwargs)


def errnet(in_channels, out_channels, **kwargs):
    return DRNet(in_channels, out_channels, 256, 13, norm=None, res_scale=0.1, se_reduction=8, bottom_kernel_size=1, pyramid=True, **kwargs)


from .errnet_transformer import ERRNetTransformer

def errnet_transformer(in_channels, out_channels, **kwargs):
    return ERRNetTransformer(in_channels, out_channels, 256, 8, res_scale=0.1, bottom_kernel_size=1, pyramid=True, **kwargs)
