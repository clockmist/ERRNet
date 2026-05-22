import torch
import torch.nn as nn
import torch.nn.functional as F

from .default import PyramidPooling, ConvLayer as _ConvLayer


# ---------------------------------------------------------------------------
# Haar Wavelet Decomposition
# ---------------------------------------------------------------------------

def _make_haar_kernels(device):
    """4 Haar 2D decomposition filters: LL, LH, HL, HH. Stride=2."""
    h_LL = torch.tensor([[1., 1.], [1., 1.]], device=device) / 4.0
    h_LH = torch.tensor([[-1., 1.], [-1., 1.]], device=device) / 4.0
    h_HL = torch.tensor([[-1., -1.], [1., 1.]], device=device) / 4.0
    h_HH = torch.tensor([[1., -1.], [-1., 1.]], device=device) / 4.0
    return h_LL, h_LH, h_HL, h_HH


class HaarWaveletDecomp(nn.Module):
    """Fixed Haar wavelet input decomposition. 3ch RGB -> 12ch (3x4 subbands)."""
    def __init__(self):
        super(HaarWaveletDecomp, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        device = x.device
        kernels = _make_haar_kernels(device)
        out = []
        for c in range(x.size(1)):
            x_c = x[:, c:c + 1, :, :]
            for k in kernels:
                weight = k.view(1, 1, 2, 2).repeat(1, 1, 1, 1)
                sub = F.conv2d(x_c, weight, stride=2)
                out.append(sub)
        return torch.cat(out, dim=1)


# ---------------------------------------------------------------------------
# CBAM (Convolutional Block Attention Module)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(pooled))


class CBAM(nn.Module):
    """Sequential channel + spatial attention."""
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ---------------------------------------------------------------------------
# CBAM Residual Block
# ---------------------------------------------------------------------------

class CBAMResBlock(nn.Module):
    def __init__(self, channels, norm=None, act=nn.ReLU(True), res_scale=0.1):
        super(CBAMResBlock, self).__init__()
        conv = nn.Conv2d
        self.conv1 = _ConvLayer(conv, channels, channels, 3, 1, norm=norm, act=act)
        self.conv2 = _ConvLayer(conv, channels, channels, 3, 1, norm=norm, act=None)
        self.cbam = CBAM(channels)
        self.res_scale = res_scale

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.cbam(out)
        out = out * self.res_scale
        return out + residual


# ---------------------------------------------------------------------------
# WEGNet
# ---------------------------------------------------------------------------

class WEGNet(nn.Module):
    """
    Wavelet-Enhanced Gradient-guided Network for SIRR.
    Inputs:
      in_channels  -- total after wavelet + VGG HyperColumn merge (1728 typical)
      out_channels -- output channels (3 for RGB)
    """

    def __init__(self, in_channels, out_channels, n_feats=256, n_resblocks=16,
                 norm=None, res_scale=0.1, pyramid=True):
        super(WEGNet, self).__init__()
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)

        # Haar decomposition: 3ch RGB -> 12ch @ H/2,W/2
        self.haar = HaarWaveletDecomp()

        # Project wavelet output to n_feats
        self.wavelet_proj = _ConvLayer(conv, 12, n_feats, 1, 1, norm=None, act=act)

        # Merge wavelet + VGG HyperColumn and project to n_feats
        # VGG HyperColumn is 1472ch, concatenated externally
        self.fusion = _ConvLayer(conv, in_channels, n_feats, 1, 1, norm=None, act=act)

        # Encoder at H/2,W/2 resolution
        self.enc_conv = _ConvLayer(conv, n_feats, n_feats, 3, 1, norm=norm, act=act)

        # Stack of CBAM residual blocks
        self.res_blocks = nn.Sequential(*[
            CBAMResBlock(n_feats, norm=norm, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)
        ])

        # Upsample back to H,W
        self.deconv = _ConvLayer(deconv, n_feats, n_feats, 4, 2, padding=1, norm=norm, act=act)

        # Output head
        self.out_conv = _ConvLayer(conv, n_feats, n_feats, 3, 1, norm=norm, act=act)

        self.pyramid = None
        if pyramid:
            self.pyramid = PyramidPooling(n_feats, n_feats, scales=(4, 8, 16, 32), ct_channels=n_feats // 4)

        self.out_proj = _ConvLayer(conv, n_feats, out_channels, 1, 1, norm=None, act=None)

    def forward(self, x):
        """
        x: concatenated [Haar-projected wavelet, VGG HyperColumn] at H/2,W/2
           OR just wavelet features if VGG is handled externally.
        This is called from ERRNetModel.forward() where:
          - Haar is applied to input image
          - VGG HyperColumn features are interpolated to H/2,W/2
          - Both are concatenated and passed here
        """
        x = self.fusion(x)
        x = self.enc_conv(x)
        x = self.res_blocks(x)
        x = self.deconv(x)
        x = self.out_conv(x)
        if self.pyramid is not None:
            x = self.pyramid(x)
        x = self.out_proj(x)
        return x
