"""Transformer-based backbone for ERRNet, using Restormer-style blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention (from Restormer)."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3, channels * 3,
            kernel_size=3, stride=1, padding=1,
            groups=channels * 3, bias=False,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(b, self.channels, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network (from Restormer)."""

    def __init__(self, channels, expansion_factor=2.66):
        super().__init__()
        hidden = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden * 2, bias=False,
        )
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class TransformerBlock(nn.Module):
    """Transformer block: MDTA + GDFN, with LayerNorm and residual connections."""

    def __init__(self, channels, num_heads=4, expansion_factor=2.66, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor=expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(x_norm) * self.res_scale

        x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.ffn(x_norm) * self.res_scale
        return x


class ERRNetTransformer(nn.Module):
    """Transformer-based reflection removal network.

    Same encode-decode skeleton as DRNet, but replaces ResidualBlocks
    with TransformerBlocks for global context modeling.
    """

    def __init__(
        self, in_channels, out_channels,
        n_feats=256, n_blocks=8, num_heads=4,
        norm=nn.BatchNorm2d, res_scale=1.0,
        bottom_kernel_size=3, pyramid=True,
        se_reduction=None,
    ):
        super().__init__()
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)

        self.conv1 = self._conv_layer(conv, in_channels, n_feats, bottom_kernel_size, 1, norm=None, act=act)
        self.conv2 = self._conv_layer(conv, n_feats, n_feats, 3, 1, norm=norm, act=act)
        self.conv3 = self._conv_layer(conv, n_feats, n_feats, 3, 2, norm=norm, act=act)

        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(n_feats, num_heads=num_heads, res_scale=res_scale)
            for _ in range(n_blocks)
        ])

        self.deconv1 = self._conv_layer(deconv, n_feats, n_feats, 4, 2, padding=1, norm=norm, act=act)
        self.deconv2 = self._conv_layer(conv, n_feats, n_feats, 3, 1, norm=norm, act=act)

        from models.arch.default import PyramidPooling
        self.pyramid = PyramidPooling(n_feats, n_feats, scales=(4, 8, 16, 32), ct_channels=n_feats // 4) if pyramid else None

        self.deconv3 = self._conv_layer(conv, n_feats, out_channels, 1, 1, norm=None, act=None)

    @staticmethod
    def _conv_layer(conv, in_c, out_c, kernel_size, stride, padding=None, norm=None, act=None):
        layers = [conv(in_c, out_c, kernel_size, stride, padding if padding is not None else kernel_size // 2)]
        if norm is not None:
            layers.append(norm(out_c))
        if act is not None:
            layers.append(act)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def forward_features(self, x):
        """Return features before the final conv (for FEM injection)."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.transformer_blocks(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        if self.pyramid is not None:
            x = self.pyramid(x)
        return x

    def forward_head(self, x):
        """Final conv after feature processing."""
        return self.deconv3(x)
