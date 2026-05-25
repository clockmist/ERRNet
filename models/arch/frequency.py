"""Frequency Enhancement Module - pluggable, independently detachable."""

import torch
import torch.nn as nn


class FrequencyEnhancementModule(nn.Module):
    """Pluggable frequency-domain enhancement.

    When enabled, applies learned frequency masking to the feature map
    and adds the result back via residual connection.

    When disabled (enabled=False), acts as identity.
    """

    def __init__(self, channels, enabled=True):
        super().__init__()
        self.enabled = enabled
        if not enabled:
            return
        self.mask_real = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.mask_imag = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Tanh(),
        )
        self.post_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        if not self.enabled:
            return x

        x_fft = torch.fft.rfft2(x, norm='ortho')
        real = torch.real(x_fft)
        imag = torch.imag(x_fft)

        real_masked = real * self.mask_real(real)
        imag_masked = imag * self.mask_imag(imag)

        x_fft_masked = torch.complex(real_masked, imag_masked)
        out = torch.fft.irfft2(x_fft_masked, s=(x.size(2), x.size(3)), norm='ortho')
        out = self.post_conv(out)

        return x + out
