import torch
from torch import nn


class FrequencyAwareAttention(nn.Module):
    """Per-channel learnable frequency-domain gating with zero-init residual.

    Applies a learnable complex weight per channel in the frequency domain,
    enabling the network to independently amplify or attenuate the magnitude
    and shift the phase of each channel's frequency components.

    Zero-initialized residual scale ensures the module starts as an identity
    mapping, safe for insertion into pretrained networks.
    """

    def __init__(self, channels):
        super().__init__()
        self.freq_weight = nn.Parameter(torch.ones(channels, dtype=torch.cfloat))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_fft = torch.fft.rfft2(x)
        x_fft = x_fft * self.freq_weight.view(1, -1, 1, 1)
        x_freq = torch.fft.irfft2(x_fft, s=x.shape[-2:])
        return x + self.alpha * x_freq
