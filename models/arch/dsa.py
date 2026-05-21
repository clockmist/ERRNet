import torch
import torch.nn as nn
import torch.nn.functional as F


def _window_partition(x, window_size):
    """x: [B, H, W, C] -> windows: [B*N, M², C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size * window_size, C)
    return windows


def _window_reverse(windows, window_size, H, W):
    """windows: [B*N, M², C] -> x: [B, H, W, C]"""
    B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class CrossWindowAttention(nn.Module):
    """Window-based cross-attention: Q from one stream, K/V from another."""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)

        # relative position bias
        self.rel_pos = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rel_pos, std=0.02)

        # precomputed relative position index
        self.register_buffer("rel_index", self._build_rel_index())

    def _build_rel_index(self):
        M = self.window_size
        coords = torch.arange(M)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords = coords.reshape(2, -1)  # [2, M²]
        rel = coords[:, :, None] - coords[:, None, :]  # [2, M², M²]
        rel = rel[0] + (2 * M - 1) * (rel[1] + M - 1)  # [M², M²]
        return rel

    def forward(self, query, key_value):
        # query, key_value: [B, H, W, C]
        B, H_orig, W_orig, C = query.shape
        M = self.window_size

        # Pad to make H,W divisible by window_size
        pad_h = (M - H_orig % M) % M
        pad_w = (M - W_orig % M) % M
        if pad_h > 0 or pad_w > 0:
            query = F.pad(query, (0, 0, 0, pad_w, 0, pad_h))
            key_value = F.pad(key_value, (0, 0, 0, pad_w, 0, pad_h))

        q_windows = _window_partition(query, M)        # [N, M², C]
        kv_windows = _window_partition(key_value, M)   # [N, M², C]

        Q = self.q_proj(q_windows)
        K, V = self.kv_proj(kv_windows).chunk(2, dim=-1)

        Q = Q.reshape(-1, self.window_size ** 2, self.num_heads,
                      self.head_dim).permute(0, 2, 1, 3)  # [N, h, M², d]
        K = K.reshape(-1, self.window_size ** 2, self.num_heads,
                      self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(-1, self.window_size ** 2, self.num_heads,
                      self.head_dim).permute(0, 2, 1, 3)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_pos[self.rel_index].permute(2, 0, 1)[None]
        attn = F.softmax(attn, dim=-1)

        out = attn @ V
        out = out.permute(0, 2, 1, 3).reshape(-1, M ** 2, self.dim)
        out = self.proj(out)
        out = _window_reverse(out, M, H_orig + pad_h, W_orig + pad_w)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :H_orig, :W_orig, :]

        return out


class DSABlock(nn.Module):
    """One direction of cross-stream interaction: query stream learns from kv stream."""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = CrossWindowAttention(dim, window_size, num_heads)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, query_stream, kv_stream):
        # query_stream, kv_stream: [B, C, H, W]
        q = query_stream.permute(0, 2, 3, 1)   # [B, H, W, C]
        k = kv_stream.permute(0, 2, 3, 1)

        q = q + self.cross_attn(self.norm_q(q), self.norm_kv(k))
        q = q + self.ffn(self.norm_ffn(q))

        return q.permute(0, 3, 1, 2)  # [B, C, H, W]


class DSA(nn.Module):
    """Dual-Stream Cross-Attention: T and R features exchange information.

    Each stream generates queries from its own features and uses the other
    stream's features as keys/values.  This lets T learn what NOT to include
    (reflection patterns) and R learn what NOT to steal (transmission details).
    """

    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()
        self.t_block = DSABlock(dim, window_size, num_heads)
        self.r_block = DSABlock(dim, window_size, num_heads)

    def forward(self, ft, fr):
        ft = self.t_block(ft, fr)  # T cross-attends to R
        fr = self.r_block(fr, ft)  # R cross-attends to T
        return ft, fr
