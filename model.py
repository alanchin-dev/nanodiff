"""
UNet noise predictor for diffusion models (DDPM architecture).

Input:  (B, 3, 32, 32) noisy image  +  (B,) timestep t
Output: (B, 3, 32, 32) predicted noise ε

Architecture (Ho et al. 2020):
  Encoder  32→16→8→4  (channels: C, C, 2C, 2C)
  Bottleneck 4×4       (2C with attention)
  Decoder  4→8→16→32  (mirror encoder, skip connections from encoder)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t, dim):
    """Sinusoidal timestep encoding — same trick as positional encodings in Transformers."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    x = t.float()[:, None] * freqs[None]
    return torch.cat([x.sin(), x.cos()], dim=1)  # (B, dim)


class ResBlock(nn.Module):
    """Conv residual block conditioned on timestep embedding t."""
    def __init__(self, in_ch, out_ch, tdim):
        super().__init__()
        self.norm1  = nn.GroupNorm(8, in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(tdim, out_ch)       # add time signal after first conv
        self.norm2  = nn.GroupNorm(8, out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip   = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t))[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class Attention(nn.Module):
    """Self-attention block. Uses Flash Attention automatically via F.scaled_dot_product_attention."""
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm     = nn.GroupNorm(8, ch)
        self.heads    = heads
        self.head_dim = ch // heads
        self.qkv      = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.proj     = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        def split(t): return t.reshape(B, self.heads, self.head_dim, H * W).transpose(2, 3)
        h = F.scaled_dot_product_attention(split(q), split(k), split(v))
        return x + self.proj(h.transpose(2, 3).reshape(B, C, H, W))


class UNet(nn.Module):
    """
    DDPM UNet for 32×32 RGB images.

    Default: base_ch=128 → ~35M parameters.
    Use base_ch=64 for a smaller model (~9M) that trains faster on CPU/MPS.
    """
    def __init__(self, image_ch=3, base_ch=128):
        super().__init__()
        C    = base_ch
        tdim = C * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(C, tdim), nn.SiLU(), nn.Linear(tdim, tdim)
        )

        # ── Encoder ──────────────────────────────────────────────────────────
        self.init_conv = nn.Conv2d(image_ch, C, 3, padding=1)  # 32×32, C

        self.enc1a = ResBlock(C,   C,   tdim)                  # 32×32, C
        self.enc1b = ResBlock(C,   C,   tdim)
        self.pool1 = nn.Conv2d(C, C, 4, stride=2, padding=1)   # 32 → 16

        self.enc2a = ResBlock(C,   2*C, tdim)                  # 16×16, 2C
        self.enc2b = ResBlock(2*C, 2*C, tdim)
        self.attn2 = Attention(2*C)
        self.pool2 = nn.Conv2d(2*C, 2*C, 4, stride=2, padding=1)  # 16 → 8

        self.enc3a = ResBlock(2*C, 2*C, tdim)                  # 8×8, 2C
        self.enc3b = ResBlock(2*C, 2*C, tdim)
        self.attn3 = Attention(2*C)
        self.pool3 = nn.Conv2d(2*C, 2*C, 4, stride=2, padding=1)  # 8 → 4

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.mid1  = ResBlock(2*C, 2*C, tdim)                  # 4×4, 2C
        self.mid_a = Attention(2*C)
        self.mid2  = ResBlock(2*C, 2*C, tdim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up3   = nn.ConvTranspose2d(2*C, 2*C, 4, stride=2, padding=1)  # 4 → 8
        self.dec3a = ResBlock(4*C, 2*C, tdim)                  # 4C = 2C + 2C skip
        self.dec3b = ResBlock(2*C, 2*C, tdim)
        self.dattn3 = Attention(2*C)

        self.up2   = nn.ConvTranspose2d(2*C, 2*C, 4, stride=2, padding=1)  # 8 → 16
        self.dec2a = ResBlock(4*C, 2*C, tdim)
        self.dec2b = ResBlock(2*C, 2*C, tdim)
        self.dattn2 = Attention(2*C)

        self.up1   = nn.ConvTranspose2d(2*C, C, 4, stride=2, padding=1)    # 16 → 32
        self.dec1a = ResBlock(2*C, C, tdim)                    # 2C = C + C skip
        self.dec1b = ResBlock(C,   C, tdim)

        self.final = nn.Sequential(
            nn.GroupNorm(8, C), nn.SiLU(), nn.Conv2d(C, image_ch, 3, padding=1)
        )

    def forward(self, x, t):
        t = self.time_mlp(sinusoidal_embedding(t, self.time_mlp[0].in_features))

        # Encoder — save skip connections e1, e2, e3
        h      = self.init_conv(x)
        h = e1 = self.enc1b(self.enc1a(h, t), t)
        h      = self.pool1(h)
        h      = self.attn2(self.enc2b(self.enc2a(h, t), t))
        h = e2 = h
        h      = self.pool2(h)
        h      = self.attn3(self.enc3b(self.enc3a(h, t), t))
        h = e3 = h
        h      = self.pool3(h)

        # Bottleneck
        h = self.mid2(self.mid_a(self.mid1(h, t)), t)

        # Decoder — concat encoder skip at each level
        h = self.up3(h)
        h = self.dattn3(self.dec3b(self.dec3a(torch.cat([h, e3], 1), t), t))
        h = self.up2(h)
        h = self.dattn2(self.dec2b(self.dec2a(torch.cat([h, e2], 1), t), t))
        h = self.up1(h)
        h = self.dec1b(self.dec1a(torch.cat([h, e1], 1), t), t)

        return self.final(h)
