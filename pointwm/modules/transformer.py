from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth, per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = torch.empty((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x.div(keep) * mask


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.heads = heads
        self.attn_drop = dropout
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(
            qkv[0], qkv[1], qkv[2], attn_mask=attn_mask, dropout_p=self.attn_drop if self.training else 0.0
        )
        x = x + self.drop_path(self.proj(attn.permute(0, 2, 1, 3).reshape(b, n, c)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


def _init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio=4.0, dropout=0.0, drop_path_rate=0.0):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, dropout, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``attn_mask`` (bool, True = may attend) is only used by causal
        online segmentation (Sec. 4.4); pretraining attends everywhere."""
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.norm(x)


def frame_causal_mask(tprime: int, nprime: int, device) -> torch.Tensor:
    """Bool [T'N', T'N'] mask for frame-major token order: a token of anchor t
    attends to every token of anchors <= t (Sec. 4.4, causal attention)."""
    f = torch.arange(tprime, device=device).repeat_interleave(nprime)
    return f.view(-1, 1) >= f.view(1, -1)
