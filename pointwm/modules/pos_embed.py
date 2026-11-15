from __future__ import annotations

import torch
import torch.nn as nn


class XYZTPosEmbed(nn.Module):
    """Positional embedding of u = [x, y, z, tau]: a two-layer pointwise MLP.

    It is an MLP of the coordinates rather than a lookup table, so a 150-frame
    input (more anchors in the same tau range [-1, 1]) reuses it without
    interpolation (Supp. Sec. A.3).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv1d(4, dim, 1), nn.GELU(), nn.Conv1d(dim, dim, 1))
        for m in self.proj.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, xyzt: torch.Tensor) -> torch.Tensor:
        return self.proj(xyzt.transpose(1, 2)).transpose(1, 2)
