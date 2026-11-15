from __future__ import annotations

import torch.nn as nn

from pointwm.modules.pos_embed import XYZTPosEmbed
from pointwm.modules.transformer import Transformer


class TokenEncoder(nn.Module):
    """P4Transformer encoder over (x, y, z, t) tokens: x + PE(u) -> transformer.
    Student: dim 384, depth 6, 6 heads (Table S1)."""

    def __init__(self, dim=384, depth=6, heads=6, mlp_ratio=4.0, dropout=0.0, drop_path_rate=0.1):
        super().__init__()
        self.pos = XYZTPosEmbed(dim)
        self.tr = Transformer(dim, depth, heads, mlp_ratio, dropout, drop_path_rate)

    def forward(self, tokens, xyzt, attn_mask=None):
        return self.tr(tokens + self.pos(xyzt), attn_mask)
