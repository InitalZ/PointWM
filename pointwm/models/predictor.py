from __future__ import annotations

import torch
import torch.nn as nn

from pointwm.modules.pos_embed import XYZTPosEmbed
from pointwm.modules.transformer import Transformer


class Predictor(nn.Module):
    """g_phi (Eq. 2): a transformer over encoded context tokens and target
    mask tokens (dim 256, depth 4, 4 heads).

    Context token: in_proj(z_c) + PE_ctx(u_C). Target query: mask token +
    PE_tgt(u~) [+ act_proj(a_t)] [+ e_j]. Outputs the latent prediction
    h^ (out_proj, back to the encoder width) and, with ``position_head``, the
    anchor residual dc^ (Sec. 3.2). ``in_proj``/``out_proj`` are the linear
    projections to and from the width-256 predictor (Supp. Sec. A.4).

    ``num_slots > 0`` builds the learned slot embedding e_j of the time-only
    query ablation (Table 5, row l); the caller then zeroes the query xyz so
    only tau_t enters PE_tgt.
    """

    def __init__(self, dim=384, pred_dim=256, depth=4, heads=4, position_head=False, act_dim=0, num_slots=0):
        super().__init__()
        self.in_proj = nn.Linear(dim, pred_dim)
        self.ctx_pos = XYZTPosEmbed(pred_dim)
        self.tgt_pos = XYZTPosEmbed(pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.tr = Transformer(pred_dim, depth, heads, drop_path_rate=0.1)
        self.out_proj = nn.Linear(pred_dim, dim)
        self.pos_head = nn.Linear(pred_dim, 3) if position_head else None
        self.act_proj = nn.Linear(act_dim, pred_dim, bias=False) if act_dim > 0 else None
        self.slot_embed = None
        if num_slots > 0:
            self.slot_embed = nn.Embedding(num_slots, pred_dim)
            nn.init.trunc_normal_(self.slot_embed.weight, std=0.02)

    def forward(self, ctx, ctx_xyzt, qry_xyzt, act=None, slot_idx=None):
        """ctx [B, S, dim], ctx_xyzt [B, S, 4], qry_xyzt [B, Q, 4],
        act [B, Q, A] or None, slot_idx [B, Q] or None.
        Returns (h^ [B, Q, dim], dc^ [B, Q, 3] or None)."""
        x = self.in_proj(ctx) + self.ctx_pos(ctx_xyzt)
        b, q, _ = qry_xyzt.shape
        qry = self.mask_token.expand(b, q, -1) + self.tgt_pos(qry_xyzt)
        if slot_idx is not None:
            qry = qry + self.slot_embed(slot_idx)
        if act is not None:
            qry = qry + self.act_proj(act.to(qry.dtype))
        y = self.tr(torch.cat([x, qry], dim=1))[:, -q:]
        return self.out_proj(y), (self.pos_head(y) if self.pos_head is not None else None)
