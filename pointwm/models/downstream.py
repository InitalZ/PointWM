"""Downstream models: only the student tokenizer and encoder are kept
(Sec. 3.4), plus a task head. Module names match PointWM's student
(``tokenizer.*``, ``encoder.*``), so a pretraining checkpoint loads by name."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointwm.models.encoder import TokenEncoder
from pointwm.models.tokenizer import PointTubeTokenizer
from pointwm.modules.transformer import frame_causal_mask


class FrameSegHead(nn.Module):
    """Per-frame MLP on the mean of the spatial tokens of each frame
    (Supp. Sec. A.3). The MLP is the old FrameSegHead funnel without TCN
    stages (LN -> 256 -> 128 -> K, ReLU, dropout 0.3); the old final configs
    used a 4-stage MS-TCN decoder, which the paper does not describe."""

    def __init__(self, dim=384, num_classes=19, dropout=0.3):
        super().__init__()
        self.cls = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, tokens, tprime, nprime, out_t):
        b, _, c = tokens.shape
        x = tokens.view(b, tprime, nprime, c).mean(dim=2)
        if out_t != tprime:
            x = F.interpolate(x.transpose(1, 2), size=out_t, mode="linear", align_corners=True).transpose(1, 2)
        return self.cls(x)


class SegModel(nn.Module):
    """HOI4D action segmentation. ``causal=True`` is the online protocol
    (Sec. 4.4): causal P4DConv kernel over frames t-2..t and a causal
    attention mask, so the label of frame t depends only on frames <= t."""

    def __init__(self, dim=384, depth=6, heads=6, num_classes=19, dropout=0.3, causal=False, tokenizer=None):
        super().__init__()
        self.causal = bool(causal)
        self.tokenizer = PointTubeTokenizer(dim, causal=self.causal, **(tokenizer or {}))
        self.encoder = TokenEncoder(dim=dim, depth=depth, heads=heads)
        self.head = FrameSegHead(dim, num_classes, dropout)
        self._mask_cache = {}

    def _mask(self, tprime, nprime, device):
        key = (tprime, nprime, str(device))
        if key not in self._mask_cache:
            self._mask_cache[key] = frame_causal_mask(tprime, nprime, device)
        return self._mask_cache[key]

    def forward(self, clip):
        tok = self.tokenizer(clip)
        mask = self._mask(tok["T"], tok["M"], clip.device) if self.causal else None
        z = self.encoder(tok["tokens"], tok["xyzt"], attn_mask=mask)
        out_t = (tok["T"] - 1) * self.tokenizer.p4d.ts + 1
        return self.head(z, tok["T"], tok["M"], out_t)


class ClsModel(nn.Module):
    """MSR-Action3D recognition: max-pool over all tokens, then LayerNorm and
    an MLP (width 512, GELU, dropout 0.5) (Supp. Sec. A.3)."""

    def __init__(self, dim=384, depth=6, heads=6, num_classes=20, mlp_dim=512, dropout=0.5, tokenizer=None):
        super().__init__()
        self.tokenizer = PointTubeTokenizer(dim, **(tokenizer or {}))
        self.encoder = TokenEncoder(dim=dim, depth=depth, heads=heads)
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, clip):
        tok = self.tokenizer(clip)
        z = self.encoder(tok["tokens"], tok["xyzt"])
        return self.head(z.max(dim=1).values)


def load_backbone(model: nn.Module, ckpt_path: str) -> dict:
    """Load the student ``tokenizer.*`` / ``encoder.*`` weights of a PointWM
    checkpoint. The teacher, predictor, IDM and prior are dropped. Raises if
    any backbone tensor is missing or mis-shaped."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k: v for k, v in ck["model"].items() if k.startswith(("tokenizer.", "encoder."))}
    missing, unexpected = model.load_state_dict(state, strict=False)
    stray = [k for k in missing if not k.startswith("head.")]
    if stray or unexpected:
        raise ValueError(f"{ckpt_path}: missing backbone tensors {stray[:5]}, unexpected {unexpected[:5]}")
    return {"loaded": len(state), "pretrain_config": ck.get("config")}
