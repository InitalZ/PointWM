from __future__ import annotations

import torch
import torch.nn as nn

from pointwm.modules.p4dconv import P4DConv


def anchor_times(tprime: int, device, dtype=torch.float32) -> torch.Tensor:
    """tau_t: the anchor index rescaled linearly to [-1, 1] over the anchors
    of the input (Supp. Sec. A.3), shape [T']."""
    return torch.arange(tprime, device=device, dtype=dtype) / max(tprime - 1, 1) * 2.0 - 1.0


class PointTubeTokenizer(nn.Module):
    """FPS + spatiotemporal ball query + P4DConv -> tokens x_{t,j}, anchors
    c_{t,j} and positions u_{t,j} = [c_{t,j}; tau_t] (Sec. 3.1).

    Defaults are Table S1: 64 FPS anchors from 2048 points (spatial stride
    32), ball-query radius 0.05 with 32 neighbours, temporal kernel 3 and
    stride 1, boundary frames repeated so that T' equals the number of input
    frames. ``causal=True`` gives the online kernel over frames t-2..t.
    """

    def __init__(
        self,
        dim: int,
        radius: float = 0.05,
        nsamples: int = 32,
        spatial_stride: int = 32,
        temporal_kernel: int = 3,
        temporal_stride: int = 1,
        causal: bool = False,
        temporal_fuse: str = "max",
    ):
        super().__init__()
        if causal:
            offsets = tuple(range(-(temporal_kernel - 1), 1))
        else:
            if temporal_kernel % 2 != 1:
                raise ValueError("a centred temporal kernel needs an odd size")
            offsets = tuple(range(-(temporal_kernel // 2), temporal_kernel // 2 + 1))
        self.p4d = P4DConv(dim, radius, nsamples, spatial_stride, offsets, temporal_stride, temporal_fuse)

    def gather(self, clip: torch.Tensor):
        """Weight-free geometry: clip [B, T, N, 3] -> (geom, anchor_xyz [B, T', M, 3])."""
        return self.p4d.gather(clip)

    def assemble(self, geom, anchor_xyz: torch.Tensor) -> dict:
        feat = self.p4d.apply_convs(geom).permute(0, 1, 3, 2)
        b, t, m, c = feat.shape
        tau = anchor_times(t, anchor_xyz.device, anchor_xyz.dtype).view(1, t, 1, 1).expand(b, t, m, 1)
        xyzt = torch.cat([anchor_xyz, tau], dim=-1).reshape(b, t * m, 4)
        return {"tokens": feat.reshape(b, t * m, c), "xyzt": xyzt, "anchor_xyz": anchor_xyz, "T": t, "M": m}

    def forward(self, clip: torch.Tensor) -> dict:
        geom, anchor_xyz = self.gather(clip)
        return self.assemble(geom, anchor_xyz)
