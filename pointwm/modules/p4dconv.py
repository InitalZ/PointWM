from __future__ import annotations

import torch
import torch.nn as nn

from pointwm.ops import ball_query, furthest_point_sample, gather_operation, grouping_operation


class P4DConv(nn.Module):
    """P4DConv point-tube convolution (P4Transformer), split into a
    weight-free geometry pass (``gather``) and a weight-dependent feature pass
    (``apply_convs``), so the student and the EMA teacher tokenizers share
    identical FPS anchors and ball-query neighbourhoods.

    The temporal kernel is given by its frame offsets. Pretraining and offline
    fine-tuning use ``(-1, 0, 1)`` (kernel 3, stride 1: token t aggregates
    frames t-1..t+1, Sec. 3.1). Causal online segmentation uses ``(-2, -1, 0)``
    (Sec. 4.4). The three temporal slots share ``conv_d``/``mlp``; the offset
    enters only as the temporal displacement channel, so moving from one set
    of offsets to the other keeps the slot order and every weight, and the
    temporal coordinate shifts with the slot (Supp. Sec. A.3). Frames outside
    the clip repeat the boundary frame.
    """

    def __init__(
        self,
        out_planes: int,
        radius: float,
        nsamples: int,
        spatial_stride: int,
        temporal_offsets=(-1, 0, 1),
        temporal_stride: int = 1,
        temporal_fuse: str = "max",
    ):
        super().__init__()
        self.radius = radius
        self.nsamples = nsamples
        self.spatial_stride = spatial_stride
        self.offsets = tuple(int(o) for o in temporal_offsets)
        self.ts = temporal_stride
        if temporal_fuse not in ("max", "concat"):
            raise ValueError(f"temporal_fuse must be max|concat, got {temporal_fuse!r}")
        self.temporal_fuse = temporal_fuse
        self.conv_d = nn.Sequential(nn.Conv2d(4, out_planes, 1), nn.GELU())
        self.mlp = nn.Sequential(nn.Conv2d(out_planes, out_planes, 1), nn.GELU())
        self.fuse = nn.Conv1d(len(self.offsets) * out_planes, out_planes, 1) if temporal_fuse == "concat" else None

    @torch.no_grad()
    def gather(self, xyzs: torch.Tensor):
        """xyzs [B, T, N, 3] -> (geom, anchor_xyz [B, T', M, 3]).

        ``geom`` holds, per output anchor frame, a list of ``(offset, disp)``
        with ``disp [B, 3, M, k]`` the neighbour displacements in frame t+offset.
        """
        b, t, n, _ = xyzs.shape
        m = n // self.spatial_stride
        xyzs = xyzs.float()
        geom, anchors = [], []
        for tc in range(0, t, self.ts):
            frame = xyzs[:, tc].contiguous()
            aidx = furthest_point_sample(frame, m)
            anchor = gather_operation(frame.transpose(1, 2).contiguous(), aidx).transpose(1, 2).contiguous()
            centre = anchor.transpose(1, 2).unsqueeze(-1)
            per_t = []
            for o in self.offsets:
                src = xyzs[:, min(max(tc + o, 0), t - 1)].contiguous()
                idx = ball_query(self.radius, self.nsamples, src, anchor)
                neigh = grouping_operation(src.transpose(1, 2).contiguous(), idx)
                per_t.append((float(o), neigh - centre))
            geom.append(per_t)
            anchors.append(anchor)
        return geom, torch.stack(anchors, dim=1)

    def apply_convs(self, geom) -> torch.Tensor:
        """-> features [B, T', C, M]."""
        out = []
        for per_t in geom:
            feats = []
            for o, disp in per_t:
                tdisp = torch.full_like(disp[:, :1], o)
                f = self.mlp(self.conv_d(torch.cat([disp, tdisp], dim=1)))
                feats.append(f.max(dim=-1).values)
            stacked = torch.stack(feats, dim=1)
            if self.fuse is None:
                out.append(stacked.max(dim=1).values)
            else:
                b, tk, c, m = stacked.shape
                out.append(self.fuse(stacked.reshape(b, tk * c, m)))
        return torch.stack(out, dim=1)
