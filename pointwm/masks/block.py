from __future__ import annotations

import torch


def _uniform(lo: float, hi: float) -> float:
    return float(torch.empty(()).uniform_(lo, hi))


def _randint(lo: int, hi: int) -> int:
    """Uniform integer in [lo, hi)."""
    return int(torch.randint(lo, hi, ()))


class BlockMask:
    """Spatiotemporal block mask of the controlled variants (Supp. Sec. A.2).

    The old pcv_jepa block mask (masks/st_block.py) was not packaged in the
    KB, so this is written from the supplement's description:

    * each of ``num_blocks`` = 8 blocks covers a run of round(f_t * T')
      consecutive anchor frames, f_t ~ U[0.25, 0.50];
    * within those frames it covers the round(f_s * M) anchors nearest to a
      random centre anchor, f_s ~ U[0.15, 0.35] (the centre is a random slot
      of a random frame of the block; nearness is to the centre's xyz);
    * the union is truncated, or padded with random tokens, to a fixed mask
      ratio of 0.70, so every clip of a batch has the same number of targets;
    * anchor frame 0 is never masked (blocks start at frame >= 1 and padding
      draws from frames >= 1), so a fully visible frame precedes the
      earliest masked frame and every last-seen query is defined.
    """

    kind = "block"

    def __init__(self, num_blocks=8, temporal_scale=(0.25, 0.50), spatial_scale=(0.15, 0.35), mask_ratio=0.70):
        self.num_blocks = int(num_blocks)
        self.temporal_scale = tuple(float(x) for x in temporal_scale)
        self.spatial_scale = tuple(float(x) for x in spatial_scale)
        self.mask_ratio = float(mask_ratio)

    @torch.no_grad()
    def __call__(self, anchor_xyz: torch.Tensor):
        """anchor_xyz [B, T', M, 3] -> (visible [B, T', M], target [B, T', M]) bool."""
        b, t, m, _ = anchor_xyz.shape
        dev = anchor_xyz.device
        n_tgt = int(round(self.mask_ratio * t * m))
        if t < 2 or n_tgt > (t - 1) * m:
            raise ValueError(f"mask ratio {self.mask_ratio} cannot keep frame 0 visible for T'={t}")
        masked = torch.zeros(b, t, m, dtype=torch.bool, device=dev)
        eligible = torch.arange(t * m, device=dev) >= m
        for i in range(b):
            for _ in range(self.num_blocks):
                lt = min(max(int(round(_uniform(*self.temporal_scale) * t)), 1), t - 1)
                ts = _randint(1, t - lt + 1)
                ns = min(max(int(round(_uniform(*self.spatial_scale) * m)), 1), m)
                centre = anchor_xyz[i, ts + _randint(0, lt), _randint(0, m)]
                d = (anchor_xyz[i, ts : ts + lt].float() - centre.float()).norm(dim=-1)
                nearest = d.topk(ns, dim=-1, largest=False).indices
                masked[i, ts : ts + lt].scatter_(1, nearest, True)
            flat = masked[i].view(-1)
            cur = int(flat.sum())
            if cur > n_tgt:
                on = flat.nonzero().squeeze(1)
                flat[on[torch.randperm(on.numel(), device=dev)[: cur - n_tgt]]] = False
            elif cur < n_tgt:
                off = (~flat & eligible).nonzero().squeeze(1)
                flat[off[torch.randperm(off.numel(), device=dev)[: n_tgt - cur]]] = True
        return ~masked, masked
