from __future__ import annotations

import numpy as np
import torch


def sample_t0(tprime: int, num_targets: int, gap: int, min_context: int, rng: np.random.Generator) -> int:
    """Starting target anchor t0, uniform over the integers in
    [min_context + g, T' - K] (Supp. Sec. A.2, zero-based)."""
    lo, hi = min_context + gap, tprime - num_targets
    if lo > hi:
        raise ValueError(f"no valid t0 for T'={tprime}, K={num_targets}, g={gap}, min context {min_context}")
    return int(rng.integers(lo, hi + 1))


class CausalMask:
    """Causal temporal mask of Sec. 3.1. Targets T = {t0, ..., t0+K-1}, all M
    spatial tokens of each; context C = {t : t < t0 - g}. The g anchors in
    between belong to neither set. Defaults K=6, g=1, at least 12 context
    anchors. t0 is shared by all clips of a batch."""

    kind = "causal"

    def __init__(self, num_targets: int = 6, gap: int = 1, min_context: int = 12):
        self.num_targets = int(num_targets)
        self.gap = int(gap)
        self.min_context = int(min_context)

    def sample_t0(self, tprime: int, rng: np.random.Generator) -> int:
        return sample_t0(tprime, self.num_targets, self.gap, self.min_context, rng)

    def __call__(self, b: int, tprime: int, nprime: int, t0: int, device):
        """-> (visible [B, T', M], target [B, T', M]) bool."""
        if not self.min_context + self.gap <= t0 <= tprime - self.num_targets:
            raise ValueError(f"t0={t0} outside the valid range for T'={tprime}")
        t = torch.arange(tprime, device=device).view(1, tprime, 1).expand(b, tprime, nprime)
        visible = t < t0 - self.gap
        target = (t >= t0) & (t < t0 + self.num_targets)
        return visible, target
