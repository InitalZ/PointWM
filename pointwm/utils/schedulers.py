from __future__ import annotations

import math


class WarmupCosine:
    """Linear warmup from 0 to ``peak`` over ``warmup`` steps, then cosine
    decay to ``end`` at ``total`` steps. ``step()`` returns the value for the
    next optimizer step (call it before ``optimizer.step()``)."""

    def __init__(self, peak, end, total, warmup=0):
        self.peak, self.end = peak, end
        self.total, self.warmup = max(1, total), warmup
        self.step_id = 0

    def step(self):
        self.step_id += 1
        if self.step_id <= self.warmup:
            return self.peak * self.step_id / max(1, self.warmup)
        p = min(1.0, (self.step_id - self.warmup) / max(1, self.total - self.warmup))
        return self.end + 0.5 * (self.peak - self.end) * (1.0 + math.cos(math.pi * p))


class EMAMomentum:
    """EMA momentum scheduled from ``start`` (0.996) to ``end`` (1.0) with a
    cosine ramp over training (Sec. 3.1)."""

    def __init__(self, start, end, total):
        self.start, self.end, self.total = start, end, max(1, total)
        self.step_id = 0

    def step(self):
        p = min(1.0, self.step_id / self.total)
        self.step_id += 1
        return self.start + (self.end - self.start) * 0.5 * (1.0 - math.cos(math.pi * p))
