from __future__ import annotations

import torch


def gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B, L, ...], idx [B, S] (flat token indices) -> [B, S, ...]."""
    view = idx.view(idx.shape[0], idx.shape[1], *([1] * (x.dim() - 2)))
    return torch.gather(x, 1, view.expand(idx.shape[0], idx.shape[1], *x.shape[2:]))


def last_seen_frames(visible: torch.Tensor, rule: str = "slot") -> torch.Tensor:
    """For every token (t, j), the nearest preceding visible anchor frame
    t' < t, or -1 if there is none. ``visible [B, T', M]`` is the context set.

    ``rule="slot"``: t' is the latest earlier frame in which slot j itself is
    visible. ``rule="frame"``: t' is the latest earlier frame with all M slots
    visible. Under the causal mask both give t_c = max C for every target.
    The supplement's guarantee (a fully visible frame precedes the earliest
    masked frame) makes both well defined for the block masks.
    """
    if rule not in ("slot", "frame"):
        raise ValueError(f"last_seen rule must be slot|frame, got {rule!r}")
    b, t, m = visible.shape
    vis = visible.all(dim=-1, keepdim=True).expand(b, t, m) if rule == "frame" else visible
    tt = torch.arange(t, device=visible.device).view(1, t, 1).expand(b, t, m)
    upto = torch.where(vis, tt, torch.full_like(tt, -1)).cummax(dim=1).values
    return torch.cat([torch.full_like(upto[:, :1], -1), upto[:, :-1]], dim=1)


def anchors_at(anchor_xyz: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """anchor_xyz [B, T', M, 3], frames [B, T', M] -> c_{frames[b,t,j], j} [B, T', M, 3]."""
    return anchor_xyz.gather(1, frames.clamp_min(0).unsqueeze(-1).expand(*frames.shape, 3))
