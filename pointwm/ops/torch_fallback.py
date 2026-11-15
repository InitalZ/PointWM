"""Pure-torch versions of the pointnet2 ops, matching the CUDA semantics.

Two differences from the old fallback in pcv_jepa: FPS starts at index 0
(as the CUDA kernel does) instead of a random index, and ball query returns
the first ``nsample`` in-radius points in index order, padded with the first
one, instead of filling the remainder with out-of-radius points.
"""
from __future__ import annotations

import torch


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """xyz [B, N, 3] -> idx [B, npoint]"""
    b, n, _ = xyz.shape
    xyz = xyz.float()
    idx = torch.zeros(b, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((b, n), 1e10, device=xyz.device)
    farthest = torch.zeros(b, dtype=torch.long, device=xyz.device)
    batch = torch.arange(b, device=xyz.device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)
        distance = torch.minimum(distance, ((xyz - centroid) ** 2).sum(-1))
        farthest = distance.max(dim=-1).indices
    return idx


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features [B, C, N], idx [B, S] -> [B, C, S]"""
    return torch.gather(features, 2, idx.long().unsqueeze(1).expand(-1, features.shape[1], -1))


def ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """xyz [B, N, 3], new_xyz [B, S, 3] -> idx [B, S, nsample]"""
    n = xyz.shape[1]
    d2 = torch.cdist(new_xyz.float(), xyz.float()).pow(2)
    order = torch.arange(n, device=xyz.device).expand_as(d2)
    order = torch.where(d2 < radius**2, order, torch.full_like(order, n))
    idx = order.sort(dim=-1).values[..., :nsample]
    first = idx[..., :1]
    first = torch.where(first == n, torch.zeros_like(first), first)
    return torch.where(idx == n, first.expand_as(idx), idx)


def grouping_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features [B, C, N], idx [B, S, K] -> [B, C, S, K]"""
    b, c, _ = features.shape
    s, k = idx.shape[1], idx.shape[2]
    flat = idx.long().reshape(b, 1, s * k).expand(b, c, s * k)
    return torch.gather(features, 2, flat).view(b, c, s, k)
