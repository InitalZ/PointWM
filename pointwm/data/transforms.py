"""Clip normalisation and augmentation (Supp. Sec. A.1).

Normalisation: centre by the clip-wide centroid, scale by the clip-wide
maximum radius. Augmentation of the normalised clip: one isotropic scale
factor from [0.9, 1.1], a rotation about the vertical (y) axis from
[-15, 15] degrees, per-point Gaussian jitter sigma 0.005 clipped to +-0.02.

The torch versions serve the context-only normalisation ablation (Table S9),
whose statistics depend on t0 and are therefore applied on the GPU after t0
has been drawn.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def resample_points(p: np.ndarray, num_points: int, rng) -> np.ndarray:
    """Resample one frame to ``num_points`` points (without replacement when
    possible, otherwise by repetition)."""
    n = p.shape[0]
    if n >= num_points:
        sel = rng.choice(n, num_points, replace=False)
    else:
        rep, rem = divmod(num_points, n)
        sel = np.concatenate([np.arange(n)] * rep + [rng.choice(n, rem, replace=False)])
    return p[sel]


def clip_normalize(clip: np.ndarray, upto: int | None = None) -> np.ndarray:
    """clip [T, N, 3]; statistics from frames [0, upto) (all frames if None)."""
    pc = (clip if upto is None else clip[:upto]).reshape(-1, 3)
    centroid = pc.mean(axis=0)
    scale = np.sqrt(((pc - centroid) ** 2).sum(axis=1)).max() + 1e-8
    return (clip - centroid) / scale


def random_scale(clip, rng, low=0.9, high=1.1):
    return clip * np.float32(rng.uniform(low, high))


def random_yaw(clip, rng, max_deg=15.0):
    rad = np.deg2rad(rng.uniform(-max_deg, max_deg))
    c, s = np.cos(rad), np.sin(rad)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return clip @ rot.T


def jitter(clip, rng, sigma=0.005, clip_val=0.02):
    return clip + np.clip(rng.standard_normal(clip.shape) * sigma, -clip_val, clip_val).astype(np.float32)


def augment(clip, rng):
    return jitter(random_yaw(random_scale(clip, rng), rng), rng)


def clip_normalize_torch(clip: torch.Tensor, upto: int | None = None) -> torch.Tensor:
    src = clip if upto is None else clip[:, :upto]
    pc = src.reshape(clip.shape[0], -1, 3)
    centroid = pc.mean(dim=1, keepdim=True)
    scale = (pc - centroid).norm(dim=-1).amax(dim=1).add(1e-8).view(-1, 1, 1, 1)
    return (clip - centroid.unsqueeze(1)) / scale


def augment_torch(clip: torch.Tensor, max_deg=15.0, low=0.9, high=1.1, sigma=0.005, clip_val=0.02) -> torch.Tensor:
    b, dev = clip.shape[0], clip.device
    s = torch.empty(b, device=dev).uniform_(low, high).view(b, 1, 1, 1)
    rad = torch.empty(b, device=dev).uniform_(-max_deg, max_deg) * (math.pi / 180.0)
    c, sn = rad.cos(), rad.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    rot = torch.stack([torch.stack([c, z, sn], -1), torch.stack([z, o, z], -1), torch.stack([-sn, z, c], -1)], -2)
    out = torch.einsum("btnj,bij->btni", clip * s, rot)
    return out + (torch.randn_like(out) * sigma).clamp(-clip_val, clip_val)
