"""Segmentation metrics (Supp. Sec. A.3) and representation metrics
(Supp. Sec. B.2)."""
from __future__ import annotations

import numpy as np
import torch


def segments(arr):
    out, s = [], 0
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            out.append((arr[s], s, i - 1))
            s = i
    out.append((arr[s], s, len(arr) - 1))
    return out


def frame_acc(pred, gt) -> float:
    return 100.0 * float((np.asarray(pred) == np.asarray(gt)).mean())


def edit_score(pred, gt) -> float:
    """Segmental edit score: 100 * (1 - Levenshtein / max(#segments))."""
    p = [s[0] for s in segments(pred)]
    g = [s[0] for s in segments(gt)]
    dp = np.zeros((len(p) + 1, len(g) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(p) + 1)
    dp[0, :] = np.arange(len(g) + 1)
    for i in range(1, len(p) + 1):
        for j in range(1, len(g) + 1):
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + (p[i - 1] != g[j - 1]))
    return 100.0 * (1.0 - dp[-1, -1] / max(len(p), len(g), 1))


def f1_at_k(pred, gt, thr: float) -> float:
    """A predicted segment is a hit when its IoU with an unmatched
    ground-truth segment of the same class reaches ``thr``."""
    pseg, gseg = segments(pred), segments(gt)
    used, hits = set(), 0
    for pc, ps, pe in pseg:
        best, best_iou = -1, 0.0
        for i, (gc, gs, ge) in enumerate(gseg):
            if i in used or gc != pc:
                continue
            inter = max(0, min(pe, ge) - max(ps, gs) + 1)
            iou = inter / ((pe - ps + 1) + (ge - gs + 1) - inter)
            if iou > best_iou:
                best_iou, best = iou, i
        if best >= 0 and best_iou >= thr:
            used.add(best)
            hits += 1
    prec = hits / max(len(pseg), 1)
    rec = hits / max(len(gseg), 1)
    return 100.0 * 2 * prec * rec / max(prec + rec, 1e-8)


def mode_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Sliding-window majority vote of width k (offline smoothing, k = 9)."""
    if k <= 1:
        return x
    pad = k // 2
    win = np.lib.stride_tricks.sliding_window_view(np.pad(x, pad, mode="edge"), k)
    return np.array([np.bincount(w).argmax() for w in win])


SEG_KEYS = ("acc", "edit", "f1_10", "f1_25", "f1_50")


def seg_metrics(pred, gt) -> dict:
    return {
        "acc": frame_acc(pred, gt), "edit": edit_score(pred, gt),
        "f1_10": f1_at_k(pred, gt, 0.10), "f1_25": f1_at_k(pred, gt, 0.25), "f1_50": f1_at_k(pred, gt, 0.50),
    }


@torch.no_grad()
def effective_rank(x: torch.Tensor) -> float:
    """exp(-sum_i p_i log p_i), p_i = sigma_i / sum_k sigma_k over the
    singular values of the channel-centred matrix (no standardisation);
    zero singular values are omitted."""
    y = x.reshape(-1, x.shape[-1]).double()
    s = torch.linalg.svdvals(y - y.mean(dim=0, keepdim=True))
    s = s[s > s.max() * 1e-12]
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))
