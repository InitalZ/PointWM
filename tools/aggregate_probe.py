#!/usr/bin/env python
"""Seed-averaged horizon curves with 95% bootstrap CIs (Fig. 4, Supp. Sec. B.1).

Skill: for every clip, slot and horizon the model and reference cosines are
first averaged over the pretraining seeds; a paired percentile bootstrap
with 2,000 resamples of the clips (keeping all slots and horizons of a clip)
then recomputes the model and reference means before taking the ratio.
dR^2 (not specified in the paper): per-seed dR^2 on the same resampled
clips, averaged over seeds.

    python tools/aggregate_probe.py --out fig/data/horizon_probe.csv \
        --series a_target=tq:probe/block_s0.npz,probe/block_s1.npz,probe/block_s2.npz \
        --series a_anchor=model:probe/block_s0.npz,... \
        --series leak_target=tq:... --series leak_anchor=model:... \
        --series last=model:... --series pointwm=model:...

``key`` selects the query: ``model`` (last-seen), ``tq`` (true target
anchors), ``rollout``. Output columns: metric,series,horizon,mean,ci_low,ci_high.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np


def parse(spec):
    name, _, rest = spec.partition("=")
    key, _, paths = rest.partition(":")
    return name, key, paths.split(",")


def skill_curve(cm, cb):
    """cm, cb [N, K] clip-level mean cosines -> skill per horizon [K]."""
    m, b = cm.mean(axis=0), cb.mean(axis=0)
    return (m - b) / np.maximum(1.0 - b, 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True, help="name=key:seed0.npz,seed1.npz,seed2.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for spec in args.series:
        name, key, paths = parse(spec)
        runs = [np.load(p) for p in paths]
        n = runs[0]["cos_cl"].shape[0]
        if any(r["cos_cl"].shape[0] != n for r in runs):
            raise ValueError(f"{name}: seeds were probed on different clip sets")
        idx = rng.integers(0, n, size=(args.resamples, n))
        cm = np.mean([r[f"cos_{key}"] for r in runs], axis=0).mean(axis=2)
        cb = np.mean([r["cos_cl"] for r in runs], axis=0).mean(axis=2)
        point = skill_curve(cm, cb)
        boot = np.stack([skill_curve(cm[i], cb[i]) for i in idx])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        rows += [("skill", name, h + 1, point[h], lo[h], hi[h]) for h in range(point.shape[0])]
        e = np.stack([r[f"err2_{key}"].sum(axis=2) for r in runs])
        g = np.stack([r["mag2"].sum(axis=2) for r in runs])
        r2 = lambda ii: (1.0 - e[:, ii].sum(axis=1) / np.maximum(g[:, ii].sum(axis=1), 1e-12)).mean(axis=0)
        point = r2(np.arange(n))
        boot = np.stack([r2(i) for i in idx])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        rows += [("delta_r2", name, h + 1, point[h], lo[h], hi[h]) for h in range(point.shape[0])]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "series", "horizon", "mean", "ci_low", "ci_high"])
        for m, s, h, a, b, c in rows:
            w.writerow([m, s, h, f"{a:.3f}", f"{b:.3f}", f"{c:.3f}"])
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
