#!/usr/bin/env python
"""Mean +- std over seeds of downstream results.json files (std with the
sample denominator n - 1, Supp. Sec. A.3), or of probe summaries.

    python tools/summarize_seeds.py runs/seg_offline/pointwm/seed*/results.json
    python tools/summarize_seeds.py --probe probe/pointwm_s*.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--probe", action="store_true", help="files are probe_forecast.py summaries")
    args = ap.parse_args()
    recs = []
    for p in args.files:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        recs.append(flat(r if args.probe else r["test"]))
    keys = [k for k in recs[0] if all(k in r for r in recs)]
    print(f"{len(recs)} runs")
    for k in keys:
        v = np.array([r[k] for r in recs])
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        print(f"{k:40s} {v.mean():9.3f} +- {sd:.3f}   ({', '.join(f'{x:.3f}' for x in v)})")


if __name__ == "__main__":
    main()
