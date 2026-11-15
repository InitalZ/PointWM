#!/usr/bin/env python
"""HOI4D action segmentation fine-tuning (Sec. 4.3 / 4.4, Supp. Sec. A.3).

    torchrun --nproc_per_node 4 finetune_hoi4d_seg.py --config configs/finetune/hoi4d_offline.yaml \
        --seed 0 --init_ckpt <pretrain out_dir>/seed0/epoch099.pt

Offline: 150-frame inputs in normalised coordinates, temporal mode filter of
width 9. Online: raw HOI4D coordinates, causal P4DConv kernel over frames
t-2..t and causal attention, no smoothing. Both: AdamW (wd 0.05), lr 1e-4,
global batch 8, 30 epochs, 3-epoch linear warmup and cosine decay to 1e-6,
selection on validation F1@50 (under the protocol's own smoothing rule) and
a single test evaluation of the selected checkpoint. ``--init_ckpt none``
trains from scratch. ``data.label_frac`` gives the label-efficiency runs
(Seg_10% is 0.1).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pointwm.data.hoi4d import NUM_CLASSES, Hoi4dSequences, hoi4d_videos
from pointwm.models.downstream import SegModel, load_backbone
from pointwm.utils.config import apply_overrides, load_config
from pointwm.utils.dist import init_distributed, is_main, per_gpu_batch, seed_everything, worker_init
from pointwm.utils.io import atomic_save, write_json
from pointwm.utils.metrics import SEG_KEYS, mode_filter, seg_metrics
from pointwm.utils.schedulers import WarmupCosine


@torch.no_grad()
def evaluate(model, ds, device, smooth, batch, workers, amp, dist_on, world):
    """Per-video metrics on whole sequences, averaged over videos."""
    model.eval()
    sampler = DistributedSampler(ds, shuffle=False) if dist_on else None
    dl = DataLoader(ds, batch_size=batch, shuffle=False, sampler=sampler, num_workers=workers, pin_memory=True)
    res = {}
    for clip, y, vi in dl:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(clip.to(device, non_blocking=True).float())
        pred = logits.float().argmax(-1).cpu().numpy()
        for b in range(pred.shape[0]):
            res[int(vi[b])] = seg_metrics(mode_filter(pred[b], smooth), y[b].numpy())
    if dist_on:
        parts = [None] * world
        torch.distributed.all_gather_object(parts, res)
        res = {k: v for p in parts for k, v in p.items()}
    if len(res) != len(ds):
        raise RuntimeError(f"evaluated {len(res)} of {len(ds)} videos")
    return {k: float(np.mean([r[k] for r in res.values()])) for k in SEG_KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--init_ckpt", default=None, help="pretraining checkpoint, or 'none' for scratch")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    init = args.init_ckpt if args.init_ckpt is not None else cfg.get("init_ckpt")
    init = None if init in (None, "", "none") else init
    out_dir = args.out_dir or os.path.join(cfg["out_dir"], f"seed{seed}")

    dist_on, rank, world, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(seed * 100 + rank)
    d, o, mode = cfg["data"], cfg["optim"], cfg["mode"]
    if mode not in ("offline", "online"):
        raise ValueError(f"mode must be offline|online, got {mode!r}")
    normalize = "full" if mode == "offline" else "none"
    smooth = int(cfg["eval"]["smoothing"])
    if mode == "online" and smooth != 1:
        raise ValueError("online predictions are not smoothed")

    split = dict(split_seed=d.get("split_seed", 17), val_frac=d.get("val_frac", 0.1))
    train_v = hoi4d_videos(d["root"], "train", label_frac=d.get("label_frac", 1.0), **split)
    common = dict(num_points=d["num_points"], normalize=normalize)
    train_ds = Hoi4dSequences(d["root"], train_v, augment=d.get("augment", True), **common)
    val_ds = Hoi4dSequences(d["root"], hoi4d_videos(d["root"], "val", **split), augment=False, **common)
    test_ds = Hoi4dSequences(d["root"], hoi4d_videos(d["root"], "test", **split), augment=False, **common)
    bs = per_gpu_batch(o["batch_size"], world)
    sampler = DistributedSampler(train_ds, shuffle=True, seed=seed, drop_last=True) if dist_on else None
    train_dl = DataLoader(
        train_ds, batch_size=bs, shuffle=sampler is None, sampler=sampler, num_workers=d["workers"], drop_last=True,
        pin_memory=True, persistent_workers=d["workers"] > 0, worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(seed * 100 + rank),
    )

    model = SegModel(**cfg["model"], num_classes=NUM_CLASSES, causal=(mode == "online")).to(device)
    if init:
        info = load_backbone(model, init)
        if is_main(rank):
            print(f"[seg] loaded {info['loaded']} backbone tensors from {init}")
    elif is_main(rank):
        print("[seg] scratch (no pretraining)")
    if dist_on:
        model = DDP(model, device_ids=[local_rank])
    raw = model.module if dist_on else model

    optim = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    total = o["epochs"] * len(train_dl)
    sched = WarmupCosine(o["lr"], o["lr_end"], total, o["warmup_epochs"] * len(train_dl))
    amp = bool(o.get("amp", True)) and device.type == "cuda"
    if is_main(rank):
        print(f"[seg] {mode}: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test videos, "
              f"{len(train_dl)} steps/epoch, global batch {o['batch_size']}")

    best, history = -1.0, []
    for ep in range(o["epochs"]):
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        tot, n = 0.0, 0
        for clip, y, _ in train_dl:
            lr = sched.step()
            for g in optim.param_groups:
                g["lr"] = lr
            clip, y = clip.to(device, non_blocking=True).float(), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(clip)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), o["clip_grad"])
            optim.step()
            tot, n = tot + float(loss), n + 1
        val = evaluate(model, val_ds, device, smooth, bs, d["workers"], amp, dist_on, world)
        history.append({"epoch": ep, "train_loss": tot / max(n, 1), **{f"val_{k}": v for k, v in val.items()}})
        if is_main(rank):
            print(f"[seg] epoch {ep} loss {tot / max(n, 1):.4f} val " + " ".join(f"{k}={v:.2f}" for k, v in val.items()), flush=True)
            if val["f1_50"] > best:
                best = val["f1_50"]
                atomic_save({"model": raw.state_dict(), "epoch": ep, "val": val, "config": cfg}, os.path.join(out_dir, "best.pt"))
        if dist_on:
            torch.distributed.barrier()

    ck = torch.load(os.path.join(out_dir, "best.pt"), map_location="cpu", weights_only=False)
    raw.load_state_dict(ck["model"])
    test = evaluate(model, test_ds, device, smooth, bs, d["workers"], amp, dist_on, world)
    if is_main(rank):
        print(f"[seg] test (best val epoch {ck['epoch']}): " + " ".join(f"{k}={v:.2f}" for k, v in test.items()))
        write_json({"mode": mode, "seed": seed, "init_ckpt": init, "label_frac": d.get("label_frac", 1.0),
                    "best_epoch": ck["epoch"], "val": ck["val"], "test": test, "history": history},
                   os.path.join(out_dir, "results.json"))


if __name__ == "__main__":
    main()
