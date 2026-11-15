#!/usr/bin/env python
"""PointWM pretraining on unlabelled HOI4D (Sec. 3, Table S1, Algorithm S1).

    torchrun --nproc_per_node 4 train_pretrain.py --config configs/pretrain/pointwm.yaml --seed 0

Every controlled variant and ablation runs through this script with its own
config (configs/pretrain/), or with ``--set`` overrides for the sensitivity
sweep (scripts/pretrain_sensitivity.sh).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pointwm.data.hoi4d import Hoi4dClips, hoi4d_videos
from pointwm.data.transforms import augment_torch, clip_normalize_torch
from pointwm.models.pointwm import PointWM, total_loss
from pointwm.utils.config import apply_overrides, load_config
from pointwm.utils.dist import init_distributed, is_main, per_gpu_batch, seed_everything, worker_init
from pointwm.utils.io import JsonlLogger, atomic_save
from pointwm.utils.schedulers import EMAMomentum, WarmupCosine


def param_groups(model, wd):
    """Weight decay on weight matrices only; none on 1-D parameters, the mask
    token and the positional embeddings (as in the old code)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "mask_token" in n or "pos" in n else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="overrides cfg.seed (paper: 0, 1, 2)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, key.sub=value")
    ap.add_argument("--resume", default="", help="checkpoint path, or 'auto' for <out_dir>/last.pt")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    cfg["seed"] = seed

    dist_on, rank, world, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(seed * 100 + rank)
    out_dir = os.path.join(cfg["out_dir"], f"seed{seed}")

    d, o, lcfg = cfg["data"], cfg["optim"], cfg["loss"]
    normalize = d.get("normalize", "full")
    if normalize not in ("full", "context"):
        raise ValueError(f"data.normalize must be full|context, got {normalize!r}")
    videos = hoi4d_videos(d["root"], "train", split_seed=d.get("split_seed", 17), val_frac=d.get("val_frac", 0.1))
    ds = Hoi4dClips(
        d["root"], videos, clip_len=d["clip_len"], frame_interval=d["frame_interval"], clip_stride=d["clip_stride"],
        num_points=d["num_points"],
        normalize="full" if normalize == "full" else "none", augment=(normalize == "full"),
    )
    sampler = DistributedSampler(ds, shuffle=True, seed=seed, drop_last=True) if dist_on else None
    dl = DataLoader(
        ds, batch_size=per_gpu_batch(o["batch_size"], world), shuffle=sampler is None, sampler=sampler,
        num_workers=d["workers"], drop_last=True, pin_memory=True, persistent_workers=d["workers"] > 0,
        worker_init_fn=worker_init, generator=torch.Generator().manual_seed(seed * 100 + rank),
    )

    model = PointWM(**cfg["model"]).to(device)
    if normalize == "context" and model.masks != ["causal"]:
        raise ValueError("context-only normalisation is defined for the causal mask only")
    if dist_on:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=bool(o.get("find_unused", False)))
    raw = model.module if dist_on else model

    optim = torch.optim.AdamW(param_groups(raw, o["weight_decay"]), lr=o["lr"])
    spe = len(dl)
    total_steps = o["epochs"] * spe
    lr_sched = WarmupCosine(o["lr"], o["lr_end"], total_steps, o["warmup_epochs"] * spe)
    ema_sched = EMAMomentum(o["ema"][0], o["ema"][1], total_steps)

    start_epoch, step = 0, 0
    resume = os.path.join(out_dir, "last.pt") if args.resume == "auto" else args.resume
    if resume and os.path.isfile(resume):
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        raw.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optimizer"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        lr_sched.step_id, ema_sched.step_id = ck["lr_step"], ck["ema_step"]
        if is_main(rank):
            print(f"[pretrain] resumed {resume} at epoch {start_epoch}, step {step}")

    logger = JsonlLogger(out_dir) if is_main(rank) else None
    if is_main(rank):
        n_train = sum(p.numel() for p in raw.parameters() if p.requires_grad)
        print(f"[pretrain] {len(videos)} videos, {len(ds)} clips, {spe} steps/epoch, {total_steps} steps, "
              f"global batch {o['batch_size']} on {world} GPU(s), trainable params {n_train / 1e6:.2f}M")

    amp = bool(o.get("amp", True)) and device.type == "cuda"
    has_causal = "causal" in raw.masks
    model.train()
    for ep in range(start_epoch, o["epochs"]):
        if sampler is not None:
            sampler.set_epoch(ep)
        for clip, _, _, _ in dl:
            clip = clip.to(device, non_blocking=True).float()
            t0 = None
            if has_causal:
                tprime = clip.shape[1] // raw.tokenizer.p4d.ts
                t0 = raw.causal.sample_t0(tprime, np.random.default_rng([seed, step]))
            if normalize == "context":
                clip = augment_torch(clip_normalize_torch(clip, upto=t0 - raw.causal.gap + 1))
            lr = lr_sched.step()
            for g in optim.param_groups:
                g["lr"] = lr
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(clip, t0=t0)
                loss, logs = total_loss(out, lcfg)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), o["clip_grad"])
            optim.step()
            momentum = ema_sched.step()
            raw.ema_update(momentum)
            if logger is not None and step % cfg.get("log_every", 50) == 0:
                scalars = {k: float(v) for k, v in logs.items()}
                logger.log(step, epoch=ep, lr=lr, ema=momentum, t0=t0 if t0 is not None else -1, **scalars)
                print(f"[ep {ep} step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in scalars.items()), flush=True)
            step += 1

        if is_main(rank):
            ck = {
                "model": raw.state_dict(), "optimizer": optim.state_dict(), "epoch": ep, "step": step,
                "lr_step": lr_sched.step_id, "ema_step": ema_sched.step_id, "config": cfg,
            }
            atomic_save(ck, os.path.join(out_dir, "last.pt"))
            if (ep + 1) % cfg.get("ckpt_every", 10) == 0 or ep + 1 == o["epochs"]:
                atomic_save({k: v for k, v in ck.items() if k != "optimizer"}, os.path.join(out_dir, f"epoch{ep:03d}.pt"))
        if dist_on:
            torch.distributed.barrier()
    if is_main(rank):
        final = os.path.join(out_dir, f"epoch{o['epochs'] - 1:03d}.pt")
        print(f"[pretrain] done; downstream checkpoint: {final}")
        logger.close()


if __name__ == "__main__":
    main()
