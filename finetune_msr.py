#!/usr/bin/env python
"""MSR-Action3D recognition, fine-tuned end to end from the HOI4D checkpoint
(Sec. 4.3, Supp. Sec. A.3): 24-frame clips at interval 1, AdamW (wd 0.05),
lr 5e-4, global batch 32, 30 epochs, 3-epoch warmup, cosine decay to 1e-6.
Evaluated once, at the last epoch, on all 297 test videos (no selection on
the test set).

    torchrun --nproc_per_node 4 finetune_msr.py --config configs/finetune/msr.yaml \
        --seed 0 --init_ckpt <pretrain out_dir>/seed0/epoch099.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pointwm.data.msr import NUM_CLASSES, MsrClips
from pointwm.models.downstream import ClsModel, load_backbone
from pointwm.utils.config import apply_overrides, load_config
from pointwm.utils.dist import init_distributed, is_main, per_gpu_batch, seed_everything, worker_init
from pointwm.utils.io import atomic_save, write_json
from pointwm.utils.schedulers import WarmupCosine


@torch.no_grad()
def evaluate(model, ds, device, batch, workers, amp):
    """Video accuracy: softmax summed over all clips of a video."""
    model.eval()
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    probs = torch.zeros(len(ds.videos), NUM_CLASSES)
    for clip, _, vi in dl:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(clip.to(device, non_blocking=True))
        probs.index_add_(0, vi, logits.float().softmax(-1).cpu())
    hit = (probs.argmax(-1) == torch.tensor(ds.labels)).float()
    span = ds.frame_interval * (ds.clip_len - 1)
    long_enough = torch.tensor([len(v) > span for v in ds.videos])
    return {
        "video_acc": 100.0 * float(hit.mean()), "n_videos": len(ds.videos),
        "video_acc_full_clips": 100.0 * float(hit[long_enough].mean()), "n_full_clip_videos": int(long_enough.sum()),
    }


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
    d, o = cfg["data"], cfg["optim"]
    common = dict(clip_len=d["clip_len"], frame_interval=d["frame_interval"], num_points=d["num_points"])
    train_ds = MsrClips(d["root"], "train", **common)
    sampler = DistributedSampler(train_ds, shuffle=True, seed=seed, drop_last=True) if dist_on else None
    train_dl = DataLoader(
        train_ds, batch_size=per_gpu_batch(o["batch_size"], world), shuffle=sampler is None, sampler=sampler,
        num_workers=d["workers"], drop_last=True, pin_memory=True, persistent_workers=d["workers"] > 0,
        worker_init_fn=worker_init, generator=torch.Generator().manual_seed(seed * 100 + rank),
    )

    model = ClsModel(**cfg["model"], num_classes=NUM_CLASSES).to(device)
    if init:
        info = load_backbone(model, init)
        if is_main(rank):
            print(f"[msr] loaded {info['loaded']} backbone tensors from {init}")
    elif is_main(rank):
        print("[msr] scratch (no pretraining)")
    if dist_on:
        model = DDP(model, device_ids=[local_rank])
    raw = model.module if dist_on else model

    optim = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    sched = WarmupCosine(o["lr"], o["lr_end"], o["epochs"] * len(train_dl), o["warmup_epochs"] * len(train_dl))
    amp = bool(o.get("amp", True)) and device.type == "cuda"
    for ep in range(o["epochs"]):
        if sampler is not None:
            sampler.set_epoch(ep)
        model.train()
        tot, n = 0.0, 0
        for clip, y, _ in train_dl:
            lr = sched.step()
            for g in optim.param_groups:
                g["lr"] = lr
            clip, y = clip.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(clip)
            loss = F.cross_entropy(logits.float(), y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), o["clip_grad"])
            optim.step()
            tot, n = tot + float(loss), n + 1
        if is_main(rank):
            print(f"[msr] epoch {ep} loss {tot / max(n, 1):.4f}", flush=True)

    if is_main(rank):
        test = evaluate(raw, MsrClips(d["root"], "test", augment=False, **common), device,
                        o.get("eval_batch_size", 64), d["workers"], amp)
        print(f"[msr] last epoch: video acc {test['video_acc']:.2f} over {test['n_videos']} videos")
        atomic_save({"model": raw.state_dict(), "config": cfg, "test": test}, os.path.join(out_dir, "last.pt"))
        write_json({"seed": seed, "init_ckpt": init, "test": test}, os.path.join(out_dir, "results.json"))
    if dist_on:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
