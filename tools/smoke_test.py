#!/usr/bin/env python
"""CPU smoke test on synthetic data (no HOI4D / MSR needed).

1. every pretraining config: forward, loss, backward; every trainable
   parameter must receive a gradient (DDP runs with find_unused_parameters
   off);
2. leakage checks: PointWM's forecast must not change when frames >= t0 are
   altered, and the online segmentation logits of frame t must not change
   when frames > t are altered;
3. block mask invariants;
4. end-to-end runs of train_pretrain.py, tools/probe_forecast.py,
   finetune_hoi4d_seg.py (offline + online) and finetune_msr.py on tiny
   synthetic HDF5 / npz files.

    python tools/smoke_test.py
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pointwm.masks import BlockMask, last_seen_frames
from pointwm.models.downstream import SegModel
from pointwm.models.pointwm import PointWM, total_loss
from pointwm.utils.config import apply_overrides, load_config

NPTS, M = 256, 8
SMALL = ["model.num_anchors=8"]


def synthetic_clip(b, t, n=NPTS, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(b, 1, n, 3, generator=g) * 0.3
    drift = torch.linspace(0, 0.2, t).view(1, t, 1, 1) * torch.randn(b, 1, 1, 3, generator=g)
    return base + drift + 0.01 * torch.randn(b, t, n, 3, generator=g)


def check_configs():
    for path in sorted(glob.glob(os.path.join(ROOT, "configs/pretrain/**/*.yaml"), recursive=True)):
        cfg = apply_overrides(load_config(path), SMALL)
        torch.manual_seed(0)
        model = PointWM(**cfg["model"]).train()
        clip = synthetic_clip(2, 32)
        t0 = model.causal.sample_t0(32, np.random.default_rng(0)) if "causal" in model.masks else None
        out = model(clip, t0=t0)
        loss, logs = total_loss(out, cfg["loss"])
        loss.backward()
        assert torch.isfinite(loss), path
        no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not no_grad, f"{path}: parameters without gradient: {no_grad[:6]}"
        enc_grad = model.encoder.tr.blocks[0].qkv.weight.grad.abs().sum()
        assert enc_grad > 0, path
        print(f"  ok {os.path.relpath(path, ROOT):45s} " + " ".join(f"{k}={float(v):.3f}" for k, v in logs.items()))


@torch.no_grad()
def check_forecast_causality():
    cfg = apply_overrides(load_config(os.path.join(ROOT, "configs/pretrain/pointwm.yaml")), SMALL)
    model = PointWM(**cfg["model"]).eval()
    n_ctx, t0, K = 25, 26, 6

    def forecast(clip):
        tok, _ = model.encode(clip)
        z_c = model.encoder(tok["tokens"][:, : n_ctx * M], tok["xyzt"][:, : n_ctx * M])
        k = torch.arange(t0, t0 + K).view(1, -1) - (n_ctx - 1)
        a = model.prior_params(model.context_state(z_c), k.expand(clip.shape[0], -1))[0].repeat_interleave(M, 1)
        last = tok["anchor_xyz"][:, n_ctx - 1].repeat(1, K, 1)
        q = torch.cat([last, tok["xyzt"][:, t0 * M : (t0 + K) * M, 3:]], -1)
        return model.predictor(z_c, tok["xyzt"][:, : n_ctx * M], q, a)

    clip = synthetic_clip(2, 32)
    alt = clip.clone()
    alt[:, t0:] += torch.randn_like(alt[:, t0:])
    p1, d1 = forecast(clip)
    p2, d2 = forecast(alt)
    assert torch.allclose(p1, p2) and torch.allclose(d1, d2), "forecast depends on frames >= t0"
    alt2 = clip.clone()
    alt2[:, n_ctx] += 1.0
    assert not torch.allclose(forecast(alt2)[0], p1), "receptive field check failed"
    print("  ok forecast uses only input frames < t0")


@torch.no_grad()
def check_online_causality():
    model = SegModel(num_classes=19, causal=True, dropout=0.0).eval()
    clip = synthetic_clip(1, 40)
    t = 20
    alt = clip.clone()
    alt[:, t + 1 :] += torch.randn_like(alt[:, t + 1 :])
    l1, l2 = model(clip), model(alt)
    assert torch.allclose(l1[:, : t + 1], l2[:, : t + 1], atol=1e-5), "online logits depend on future frames"
    assert not torch.allclose(l1[:, t + 1 :], l2[:, t + 1 :]), "online model ignores the frames it should see"
    off = SegModel(num_classes=19, causal=False, dropout=0.0).eval()
    off.load_state_dict(model.state_dict())
    assert not torch.allclose(off(clip)[:, : t + 1], off(alt)[:, : t + 1]), "offline model should look ahead"
    print("  ok online segmentation is causal; offline is not")


def check_block_mask():
    torch.manual_seed(0)
    anchors = synthetic_clip(4, 32)[:, :, :M]
    vis, tgt = BlockMask()(anchors)
    counts = tgt.reshape(4, -1).sum(1)
    assert (counts == round(0.7 * 32 * M)).all(), counts
    assert vis[:, 0].all(), "frame 0 must stay visible"
    for rule in ("slot", "frame"):
        ls = last_seen_frames(vis, rule)
        assert (ls[tgt] >= 0).all() and (ls[tgt] < torch.arange(32).view(1, 32, 1).expand_as(ls)[tgt]).all()
        if rule == "slot":
            b, t, j = tgt.nonzero(as_tuple=True)
            assert vis[b, ls[b, t, j], j].all(), "slot rule must point at a visible token"
    print("  ok block mask: fixed count, frame 0 visible, last-seen defined")


def make_h5(root):
    import h5py

    rng = np.random.default_rng(0)
    for shard, n in (("train1", 6), ("train2", 6), ("train3", 6), ("train4", 6), ("test1", 4), ("test2", 4)):
        pcd = np.empty((n, 150, NPTS, 3), np.float32)
        lab = np.empty((n, 150), np.int64)
        for i in range(n):
            pcd[i] = synthetic_clip(1, 150, seed=int(rng.integers(1 << 30)))[0].numpy()
            cuts = np.sort(rng.choice(np.arange(10, 140), 3, replace=False))
            lab[i] = np.digitize(np.arange(150), cuts) + rng.integers(0, 15)
        with h5py.File(os.path.join(root, f"{shard}.h5"), "w") as f:
            f["pcd"], f["label"] = pcd, lab


def make_msr(root):
    rng = np.random.default_rng(0)
    for a in (1, 2):
        for s in (1, 6):
            n = 30 if s == 1 else 20
            frames = [rng.normal(size=(int(rng.integers(200, 300)), 3)).astype(np.float32) for _ in range(n)]
            np.savez(os.path.join(root, f"a{a:02d}_s{s:02d}_e01_sdepth.npz"),
                     point_clouds=np.array(frames, dtype=object), allow_pickle=True)


def run(args):
    print("  $ " + " ".join(os.path.basename(a) if os.path.isabs(a) else a for a in args))
    subprocess.run([sys.executable] + args, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def check_scripts(tmp):
    h5, msr = os.path.join(tmp, "h5"), os.path.join(tmp, "msr")
    os.makedirs(h5)
    os.makedirs(msr)
    make_h5(h5)
    make_msr(msr)
    small_pre = [f"data.root={h5}", f"data.num_points={NPTS}", "data.workers=0", "data.clip_stride=40",
                 "model.num_anchors=8", "optim.epochs=1", "optim.warmup_epochs=1", "optim.batch_size=2", "log_every=1"]
    for name in ("pointwm.yaml", "last.yaml", "ablation/norm_context.yaml"):
        run(["train_pretrain.py", "--config", f"configs/pretrain/{name}", "--seed", "0",
             "--set", *small_pre, f"out_dir={tmp}/pre_{os.path.basename(name)[:-5]}"])
    ck = f"{tmp}/pre_pointwm/seed0/epoch000.pt"
    for name in ("pointwm", "last", "norm_context"):
        run(["tools/probe_forecast.py", "--ckpt", f"{tmp}/pre_{name}/seed0/epoch000.pt", "--root", h5,
             "--out", f"{tmp}/probe/{name}", "--n_clips", "8", "--workers", "0", "--export_anchor_clips", "2"])
    run(["tools/aggregate_probe.py", "--out", f"{tmp}/probe/curve.csv", "--resamples", "50",
         "--series", f"pointwm=model:{tmp}/probe/pointwm.npz,{tmp}/probe/pointwm.npz",
         "--series", f"last_target=tq:{tmp}/probe/last.npz"])
    small_ft = [f"data.root={h5}", f"data.num_points={NPTS}", "data.workers=0", "optim.epochs=1",
                "optim.warmup_epochs=1", "optim.batch_size=2"]
    for cfg in ("hoi4d_offline.yaml", "hoi4d_online.yaml"):
        run(["finetune_hoi4d_seg.py", "--config", f"configs/finetune/{cfg}", "--init_ckpt", ck,
             "--out_dir", f"{tmp}/{cfg[:-5]}", "--set", *small_ft, "data.label_frac=0.5"])
    run(["finetune_msr.py", "--config", "configs/finetune/msr.yaml", "--init_ckpt", ck, "--out_dir", f"{tmp}/msr_run",
         "--set", f"data.root={msr}", f"data.num_points={NPTS}", "data.workers=0", "optim.epochs=1",
         "optim.warmup_epochs=1", "optim.batch_size=4"])
    run(["tools/summarize_seeds.py", f"{tmp}/hoi4d_offline/results.json", f"{tmp}/hoi4d_online/results.json"])
    print("  ok all entry points ran end to end")


def main():
    print("[1] configs"); check_configs()
    print("[2] leakage"); check_forecast_causality(); check_online_causality()
    print("[3] block mask"); check_block_mask()
    print("[4] scripts")
    with tempfile.TemporaryDirectory() as tmp:
        check_scripts(tmp)
    print("smoke test passed")


if __name__ == "__main__":
    main()
