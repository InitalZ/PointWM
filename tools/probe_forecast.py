#!/usr/bin/env python
"""Forecasting probe of a frozen pretrained model (Sec. 4.2, Supp. Sec. B).

Protocol: 200 unaugmented 32-frame test clips, taken at evenly spaced
positions in the list of all windows cut from the HOI4D test videos with
stride 16. The predictor gets the first 25 anchor frames as context and
forecasts the 6 frames after a one-frame gap, queried with last-seen anchors
(a time-only model is queried with its own time-only query) and, for
latent-action models, conditioned on the prior mean. Targets are the
model's own full-clip teacher latents, paired by slot. Batches of 8 clips.

Reported (all per forecast step t+1..t+6 and averaged over the steps):
  skill vs copy-last (CL), context mean (CM), linear extrapolation (LE);
  CL* (queried with the true target anchors); rollout (one anchor at a time);
  dR^2 over all and moving tokens, direction cosine on moving tokens;
  anchor Chamfer of copy-last and of the position head (Table S8);
  effective rank of frame-pooled student features (Table 5);
  posterior skill at t+6, Delta_act and clip variance (Table 6).

    python tools/probe_forecast.py --ckpt <out_dir>/seed0/epoch099.pt --root <HOI4D h5 dir> --out probe/pointwm_s0

writes <out>.json (summary) and <out>.npz (per-clip arrays for
tools/aggregate_probe.py, which averages seeds and bootstraps the CIs).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointwm.data.hoi4d import Hoi4dClips, hoi4d_videos
from pointwm.data.transforms import clip_normalize_torch
from pointwm.models.pointwm import PointWM, frame_chamfer
from pointwm.utils.metrics import effective_rank


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = PointWM(**ck["config"]["model"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["config"]


def probe_loader(root, pcfg, n_clips, batch, workers, normalize):
    ds = Hoi4dClips(root, hoi4d_videos(root, "test"), clip_len=pcfg["clip_len"], frame_interval=pcfg["frame_interval"],
                    clip_stride=16, num_points=pcfg["num_points"], normalize=normalize, augment=False)
    sel = np.linspace(0, len(ds) - 1, n_clips).astype(int).tolist()
    return DataLoader(Subset(ds, sel), batch_size=batch, shuffle=False, num_workers=workers)


def cos(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def skill(cm, cb):
    return (cm - cb) / max(1e-8, 1.0 - cb)


@torch.no_grad()
def run(model, dl, device, n_ctx=25, gap=1, horizon=6, normalize="full", export=0):
    pred = model.predictor
    acc = {}
    rows, mu_all, exported = [], [], []

    def put(k, v):
        acc.setdefault(k, []).append(v.detach().float().cpu().numpy())

    for clip, _, _, _ in dl:
        clip = clip.to(device).float()
        t0 = n_ctx + gap
        if normalize == "context":
            clip = clip_normalize_torch(clip, upto=n_ctx + 1)
        tok, h = model.encode(clip)
        b, tprime, m = clip.shape[0], tok["T"], tok["M"]
        if t0 + horizon > tprime:
            raise ValueError(f"T'={tprime} too short for {n_ctx} context + {gap} gap + {horizon} targets")
        t_c = n_ctx - 1
        t_idx = torch.arange(t0, t0 + horizon, device=device)
        k = (t_idx - t_c).view(1, -1).expand(b, -1)
        tokens, xyzt, anchor = tok["tokens"], tok["xyzt"], tok["anchor_xyz"]
        tgt = slice(t0 * m, (t0 + horizon) * m)
        tgt_idx = torch.arange(t0 * m, (t0 + horizon) * m, device=device).view(1, -1).expand(b, -1)
        ctx_xyzt = xyzt[:, : n_ctx * m]
        z_c = model.encoder(tokens[:, : n_ctx * m], ctx_xyzt)
        z_f = model.encoder(tokens, xyzt)
        true_xyzt = xyzt[:, tgt]
        last_xyz = anchor[:, t_c].repeat(1, horizon, 1)

        h4 = h.view(b, tprime, m, -1).float()
        h_t, h_last, h_prev = h4[:, t0 : t0 + horizon], h4[:, t_c : t_c + 1], h4[:, t_c - 1 : t_c]
        h_cm = h4[:, :n_ctx].mean(dim=1, keepdim=True)
        h_le = h_last + k[0].view(1, -1, 1, 1).float() * (h_last - h_prev)

        own = "time_only" if model.query == "time_only" else "last_seen"
        q_ls, slot = model.make_query(own, true_xyzt, last_xyz, tgt_idx)
        q_tq, slot_tq = model.make_query("time_only" if own == "time_only" else "target", true_xyzt, last_xyz, tgt_idx)

        a_prior = a_post = None
        if model.la_enabled:
            mu_p, _ = model.prior_params(model.context_state(z_c), k)
            la_in = F.layer_norm(z_f, (z_f.shape[-1],)) if model.la_source == "student" else h
            states = model.frame_states(la_in, tprime)
            mu_q, _ = model.posterior(states, t_idx)
            a_prior, a_post = mu_p.repeat_interleave(m, 1), mu_q.repeat_interleave(m, 1)
            mu_all.append(model.posterior(states, torch.arange(1, tprime, device=device))[0].cpu().numpy())

        def forecast(q, a, s):
            p, dp = pred(z_c, ctx_xyzt, q, a, s)
            return p.float().view(b, horizon, m, -1), dp

        p_ls, dpos = forecast(q_ls, a_prior, slot)
        p_tq, _ = forecast(q_tq, a_prior, slot_tq)
        ctx_r, xyzt_r, p_ro = z_c, ctx_xyzt, []
        for s in range(horizon):
            sl = slice(s * m, (s + 1) * m)
            p_s, _ = pred(ctx_r, xyzt_r, q_ls[:, sl], None if a_prior is None else a_prior[:, sl],
                          None if slot is None else slot[:, sl])
            p_ro.append(p_s.float())
            ctx_r, xyzt_r = torch.cat([ctx_r, p_s.to(ctx_r.dtype)], 1), torch.cat([xyzt_r, q_ls[:, sl]], 1)
        p_ro = torch.stack(p_ro, dim=1)

        put("cos_model", cos(p_ls, h_t))
        put("cos_tq", cos(p_tq, h_t))
        put("cos_rollout", cos(p_ro, h_t))
        put("cos_cl", cos(h_last.expand_as(h_t), h_t))
        put("cos_cm", cos(h_cm.expand_as(h_t), h_t))
        put("cos_le", cos(h_le, h_t))
        delta = h_t - h_last
        mag2 = delta.pow(2).sum(-1)
        put("mag2", mag2)
        for name, p in (("model", p_ls), ("tq", p_tq), ("rollout", p_ro)):
            put(f"err2_{name}", (p - h_last - delta).pow(2).sum(-1))
        put("moving", (mag2.sqrt() >= mag2.sqrt().median()).float())
        put("dircos_model", cos(p_ls - h_last, delta))

        true_c = true_xyzt[..., :3].view(b, horizon, m, 3)
        put("chamfer_copy_last", frame_chamfer(last_xyz.view(b, horizon, m, 3), true_c))
        if dpos is not None:
            c_hat = (last_xyz + dpos.float()).view(b, horizon, m, 3)
            put("chamfer_forecast", frame_chamfer(c_hat, true_c))
            if len(exported) < export:
                for i in range(min(b, export - len(exported))):
                    exported.append({"points": clip[i, t0 : t0 + horizon].cpu().numpy(), "true": true_c[i].cpu().numpy(),
                                     "last_seen": last_xyz.view(b, horizon, m, 3)[i].cpu().numpy(),
                                     "forecast": c_hat[i].cpu().numpy()})
        if a_post is not None:
            p_post, _ = forecast(q_ls, a_post, slot)
            p_shift, _ = forecast(q_ls, a_post.roll(-1, dims=0), slot)
            put("cos_post", cos(p_post, h_t))
            put("l1_own_last", (p_post[:, -1] - h_t[:, -1]).abs().mean(dim=(1, 2)))
            put("l1_shift_last", (p_shift[:, -1] - h_t[:, -1]).abs().mean(dim=(1, 2)))
        rows.append(z_f.float().view(b, tprime, m, -1).mean(dim=2).reshape(-1, z_f.shape[-1]).cpu())

    arr = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    return arr, torch.cat(rows), (np.concatenate(mu_all) if mu_all else None), exported


def per_step_skill(arr, key, base):
    m, c = arr[key].mean(axis=(0, 2)), arr[base].mean(axis=(0, 2))
    return [skill(float(a), float(b)) for a, b in zip(m, c)]


def per_step_r2(arr, key, mask=None):
    e, g = arr[f"err2_{key}"], arr["mag2"]
    if mask is not None:
        e, g = e * mask, g * mask
    return [1.0 - float(e[:, h].sum()) / max(float(g[:, h].sum()), 1e-12) for h in range(e.shape[1])]


def clip_variance(mu):
    """Between-clip share (%) of the posterior-mean variance (Eq. S3)."""
    gm = mu.reshape(-1, mu.shape[-1]).mean(axis=0)
    between = (((mu.mean(axis=1) - gm) ** 2).sum(axis=1) * mu.shape[1]).sum()
    total = ((mu - gm) ** 2).sum()
    return 100.0 * float(between / max(total, 1e-12))


def summarize(arr, rank, mu):
    avg = lambda x: float(np.mean(x))
    out = {}
    for name, key, base in (("skill_cl", "cos_model", "cos_cl"), ("skill_cm", "cos_model", "cos_cm"),
                            ("skill_le", "cos_model", "cos_le"), ("skill_cl_target_query", "cos_tq", "cos_cl"),
                            ("skill_cl_rollout", "cos_rollout", "cos_cl")):
        out[name] = {"per_step": per_step_skill(arr, key, base)}
        out[name]["avg"] = avg(out[name]["per_step"])
    for name, key, mask in (("dr2", "model", None), ("dr2_moving", "model", arr["moving"]),
                            ("dr2_target_query", "tq", None), ("dr2_rollout", "rollout", None)):
        out[name] = {"per_step": per_step_r2(arr, key, mask)}
        out[name]["avg"] = avg(out[name]["per_step"])
    mv = arr["moving"] > 0
    out["dircos_moving"] = float(arr["dircos_model"][mv].mean())
    out["chamfer_copy_last"] = arr["chamfer_copy_last"].mean(axis=0).tolist()
    if "chamfer_forecast" in arr:
        out["chamfer_forecast"] = arr["chamfer_forecast"].mean(axis=0).tolist()
    out["effective_rank"] = effective_rank(rank)
    if "cos_post" in arr:
        out["skill_cl_posterior"] = {"per_step": per_step_skill(arr, "cos_post", "cos_cl")}
        out["prior_skill_t6"] = out["skill_cl"]["per_step"][-1]
        out["posterior_skill_t6"] = out["skill_cl_posterior"]["per_step"][-1]
        out["delta_act"] = float(arr["l1_shift_last"].mean() - arr["l1_own_last"].mean())
        out["clip_var"] = clip_variance(mu)
    out["n_clips"] = int(arr["cos_model"].shape[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True, help="HOI4D h5 directory")
    ap.add_argument("--out", required=True, help="output prefix (.json / .npz)")
    ap.add_argument("--n_clips", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--context", type=int, default=25)
    ap.add_argument("--gap", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--export_anchor_clips", type=int, default=0, help="save anchor forecasts of the first N clips")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, cfg = load_model(args.ckpt, device)
    normalize = cfg["data"].get("normalize", "full")
    dl = probe_loader(args.root, cfg["data"], args.n_clips, args.batch, args.workers,
                      "full" if normalize == "full" else "none")
    arr, rank_rows, mu, exported = run(model, dl, device, args.context, args.gap, args.horizon, normalize,
                                       args.export_anchor_clips)
    summary = summarize(arr, rank_rows, mu)
    summary["ckpt"] = os.path.abspath(args.ckpt)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out + ".npz", **arr)
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    if exported:
        np.savez_compressed(args.out + "_anchors.npz", **{f"{k}_{i}": v for i, e in enumerate(exported) for k, v in e.items()})
    print(json.dumps({k: (v["avg"] if isinstance(v, dict) and "avg" in v else v) for k, v in summary.items()
                      if k not in ("chamfer_copy_last", "chamfer_forecast")}, indent=1))


if __name__ == "__main__":
    main()
