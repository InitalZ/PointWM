from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointwm.masks import BlockMask, CausalMask, gather_tokens, last_seen_frames
from pointwm.masks.utils import anchors_at
from pointwm.models.encoder import TokenEncoder
from pointwm.models.latent_action import InverseDynamics, LatentActionPrior, gaussian_kl, std_normal_kl
from pointwm.models.predictor import Predictor
from pointwm.models.tokenizer import PointTubeTokenizer

QUERIES = ("target", "last_seen", "noised", "time_only")


def vicreg_var_cov(z: torch.Tensor, cov_weight: float = 0.04) -> torch.Tensor:
    """Eq. 7 on rows z [n, d]: (1/d) sum_r max(0, 1 - sigma_r)
    + (cov_weight/d) sum_{r != s} C_rs^2, computed in fp32."""
    with torch.autocast(device_type=z.device.type, enabled=False):
        z = z.float()
        std = z.var(dim=0).add(1e-4).sqrt()
        var_loss = F.relu(1.0 - std).mean()
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(1, z.shape[0] - 1)
        d = cov.shape[0]
        cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / d
    return var_loss + cov_weight * cov_loss


def frame_chamfer(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Eq. 4: symmetric set distance between predicted and true anchors, per
    target frame. pred / true [B, K, M, 3] -> [B, K]
    (0.5 * (mean_j min_j' ||.||_2 + mean_j' min_j ||.||_2))."""
    with torch.autocast(device_type=pred.device.type, enabled=False):
        b, k, m, _ = pred.shape
        d = torch.cdist(pred.float().reshape(b * k, m, 3), true.float().reshape(b * k, m, 3))
        cd = 0.5 * (d.min(dim=2).values.mean(dim=1) + d.min(dim=1).values.mean(dim=1))
    return cd.view(b, k)


class PointWM(nn.Module):
    """Student tokenizer + encoder f_theta, EMA teacher f_theta_bar, predictor
    g_phi, IDM q_psi and prior p_omega (Fig. 3, Sec. 3).

    The same class builds every controlled variant and ablation of the paper;
    they differ only in the constructor arguments (see configs/pretrain):

    ``masks``           ["causal"] (PointWM), ["block"] (Block) or
                        ["block", "causal"] (Leak, Last). With several mask
                        types every clip is processed under each, with a
                        separate context pass, and L_lat is averaged over them.
    ``query``           target query position: "target" (true anchors c_{t,j},
                        Block / Leak / row i), "last_seen" (c_{t_c,j}, PointWM
                        and Last), "noised" (c_{t,j} + N(0, sigma^2 I), rows
                        j/k) or "time_only" (tau_t plus a slot embedding,
                        row l).
    ``last_seen_rule``  block-mask targets only: "slot" or "frame" (see
                        masks.utils.last_seen_frames).
    ``position_head``   residual anchor head with L_pos (causal targets).
    ``latent_action``   IDM + prior (causal-only models).
    ``vicreg``          "frame" (PointWM), "token" (row h) or "none".
    """

    def __init__(
        self,
        dim=384,
        depth=6,
        heads=6,
        pred_dim=256,
        pred_depth=4,
        pred_heads=4,
        num_anchors=64,
        tokenizer=None,
        masks=("causal",),
        causal=None,
        block=None,
        query="last_seen",
        query_noise_std=0.0,
        last_seen_rule="slot",
        position_head=True,
        latent_action=None,
        vicreg="frame",
    ):
        super().__init__()
        self.masks = list(masks)
        if not self.masks or any(k not in ("causal", "block") for k in self.masks) or len(set(self.masks)) != len(self.masks):
            raise ValueError(f"masks must be a non-empty subset of [block, causal], got {self.masks}")
        if query not in QUERIES:
            raise ValueError(f"query must be one of {QUERIES}, got {query!r}")
        if query in ("noised", "time_only") and self.masks != ["causal"]:
            raise ValueError(f"query={query!r} is a PointWM ablation and needs masks=[causal]")
        if position_head and "causal" not in self.masks:
            raise ValueError("the position head is supervised on causal (whole-frame) targets only")
        if vicreg not in ("none", "frame", "token"):
            raise ValueError(f"vicreg must be none|frame|token, got {vicreg!r}")
        self.num_anchors = int(num_anchors)
        self.query = query
        self.query_noise_std = float(query_noise_std)
        self.last_seen_rule = last_seen_rule
        self.vicreg = vicreg
        self.causal = CausalMask(**(causal or {}))
        self.block = BlockMask(**(block or {})) if "block" in self.masks else None

        la = dict(latent_action or {})
        self.la_enabled = bool(la.get("enabled", False))
        if self.la_enabled and self.masks != ["causal"]:
            raise ValueError("the latent action is defined on causal targets only (masks=[causal])")
        self.la_source = str(la.get("source", "student"))
        self.la_prior_mode = str(la.get("prior", "learned"))
        self.la_prior_sub_p = float(la.get("prior_sub_p", 0.3))
        self.la_kl_detach = bool(la.get("kl_detach", True))
        if self.la_source not in ("student", "teacher"):
            raise ValueError(f"latent_action.source must be student|teacher, got {self.la_source!r}")
        if self.la_prior_mode not in ("learned", "fixed"):
            raise ValueError(f"latent_action.prior must be learned|fixed, got {self.la_prior_mode!r}")
        if self.la_prior_mode == "fixed" and self.la_prior_sub_p > 0:
            raise ValueError("the fixed N(0, I) prior drops the prior-mean replacement (Supp. Sec. A.2)")
        act_dim = int(la.get("act_dim", 8))
        hidden = int(la.get("hidden", 256))

        self.tokenizer = PointTubeTokenizer(dim, **(tokenizer or {}))
        self.target_tokenizer = copy.deepcopy(self.tokenizer)
        self.encoder = TokenEncoder(dim=dim, depth=depth, heads=heads)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in list(self.target_tokenizer.parameters()) + list(self.target_encoder.parameters()):
            p.requires_grad = False
        self.predictor = Predictor(
            dim, pred_dim, pred_depth, pred_heads,
            position_head=position_head,
            act_dim=act_dim if self.la_enabled else 0,
            num_slots=self.num_anchors if query == "time_only" else 0,
        )
        self.idm = None
        self.prior = None
        if self.la_enabled:
            idm_input = str(la.get("idm_input", "delta"))
            if idm_input not in ("delta", "state_delta"):
                raise ValueError(f"latent_action.idm_input must be delta|state_delta, got {idm_input!r}")
            self.idm = InverseDynamics(dim, act_dim, hidden, use_state=(idm_input == "state_delta"))
            if self.la_prior_mode == "learned":
                self.prior = LatentActionPrior(dim, act_dim, hidden)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_tokenizer.eval()
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def ema_update(self, momentum: float):
        """theta_bar <- m * theta_bar + (1 - m) * theta, tokenizer and encoder."""
        tgt = list(self.target_encoder.parameters()) + list(self.target_tokenizer.parameters())
        src = list(self.encoder.parameters()) + list(self.tokenizer.parameters())
        torch._foreach_mul_(tgt, momentum)
        torch._foreach_add_(tgt, src, alpha=1.0 - momentum)

    @property
    def needs_full_pass(self) -> bool:
        """Whether the observation branch z_f = f_theta(x, u) is run."""
        return self.vicreg != "none" or (self.la_enabled and self.la_source == "student")

    def encode(self, clip: torch.Tensor):
        """Tokenize once; return the student tokens and the target-branch
        latents h = LN(f_theta_bar(P)) for every token (Eq. 1)."""
        geom, anchor_xyz = self.tokenizer.gather(clip)
        tok = self.tokenizer.assemble(geom, anchor_xyz)
        if tok["M"] != self.num_anchors:
            raise ValueError(f"tokenizer produced M={tok['M']} anchors, model expects {self.num_anchors}")
        with torch.no_grad():
            ttok = self.target_tokenizer.assemble(geom, anchor_xyz)
            h = self.target_encoder(ttok["tokens"], ttok["xyzt"])
            h = F.layer_norm(h, (h.shape[-1],))
        return tok, h

    def make_query(self, mode, true_xyzt, last_xyz, tgt_idx):
        """Target query u~ for ``mode`` (see the class docstring) ->
        (query xyzt [B, Q, 4], slot index [B, Q] or None)."""
        tau = true_xyzt[..., 3:]
        if mode == "target":
            return true_xyzt, None
        if mode == "last_seen":
            return torch.cat([last_xyz, tau], dim=-1), None
        if mode == "noised":
            noisy = true_xyzt[..., :3] + self.query_noise_std * torch.randn_like(true_xyzt[..., :3])
            return torch.cat([noisy, tau], dim=-1), None
        if mode == "time_only":
            return torch.cat([torch.zeros_like(last_xyz), tau], dim=-1), tgt_idx % self.num_anchors
        raise ValueError(mode)

    def frame_states(self, la_in: torch.Tensor, tprime: int) -> torch.Tensor:
        """s_t = (1/M) sum_j LN(.)_{t,j} (Eq. 5) -> [B, T', C]. ``la_in`` is
        already layer-normed."""
        b = la_in.shape[0]
        return la_in.view(b, tprime, self.num_anchors, -1).mean(dim=2).float()

    def posterior(self, states: torch.Tensor, t_idx: torch.Tensor, detach: bool = False):
        """q_psi(a_t | delta_t) for anchors ``t_idx [K]`` (all >= 1) ->
        (mu, logvar) [B, K, A]."""
        cur, prev = states[:, t_idx], states[:, t_idx - 1]
        delta = cur - prev
        if detach:
            delta, prev = delta.detach(), prev.detach()
        return self.idm(delta, prev)

    def context_state(self, z_c: torch.Tensor) -> torch.Tensor:
        """s^c: layer-normed, frame-pooled context latents of the last context
        anchor t_c. The context tokens are in ascending anchor order, so t_c's
        M tokens are the last M rows of z_c."""
        last = z_c[:, -self.num_anchors :]
        return F.layer_norm(last, (last.shape[-1],)).mean(dim=1).float()

    def prior_params(self, s_c: torch.Tensor, k: torch.Tensor):
        """p_omega(a_t | s^c, k) -> (mu, logvar) [B, K, A]; N(0, I) for the
        fixed prior."""
        if self.prior is None:
            z = s_c.new_zeros(s_c.shape[0], k.shape[1], self.idm.act_dim)
            return z, z
        return self.prior(s_c, k)

    def forward(self, clip: torch.Tensor, t0: int | None = None) -> dict:
        """One pretraining step's forward (Algorithm S1). ``t0`` is the shared
        starting target anchor of the causal mask (sampled here if None).
        Returns the loss components; ``total_loss`` combines them."""
        tok, h = self.encode(clip)
        b, tprime = clip.shape[0], tok["T"]
        z_f = self.encoder(tok["tokens"], tok["xyzt"]) if self.needs_full_pass else None
        if "causal" in self.masks and t0 is None:
            t0 = self.causal.sample_t0(tprime, np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()))))

        out = {"lat_per_mask": {}}
        for kind in self.masks:
            if kind == "causal":
                visible, target = self.causal(b, tprime, self.num_anchors, t0, clip.device)
            else:
                visible, target = self.block(tok["anchor_xyz"])
            res = self._masked_prediction(kind, tok, h, z_f, visible, target, t0)
            out["lat_per_mask"][kind] = res.pop("lat")
            out.update(res)
        out["lat"] = torch.stack(list(out["lat_per_mask"].values())).mean()

        if self.vicreg != "none":
            c = z_f.shape[-1]
            if self.vicreg == "frame":
                rows = z_f.view(b, tprime, self.num_anchors, c).mean(dim=2).reshape(-1, c)
            else:
                rows = z_f.reshape(-1, c)
                if rows.shape[0] > 65536:
                    rows = rows[torch.randperm(rows.shape[0], device=rows.device)[:65536]]
            out["vic"] = vicreg_var_cov(rows)
        out["t0"] = t0
        return out

    def _masked_prediction(self, kind, tok, h, z_f, visible, target, t0):
        b, tprime, m = visible.shape
        tokens, xyzt, anchor_xyz = tok["tokens"], tok["xyzt"], tok["anchor_xyz"]
        ctx_idx = visible.reshape(b, -1).nonzero()[:, 1].view(b, -1)
        tgt_idx = target.reshape(b, -1).nonzero()[:, 1].view(b, -1)
        ctx_xyzt = gather_tokens(xyzt, ctx_idx)
        z_c = self.encoder(gather_tokens(tokens, ctx_idx), ctx_xyzt)
        h_t = gather_tokens(h, tgt_idx)
        true_xyzt = gather_tokens(xyzt, tgt_idx)
        ls = last_seen_frames(visible, self.last_seen_rule)
        if bool((gather_tokens(ls.reshape(b, -1), tgt_idx) < 0).any()):
            raise RuntimeError("a target has no preceding visible frame")
        last_xyz = gather_tokens(anchors_at(anchor_xyz, ls).reshape(b, -1, 3), tgt_idx)
        qry, slot = self.make_query(self.query, true_xyzt, last_xyz, tgt_idx)

        res = {}
        act = None
        if kind == "causal" and self.la_enabled:
            act, kl = self._latent_actions(h, z_f, z_c, tprime, t0)
            res.update(kl)
        pred, dpos = self.predictor(z_c, ctx_xyzt, qry, act, slot)
        res["lat"] = (pred.float() - h_t.float()).abs().mean()
        if kind == "causal" and dpos is not None:
            k = tgt_idx.shape[1] // m
            c_hat = (last_xyz + dpos.float()).view(b, k, m, 3)
            res["pos"] = frame_chamfer(c_hat, true_xyzt[..., :3].view(b, k, m, 3)).mean()
        return res

    def _latent_actions(self, h, z_f, z_c, tprime, t0):
        """a_t for the K causal targets, plus the KL statistics (Sec. 3.3)."""
        b, m, kk = h.shape[0], self.num_anchors, self.causal.num_targets
        dev = h.device
        t_idx = torch.arange(t0, t0 + kk, device=dev)
        t_c = t0 - self.causal.gap - 1
        k = (t_idx - t_c).view(1, kk).expand(b, kk)
        la_in = F.layer_norm(z_f, (z_f.shape[-1],)) if self.la_source == "student" else h
        states = self.frame_states(la_in, tprime)
        mu_q, lv_q = self.posterior(states, t_idx)
        a = mu_q + torch.randn_like(mu_q) * (0.5 * lv_q).exp() if self.training else mu_q
        mu_qk, lv_qk = self.posterior(states, t_idx, detach=True) if self.la_kl_detach else (mu_q, lv_q)
        s_c = self.context_state(z_c)
        res = {}
        if self.prior is not None:
            mu_p, lv_p = self.prior(s_c.detach() if self.la_kl_detach else s_c, k)
            res["kl_q_dim"] = gaussian_kl(mu_qk, lv_qk, mu_p.detach(), lv_p.detach()).mean(dim=(0, 1))
            res["kl_p"] = gaussian_kl(mu_qk.detach(), lv_qk.detach(), mu_p, lv_p).sum(-1).mean()
            if self.training:
                if self.la_prior_sub_p > 0:
                    mu_sub = self.prior(s_c, k)[0] if self.la_kl_detach else mu_p
                    sub = torch.rand(b, 1, 1, device=dev) < self.la_prior_sub_p
                    a = torch.where(sub, mu_sub.to(a.dtype), a)
            else:
                a = mu_p
        else:
            res["kl_q_dim"] = std_normal_kl(mu_qk, lv_qk).mean(dim=(0, 1))
            if not self.training:
                a = torch.zeros_like(a)
        return a.repeat_interleave(m, dim=1), res


def total_loss(out: dict, loss_cfg: dict):
    """Eq. 8: L = L_lat + lambda_pos L_pos + beta_q L_q + beta_p L_p
    + lambda_VIC L_VIC, with L_q = sum_i max(E[KL_i], lambda_fb).
    Returns (total, logs); logs are detached tensors (no host sync)."""
    total = out["lat"]
    logs = {"lat": out["lat"]}
    for kind, v in out["lat_per_mask"].items():
        logs[f"lat_{kind}"] = v
    if "pos" in out:
        total = total + float(loss_cfg["lambda_pos"]) * out["pos"]
        logs["pos"] = out["pos"]
    if "kl_q_dim" in out:
        kl_dim = out["kl_q_dim"]
        fb = float(loss_cfg["free_bits"])
        total = total + float(loss_cfg["beta_q"]) * kl_dim.clamp_min(fb).sum()
        logs["kl_q"] = kl_dim.sum()
        logs["kl_active_dims"] = (kl_dim > 0.5 * fb).sum()
    if "kl_p" in out:
        total = total + float(loss_cfg["beta_p"]) * out["kl_p"]
        logs["kl_p"] = out["kl_p"]
    if "vic" in out:
        total = total + float(loss_cfg["lambda_vic"]) * out["vic"]
        logs["vic"] = out["vic"]
    logs["total"] = total
    return total, {k: v.detach() for k, v in logs.items()}
