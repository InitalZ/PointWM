from __future__ import annotations

import torch
import torch.nn as nn


class InverseDynamics(nn.Module):
    """IDM q_psi(a_t | delta_t) (Sec. 3.3, Supp. Sec. A.1): LayerNorm, a
    two-layer MLP (width 256, GELU), and mean / log-variance heads of a
    diagonal Gaussian over a_t in R^{d_a}, log-variance clamped to [-8, 4].

    ``use_state=True`` is the Table 6 variant that reads [s_{t-1}; delta_t].
    """

    def __init__(self, dim: int, act_dim: int = 8, hidden: int = 256, use_state: bool = False):
        super().__init__()
        self.act_dim = act_dim
        self.use_state = bool(use_state)
        in_dim = 2 * dim if self.use_state else dim
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.mu = nn.Linear(hidden, act_dim)
        self.logvar = nn.Linear(hidden, act_dim)

    def forward(self, delta: torch.Tensor, prev: torch.Tensor | None = None):
        x = torch.cat([prev, delta], dim=-1) if self.use_state else delta
        h = self.net(self.norm(x.float()))
        return self.mu(h), self.logvar(h).clamp(-8.0, 4.0)


class LatentActionPrior(nn.Module):
    """Learned prior p_omega(a_t | s^c, k) (Sec. 3.3, Supp. Sec. A.1):
    LayerNorm and a width-256 linear layer on s^c, plus a learned embedding of
    the horizon k = t - t_c (one entry per integer k), then GELU, linear,
    GELU and mean / log-variance heads. The heads are zero-initialised, so the
    prior starts at N(0, I)."""

    def __init__(self, dim: int, act_dim: int = 8, hidden: int = 256, max_k: int = 64):
        super().__init__()
        self.max_k = int(max_k)
        self.norm = nn.LayerNorm(dim)
        self.inp = nn.Linear(dim, hidden)
        self.k_emb = nn.Embedding(self.max_k, hidden)
        self.net = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.mu = nn.Linear(hidden, act_dim)
        self.logvar = nn.Linear(hidden, act_dim)
        for m in (self.mu, self.logvar):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, k: torch.Tensor):
        """state [B, C], k [B, K] (long) -> (mu, logvar), each [B, K, A]."""
        if int(k.max()) >= self.max_k or int(k.min()) < 0:
            raise ValueError(f"horizon k out of range [0, {self.max_k})")
        h = self.inp(self.norm(state.float())).unsqueeze(1) + self.k_emb(k)
        h = self.net(h)
        return self.mu(h), self.logvar(h).clamp(-8.0, 4.0)


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    """Per-dimension KL(N(mu_q, e^logvar_q) || N(mu_p, e^logvar_p)), nats."""
    return 0.5 * (logvar_p - logvar_q + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp() - 1.0)


def std_normal_kl(mu, logvar):
    """Per-dimension KL(N(mu, e^logvar) || N(0, 1)), nats."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
