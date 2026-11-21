"""TopoNet3: strong non-topological backbone + gated topological / GNN residuals.

out = lin(x_b) + mlp(x_b)                                  (backbone on comp, marker means, quantiles)
    + g_t * head_t(stats(PH of learned + per-type fields))   (gate g_t init ~0)
    + g_g * head_g(attn-pooled 2-layer GIN on Delaunay)      (gate g_g init ~0)

Importing this module registers methods topo3 / topo3_nognn / topo3_notopo in models.METHODS
and wraps models.build_model (import it BEFORE `from models import build_model`).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from models import scatter_sum
from topo_core import persistence_pairs

TOPO3 = {"topo3": dict(use_topo=True, use_gnn=True),
         "topo3_nognn": dict(use_topo=True, use_gnn=False),
         "topo3_notopo": dict(use_topo=False, use_gnn=False),
         "topo3g": dict(use_topo=True, use_gnn=False, gate0=0.3)}


class TopoNet3(nn.Module):
    def __init__(self, p_dim, comp_dim, out_dim, n_types, init_idx=None, use_topo=True, use_gnn=True, gate0=0.01,
                 K=8, quantiles=(0.1, 0.5, 0.9), hidden=32, drop_in=0.2, drop=0.5, gnn_hidden=32,
                 topo_hidden=16, rank_w=0.5, wd=5e-2):
        super().__init__()
        self.nT = n_types; self.nM = p_dim - n_types
        self.use_topo, self.use_gnn = use_topo, use_gnn
        self.rank_w, self.wd = rank_w, wd
        self.register_buffer("qs", torch.tensor(quantiles, dtype=torch.float32))
        nq = (len(quantiles) + 1) * self.nM if self.nM > 0 else 0
        self.base_dim = comp_dim + nq
        self.register_buffer("base_mu", torch.zeros(self.base_dim))
        self.register_buffer("base_sd", torch.ones(self.base_dim))
        # ---------------- backbone ----------------
        self.drop_in = nn.Dropout(drop_in)
        self.lin = nn.Linear(self.base_dim, out_dim)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)
        self.mlp = nn.Sequential(nn.Linear(self.base_dim, hidden), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(hidden, out_dim))
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.1); self.mlp[-1].bias.zero_()
        # ---------------- topology ----------------
        if use_topo:
            W = torch.randn(p_dim, K) * 0.05
            if init_idx is not None:
                for k, j in enumerate(list(init_idx)[:K]):
                    W[int(j), k] += 1.0
            self.W = nn.Parameter(W)
            Wfix = torch.zeros(p_dim, n_types)
            Wfix[torch.arange(n_types), torch.arange(n_types)] = 2.0   # f_v = -2 on cells of type k, 0 elsewhere
            self.register_buffer("Wfix", Wfix)
            self.nF = K + n_types
            self.beta = nn.Parameter(torch.zeros(self.nF))                       # softplus(0)=0.69
            self.tau_raw = nn.Parameter(torch.full((self.nF, 2), math.log(math.expm1(0.5))))  # thr 0.5
            self.tdim = self.nF * 2 * 4
            self.register_buffer("topo_mu", torch.zeros(self.tdim))
            self.register_buffer("topo_sd", torch.ones(self.tdim))
            self.topo_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.tdim, topo_hidden), nn.ReLU(),
                                           nn.Dropout(0.3), nn.Linear(topo_hidden, out_dim))
            self.gate_t = nn.Parameter(torch.full((out_dim,), float(gate0)))
        # ---------------- GIN ----------------
        if use_gnn:
            gh = gnn_hidden
            self.gin_in = nn.Linear(p_dim, gh)
            self.gin = nn.ModuleList([nn.Sequential(nn.Linear(gh, gh), nn.ReLU(), nn.Linear(gh, gh)) for _ in range(2)])
            self.gin_eps = nn.Parameter(torch.zeros(2))
            self.gin_norm = nn.ModuleList([nn.LayerNorm(gh) for _ in range(2)])
            self.att = nn.Linear(gh, 1)
            self.gnn_out = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * gh, out_dim))
            self.gate_g = nn.Parameter(torch.full((out_dim,), 0.01))
        # read by train.py (backward compatible: other models have no train_cfg)
        self.train_cfg = dict(cosine=True, small_n=200, small_mult=2.0, patience=16)

    # ------------------------------------------------ features
    def base_feats(self, t, comp):
        if self.nM == 0:
            return comp
        if "_b3" not in t:
            mk = t["mk"]
            q = torch.quantile(mk, self.qs, dim=0).reshape(-1)          # (nq*nM,)
            fp = (mk > 1.0).float().mean(0)                              # (nM,) fraction "positive"
            t["_b3"] = torch.cat([q, fp]).detach()
        return torch.cat([comp, t["_b3"]])

    @staticmethod
    def _stats(b, d, n, tau):
        if b.numel() == 0:
            return torch.zeros(4)
        pers = (d - b).clamp(min=0)
        tot = pers.sum()
        wb = (pers * b).sum() / (tot + 1e-6)                           # persistence-weighted mean birth
        cnt = torch.sigmoid((pers - tau) / 0.1).sum()                   # soft count above threshold
        return torch.stack([tot / n * 10.0, wb, cnt / n * 10.0, pers.max()])

    def topo_stats(self, im, t, p):
        Wall = torch.cat([self.W, self.Wfix], 1)                       # (p_dim, nF)
        fv = -torch.clamp(p @ Wall, -4, 4)                              # (N, nF)
        e = t["e"]; el = t["el"]
        fe = torch.maximum(fv[e[:, 0]], fv[e[:, 1]]) + F.softplus(self.beta)[None] * el[:, None]  # (E, nF)
        fv_np = fv.detach().double().numpy(); fe_np = fe.detach().double().numpy()
        tau = F.softplus(self.tau_raw)
        N = p.shape[0]; out = []
        for k in range(self.nF):
            h0b, h0d, h1b, h1d = persistence_pairs(fv_np[:, k], fe_np[:, k], im["edges"], im["tri_edges"], im["faces"])
            out.append(self._stats(fv[torch.from_numpy(h0b), k], fe[torch.from_numpy(h0d), k], N, tau[k, 0]))
            out.append(self._stats(fe[torch.from_numpy(h1b), k], fe[torch.from_numpy(h1d), k], N, tau[k, 1]))
        return torch.cat(out)                                           # (nF*8,)

    def gnn_embed(self, t, p):
        h = F.relu(self.gin_in(p)); N = h.shape[0]
        e = t["e"]; src = torch.cat([e[:, 0], e[:, 1]]); dst = torch.cat([e[:, 1], e[:, 0]])
        for l in range(2):
            agg = scatter_sum(h[src], dst, N)
            h = self.gin_norm[l](h + self.gin[l]((1 + self.gin_eps[l]) * h + agg))
        a = torch.softmax(self.att(h).squeeze(1), 0)                   # (N,)
        return torch.cat([(a[:, None] * h).sum(0), h.mean(0)])         # (2*gh,)

    def topo_features(self, im, t, p):
        """for stack_predict compatibility."""
        return self.topo_stats(im, t, p) if self.use_topo else torch.zeros(1)

    # ------------------------------------------------ forward
    def forward(self, im, t, p, comp):
        xb = ((self.base_feats(t, comp) - self.base_mu) / self.base_sd).clamp(-5, 5)
        xb = self.drop_in(xb)
        out = self.lin(xb) + self.mlp(xb)
        if self.use_topo:
            xt = ((self.topo_stats(im, t, p) - self.topo_mu) / self.topo_sd).clamp(-5, 5)
            out = out + self.gate_t * self.topo_head(xt)
        if self.use_gnn:
            out = out + self.gate_g * self.gnn_out(self.gnn_embed(t, p))
        return out

    # ------------------------------------------------ training hooks (used by train.py if present)
    @torch.no_grad()
    def init_stats(self, samples):
        """samples: list of (im, t, p, comp) from TRAINING images; sets feature standardisation."""
        if not samples:
            return
        B = torch.stack([self.base_feats(t, c) for im, t, p, c in samples])
        self.base_mu.copy_(B.mean(0)); self.base_sd.copy_(B.std(0).clamp(min=1e-3) if len(B) > 1 else torch.ones_like(self.base_sd))
        if self.use_topo:
            T = torch.stack([self.topo_stats(im, t, p) for im, t, p, c in samples])
            self.topo_mu.copy_(T.mean(0)); self.topo_sd.copy_(T.std(0).clamp(min=1e-3) if len(T) > 1 else torch.ones_like(self.topo_sd))

    def param_groups(self):
        decay, nodecay = [], []
        for n, q in self.named_parameters():
            (decay if (q.ndim == 2 and n != "W") else nodecay).append(q)
        return [{"params": decay, "weight_decay": self.wd}, {"params": nodecay, "weight_decay": 0.0}]

    def surv_aux(self, risk, time, event):
        """pairwise logistic C-index surrogate: comparable pairs (event_i, t_i < t_j) want r_i > r_j."""
        mask = (time[:, None] < time[None, :]) & (event[:, None] > 0)
        if mask.sum() == 0:
            return risk.sum() * 0.0
        diff = risk[:, None] - risk[None, :]
        return self.rank_w * F.softplus(-diff)[mask].mean()


# ---------------------------------------------------------------- registration
models.METHODS.update({k: k for k in TOPO3})
_prev_build = models.build_model


def build_model(method, p_dim, comp_dim, out_dim, init_idx=None, n_types=None, **kw):
    if method in TOPO3:
        if n_types is None:
            raise ValueError("TopoNet3 needs n_types")
        return TopoNet3(p_dim, comp_dim, out_dim, n_types, init_idx=init_idx, **TOPO3[method], **kw)
    return _prev_build(method, p_dim, comp_dim, out_dim, init_idx=init_idx)


if getattr(models.build_model, "__module__", None) != __name__:
    models.build_model = build_model
