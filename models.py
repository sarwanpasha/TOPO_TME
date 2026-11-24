"""Learnable marked persistence (MPH) and GNN baselines (pure torch)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from topo_core import persistence_pairs


def to_t(im, scale, device="cpu"):
    """cache tensors on the image dict (once)."""
    if "_t" not in im:
        tps = torch.from_numpy(im["types"].astype(np.int64))
        im["_t"] = dict(
            xy=torch.from_numpy(im["coords"].astype(np.float32) / scale),
            mk=torch.from_numpy(im["markers"].astype(np.float32)),
            types=tps,
            e=torch.from_numpy(im["edges"]),
            ki=torch.from_numpy(im["knn_idx"].astype(np.int64)),
            kd=torch.from_numpy(im["knn_d"].astype(np.float32) / scale),
        )
        xy = im["_t"]["xy"]; e = im["_t"]["e"]
        im["_t"]["el"] = (xy[e[:, 0]] - xy[e[:, 1]]).norm(dim=1)
    return im["_t"]


def phen(t, n_types, use_markers=True, use_types=True):
    parts = []
    if use_types:
        parts.append(F.one_hot(t["types"], n_types).float())
    if use_markers:
        parts.append(t["mk"])
    return torch.cat(parts, 1)


class PersImage(nn.Module):
    """differentiable persistence image on (log1p b, log-persistence) grid."""
    def __init__(self, G=6, lim=2.5, sig=0.3):
        super().__init__()
        self.register_buffer("c", torch.linspace(0, lim, G))
        self.sig = sig; self.G = G

    def forward(self, b, d, n):
        lb = torch.log1p(b); lp = torch.log1p(d) - lb
        if lb.numel() == 0:
            return torch.zeros(self.G * self.G + 2)
        wb = torch.exp(-(lb[:, None] - self.c[None]) ** 2 / (2 * self.sig ** 2))
        wp = torch.exp(-(lp[:, None] - self.c[None]) ** 2 / (2 * self.sig ** 2))
        img = torch.einsum("p,pi,pj->ij", lp, wb, wp).reshape(-1)
        summ = torch.stack([lp.sum(), (lp ** 2).sum()])
        return torch.cat([img, summ]) / n * 10.0


class MarkedPH(nn.Module):
    """K learned filtrations: vertex = phenotype-weighted DTM, edge = phenotype-warped length.
    f_e = max(w_e, f_u, f_v);  w_e = |x_u-x_v| exp(M tanh(.)) ; exact H0 + H1 on Delaunay complex."""
    def __init__(self, p_dim, comp_dim, out_dim, K=4, M=1.0, dims=(0, 1), use_vertex=True,
                 use_edge=True, use_comp=True, G=6, emb=32, hidden=64, dropout=0.2, gnn=None):
        super().__init__()
        self.K, self.M, self.dims = K, M, dims
        self.use_vertex, self.use_edge, self.use_comp = use_vertex, use_edge, use_comp
        self.phi = nn.Sequential(nn.Linear(p_dim, emb), nn.ReLU(), nn.Linear(emb, emb))
        self.A = nn.Linear(emb, K); self.B = nn.Linear(emb, K, bias=False)
        self.av = nn.Linear(emb, K)
        self.gam = nn.Parameter(torch.zeros(K))
        for l in (self.A, self.B, self.av):
            nn.init.normal_(l.weight, std=0.05)
        nn.init.zeros_(self.A.bias); nn.init.zeros_(self.av.bias)
        self.pi = PersImage(G)
        fdim = K * len(dims) * (G * G + 2)
        self.gnn = gnn
        in_dim = fdim + (comp_dim if use_comp else 0) + (gnn.out_dim if gnn is not None else 0)
        self.norm = nn.LayerNorm(in_dim)
        self.head = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, out_dim))

    def filtrations(self, t, p):
        z = self.phi(p)
        e = t["e"]; el = t["el"]
        N = p.shape[0]
        if self.use_edge:
            zu, zv = z[e[:, 0]], z[e[:, 1]]
            g = self.M * torch.tanh(self.A(zu + zv) + self.B(zu * zv))
            w = el[:, None] * torch.exp(g)
        else:
            w = el[:, None].expand(-1, self.K)
        if self.use_vertex:
            a = torch.exp(self.M * torch.tanh(self.av(z)))       # (N,K) relevance
            ak = a[t["ki"]]                                       # (N,k,K)
            dtm = torch.sqrt((ak * t["kd"][:, :, None] ** 2).sum(1) / ak.sum(1) + 1e-8)
            fv = torch.sigmoid(self.gam)[None] * dtm
        else:
            fv = torch.zeros(N, self.K)
        fe = torch.maximum(w, torch.maximum(fv[e[:, 0]], fv[e[:, 1]]))
        return fv, fe

    def topo_features(self, im, t, p):
        fv, fe = self.filtrations(t, p)
        fv_np = fv.detach().double().numpy(); fe_np = fe.detach().double().numpy()
        N = p.shape[0]
        feats = []
        for k in range(self.K):
            h0b, h0d, h1b, h1d = persistence_pairs(fv_np[:, k], fe_np[:, k], im["edges"],
                                                   im["tri_edges"], im["faces"])
            if 0 in self.dims:
                feats.append(self.pi(fv[torch.from_numpy(h0b), k], fe[torch.from_numpy(h0d), k], N))
            if 1 in self.dims:
                feats.append(self.pi(fe[torch.from_numpy(h1b), k], fe[torch.from_numpy(h1d), k], N))
        return torch.cat(feats)

    def forward(self, im, t, p, comp):
        parts = [self.topo_features(im, t, p)]
        if self.use_comp:
            parts.append(comp)
        if self.gnn is not None:
            parts.append(self.gnn.embed(t, p))
        return self.head(self.norm(torch.cat(parts)))


def scatter_sum(src, idx, n):
    out = torch.zeros(n, src.shape[1]); return out.index_add_(0, idx, src)


class GNN(nn.Module):
    """GIN or GAT on the Delaunay graph; mean+max pooling."""
    def __init__(self, p_dim, out_dim, kind="gin", hidden=64, layers=3, dropout=0.2, head=True):
        super().__init__()
        self.kind = kind; self.out_dim = 2 * hidden
        self.inp = nn.Linear(p_dim, hidden)
        if kind == "gin":
            self.layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                                       nn.Linear(hidden, hidden)) for _ in range(layers)])
            self.eps = nn.Parameter(torch.zeros(layers))
        else:
            self.W = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
            self.att = nn.ParameterList([nn.Parameter(torch.randn(2 * hidden) * 0.1) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, out_dim)) if head else None

    def embed(self, t, p):
        h = F.relu(self.inp(p)); N = h.shape[0]
        e = t["e"]; src = torch.cat([e[:, 0], e[:, 1]]); dst = torch.cat([e[:, 1], e[:, 0]])
        for l in range(len(self.norms)):
            if self.kind == "gin":
                agg = scatter_sum(h[src], dst, N)
                hn = self.layers[l]((1 + self.eps[l]) * h + agg)
            else:
                wh = self.W[l](h); H = wh.shape[1]
                s = F.leaky_relu((wh[src] * self.att[l][:H]).sum(1) + (wh[dst] * self.att[l][H:]).sum(1), 0.2)
                s = s - s.max()
                a = torch.exp(s)
                den = torch.zeros(N).index_add_(0, dst, a) + 1e-9
                hn = scatter_sum(wh[src] * (a / den[dst])[:, None], dst, N)
            h = self.norms[l](h + F.relu(hn))
        return torch.cat([h.mean(0), h.max(0).values])

    def forward(self, im, t, p, comp):
        return self.head(self.embed(t, p))


METHODS = {
    # name: constructor kwargs for MarkedPH (or 'gnn')
    "mph":          dict(K=4, dims=(0, 1), use_vertex=True, use_edge=True, use_comp=True),
    "mph_nocomp":   dict(K=4, dims=(0, 1), use_vertex=True, use_edge=True, use_comp=False),
    "mph_K1":       dict(K=1, dims=(0, 1), use_vertex=True, use_edge=True, use_comp=True),
    "mph_h0":       dict(K=4, dims=(0,), use_vertex=True, use_edge=True, use_comp=True),
    "mph_h1":       dict(K=4, dims=(1,), use_vertex=True, use_edge=True, use_comp=True),
    "mph_novertex": dict(K=4, dims=(0, 1), use_vertex=False, use_edge=True, use_comp=True),
    "mph_noedge":   dict(K=4, dims=(0, 1), use_vertex=True, use_edge=False, use_comp=True),
    "mph_fixed":    dict(K=1, dims=(0, 1), use_vertex=False, use_edge=False, use_comp=True),
    "mph_fixed_nocomp": dict(K=1, dims=(0, 1), use_vertex=False, use_edge=False, use_comp=False),
    "paph":         dict(K=1, dims=(0,), use_vertex=False, use_edge=True, use_comp=False),  # previous paper
    "mph_gin":      dict(K=4, dims=(0, 1), use_vertex=True, use_edge=True, use_comp=True, gnn="gin"),
    "gin": "gin", "gat": "gat",
}


def build_model(method, p_dim, comp_dim, out_dim):
    spec = METHODS[method]
    if isinstance(spec, str):
        return GNN(p_dim, out_dim, kind=spec)
    spec = dict(spec)
    g = spec.pop("gnn", None)
    gnn = GNN(p_dim, out_dim, kind=g, head=False) if g else None
    return MarkedPH(p_dim, comp_dim, out_dim, gnn=gnn, **spec)


class FieldPH(nn.Module):
    """Learned phenotype-field persistence (v2).
    K heads; head k: vertex value f_v = -(w_k . p_v) (learned projection of marker/type vector),
    edge value f_e = max(f_u, f_v) + softplus(beta_k) * |x_u - x_v|  (lower-star + geometry).
    Exact H0 + H1 on the Delaunay complex; persistence images on a fixed grid."""
    def __init__(self, p_dim, comp_dim, out_dim, K=8, use_comp=True, hidden=64, dropout=0.3,
                 GB=(-3, 3), GP=(0, 4), G=5, sig=0.75, init_idx=None):
        super().__init__()
        self.K = K; self.use_comp = use_comp
        W = torch.randn(p_dim, K) * 0.05
        if init_idx is not None:
            for k, j in enumerate(init_idx[:K]):
                W[j, k] += 1.0
        self.W = nn.Parameter(W)
        self.beta = nn.Parameter(torch.full((K,), 0.0))   # softplus(0)=0.69
        self.register_buffer("cb", torch.linspace(GB[0], GB[1], G))
        self.register_buffer("cp", torch.linspace(GP[0], GP[1], G))
        self.sig = sig; self.G = G
        self.fdim = K * 2 * (G * G + 3)
        in_dim = self.fdim + (comp_dim if use_comp else 0)
        self.norm = nn.LayerNorm(in_dim)
        self.head = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def pimg(self, b, d, n):
        if b.numel() == 0:
            return torch.zeros(self.G * self.G + 3)
        p = d - b
        wb = torch.exp(-(b[:, None] - self.cb[None]) ** 2 / (2 * self.sig ** 2))
        wp = torch.exp(-(p[:, None] - self.cp[None]) ** 2 / (2 * self.sig ** 2))
        img = torch.einsum("p,pi,pj->ij", p, wb, wp).reshape(-1)
        return torch.cat([img, torch.stack([p.sum(), torch.sigmoid(4 * (p - 0.5)).sum(), (p * b).sum()])]) / n * 100.0

    def topo_features(self, im, t, p):
        z = torch.clamp(p @ self.W, -4, 4)          # (N,K)
        fv = -z
        e = t["e"]; el = t["el"]
        fe = torch.maximum(fv[e[:, 0]], fv[e[:, 1]]) + F.softplus(self.beta)[None] * el[:, None]
        fv_np = fv.detach().double().numpy(); fe_np = fe.detach().double().numpy()
        N = p.shape[0]; out = []
        for k in range(self.K):
            h0b, h0d, h1b, h1d = persistence_pairs(fv_np[:, k], fe_np[:, k], im["edges"], im["tri_edges"], im["faces"])
            out.append(self.pimg(fv[torch.from_numpy(h0b), k], fe[torch.from_numpy(h0d), k], N))
            out.append(self.pimg(fe[torch.from_numpy(h1b), k], fe[torch.from_numpy(h1d), k], N))
        return torch.cat(out)

    def forward(self, im, t, p, comp):
        parts = [self.topo_features(im, t, p)]
        if self.use_comp:
            parts.append(comp)
        return self.head(self.norm(torch.cat(parts)))


METHODS.update({"mph2": "field", "mph2_nocomp": "field_nocomp", "mph2_stack": "field", "mph2_stack2": "field"})
_build_old = build_model


def build_model(method, p_dim, comp_dim, out_dim, init_idx=None):
    spec = METHODS[method]
    if spec in ("field", "field_nocomp"):
        return FieldPH(p_dim, comp_dim, out_dim, use_comp=(spec == "field"), init_idx=init_idx)
    return _build_old(method, p_dim, comp_dim, out_dim)


# ---- GIN with a composition skip connection and a small hyperparameter grid (tuned by inner validation) ----
class GNNc(nn.Module):
    """GIN embedding (mean+max pooling) concatenated with the image composition vector before the head, so that the
    graph model has at least the information of the composition model."""
    def __init__(self, p_dim, comp_dim, out_dim, hidden=64, layers=3, dropout=0.2):
        super().__init__()
        self.g = GNN(p_dim, out_dim, kind="gin", hidden=hidden, layers=layers, dropout=dropout, head=False)
        self.c = nn.Sequential(nn.Linear(comp_dim, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, im, t, p, comp):
        return self.head(torch.cat([self.g.embed(t, p), self.c(comp)]))


GNNC_GRID = {"ginc": dict(hidden=64, layers=3, dropout=0.2), "ginc_l2": dict(hidden=64, layers=2, dropout=0.2),
             "ginc_h128": dict(hidden=128, layers=3, dropout=0.2), "ginc_d5": dict(hidden=64, layers=3, dropout=0.5)}
METHODS.update({k: "ginc" for k in GNNC_GRID})
_build_prev = build_model


def build_model(method, p_dim, comp_dim, out_dim, init_idx=None):
    if method in GNNC_GRID:
        return GNNc(p_dim, comp_dim, out_dim, **GNNC_GRID[method])
    return _build_prev(method, p_dim, comp_dim, out_dim, init_idx=init_idx)
