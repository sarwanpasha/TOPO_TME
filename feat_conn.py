"""Plain graph-connectivity baseline (no homology). For every cell type and for the high/low level set of every marker
(z > 1, z < -1): connected components of the induced Delaunay subgraph keeping edges shorter than L x the median edge
length, L in {1, 1.5, 2.5, inf}: [largest component / all cells, largest component / in-set cells, components per 100
cells], plus the mean within-set degree. The same permutation null as TopoNull gives a calibrated version, so the
comparison with persistence isolates the contribution of homology.
NAME conn: [raw | calibrated], each 13 per type + 26 per marker."""
import zlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

NAME = "conn"
LS = (1.0, 1.5, 2.5, np.inf)
K_NULL, EPS = 8, 0.05


def _set_feats(vin, E, el, N):
    n_in = int(vin.sum())
    if n_in < 2:
        return np.zeros(3 * len(LS) + 1)
    both = vin[E[:, 0]] & vin[E[:, 1]]
    out = []
    for L in LS:
        ek = both & (el <= L)
        A = coo_matrix((np.ones(int(ek.sum())), (E[ek, 0], E[ek, 1])), shape=(N, N))
        _, lab = connected_components(A, directed=False)
        cnt = np.bincount(lab[vin]); cnt = cnt[cnt > 0]
        out += [cnt.max() / N, cnt.max() / n_in, len(cnt) / N * 100.0]
    out.append(2.0 * both.sum() / n_in)
    return np.array(out)


def _all(t, Z, E, el, N, n_types):
    out = [_set_feats(t == k, E, el, N) for k in range(n_types)]
    for m in range(Z.shape[1]):
        out.append(_set_feats(Z[:, m] > 1.0, E, el, N)); out.append(_set_feats(Z[:, m] < -1.0, E, el, N))
    return np.concatenate(out)


def compute(im, n_types):
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(im["coords"][:50]).tobytes()) + 3)
    E = im["edges"]; xy = im["coords"].astype(np.float64) / im["scale"]; N = len(xy)
    el = np.linalg.norm(xy[E[:, 0]] - xy[E[:, 1]], axis=1)
    t = np.asarray(im["types"]).astype(int); Z = np.clip(im["markers"].astype(np.float64), -4, 4)
    obs = _all(t, Z, E, el, N, n_types)
    groups = [np.flatnonzero(t == k) for k in range(n_types)]
    nul = []
    for _ in range(K_NULL):
        tp = rng.permutation(t); Zp = Z.copy()
        for g in groups:
            if len(g) > 1:
                Zp[g] = Z[rng.permutation(g)]
        nul.append(_all(tp, Zp, E, el, N, n_types))
    nul = np.stack(nul); mu, sd = nul.mean(0), nul.std(0)
    z = np.clip(np.nan_to_num((obs - mu) / (sd + EPS * (np.abs(mu) + 1e-3) + 1e-6)), -8, 8)
    return np.concatenate([np.nan_to_num(obs), z]).astype(np.float32)
