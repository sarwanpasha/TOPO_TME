"""Density-aware TopoNull. Same descriptors as feat_null, but the permutation null also preserves the relation
between cell identity and local cell density: cells are stratified into quintiles of local density (mean length of
their incident Delaunay edges); cell-type labels are permuted WITHIN density strata, and marker vectors are permuted
within (cell type x density stratum). Layout of the output is identical to feat_null (type fields, cross, markers)."""
import zlib
import numpy as np
import feat_null as FN

NAME = "znulld"
N_STRATA = 5


def density_strata(im):
    E = im["edges"]; xy = im["coords"].astype(np.float64)
    el = np.linalg.norm(xy[E[:, 0]] - xy[E[:, 1]], axis=1); n = len(xy)
    s = np.bincount(E[:, 0], el, n) + np.bincount(E[:, 1], el, n)
    d = np.bincount(E[:, 0], minlength=n) + np.bincount(E[:, 1], minlength=n)
    m = s / np.maximum(d, 1)
    q = np.quantile(m, np.linspace(0, 1, N_STRATA + 1)[1:-1])
    return np.searchsorted(q, m)


def compute(im, n_types):
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(im["coords"][:50]).tobytes()) + 7)
    t = np.asarray(im["types"]).astype(int)
    Z = np.clip(im["markers"].astype(np.float64), -4, 4)
    st = density_strata(im)
    obs = np.concatenate([FN._type_part(im, t, n_types), FN._marker_part(im, Z)])
    sgroups = [np.flatnonzero(st == s) for s in range(N_STRATA)]
    mgroups = [np.flatnonzero((t == k) & (st == s)) for k in range(n_types) for s in range(N_STRATA)]
    nul = []
    for _ in range(FN.K_NULL):
        tp = t.copy()
        for g in sgroups:
            if len(g) > 1:
                tp[g] = t[rng.permutation(g)]
        Zp = Z.copy()
        for g in mgroups:
            if len(g) > 1:
                Zp[g] = Z[rng.permutation(g)]
        nul.append(np.concatenate([FN._type_part(im, tp, n_types), FN._marker_part(im, Zp)]))
    nul = np.stack(nul)
    mu, sd = nul.mean(0), nul.std(0)
    z = (obs - mu) / (sd + FN.EPS * (np.abs(mu) + 1e-3) + 1e-6)
    return np.clip(np.nan_to_num(z), -8, 8).astype(np.float32)
