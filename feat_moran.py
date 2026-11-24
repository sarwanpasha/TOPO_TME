"""Spatial-autocorrelation baseline (no homology). On the Delaunay graph of the image, at 2 scales (all Delaunay edges;
edges no longer than the median edge length):
  per marker : Moran's I and Geary's C over all edges, and Moran's I over homotypic edges only (both cells of the same
               type), i.e. autocorrelation of expression within cell types;                         3 x 2 = 6 per marker
  per type   : Moran's I of the type indicator (join-count type statistic).                          1 x 2 = 2 per type
The same permutation null as TopoNull (type labels permuted over positions; marker vectors permuted within type) gives a
calibrated version, so the comparison with TopoNull separates homology from calibrated autocorrelation.
NAME moran: [raw | calibrated], each 2 per type + 6 per marker."""
import zlib
import numpy as np

NAME = "moran"
K_NULL, EPS = 8, 0.05


def _moran(x, a, b):
    if len(a) == 0:
        return 0.0
    z = x - x.mean(); den = (z ** 2).sum()
    if den <= 1e-12:
        return 0.0
    return float(len(x) / (2.0 * len(a)) * (2.0 * (z[a] * z[b]).sum()) / den)


def _geary(x, a, b):
    if len(a) == 0:
        return 1.0
    z = x - x.mean(); den = (z ** 2).sum()
    if den <= 1e-12:
        return 1.0
    return float((len(x) - 1) * ((x[a] - x[b]) ** 2).sum() / (2.0 * len(a) * den))


def _all(t, Z, E, keep_sets, n_types):
    out = []
    for ek in keep_sets:
        a, b = E[ek, 0], E[ek, 1]
        for k in range(n_types):
            out.append(_moran((t == k).astype(np.float64), a, b))
        same = t[a] == t[b]
        for m in range(Z.shape[1]):
            x = Z[:, m]
            out += [_moran(x, a, b), _geary(x, a, b), _moran(x, a[same], b[same])]
    return np.array(out)


def compute(im, n_types):
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(im["coords"][:50]).tobytes()) + 7)
    E = np.asarray(im["edges"]); xy = im["coords"].astype(np.float64) / im["scale"]
    el = np.linalg.norm(xy[E[:, 0]] - xy[E[:, 1]], axis=1)
    keep_sets = (np.ones(len(E), bool), el <= np.median(el))
    t = np.asarray(im["types"]).astype(int); Z = np.clip(im["markers"].astype(np.float64), -4, 4)
    obs = _all(t, Z, E, keep_sets, n_types)
    groups = [np.flatnonzero(t == k) for k in range(n_types)]
    nul = []
    for _ in range(K_NULL):
        tp = rng.permutation(t); Zp = Z.copy()
        for g in groups:
            if len(g) > 1:
                Zp[g] = Z[rng.permutation(g)]
        nul.append(_all(tp, Zp, E, keep_sets, n_types))
    nul = np.stack(nul); mu, sd = nul.mean(0), nul.std(0)
    z = np.clip(np.nan_to_num((obs - mu) / (sd + EPS * (np.abs(mu) + 1e-3) + 1e-6)), -8, 8)
    return np.concatenate([np.nan_to_num(obs), z]).astype(np.float32)
