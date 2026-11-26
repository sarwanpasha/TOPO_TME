"""Multiscale co-localization baseline in the style of CRAWDAD (dos Santos Peixoto et al., Nat Commun 2025), no homology.
For every unordered pair of cell types (a, b), including a = b, the statistic is the number of a-b cell pairs closer than
R median Delaunay edges. It is standardized against null distributions in which the type labels are shuffled
  (i) over the whole image and (ii)-(iv) only within square tiles whose side is 1/2, 1/4 and 1/8 of the image extent,
so that each scale asks whether the pair co-localizes beyond what the composition at that scale implies.
NAME crawdad: 4 scales x n_pairs calibrated values (same standardization and clipping as TopoNull)."""
import zlib
import numpy as np
from scipy.spatial import cKDTree
from feat_ladder import _perm_within

NAME = "crawdad"
K_NULL, EPS, R = 8, 0.05, 3.0
SCALES = (None, 2, 4, 8)


def _counts(t, i, j, n_types):
    C = np.zeros((n_types, n_types))
    np.add.at(C, (t[i], t[j]), 1.0)
    C = C + C.T - np.diag(np.diag(C))
    return C[np.triu_indices(n_types)]


def compute(im, n_types):
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(im["coords"][:50]).tobytes()) + 31)
    xy = im["coords"].astype(np.float64) / im["scale"]
    t = np.asarray(im["types"]).astype(int)
    pr = cKDTree(xy).query_pairs(R, output_type="ndarray")
    i, j = (pr[:, 0], pr[:, 1]) if len(pr) else (np.zeros(0, int), np.zeros(0, int))
    obs = _counts(t, i, j, n_types)
    lo = xy.min(0); ext = (xy.max(0) - lo).max() + 1e-9
    out = []
    for s in SCALES:
        if s is None:
            strata = np.zeros(len(t), int)
        else:
            g = np.minimum((s * (xy - lo) / ext).astype(int), s - 1)
            strata = g[:, 0] * s + g[:, 1]
        nul = np.stack([_counts(t[_perm_within(rng, strata)], i, j, n_types) for _ in range(K_NULL)])
        mu, sd = nul.mean(0), nul.std(0)
        out.append(np.clip(np.nan_to_num((obs - mu) / (sd + EPS * (np.abs(mu) + 1e-3) + 1e-6)), -8, 8))
    return np.concatenate(out).astype(np.float32)
