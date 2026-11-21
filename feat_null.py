"""TopoNull: composition-calibrated topological excess (TEx) features.

For every image we compare the observed persistence descriptors with a permutation null
that keeps cell positions and composition fixed but destroys spatial arrangement:
  * type-level descriptors (per-type indicator-field persistence, 15/type, and cross-type
    interaction persistence, 6/pair): null = cell-type labels permuted over the fixed cell positions;
  * marker-level descriptors (per-marker field persistence, 30/marker): null = marker vectors
    permuted WITHIN each cell type (keeps the cell-type layout and each type's expression
    distribution; tests spatial organisation of expression beyond cell-type layout).
TEx = (observed - mean_null) / (sd_null + eps), clipped to [-8, 8].
By construction E[TEx | composition, positions] = 0 under spatial randomness, so TEx carries no
composition signal to first order -- it is exactly the spatial-organisation information that
composition- and abundance-based analyses cannot see.
"""
import zlib
import numpy as np
import feat_compact as FC
import feat_cross as FX

NAME = "znull"
K_NULL = 8
EPS = 0.05


def _type_part(im, t, n_types):
    E, TE, Fc = im["edges"], im["tri_edges"], im["faces"]
    xy = im["coords"].astype(np.float64) / im["scale"]
    el = FC.BETA * np.linalg.norm(xy[E[:, 0]] - xy[E[:, 1]], axis=1)
    out = []
    for k in range(n_types):
        if (t == k).sum() < 2:
            out.append(np.zeros(FC.PER_TYPE)); continue
        fv = (t != k).astype(np.float64)
        fe = np.maximum(fv[E[:, 0]], fv[E[:, 1]]) + el
        out.append(FC.field_feats(fv, fe, E, TE, Fc, FC.BETA * FC.LVL_LEN, bmax=1.0))
    cr = FX.compute(dict(types=t, coords=im["coords"], scale=im["scale"]), n_types)
    return np.concatenate(out + [cr])


def _marker_part(im, Z):
    E, TE, Fc = im["edges"], im["tri_edges"], im["faces"]
    xy = im["coords"].astype(np.float64) / im["scale"]
    el = FC.BETA * np.linalg.norm(xy[E[:, 0]] - xy[E[:, 1]], axis=1)
    out = []
    for m in range(Z.shape[1]):
        for sign in (-1.0, 1.0):
            fv = sign * Z[:, m]
            fe = np.maximum(fv[E[:, 0]], fv[E[:, 1]]) + el
            out.append(FC.field_feats(fv, fe, E, TE, Fc, FC.LEVEL))
    return np.concatenate(out)


def compute(im, n_types):
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(im["coords"][:50]).tobytes()))
    t = np.asarray(im["types"]).astype(int)
    Z = np.clip(im["markers"].astype(np.float64), -4, 4)
    obs = np.concatenate([_type_part(im, t, n_types), _marker_part(im, Z)])
    nul = []
    groups = [np.flatnonzero(t == k) for k in range(n_types)]
    for _ in range(K_NULL):
        tp = rng.permutation(t)
        Zp = Z.copy()
        for g in groups:
            if len(g) > 1:
                Zp[g] = Z[rng.permutation(g)]
        nul.append(np.concatenate([_type_part(im, tp, n_types), _marker_part(im, Zp)]))
    nul = np.stack(nul)
    mu, sd = nul.mean(0), nul.std(0)
    z = (obs - mu) / (sd + EPS * (np.abs(mu) + 1e-3) + 1e-6)
    return np.clip(np.nan_to_num(z), -8, 8).astype(np.float32)


def feature_names(markers, types):
    d = ["tp", "n_sig", "wbirth", "pmax", "ent", "top5"]; cn = ["lcc_all", "lcc_set", "ncomp"]
    nm = []
    for ty in types:
        nm += [f"zT:{ty}|H0|{x}" for x in d] + [f"zT:{ty}|H1|{x}" for x in d] + [f"zT:{ty}|lvl|{x}" for x in cn]
    xs = ["H0tp", "ncomp@1", "ncomp@2", "nloops", "H1tp", "H1max"]
    for a in range(len(types)):
        for b in range(a, len(types)):
            nm += [f"zX:{types[a]}~{types[b]}|{x}" for x in xs]
    for m in markers:
        for s in ("sup", "sub"):
            nm += [f"zM:{m}|{s}|H0|{x}" for x in d] + [f"zM:{m}|{s}|H1|{x}" for x in d] + [f"zM:{m}|{s}|lvl|{x}" for x in cn]
    return nm
