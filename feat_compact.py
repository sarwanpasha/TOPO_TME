"""H3: COMPACT marker-field / cell-type persistence descriptor (few interpretable scalars).

Per marker m and sign s in {super (f=-z), sub (f=+z)}:
  f_v = s*z_m (clipped to [-4,4]),  f_e = max(f_u, f_v) + BETA * len / scale
  6 scalars from H0 and 6 from H1 (see diag_stats) + 3 level-set connectivity
  scalars at f=-1 (z>1 for super, z<-1 for sub)            -> 30 per marker.
Per cell type k: indicator field f_v = 0 (type k) / 1 (other), same edge rule,
  6 H0 + 6 H1 scalars (pairs born < 1, i.e. features of the type-k cells) + 3 connectivity scalars at level BETA*LVL_LEN
  (type-k cells joined by type-k edges shorter than LVL_LEN*scale) -> 15 per type.
All counts / sums are per 100 cells -> independent of image size.
python feat_compact.py <cohort>  -> feats_compact.npz (keys compact, image_id)
"""
import os, sys, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from topo_core import persistence_pairs

NAME = "compact"
BETA = 0.5
P_THR = 0.5        # "significant" persistence threshold (z units)
LEVEL = -1.0       # marker level-set (z > 1 / z < -1)
LVL_LEN = 1.5      # type connectivity: edges up to 1.5 x median Delaunay edge
N_DIAG = 6
N_CONN = 3
PER_MARKER = 2 * (2 * N_DIAG + N_CONN)   # 30
PER_TYPE = 2 * N_DIAG + N_CONN           # 15


def diag_stats(b, d, N):
    """6 robust scalars of a diagram: total pers/100 cells, #(p>P_THR)/100 cells,
    persistence-weighted mean birth, max pers, normalised persistent entropy,
    mean of top-5 persistences."""
    p = d - b
    k = p > 1e-9
    b, p = b[k], p[k]
    if len(p) == 0:
        return np.zeros(N_DIAG)
    S = p.sum()
    q = p / S
    ent = -(q * np.log(q)).sum() / np.log(len(p)) if len(p) > 1 else 0.0
    top = np.sort(p)[-5:].mean()
    return np.array([S / N * 100.0, (p > P_THR).sum() / N * 100.0, (p * b).sum() / S,
                     p.max(), ent, top])


def level_conn(fv, fe, E, t):
    """connectivity of sublevel set {f <= t}: [frac of ALL cells in largest comp,
    frac of in-set cells in largest comp, #components per 100 cells]."""
    N = len(fv)
    vin = fv <= t
    n_in = int(vin.sum())
    if n_in == 0:
        return np.zeros(N_CONN)
    ek = fe <= t
    A = coo_matrix((np.ones(int(ek.sum())), (E[ek, 0], E[ek, 1])), shape=(N, N))
    _, lab = connected_components(A, directed=False)
    cnt = np.bincount(lab[vin])
    cnt = cnt[cnt > 0]
    return np.array([cnt.max() / N, cnt.max() / n_in, len(cnt) / N * 100.0])


def field_feats(fv, fe, E, TE, Fc, t_level, bmax=np.inf):
    """bmax: keep only pairs born below bmax (indicator fields: features of the type-k
    population itself, not of the complementary cells)."""
    N = len(fv)
    h0b, h0d, h1b, h1d = persistence_pairs(fv, fe, E, TE, Fc)
    b0, d0, b1, d1 = fv[h0b], fe[h0d], fe[h1b], fe[h1d]
    k0, k1 = b0 < bmax, b1 < bmax
    return np.concatenate([diag_stats(b0[k0], d0[k0], N), diag_stats(b1[k1], d1[k1], N),
                           level_conn(fv, fe, E, t_level)])


def compute(im, n_types=None):
    E, TE, Fc = im["edges"], im["tri_edges"], im["faces"]
    xy = im["coords"].astype(np.float64) / im["scale"]
    el = BETA * np.linalg.norm(xy[E[:, 0]] - xy[E[:, 1]], axis=1)
    Z = np.clip(im["markers"].astype(np.float64), -4, 4)
    out = []
    for m in range(Z.shape[1]):
        for sign in (-1.0, 1.0):
            fv = sign * Z[:, m]
            fe = np.maximum(fv[E[:, 0]], fv[E[:, 1]]) + el
            out.append(field_feats(fv, fe, E, TE, Fc, LEVEL))
    if n_types is None:
        n_types = im.get("n_types")
    if n_types is not None:
        t = im["types"].astype(int)
        for k in range(int(n_types)):
            if (t == k).sum() < 2:
                out.append(np.zeros(PER_TYPE)); continue
            fv = (t != k).astype(np.float64)
            fe = np.maximum(fv[E[:, 0]], fv[E[:, 1]]) + el
            out.append(field_feats(fv, fe, E, TE, Fc, BETA * LVL_LEN, bmax=1.0))
    return np.nan_to_num(np.concatenate(out)).astype(np.float32)


def feature_names(markers, types):
    d = ["tp", "n_sig", "wbirth", "pmax", "ent", "top5"]
    cn = ["lcc_all", "lcc_set", "ncomp"]
    nm = []
    for m in markers:
        for s in ("sup", "sub"):
            nm += [f"{m}|{s}|H0|{x}" for x in d] + [f"{m}|{s}|H1|{x}" for x in d] + \
                  [f"{m}|{s}|lvl|{x}" for x in cn]
    for t in types:
        nm += [f"T:{t}|H0|{x}" for x in d] + [f"T:{t}|H1|{x}" for x in d] + [f"T:{t}|lvl|{x}" for x in cn]
    return nm


if __name__ == "__main__":
    import common as C
    c = sys.argv[1]
    imgs, pat, meta = C.load_cohort(c)
    nt = len(meta["types"])
    t0 = time.time(); F = []
    for i, im in enumerate(imgs):
        im["scale"] = meta["scale"]
        F.append(compute(im, nt))
        if i % 100 == 0:
            print(c, i, len(imgs), f"{time.time()-t0:.0f}s", flush=True)
    F = np.stack(F).astype(np.float32)
    np.savez_compressed(os.path.join(C.cohort_dir(c), "feats_compact.npz"), compact=F,
                        image_id=np.array([im["image_id"] for im in imgs]),
                        names=np.array(feature_names(meta["markers"], meta["types"])))
    print("DONE", c, F.shape, flush=True)
