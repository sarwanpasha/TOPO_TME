"""Nested-null decomposition of topological descriptors (TopoNull ladder).

A chain of permutation groups G_0 > G_1 > ... > G_L acts on the marks of a cell map with fixed positions. For a
descriptor phi let mu_j = E[phi | orbit under G_j] (estimated by K permutations drawn from G_j). The telescoping sum
    phi = mu_0 + (mu_1 - mu_0) + ... + (mu_L - mu_{L-1}) + (phi - mu_L)
splits phi into the part fixed by abundance, the parts explained by successively finer spatial constraints, and a
residual that none of the constraints explains. Under a G_0-invariant law the terms are martingale differences and
therefore mutually uncorrelated (an ANOVA identity for topology).

Type-level descriptors (cell-type fields, cross-type persistence); strata depend on positions only:
    G_0 all label permutations                       -> A  abundance
    G_1 within local-density quintiles               -> D  density-linked
    G_2 within density quintile x spatial region     -> R  regional layout (3 x 3 grid of the bounding box)
    residual                                         -> M  micro-organisation
Marker-level descriptors (marker fields): marker vectors are permuted
    H_0 over all cells                               -> A  expression distribution only
    H_1 within cell type                             -> T  explained by the cell-type layout
    H_2 within type x density quintile               -> D
    H_3 within type x density x region               -> R
    residual                                         -> M
Output per image: [obs | mu_0..mu_L | sd_L] for the type block followed by the same for the marker block.
python feat_ladder.py <cohort> [workers]  -> feats_ladder.npz {ladder, image_id, n_type_desc, n_marker_desc}"""
import os, sys, time, zlib
import numpy as np
import feat_null as FN
from feat_nulld import density_strata

NAME = "ladder"
K = 8
GRID = 3


def regions(im):
    xy = im["coords"].astype(np.float64)
    lo, hi = xy.min(0), xy.max(0) + 1e-9
    g = np.minimum((GRID * (xy - lo) / (hi - lo)).astype(int), GRID - 1)
    return g[:, 0] * GRID + g[:, 1]


def _perm_within(rng, strata):
    """index permutation that only moves cells inside their stratum."""
    p = np.arange(len(strata)); o = np.argsort(strata, kind="stable"); s = strata[o]
    b = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    for a, c in zip(b[:-1], b[1:]):
        if c - a > 1:
            p[o[a:c]] = o[a:c][rng.permutation(c - a)]
    return p


def compute(im, n_types):
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(im["coords"][:50]).tobytes()) + 11)
    t = np.asarray(im["types"]).astype(int)
    Z = np.clip(im["markers"].astype(np.float64), -4, 4)
    n = len(t); dens = density_strata(im); reg = regions(im)
    zero = np.zeros(n, int)
    tl = [zero, dens, dens * 100 + reg]                                  # type ladder strata
    ml = [zero, t, t * 10 + dens, (t * 10 + dens) * 100 + reg]           # marker ladder strata
    obsT = FN._type_part(im, t, n_types); obsM = FN._marker_part(im, Z)
    outT = [obsT]; sdT = None
    for s in tl:
        R = np.stack([FN._type_part(im, t[_perm_within(rng, s)], n_types) for _ in range(K)])
        outT.append(R.mean(0)); sdT = R.std(0)
    outT.append(sdT)
    outM = [obsM]; sdM = None
    for s in ml:
        R = np.stack([FN._marker_part(im, Z[_perm_within(rng, s)]) for _ in range(K)])
        outM.append(R.mean(0)); sdM = R.std(0)
    outM.append(sdM)
    return np.nan_to_num(np.concatenate(outT + outM)).astype(np.float32)


if __name__ == "__main__":
    import multiprocessing as mp
    import common as C
    c = sys.argv[1]; w = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    imgs, pat, meta = C.load_cohort(c); nt = len(meta["types"])
    keep = ("coords", "types", "markers", "edges", "tri_edges", "faces")
    args = [({**{k: im[k] for k in keep}, "scale": meta["scale"]}, nt) for im in imgs]
    t0 = time.time()
    with mp.get_context("spawn").Pool(w) as p:
        F = p.starmap(compute, args, chunksize=2)
    F = np.stack(F)
    nT = nt * 15 + nt * (nt + 1) // 2 * 6; nM = len(meta["markers"]) * 30
    assert F.shape[1] == nT * 5 + nM * 6, (F.shape, nT, nM)
    np.savez_compressed(os.path.join(C.cohort_dir(c), "feats_ladder.npz"), ladder=F, n_type_desc=nT, n_marker_desc=nM,
                        image_id=np.array([im["image_id"] for im in imgs]))
    print("DONE ladder", c, F.shape, f"{time.time()-t0:.0f}s", flush=True)
