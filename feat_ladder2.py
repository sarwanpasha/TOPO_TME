"""Second, independent run of the nested-null decomposition, for 2 purposes:
  (1) Monte Carlo replicate with a new seed: agreement with feats_ladder.npz gives the Monte Carlo error of every
      component (each level is a mean of K permutations);
  (2) the other order of the 2 middle constraints (region before density), so that the attribution can be averaged
      over both orders (Shapley value over the 2 orderings of density and region).
Type ladder, density-first:  0 | dens | dens x reg          region-first: 0 | reg | reg x dens (same last level)
Marker ladder, density-first: 0 | t | t x dens | t x dens x reg   region-first: 0 | t | t x reg | (same last level)
Output per image: type block [obs, mu0, muD, muR_alt, muDR, sd0, sdD, sdR_alt, sdDR] and marker block
[obs, mu0, muT, muTD, muTR_alt, muTDR, sd0, sdT, sdTD, sdTR_alt, sdTDR].
python feat_ladder2.py <cohort> [workers]  -> feats_ladder2.npz"""
import os, sys, time, zlib
import numpy as np
import feat_null as FN
from feat_nulld import density_strata
from feat_ladder import regions, _perm_within

NAME = "ladder2"
K = 8


def _lvl(rng, fn, strata):
    R = np.stack([fn(strata) for _ in range(K)])
    return R.mean(0), R.std(0)


def compute(im, n_types):
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(im["coords"][:50]).tobytes()) + 977)
    t = np.asarray(im["types"]).astype(int)
    Z = np.clip(im["markers"].astype(np.float64), -4, 4)
    n = len(t); dens = density_strata(im); reg = regions(im); zero = np.zeros(n, int)
    ft = lambda s: FN._type_part(im, t[_perm_within(rng, s)], n_types)
    fm = lambda s: FN._marker_part(im, Z[_perm_within(rng, s)])
    obsT = FN._type_part(im, t, n_types); obsM = FN._marker_part(im, Z)
    tl = [zero, dens, reg, dens * 100 + reg]
    ml = [zero, t, t * 10 + dens, t * 100 + reg, (t * 10 + dens) * 100 + reg]
    LT = [_lvl(rng, ft, s) for s in tl]; LM = [_lvl(rng, fm, s) for s in ml]
    outT = [obsT] + [m for m, _ in LT] + [s for _, s in LT]
    outM = [obsM] + [m for m, _ in LM] + [s for _, s in LM]
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
    assert F.shape[1] == nT * 9 + nM * 11, (F.shape, nT, nM)
    np.savez_compressed(os.path.join(C.cohort_dir(c), "feats_ladder2.npz"), ladder2=F, n_type_desc=nT, n_marker_desc=nM,
                        image_id=np.array([im["image_id"] for im in imgs]))
    print("DONE ladder2", c, F.shape, f"{time.time()-t0:.0f}s", flush=True)
