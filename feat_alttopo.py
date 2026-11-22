"""Alternative topological descriptors used as comparators (GUDHI):
  alpha : per cell type, persistence (H0, H1) of the alpha complex of the cells of that type (the standard
          distance filtration; no fields, no calibration).                                     12 per type
  mp    : density-sliced two-parameter persistence. For each type, alpha persistence of the cells whose local
          density (inverse distance to the 5th nearest same-type cell) is above the 25/50/75% quantile, i.e.
          horizontal slices of the density-alpha bifiltration.                                  36 per type
  dowker: sparse Dowker (witness) persistence for ordered pairs (a landmarks, b witnesses) of the 6 most abundant
          cell types of the cohort ordering in the image: a simplex of <= 3 landmarks enters when some witness has
          all its vertices among its 3 nearest landmarks, at the largest of those distances.    12 per ordered pair
NAME alttopo = [alpha | mp | dowker]; block sizes: 12*K, 36*K, 12*30."""
import numpy as np
import gudhi
from scipy.spatial import cKDTree

NAME = "alttopo"
TOPK = 6
CAP = 8.0


def _stats(b, d, N):
    p = np.minimum(d, CAP) - b
    k = p > 1e-9; p = p[k]; b = b[k]
    if len(p) == 0:
        return np.zeros(6)
    S = p.sum(); q = p / S
    ent = -(q * np.log(q)).sum() / np.log(len(p)) if len(p) > 1 else 0.0
    return np.array([S / N * 100.0, (p > 0.3).sum() / N * 100.0, (p * b).sum() / S,
                     p.max(), ent, np.sort(p)[-5:].mean()])


def _pers(st, N):
    st.compute_persistence(homology_coeff_field=2, min_persistence=0)
    out = []
    for dim in (0, 1):
        iv = st.persistence_intervals_in_dimension(dim)
        if len(iv) == 0:
            out.append(np.zeros(6)); continue
        b = np.sqrt(np.maximum(iv[:, 0], 0)); d = iv[:, 1].copy(); d[~np.isfinite(d)] = CAP ** 2; d = np.sqrt(np.maximum(d, 0))
        out.append(_stats(b, d, N))
    return np.concatenate(out)


def _alpha(xy):
    if len(xy) < 4:
        return np.zeros(12)
    try:
        st = gudhi.AlphaComplex(points=xy.tolist()).create_simplex_tree()
        return _pers(st, len(xy))
    except Exception:
        return np.zeros(12)


def _dowker(L, W):
    if len(L) < 4 or len(W) < 4:
        return np.zeros(12)
    k = min(3, len(L))
    d, ix = cKDTree(L).query(W, k=k)
    best = {}
    for dd, ii in zip(d, ix):
        for r in range(k):
            for s in ([(ii[r],)] + [tuple(sorted((ii[a], ii[r]))) for a in range(r)] + ([tuple(sorted(ii[:3]))] if r == 2 else [])):
                v = dd[r]
                if v < best.get(s, np.inf):
                    best[s] = v
    st = gudhi.SimplexTree()
    for s, v in sorted(best.items(), key=lambda kv: (len(kv[0]), kv[1])):
        st.insert(list(map(int, s)), filtration=float(v) ** 2)
    st.make_filtration_non_decreasing()
    return _pers(st, len(L))


def compute(im, n_types):
    t = np.asarray(im["types"]).astype(int)
    xy = np.asarray(im["coords"], np.float64) / float(im["scale"])
    A, M = [], []
    for k in range(n_types):
        p = xy[t == k]
        A.append(_alpha(p))
        if len(p) >= 8:
            d5 = cKDTree(p).query(p, k=min(6, len(p)))[0][:, -1]
            for q in (0.25, 0.5, 0.75):
                M.append(_alpha(p[d5 <= np.quantile(d5, 1 - q)]))
        else:
            M += [np.zeros(12)] * 3
    cnt = np.bincount(t, minlength=n_types); top = TOP
    D = []
    for a in top:
        for b in top:
            if a != b:
                D.append(_dowker(xy[t == a], xy[t == b]) if cnt[a] >= 10 and cnt[b] >= 10 else np.zeros(12))
    while len(D) < TOPK * (TOPK - 1):
        D.append(np.zeros(12))
    return np.nan_to_num(np.concatenate(A + M + D)).astype(np.float32)


TOP = None


def _init(top):
    global TOP
    TOP = top


if __name__ == "__main__":
    import os, sys, time, multiprocessing as mp
    import common as C
    c = sys.argv[1]; w = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    imgs, pat, meta = C.load_cohort(c); nt = len(meta["types"])
    base = "basel" if c == "zurich" else c
    bi = imgs if base == c else C.load_cohort(base)[0]
    tot = np.zeros(nt)
    for im in bi:
        tot += np.bincount(np.asarray(im["types"]).astype(int), minlength=nt)
    top = np.sort(np.argsort(-tot, kind="stable")[:TOPK])
    args = [dict(coords=im["coords"], types=im["types"], scale=meta["scale"]) for im in imgs]
    t0 = time.time()
    with mp.get_context("spawn").Pool(w, initializer=_init, initargs=(top,)) as p:
        F = p.starmap(compute, [(a, nt) for a in args], chunksize=4)
    F = np.stack(F).astype(np.float32)
    np.savez_compressed(os.path.join(C.cohort_dir(c), "feats_alttopo.npz"), alttopo=F, top=top,
                        image_id=np.array([im["image_id"] for im in imgs]))
    print("DONE alttopo", c, F.shape, [meta["types"][i] for i in top], f"{time.time()-t0:.0f}s", flush=True)
