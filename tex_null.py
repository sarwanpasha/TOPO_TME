"""Empirical null distribution of the shipped TEx estimator (R = 8, regularized, clipped): apply TopoNull to images whose
cell-type labels (over positions) and marker vectors (within type) have been randomly permuted once, so that the data
satisfy the null exactly while composition and expression distributions match real tissue.
python tex_null.py <cohort> <n_images> [workers]"""
import sys, zlib, numpy as np, multiprocessing as mp
import common as C, feat_null as FN


def one(a):
    im, nt, seed = a
    rng = np.random.default_rng(seed)
    t = np.asarray(im["types"]).astype(int); Z = np.asarray(im["markers"]).copy()
    tp = rng.permutation(t)
    for k in range(nt):
        g = np.flatnonzero(tp == k)
        if len(g) > 1:
            Z[g] = Z[rng.permutation(g)]
    jm = dict(im); jm["types"] = tp; jm["markers"] = Z
    return FN.compute(jm, nt)


if __name__ == "__main__":
    c, n = sys.argv[1], int(sys.argv[2]); W = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    imgs, pat, meta = C.load_cohort(c); nt = len(meta["types"])
    rng = np.random.default_rng(1)
    pick = rng.choice(len(imgs), min(n, len(imgs)), replace=False)
    args = []
    for i in pick:
        im = imgs[i]; im["scale"] = meta["scale"]
        args.append(({k: im[k] for k in ("coords", "types", "markers", "edges", "tri_edges", "faces", "scale")}, nt, int(i)))
    with mp.get_context("spawn").Pool(W) as p:
        T = np.stack(p.map(one, args))
    v = T[:, T.std(0) > 1e-9]
    print(f"{c}: images {len(pick)}, descriptors {v.shape[1]}")
    print(f"null TEx: mean {v.mean():+.3f}, SD {v.std():.3f}, median |TEx| {np.median(np.abs(v)):.3f}, "
          f"fraction |TEx|>2 {np.mean(np.abs(v) > 2):.3f} (Gaussian 0.046), clipped {np.mean(np.abs(v) >= 7.99):.4f}")
    md = v.mean(0); sd = v.std(0)
    print(f"per-descriptor mean: median {np.median(md):+.3f} (IQR {np.percentile(md,25):+.3f} to {np.percentile(md,75):+.3f}); "
          f"per-descriptor SD: median {np.median(sd):.3f} (IQR {np.percentile(sd,25):.3f} to {np.percentile(sd,75):.3f})")
    # dependence on composition: correlation of null TEx with the abundance of the relevant type, pooled
    comp = np.stack([np.bincount(np.asarray(imgs[i]["types"]).astype(int), minlength=nt) / len(imgs[i]["types"]) for i in pick])
    r = [abs(np.corrcoef(v[:, j], comp[:, k])[0, 1]) for j in range(0, v.shape[1], max(1, v.shape[1] // 200)) for k in range(nt) if comp[:, k].std() > 0]
    print(f"|corr(null TEx, cell-type fraction)|: median {np.nanmedian(r):.3f}, 95th percentile {np.nanpercentile(r, 95):.3f}")
