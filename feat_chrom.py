"""Chromatic alpha comparator (chalc, Delaunay-Cech filtration, which has the persistent homology of the chromatic
alpha filtration). Per image: cells of the 4 most abundant cell types of the cohort, coloured by type (at most CAPN cells,
uniform subsample with a fixed seed). For each of the 4 types a, the 6-pack of the inclusion of the subcomplex spanned by
type a into the full chromatic complex; we keep the kernel, cokernel and image diagrams (H0, H1), plus the codomain once.
Each diagram is summarized by the same 6 statistics as the other comparators.   NAME chrom; 4*36 + 12 = 156 features."""
import sys, types
_w = types.ModuleType("phimaker._washboard_interface")  # the optional web viewer of phimaker is not needed
_w.WashboardServer = object
_w.build_washboard_object = lambda *a, **k: None
sys.modules["phimaker._washboard_interface"] = _w
import warnings
import numpy as np
import chalc.sixpack as six
from feat_alttopo import _stats

NAME = "chrom"
TOPK = 4
CAPN = 1500
TOP = None


def _summ(m, N):
    m = np.asarray(m, np.float64).reshape(-1, 2)
    if len(m) == 0:
        return np.zeros(6)
    b = m[:, 0]; d = m[:, 1].copy(); d[~np.isfinite(d)] = 8.0
    return _stats(b, d, N)


def compute(im, seed):
    t = np.asarray(im["types"]).astype(int)
    xy = np.asarray(im["coords"], np.float64) / float(im["scale"])
    keep = np.isin(t, TOP)
    xy, t = xy[keep], t[keep]
    if len(t) > CAPN:
        ix = np.random.default_rng(seed).choice(len(t), CAPN, replace=False); xy, t = xy[ix], t[ix]
    col = np.searchsorted(TOP, t)
    out = []
    N = max(len(t), 1)
    cod = None
    for a in range(TOPK):
        if (col == a).sum() < 5 or len(np.unique(col)) < 2 or len(t) < 10:
            out.append(np.zeros(36)); continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = six.compute(xy.T.copy(), col.tolist(), dom=[a], method="delcech", max_diagram_dimension=1)
            for name in ("ker", "cok", "im"):
                for dim in (0, 1):
                    out.append(_summ(d.get_matrix(name, dim), N))
            if cod is None:
                cod = np.concatenate([_summ(d.get_matrix("cod", dim), N) for dim in (0, 1)])
        except Exception:
            out.append(np.zeros(36))
    out = [np.asarray(o).ravel() for o in out]
    out.append(cod if cod is not None else np.zeros(12))
    return np.nan_to_num(np.concatenate(out)).astype(np.float32)


def _init(top):
    global TOP
    TOP = top


if __name__ == "__main__":
    import os, time, multiprocessing as mp
    import common as C
    c = sys.argv[1]; w = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    out = os.path.join(C.cohort_dir(c), "feats_chrom.npz")
    if os.path.exists(out):
        print("SKIP", c, flush=True); sys.exit()
    imgs, pat, meta = C.load_cohort(c); nt = len(meta["types"])
    base = "basel" if c == "zurich" else c
    bi = imgs if base == c else C.load_cohort(base)[0]
    tot = np.zeros(nt)
    for im in bi:
        tot += np.bincount(np.asarray(im["types"]).astype(int), minlength=nt)
    top = np.sort(np.argsort(-tot, kind="stable")[:TOPK])
    args = [(dict(coords=im["coords"], types=im["types"], scale=meta["scale"]), i) for i, im in enumerate(imgs)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(w, initializer=_init, initargs=(top,)) as p:
        F = []
        for i, f in enumerate(p.starmap(compute, args, chunksize=2)):
            F.append(f)
    F = np.stack(F).astype(np.float32)
    np.savez_compressed(out, chrom=F, top=top, image_id=np.array([im["image_id"] for im in imgs]))
    print("DONE chrom", c, F.shape, [meta["types"][i] for i in top], f"{time.time()-t0:.0f}s", flush=True)
