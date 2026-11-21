"""H4: cross-type interaction persistence (multiscale generalisation of contact counts).

For every unordered type pair (A,B), A<=B (fixed cohort ordering = type index), build the
Delaunay complex of the cells of type A or B (sub-Delaunay: cells of other types are removed,
so A/B cells separated by other cells still connect at larger scales).
Filtration: f_v = 0 for all A/B cells; edge value = length/scale for A-B edges,
length/scale + PEN for A-A / B-B edges (mixed adjacency enters first; homotypic adjacency
only after the penalty). For A==B: plain single-type Delaunay filtration (no penalty).
fe >= 0 = max(fv) always holds.

Per pair (K=6 scalars, zeros if a type has < MIN_N cells):
  0 H0 total persistence (deaths capped at PEN) / n_AB
  1 #components alive at scale 1 / n_AB
  2 #components alive at scale 2 / n_AB
  3 #H1 loops with persistence > 0.3 / n_AB
  4 H1 total persistence (capped at PEN) / n_AB
  5 max H1 persistence (capped at PEN)
python feat_cross.py <cohort>  -> cohort_dir/feats_cross.npz {cross, image_id}
"""
import os, sys, time
import numpy as np
import topo_core as T

NAME = "cross"
K = 6
PEN = 3.0
MIN_N = 10
H1_THR = 0.3


def _pair_feats(xy, is_a, same_pen):
    n = len(xy)
    if n < 4:
        return np.zeros(K, np.float32)
    try:
        edges, tri_edges, faces = T.build_complex(xy)
    except Exception:
        return np.zeros(K, np.float32)
    fv = np.zeros(n)
    fe = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)
    if same_pen:
        fe = fe + PEN * (is_a[edges[:, 0]] == is_a[edges[:, 1]])
    h0b, h0d, h1b, h1d = T.persistence_pairs(fv, fe, edges, tri_edges, faces)
    d0 = fe[h0d]                       # births are all 0
    p1 = np.minimum(fe[h1d] - fe[h1b], PEN)
    p1 = p1[p1 > 1e-9]
    out = np.zeros(K)
    out[0] = np.minimum(d0, PEN).sum() / n
    out[1] = (n - (d0 <= 1.0).sum()) / n
    out[2] = (n - (d0 <= 2.0).sum()) / n
    out[3] = (p1 > H1_THR).sum() / n
    out[4] = p1.sum() / n
    out[5] = p1.max() if len(p1) else 0.0
    return out.astype(np.float32)


def compute(im, n_types):
    t = np.asarray(im["types"]).astype(int)
    xy = np.asarray(im["coords"], np.float64) / float(im["scale"])
    cnt = np.bincount(t, minlength=n_types)
    feats = []
    for a in range(n_types):
        for b in range(a, n_types):
            if cnt[a] < MIN_N or cnt[b] < MIN_N:
                feats.append(np.zeros(K, np.float32)); continue
            m = (t == a) | (t == b)
            feats.append(_pair_feats(xy[m], t[m] == a, a != b))
    return np.concatenate(feats).astype(np.float32)


if __name__ == "__main__":
    import common as C
    c = sys.argv[1]
    images, pat, meta = C.load_cohort(c)
    n_types = len(meta["types"])
    t0 = time.time(); X = []
    for i, im in enumerate(images):
        im["scale"] = meta["scale"]
        X.append(compute(im, n_types))
        if i % 50 == 0:
            print(c, i, len(images), f"{time.time() - t0:.1f}s", flush=True)
    np.savez(os.path.join(C.cohort_dir(c), "feats_cross.npz"), cross=np.stack(X),
             image_id=np.array([im["image_id"] for im in images]))
    print(c, "done", np.stack(X).shape, f"{(time.time() - t0) / len(images):.2f}s/img", flush=True)
