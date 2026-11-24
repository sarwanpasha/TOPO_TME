"""TopoNull on perturbed cell maps (robustness to phenotyping and segmentation error). NAME znullp.
10% of the cells receive a type drawn at random from the image composition (excluding their own type), and every
centroid is moved by Gaussian noise with SD 0.25 median Delaunay edge lengths per axis; the Delaunay complex is rebuilt."""
import zlib
import numpy as np
import feat_null as FN
import topo_core as T

NAME = "znullp"
FRAC, JIT = 0.10, 0.25


def compute(im, n_types):
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(im["coords"][:50]).tobytes()) + 11)
    t = np.asarray(im["types"]).astype(int).copy()
    n = len(t)
    idx = rng.choice(n, int(round(FRAC * n)), replace=False)
    present = np.unique(t)
    if len(present) > 1:
        p = np.bincount(t, minlength=n_types).astype(float)
        for i in idx:
            q = p.copy(); q[t[i]] = 0
            t[i] = rng.choice(n_types, p=q / q.sum())
    xy = np.asarray(im["coords"], np.float64) + rng.normal(0, JIT * float(im["scale"]), size=(n, 2))
    E, TE, Fc = T.build_complex(xy / float(im["scale"]))
    jm = dict(im); jm.update(coords=xy, types=t, edges=E, tri_edges=TE, faces=Fc)
    return FN.compute(jm, n_types)
