"""Marker-field persistence bank: for every marker m, superlevel persistence of the
expression field on the Delaunay complex (high-expressing cells enter first),
f_v = -z_m(v), f_e = max(f_u, f_v) + beta * |x_u - x_v| / scale.
H0 + H1 persistence images per marker -> feats_mfield.npz (per image).
python bank.py <cohort>
"""
import os, sys, time
import numpy as np
import common as C
from topo_core import persistence_pairs

BETA = 0.5
GB = np.linspace(-3, 3, 5); GP = np.linspace(0, 4, 5); SIG = 0.75


def pimg(b, d):
    if len(b) == 0:
        return np.zeros(len(GB) * len(GP) + 3)
    p = d - b
    wb = np.exp(-(b[:, None] - GB[None]) ** 2 / (2 * SIG ** 2))
    wp = np.exp(-(p[:, None] - GP[None]) ** 2 / (2 * SIG ** 2))
    img = (p[:, None, None] * wb[:, :, None] * wp[:, None, :]).sum(0).ravel()
    return np.concatenate([img, [p.sum(), (p > 0.5).sum(), (p * b).sum()]])


GB2 = np.linspace(-2.5, 2.5, 3); GP2 = np.linspace(0, 3, 3); SIG2 = 1.0


def pimg2(b, d):
    if len(b) == 0:
        return np.zeros(len(GB2) * len(GP2) + 3)
    p = d - b
    wb = np.exp(-(b[:, None] - GB2[None]) ** 2 / (2 * SIG2 ** 2))
    wp = np.exp(-(p[:, None] - GP2[None]) ** 2 / (2 * SIG2 ** 2))
    img = (p[:, None, None] * wb[:, :, None] * wp[:, None, :]).sum(0).ravel()
    return np.concatenate([img, [p.sum(), (p > 0.5).sum(), (p * b).sum()]])


def image_bank2(im):
    """multi-scale, two-sided marker-field persistence (super- and sub-level, beta in {0.25, 1.5})."""
    E, TE, Fc = im["edges"], im["tri_edges"], im["faces"]
    xy = im["coords"].astype(np.float64) / im["scale"]
    el = np.linalg.norm(xy[E[:, 0]] - xy[E[:, 1]], axis=1)
    Z = np.clip(im["markers"].astype(np.float64), -4, 4)
    N = len(Z); out = []
    for m in range(Z.shape[1]):
        for sign in (-1.0, 1.0):
            fv = sign * Z[:, m]
            for beta in (0.25, 1.5):
                fe = np.maximum(fv[E[:, 0]], fv[E[:, 1]]) + beta * el
                h0b, h0d, h1b, h1d = persistence_pairs(fv, fe, E, TE, Fc)
                out.append(pimg2(fv[h0b], fe[h0d])); out.append(pimg2(fe[h1b], fe[h1d]))
    return np.concatenate(out) / N * 100.0


def image_bank(im):
    E, TE, Fc = im["edges"], im["tri_edges"], im["faces"]
    xy = im["coords"].astype(np.float64) / im["scale"]
    el = np.linalg.norm(xy[E[:, 0]] - xy[E[:, 1]], axis=1)
    Z = np.clip(im["markers"].astype(np.float64), -4, 4)
    N = len(Z)
    out = []
    for m in range(Z.shape[1]):
        fv = -Z[:, m]
        fe = np.maximum(fv[E[:, 0]], fv[E[:, 1]]) + BETA * el
        h0b, h0d, h1b, h1d = persistence_pairs(fv, fe, E, TE, Fc)
        out.append(pimg(fv[h0b], fe[h0d]))
        out.append(pimg(fe[h1b], fe[h1d]))
    return np.concatenate(out) / N * 100.0


if __name__ == "__main__":
    c = sys.argv[1]; v2 = len(sys.argv) > 2 and sys.argv[2] == "v2"
    imgs, pat, meta = C.load_cohort(c)
    t0 = time.time()
    F = []
    for i, im in enumerate(imgs):
        im["scale"] = meta["scale"]
        F.append(image_bank2(im) if v2 else image_bank(im))
        if i % 100 == 0:
            print(c, i, len(imgs), f"{time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(os.path.join(C.cohort_dir(c), "feats_mfield2.npz" if v2 else "feats_mfield.npz"),
                        **{("mfield2" if v2 else "mfield"): np.stack(F).astype(np.float32)},
                        image_id=np.array([im["image_id"] for im in imgs]))
    print("DONE", c, np.stack(F).shape, flush=True)
