"""Robustness of receptor- and vessel-field associations to (1) lateral signal spillover and (2) the conditioning set.
(1) Spillover between touching cells adds a fraction s of the neighbours' signal to each cell. We undo a spillover of
    s in {0, 0.1, 0.2, 0.3} on the standardized intensities, x' = (x - s * mean of Delaunay neighbours) / (1 - s),
    recompute the field descriptors of the marker under the standard TopoNull null (marker vectors permuted within cell
    type) and repeat the likelihood-ratio test conditional on composition, mean markers and log cell count.
(2) For s = 0, the test is repeated conditioning on 5, 15 (default) or 30 principal components, or on the full
    standardized vector of composition, mean markers and log cell count without compression.
python spill.py <cohort> <task> <marker name> <descriptor, e.g. sup|H0|tp> [workers]"""
import os, sys, zlib, time
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import common as C
from interp import patient_matrix, loglik
from rephen import field, STATS

SPILL = (0.0, 0.1, 0.2, 0.3)


def one(args):
    im, mi, scale = args
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(im["coords"][:50]).tobytes()) + 5)
    x = np.clip(im["markers"][:, mi].astype(np.float64), -4, 4)
    E = np.asarray(im["edges"]); n = len(x)
    deg = np.bincount(E[:, 0], minlength=n) + np.bincount(E[:, 1], minlength=n)
    nb = (np.bincount(E[:, 0], weights=x[E[:, 1]], minlength=n) + np.bincount(E[:, 1], weights=x[E[:, 0]], minlength=n)) / np.maximum(deg, 1)
    t = np.asarray(im["types"]).astype(int)
    groups = [np.flatnonzero(t == k) for k in np.unique(t)]
    out = []
    for s in SPILL:
        z = np.clip((x - s * nb) / (1 - s), -4, 4)
        obs = field(im, z, scale); nul = []
        for _ in range(8):
            zp = z.copy()
            for g in groups:
                if len(g) > 1:
                    zp[g] = z[rng.permutation(g)]
            nul.append(field(im, zp, scale))
        nul = np.stack(nul); mu, sd = nul.mean(0), nul.std(0)
        out.append(np.clip(np.nan_to_num((obs - mu) / (sd + 0.05 * (np.abs(mu) + 1e-3) + 1e-6)), -8, 8))
    return np.concatenate(out)


if __name__ == "__main__":
    import multiprocessing as mp
    cohort, task, mname, desc = sys.argv[1:5]; W = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    imgs, pat, meta = C.load_cohort(cohort); mi = meta["markers"].index(mname)
    t0 = time.time()
    with mp.get_context("spawn").Pool(W) as p:
        F = np.stack(p.map(one, [(dict(coords=im["coords"], markers=im["markers"], edges=im["edges"], tri_edges=im["tri_edges"],
                                       faces=im["faces"], types=im["types"]), mi, meta["scale"]) for im in imgs], chunksize=4))
    names = [f"{s}|{h}|{x}" for s in ("sup", "sub") for h in ("H0", "H1") for x in STATS]
    j = names.index(desc); nd = len(names)
    pid = np.array([str(im["patient_id"]) for im in imgs])
    cols = {f"s{s}": F[:, k * nd + j] for k, s in enumerate(SPILL)}
    T = pd.DataFrame(cols); T["pid"] = pid; T = T.groupby("pid").mean()
    df, kind = C.task_frame(pat, cohort, task)
    base = pd.concat([patient_matrix(cohort, k, None).add_prefix(k + "_") for k in ("comp", "mm", "logn")], axis=1)
    ids = [q for q in df["patient_id"] if q in base.index and q in T.index]
    y = df.set_index("patient_id").loc[ids, "y"].values.astype(int)
    Bs = StandardScaler().fit_transform(np.nan_to_num(base.loc[ids].values))
    Bs = Bs[:, Bs.std(0) > 0]

    def test(B, x):
        x = (x - x.mean()) / x.std()
        lr = 2 * (loglik(np.hstack([B, x[:, None]]), y) - loglik(B, y))
        return chi2.sf(max(lr, 0), 1)
    print(cohort, task, mname, desc, "n", len(ids), f"{time.time()-t0:.0f}s", flush=True)
    B15 = PCA(15, random_state=0).fit_transform(Bs)
    for s in SPILL:
        print(f"spillover {s:.1f}: p = {test(B15, T.loc[ids, f's{s}'].values):.2e}", flush=True)
    x0 = T.loc[ids, "s0.0"].values
    for k in (5, 15, 30):
        print(f"conditioning on {k} PCs: p = {test(PCA(k, random_state=0).fit_transform(Bs), x0):.2e}", flush=True)
    from sklearn.linear_model import LogisticRegression

    def ll_pen(B):  # lightly ridge-penalized fit: the unpenalized fit separates with many columns
        m = LogisticRegression(C=10.0, max_iter=20000).fit(B, y); p = np.clip(m.predict_proba(B)[:, 1], 1e-12, 1 - 1e-12)
        return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    xs = ((x0 - x0.mean()) / x0.std())[:, None]
    lr = 2 * (ll_pen(np.hstack([Bs, xs])) - ll_pen(Bs))
    print(f"conditioning on the full vector ({Bs.shape[1]} columns, ridge C=10): p = {chi2.sf(max(lr, 0), 1):.2e}", flush=True)
