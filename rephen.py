"""Receptor-field descriptors with cell types re-derived WITHOUT receptor markers (circularity check).
Cells of each cohort are clustered (k-means on the standardized panel minus ER/PR/HER2/GATA3) into as many clusters as the
original taxonomy has types; the receptor-field TopoNull descriptors are recomputed with marker vectors permuted within
these receptor-blind types, and tested for association with receptor status conditional on composition, mean markers and
log cell count (first 15 principal components), exactly as in interp.py.
python rephen.py <cohort> <task> <field marker index> [workers]"""
import os, sys, zlib, time
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import common as C
import feat_compact as FC
from interp import patient_matrix, loglik

EXCL = {"danenberg": ["ER", "HER2 (3B5)", "HER2 (D8F12)"], "basel": ["Estroge", "Progest", "cerbB", "GATA3"]}
STATS = ["tp", "n_sig", "wbirth", "pmax", "ent", "top5"]


def field(im, z, scale):
    E, TE, Fc = im["edges"], im["tri_edges"], im["faces"]
    xy = im["coords"].astype(np.float64) / scale
    el = FC.BETA * np.linalg.norm(xy[E[:, 0]] - xy[E[:, 1]], axis=1)
    out = []
    for sign in (-1.0, 1.0):  # sup, sub
        fv = sign * z
        fe = np.maximum(fv[E[:, 0]], fv[E[:, 1]]) + el
        out.append(FC.field_feats(fv, fe, E, TE, Fc, FC.LEVEL)[:12])
    return np.concatenate(out)


def one(args):
    im, lab, mi, scale = args
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(im["coords"][:50]).tobytes()) + 5)
    z = np.clip(im["markers"][:, mi].astype(np.float64), -4, 4)
    obs = field(im, z, scale)
    groups = [np.flatnonzero(lab == k) for k in np.unique(lab)]
    nul = []
    for _ in range(8):
        zp = z.copy()
        for g in groups:
            if len(g) > 1:
                zp[g] = z[rng.permutation(g)]
        nul.append(field(im, zp, scale))
    nul = np.stack(nul); mu, sd = nul.mean(0), nul.std(0)
    return np.clip(np.nan_to_num((obs - mu) / (sd + 0.05 * (np.abs(mu) + 1e-3) + 1e-6)), -8, 8)


if __name__ == "__main__":
    import multiprocessing as mp
    cohort, task, mi = sys.argv[1], sys.argv[2], int(sys.argv[3]); W = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    imgs, pat, meta = C.load_cohort(cohort); markers = meta["markers"]
    keep = [j for j, m in enumerate(markers) if m not in EXCL[cohort]]
    print("field marker", markers[mi], "| excluded", EXCL[cohort], flush=True)
    rng = np.random.default_rng(0)
    allZ = np.concatenate([im["markers"][:, keep] for im in imgs])
    sub = allZ[rng.choice(len(allZ), min(300000, len(allZ)), replace=False)]
    sc = StandardScaler().fit(sub)
    km = MiniBatchKMeans(len(meta["types"]), random_state=0, n_init=5, batch_size=8192).fit(sc.transform(sub))
    labs = [km.predict(sc.transform(im["markers"][:, keep])) for im in imgs]
    # agreement with the original taxonomy
    ot = np.concatenate([np.asarray(im["types"]).astype(int) for im in imgs]); nl = np.concatenate(labs)
    from sklearn.metrics import adjusted_rand_score
    print("ARI new vs original types", round(adjusted_rand_score(ot[::20], nl[::20]), 3), flush=True)
    JOINT = len(sys.argv) > 5 and sys.argv[5] == "joint"
    if JOINT:  # strictest null: permute within (original type x receptor-blind cluster)
        labs = [np.asarray(im["types"]).astype(int) * 1000 + l for im, l in zip(imgs, labs)]
        print("joint strata: original type x receptor-blind cluster", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(W) as p:
        F = np.stack(p.map(one, [(dict(coords=im["coords"], markers=im["markers"], edges=im["edges"], tri_edges=im["tri_edges"],
                                        faces=im["faces"]), l, mi, meta["scale"]) for im, l in zip(imgs, labs)], chunksize=4))
    print("features", F.shape, f"{time.time()-t0:.0f}s", flush=True)
    names = [f"zM:{markers[mi]}|{s}|{h}|{x}" for s in ("sup", "sub") for h in ("H0", "H1") for x in STATS]
    pid = np.array([str(im["patient_id"]) for im in imgs])
    T = pd.DataFrame(F, columns=names); T["pid"] = pid; T = T.groupby("pid").mean()
    df, kind = C.task_frame(pat, cohort, task)
    base = pd.concat([patient_matrix(cohort, k, None).add_prefix(k + "_") for k in ("comp", "mm", "logn")], axis=1)
    ids = [q for q in df["patient_id"] if q in base.index and q in T.index]
    y = df.set_index("patient_id").loc[ids, "y"].values.astype(int)
    Bp = PCA(15, random_state=0).fit_transform(StandardScaler().fit_transform(np.nan_to_num(base.loc[ids].values)))
    l0 = loglik(Bp, y)
    rows = []
    for n in names:
        x = T.loc[ids, n].values
        if x.std() < 1e-9:
            continue
        x = (x - x.mean()) / x.std()
        lr = 2 * (loglik(np.hstack([Bp, x[:, None]]), y) - l0)
        rows.append((n, roc_auc_score(y, x), lr, chi2.sf(max(lr, 0), 1)))
    R = pd.DataFrame(rows, columns=["feature", "uni_auc", "lr", "p"]).sort_values("p")
    pd.set_option("display.width", 200)
    print(R.to_string(index=False), flush=True)
    R.to_csv(os.path.join(C.RES, "interp", f"rephen_{cohort}_{task}{'_joint' if JOINT else ''}.csv"), index=False)
