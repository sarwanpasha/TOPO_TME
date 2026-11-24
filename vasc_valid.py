"""Orthogonal vascular metrics for LUAD (microvessel-density style counts from the cell map) and their relation to the
calibrated CD31-field descriptors that separate solid from non-solid tumors.
Vessel proxy: connected components of endothelial cells in the Delaunay graph using edges no longer than 1.5 median edge
lengths (the counting rule of microvessel density: every separate endothelial cluster is 1 vessel).
Per image: vessels per 1,000 cells, mean and maximum vessel size (cells), fraction of endothelial cells in vessels of
>= 5 cells, and the fraction of vessels that enclose a non-endothelial cell (lumen-like: an H1 loop in the vessel graph)."""
import os, numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import common as C
import feat_null as FN
from interp import patient_matrix, loglik

imgs, pat, meta = C.load_cohort("luad"); types = meta["types"]; ei = types.index("Endothelial cell")
rows = []
for im in imgs:
    t = np.asarray(im["types"]).astype(int); n = len(t)
    E = np.asarray(im["edges"]); xy = im["coords"].astype(float) / meta["scale"]
    el = np.linalg.norm(xy[E[:, 0]] - xy[E[:, 1]], axis=1)
    en = t == ei; k = en[E[:, 0]] & en[E[:, 1]] & (el <= 1.5)
    A = coo_matrix((np.ones(k.sum()), (E[k, 0], E[k, 1])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab[en]); sizes = sizes[sizes > 0]
    # cycles in the vessel graph: edges - vertices + components (graph cycle rank)
    ncyc = int(k.sum() - en.sum() + len(sizes)) if en.sum() else 0
    rows.append(dict(pid=str(im["patient_id"]), vessels_per_1000=1000 * len(sizes) / n,
                     vessel_mean=sizes.mean() if len(sizes) else 0, vessel_max=sizes.max() if len(sizes) else 0,
                     frac_in_large=(sizes[sizes >= 5].sum() / max(en.sum(), 1)) if len(sizes) else 0,
                     cycles_per_vessel=ncyc / max(len(sizes), 1), endo_frac=en.mean()))
V = pd.DataFrame(rows).groupby("pid").mean()
zn = patient_matrix("luad", "znull", FN.feature_names(meta["markers"], types))
cd = [c for c in zn.columns if c.startswith("zM:CD31|")]
df, kind = C.task_frame(pat, "luad", "solid")
ids = [q for q in df["patient_id"] if q in V.index and q in zn.index]
y = df.set_index("patient_id").loc[ids, "y"].values.astype(int)
print("patients", len(ids), "solid", y.sum())
print("\n== solid vs non-solid, median vascular metrics")
for c in V.columns:
    print(f"   {c:18s} non-solid {np.median(V.loc[ids, c].values[y == 0]):.3f}  solid {np.median(V.loc[ids, c].values[y == 1]):.3f}")
top = pd.read_csv(os.path.join(C.RES, "interp", "luad_solid_top.csv"))
feats = [f for f in top["feature"] if f.startswith("zM:CD31|")][:4]
print("\n== Spearman correlation of leading CD31-field descriptors with vascular metrics")
for f in feats:
    print("  ", f, " ".join(f"{c}={spearmanr(zn.loc[ids, f], V.loc[ids, c])[0]:+.2f}" for c in V.columns))
base = pd.concat([patient_matrix("luad", k, None).add_prefix(k + "_") for k in ("comp", "mm", "logn")], axis=1)
Bp = PCA(15, random_state=0).fit_transform(StandardScaler().fit_transform(np.nan_to_num(base.loc[ids].values)))
Vs = StandardScaler().fit_transform(V.loc[ids].values)
B2 = np.hstack([Bp, Vs])
print("\n== LR test for solid histology: descriptor given base, and given base + all vascular metrics")
for f in feats:
    x = zn.loc[ids, f].values; x = ((x - x.mean()) / x.std())[:, None]
    l1 = 2 * (loglik(np.hstack([Bp, x]), y) - loglik(Bp, y)); l2 = 2 * (loglik(np.hstack([B2, x]), y) - loglik(B2, y))
    print(f"   {f:32s} base p={chi2.sf(l1, 1):.2e}   base+vascular p={chi2.sf(l2, 1):.2e}")
lv = 2 * (loglik(B2, y) - loglik(Bp, y))
print(f"vascular metrics given base: LR={lv:.1f} df={Vs.shape[1]} p={chi2.sf(lv, Vs.shape[1]):.2e}")
