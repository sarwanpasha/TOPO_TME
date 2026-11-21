"""Interpretation: which topological descriptors carry signal BEYOND composition?
For a cohort/task: base = composition + mean markers + log(#cells) (standardised, PCA to 15 comps);
for every topology descriptor j (TopoNull TEx, raw cell-type fields, cross-type persistence):
  * 5-fold x 3 repeats CV logistic regression: AUROC(base + j) - AUROC(base)
  * likelihood-ratio test of j given base (full data), Benjamini-Hochberg q-values
  * univariate AUROC of j and its correlation with composition (max |r| over composition columns)
Writes results/interp/<cohort>_<task>.csv and prints the top descriptors.
python interp.py <cohort> <task> [fset_list]"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import common as C
import feat_compact as FC
import feat_null as FN


def patient_matrix(cohort, key, names):
    fe = np.load(os.path.join(C.cohort_dir(cohort), "feats.npz"))
    pid = fe["patient_id"].astype(str)
    if key in ("comp", "mm", "logn"):
        X = fe[key].reshape(len(pid), -1)
    else:
        z = np.load(os.path.join(C.cohort_dir(cohort), f"feats_{key}.npz"))
        assert (z["image_id"] == fe["image_id"]).all()
        X = z[key]
    df = pd.DataFrame(X, columns=names[:X.shape[1]] if names is not None else None); df["pid"] = pid
    return df.groupby("pid").mean()


def cross_names(types):
    xs = ["H0tp", "ncomp@1", "ncomp@2", "nloops", "H1tp", "H1max"]
    return [f"X:{types[a]}~{types[b]}|{x}" for a in range(len(types)) for b in range(a, len(types)) for x in xs]


def loglik(X, y):
    m = LogisticRegression(C=1e4, max_iter=5000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def bh(p):
    p = np.nan_to_num(np.asarray(p, float), nan=1.0); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


if __name__ == "__main__":
    cohort, task = sys.argv[1], sys.argv[2]
    _, pat, meta = C.load_cohort(cohort)
    types, markers = meta["types"], meta["markers"]
    df, kind = C.task_frame(pat, cohort, task)
    assert kind == "bin"
    base = pd.concat([patient_matrix(cohort, k, None).add_prefix(k + "_") for k in ("comp", "mm", "logn")], axis=1)
    blocks = {}
    nc = FC.feature_names(markers, types)
    comp_names = [n for n in nc if n.startswith("T:")]
    try:
        blocks["znull"] = patient_matrix(cohort, "znull", FN.feature_names(markers, types))
    except FileNotFoundError:
        pass
    cp = patient_matrix(cohort, "compact", nc)
    blocks["typefield"] = cp[[c for c in cp.columns if str(c).startswith("T:")]]
    blocks["cross"] = patient_matrix(cohort, "cross", cross_names(types))
    T = pd.concat(blocks.values(), axis=1)
    ids = [p for p in df["patient_id"] if p in base.index and p in T.index]
    y = df.set_index("patient_id").loc[ids, "y"].values.astype(int)
    B = StandardScaler().fit_transform(np.nan_to_num(base.loc[ids].values))
    Bp = PCA(min(15, B.shape[1]), random_state=0).fit_transform(B)
    Tm = np.nan_to_num(T.loc[ids].values); Tm = np.where(np.isfinite(Tm), Tm, 0)
    keep = Tm.std(0) > 1e-9
    Tm = StandardScaler().fit_transform(Tm[:, keep]); tnames = np.array(T.columns)[keep]
    Bc = base.loc[ids].values
    # CV AUROC of base alone (shared splits)
    splits = [s for r in range(3) for s in StratifiedKFold(5, shuffle=True, random_state=r).split(Bp, y)]
    def cv_auc(X):
        out = []
        for tr, te in splits:
            m = LogisticRegression(C=0.5, max_iter=3000).fit(X[tr], y[tr])
            out.append(roc_auc_score(y[te], m.decision_function(X[te])))
        return float(np.mean(out))
    a0 = cv_auc(Bp); l0 = loglik(Bp, y)
    rows = []
    for j, n in enumerate(tnames):
        x = Tm[:, j:j + 1]
        X1 = np.hstack([Bp, x])
        lr = 2 * (loglik(X1, y) - l0)
        r = np.nanmax(np.abs([np.corrcoef(x[:, 0], Bc[:, k])[0, 1] for k in range(Bc.shape[1]) if Bc[:, k].std() > 0]))
        au = roc_auc_score(y, x[:, 0])
        rows.append(dict(feature=n, family=n.split(":")[0], uni_auc=au, direction=np.sign(au - 0.5),
                         lr_stat=lr, p=float(chi2.sf(max(lr, 0), 1)), max_abs_r_comp=r))
    R = pd.DataFrame(rows)
    R["q"] = bh(R["p"].values)
    R = R.sort_values("p")
    top = R.head(40).copy()
    top["cv_delta_auc"] = [cv_auc(np.hstack([Bp, Tm[:, list(tnames).index(f):list(tnames).index(f) + 1]])) - a0 for f in top["feature"]]
    od = os.path.join(C.RES, "interp"); os.makedirs(od, exist_ok=True)
    R.to_csv(os.path.join(od, f"{cohort}_{task}_all.csv"), index=False)
    top.to_csv(os.path.join(od, f"{cohort}_{task}_top.csv"), index=False)
    print(f"{cohort} {task}: n={len(y)} pos={y.sum()} base CV AUROC={a0:.3f}; descriptors={len(R)}; q<0.05: {(R.q < 0.05).sum()}")
    pd.set_option("display.width", 220)
    print(top[["feature", "uni_auc", "p", "q", "cv_delta_auc", "max_abs_r_comp"]].head(20).to_string(index=False))
