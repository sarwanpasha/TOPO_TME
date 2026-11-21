"""Does topology add prognostic information BEYOND clinical covariates (and composition)?
For a cohort/endpoint: penalised Cox (lifelines) on clinical covariates [+ composition PCs]; for each topological
descriptor: likelihood-ratio test of adding it (BH-FDR over all descriptors), hazard ratio per SD, and the
5x3 cross-validated C-index gain. Also reports PRE-SPECIFIED descriptors (chosen from the classification
interpretation, i.e. on different endpoints) so that their tests need no multiplicity correction.
python surv_interp.py <cohort> <endpoint> [prespecified,comma,separated]"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import common as C
import feat_compact as FC
import feat_null as FN
from interp import patient_matrix, cross_names, bh
warnings.filterwarnings("ignore")


def fit_ll(X, t, e, pen=0.1):
    d = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]); d["T"] = t; d["E"] = e
    m = CoxPHFitter(penalizer=pen).fit(d, "T", "E")
    return m, float(m.log_likelihood_)


def cv_cindex(X, t, e, splits, pen=0.1):
    out = []
    for tr, te in splits:
        m, _ = fit_ll(X[tr], t[tr], e[tr], pen)
        r = m.predict_partial_hazard(pd.DataFrame(X[te], columns=[f"x{i}" for i in range(X.shape[1])])).values.ravel()
        out.append(concordance_index(t[te], -r, e[te]))
    return float(np.mean(out))


if __name__ == "__main__":
    cohort, ep = sys.argv[1], sys.argv[2]
    pre = sys.argv[3].split(",") if len(sys.argv) > 3 else []
    _, pat, meta = C.load_cohort(cohort)
    df, kind = C.task_frame(pat, cohort, ep)
    cov = [c for c in meta.get("covariates", []) if c in pat.columns]
    cl = pat.set_index("patient_id")[cov].apply(pd.to_numeric, errors="coerce"); cl = cl.fillna(cl.median())
    comp = pd.concat([patient_matrix(cohort, k, None).add_prefix(k + "_") for k in ("comp", "logn")], axis=1)
    types, markers = meta["types"], meta["markers"]
    blocks = [patient_matrix(cohort, "znull", FN.feature_names(markers, types))]
    cp = patient_matrix(cohort, "compact", FC.feature_names(markers, types))
    blocks.append(cp[[c for c in cp.columns if str(c).startswith("T:")]])
    blocks.append(patient_matrix(cohort, "cross", cross_names(types)))
    T = pd.concat(blocks, axis=1)
    ids = [p for p in df["patient_id"] if p in comp.index and p in T.index and p in cl.index]
    d = df.set_index("patient_id").loc[ids]; t = d["time"].values.astype(float); e = d["event"].values.astype(int)
    Bcl = StandardScaler().fit_transform(cl.loc[ids].values) if cov else np.zeros((len(ids), 0))
    Bco = PCA(8, random_state=0).fit_transform(StandardScaler().fit_transform(np.nan_to_num(comp.loc[ids].values)))
    Tm = np.nan_to_num(T.loc[ids].values); keep = Tm.std(0) > 1e-9
    Tm = np.clip(StandardScaler().fit_transform(Tm[:, keep]), -5, 5); tn = np.array(T.columns)[keep]
    splits = [s for r in range(3) for s in StratifiedKFold(5, shuffle=True, random_state=r).split(Tm, e)]
    res = {}
    for base_name, B in (("clinical", Bcl), ("clinical+composition", np.hstack([Bcl, Bco]))):
        if B.shape[1] == 0:
            continue
        _, l0 = fit_ll(B, t, e); c0 = cv_cindex(B, t, e, splits)
        rows = []
        for j, n in enumerate(tn):
            try:
                m, l1 = fit_ll(np.hstack([B, Tm[:, j:j + 1]]), t, e)
                hr = float(np.exp(m.params_.iloc[-1]))
            except Exception:
                l1, hr = l0, 1.0
            lr = 2 * (l1 - l0)
            rows.append(dict(feature=n, lr=lr, p=float(chi2.sf(max(lr, 0), 1)), hr_per_sd=hr))
        R = pd.DataFrame(rows); R["q"] = bh(R["p"].values); R = R.sort_values("p")
        top = R.head(10).copy()
        top["cv_dC"] = [cv_cindex(np.hstack([B, Tm[:, list(tn).index(f):list(tn).index(f) + 1]]), t, e, splits) - c0 for f in top["feature"]]
        ps = R[R["feature"].isin(pre)].copy()
        if len(ps):
            ps["cv_dC"] = [cv_cindex(np.hstack([B, Tm[:, list(tn).index(f):list(tn).index(f) + 1]]), t, e, splits) - c0 for f in ps["feature"]]
        print(f"== {cohort} {ep} | base={base_name} n={len(t)} events={e.sum()} CV C={c0:.3f} | descriptors={len(R)} q<0.05: {(R.q < 0.05).sum()} q<0.10: {(R.q < 0.10).sum()}")
        print(top[["feature", "hr_per_sd", "p", "q", "cv_dC"]].to_string(index=False))
        if len(ps):
            print("  pre-specified:"); print(ps[["feature", "hr_per_sd", "p", "cv_dC"]].to_string(index=False))
        od = os.path.join(C.RES, "interp"); os.makedirs(od, exist_ok=True)
        R.to_csv(os.path.join(od, f"surv_{cohort}_{ep}_{base_name.replace('+', '_')}.csv"), index=False)
