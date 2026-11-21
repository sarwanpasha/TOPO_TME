"""Nested cross-validated topology risk score: does it add prognostic information beyond clinical covariates?
For each outer fold (5-fold, repeated R times): on the TRAINING patients only,
  (1) a ridge Cox model on clinical covariates gives an offset;
  (2) topological descriptors are screened by the score test beyond that offset (top-k);
  (3) a ridge Cox model on the k descriptors is fitted on top of the offset; (k, lambda) chosen by inner 3-fold CV.
The frozen model scores the held-out patients. The out-of-fold topology score (averaged over repeats; no patient's
score ever used its own outcome) then enters ONE Cox model together with the clinical covariates on the full cohort:
likelihood-ratio test (1 df), HR per SD, and C-index. Optionally the base also includes composition PCs.
python surv_oof.py <cohort> <endpoint> [repeats]"""
import os, sys, json, warnings
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
from interp import patient_matrix, cross_names
from surv_loco import strat_ridgecox
warnings.filterwarnings("ignore")
KS, LAMS = (10, 25, 50), (10.0, 50.0, 200.0)


def cox_fit(B, t, e, pen=0.1):
    d = pd.DataFrame(B, columns=[f"b{i}" for i in range(B.shape[1])]); d["T"] = t; d["E"] = e
    return CoxPHFitter(penalizer=pen).fit(d, "T", "E")


def lp(m, B):
    return m.predict_log_partial_hazard(pd.DataFrame(B, columns=[f"b{i}" for i in range(B.shape[1])])).values.ravel()


def score_test(Z, t, e, off):
    o = np.argsort(t, kind="stable"); Zs, es = Z[o], e[o]; w = np.exp(np.clip(off[o], -30, 30))
    S0 = np.cumsum(w[::-1])[::-1]; S1 = np.cumsum((w[:, None] * Zs)[::-1], 0)[::-1]
    S2 = np.cumsum((w[:, None] * Zs ** 2)[::-1], 0)[::-1]
    mu = S1 / S0[:, None]
    U = (es[:, None] * (Zs - mu)).sum(0); I = (es[:, None] * (S2 / S0[:, None] - mu ** 2)).sum(0)
    return U ** 2 / (I + 1e-9)


def fit_topo(Z, t, e, off, k, lam):
    sel = np.argsort(-score_test(Z, t, e, off))[:k]
    b = strat_ridgecox([(Z[:, sel], t, e, off)], lam)
    return sel, b


def train_predict(Btr, Ztr, ttr, etr, Bte, Zte, seed):
    m = cox_fit(Btr, ttr, etr); off = lp(m, Btr)
    best, arg = -9, None
    inner = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(Ztr, etr))
    for k in KS:
        for lam in LAMS:
            ds = []
            for a, v in inner:
                mi = cox_fit(Btr[a], ttr[a], etr[a]); oa, ov = lp(mi, Btr[a]), lp(mi, Btr[v])
                sel, b = fit_topo(Ztr[a], ttr[a], etr[a], oa, k, lam)
                ds.append(concordance_index(ttr[v], -(ov + Zte_safe(Ztr[v], sel) @ b), etr[v]) - concordance_index(ttr[v], -ov, etr[v]))
            if np.mean(ds) > best:
                best, arg = np.mean(ds), (k, lam)
    sel, b = fit_topo(Ztr, ttr, etr, off, *arg)
    return Zte[:, sel] @ b, arg


def Zte_safe(Z, sel):
    return Z[:, sel]


def final_test(B, s, t, e):
    s = (s - s.mean()) / (s.std() + 1e-9)
    m0 = cox_fit(B, t, e, 0.01); m1 = cox_fit(np.hstack([B, s[:, None]]), t, e, 0.01)
    lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
    nm = f"b{B.shape[1]}"
    return dict(hr_per_sd=float(np.exp(m1.params_[nm])), z=float(m1.summary.loc[nm, "z"]), lr=float(lr),
                p=float(chi2.sf(max(lr, 0), 1)),
                C_base=float(concordance_index(t, -lp(m0, B), e)),
                C_base_topo=float(concordance_index(t, -lp(m1, np.hstack([B, s[:, None]])), e)))


if __name__ == "__main__":
    cohort, ep = sys.argv[1], sys.argv[2]; R = int(sys.argv[3]) if len(sys.argv) > 3 else 5
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
    Bcl = StandardScaler().fit_transform(cl.loc[ids].values)
    Bco = PCA(8, random_state=0).fit_transform(StandardScaler().fit_transform(np.nan_to_num(comp.loc[ids].values)))
    Tm = np.nan_to_num(T.loc[ids].values); keep = Tm.std(0) > 1e-9
    Tm = np.clip(StandardScaler().fit_transform(Tm[:, keep]), -5, 5)
    od = os.path.join(C.RES, "interp"); os.makedirs(od, exist_ok=True)
    for base_name, B in (("clinical", Bcl), ("clinical+composition", np.hstack([Bcl, Bco]))):
        S = np.zeros((R, len(t))); picks = []
        for r in range(R):
            for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=100 + r).split(Tm, e)):
                s, arg = train_predict(B[tr], Tm[tr], t[tr], e[tr], B[te], Tm[te], seed=r * 10 + f)
                S[r, te] = (s - s.mean()) / (s.std() + 1e-9); picks.append(arg)
        per = [final_test(B, S[r], t, e)["p"] for r in range(R)]
        res = final_test(B, S.mean(0), t, e)
        res.update(cohort=cohort, endpoint=ep, base=base_name, n=len(t), events=int(e.sum()), repeats=R,
                   p_per_repeat_median=float(np.median(per)), p_per_repeat_max=float(np.max(per)),
                   choices=pd.Series([str(a) for a in picks]).value_counts().head(3).to_dict())
        print(json.dumps(res), flush=True)
        pd.DataFrame([res]).to_csv(os.path.join(od, f"oof_{cohort}_{ep}_{base_name.replace('+', '_')}.csv"), index=False)
        np.save(os.path.join(od, f"oofscore_{cohort}_{ep}_{base_name.replace('+', '_')}.npy"), S.mean(0))
