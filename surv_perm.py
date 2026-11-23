"""Permutation calibration of the nested cross-validated topology risk score (addresses the optimism of testing
pre-validated scores with a nominal chi-square reference).
The whole pipeline of surv_oof.py (screening, ridge Cox on a clinical offset, inner-CV choice of k and lambda, R repeats
of 5-fold CV, final likelihood-ratio statistic) is re-run after permuting the rows of the topology matrix across
patients, which breaks any link between topology and (outcome, covariates) while keeping everything else fixed.
p_perm = (1 + #{LR_perm >= LR_obs}) / (1 + n_perm). Clinical models do not depend on topology and are cached.
python surv_perm.py <cohort> <endpoint> [n_perm=500] [repeats=5] [workers=5]"""
import os, sys, json, warnings
import numpy as np, pandas as pd, multiprocessing as mp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index
import common as C
import feat_compact as FC, feat_null as FN
from interp import patient_matrix, cross_names
from surv_oof import cox_fit, lp, fit_topo, final_test, KS, LAMS
warnings.filterwarnings("ignore")
G = {}


def load(cohort, ep):
    _, pat, meta = C.load_cohort(cohort)
    df, _ = C.task_frame(pat, cohort, ep)
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
    return t, e, Bcl, np.hstack([Bcl, Bco]), Tm


def build_cache(B, t, e, R):
    """clinical offsets for every outer fold and every inner fold (independent of topology)."""
    cache = []
    for r in range(R):
        for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=100 + r).split(B, e)):
            m = cox_fit(B[tr], t[tr], e[tr]); inner = []
            for a, v in StratifiedKFold(3, shuffle=True, random_state=r * 10 + f).split(B[tr], e[tr]):
                mi = cox_fit(B[tr][a], t[tr][a], e[tr][a]); ov = lp(mi, B[tr][v])
                inner.append((a, v, lp(mi, B[tr][a]), ov, concordance_index(t[tr][v], -ov, e[tr][v])))
            cache.append((r, tr, te, lp(m, B[tr]), inner))
    return cache


def pipeline(Tm, B, t, e, cache, R):
    S = np.zeros((R, len(t)))
    for r, tr, te, off, inner in cache:
        Ztr, ttr, etr = Tm[tr], t[tr], e[tr]
        best, arg = -9, None
        for k in KS:
            for lam in LAMS:
                ds = []
                for a, v, oa, ov, c0 in inner:
                    sel, b = fit_topo(Ztr[a], ttr[a], etr[a], oa, k, lam)
                    ds.append(concordance_index(ttr[v], -(ov + Ztr[v][:, sel] @ b), etr[v]) - c0)
                if np.mean(ds) > best:
                    best, arg = np.mean(ds), (k, lam)
        sel, b = fit_topo(Ztr, ttr, etr, off, *arg)
        s = Tm[te][:, sel] @ b
        S[r, te] = (s - s.mean()) / (s.std() + 1e-9)
    return final_test(B, S.mean(0), t, e)


def _init(g):
    G.update(g)


def one(seed):
    rng = np.random.default_rng(seed)
    return pipeline(G["Tm"][rng.permutation(len(G["t"]))], G["B"], G["t"], G["e"], G["cache"], G["R"])["lr"]


if __name__ == "__main__":
    cohort, ep = sys.argv[1], sys.argv[2]
    NP = int(sys.argv[3]) if len(sys.argv) > 3 else 500; R = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    W = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    t, e, Bcl, Bcc, Tm = load(cohort, ep)
    _wf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perm_workers.txt")
    if os.path.exists(_wf):
        W = int(open(_wf).read().strip())
    BASES = sys.argv[6].split(",") if len(sys.argv) > 6 else ["clinical", "clinical+composition"]
    for base_name, B in (("clinical", Bcl), ("clinical+composition", Bcc)):
        if base_name not in BASES:
            continue
        cache = build_cache(B, t, e, R)
        obs = pipeline(Tm, B, t, e, cache, R)
        with mp.get_context("spawn").Pool(W, initializer=_init, initargs=(dict(Tm=Tm, B=B, t=t, e=e, cache=cache, R=R),)) as p:
            null = []
            for v in p.imap_unordered(one, range(1000, 1000 + NP), chunksize=1):
                null.append(v)
                if len(null) % 5 == 0:
                    print("progress", cohort, ep, base_name, len(null), NP, flush=True)
            null = np.array(null)
        res = dict(cohort=cohort, endpoint=ep, base=base_name, n=len(t), events=int(e.sum()), hr_per_sd=obs["hr_per_sd"],
                   lr_obs=obs["lr"], p_chi2=obs["p"], n_perm=NP, repeats=R, p_perm=float((1 + (null >= obs["lr"]).sum()) / (1 + NP)),
                   null_mean_lr=float(null.mean()), null_q95=float(np.quantile(null, 0.95)), frac_null_gt_3p84=float((null > 3.84).mean()))
        print(json.dumps(res), flush=True)
        np.save(os.path.join(C.RES, "interp", f"permnull_{cohort}_{ep}_{base_name.replace('+', '_')}.npy"), null)

