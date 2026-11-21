"""Leave-one-cohort-out external validation: does harmonised topology add prognostic information beyond clinical covariates?
Training cohorts: per cohort, a ridge Cox model on that cohort's clinical covariates gives an offset (clinical linear predictor);
a single stratified ridge Cox model on harmonised TopoNull descriptors (z-scored within cohort) is fitted ON TOP of the offsets.
Held-out cohort: the frozen topology score enters a Cox model together with the held-out cohort's own clinical covariates;
one likelihood-ratio test per held-out cohort (pre-specified), then Stouffer meta-analysis across cohorts.
python surv_loco.py [endpoint_map_json] [--comp]"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2, norm
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import common as C
warnings.filterwarnings("ignore")

EP = {"danenberg": "OS", "basel": "OS", "luad": "OS"}
USE_COMP = "--comp" in sys.argv
for a in sys.argv[1:]:
    if a.startswith("{"):
        EP = json.loads(a)


def load(c):
    _, pat, meta = C.load_cohort(c)
    df, _ = C.task_frame(pat, c, EP[c])
    fe = np.load(os.path.join(C.cohort_dir(c), "feats.npz")); h = np.load(os.path.join(C.cohort_dir(c), "feats_hnull.npz"))
    assert (h["image_id"] == fe["image_id"]).all()
    H = pd.DataFrame(h["hnull"], columns=list(h["names"])); H["pid"] = fe["patient_id"].astype(str)
    H = H.groupby("pid").mean()
    cov = [x for x in meta.get("covariates", []) if x in pat.columns]
    cl = pat.set_index("patient_id")[cov].apply(pd.to_numeric, errors="coerce"); cl = cl.fillna(cl.median())
    ids = [p for p in df["patient_id"] if p in H.index and p in cl.index]
    d = df.set_index("patient_id").loc[ids]
    topo = H.loc[ids, [n for n in H.columns if not n.startswith("hfrac:")]]
    frac = H.loc[ids, [n for n in H.columns if n.startswith("hfrac:")]]
    Z = topo.values; Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9); Z = np.clip(np.nan_to_num(Z), -5, 5)
    B = cl.loc[ids].values; B = (B - B.mean(0)) / (B.std(0) + 1e-9)
    if USE_COMP:
        F = frac.values; F = (F - F.mean(0)) / (F.std(0) + 1e-9); B = np.hstack([B, np.nan_to_num(F)])
    return dict(t=d["time"].values.astype(float), e=d["event"].values.astype(int), Z=Z, B=np.nan_to_num(B), names=list(topo.columns))


def cox_lp(B, t, e, pen=0.1):
    keep = B.std(0) > 1e-9; B = B[:, keep]
    d = pd.DataFrame(B, columns=[f"b{i}" for i in range(B.shape[1])]); d["T"] = t; d["E"] = e
    m = CoxPHFitter(penalizer=pen).fit(d, "T", "E")
    return m.predict_log_partial_hazard(d).values.ravel()


def strat_ridgecox(blocks, lam):
    """blocks: list of (Z, t, e, offset); shared coefficients, stratum-specific baseline hazards."""
    p = blocks[0][0].shape[1]; b = np.zeros(p)
    for _ in range(50):
        g = -lam * b; H = lam * np.eye(p)
        for Z, t, e, off in blocks:
            o = np.argsort(t, kind="stable"); Zs, es, os_ = Z[o], e[o], off[o]
            w = np.exp(np.clip(Zs @ b + os_, -30, 30))
            S0 = np.cumsum(w[::-1])[::-1]; S1 = np.cumsum((w[:, None] * Zs)[::-1], 0)[::-1]; mu = S1 / S0[:, None]
            g += (es[:, None] * (Zs - mu)).sum(0)
            S2 = np.cumsum((w[:, None, None] * Zs[:, :, None] * Zs[:, None, :])[::-1], 0)[::-1]
            H += (es[:, None, None] * (S2 / S0[:, None, None] - mu[:, :, None] * mu[:, None, :])).sum(0)
        st = np.linalg.solve(H, g); b = b + st
        if np.abs(st).max() < 1e-7:
            break
    return b


def held_out_test(D, score):
    B = D["B"]; keep = B.std(0) > 1e-9; B = B[:, keep]
    s = (score - score.mean()) / (score.std() + 1e-9)
    d0 = pd.DataFrame(B, columns=[f"b{i}" for i in range(B.shape[1])]); d0["T"] = D["t"]; d0["E"] = D["e"]
    m0 = CoxPHFitter(penalizer=0.01).fit(d0, "T", "E")
    d1 = d0.copy(); d1["topo"] = s
    m1 = CoxPHFitter(penalizer=0.01).fit(d1, "T", "E")
    lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
    hr = float(np.exp(m1.params_["topo"])); z = float(m1.summary.loc["topo", "z"])
    c0 = concordance_index(D["t"], -m0.predict_partial_hazard(d0).values.ravel(), D["e"])
    c1 = concordance_index(D["t"], -m1.predict_partial_hazard(d1).values.ravel(), D["e"])
    return dict(hr_per_sd=hr, z=z, lr=float(lr), p=float(chi2.sf(max(lr, 0), 1)), C_clin=c0, C_clin_topo=c1)


if __name__ == "__main__":
    data = {c: load(c) for c in EP}
    names = data[next(iter(data))]["names"]
    for c in data:
        assert data[c]["names"] == names
    rows = []
    for test in data:
        train = [c for c in data if c != test]
        blocks = [(data[c]["Z"], data[c]["t"], data[c]["e"], cox_lp(data[c]["B"], data[c]["t"], data[c]["e"])) for c in train]
        # lambda by leave-one-training-cohort-out C-index (beyond offset)
        best, bl = -1, None
        for lam in (1, 10, 30, 100, 300, 1000):
            cs = []
            for i in range(len(blocks)):
                b = strat_ridgecox([blk for j, blk in enumerate(blocks) if j != i], lam)
                Z, t, e, off = blocks[i]
                cs.append(concordance_index(t, -(off + Z @ b), e) - concordance_index(t, -off, e))
            if np.mean(cs) > best:
                best, bl = np.mean(cs), lam
        b = strat_ridgecox(blocks, bl)
        r = held_out_test(data[test], data[test]["Z"] @ b)
        top = np.argsort(-np.abs(b))[:5]
        r.update(test=test, endpoint=EP[test], train="+".join(train), lam=bl, n=len(data[test]["t"]), events=int(data[test]["e"].sum()),
                 top_weights="; ".join(f"{names[i]}={b[i]:+.3f}" for i in top))
        rows.append(r)
        print(json.dumps(r), flush=True)
    R = pd.DataFrame(rows)
    zs = np.sign(-np.log(R["hr_per_sd"])) * norm.isf(R["p"] / 2)   # protective direction positive
    zcomb = zs.sum() / np.sqrt(len(zs))
    print(f"STOUFFER (risk-direction-aligned): z={zcomb:.3f}  p_two_sided={2*norm.sf(abs(zcomb)):.4g}  "
          f"(using |z| per cohort with sign of association)")
    zr = R["z"].values; zc2 = zr.sum() / np.sqrt(len(zr))
    print(f"STOUFFER (signed Wald z of score; positive=score increases hazard): z={zc2:.3f} p={2*norm.sf(abs(zc2)):.4g}")
    tag = ("comp_" if USE_COMP else "") + "_".join(f"{k}{v}" for k, v in EP.items())
    R.to_csv(os.path.join(C.RES, "interp", f"loco_{tag}.csv"), index=False)
