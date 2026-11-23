"""Nested-null decomposition: components and variance shares.
components(F, nT, nM) -> dict of image-level blocks LA, LD, LT, LR, LM.
python ladder_analysis.py : for every cohort, the share of the between-patient variance of each descriptor carried by
each component, share_j = Cov(c_j, phi) / Var(phi) (the shares sum to 1 exactly because the components sum to phi),
summarised by descriptor family (cell-type fields, cross-type, marker fields)."""
import os, sys, json
import numpy as np, pandas as pd

EPS = 0.05


def _split(F, nT, nM):
    T = F[:, :5 * nT].reshape(len(F), 5, nT); M = F[:, 5 * nT:].reshape(len(F), 6, nM)
    return T, M


def parts(F, nT, nM):
    """raw additive components; type: A, D, R, M ; marker: A, T, D, R, M"""
    T, M = _split(F, nT, nM)
    pt = dict(A=T[:, 1], D=T[:, 2] - T[:, 1], R=T[:, 3] - T[:, 2], M=T[:, 0] - T[:, 3])
    pm = dict(A=M[:, 1], T=M[:, 2] - M[:, 1], D=M[:, 3] - M[:, 2], R=M[:, 4] - M[:, 3], M=M[:, 0] - M[:, 4])
    return T, M, pt, pm


def components(F, nT, nM):
    T, M, pt, pm = parts(F, nT, nM)
    zt = pt["M"] / (T[:, 4] + EPS * (np.abs(T[:, 3]) + 1e-3) + 1e-6)
    zm = pm["M"] / (M[:, 5] + EPS * (np.abs(M[:, 4]) + 1e-3) + 1e-6)
    f32 = lambda a: np.nan_to_num(a).astype(np.float32)
    return dict(LA=f32(np.hstack([pt["A"], pm["A"]])), LD=f32(np.hstack([pt["D"], pm["D"]])), LT=f32(pm["T"]),
                LR=f32(np.hstack([pt["R"], pm["R"]])), LM=f32(np.clip(np.hstack([zt, zm]), -8, 8)))


if __name__ == "__main__":
    import common as C
    rows = []
    for c in ("danenberg", "basel", "zurich", "codex", "luad"):
        f = os.path.join(C.cohort_dir(c), "feats_ladder.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f); nT, nM = int(z["n_type_desc"]), int(z["n_marker_desc"])
        fe = np.load(os.path.join(C.cohort_dir(c), "feats.npz")); pid = fe["patient_id"].astype(str)
        meta = json.load(open(os.path.join(C.cohort_dir(c), "meta.json"))); K = len(meta["types"])
        T, M, pt, pm = parts(z["ladder"].astype(np.float64), nT, nM)
        agg = lambda a: pd.DataFrame(a).groupby(pid).mean().values
        fam_t = np.r_[np.repeat("cell-type fields", K * 15), np.repeat("cross-type", nT - K * 15)]
        for fam, obs, P in [("T", agg(T[:, 0]), {k: agg(v) for k, v in pt.items()}), ("M", agg(M[:, 0]), {k: agg(v) for k, v in pm.items()})]:
            v = obs.var(0); ok = v > 1e-12
            sh = {k: ((a - a.mean(0)) * (obs - obs.mean(0))).mean(0) / np.where(ok, v, 1) for k, a in P.items()}
            groups = {"marker fields": ok} if fam == "M" else {g: ok & (fam_t == g) for g in ("cell-type fields", "cross-type")}
            for g, m in groups.items():
                r = dict(cohort=c, family=g, n_desc=int(m.sum()))
                for k in ("A", "T", "D", "R", "M"):
                    if k in sh:
                        r[k] = float(np.median(sh[k][m])); r[k + "_mean"] = float(np.clip(sh[k][m], -1, 2).mean())
                r["sum_check"] = float(np.median(sum(sh.values())[m]))
                rows.append(r)
    R = pd.DataFrame(rows); pd.set_option("display.width", 220)
    print(R.round(3).to_string(index=False))
    R.to_csv(os.path.join(C.RES, "interp", "ladder_shares.csv"), index=False)
