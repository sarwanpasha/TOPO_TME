"""Monte Carlo error and order sensitivity of the nested-null decomposition (feats_ladder.npz vs feats_ladder2.npz).
For every cohort and descriptor family:
  reliability : correlation across images between the components of 2 independent Monte Carlo runs (density first),
                median over descriptors; 1 - reliability is the share of between-image variance due to Monte Carlo error
  shares      : variance shares Cov(c_j, phi)/Var(phi) of density and regional components with density first,
                region first, and the Shapley average over the 2 orders (abundance, type-layout and micro components
                do not depend on the order)."""
import os, json
import numpy as np, pandas as pd
import common as C
from ladder_analysis import parts


def shares(obs, P):
    v = obs.var(0); ok = v > 1e-12
    return {k: ((a - a.mean(0)) * (obs - obs.mean(0))).mean(0) / np.where(ok, v, 1) for k, a in P.items()}, ok


def corr_cols(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    d = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return np.where(d > 1e-12, (a * b).sum(0) / np.where(d > 1e-12, d, 1), np.nan)


rows = []
for c in ("danenberg", "basel", "zurich", "codex", "luad"):
    f1, f2 = (os.path.join(C.cohort_dir(c), f) for f in ("feats_ladder.npz", "feats_ladder2.npz"))
    if not (os.path.exists(f1) and os.path.exists(f2)):
        continue
    z1, z2 = np.load(f1), np.load(f2); nT, nM = int(z1["n_type_desc"]), int(z1["n_marker_desc"])
    assert (z1["image_id"] == z2["image_id"]).all()
    K = len(json.load(open(os.path.join(C.cohort_dir(c), "meta.json")))["types"])
    _, _, pt1, pm1 = parts(z1["ladder"].astype(np.float64), nT, nM)
    F2 = z2["ladder2"].astype(np.float64); n = len(F2)
    T2 = F2[:, :9 * nT].reshape(n, 9, nT); M2 = F2[:, 9 * nT:].reshape(n, 11, nM)
    obsT, obsM = T2[:, 0], M2[:, 0]
    # run 2, density first and region first
    t_df = dict(A=T2[:, 1], D=T2[:, 2] - T2[:, 1], R=T2[:, 4] - T2[:, 2], M=obsT - T2[:, 4])
    t_rf = dict(R=T2[:, 3] - T2[:, 1], D=T2[:, 4] - T2[:, 3])
    m_df = dict(A=M2[:, 1], T=M2[:, 2] - M2[:, 1], D=M2[:, 3] - M2[:, 2], R=M2[:, 5] - M2[:, 3], M=obsM - M2[:, 5])
    m_rf = dict(R=M2[:, 4] - M2[:, 2], D=M2[:, 5] - M2[:, 4])
    fam_t = np.r_[np.repeat("cell-type fields", K * 15), np.repeat("cross-type", nT - K * 15)]
    for fam, obs, p1, df, rf in (("T", obsT, pt1, t_df, t_rf), ("M", obsM, pm1, m_df, m_rf)):
        sh_df, ok = shares(obs, df)
        sh_rf, _ = shares(obs, rf)
        rel = {k: corr_cols(p1[k], df[k]) for k in df}
        groups = {"marker fields": ok} if fam == "M" else {g: ok & (fam_t == g) for g in ("cell-type fields", "cross-type")}
        for g, m in groups.items():
            r = dict(cohort=c, family=g, n_desc=int(m.sum()))
            for k in df:
                r[f"rel_{k}"] = float(np.nanmedian(rel[k][m]))
            for k in ("D", "R"):
                r[f"{k}_dfirst"] = float(np.median(sh_df[k][m])); r[f"{k}_rfirst"] = float(np.median(sh_rf[k][m]))
                r[f"{k}_shapley"] = float(np.median(0.5 * (sh_df[k][m] + sh_rf[k][m])))
            rows.append(r)
R = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
R.to_csv(os.path.join(C.RES, "interp", "ladder2_mc_order.csv"), index=False)
