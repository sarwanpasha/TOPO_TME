"""Summarise results: mean +- sd of test metric over folds x seeds, per cohort/task/method."""
import os, json, glob, sys
import pandas as pd, numpy as np
R = os.path.join(os.environ.get("TOPO_ROOT", "."), "results")
rows = []
for f in glob.glob(os.path.join(R, "*", "*", "*", "*.json")):
    c, t, m = f.split(os.sep)[-4:-1]
    if c.endswith("_smoke"):
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rows.append(dict(cohort=c, task=t, method=m, metric=d.get("test_metric"), kind=d.get("kind")))
df = pd.DataFrame(rows)
g = df.groupby(["cohort", "task", "method"])["metric"].agg(["mean", "std", "count"]).reset_index()
g.to_csv(os.path.join(R, "..", "summary_all.csv"), index=False)
OURS = lambda m: m.startswith("mph2") or m in ("topo3", "topo3_nognn", "topo3g") or any(m.endswith(x) for x in ("_mfield", "_bank", "_mfield2", "_bank2", "_compact", "_cross", "_ours", "_c_compact", "_c_cross", "_c_mfield2", "_c_ours", "_c_ctype", "_c_ctype_cross"))
for (c, t), sub in g.groupby(["cohort", "task"]):
    sub = sub.set_index("method")["mean"]
    o = sub[[m for m in sub.index if OURS(m)]]; b = sub[[m for m in sub.index if not OURS(m)]]
    st = sub.get("mph2_stack", np.nan); bk = max(sub.get("lgbm_bank", np.nan) if not np.isnan(sub.get("lgbm_bank", np.nan)) else -1,
                                                sub.get("rsf_bank", -1))
    print(f"{c:14s} {t:7s} stack={st:.3f} mph2={sub.get('mph2', np.nan):.3f} bank={bk:.3f} | bestOurs={o.idxmax()}:{o.max():.3f} bestBase={b.idxmax()}:{b.max():.3f} delta={o.max()-b.max():+.3f}")

print("\n=== same-learner comparison (ours = marker-field persistence features) ===")
for (c, t), sub in g.groupby(["cohort", "task"]):
    sub = sub.set_index("method")["mean"]; line = []
    for L in ("gbscv", "cnetcv", "lgbmsel", "lrsel", "coxsel", "rsfsel", "fuselgbm", "fuselr", "fusersf", "fusecox", "lgbmcv", "lrcv", "coxcv", "rsfcv", "lgbm", "lr", "cox", "rsf"):
        o = [sub[m] for m in sub.index if m.startswith(L + "_") and OURS(m)]
        b = [sub[m] for m in sub.index if m.startswith(L + "_") and not OURS(m)]
        if o and b:
            line.append(f"{L}:{max(o):.3f}/{max(b):.3f}({max(o)-max(b):+.3f})")
    nb = [sub[m] for m in ("gin", "gat", "paph", "mph") if m in sub]
    if "mph2" in sub and nb:
        line.append(f"neural mph2:{sub['mph2']:.3f}/{max(nb):.3f}")
    print(f"{c:14s} {t:7s} " + " ".join(line))

print("\n=== SINGLE-METHOD: best ours (no fusion) vs best prior (no fusion) ===")
for (c, t), sub in g.groupby(["cohort", "task"]):
    sub = sub[~sub.method.str.startswith("fuse")].set_index("method")["mean"]
    o = sub[[m for m in sub.index if OURS(m)]]; b = sub[[m for m in sub.index if not OURS(m)]]
    oo = {k: sub.get(k, np.nan) for k in ("lgbmcv_ours", "lrcv_ours", "rsfcv_ours", "coxcv_ours", "topo3")}
    print(f"{c:14s} {t:7s} ours={o.idxmax()}:{o.max():.3f} prior={b.idxmax()}:{b.max():.3f} d={o.max()-b.max():+.3f} | " +
          " ".join(f"{k}={v:.3f}" for k, v in oo.items() if not np.isnan(v)))
