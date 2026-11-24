"""Corrected resampled t-test (Nadeau and Bengio 2003) for paired comparisons over repeated k-fold CV.
For each task: d_j = metric(ours) - metric(prior) on the same (fold, seed); t = mean(d) / sqrt((1/J + n_te/n_tr) s^2), df J-1.
python nb_tests.py ours1:prior1 ...   (cohorts: development + external + cords)"""
import os, sys, glob, json
import numpy as np
from scipy.stats import t as tdist
import common as C

COH = ("danenberg", "basel", "basel_ext", "codex", "luad", "cords")


def runs(c, task, m):
    out = {}
    for f in glob.glob(os.path.join(C.RES, c, task, m, "f*_s*.json")):
        d = json.load(open(f)); out[(d["fold"], d["seed"])] = d["test_metric"]
    return out


for pair in sys.argv[1:]:
    ours, prior = pair.split(":")
    res = []
    for c in COH:
        base = c[:-4] if c.endswith("_ext") else c
        d = os.path.join(C.RES, c)
        if not os.path.isdir(d):
            continue
        for task in sorted(os.listdir(d)):
            if C.TASKS.get(base, {}).get(task, ("",))[0] != "bin":
                continue
            a, b = runs(c, task, ours), runs(c, task, prior)
            k = sorted(set(a) & set(b))
            if len(k) < 3:
                continue
            dd = np.array([a[x] - b[x] for x in k], float); dd = dd[np.isfinite(dd)]
            J = len(dd); kf = 5; ratio = 1.0 / (kf - 1)
            s2 = dd.var(ddof=1)
            if s2 <= 0:
                tt, p = 0.0, 1.0
            else:
                tt = dd.mean() / np.sqrt((1.0 / J + ratio) * s2); p = 2 * tdist.sf(abs(tt), J - 1)
            res.append((c, task, dd.mean(), p, J))
    if not res:
        print(pair, "no results"); continue
    dev = [r for r in res if r[0] != "cords"]; nsc = [r for r in res if r[0] == "cords"]
    for name, R in (("development+external", dev), ("NSCLC", nsc)):
        if not R:
            continue
        d = np.array([r[2] for r in R]); p = np.array([r[3] for r in R])
        print(f"=== {ours} vs {prior} [{name}] tasks={len(R)} mean={d.mean():+.4f} NB sigwins={((d>0)&(p<.05)).sum()} "
              f"NB siglosses={((d<0)&(p<.05)).sum()}", flush=True)
        for r in R:
            print(f"   {r[0]:9s} {r[1]:11s} diff={r[2]:+.3f} p_NB={r[3]:.4f} runs={r[4]}")
