"""Cohort-level inference that does not treat tasks as independent.
Within a cohort, patients are resampled with replacement ONCE per bootstrap replicate and the same resample is used for
every task, fold and seed of that cohort, so correlations between tasks that share patients are preserved. The cohort
effect is the mean paired difference over its tasks; its standard error is the bootstrap SD. Cohort effects are then
combined by fixed-effect and DerSimonian-Laird random-effects meta-analysis (cohorts have disjoint test patients).
python cohort_inference.py ours:prior [ours:prior ...] [--B 500]"""
import os, sys
import numpy as np
from scipy.stats import norm
import common as C
import paired, paired_fast
met = paired_fast.met
COH = tuple(os.environ.get("COHORTS", "danenberg,basel,basel_ext,codex,luad").split(","))
B = int(sys.argv[sys.argv.index("--B") + 1]) if "--B" in sys.argv else 500


def cohort_effect(c, ours, prior, rng):
    base = c[:-4] if c.endswith("_ext") else c
    runs = []
    for t in sorted(os.listdir(os.path.join(C.RES, c))):
        if C.TASKS.get(base, {}).get(t, ("",))[0] != "bin":
            continue
        A, P = paired.load(c, t, ours), paired.load(c, t, prior)
        for k in sorted(set(A) & set(P)):
            a, p = A[k], P[k]
            runs.append((t, a, np.asarray(a["pred"], float), np.asarray(p["pred"], float), np.asarray(a["patients"]).astype(str)))
    if not runs:
        return None
    tasks = sorted({r[0] for r in runs})
    pats = np.unique(np.concatenate([r[4] for r in runs]))
    pos = {p: i for i, p in enumerate(pats)}
    ridx = [np.array([pos[p] for p in r[4]]) for r in runs]

    def stat(w):
        per = {t: [] for t in tasks}
        for (t, a, pa, pp, _), ri in zip(runs, ridx):
            idx = np.repeat(np.arange(len(ri)), w[ri])
            if len(idx) < 4:
                continue
            da, dp = met("bin", a, pa, idx), met("bin", a, pp, idx)
            if np.isfinite(da) and np.isfinite(dp):
                per[t].append(da - dp)
        return np.mean([np.mean(v) for v in per.values() if v])

    obs = stat(np.ones(len(pats), int))
    bs = np.array([stat(np.bincount(rng.integers(0, len(pats), len(pats)), minlength=len(pats))) for _ in range(B)])
    return dict(cohort=c, tasks=len(tasks), patients=len(pats), d=obs, se=bs.std(ddof=1), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5))


if __name__ == "__main__":
    met("bin", {"y": [0, 1]}, np.array([0.1, 0.2]), np.arange(2))
    for pair in [a for a in sys.argv[1:] if ":" in a]:
        ours, prior = pair.split(":"); rng = np.random.default_rng(0)
        R = [r for r in (cohort_effect(c, ours, prior, rng) for c in COH) if r]
        for r in R:
            if r["se"] == 0:
                print(f"   {r['cohort']:10s} identical predictions (no information; excluded)")
        R = [r for r in R if r["se"] > 0]
        d = np.array([r["d"] for r in R]); se = np.array([r["se"] for r in R]); w = 1 / se ** 2
        fe = (w * d).sum() / w.sum(); fse = w.sum() ** -0.5
        Q = (w * (d - fe) ** 2).sum(); k = len(d)
        tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        wr = 1 / (se ** 2 + tau2); re = (wr * d).sum() / wr.sum(); rse = wr.sum() ** -0.5
        print(f"=== {ours} vs {prior}", flush=True)
        for r in R:
            print(f"   {r['cohort']:10s} tasks={r['tasks']} n={r['patients']:4d} d={r['d']:+.4f} se={r['se']:.4f} CI[{r['lo']:+.4f},{r['hi']:+.4f}] z={r['d']/r['se']:.2f}")
        print(f"   fixed  : {fe:+.4f} (se {fse:.4f}) z={fe/fse:.2f} p={2*norm.sf(abs(fe/fse)):.3g}")
        print(f"   random : {re:+.4f} (se {rse:.4f}) z={re/rse:.2f} p={2*norm.sf(abs(re/rse)):.3g} tau2={tau2:.2e} Q={Q:.2f}", flush=True)

