"""Same tests as paired.py (patient-level bootstrap within test folds, B=1000; Wilcoxon across tasks),
but numba metrics + one process per (setting, cohort, task). Writes paired_tests.txt / .csv."""
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import sys, zlib
import numpy as np, pandas as pd
import multiprocessing as mp
from numba import njit
from scipy.stats import wilcoxon
import common as C
import paired as Pd


@njit(cache=True)
def _auc(y, s):
    pos = 0.0; num = 0.0; n = len(y)
    npos = 0; nneg = 0
    for i in range(n):
        if y[i] == 1:
            npos += 1
        else:
            nneg += 1
    if npos == 0 or nneg == 0:
        return np.nan
    for i in range(n):
        if y[i] != 1:
            continue
        for j in range(n):
            if y[j] == 1:
                continue
            if s[i] > s[j]:
                num += 1.0
            elif s[i] == s[j]:
                num += 0.5
    return num / (npos * nneg)


@njit(cache=True)
def _cidx(t, e, r):
    num = 0.0; den = 0.0; n = len(t)
    for i in range(n):
        if e[i] == 0:
            continue
        for j in range(n):
            if t[j] > t[i]:
                den += 1.0
                if r[i] > r[j]:
                    num += 1.0
                elif r[i] == r[j]:
                    num += 0.5
    return num / den if den > 0 else np.nan


def met(kind, d, pred, idx):
    if kind == "bin":
        return _auc(np.asarray(d["y"], np.int64)[idx], pred[idx])
    return _cidx(np.asarray(d["time"], np.float64)[idx], np.asarray(d["event"], np.int64)[idx], pred[idx])


Pd.met = met


def job(args):
    name, c, t, ours, prior, kind = args
    rng = np.random.default_rng(zlib.crc32(f"{name}|{c}|{t}".encode()))
    r = Pd.compare(c, t, ours, prior, kind, rng)
    if r:
        r["setting"] = name
    return r


if __name__ == "__main__":
    cohorts = [c for c in sorted(os.listdir(C.RES)) if not c.startswith("synth") and not c.endswith("_smoke")]
    jobs = []
    for name, (ours, prior, kind) in Pd.PAIRS.items():
        for c in cohorts:
            base = c[:-4] if c.endswith("_ext") else c
            for t in sorted(os.listdir(os.path.join(C.RES, c))):
                if C.TASKS.get(base, {}).get(t, ("",))[0] == kind:
                    jobs.append((name, c, t, ours, prior, kind))
    met("bin", {"y": [0, 1]}, np.array([0.1, 0.2]), np.arange(2))
    met("surv", {"time": [1.0, 2.0], "event": [1, 0]}, np.array([0.1, 0.2]), np.arange(2))
    with mp.get_context("spawn").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as pool:
        res = [r for r in pool.map(job, jobs) if r]
    rows = []; lines = []
    for name, (ours, prior, kind) in Pd.PAIRS.items():
        sub = [r for r in res if r["setting"] == name]
        if not sub:
            continue
        rows += sub; d = np.array([r["diff"] for r in sub])
        w = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else float("nan")
        lines.append(f"\n=== {name}: {ours} vs {prior} | tasks={len(d)} mean diff={d.mean():+.4f} "
                     f"wins={int((d > 0).sum())}/{len(d)} sig.wins(p<.05)={sum(r['diff'] > 0 and r['p_boot'] < .05 for r in sub)} "
                     f"sig.losses={sum(r['diff'] < 0 and r['p_boot'] < .05 for r in sub)} Wilcoxon p={w:.4f}")
        for r in sub:
            lines.append(f"  {r['cohort']:10s} {r['task']:7s} ours={r['ours_mean']:.3f} prior={r['prior_mean']:.3f} "
                         f"diff={r['diff']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p_boot']:.3f} (runs {r['n_runs']})")
    txt = "\n".join(lines) + "\n"
    print(txt, flush=True)
    open(os.path.join(C.ROOT, "paired_tests.txt"), "w").write(txt)
    pd.DataFrame(rows).to_csv(os.path.join(C.ROOT, "paired_tests.csv"), index=False)
