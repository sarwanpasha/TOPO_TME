"""Paired tests for ablation pairs over all binary tasks (incl. external); per-task BH and Wilcoxon across tasks.
python ablate_tests.py ours1:prior1 ours2:prior2 ..."""
import os, sys
import numpy as np
from scipy.stats import wilcoxon
import common as C
import paired, paired_fast
paired.met = paired_fast.met
from paired import compare


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


rng = np.random.default_rng(0)
for pair in sys.argv[1:]:
    ours, prior = pair.split(":")
    rows = []
    for c in ("danenberg", "basel", "basel_ext", "codex", "luad"):
        base = c[:-4] if c.endswith("_ext") else c
        d = os.path.join(C.RES, c)
        for t in sorted(os.listdir(d)):
            if C.TASKS.get(base, {}).get(t, ("",))[0] != "bin":
                continue
            if not (os.path.isdir(os.path.join(d, t, ours)) and os.path.isdir(os.path.join(d, t, prior))):
                continue
            r = compare(c, t, ours, prior, "bin", rng)
            if r:
                rows.append(r)
    if not rows:
        print(pair, "no results"); continue
    dd = np.array([r["diff"] for r in rows]); p = np.array([r["p_boot"] for r in rows]); q = bh(p)
    w = wilcoxon(dd).pvalue if np.any(dd != 0) else float("nan")
    print(f"=== {ours} vs {prior}: tasks={len(dd)} mean={dd.mean():+.4f} wins={(dd>0).sum()} sigwins={((dd>0)&(p<.05)).sum()} "
          f"siglosses={((dd<0)&(p<.05)).sum()} BHwins={((dd>0)&(q<.05)).sum()} BHlosses={((dd<0)&(q<.05)).sum()} Wilcoxon={w:.4g}", flush=True)
    for r, qq in zip(rows, q):
        print(f"   {r['cohort']:9s} {r['task']:11s} ours={r['ours_mean']:.3f} prior={r['prior_mean']:.3f} diff={r['diff']:+.3f} p={r['p_boot']:.3f} q={qq:.3f}", flush=True)
