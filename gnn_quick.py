"""Interim paired tests vs GNNs on cohorts where GNN runs exist (does not touch paired_tests.txt)."""
import os, sys
import numpy as np
import common as C
import paired, paired_fast
paired.met = paired_fast.met
from paired import compare
ours = "lgbmgate4_cnz"
rng = np.random.default_rng(0)
for prior in ("gin", "gat"):
    for c in ("codex", "luad", "basel", "danenberg"):
        d = os.path.join(C.RES, c)
        for t in sorted(os.listdir(d)):
            if C.TASKS.get(c, {}).get(t, ("",))[0] != "bin" or not os.path.isdir(os.path.join(d, t, prior)):
                continue
            try:
                r = compare(c, t, ours, prior, "bin", rng)
            except Exception as ex:
                print(prior, c, t, "ERR", ex); continue
            if r:
                print(f"{prior} {c:6s} {t:8s} ours={r['ours_mean']:.3f} gnn={r['prior_mean']:.3f} diff={r['diff']:+.3f} "
                      f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p_boot']:.3f} runs={r['n_runs']}", flush=True)
