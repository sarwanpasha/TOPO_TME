"""Tuned GNN: for every (task, fold, seed) choose, among GIN and the GIN-with-composition grid, the configuration with the
best inner-validation metric, and store its test result as method 'gin_tuned' (selection never sees the test fold).
python tuned_select.py <cohort> [<cohort> ...]"""
import os, sys, glob, json, collections
import common as C

CANDS = ("gin", "ginc", "ginc_l2", "ginc_h128", "ginc_d5")
for c in sys.argv[1:]:
    d = os.path.join(C.RES, c)
    for task in sorted(os.listdir(d)):
        runs = collections.defaultdict(dict)
        for m in CANDS:
            for f in glob.glob(os.path.join(d, task, m, "f*_s*.json")):
                r = json.load(open(f)); runs[(r["fold"], r["seed"])][m] = r
        chosen = collections.Counter(); n = 0
        for key, rs in runs.items():
            if not all(m in rs for m in CANDS[1:]):
                continue
            best = max(rs.values(), key=lambda r: (r.get("val_metric") if r.get("val_metric") is not None else -1))
            out = dict(best); out["method"] = "gin_tuned"; out["chosen"] = best["method"]
            p = C.result_path(c, task, "gin_tuned", key[0], key[1], "")
            os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(out, open(p, "w"))
            chosen[best["method"]] += 1; n += 1
        if n:
            print(c, task, "runs", n, dict(chosen), flush=True)
