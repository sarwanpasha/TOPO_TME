"""Topology-augmented late fusion with inner-CV mixing weight.
pred = (1-a)*rank(base model on comp+mm) + a*rank(model on partner features), a chosen by inner CV
on training patients only. Partners: prior spatial features (contact, topo_fixed, all) vs ours (mfield, mfield2, bank2),
so every topology representation gets the identical fusion protocol.
python fusion.py --cohort danenberg [--external]
"""
import os, json, argparse, traceback
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import multiprocessing as mp
from scipy.stats import rankdata
import common as C
import baselines as BL

PARTNERS = ["contact", "topo_fixed", "all", "mfield", "mfield2", "bank2"]
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8]
G = {}


def r01(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def score(kind, d, p):
    return C.auroc(d["y"].values, p) if kind == "bin" else C.cindex(d["time"].values, d["event"].values, p)


def yof(kind, d):
    return d["y"].values if kind == "bin" else (d["time"].values, d["event"].values)


def run_one(cfg):
    task, partner, learner, fold, seed, external = cfg
    method = f"fuse{learner}_{partner}"
    out = C.result_path(G["cohort"], task, method, fold, seed, "_ext" if external else "")
    if os.path.exists(out):
        return out, "skip"
    try:
        df, kind = C.task_frame(G["pat"], G["cohort"], task)
        PB, PP = G["pf"]["comp_mm"], G["pf"][partner]
        df = df[df["patient_id"].isin(PB.index)].reset_index(drop=True)
        if external:
            edf, _ = C.task_frame(G["epat"], G["ecohort"], task)
            edf = edf[edf["patient_id"].isin(G["epf"]["comp_mm"].index)].reset_index(drop=True)
            trd, ted = df, edf
            EB, EP = G["epf"]["comp_mm"], G["epf"][partner]
        else:
            tr, te = C.make_folds(df, kind)[fold]
            trd, ted = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
            EB, EP = PB, PP
        Xb, Xp = PB.loc[trd["patient_id"]].values, PP.loc[trd["patient_id"]].values
        # inner CV OOF predictions
        from sklearn.model_selection import StratifiedKFold
        strat = trd["y"].values if kind == "bin" else trd["event"].values
        ob, op = np.zeros(len(trd)), np.zeros(len(trd))
        for a, b in StratifiedKFold(3, shuffle=True, random_state=seed + 7).split(Xb, strat):
            ya = yof(kind, trd.iloc[a])
            ob[b] = r01(BL.fit_predict(kind, learner, Xb[a], ya, Xb[b], seed))
            op[b] = r01(BL.fit_predict(kind, learner, Xp[a], ya, Xp[b], seed))
        alpha = max(ALPHAS, key=lambda al: score(kind, trd, (1 - al) * ob + al * op))
        y = yof(kind, trd)
        pb = r01(BL.fit_predict(kind, learner, Xb, y, EB.loc[ted["patient_id"]].values, seed))
        pp = r01(BL.fit_predict(kind, learner, Xp, y, EP.loc[ted["patient_id"]].values, seed))
        pr = (1 - alpha) * pb + alpha * pp
        res = dict(cohort=G["cohort"], task=task, method=method, fold=fold, seed=seed, kind=kind, external=external,
                   alpha=alpha, test_metric=score(kind, ted, pr), patients=ted["patient_id"].tolist(), pred=pr.tolist())
        if kind == "bin":
            res["y"] = ted["y"].tolist()
        else:
            res["time"] = ted["time"].tolist(); res["event"] = ted["event"].tolist()
        json.dump(res, open(out, "w"))
        return out, f"ok {res['test_metric']:.3f} a={alpha}"
    except Exception:
        return out, "FAIL\n" + traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort"); ap.add_argument("--external", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    _, pat, _ = C.load_cohort(a.cohort)
    G.update(cohort=a.cohort, pat=pat, pf=BL.patient_feats(a.cohort))
    if a.external:
        ec = C.EXTERNAL[a.cohort]; _, ep, _ = C.load_cohort(ec)
        G.update(ecohort=ec, epat=ep, epf=BL.patient_feats(ec))
    cfgs = []
    for task, (kind, _) in C.TASKS[a.cohort].items():
        if C.task_frame(pat, a.cohort, task)[0] is None or (a.external and task not in C.TASKS[C.EXTERNAL[a.cohort]]):
            continue
        for p in [p for p in PARTNERS if p in G["pf"]]:
            for L in (["lgbm", "lr"] if kind == "bin" else ["rsf", "cox"]):
                for f in ([0] if a.external else range(5)):
                    for s in range(3):
                        cfgs.append((task, p, L, f, s, a.external))
    print(a.cohort, "runs", len(cfgs), flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool:
        for out, msg in pool.imap_unordered(run_one, cfgs):
            print(out, msg, flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
