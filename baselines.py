"""Non-neural baselines on hand-crafted per-patient features.

feature sets: comp, mm, comp_mm, contact (2nd-order type-pair contacts),
topo_fixed (phenotype-aware *fixed* PH: all cells + per-type Delaunay PH), all.
binary: LightGBM + L2-logistic ; survival: penalised Cox on PCA + random survival forest.
python baselines.py --cohort danenberg [--external]
"""
import os, json, time, argparse, traceback
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import pandas as pd
import multiprocessing as mp
import common as C

FSETS = {"comp": ["comp", "logn"], "mm": ["mm", "logn"], "comp_mm": ["comp", "mm", "logn"],
         "contact": ["comp", "logn", "contact"], "topo_fixed": ["comp", "logn", "topo_all", "topo_type"],
         "all": ["comp", "mm", "logn", "contact", "topo_all", "topo_type"],
         "mfield": ["comp", "mm", "logn", "mfield"],
         "bank": ["comp", "mm", "logn", "contact", "topo_all", "topo_type", "mfield"],
         "mfield2": ["comp", "mm", "logn", "mfield2"],
         "bank2": ["comp", "mm", "logn", "contact", "topo_type", "mfield2"],
         "compact": ["comp", "mm", "logn", "compact"],
         "cross": ["comp", "mm", "logn", "cross"],
         "ours": ["comp", "mm", "logn", "compact", "cross"],
         "c_compact": ["comp", "logn", "compact"], "c_cross": ["comp", "logn", "cross"],
         "c_mfield2": ["comp", "logn", "mfield2"], "c_ours": ["comp", "logn", "compact", "cross"],
         "emb": ["comp", "mm", "logn", "emb"], "c_emb": ["comp", "logn", "emb"],
         "c_emb_compact": ["comp", "logn", "emb", "compact"],
         "nbr": ["comp", "mm", "logn", "contact"], "c_nbr": ["comp", "logn", "contact"],
         "cn": ["comp", "mm", "logn", "cn"], "c_cn": ["comp", "logn", "cn"],
         "znull": ["comp", "mm", "logn", "znull"], "c_znull": ["comp", "logn", "znull"],
         "zall": ["comp", "mm", "logn", "znull", "compact", "cross"],
         "cnzT": ["comp", "mm", "logn", "cn", "zT"], "cnzX": ["comp", "mm", "logn", "cn", "zX"],
         "cnzM": ["comp", "mm", "logn", "cn", "zM"], "cnzTX": ["comp", "mm", "logn", "cn", "zT", "zX"],
         "cnz": ["comp", "mm", "logn", "cn", "znull"], "c_cnz": ["comp", "logn", "cn", "znull"],
         "c_ctype": ["comp", "logn", "ctype"], "c_ctype_cross": ["comp", "logn", "ctype", "cross"]}
G = {}


def patient_feats(cohort):
    fe = dict(np.load(os.path.join(C.cohort_dir(cohort), "feats.npz")))
    pid = fe["patient_id"].astype(str)
    mf = os.path.join(C.cohort_dir(cohort), "feats_mfield.npz")
    if os.path.exists(mf):
        z = np.load(mf); assert (z["image_id"] == fe["image_id"]).all(); fe["mfield"] = z["mfield"]
    mf2 = os.path.join(C.cohort_dir(cohort), "feats_mfield2.npz")
    if os.path.exists(mf2):
        z = np.load(mf2); assert (z["image_id"] == fe["image_id"]).all(); fe["mfield2"] = z["mfield2"]
    mfc = os.path.join(C.cohort_dir(cohort), "feats_compact.npz")
    if os.path.exists(mfc):
        z = np.load(mfc); assert (z["image_id"] == fe["image_id"]).all(); fe["compact"] = z["compact"]
        nM = fe["mm"].shape[1]
        fe["ctype"] = z["compact"][:, nM * 30:]          # per-cell-type indicator-field block of compact
    mfe = os.path.join(C.cohort_dir(cohort), "feats_emb.npz")
    if os.path.exists(mfe):
        z = np.load(mfe); assert (z["image_id"] == fe["image_id"]).all(); fe["emb"] = z["emb"]
    mfx = os.path.join(C.cohort_dir(cohort), "feats_cross.npz")
    if os.path.exists(mfx):
        z = np.load(mfx); assert (z["image_id"] == fe["image_id"]).all(); fe["cross"] = z["cross"]
    for _k in ("cn", "znull"):
        _f = os.path.join(C.cohort_dir(cohort), f"feats_{_k}.npz")
        if os.path.exists(_f):
            z = np.load(_f); assert (z["image_id"] == fe["image_id"]).all(); fe[_k] = z[_k]
    if "znull" in fe:
        _nt = fe["comp"].reshape(len(pid), -1).shape[1]; _nm = fe["mm"].reshape(len(pid), -1).shape[1]
        _a = _nt * 15; _b = _a + _nt * (_nt + 1) // 2 * 6; _Z = fe["znull"].reshape(len(pid), -1)
        if _b + _nm * 30 == _Z.shape[1]:
            fe["zT"], fe["zX"], fe["zM"] = _Z[:, :_a], _Z[:, _a:_b], _Z[:, _b:]
    out = {}
    for name, keys in FSETS.items():
        if any(k not in fe for k in keys):
            continue
        X = np.concatenate([fe[k].reshape(len(pid), -1) for k in keys], 1)
        NBASE[name] = sum(fe[k].reshape(len(pid), -1).shape[1] for k in keys if k in ("comp", "mm", "logn"))
        if name in ("cnz", "c_cnz") or name.startswith("cnz"):
            NBASE0[name] = NBASE[name]; NBASE[name] += fe["cn"].reshape(len(pid), -1).shape[1]
        df = pd.DataFrame(X); df["pid"] = pid
        out[name] = df.groupby("pid").mean()
    return out


NBASE = {}
NBASE0 = {}
CUR = {"nbase": 0}


def _screen(kind, X, y):
    """univariate screening score per column (training data only)."""
    if kind == "bin":
        from scipy.stats import rankdata
        y = np.asarray(y).astype(int); n1 = int(y.sum()); n0 = len(y) - n1
        if n1 == 0 or n0 == 0:
            return np.zeros(X.shape[1])
        R = rankdata(X, axis=0)
        auc = (R[y == 1].sum(0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
        return np.nan_to_num(np.abs(auc - 0.5) * (X.std(0) >= 1e-9))
    sc = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        if X[:, j].std() < 1e-9:
            continue
        sc[j] = abs((C.auroc(y, X[:, j]) if kind == "bin" else C.cindex(y[0], y[1], X[:, j])) - 0.5)
    return np.nan_to_num(sc)


def _inner_cv_pick(kind, model, Xtr, ytr, seed, grid):
    from sklearn.model_selection import StratifiedKFold
    strat = ytr if kind == "bin" else ytr[1]
    skf = StratifiedKFold(3, shuffle=True, random_state=seed + 100)
    best, bs = None, -1
    for g in grid:
        sc = []
        for a, bb in skf.split(Xtr, strat):
            ya = ytr[a] if kind == "bin" else (ytr[0][a], ytr[1][a])
            pr = fit_predict(kind, model, Xtr[a], ya, Xtr[bb], seed, g)
            if kind == "bin":
                sc.append(C.auroc(ytr[bb], pr))
            else:
                sc.append(C.cindex(ytr[0][bb], ytr[1][bb], pr))
        m = np.nanmean(sc)
        if m > bs:
            bs, best = m, g
    return best


GRIDS = {"lgbm": [dict(nl=nl, ne=ne, cs=cs) for nl in (4, 15) for ne in (150, 400) for cs in (0.1, 0.5)],
         "lr": [dict(C=c) for c in (0.003, 0.03, 0.3, 3.0)],
         "cox": [dict(k=k, pen=p) for k in (5, 20) for p in (0.05, 1.0)],
         "rsf": [dict(leaf=l, mf=m) for l in (5, 20) for m in ("sqrt", 0.1)],
         "rsf2": [dict(ne=500, leaf=l, mf=m) for l in (3, 10, 30) for m in ("sqrt", 0.05, 0.3)],
         "est": [dict(ne=500, leaf=l, mf=m) for l in (5, 15, 40) for m in ("sqrt", 0.1)],
         "gbs": [dict(ne=n, lr=0.05, d=d) for n in (100, 300) for d in (1, 2)],
         "cnet": [dict(l1=l, ai=a) for l in (0.2, 0.8) for a in (8, 16, 24)]}



def _ridgecox(Z, t, e, off, lam):
    """ridge-penalised Cox (Breslow ties) with offset, Newton; numpy only."""
    order = np.argsort(t, kind="stable"); Zs, es, os_ = Z[order], e[order], off[order]
    b = np.zeros(Z.shape[1])
    for _ in range(40):
        w = np.exp(np.clip(Zs @ b + os_, -20, 20))
        S0 = np.cumsum(w[::-1])[::-1]; S1 = np.cumsum((w[:, None] * Zs)[::-1], 0)[::-1]; mu = S1 / S0[:, None]
        g = (es[:, None] * (Zs - mu)).sum(0) - lam * b
        S2 = np.cumsum((w[:, None, None] * Zs[:, :, None] * Zs[:, None, :])[::-1], 0)[::-1]
        Hm = (es[:, None, None] * (S2 / S0[:, None, None] - mu[:, :, None] * mu[:, None, :])).sum(0) + lam * np.eye(len(b))
        st = np.linalg.solve(Hm, g); b = b + st
        if not np.all(np.isfinite(b)):
            return np.zeros(Z.shape[1])
        if np.abs(st).max() < 1e-6:
            break
    return b


def _martingale(lp, t, e):
    order = np.argsort(t, kind="stable"); w = np.exp(np.clip(lp, -20, 20))
    rs = np.cumsum(w[order][::-1])[::-1]
    h = np.where(e[order] == 1, 1.0 / np.maximum(rs, 1e-12), 0.0)
    H = np.empty(len(t)); H[order] = np.cumsum(h)
    return e - H * w


def _res_parts(Xa, ya, Xb, seed, nb, lin):
    from sklearn.preprocessing import StandardScaler
    t, e = np.asarray(ya[0], float), np.asarray(ya[1], int)
    pb = fit_predict("surv", "est", Xa[:, :nb], ya, Xb[:, :nb], seed)
    Ta, Tb = Xa[:, nb:], Xb[:, nb:]
    if Ta.shape[1] == 0:
        return pb, np.zeros(len(Xb))
    s1 = StandardScaler().fit(Xa[:, :nb]); Za = np.clip(np.nan_to_num(s1.transform(Xa[:, :nb])), -5, 5)
    keep = Za.std(0) > 1e-6; Za = Za[:, keep]
    off = Za @ _ridgecox(Za, t, e, np.zeros(len(t)), 0.05 * len(t))
    if lin:
        from sklearn.decomposition import PCA
        s2 = StandardScaler().fit(Ta); A = np.clip(np.nan_to_num(s2.transform(Ta)), -5, 5)
        Bm = np.clip(np.nan_to_num(s2.transform(Tb)), -5, 5)
        p = PCA(min(20, A.shape[0] - 2, A.shape[1]), random_state=seed).fit(A)
        PA, PB = p.transform(A), p.transform(Bm); sd = PA.std(0) + 1e-9
        bt = _ridgecox(PA / sd, t, e, off, 0.05 * len(t))
        pt = (PB / sd) @ bt
    else:
        from sklearn.ensemble import ExtraTreesRegressor
        M = _martingale(off, t, e)
        r = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.3, random_state=seed, n_jobs=1)
        pt = r.fit(np.nan_to_num(Ta), M).predict(np.nan_to_num(Tb))
    return pb, pt


def _rk(x):
    from scipy.stats import rankdata
    return (rankdata(x) - 0.5) / len(x)


ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)


GATE = {"lgbmgate": ("lgbm", "lgbmsel", True), "rsf2gate": ("rsf2", "estrisk", True), "estgate": ("est", "estrisk", False)}
GALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def _grk(x):
    from scipy.stats import rankdata
    return (rankdata(x) - 0.5) / len(x)


def _gate(kind, model, Xtr, ytr, Xte, seed):
    from sklearn.model_selection import StratifiedKFold
    base_m, topo_m, tune = GATE[model]; nb = CUR["nbase"]
    Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
    sub = (lambda idx: ytr[idx]) if kind == "bin" else (lambda idx: (ytr[0][idx], ytr[1][idx]))
    score = (lambda idx, p: C.auroc(ytr[idx], p)) if kind == "bin" else (lambda idx, p: C.cindex(ytr[0][idx], ytr[1][idx], p))
    bhp = _inner_cv_pick(kind, base_m, Xtr[:, :nb], ytr, seed, GRIDS[base_m]) if tune else {}
    sc = {al: [] for al in GALPHAS}
    strat = ytr if kind == "bin" else ytr[1]
    for a_, b_ in StratifiedKFold(3, shuffle=True, random_state=seed + 600).split(Xtr, strat):
        pb = fit_predict(kind, base_m, Xtr[a_][:, :nb], sub(a_), Xtr[b_][:, :nb], seed, bhp)
        pt = fit_predict(kind, topo_m, Xtr[a_], sub(a_), Xtr[b_], seed)
        for al in GALPHAS:
            sc[al].append(score(b_, _grk(pb) + al * _grk(pt)))
    al = max(GALPHAS, key=lambda v: (np.nanmean(sc[v]), -v))
    pb = fit_predict(kind, base_m, Xtr[:, :nb], ytr, Xte[:, :nb], seed, bhp)
    if al == 0:
        return _grk(pb)
    pt = fit_predict(kind, topo_m, Xtr, ytr, Xte, seed)
    return _grk(pb) + al * _grk(pt)


def _gate2(kind, Xtr, ytr, Xte, seed):
    from sklearn.model_selection import StratifiedKFold
    nb = CUR["nbase"]
    Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
    sub = (lambda idx: ytr[idx]) if kind == "bin" else (lambda idx: (ytr[0][idx], ytr[1][idx]))
    score = (lambda idx, p: C.auroc(ytr[idx], p)) if kind == "bin" else (lambda idx, p: C.cindex(ytr[0][idx], ytr[1][idx], p))
    cands = {"default": {}, "tuned": _inner_cv_pick(kind, "lgbm", Xtr[:, :nb], ytr, seed, GRIDS["lgbm"])}
    sc = {(c, al): [] for c in cands for al in GALPHAS}
    strat = ytr if kind == "bin" else ytr[1]
    for a_, b_ in StratifiedKFold(3, shuffle=True, random_state=seed + 600).split(Xtr, strat):
        pt = fit_predict(kind, "lgbmsel", Xtr[a_], sub(a_), Xtr[b_], seed)
        for c, hp in cands.items():
            pb = fit_predict(kind, "lgbm", Xtr[a_][:, :nb], sub(a_), Xtr[b_][:, :nb], seed, hp)
            for al in GALPHAS:
                sc[(c, al)].append(score(b_, _grk(pb) + al * _grk(pt)))
    c, al = max(sc, key=lambda k: (np.nanmean(sc[k]), -k[1], k[0] == "default"))
    pb = fit_predict(kind, "lgbm", Xtr[:, :nb], ytr, Xte[:, :nb], seed, cands[c])
    if al == 0:
        return _grk(pb)
    pt = fit_predict(kind, "lgbmsel", Xtr, ytr, Xte, seed)
    return _grk(pb) + al * _grk(pt)


def _sgate2(kind, Xtr, ytr, Xte, seed):
    from sklearn.model_selection import StratifiedKFold
    nb = CUR["nbase"]
    Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
    sub = lambda idx: (ytr[0][idx], ytr[1][idx])
    score = lambda idx, p: C.cindex(ytr[0][idx], ytr[1][idx], p)
    cands = {m: _inner_cv_pick(kind, m, Xtr[:, :nb], ytr, seed, GRIDS[m]) for m in ("rsf2", "rsf", "est")}
    sc = {(c, al): [] for c in cands for al in GALPHAS}
    for a_, b_ in StratifiedKFold(3, shuffle=True, random_state=seed + 600).split(Xtr, ytr[1]):
        pt = fit_predict(kind, "estrisk", Xtr[a_], sub(a_), Xtr[b_], seed)
        for c, hp in cands.items():
            pb = fit_predict(kind, c, Xtr[a_][:, :nb], sub(a_), Xtr[b_][:, :nb], seed, hp)
            for al in GALPHAS:
                sc[(c, al)].append(score(b_, _grk(pb) + al * _grk(pt)))
    c, al = max(sc, key=lambda k: (np.nanmean(sc[k]), -k[1]))
    pb = fit_predict(kind, c, Xtr[:, :nb], ytr, Xte[:, :nb], seed, cands[c])
    if al == 0:
        return _grk(pb)
    pt = fit_predict(kind, "estrisk", Xtr, ytr, Xte, seed)
    return _grk(pb) + al * _grk(pt)


def _gate4(kind, Xtr, ytr, Xte, seed):
    from sklearn.model_selection import StratifiedKFold
    nb, nb0 = CUR["nbase"], CUR.get("nbase0", CUR["nbase"])
    Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
    sub = (lambda idx: ytr[idx]) if kind == "bin" else (lambda idx: (ytr[0][idx], ytr[1][idx]))
    score = (lambda idx, p: C.auroc(ytr[idx], p)) if kind == "bin" else (lambda idx, p: C.cindex(ytr[0][idx], ytr[1][idx], p))
    if kind == "bin":
        cands = {}
        for tag, k in (("cm", nb0), ("cmcn", nb)):
            cands[(tag, "default")] = ("lgbm", k, {})
            cands[(tag, "tuned")] = ("lgbm", k, _inner_cv_pick(kind, "lgbm", Xtr[:, :k], ytr, seed, GRIDS["lgbm"]))
        topo = "lgbmsel"
    else:
        cands = {}
        for tag, k in (("c", nb0), ("ccn", nb)):
            cands[(tag, "rsf2")] = ("rsf2", k, _inner_cv_pick(kind, "rsf2", Xtr[:, :k], ytr, seed, GRIDS["rsf2"]))
        topo = "estrisk"
    sc = {(c, al): [] for c in cands for al in GALPHAS}
    strat = ytr if kind == "bin" else ytr[1]
    for a_, b_ in StratifiedKFold(3, shuffle=True, random_state=seed + 600).split(Xtr, strat):
        CUR["nbase"] = nb
        pt = fit_predict(kind, topo, Xtr[a_], sub(a_), Xtr[b_], seed)
        for c, (m, k, hp) in cands.items():
            pb = fit_predict(kind, m, Xtr[a_][:, :k], sub(a_), Xtr[b_][:, :k], seed, hp)
            for al in GALPHAS:
                sc[(c, al)].append(score(b_, _grk(pb) + al * _grk(pt)))
    c, al = max(sc, key=lambda q: (np.nanmean(sc[q]), -q[1]))
    if CUR.get("log"):
        rec = dict(CUR.get("tag", {}), kind=kind, base="/".join(c), alpha=al,
                   scores={"/".join(q[0]) + "|" + str(q[1]): float(np.nanmean(v)) for q, v in sc.items()})
        with open(os.path.join(C.RES, "gatelog", f"{os.getpid()}.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    m, k, hp = cands[c]
    pb = fit_predict(kind, m, Xtr[:, :k], ytr, Xte[:, :k], seed, hp)
    if al == 0:
        return _grk(pb)
    CUR["nbase"] = nb
    pt = fit_predict(kind, topo, Xtr, ytr, Xte, seed)
    return _grk(pb) + al * _grk(pt)

def fit_predict(kind, model, Xtr, ytr, Xte, seed, hp=None):
    if model in ("lgbmgate4", "rsfgate4", "lgbmgate4L", "rsfgate4L"):
        CUR["log"] = model.endswith("L")
        return _gate4(kind, Xtr, ytr, Xte, seed)
    if model == "rsfgate2":
        return _sgate2(kind, Xtr, ytr, Xte, seed)
    if model == "lgbmgate2":
        return _gate2(kind, Xtr, ytr, Xte, seed)
    if model in GATE:
        return _gate(kind, model, Xtr, ytr, Xte, seed)
    if model in ("estres", "estresL"):
        lin = model == "estresL"; nb = CUR["nbase"]
        Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
        if hp is None:
            from sklearn.model_selection import StratifiedKFold
            sc = {al: [] for al in ALPHAS}
            for a_, b_ in StratifiedKFold(3, shuffle=True, random_state=seed + 500).split(Xtr, ytr[1]):
                pb, pt = _res_parts(Xtr[a_], (ytr[0][a_], ytr[1][a_]), Xtr[b_], seed, nb, lin)
                for al in ALPHAS:
                    sc[al].append(C.cindex(ytr[0][b_], ytr[1][b_], _rk(pb) + al * _rk(pt)))
            hp = dict(alpha=max(ALPHAS, key=lambda al: (np.nanmean(sc[al]), -al)))
        pb, pt = _res_parts(Xtr, ytr, Xte, seed, nb, lin)
        return _rk(pb) + hp["alpha"] * _rk(pt)
    if model.endswith("risk") or model.endswith("riskE"):
        # topology risk score: a model on the topological columns is turned into ONE out-of-fold score
        # (inner CV on the training fold), appended to the base columns (clinical + composition)
        from sklearn.model_selection import StratifiedKFold
        from scipy.stats import rankdata
        useE = model.endswith("riskE")
        base = model[:-5] if useE else model[:-4]; nb = CUR["nbase"]
        Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
        if Xtr.shape[1] <= nb:
            return fit_predict(kind, base, Xtr, ytr, Xte, seed, hp)
        sub_m = "lgbm" if kind == "bin" else ("est" if useE else "rsf")
        strat = ytr if kind == "bin" else ytr[1]
        oof = np.zeros(len(Xtr)); pte = np.zeros(len(Xte)); nf = 0
        for a, bb in StratifiedKFold(5, shuffle=True, random_state=seed + 400).split(Xtr, strat):
            ya = ytr[a] if kind == "bin" else (ytr[0][a], ytr[1][a])
            p = fit_predict(kind, sub_m, Xtr[a][:, nb:], ya, np.vstack([Xtr[bb][:, nb:], Xte[:, nb:]]), seed)
            pv, pt = p[:len(bb)], p[len(bb):]
            oof[bb] = (rankdata(pv) - 0.5) / len(pv)
            pte += np.searchsorted(np.sort(pv), pt) / len(pv); nf += 1
        pte /= nf
        Ztr = np.hstack([Xtr[:, :nb], oof[:, None]]); Zte = np.hstack([Xte[:, :nb], pte[:, None]])
        return fit_predict(kind, base, Ztr, ytr, Zte, seed, hp)
    if model.endswith("pca") and model not in ("pca",):
        # base columns kept; topological columns standardised and compressed to k PCs (k by inner CV)
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        base = model[:-3]; nb = CUR["nbase"]
        Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
        def comp(Xa, Xb, k):
            if Xa.shape[1] <= nb:
                return Xa, Xb
            sc = StandardScaler().fit(Xa[:, nb:]); p = PCA(min(k, Xa.shape[0] - 2, Xa.shape[1] - nb), random_state=seed)
            Za = p.fit_transform(np.clip(sc.transform(Xa[:, nb:]), -5, 5)); Zb = p.transform(np.clip(sc.transform(Xb[:, nb:]), -5, 5))
            return np.hstack([Xa[:, :nb], Za]), np.hstack([Xb[:, :nb], Zb])
        if hp is None:
            from sklearn.model_selection import StratifiedKFold
            strat = ytr if kind == "bin" else ytr[1]; best, bs = 5, -1
            for k in (3, 8, 20):
                scs = []
                for a, bb in StratifiedKFold(3, shuffle=True, random_state=seed + 300).split(Xtr, strat):
                    ya = ytr[a] if kind == "bin" else (ytr[0][a], ytr[1][a])
                    Za, Zb = comp(Xtr[a], Xtr[bb], k)
                    pr = fit_predict(kind, base, Za, ya, Zb, seed)
                    scs.append(C.auroc(ytr[bb], pr) if kind == "bin" else C.cindex(ytr[0][bb], ytr[1][bb], pr))
                if np.nanmean(scs) > bs:
                    bs, best = np.nanmean(scs), k
            hp = dict(k=best)
        Za, Zb = comp(Xtr, Xte, hp["k"])
        return fit_predict(kind, base, Za, ytr, Zb, seed)
    if model.endswith("sel") or model.endswith("selS"):
        # topology-screened model: keep all base columns (comp, mm, logN) + top-K topological columns,
        # columns ranked by univariate association on the training fold; K chosen by inner CV
        small = model.endswith("selS")
        base = model[:-4] if small else model[:-3]; nb = CUR["nbase"]
        Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
        order = nb + np.argsort(-_screen(kind, Xtr[:, nb:], ytr))
        def sub(K):
            idx = np.concatenate([np.arange(nb), order[:K]]).astype(int)
            return idx
        if hp is None:
            grid = [dict(K=k) for k in ((2, 5, 10) if small else (10, 30, 100)) if k <= Xtr.shape[1] - nb]
            best, bs = None, -1
            from sklearn.model_selection import StratifiedKFold
            strat = ytr if kind == "bin" else ytr[1]
            for g in grid:
                scs = []
                for a, bb in StratifiedKFold(3, shuffle=True, random_state=seed + 200).split(Xtr, strat):
                    ya = ytr[a] if kind == "bin" else (ytr[0][a], ytr[1][a])
                    oa = nb + np.argsort(-_screen(kind, Xtr[a][:, nb:], ya))
                    idx = np.concatenate([np.arange(nb), oa[:g["K"]]]).astype(int)
                    pr = fit_predict(kind, base, Xtr[a][:, idx], ya, Xtr[bb][:, idx], seed)
                    scs.append(C.auroc(ytr[bb], pr) if kind == "bin" else C.cindex(ytr[0][bb], ytr[1][bb], pr))
                if np.nanmean(scs) > bs:
                    bs, best = np.nanmean(scs), g
            hp = best or dict(K=10)
        idx = sub(hp["K"])
        return fit_predict(kind, base, Xtr[:, idx], ytr, Xte[:, idx], seed)
    if model.endswith("cv"):
        base = model[:-2]
        hp = _inner_cv_pick(kind, base, np.nan_to_num(Xtr), ytr, seed, GRIDS[base])
        return fit_predict(kind, base, Xtr, ytr, Xte, seed, hp)
    hp = hp or {}
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    Xtr = np.nan_to_num(Xtr); Xte = np.nan_to_num(Xte)
    if kind == "bin":
        if model == "lgbm":
            import lightgbm as lgb
            m = lgb.LGBMClassifier(n_estimators=hp.get("ne", 300), learning_rate=0.03, num_leaves=hp.get("nl", 15),
                                   min_child_samples=10, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=hp.get("cs", 0.5), reg_lambda=1.0,
                                   random_state=seed, verbose=-1, n_jobs=1)
            m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1]
        from sklearn.linear_model import LogisticRegression
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=hp.get("C", 0.1), max_iter=3000, class_weight="balanced")
        m.fit(sc.transform(Xtr), ytr); return m.decision_function(sc.transform(Xte))
    t, e = ytr
    if model == "cox":
        from lifelines import CoxPHFitter
        sc = StandardScaler().fit(Xtr)
        k = min(hp.get("k", 10), Xtr.shape[1], len(Xtr) - 2)
        pca = PCA(k, random_state=seed).fit(sc.transform(Xtr))
        Ztr = pca.transform(sc.transform(Xtr)); Zte = pca.transform(sc.transform(Xte))
        d = pd.DataFrame(Ztr, columns=[f"z{i}" for i in range(k)]); d["T"] = t; d["E"] = e
        m = CoxPHFitter(penalizer=hp.get("pen", 0.1)).fit(d, "T", "E")
        return m.predict_partial_hazard(pd.DataFrame(Zte, columns=[f"z{i}" for i in range(k)])).values.ravel()
    if model == "est":
        from sksurv.ensemble import ExtraSurvivalTrees
        from sksurv.util import Surv
        m = ExtraSurvivalTrees(n_estimators=hp.get("ne", 500), min_samples_leaf=hp.get("leaf", 10),
                               max_features=hp.get("mf", "sqrt"), random_state=seed, n_jobs=1)
        m.fit(np.nan_to_num(Xtr), Surv.from_arrays(e.astype(bool), t)); return m.predict(np.nan_to_num(Xte))
    if model == "gbs":
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        from sksurv.util import Surv
        m = GradientBoostingSurvivalAnalysis(n_estimators=hp.get("ne", 200), learning_rate=hp.get("lr", 0.05),
                                             max_depth=hp.get("d", 2), subsample=0.8, max_features="sqrt",
                                             random_state=seed)
        m.fit(np.nan_to_num(Xtr), Surv.from_arrays(e.astype(bool), t)); return m.predict(np.nan_to_num(Xte))
    if model == "cnet":
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
        sc = StandardScaler().fit(np.nan_to_num(Xtr))
        m = CoxnetSurvivalAnalysis(l1_ratio=hp.get("l1", 0.5), alpha_min_ratio=0.05, n_alphas=30, max_iter=20000)
        Ztr = np.clip(sc.transform(np.nan_to_num(Xtr)), -5, 5); Zte = np.clip(sc.transform(np.nan_to_num(Xte)), -5, 5)
        m.fit(Ztr, Surv.from_arrays(e.astype(bool), t))
        k = hp.get("ai", len(m.alphas_) // 2)
        return Zte @ m.coef_[:, min(k, m.coef_.shape[1] - 1)]
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    m = RandomSurvivalForest(n_estimators=hp.get("ne", 200), min_samples_leaf=hp.get("leaf", 10), max_features=hp.get("mf", "sqrt"),
                             random_state=seed, n_jobs=1)
    m.fit(Xtr, Surv.from_arrays(e.astype(bool), t)); return m.predict(Xte)


def run_one(cfg):
    task, fset, model, fold, seed, external = cfg
    method = f"{model}_{fset}"
    out = C.result_path(G["cohort"], task, method, fold, seed, "_ext" if external else "")
    if os.path.exists(out):
        return out, "skip"
    try:
        df, kind = C.task_frame(G["pat"], G["cohort"], task)
        PF = G["pf"][fset]
        CUR["nbase"] = NBASE.get(fset, 0); CUR["nbase0"] = NBASE0.get(fset, CUR["nbase"])
        CUR["tag"] = dict(cohort=G["cohort"], task=task, fold=fold, seed=seed, external=external)
        df = df[df["patient_id"].isin(PF.index)].reset_index(drop=True)
        if external:
            edf, _ = C.task_frame(G["epat"], G["ecohort"], task)
            EPF = G["epf"][fset]
            edf = edf[edf["patient_id"].isin(EPF.index)].reset_index(drop=True)
            trd, ted, Xte = df, edf, EPF.loc[edf["patient_id"]].values
        else:
            tr, te = C.make_folds(df, kind)[fold]
            trd, ted = df.iloc[tr], df.iloc[te].reset_index(drop=True)
            Xte = PF.loc[ted["patient_id"]].values
        Xtr = PF.loc[trd["patient_id"]].values
        y = trd["y"].values if kind == "bin" else (trd["time"].values, trd["event"].values)
        pr = fit_predict(kind, model, Xtr, y, Xte, seed)
        res = dict(cohort=G["cohort"], task=task, method=method, fold=fold, seed=seed, kind=kind,
                   external=external, test_metric=C.metric(kind, ted, pr),
                   patients=ted["patient_id"].tolist(), pred=np.asarray(pr, float).tolist())
        if kind == "bin":
            res["y"] = ted["y"].tolist()
        else:
            res["time"] = ted["time"].tolist(); res["event"] = ted["event"].tolist()
        json.dump(res, open(out, "w"))
        return out, f"ok {res['test_metric']:.3f}"
    except Exception:
        return out, "FAIL\n" + traceback.format_exc()


def add_clinical(cohort, pat, pf):
    """'k_<fset>' = clinical covariates (age, grade, size, nodes, ER, HER2; median-imputed) + <fset>."""
    meta = json.load(open(os.path.join(C.cohort_dir(cohort), "meta.json")))
    cov = [c for c in meta.get("covariates", []) if c in pat.columns]
    if not cov:
        return
    CL = pat.set_index("patient_id")[cov].apply(pd.to_numeric, errors="coerce")
    CL = CL.fillna(CL.median())
    for name in list(pf):
        if name.startswith("k_"):
            continue
        base = pf[name]
        J = CL.reindex(base.index)
        J = J.fillna(CL.median())
        pf["k_" + name] = pd.concat([J.set_axis([f"clin_{c}" for c in cov], axis=1), base], axis=1)
        NBASE["k_" + name] = len(cov) + NBASE.get(name, 0)


def _init(cohort, external):
    """build globals (also used as pool initializer so 'spawn' platforms such as Windows work)."""
    if G.get("cohort") == cohort:
        return
    _, pat, _ = C.load_cohort(cohort)
    G.update(cohort=cohort, pat=pat, pf=patient_feats(cohort))
    add_clinical(cohort, pat, G["pf"])
    if external:
        ec = C.EXTERNAL[cohort]; _, ep, _ = C.load_cohort(ec)
        G.update(ecohort=ec, epat=ep, epf=patient_feats(ec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort"); ap.add_argument("--external", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tasks", default="all"); ap.add_argument("--only", default="")
    ap.add_argument("--fsets", default="all")
    ap.add_argument("--models", default="default")
    a = ap.parse_args()
    _init(a.cohort, a.external); pat = G["pat"]
    cfgs = []
    for task, (kind, _) in C.TASKS[a.cohort].items():
        if C.task_frame(pat, a.cohort, task)[0] is None or (a.tasks != "all" and task not in a.tasks.split(",")):
            continue
        if a.external and task not in C.TASKS[C.EXTERNAL[a.cohort]]:
            continue
        models = {"default": (["lgbm", "lr"] if kind == "bin" else ["cox", "rsf"]),
                  "cv": (["lgbmcv", "lrcv"] if kind == "bin" else ["coxcv", "rsfcv"]),
                  "sel": (["lgbmsel", "lrsel"] if kind == "bin" else ["coxsel", "rsfsel"]),
                  "surv": ([] if kind == "bin" else ["gbscv", "cnetcv"]),
                  "rsf2": ([] if kind == "bin" else ["rsf2cv", "rsf2sel"]),
                  "surv5": ([] if kind == "bin" else ["estcv", "rsfpca", "coxpca", "coxcv"]),
                  "risk2": ([] if kind == "bin" else ["estcvrisk", "estcvriskE", "estriskE"]),
                  "risk": (["lgbmrisk"] if kind == "bin" else ["estrisk", "coxrisk", "rsfrisk"]),
                  "clin2": ([] if kind == "bin" else ["estselS", "rsfselS", "coxselS", "estpca"]),
                  "clin": ([] if kind == "bin" else ["rsf2cv", "coxcv", "cnetcv", "rsfpca", "coxpca"])}[a.models]
        for fs in ([f for f in FSETS if f in G["pf"]] if a.fsets == "all" else [f for f in a.fsets.split(",") if f in G["pf"]]):
            for m in models:
                for f in ([0] if a.external else range(5)):
                    for s in range(3):
                        cfgs.append((task, fs, m, f, s, a.external))
    print(a.cohort, "runs", len(cfgs), flush=True)
    if a.only:   # explicit list of method names "model_fset", e.g. lgbm_comp_mm,estrisk_c_ours
        want = set(a.only.split(","))
        fs_all = sorted(G["pf"], key=len, reverse=True)
        cfgs = []
        for task, (kind, _) in C.TASKS[a.cohort].items():
            if C.task_frame(pat, a.cohort, task)[0] is None or (a.tasks != "all" and task not in a.tasks.split(",")):
                continue
            if a.external and task not in C.TASKS[C.EXTERNAL[a.cohort]]:
                continue
            for w in want:
                fs = next((f for f in fs_all if w.endswith("_" + f)), None)
                if fs is None:
                    continue
                m = w[:-(len(fs) + 1)]
                surv_m = m.startswith(("cox", "rsf", "est", "gbs", "cnet"))
                if surv_m != (kind == "surv"):
                    continue
                for f in ([0] if a.external else range(5)):
                    for sd in range(3):
                        cfgs.append((task, fs, m, f, sd, a.external))
    print(a.cohort, "runs", len(cfgs), flush=True)
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.cohort, a.external)) as pool:
        for out, msg in pool.imap_unordered(run_one, cfgs):
            print(out, msg, flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
