"""Train/evaluate neural methods. One process per (task, fold, seed); pooled with fork.

python train.py --cohort danenberg --method mph --workers 16
python train.py --cohort basel --method mph --external     (train Basel -> test Zurich)
python train.py --smoke
"""
import os, sys, json, time, argparse, traceback
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn.functional as F
import multiprocessing as mp
import common as C
import models3  # noqa: F401  registers topo3* and wraps models.build_model (must precede next line)
from models import build_model, to_t, phen, METHODS

G = {}  # globals shared via fork


def setup(cohort, external=False):
    if G.get("cohort") == cohort:
        return
    imgs, pat, meta = C.load_cohort(cohort)
    G.update(cohort=cohort, imgs=imgs, pat=pat, meta=meta, n_types=len(meta["types"]))
    fe = np.load(os.path.join(C.cohort_dir(cohort), "feats.npz"))
    G["comp"] = {iid: np.concatenate([a, b, c]) for iid, a, b, c in
                 zip(fe["image_id"], fe["comp"], fe["mm"], fe["logn"])}
    if external:
        ext = C.EXTERNAL[cohort]
        ei, ep, em = C.load_cohort(ext)
        assert em["types"] == meta["types"] and em["markers"] == meta["markers"]
        fe2 = np.load(os.path.join(C.cohort_dir(ext), "feats.npz"))
        G["ext"] = dict(cohort=ext, imgs=ei, pat=ep, meta=em)
        G["comp"].update({"EXT:" + iid: np.concatenate([a, b, c]) for iid, a, b, c in
                          zip(fe2["image_id"], fe2["comp"], fe2["mm"], fe2["logn"])})
        for im in ei:
            im["_ext"] = True


def by_patient(imgs):
    d = {}
    for i, im in enumerate(imgs):
        d.setdefault(im["patient_id"], []).append(i)
    return d


def predict_patient(model, imgs, idxs, scale):
    outs = []
    for i in idxs:
        im = imgs[i]
        t = to_t(im, scale)
        p = phen(t, G["n_types"])
        key = ("EXT:" if im.get("_ext") else "") + im["image_id"]
        comp = torch.from_numpy(G["comp"][key]).float()
        outs.append(model(im, t, p, comp))
    return torch.stack(outs).mean(0)


def select_markers(df, tr_all, kind, pmap, imgs, K=8):
    """init heads on the K markers most associated with the outcome in TRAINING patients only."""
    nT = G["n_types"]; nM = len(G["meta"]["markers"])
    X = np.stack([np.mean([imgs[i]["markers"].astype(np.float32).mean(0) for i in pmap[pid]], 0)
                  for pid in df["patient_id"].values[tr_all]])
    if kind == "bin":
        y = df["y"].values[tr_all]
        sc = np.array([abs(C.auroc(y, X[:, j]) - 0.5) if X[:, j].std() > 0 else 0 for j in range(nM)])
    else:
        y = np.log1p(df["time"].values[tr_all]) * (2 * df["event"].values[tr_all] - 1)
        sc = np.nan_to_num(np.abs([np.corrcoef(X[:, j], y)[0, 1] if X[:, j].std() > 0 else 0 for j in range(nM)]))
    return [nT + int(j) for j in np.argsort(-sc)[:K]]


def bank_feats(cohort_key):
    fs = "compact" if G.get("cur_method", "").endswith("_stack2") else "bank"
    key = cohort_key + "_" + fs
    if key not in G:
        import baselines as BL
        G[key] = BL.patient_feats(G["cohort"] if cohort_key == "bank_main" else G["ext"]["cohort"]).get(fs)
    return G[key]


def stack_predict(model, trdf, imgs, pmap, scale, tedf, timgs, tpmap, tscale, kind, seed):
    """second stage: learned topological features + fixed feature bank -> LightGBM / RSF."""
    model.eval()
    def feats(sub, im_list, pm, sc):
        out = []
        with torch.no_grad():
            for pid in sub["patient_id"]:
                v = []
                for i in pm[pid]:
                    im = im_list[i]; t = to_t(im, sc); p = phen(t, G["n_types"])
                    v.append(model.topo_features(im, t, p).numpy())
                out.append(np.mean(v, 0))
        return np.stack(out)
    Ftr = feats(trdf, imgs, pmap, scale); Fte = feats(tedf, timgs, tpmap, tscale)
    btr = bank_feats("bank_main"); bte = bank_feats("bank_ext" if "ext" in G and timgs is G["ext"]["imgs"] else "bank_main")
    if btr is not None:
        Ftr = np.concatenate([Ftr, btr.loc[trdf["patient_id"]].values], 1)
        Fte = np.concatenate([Fte, bte.loc[tedf["patient_id"]].values], 1)
    import baselines as BL
    y = trdf["y"].values if kind == "bin" else (trdf["time"].values, trdf["event"].values)
    if G.get("cur_method", "").endswith("_stack2"):
        BL.CUR["nbase"] = 0
        return np.asarray(BL.fit_predict(kind, "lgbmcv" if kind == "bin" else "gbscv", Ftr, y, Fte, seed), float)
    return np.asarray(BL.fit_predict(kind, "lgbm" if kind == "bin" else "rsf", Ftr, y, Fte, seed), float)


def run_one(cfg):
    task, method, fold, seed, args = cfg
    tag = "_smoke" if args.smoke else ("_ext" if args.external else "")
    out = C.result_path(G["cohort"], task, method, fold, seed, tag)
    if os.path.exists(out) and not args.overwrite:
        return out, "skip"
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(1)
    t0 = time.time()
    try:
        df, kind = C.task_frame(G["pat"], G["cohort"], task)
        scale = G["meta"]["scale"]
        imgs = G["imgs"]
        pmap = by_patient(imgs)
        df = df[df["patient_id"].isin(pmap)].reset_index(drop=True)
        if args.external:
            tr_all = np.arange(len(df))
            edf, _ = C.task_frame(G["ext"]["pat"], G["ext"]["cohort"], task)
            epmap = by_patient(G["ext"]["imgs"])
            edf = edf[edf["patient_id"].isin(epmap)].reset_index(drop=True)
            test_imgs, test_pmap, test_df = G["ext"]["imgs"], epmap, edf
            scale_test = G["ext"]["meta"]["scale"]
            tr, va = C.inner_split(df, tr_all, kind, seed)
        else:
            folds = C.make_folds(df, kind)
            tr_all, te = folds[fold]
            tr, va = C.inner_split(df, tr_all, kind, seed)
            test_imgs, test_pmap, test_df, scale_test = imgs, pmap, df.iloc[te].reset_index(drop=True), scale
        p_dim = G["n_types"] + len(G["meta"]["markers"])
        comp_dim = len(next(iter(G["comp"].values())))
        init_idx = None
        is3 = method in models3.TOPO3
        if METHODS[method] in ("field", "field_nocomp") or is3:
            init_idx = select_markers(df, tr_all, kind, pmap, imgs)
        if is3:
            model = build_model(method, p_dim, comp_dim, 1, init_idx=init_idx, n_types=G["n_types"])
        else:
            model = build_model(method, p_dim, comp_dim, 1, init_idx=init_idx)
        tcfg = getattr(model, "train_cfg", {})
        if hasattr(model, "init_stats"):   # feature standardisation from (a subsample of) training images
            tr_im = [i for pid in df["patient_id"].values[tr] for i in pmap[pid]]
            tr_im = np.random.default_rng(seed).permutation(tr_im)[:256]
            samp = []
            for i in tr_im:
                t_ = to_t(imgs[i], scale)
                samp.append((imgs[i], t_, phen(t_, G["n_types"]), torch.from_numpy(G["comp"][imgs[i]["image_id"]]).float()))
            model.init_stats(samp)
        params = model.param_groups() if hasattr(model, "param_groups") else model.parameters()
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        n_epochs = args.epochs
        if tcfg.get("small_n") and len(tr) < tcfg["small_n"]:
            n_epochs = int(round(args.epochs * tcfg.get("small_mult", 2.0)))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=args.lr * 0.05) \
            if tcfg.get("cosine") else None
        patience = tcfg.get("patience")
        if kind == "bin":
            ytr = df["y"].values[tr]
            pw = torch.tensor(float((ytr == 0).sum()) / max((ytr == 1).sum(), 1))
        bs = args.bs if kind == "bin" else args.bs_surv
        best = (-1.0, None, -1); hist = []

        def evaluate(sub_df, im_list, pm, sc):
            model.eval()
            with torch.no_grad():
                pr = np.array([predict_patient(model, im_list, pm[pid], sc).item() for pid in sub_df["patient_id"]])
            model.train()
            return pr

        va_df = df.iloc[va].reset_index(drop=True)
        rng = np.random.default_rng(seed)
        for ep in range(n_epochs):
            perm = rng.permutation(tr)
            losses = []
            for s in range(0, len(perm), bs):
                b = perm[s:s + bs]
                preds = torch.stack([predict_patient(model, imgs, pmap[df["patient_id"].values[i]], scale)[0]
                                     for i in b])
                if kind == "bin":
                    y = torch.tensor(df["y"].values[b], dtype=torch.float32)
                    loss = F.binary_cross_entropy_with_logits(preds, y, pos_weight=pw)
                else:
                    tt = torch.tensor(df["time"].values[b], dtype=torch.float32)
                    ee = torch.tensor(df["event"].values[b], dtype=torch.float32)
                    if ee.sum() == 0:
                        continue
                    loss = C.cox_loss(preds, tt, ee)
                    if hasattr(model, "surv_aux"):
                        loss = loss + model.surv_aux(preds, tt, ee)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); losses.append(loss.item())
            if sched is not None:
                sched.step()
            if (ep + 1) % args.eval_every == 0 or ep == n_epochs - 1:
                pv = evaluate(va_df, imgs, pmap, scale)
                mv = C.metric(kind, va_df, pv)
                hist.append((ep, float(np.mean(losses)) if losses else None, mv))
                if np.isfinite(mv) and mv > best[0]:
                    best = (mv, {k: v.clone() for k, v in model.state_dict().items()}, ep)
                if patience and best[2] >= 0 and ep - best[2] >= patience:
                    break
        if best[1] is not None:
            model.load_state_dict(best[1])
        pt = evaluate(test_df, test_imgs, test_pmap, scale_test)
        if method.endswith("_stack") or method.endswith("_stack2"):
            G["cur_method"] = method
            pt = stack_predict(model, df.iloc[tr_all].reset_index(drop=True), imgs, pmap, scale,
                               test_df, test_imgs, test_pmap, scale_test, kind, seed)
        res = dict(cohort=G["cohort"], task=task, method=method, fold=fold, seed=seed, kind=kind,
                   external=bool(args.external), best_epoch=best[2], val_metric=best[0],
                   test_metric=C.metric(kind, test_df, pt), hist=hist,
                   patients=test_df["patient_id"].tolist(), pred=pt.tolist(),
                   secs=time.time() - t0, n_params=sum(p.numel() for p in model.parameters()))
        if kind == "bin":
            res["y"] = test_df["y"].tolist()
        else:
            res["time"] = test_df["time"].tolist(); res["event"] = test_df["event"].tolist()
        if args.save_model:
            torch.save(model.state_dict(), out.replace(".json", ".pt"))
        json.dump(res, open(out, "w"))
        return out, f"ok {res['test_metric']:.3f} ({res['secs']:.0f}s)"
    except Exception:
        return out, "FAIL\n" + traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort"); ap.add_argument("--method", default="mph")
    ap.add_argument("--tasks", default="all"); ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--bs_surv", type=int, default=64)
    ap.add_argument("--external", action="store_true"); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save_model", action="store_true"); ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.cohort = args.cohort or "synth_ring"; args.epochs = 1; args.folds = "0"; args.seeds = "0"
        args.overwrite = True; args.workers = 2
    setup(args.cohort, args.external)
    # warm numba in parent
    im = G["imgs"][0]
    from topo_core import persistence_pairs
    persistence_pairs(np.zeros(len(im["types"])), np.linalg.norm(im["coords"][im["edges"][:, 0]] - im["coords"][im["edges"][:, 1]], axis=1),
                      im["edges"], im["tri_edges"], im["faces"])
    tasks = list(C.TASKS[args.cohort]) if args.tasks == "all" else args.tasks.split(",")
    if args.external:
        tasks = [t for t in tasks if t in C.TASKS[C.EXTERNAL[args.cohort]]]
        folds = [0]
    else:
        folds = [int(x) for x in args.folds.split(",")]
    tasks = [t for t in tasks if C.task_frame(G["pat"], args.cohort, t)[0] is not None]
    methods = args.method.split(",")
    cfgs = [(t, m, f, s, args) for m in methods for t in tasks for f in folds for s in map(int, args.seeds.split(","))]
    print(f"cohort={args.cohort} tasks={tasks} methods={methods} runs={len(cfgs)}", flush=True)
    nfail = 0
    if args.smoke:
        cfgs = [(tasks[0], m, 0, 0, args) for m in (methods if args.method != "mph" else list(METHODS))]
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    with ctx.Pool(args.workers, initializer=setup, initargs=(args.cohort, args.external)) as pool:
        for out, msg in pool.imap_unordered(run_one, cfgs):
            print(out, msg, flush=True)
            nfail += msg.startswith("FAIL")
    print("ALL DONE fails=", nfail, flush=True)
    if args.smoke and nfail:
        sys.exit(1)


if __name__ == "__main__":
    main()
