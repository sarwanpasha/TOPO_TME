"""Learned cell-state fields using all three devices of the laptop.
  GPU (Intel Graphics, torch 'xpu'): trains a self-supervised spatial-context cell encoder
       (denoising autoencoder on [own markers, mean markers of 8 nearest neighbours]); no outcome labels used.
  NPU (Intel AI Boost, OpenVINO):    runs the trained encoder over every cell (static batch) -> K-dim cell embedding.
  CPU (numba):                       exact H0/H1 persistence of each embedding dimension as a field on the Delaunay
                                     complex (compact statistics) -> feats_emb.npz
python cellenc.py <train_cohort> [apply_cohort ...]     e.g.  cellenc.py danenberg | cellenc.py basel zurich
"""
import os, sys, time, json, importlib
import numpy as np
import common as C

K = 8; BATCH = 8192; NPU_BATCH = 4096; EPOCHS = 12


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cell_inputs(imgs):
    X = []
    for im in imgs:
        m = np.clip(im["markers"].astype(np.float32), -5, 5)
        nb = m[im["knn_idx"].astype(np.int64)].mean(1)
        X.append(np.concatenate([m, nb], 1))
    return X


def train_encoder(X, dev_pref="xpu"):
    import torch, torch.nn as nn
    dev = "cpu"
    if dev_pref == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        dev = "xpu"
    log("training device:", dev, torch.xpu.get_device_name(0) if dev == "xpu" else "")
    d = X.shape[1]
    enc = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, K))
    dec = nn.Sequential(nn.Linear(K, 64), nn.GELU(), nn.Linear(64, d))
    enc.to(dev); dec.to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=2e-3, weight_decay=1e-5)
    Xt = torch.from_numpy(X); n = len(Xt); g = torch.Generator().manual_seed(0); t0 = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g); tot = 0.0; nb = 0
        for s in range(0, n, BATCH):
            xb = Xt[perm[s:s + BATCH]].to(dev)
            mask = (torch.rand(xb.shape, device=dev) > 0.3).float()
            loss = ((dec(enc(xb * mask)) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        log(f"epoch {ep+1}/{EPOCHS} loss {tot/nb:.4f} ({time.time()-t0:.0f}s)")
    return enc.to("cpu").eval(), dev, time.time() - t0


def infer(enc, X, d):
    """encoder inference through OpenVINO with a static batch: NPU first, then GPU, then CPU (all timed and checked)."""
    import torch, openvino as ov
    core = ov.Core()
    m = ov.convert_model(enc, example_input=torch.zeros(NPU_BATCH, d), input=[(NPU_BATCH, d)])
    out = {}
    order = [x for x in ("NPU", "GPU", "CPU") if x in core.available_devices]
    probe = np.zeros((NPU_BATCH, d), np.float32); probe[:min(len(X), NPU_BATCH)] = X[:NPU_BATCH]
    with torch.no_grad():
        ref = enc(torch.from_numpy(probe)).numpy()
    used = None; emb = None
    for devn in order:
        try:
            cm = core.compile_model(m, devn); t0 = time.time()
            res = np.zeros((len(X), K), np.float32)
            for s in range(0, len(X), NPU_BATCH):
                xb = X[s:s + NPU_BATCH]
                if len(xb) < NPU_BATCH:
                    xb = np.concatenate([xb, np.zeros((NPU_BATCH - len(xb), d), np.float32)])
                res[s:s + NPU_BATCH] = cm(xb)[0][:len(X) - s]
            secs = time.time() - t0
            err = float(np.abs(cm(probe)[0] - ref).max())
            out[devn] = dict(secs=round(secs, 2), cells_per_s=int(len(X) / max(secs, 1e-6)), max_abs_err_vs_torch=err)
            log("inference", devn, out[devn])
            if used is None and err < 0.05:
                used, emb = devn, res
        except Exception as e:
            out[devn] = dict(error=repr(e)[:200]); log("inference", devn, "FAILED", repr(e)[:200])
    return emb, used, out


def topo_feats(imgs, embs, meta, mu, sd):
    FC = importlib.import_module("feat_compact"); F = []
    for im, e in zip(imgs, embs):
        z = np.clip((e - mu) / sd, -4, 4).astype(np.float16)
        j = dict(coords=im["coords"], edges=im["edges"], tri_edges=im["tri_edges"], faces=im["faces"],
                 types=im["types"], markers=z, scale=meta["scale"])
        F.append(np.concatenate([FC.compute(j), z.astype(np.float32).mean(0), z.astype(np.float32).std(0)]))
    return np.stack(F).astype(np.float32)


if __name__ == "__main__":
    cohorts = sys.argv[1:]; report = {}
    imgs, pat, meta = C.load_cohort(cohorts[0])
    Xl = cell_inputs(imgs); X = np.concatenate(Xl); d = X.shape[1]
    log(cohorts[0], "cells", len(X), "input dim", d)
    enc, tdev, tsecs = train_encoder(X)
    report["train"] = dict(device=tdev, secs=round(tsecs, 1), cells=int(len(X)))
    mu = sd = None
    for c in cohorts:
        if c != cohorts[0]:
            imgs, pat, meta = C.load_cohort(c); Xl = cell_inputs(imgs); X = np.concatenate(Xl)
        emb, used, rep = infer(enc, X, d); report["infer_" + c] = dict(used=used, **rep)
        if mu is None:
            mu, sd = emb.mean(0), emb.std(0) + 1e-6
        cuts = np.cumsum([len(x) for x in Xl])[:-1]; t0 = time.time()
        F = topo_feats(imgs, np.split(emb, cuts), meta, mu, sd)
        report["persistence_cpu_secs_" + c] = round(time.time() - t0, 1)
        np.savez_compressed(os.path.join(C.cohort_dir(c), "feats_emb.npz"), emb=F,
                            image_id=np.array([im["image_id"] for im in imgs]))
        log("DONE", c, F.shape, "inference device used:", used)
    json.dump(report, open(os.path.join(C.cohort_dir(cohorts[0]), "hardware_report.json"), "w"), indent=1)
    log("REPORT", json.dumps(report))
