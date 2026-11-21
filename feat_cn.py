"""Spatial baselines that are NOT topological (for positioning):
  cn : cellular-neighbourhood frequencies (Schuerch et al., Cell 2020): for each cell, the cell-type
       composition of its 10 nearest neighbours; cohort-wide MiniBatchKMeans (k=10, unsupervised, no
       labels) assigns each cell a neighbourhood; per-image neighbourhood fractions (10) + per-type
       neighbourhood occupancy entropy.
python feat_cn.py <cohort>  -> feats_cn.npz {cn, image_id}"""
import os, sys
import numpy as np
import common as C
from sklearn.cluster import MiniBatchKMeans

K_NN, K_CN = 10, 10

if __name__ == "__main__":
    c = sys.argv[1]
    imgs, pat, meta = C.load_cohort(c); nt = len(meta["types"])
    W = []
    for im in imgs:
        t = im["types"].astype(int); ki = im["knn_idx"][:, :K_NN]
        oh = np.eye(nt, dtype=np.float32)[t[ki]].mean(1)
        W.append(oh)
    allW = np.concatenate(W)
    rng = np.random.default_rng(0)
    sub = allW[rng.choice(len(allW), min(len(allW), 400000), replace=False)]
    km = MiniBatchKMeans(K_CN, random_state=0, n_init=5, batch_size=8192).fit(sub)
    F = []
    for im, w in zip(imgs, W):
        lab = km.predict(w); t = im["types"].astype(int)
        frac = np.bincount(lab, minlength=K_CN) / len(lab)
        ent = []
        for k in range(nt):
            m = t == k
            if m.sum() < 2:
                ent.append(0.0); continue
            p = np.bincount(lab[m], minlength=K_CN) / m.sum(); p = p[p > 0]
            ent.append(float(-(p * np.log(p)).sum()))
        F.append(np.concatenate([frac, ent]))
    np.savez_compressed(os.path.join(C.cohort_dir(c), "feats_cn.npz"), cn=np.stack(F).astype(np.float32),
                        image_id=np.array([im["image_id"] for im in imgs]))
    print("DONE cn", c, np.stack(F).shape, flush=True)
