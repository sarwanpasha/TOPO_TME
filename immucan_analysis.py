"""Analysis of immucan_ws.npz: how well do TMA-sized cores represent whole tumor regions?"""
import os, numpy as np
from scipy.stats import spearmanr
import common as C

z = np.load(os.path.join(C.RES, "interp", "immucan_ws.npz"))
names = sorted({k.split("|")[0] for k in z.files if "|" in k})
cores = [z[n + "|cores"] for n in names]; whole = np.stack([z[n + "|whole"] for n in names])
D = cores[0].shape[1]
keep = np.array([np.std(np.concatenate([c[:, j] for c in cores])) > 1e-6 for j in range(D)])
print("slides", len(names), "descriptors", D, "non-constant", keep.sum())
icc, kneed = [], []
for j in np.flatnonzero(keep):
    groups = [c[:, j] for c in cores]
    m = np.array([g.mean() for g in groups]); n = np.array([len(g) for g in groups])
    within = np.mean([g.var(ddof=1) for g in groups]); between = max(m.var(ddof=1) - within / n.mean(), 0)
    r = between / (between + within) if between + within > 0 else 0
    icc.append(r)
    kneed.append(np.inf if r <= 0 else 0.8 * (1 - r) / (0.2 * r))   # Spearman-Brown: k for reliability 0.8
icc = np.array(icc); kneed = np.array(kneed)
print(f"ICC (share of variance between slides): median {np.median(icc):.2f}, IQR {np.percentile(icc, 25):.2f}-{np.percentile(icc, 75):.2f}")
print(f"cores for reliability 0.8: median {np.median(kneed[np.isfinite(kneed)]):.1f}, IQR "
      f"{np.percentile(kneed[np.isfinite(kneed)], 25):.1f}-{np.percentile(kneed[np.isfinite(kneed)], 75):.1f}; "
      f"fraction of descriptors with ICC>=0.5: {(icc >= 0.5).mean():.2f}")
rng = np.random.default_rng(0)
for k in (1, 2, 4, 8):
    rs = []
    for j in np.flatnonzero(keep):
        vals = []
        for _ in range(20):
            mk = np.array([c[rng.choice(len(c), min(k, len(c)), replace=False), j].mean() for c in cores])
            vals.append(spearmanr(mk, whole[:, j])[0])
        rs.append(np.nanmean(vals))
    print(f"k={k} cores: median Spearman correlation with the whole-region value across slides {np.nanmedian(rs):.2f}")
