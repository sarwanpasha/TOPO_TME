"""Whole-slide analysis on IMMUcan multiplex immunofluorescence slides (Zenodo 15855866, CC BY 4.0; 10 slides of breast,
colorectal, head and neck, lung and renal cancer, 6 markers: CK, CD3, CD20, CD163, CD11c, CD15).
For each slide (cells inside the annotated tumor-tissue region):
  * TopoNull (type and marker fields) on the whole region, with wall-clock time;
  * TopoNull on virtual cores of 1 mm diameter placed on a lattice over the region (the size of a TMA core).
Outputs results/interp/immucan_ws.npz and prints how well core-level descriptors represent the slide:
fraction of the variance of core-level descriptors explained by the slide (ICC), the number of cores needed for a
reliability of 0.8 (Spearman-Brown), and the correlation across slides between the mean of k random cores and the
whole-region value.  python immucan_ws.py [workers]"""
import os, sys, glob, time, zlib
import numpy as np, pandas as pd
import common as C
import feat_compact as FC
import feat_null as FN
import topo_core as T

RAW = os.path.join(os.path.dirname(C.RES), "raw", "immucan", "mIF", "data", "whole_slide")
MK = ["CK", "CD3", "CD20", "CD163", "CD11c", "CD15"]
TYPES = ["tumor", "T cell", "B cell", "macrophage", "dendritic", "granulocyte", "other"]
CORE_UM, MIN_CELLS, MAX_CORES = 1000.0, 300, 40


def typing(ph):
    s = pd.Series(ph).astype(str)
    t = np.full(len(s), 6)
    for k, m in reversed(list(enumerate(["CK+", "CD3+", "CD20+", "CD163+", "CD11c+", "CD15+"]))):
        t[s.str.contains(m, regex=False).values] = k
    return t


def tex(xy, t, Z, nt):
    """TopoNull type-field and marker-field descriptors (no cross-type block, so that whole slides stay tractable)."""
    E, TE, Fc = T.build_complex(xy)
    sc = float(np.median(np.linalg.norm(xy[E[:, 0]] - xy[E[:, 1]], axis=1)))
    im = dict(coords=xy, scale=sc, edges=E, tri_edges=TE, faces=Fc)
    rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(xy[:50]).tobytes()))

    def both(tt, ZZ):
        el = FC.BETA * np.linalg.norm(xy[E[:, 0]] / sc - xy[E[:, 1]] / sc, axis=1)
        out = []
        for k in range(nt):
            if (tt == k).sum() < 2:
                out.append(np.zeros(FC.PER_TYPE)); continue
            fv = (tt != k).astype(np.float64); fe = np.maximum(fv[E[:, 0]], fv[E[:, 1]]) + el
            out.append(FC.field_feats(fv, fe, E, TE, Fc, FC.BETA * FC.LVL_LEN, bmax=1.0))
        return np.concatenate(out + [FN._marker_part(im, ZZ)])
    obs = both(t, Z)
    groups = [np.flatnonzero(t == k) for k in range(nt)]
    nul = []
    for _ in range(FN.K_NULL):
        tp = rng.permutation(t); Zp = Z.copy()
        for g in groups:
            if len(g) > 1:
                Zp[g] = Z[rng.permutation(g)]
        nul.append(both(tp, Zp))
    nul = np.stack(nul); mu, sd = nul.mean(0), nul.std(0)
    return np.clip(np.nan_to_num((obs - mu) / (sd + FN.EPS * (np.abs(mu) + 1e-3) + 1e-6)), -8, 8).astype(np.float32)


def core_job(a):
    xy, t, Z = a
    return tex(xy, t, Z, len(TYPES))


def load(f):
    d = pd.read_csv(f, sep="\t", usecols=["nucleus.x", "nucleus.y", "phenotype", "in.ROI.tumor_tissue"] + [m + ".score.normalized" for m in MK])
    d = d[d["in.ROI.tumor_tissue"].astype(str).str.lower() == "true"].reset_index(drop=True)
    xy = d[["nucleus.x", "nucleus.y"]].values.astype(np.float64)
    xy, ui = np.unique(xy, axis=0, return_index=True); d = d.iloc[ui].reset_index(drop=True)
    Z = np.log1p(np.clip(d[[m + ".score.normalized" for m in MK]].values.astype(np.float64), 0, None))
    Z = np.clip((Z - Z.mean(0)) / (Z.std(0) + 1e-9), -4, 4)
    return xy, typing(d["phenotype"].values), Z


if __name__ == "__main__":
    import multiprocessing as mp
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    files = sorted(glob.glob(os.path.join(RAW, "*_#_cells_properties_#_*.tsv.gz")))
    rng = np.random.default_rng(0)
    res = {}
    for f in files:
        name = os.path.basename(f).split("_#_")[0]
        xy, t, Z = load(f)
        # virtual cores on a lattice
        lo, hi = xy.min(0), xy.max(0)
        cs = [(x, y) for x in np.arange(lo[0] + CORE_UM / 2, hi[0], CORE_UM) for y in np.arange(lo[1] + CORE_UM / 2, hi[1], CORE_UM)]
        rng.shuffle(cs)
        jobs = []
        for (x, y) in cs:
            m = ((xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2) <= (CORE_UM / 2) ** 2
            if m.sum() >= MIN_CELLS:
                jobs.append((xy[m], t[m], Z[m]))
            if len(jobs) >= MAX_CORES:
                break
        t0 = time.time()
        with mp.get_context("spawn").Pool(W) as p:
            cores = np.stack(p.map(core_job, jobs, chunksize=1))
        tc = time.time() - t0
        t0 = time.time(); whole = tex(xy, t, Z, len(TYPES)); tw = time.time() - t0
        res[name] = dict(cores=cores, whole=whole, n=len(xy), area_mm2=float(np.prod(hi - lo)) / 1e6, n_cores=len(jobs))
        print(f"{name}: cells {len(xy):,} cores {len(jobs)} core-time {tc:.0f}s whole-region TopoNull {tw:.0f}s (1 core)", flush=True)
        np.savez_compressed(os.path.join(C.RES, "interp", "immucan_ws.npz"),
                            **{k + "|cores": v["cores"] for k, v in res.items()}, **{k + "|whole": v["whole"] for k, v in res.items()},
                            meta=np.array([f"{k},{v['n']},{v['n_cores']}" for k, v in res.items()]))
