"""Parallel feature extraction (spawn-safe): python featrun.py <module> <cohort> [workers]
modules: feat_compact, feat_cross, feat_mfield, feat_mfield2"""
import os, sys, importlib, inspect, time
import numpy as np, multiprocessing as mp
import common as C


def one(arg):
    mod, im, nt = arg
    f = importlib.import_module(mod).compute
    return f(im, nt) if len(inspect.signature(f).parameters) > 1 else f(im)


if __name__ == "__main__":
    mod, c = sys.argv[1], sys.argv[2]; w = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    M = importlib.import_module(mod)
    imgs, pat, meta = C.load_cohort(c)
    for im in imgs:
        im["scale"] = meta["scale"]
    nt = len(meta["types"]); t0 = time.time()
    keep = ("coords", "types", "markers", "edges", "tri_edges", "faces", "scale", "knn_idx", "knn_d")
    args = [(mod, {k: im[k] for k in keep if k in im}, nt) for im in imgs]
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    with ctx.Pool(w) as p:
        F = p.map(one, args, chunksize=4)
    F = np.stack(F).astype(np.float32)
    np.savez_compressed(os.path.join(C.cohort_dir(c), f"feats_{M.NAME}.npz"), **{M.NAME: F},
                        image_id=np.array([im["image_id"] for im in imgs]))
    print("DONE", mod, c, F.shape, f"{time.time()-t0:.0f}s", flush=True)
