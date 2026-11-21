"""Synthetic marked point-process benchmarks with ground truth.

All classes have matched composition. Tasks:
  ring     : T cells form closed rings around tumour nests (y=1) vs. arcs of the
             same spacing covering 70% of the circle + equal-spacing arcs elsewhere (y=0).
             -> differs in H1 (loops), not in composition / local spacing.
  interact : B cells attracted to T cells (y=1) vs. uniform B cells (y=0).
             -> ground-truth interacting pair (T,B) for interpretability.
  channel  : immune cells in long connected chains (y=1) vs. many short chains
             with identical step length (y=0) -> differs in H0 at mid scales.
Usage: python synth.py <task> [n_images]
"""
import sys, os
import numpy as np
import pandas as pd
from preprocess import finish_image, save_cohort, log

TYPES = ["tumour", "T cell", "stroma", "B cell", "macrophage"]
W = 600.0
NM = 8
MU = np.random.default_rng(123).normal(size=(len(TYPES) + 1, NM))
MU[5] = MU[1]; MU[5, 0] += 2.5   # hidden subtype: ring-forming T cells (m0-high), labelled "T cell"


def markers(types, rng):
    return (MU[types] + rng.normal(scale=1.0, size=(len(types), NM))).astype(np.float16)


def background(rng, n_stroma=700, n_mac=80, n_nests=6):
    pts, ty = [], []
    pts.append(rng.uniform(0, W, (n_stroma, 2))); ty += [2] * n_stroma
    pts.append(rng.uniform(0, W, (n_mac, 2))); ty += [4] * n_mac
    centers = rng.uniform(90, W - 90, (n_nests, 2))
    for c in centers:
        r = 35 * np.sqrt(rng.uniform(0, 1, 60)); a = rng.uniform(0, 2 * np.pi, 60)
        pts.append(c + np.stack([r * np.cos(a), r * np.sin(a)], 1)); ty += [0] * 60
    return pts, ty, centers


def arc(c, R, a0, frac, step, rng):
    L = 2 * np.pi * R * frac
    n = max(int(L / step), 2)
    a = a0 + np.linspace(0, 2 * np.pi * frac, n, endpoint=(frac < 1))
    p = c + R * np.stack([np.cos(a), np.sin(a)], 1)
    return p + rng.normal(scale=1.5, size=p.shape)


def gen_ring(y, rng):
    """closed rings (y=1) vs rings with one gap of GAP cells (y=0); the removed cells
    are placed as a short arc in the stroma so composition is identical."""
    pts, ty, centers = background(rng, n_nests=int(rng.integers(4, 9)))
    step, GAP = 9.0, 4
    T = []; spare = 0
    for c in centers:
        R = rng.uniform(42, 55)
        ring = arc(c, R, rng.uniform(0, 2 * np.pi), 1.0, step, rng)
        if not y:
            ring = ring[GAP:]; spare += GAP
        T.append(ring)
    while spare > 0:
        k = min(spare, 8)
        cc = rng.uniform(60, W - 60, 2)
        T.append(arc(cc, rng.uniform(30, 45), rng.uniform(0, 6.3), k * step / (2 * np.pi * 37), step, rng)[:k])
        spare -= k
    T = np.concatenate(T)
    pts.append(T); ty += [1] * len(T)
    pts.append(rng.uniform(0, W, (40, 2))); ty += [3] * 40
    return np.concatenate(pts), np.array(ty)


def gen_interact(y, rng):
    pts, ty, _ = background(rng)
    nT, nB = 150, 120
    T = rng.uniform(0, W, (nT, 2)); pts.append(T); ty += [1] * nT
    if y:
        B = T[rng.integers(0, nT, nB)] + rng.normal(scale=10, size=(nB, 2))
    else:
        B = rng.uniform(0, W, (nB, 2))
    pts.append(np.clip(B, 0, W)); ty += [3] * nB
    return np.concatenate(pts), np.array(ty)


def walk(n, step, rng, start=None):
    p = [rng.uniform(50, W - 50, 2) if start is None else start]
    ang = rng.uniform(0, 2 * np.pi)
    for _ in range(n - 1):
        ang += rng.normal(scale=0.35)
        nxt = p[-1] + step * np.array([np.cos(ang), np.sin(ang)])
        if not (0 < nxt[0] < W and 0 < nxt[1] < W):
            ang += np.pi; nxt = p[-1] + step * np.array([np.cos(ang), np.sin(ang)])
        p.append(nxt)
    return np.array(p)


def gen_channel(y, rng):
    """8 chain segments of 30 cells (same step) joined end-to-end into one connected
    channel (y=1) or left separated by gaps of ~3 steps (y=0)."""
    pts, ty, _ = background(rng)
    step, nseg, L = 9.0, 8, 30
    path = walk(nseg * L + (0 if y else 3 * (nseg - 1)), step, rng)
    if y:
        C = path
    else:
        keep = []
        for k in range(nseg):
            a = k * (L + 3); keep.append(path[a:a + L])
        C = np.concatenate(keep)
    pts.append(C); ty += [1] * len(C)
    pts.append(rng.uniform(0, W, (40, 2))); ty += [3] * 40
    return np.concatenate(pts), np.array(ty)


def gen_ringmk(y, rng):
    """as ring, but ring cells are an m0-high *subpopulation* of T cells hidden among
    many diffuse T cells of the same label -> type-level fixed PH cannot isolate them."""
    xy, t = gen_ring(y, rng)
    t = t.copy(); t[t == 1] = 5
    D = rng.uniform(0, W, (500, 2))
    return np.concatenate([xy, D]), np.concatenate([t, np.ones(500, int)])


GEN = {"ringmk": gen_ringmk, "ring": gen_ring, "interact": gen_interact, "channel": gen_channel}

if __name__ == "__main__":
    task = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    rng = np.random.default_rng({"ring": 1, "interact": 2, "channel": 3, "ringmk": 4}[task])
    images, rows = [], []
    for i in range(n):
        y = i % 2
        xy, t = GEN[task](y, rng)
        xy = np.clip(xy, 0, W)
        mk = markers(t, rng); t = np.where(t == 5, 1, t)
        images.append(finish_image(f"img{i}", f"p{i}", xy, t, mk, len(TYPES)))
        rows.append({"patient_id": f"p{i}", "y": float(y)})
        if i % 100 == 0:
            log(task, i)
    meta = dict(types=TYPES, markers=[f"m{j}" for j in range(NM)], covariates=[],
                truth={"interact": ["T cell", "B cell"]}.get(task))
    save_cohort("synth_" + task, images, pd.DataFrame(rows), meta)
    log("DONE synth", task)
