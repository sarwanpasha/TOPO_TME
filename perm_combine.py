"""Permutation-calibrated survival p-values (final JSON or checkpoints) and signed Stouffer combination."""
import os, glob, json, numpy as np, common as C
from scipy.stats import norm, chi2

# observed statistics and HR direction from the nested-CV runs
OBS = {"basel": (3.536590679933056, 1.2307), "luad": (0.2230650938126928, 1.0309),
       "danenberg": (chi2.isf(0.017, 1), 1.137), "cords": (7.442009200727625, 1.1094)}


def p_perm(c):
    f = os.path.join(C.RES, "interp", f"permnull_{c}_OS_clinical.npy")
    d = os.path.join(C.RES, "interp", "permck", f"{c}_OS_clinical")
    v = None
    if os.path.isdir(d) and len(os.listdir(d)) > 0:
        v = np.array([float(open(x).read()) for x in glob.glob(os.path.join(d, "*.txt"))])
    if (v is None or len(v) < 200) and os.path.exists(f):
        v = np.load(f)
    lr = OBS[c][0]
    return (1 + (v >= lr).sum()) / (1 + len(v)), len(v), v.mean(), (v > 3.84).mean()


def stouffer(cs, P):
    z = [np.sign(np.log(OBS[c][1])) * norm.isf(P[c] / 2) for c in cs]
    Z = sum(z) / np.sqrt(len(z))
    return Z, 2 * norm.sf(abs(Z))


P = {}
for c in OBS:
    p, n, m, fr = p_perm(c)
    P[c] = p
    print(f"{c:10s} B={n} p_perm={p:.3f} null_meanLR={m:.2f} frac>3.84={fr:.3f}")
for name, cs in (("development", ["danenberg", "basel", "luad"]), ("all4", ["danenberg", "basel", "luad", "cords"])):
    Z, p = stouffer(cs, P)
    print(f"stouffer {name}: z={Z:.2f} p={p:.4f}")
