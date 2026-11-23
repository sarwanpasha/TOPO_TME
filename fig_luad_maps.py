"""Cell-level maps for the matched LUAD pair of fig_luad.py.
Row a: cancer-cell Delaunay filtration (the cancer-cancer interaction filtration); every H1 class with persistence > 0.3
is drawn as a circle centred at the midpoint of the edge that kills it (inside the hole), radius = half its death length.
Row b: CD31 superlevel set at the pooled 90th percentile of the 2 cores; cells above the threshold are coloured by their
connected component in the Delaunay graph (the H0 classes of the superlevel filtration at that level).
Writes export/paper/fig_luad_maps.tex (pgfplots) and prints summary numbers."""
import os, numpy as np
import common as C
from interp import patient_matrix, cross_names
import feat_null as FN
import topo_core as T

imgs, pat, meta = C.load_cohort("luad")
types = meta["types"]; markers = meta["markers"]
comp = patient_matrix("luad", "comp", None)
cr = patient_matrix("luad", "cross", cross_names(types))
zn = patient_matrix("luad", "znull", FN.feature_names(markers, types))
p = pat.set_index("patient_id")
ids = [i for i in comp.index if i in p.index and not np.isnan(p.loc[i, "solid"])]
loops = cr.loc[ids, "X:Cancer~Cancer|nloops"]; cd31 = zn.loc[ids, "zM:CD31|sup|H0|wbirth"]
solid = p.loc[ids, "solid"].values == 1
score = (loops.rank(pct=True) + cd31.rank(pct=True)).values
S = np.array(ids)[solid][np.argsort(-score[solid])][:25]
N = np.array(ids)[~solid][np.argsort(score[~solid])][:60]
best = min(((s, n) for s in S for n in N), key=lambda q: np.abs(comp.loc[q[0]].values - comp.loc[q[1]].values).sum())
print("pair", best)

ci = types.index("Cancer"); mi = markers.index("CD31")
ims = [next(i for i in imgs if i["patient_id"] == pid) for pid in best]
_pool = np.concatenate([np.asarray(im["markers"])[:, mi] for im in ims])
_pos = _pool[_pool > _pool.min() + 1e-6]
thr = _pool.min() + 1e-6  # cells with detectable CD31
print("CD31 pooled: n", len(_pool), "detectable", len(_pos))
rng = np.random.default_rng(0)
COLS = ["red!85!black", "orange!90!black", "violet", "teal", "magenta!80!black", "brown"]


def norm(xy, ref):
    lo = ref.min(0); s = (ref.max(0) - lo).max()
    return (xy - lo) / s * 100, 100 / s


def pts(xy, idx, col, size, nmax=None):
    if nmax is not None and len(idx) > nmax:
        idx = rng.choice(idx, nmax, replace=False)
    if len(idx) == 0:
        return ""
    c = " ".join(f"({xy[i, 0]:.1f},{xy[i, 1]:.1f})" for i in idx)
    return f"\\addplot[only marks,mark=*,mark size={size}pt,color={col}] coordinates {{{c}}};\n"


rowA, rowB, info = [], [], []
for im in ims:
    xy0 = np.asarray(im["coords"], float); t = np.asarray(im["types"]).astype(int)
    xy, f = norm(xy0, xy0)
    sc = float(meta["scale"])
    # a: cancer loops
    cidx = np.flatnonzero(t == ci); oidx = np.flatnonzero(t != ci)
    cx = xy0[cidx] / sc
    edges, tri_edges, faces = T.build_complex(cx)
    fe = np.linalg.norm(cx[edges[:, 0]] - cx[edges[:, 1]], axis=1)
    h0b, h0d, h1b, h1d = T.persistence_pairs(np.zeros(len(cx)), fe, edges, tri_edges, faces)
    per = fe[h1d] - fe[h1b]; keep = per > 0.3
    s = pts(xy, oidx, "black!12", 0.6, 500) + pts(xy, cidx, "black!55", 0.7, 900)
    cen = np.array([xy[cidx[edges[e]]].mean(0) for e in h1d[keep]]).reshape(-1, 2)
    circ = ("\\addplot[only marks,mark=o,mark size=1.3pt,mark options={line width=0.6pt},color=blue!80!black] coordinates {"
            + " ".join(f"({a:.1f},{b:.1f})" for a, b in cen) + "};\n") if len(cen) else ""
    ext = xy.max(0)
    rowA.append((s + circ, ext))
    # b: CD31 superlevel components
    v = np.asarray(im["markers"])[:, mi]
    hi = np.flatnonzero(v > thr)
    E = np.asarray(im["edges"])
    E = E[np.isin(E[:, 0], hi) & np.isin(E[:, 1], hi)]
    par = {i: i for i in hi}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in E:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    roots = np.array([find(i) for i in hi]) if len(hi) else np.array([], int)
    labs, cnt = np.unique(roots, return_counts=True)
    order = labs[np.argsort(-cnt)]
    s = pts(xy, np.setdiff1d(np.arange(len(t)), hi), "black!12", 0.6, 700)
    s += pts(xy, hi[np.isin(roots, order[len(COLS):])], "red!35", 0.9)
    for k, lab in enumerate(order[:len(COLS)]):
        s += pts(xy, hi[roots == lab], COLS[k], 1.1)
    rowB.append((s, ext))
    info.append(dict(n_loops=int(keep.sum()), n_cancer=len(cidx), n_hi=len(hi), n_comp=len(labs),
                     largest=int(cnt.max()) if len(cnt) else 0, frac_top3=float(np.sort(cnt)[::-1][:3].sum() / max(len(hi), 1))))
print(info, "thr", round(float(thr), 3))

ax = "xmin=0,ymin=0,xmax=101,ymax=101,hide axis,axis equal image"
body = ""
for k, (a, b) in enumerate([(rowA[0], rowA[1]), (rowB[0], rowB[1])]):
    lab = "ab"[k]
    body += (f"\\nextgroupplot[title={{\\textbf{{{lab}}}\\ \\ solid}}]\n{a[0]}"
             f"\\nextgroupplot[title={{non-solid}}]\n{b[0]}")
tex = ("\\begin{figure}[t]\n\\centering\n\\begin{tikzpicture}\n"
       "\\begin{groupplot}[group style={group size=2 by 2,horizontal sep=0.8cm,vertical sep=0.9cm},width=0.46\\textwidth,"
       "height=0.46\\textwidth," + ax + ",title style={font=\\normalsize, at={(0.0,1.0)}, anchor=south west}]\n" + body +
       "\\end{groupplot}\n\\end{tikzpicture}\n")
od = os.path.join(C.ROOT, "export", "paper") if os.path.isdir(os.path.join(C.ROOT, "export")) else os.path.join(os.path.dirname(C.ROOT), "TopoTME", "export", "paper")
os.makedirs(od, exist_ok=True)
open(os.path.join(od, "fig_luad_maps.tex"), "w", encoding="utf-8").write(tex)
print("bytes", len(tex), od)
