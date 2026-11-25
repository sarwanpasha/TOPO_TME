"""Minimum working example on synthetic tissue (no downloads, about 1-2 minutes on a laptop).

Two classes of synthetic tumour cores with identical cell-type composition and identical T-cell spacing:
  y=1: 240 T cells form one long connected channel (a random walk with a fixed step)
  y=0: the same walk cut into 8 separate chains of 30 cells
Composition cannot separate the classes. TopoNull (composition-calibrated persistence) can.

Usage: python example.py [n_images]
"""
import sys, time, warnings
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import synth as S
import feat_null as FN
from preprocess import finish_image

warnings.filterwarnings("ignore")  # composition columns are constant by design in this example
n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rng = np.random.default_rng(0)
K = len(S.TYPES)

# 1. simulate single-cell maps: positions, cell types and 8 protein markers per cell
images, y = [], []
for i in range(n):
    lab = i % 2
    xy, t = S.gen_channel(lab, rng)
    xy = np.clip(xy, 0, S.W)
    mk = S.markers(t, rng)
    images.append(finish_image(f"img{i}", f"p{i}", xy, t, mk, K))  # builds the Delaunay complex
    y.append(lab)
y = np.array(y)
scale = float(np.median([im["med_edge"] for im in images]))      # length unit: median Delaunay edge
for im in images:
    im["scale"] = scale

# 2. composition (fraction of cells of each type) and TopoNull descriptors
comp = np.array([np.bincount(im["types"].astype(int), minlength=K) / len(im["types"]) for im in images])
t0 = time.time()
tex = np.array([FN.compute(im, K) for im in images])               # (n_images, n_descriptors)
names = FN.feature_names([f"m{j}" for j in range(S.NM)], S.TYPES)
print(f"{n} images, {int(np.mean([len(im['types']) for im in images]))} cells per image on average")
print(f"TopoNull: {tex.shape[1]} calibrated descriptors per image in {time.time() - t0:.1f} s")

# 3. composition is matched between the classes
print("\nmean composition (y=0 | y=1):")
for k, ty in enumerate(S.TYPES):
    print(f"  {ty:<11s} {comp[y == 0, k].mean():.3f} | {comp[y == 1, k].mean():.3f}")

# 4. which calibrated descriptors separate the classes
p = np.array([mannwhitneyu(tex[y == 1, j], tex[y == 0, j]).pvalue if tex[:, j].std() > 0 else 1.0
              for j in range(tex.shape[1])])
print("\ntop calibrated descriptors (Mann-Whitney p, mean TEx y=0 | y=1):")
for j in np.argsort(p)[:5]:
    print(f"  {names[j]:<28s} p={p[j]:.1e}  {tex[y == 0, j].mean():+.2f} | {tex[y == 1, j].mean():+.2f}")

# 5. cross-validated prediction: composition alone vs composition + TopoNull
cv = StratifiedKFold(5, shuffle=True, random_state=0)
def auc(X):
    k = min(10, X.shape[1])  # screening happens inside each training fold
    clf = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=k), LogisticRegression(C=0.1, max_iter=2000))
    return roc_auc_score(y, cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1])
print(f"\n5-fold CV AUROC, composition only:       {auc(comp):.2f}")
print(f"5-fold CV AUROC, composition + TopoNull: {auc(np.hstack([comp, tex])):.2f}")
