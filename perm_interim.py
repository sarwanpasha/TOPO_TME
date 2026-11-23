import os, glob, numpy as np, common as C
from scipy.stats import chi2
OBS = {"danenberg_OS_clinical": chi2.isf(0.017, 1), "cords_OS_clinical": 7.442009200727625}
for k, lr in OBS.items():
    v = np.array([float(open(f).read()) for f in glob.glob(os.path.join(C.RES, "interp", "permck", k, "*.txt"))])
    print(k, "n=%d obsLR=%.2f nullmean=%.2f frac>3.84=%.3f frac>=obs=%.3f p_interim=%.3f" % (
        len(v), lr, v.mean(), (v > 3.84).mean(), (v >= lr).mean(), (1 + (v >= lr).sum()) / (1 + len(v))))
