"""Sorin et al. 2023 LUAD IMC cohort (Zenodo 7760826, CC-BY-4.0): 416 patients, 1 core each, OS + clinical.
Cells: nucleus boundaries (MATLAB 1-based column-major linear indices) + cell-type labels per core.
Markers: mean of each of the 18 channels (8-bit masked tif) in a 3-px disk around the nucleus centroid.
Writes data/luad/{images.pkl,feats.npz,patients.csv,meta.json} with the same pipeline as the other cohorts."""
import os, glob, re, sys, json
import numpy as np, pandas as pd
import scipy.io as sio
from PIL import Image
import preprocess as PP

RAW = os.path.join(PP.RAW, "luad")
CHANNELS = ["CD117", "CD11c", "CD14", "CD163", "CD16", "CD20", "CD31", "CD3", "CD4", "CD68", "CD8a", "CD94",
            "DNA1", "FoxP3", "HLA-DR", "MPO", "Pancytokeratin", "TTF1"]


def load_tif(p):
    im = Image.open(p); fr = []
    for k in range(im.n_frames):
        im.seek(k); fr.append(np.asarray(im, dtype=np.float32))
    return np.stack(fr, -1)          # H x W x C


def core(p_mat, p_tif, r=3):
    m = sio.loadmat(p_mat, squeeze_me=True, struct_as_record=False)
    B = m["Boundaries"]; T = np.array([t if isinstance(t, str) else (str(np.ravel(t)[0]) if np.size(t) else "Unassigned") for t in m["cellTypes"]])
    I = load_tif(p_tif); H, W, Cn = I.shape
    xy = np.zeros((len(B), 2)); ok = np.ones(len(B), bool)
    for i, b in enumerate(B):
        b = np.atleast_1d(np.asarray(b, np.int64)) - 1
        if b.size == 0:
            ok[i] = False; continue
        rows, cols = b % H, b // H
        xy[i] = cols.mean(), rows.mean()
    ok &= T != "Unassigned"
    # disk-mean marker intensities
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; disk = (yy ** 2 + xx ** 2) <= r * r
    dy, dx = yy[disk], xx[disk]
    ci = np.clip(np.round(xy[:, 1]).astype(int)[:, None] + dy[None], 0, H - 1)
    cj = np.clip(np.round(xy[:, 0]).astype(int)[:, None] + dx[None], 0, W - 1)
    M = I[ci, cj].mean(1)            # cells x C
    return xy[ok], T[ok], M[ok], (H, W, Cn)


def main():
    mats = sorted(glob.glob(os.path.join(RAW, "LUAD_IMC_CellType", "*.mat")))
    names = [os.path.splitext(os.path.basename(p))[0] for p in mats]
    print("cores", len(mats), "e.g.", names[:3], names[-3:], flush=True)
    rows = []; XY = []; TT = []; MM = []; IMG = []
    for j, (n, p) in enumerate(zip(names, mats)):
        t = os.path.join(RAW, "LUAD_IMC_MaskTif", n + ".tif")
        if not os.path.exists(t):
            continue
        try:
            xy, T, M, shp = core(p, t)
        except Exception as e:
            print("skip", n, e, flush=True); continue
        if shp[2] != len(CHANNELS):
            print("channel mismatch", n, shp, flush=True)
        XY.append(xy); TT.append(T); MM.append(M[:, :len(CHANNELS)]); IMG.append(np.full(len(T), n))
        if j % 50 == 0:
            print("core", j, n, len(T), shp, flush=True)
    xy = np.concatenate(XY); T = np.concatenate(TT); M = np.concatenate(MM); img = np.concatenate(IMG)
    keep = [i for i, c in enumerate(CHANNELS) if c != "DNA1"]
    Mz, _ = PP.norm_markers(M[:, keep])
    types = sorted(set(T)); tcode = pd.Series(T).map({t: i for i, t in enumerate(types)}).values
    print("cells", len(T), "types", types, flush=True)
    df = pd.DataFrame({"img": img, "pid": img, "x": xy[:, 0], "y": xy[:, 1]})
    images = PP.build_images(df, "img", "pid", "x", "y", tcode, Mz, len(types))
    cl = pd.read_excel(os.path.join(RAW, "LUAD Clinical Data.xlsx"))
    col = lambda pat: [c for c in cl.columns if re.search(pat, str(c), re.I)][0]
    pat = pd.DataFrame({
        "patient_id": cl[col(r"^Key")].astype(str),
        "os_time": pd.to_numeric(cl[col(r"Survival or loss")], errors="coerce") * 365.25,
        "os_event": pd.to_numeric(cl[col(r"^Death")], errors="coerce"),
        "progression": pd.to_numeric(cl[col(r"^Progression")], errors="coerce"),
        "stage34": pd.to_numeric(cl[col(r"^Stage")], errors="coerce"),
        "solid": (pd.to_numeric(cl[col(r"histological pattern")], errors="coerce") == 5).astype(float),
        "sex": pd.to_numeric(cl[col(r"^Sex")], errors="coerce"),
        "age": pd.to_numeric(cl[col(r"^Age")], errors="coerce"),
        "bmi": pd.to_numeric(cl[col(r"^BMI")], errors="coerce"),
        "smoking": pd.to_numeric(cl[col(r"^Smoking")], errors="coerce"),
        "stage": pd.to_numeric(cl[col(r"^Stage")], errors="coerce"),
        "pattern": pd.to_numeric(cl[col(r"histological pattern")], errors="coerce")})
    meta = dict(types=types, markers=[CHANNELS[i] for i in keep],
                covariates=["sex", "age", "bmi", "smoking", "stage", "pattern"],
                source="Sorin et al. 2023 Nature, Zenodo 10.5281/zenodo.7760826 (CC-BY-4.0)")
    PP.save_cohort("luad", images, pat, meta)


if __name__ == "__main__":
    main()
