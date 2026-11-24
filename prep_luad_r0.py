"""Artifact control for LUAD: marker intensity read at the nucleus-centroid pixel only (no disk averaging, r = 0),
and the DNA1 channel kept as a negative-control marker. Same cells, types and patients as prep_luad.py.
Writes data/luad_r0/*."""
import os, glob
import numpy as np, pandas as pd
import preprocess as PP
import prep_luad as PL


def main():
    mats = sorted(glob.glob(os.path.join(PL.RAW, "LUAD_IMC_CellType", "*.mat")))
    XY, TT, MM, IMG = [], [], [], []
    for j, p in enumerate(mats):
        n = os.path.splitext(os.path.basename(p))[0]
        t = os.path.join(PL.RAW, "LUAD_IMC_MaskTif", n + ".tif")
        if not os.path.exists(t):
            continue
        try:
            xy, T, M, shp = PL.core(p, t, r=0)
        except Exception as e:
            print("skip", n, e, flush=True); continue
        XY.append(xy); TT.append(T); MM.append(M[:, :len(PL.CHANNELS)]); IMG.append(np.full(len(T), n))
        if j % 100 == 0:
            print("core", j, n, len(T), flush=True)
    xy = np.concatenate(XY); T = np.concatenate(TT); M = np.concatenate(MM); img = np.concatenate(IMG)
    Mz, _ = PP.norm_markers(M)
    types = sorted(set(T)); tcode = pd.Series(T).map({t: i for i, t in enumerate(types)}).values
    df = pd.DataFrame({"img": img, "pid": img, "x": xy[:, 0], "y": xy[:, 1]})
    images = PP.build_images(df, "img", "pid", "x", "y", tcode, Mz, len(types))
    pat = pd.read_csv(os.path.join(PP.DATA, "luad", "patients.csv"), dtype={"patient_id": str})
    meta = dict(types=types, markers=list(PL.CHANNELS), covariates=["sex", "age", "bmi", "smoking", "stage", "pattern"],
                source="Sorin et al. 2023; centroid-pixel intensities (artifact control), DNA1 kept as negative control")
    PP.save_cohort("luad_r0", images, pat, meta)
    print("done", len(images), "images", len(T), "cells", flush=True)


if __name__ == "__main__":
    main()
