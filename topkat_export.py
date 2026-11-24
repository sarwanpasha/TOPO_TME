"""Export per-cell tables (PID, id, x, y, type) and patient-level OS + clinical covariates for TopKAT.
At most 400 cells per image (uniform subsample, fixed seed) so that Rips persistence stays tractable.
python topkat_export.py <cohort>  -> raw/topkat/<cohort>_cells.csv, <cohort>_patients.csv, <cohort>_thr.txt"""
import os, sys
import numpy as np, pandas as pd
import common as C

c = sys.argv[1]
imgs, pat, meta = C.load_cohort(c)
out = os.path.join(os.path.dirname(C.RES), "raw", "topkat"); os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
rows = []
cnt = {}
for im in imgs:
    pid = str(im["patient_id"]); k = cnt.get(pid, 0) + 1; cnt[pid] = k
    xy = np.asarray(im["coords"], float); t = np.asarray(im["types"]).astype(int)
    ix = rng.choice(len(t), min(400, len(t)), replace=False)
    rows.append(pd.DataFrame({"PID": pid, "id": k, "x": xy[ix, 0], "y": xy[ix, 1], "type": [meta["types"][j] for j in t[ix]]}))
cells = pd.concat(rows, ignore_index=True)
df, kind = C.task_frame(pat, c, "OS")
cov = [v for v in meta.get("covariates", []) if v in pat.columns]
P = df[["patient_id", "time", "event"]].merge(pat[["patient_id"] + cov], on="patient_id", how="left")
P = P[P["patient_id"].isin(set(cells["PID"]))].dropna()
cells = cells[cells["PID"].isin(set(P["patient_id"]))]
cells.to_csv(os.path.join(out, f"{c}_cells.csv"), index=False)
P.to_csv(os.path.join(out, f"{c}_patients.csv"), index=False)
open(os.path.join(out, f"{c}_thr.txt"), "w").write(str(10 * float(meta["scale"])))
print(c, "patients", len(P), "events", int(P["event"].sum()), "cells", len(cells), "covariates", cov, "thr", 10 * float(meta["scale"]))
