"""Cords et al. 2024 (Cancer Cell) NSCLC IMC cohort (Zenodo 10.5281/zenodo.7961844, CC-BY-4.0;
sce_all_annotated.rds exported with export_sce2.R): 1,071 patients, 2,072 cores, 5.98 million cells, 22 cell types.
Endpoints: overall survival (OS, Ev.O), disease-free survival (DFS, Relapse), lymph-node metastasis, grade, histology.
Writes data/cords/* with the same pipeline as the other cohorts."""
import os
import numpy as np, pandas as pd
import preprocess as PP

RAW = os.path.join(PP.RAW, "cords", "flat")
DROP = ("Histone", "Iridium")
USE = ["Center_X", "Center_Y", "Tma_ac", "Patient_ID", "cell_type", "DX.name", "Age", "Gender", "Grade", "Stage", "Smok",
       "Relapse", "DFS", "Ev.O", "OS", "LN.Met", "Dist.Met", "NeoAdj", "T.new", "N"]


def main():
    cells = pd.read_csv(os.path.join(RAW, "cells.csv"), usecols=USE, low_memory=False)
    markers = [l.strip() for l in open(os.path.join(RAW, "markers.txt"), encoding="utf-8", errors="replace") if l.strip()]
    M = np.memmap(os.path.join(RAW, "expr.f32"), dtype=np.float32, mode="r").reshape(len(cells), len(markers))
    ok = (cells["DX.name"].astype(str) != "Control").values & cells["Patient_ID"].notna().values & cells["cell_type"].notna().values
    keep = [i for i, m in enumerate(markers) if not any(d in m for d in DROP)]
    Mz, _ = PP.norm_markers(np.asarray(M[ok][:, keep]))
    cells = cells[ok].reset_index(drop=True)
    T = cells["cell_type"].astype(str).values
    types = sorted(set(T)); tcode = pd.Series(T).map({t: i for i, t in enumerate(types)}).values
    df = pd.DataFrame({"img": "cords_" + cells["Tma_ac"].astype(str), "pid": cells["Patient_ID"].astype(str),
                       "x": cells["Center_X"].values, "y": cells["Center_Y"].values})
    images = PP.build_images(df, "img", "pid", "x", "y", tcode, Mz, len(types))
    p = cells.drop_duplicates("Patient_ID"); num = lambda c: pd.to_numeric(p[c], errors="coerce")
    dx = p["DX.name"].astype(str)
    pat = pd.DataFrame({
        "patient_id": p["Patient_ID"].astype(str),
        "os_time": num("OS"), "os_event": num("Ev.O"), "dfs_time": num("DFS"), "dfs_event": num("Relapse"),
        "relapse": num("Relapse"),
        "lnmet": p["LN.Met"].map({"LN Metastases": 1.0, "No LN Metastases": 0.0}),
        "grade3": (num("Grade") == 3).astype(float).where(num("Grade").notna()),
        "squamous": dx.map({"Squamous cell carcinoma": 1.0, "Adenocarcinoma": 0.0}),
        "age": num("Age"), "sex": num("Gender"), "grade": num("Grade"), "stage": num("Stage"), "smoking": num("Smok"),
        "sqc": (dx == "Squamous cell carcinoma").astype(float), "neoadj": (p["NeoAdj"].astype(str) == "NeoAdjuvantTherapy").astype(float)})
    meta = dict(types=types, markers=[markers[i] for i in keep], covariates=["age", "sex", "grade", "stage", "smoking", "sqc"],
                source="Cords et al. 2024 Cancer Cell; Zenodo 10.5281/zenodo.7961844 (CC-BY-4.0)")
    PP.save_cohort("cords", images, pat, meta)
    print(pat.describe().T.round(3).to_string(), flush=True)
    print(dx.value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
