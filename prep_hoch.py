"""Hoch, Schulz et al. 2022 (Science Immunology) metastatic melanoma IMC cohort, protein panel
(Zenodo 10.5281/zenodo.20623014, sce_full_protein.rds exported with export_sce.R): 69 patients, 167 images.
Endpoints (not measured by the antibody panel): response to immune checkpoint inhibition at 3 months (R vs NR),
relapse, death. Writes data/hoch/* with the same pipeline as the other cohorts."""
import os
import numpy as np, pandas as pd
import preprocess as PP

RAW = os.path.join(PP.RAW, "hoch", "flat")
DROP = {"HistoneH3", "DNA1", "DNA2", "Ki67Pt198"}


def main():
    cells = pd.read_csv(os.path.join(RAW, "cells.csv"), low_memory=False)
    markers = open(os.path.join(RAW, "markers.txt")).read().split()
    M = np.fromfile(os.path.join(RAW, "expr.f32"), dtype=np.float32).reshape(len(cells), len(markers))
    ok = (cells["sample_location"].astype(str) != "CTRL").values & cells["patient_id"].notna().values
    cells, M = cells[ok].reset_index(drop=True), M[ok]
    keep = [i for i, m in enumerate(markers) if m not in DROP]
    Mz, _ = PP.norm_markers(M[:, keep])
    T = cells["cell_type"].fillna("unknown").astype(str).values
    types = sorted(set(T)); tcode = pd.Series(T).map({t: i for i, t in enumerate(types)}).values
    df = pd.DataFrame({"img": "hoch_" + cells["image_number"].astype(str), "pid": cells["patient_id"].astype(str),
                       "x": cells["cell_x"].values, "y": cells["cell_y"].values})
    images = PP.build_images(df, "img", "pid", "x", "y", tcode, Mz, len(types))
    p = cells.drop_duplicates("patient_id")
    pat = pd.DataFrame({
        "patient_id": p["patient_id"].astype(str),
        "response": p["patient_status_at_3months"].map({"R": 1.0, "NR": 0.0}),
        "relapse": p["patient_relapse"].map({"relapse": 1.0, "no relapse": 0.0}),
        "death": p["patient_death"].map({"yes": 1.0, "no": 0.0}),
        "stage4": p["patient_cancer_stage"].map({"IV": 1.0, "III": 0.0}),
        "age60": p["patient_age"].astype(str).str.contains("60").astype(float),
        "male": p["patient_gender"].map({"m": 1.0, "f": 0.0})})
    meta = dict(types=types, markers=[markers[i] for i in keep], covariates=["stage4", "age60", "male"],
                source="Hoch, Schulz et al. 2022 Science Immunology; Zenodo 10.5281/zenodo.20623014")
    PP.save_cohort("hoch", images, pat, meta)
    print(pat.describe().T.to_string(), flush=True)


if __name__ == "__main__":
    main()
