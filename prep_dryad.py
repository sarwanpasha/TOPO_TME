"""Zheng & Gevaert 2025 (Science Advances) NSCLC multiplex immunofluorescence cohort (Dryad doi:10.5061/dryad.b8gtht7p4, CC0):
mIF_data.h5ad (cells x 33 markers, centroids in um, phenotype, core_id, patient_id) and deident_clinical.csv.
Endpoints: histology (LUSC vs LUAD), response to immune checkpoint therapy (among treated patients), overall survival.
Writes data/dryad/* with the same pipeline as the other cohorts."""
import os
import numpy as np, pandas as pd
import anndata as ad
import preprocess as PP

RAW = os.path.join(PP.RAW, "dryad_nsclc")


def main():
    a = ad.read_h5ad(os.path.join(RAW, "mIF_data.h5ad"))
    o = a.obs
    cx = [c for c in o.columns if c.startswith("Centroid X")][0]; cy = [c for c in o.columns if c.startswith("Centroid Y")][0]
    ok = (o["phenotype"].astype(str) != "Unclass").values & o["patient_id"].notna().values
    X = np.asarray(a.X[ok], dtype=np.float32)
    Mz, _ = PP.norm_markers(X)
    o = o[ok].reset_index(drop=True)
    T = o["phenotype"].astype(str).values
    types = sorted(set(T)); tcode = pd.Series(T).map({t: i for i, t in enumerate(types)}).values
    df = pd.DataFrame({"img": "dryad_" + o["core_id"].astype(str), "pid": o["patient_id"].astype(str),
                       "x": o[cx].astype(float).values, "y": o[cy].astype(float).values})
    images = PP.build_images(df, "img", "pid", "x", "y", tcode, Mz, len(types))
    c = pd.read_csv(os.path.join(RAW, "deident_clinical.csv"))
    col = lambda s: [k for k in c.columns if k.startswith(s)][0]
    num = lambda s: pd.to_numeric(c[col(s)], errors="coerce")
    hist = num("Histology"); io = num("IO_received"); resp = num("IO_response")
    stage = c[col("Stage")].astype(str).str.extract(r"(\d)")[0].astype(float)
    pat = pd.DataFrame({"patient_id": c["patient_id"].astype(str),
                        "os_time": num("Survival"), "os_event": num("Deceased"),
                        "lusc": hist.where(hist.isin([0, 1])),
                        "io_response": resp.where(io == 1),
                        "age70": num("Age"), "sex": num("Gender"), "stage": stage, "smoking": num("Smoking")})
    pat = pat[pat["patient_id"].isin(set(df["pid"]))].reset_index(drop=True)
    meta = dict(types=types, markers=list(a.var_names), covariates=["age70", "sex", "stage", "smoking"],
                source="Zheng & Gevaert 2025 Sci Adv; Dryad doi:10.5061/dryad.b8gtht7p4 (CC0)")
    PP.save_cohort("dryad", images, pat, meta)
    print(pat.describe().T.round(3).to_string(), flush=True)
    print("images", len(images), "patients", pat.shape[0], "lusc", pat["lusc"].value_counts().to_dict(),
          "io", pat["io_response"].value_counts().to_dict(), "deaths", pat["os_event"].sum(), flush=True)


if __name__ == "__main__":
    main()
