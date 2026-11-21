"""Build per-cohort caches: images.pkl, patients.csv, meta.json, feats.npz.

Each image dict: image_id, patient_id, coords (N,2, scaled so median Delaunay
edge = 1), types (N,), markers (N,m) z-scored, edges, tri_edges, faces,
knn_idx (N,k), knn_d (N,k).  Usage: python preprocess.py <cohort>
cohorts: danenberg | jackson (writes basel + zurich) | codex
"""
import os, sys, json, pickle, re, glob, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from common import RAW, DATA
from topo_core import build_complex, persistence_pairs

KNN = 8
MIN_CELLS = 50


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def norm_markers(M, ref=None):
    M = np.asarray(M, dtype=np.float64)
    M = np.where(np.isfinite(M), M, 0.0)
    if ref is None:
        q = np.percentile(M, 99, axis=0)
        cof = np.where(q > 20, 5.0, 0.0)
        ref = {"cof": cof}
    cof = ref["cof"]
    X = np.where(cof > 0, np.arcsinh(M / np.maximum(cof, 1e-9)), M)
    if "mu" not in ref:
        ref["mu"] = X.mean(0); ref["sd"] = X.std(0) + 1e-6
    X = np.clip((X - ref["mu"]) / ref["sd"], -5, 5)
    return X.astype(np.float16), ref


def pi_feats(b, d, G=6, lim=2.5, sig=0.3):
    """fixed persistence image in (log1p birth, log persistence) coords."""
    lb = np.log1p(b); lp = np.log1p(d) - np.log1p(b)
    c = np.linspace(0, lim, G)
    if len(lb) == 0:
        return np.zeros(G * G + 2)
    wb = np.exp(-(lb[:, None] - c[None]) ** 2 / (2 * sig ** 2))
    wp = np.exp(-(lp[:, None] - c[None]) ** 2 / (2 * sig ** 2))
    img = (lp[:, None, None] * wb[:, :, None] * wp[:, None, :]).sum(0).ravel()
    return np.concatenate([img, [lp.sum(), (lp > 0.2).sum()]])


def fixed_topo(coords, N_norm):
    """H0+H1 persistence-image of unweighted Delaunay filtration on given cells."""
    if len(coords) < 4:
        return np.zeros(2 * (6 * 6 + 2))
    try:
        E, TE, F = build_complex(coords)
    except Exception:
        return np.zeros(2 * (6 * 6 + 2))
    fe = np.linalg.norm(coords[E[:, 0]] - coords[E[:, 1]], axis=1)
    fv = np.zeros(len(coords))
    h0b, h0d, h1b, h1d = persistence_pairs(fv, fe, E, TE, F)
    f0 = pi_feats(fv[h0b], fe[h0d]); f1 = pi_feats(fe[h1b], fe[h1d])
    return np.concatenate([f0, f1]) / N_norm


def finish_image(img_id, pid, xy, types, markers, n_types):
    xy = np.asarray(xy, np.float64)
    # dedupe identical coordinates
    xy = xy + np.random.default_rng(0).normal(scale=1e-3, size=xy.shape)
    E, TE, F = build_complex(xy)
    el = np.linalg.norm(xy[E[:, 0]] - xy[E[:, 1]], axis=1)
    tree = cKDTree(xy)
    kd, ki = tree.query(xy, k=min(KNN + 1, len(xy)))
    return dict(image_id=str(img_id), patient_id=str(pid), coords=xy.astype(np.float32),
                types=np.asarray(types, np.int16), markers=markers,
                edges=E, tri_edges=TE, faces=F, knn_idx=ki[:, 1:].astype(np.int32),
                knn_d=kd[:, 1:].astype(np.float32), med_edge=float(np.median(el)))


def image_features(im, n_types):
    """hand-crafted features for baselines (per image)."""
    t = im["types"].astype(int); N = len(t)
    comp = np.bincount(t, minlength=n_types) / N
    mm = im["markers"].astype(np.float32).mean(0)
    E = im["edges"]
    a, b = t[E[:, 0]], t[E[:, 1]]
    C = np.zeros((n_types, n_types))
    np.add.at(C, (a, b), 1); np.add.at(C, (b, a), 1)
    C = C / max(C.sum(), 1)
    expct = np.outer(comp, comp) + 1e-6
    iu = np.triu_indices(n_types)
    contact = np.log((C + 1e-4) / expct)[iu] * (expct[iu] > 1e-5)
    xy = im["coords"].astype(np.float64) / im["scale"]
    topo_all = fixed_topo(xy, N)
    topo_t = [fixed_topo(xy[t == k], N) if (t == k).sum() >= 4 else np.zeros(76) for k in range(n_types)]
    return dict(comp=comp, mm=mm, logn=np.array([np.log(N)]), contact=contact,
                topo_all=topo_all, topo_type=np.concatenate(topo_t))


def save_cohort(name, images, pat, meta):
    d = os.path.join(DATA, name); os.makedirs(d, exist_ok=True)
    n_types = len(meta["types"])
    scale = float(np.median([im["med_edge"] for im in images]))
    meta["scale"] = scale
    for im in images:
        im["scale"] = scale
    feats = {}
    for i, im in enumerate(images):
        f = image_features(im, n_types)
        for k, v in f.items():
            feats.setdefault(k, []).append(v)
        if i % 200 == 0:
            log(name, "feat", i, len(images))
    feats = {k: np.stack(v).astype(np.float32) for k, v in feats.items()}
    feats["image_id"] = np.array([im["image_id"] for im in images])
    feats["patient_id"] = np.array([im["patient_id"] for im in images])
    np.savez_compressed(os.path.join(d, "feats.npz"), **feats)
    with open(os.path.join(d, "images.pkl"), "wb") as f:
        pickle.dump(images, f, protocol=4)
    pat = pat[pat["patient_id"].isin({im["patient_id"] for im in images})]
    pat.to_csv(os.path.join(d, "patients.csv"), index=False)
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    log(name, "saved", len(images), "images", len(pat), "patients", "types", n_types,
        "markers", len(meta["markers"]), "scale", scale)
    for c in pat.columns:
        if c != "patient_id":
            log("  ", c, pat[c].describe().to_dict() if pat[c].dtype != object else pat[c].value_counts().to_dict())


def build_images(df, img_col, pid_col, x_col, y_col, type_codes, M, n_types):
    images = []
    groups = df.groupby(img_col, sort=False).indices
    for j, (img, idx) in enumerate(groups.items()):
        if len(idx) < MIN_CELLS:
            continue
        pid = df[pid_col].values[idx[0]]
        xy = np.stack([df[x_col].values[idx], df[y_col].values[idx]], 1)
        try:
            images.append(finish_image(img, pid, xy, type_codes[idx], M[idx], n_types))
        except Exception as e:
            log("skip image", img, e)
        if j % 100 == 0:
            log("complex", j, len(groups))
    return images


# --------------------------------------------------------------- Danenberg
def prep_danenberg():
    f = glob.glob(os.path.join(RAW, "danenberg", "**", "SingleCells.csv"), recursive=True)[0]
    cells = pd.read_csv(f)
    cols = cells.columns.tolist(); log("danenberg cols", cols)
    xc = [c for c in cols if re.search(r"Location_Center_X|^x$", c)][0]
    yc = [c for c in cols if re.search(r"Location_Center_Y|^y$", c)][0]
    tc = "cellPhenotype" if "cellPhenotype" in cols else "phenotype"
    skip = re.compile(r"^(ImageNumber|ObjectNumber|metabric_id|cellPhenotype|phenotype)$|^is_|Location_|AreaShape|^DNA|Histone")
    names = [c for c in cols if not skip.search(c) and pd.api.types.is_numeric_dtype(cells[c])]
    log("danenberg markers", names)
    Mz, _ = norm_markers(cells[names].values)
    keep = list(range(len(names)))
    types = sorted(cells[tc].astype(str).unique())
    tcode = cells[tc].astype(str).map({t: i for i, t in enumerate(types)}).values
    images = build_images(cells, "ImageNumber", "metabric_id", xc, yc, tcode, Mz, len(types))
    # clinical
    cb = pd.read_csv(glob.glob(os.path.join(RAW, "cbioportal", "**", "data_clinical_patient.txt"), recursive=True)[0], sep="\t", comment="#")
    cs = pd.read_csv(glob.glob(os.path.join(RAW, "cbioportal", "**", "data_clinical_sample.txt"), recursive=True)[0], sep="\t", comment="#")
    log("cbio patient cols", cb.columns.tolist()); log("cbio sample cols", cs.columns.tolist())
    cs = cs.drop_duplicates("PATIENT_ID")
    cl = cb.merge(cs, on="PATIENT_ID", how="left", suffixes=("", "_s"))
    def pos(series):
        s = series.astype(str).str.lower()
        return np.where(s.str.startswith("pos"), 1.0, np.where(s.str.startswith("neg"), 0.0, np.nan))
    pat = pd.DataFrame({"patient_id": cl["PATIENT_ID"].astype(str)})
    pat["ER"] = pos(cl["ER_STATUS"]) if "ER_STATUS" in cl else pos(cl["ER_IHC"])
    pat["HER2"] = pos(cl["HER2_STATUS"]) if "HER2_STATUS" in cl else np.nan
    pat["PR"] = pos(cl["PR_STATUS"]) if "PR_STATUS" in cl else np.nan
    cs_ = cl.get("CLAUDIN_SUBTYPE")
    if cs_ is not None:
        s = cs_.astype(str)
        pat["basal"] = np.where(s.isin(["nan", "NC", "None"]), np.nan, (s == "Basal").astype(float))
    if "GRADE" in cl:
        g = pd.to_numeric(cl["GRADE"], errors="coerce")
        pat["grade3"] = np.where(g.isna(), np.nan, (g == 3).astype(float))
    pat["os_time"] = pd.to_numeric(cl.get("OS_MONTHS"), errors="coerce")
    pat["os_event"] = cl.get("OS_STATUS").astype(str).str.contains(r"^1|^DECEASED", case=False).astype(float)
    pat.loc[cl["OS_STATUS"].isna().values, "os_event"] = np.nan
    vs = cl.get("VITAL_STATUS").astype(str)
    pat["dss_event"] = np.where(vs == "nan", np.nan, (vs == "Died of Disease").astype(float))
    pat["rfs_time"] = pd.to_numeric(cl.get("RFS_MONTHS"), errors="coerce")
    pat["rfs_event"] = cl.get("RFS_STATUS").astype(str).str.contains(r"^1|^Recurred", case=False).astype(float)
    pat.loc[cl["RFS_STATUS"].isna().values, "rfs_event"] = np.nan
    # covariates
    pat["age"] = pd.to_numeric(cl.get("AGE_AT_DIAGNOSIS"), errors="coerce")
    pat["grade"] = pd.to_numeric(cl.get("GRADE"), errors="coerce")
    pat["size"] = pd.to_numeric(cl.get("TUMOR_SIZE"), errors="coerce")
    pat["nodes"] = pd.to_numeric(cl.get("LYMPH_NODES_EXAMINED_POSITIVE"), errors="coerce")
    meta = dict(types=types, markers=[names[i] for i in keep],
                covariates=["age", "grade", "size", "nodes", "ER", "HER2"])
    save_cohort("danenberg", images, pat, meta)


# --------------------------------------------------------------- Jackson (Basel + Zurich)
JUNK = re.compile(r"(Rutheni|Iridium|Carboni|\d+Di$|^Area$|Eccentricity|Solidity|Extent|EulerNumber|Perimeter|"
                  r"AxisLength|Orientation|Percent_Touching|Number_Neighbors)")


def jackson_meta(pm, site):
    log(site, "meta cols", pm.columns.tolist())
    pat = pd.DataFrame({"patient_id": pm["PID"].astype(str)})
    def st(col, alt=None):
        out = np.full(len(pm), np.nan)
        if col in pm:
            s = pm[col].astype(str).str.lower().str.strip()
            out = np.where(s.str.startswith("pos"), 1.0, np.where(s.str.startswith("neg"), 0.0, np.nan))
        if alt is not None and alt in pm:
            s = pm[alt].astype(str).str.strip()
            a = np.where(s == "+", 1.0, np.where(s == "-", 0.0, np.nan))
            out = np.where(np.isnan(out), a, out)
        return out
    pat["ER"] = st("ERStatus", "ER"); pat["PR"] = st("PRStatus", "PR"); pat["HER2"] = st("HER2Status", "HER2")
    ct = pm.get("clinical_type")
    if ct is not None:
        s = ct.astype(str)
        pat["TNBC"] = np.where(s.isin(["nan", "", "None"]), np.nan, (s == "TripleNeg").astype(float))
    g = pd.to_numeric(pm.get("grade"), errors="coerce")
    pat["grade3"] = np.where(g.isna(), np.nan, (g == 3).astype(float))
    if "OSmonth" in pm and site == "basel":
        pat["os_time"] = pd.to_numeric(pm["OSmonth"], errors="coerce")
        ps = pm["Patientstatus"].astype(str).str.lower()
        pat["os_event"] = np.where(ps == "nan", np.nan, ps.str.startswith("death").astype(float))
        pat["dss_event"] = np.where(ps == "nan", np.nan, ps.str.contains("primary").astype(float))
        pat["dfs_time"] = pd.to_numeric(pm["DFSmonth"], errors="coerce")
        # DFS event: recurrence or death (DFS < OS or died)
        pat["dfs_event"] = np.where(pat["dfs_time"].isna(), np.nan,
                                    ((pat["dfs_time"] < pat["os_time"] - 1) | (pat["os_event"] == 1)).astype(float))
    pat["age"] = pd.to_numeric(pm.get("age"), errors="coerce")
    pat["grade"] = g
    pat["size"] = pd.to_numeric(pm.get("tumor_size"), errors="coerce")
    pat = pat.groupby("patient_id", as_index=False).first()
    return pat


def jackson_raw(site):
    J = os.path.join(RAW, "jackson")
    sub = {"basel": "BaselTMA", "zurich": "ZurichTMA"}[site]
    sc = glob.glob(os.path.join(J, "**", sub, "SC_dat.csv"), recursive=True)[0]
    log(site, "reading", sc)
    locf = glob.glob(os.path.join(J, "**", ("Basel" if site == "basel" else "Zurich") + "_SC_locations.csv"), recursive=True)[0]
    ids = pd.read_csv(locf, usecols=["id"])["id"].astype(str).values
    pos0 = pd.Series(np.arange(len(ids)), index=ids)
    pos0 = pos0[~pos0.index.duplicated()]
    chan_idx = {}; M = np.zeros((len(ids), 80), np.float32); seen = np.zeros(len(ids), bool)
    # long-format file is several GB: stream it in chunks (keeps memory low on a laptop)
    for ch in pd.read_csv(sc, usecols=["id", "channel", "mc_counts"], dtype={"id": str, "channel": str}, chunksize=4_000_000):
        for c in ch["channel"].unique():
            chan_idx.setdefault(c, len(chan_idx))
        r = pos0.reindex(ch["id"].values).values
        ok = ~np.isnan(r)
        M[r[ok].astype(np.int64), ch["channel"].map(chan_idx).values[ok]] = ch["mc_counts"].values[ok]
        seen[r[ok].astype(np.int64)] = True
    chans = sorted(chan_idx)                      # same (sorted) order as the categorical version
    M = M[:, [chan_idx[c] for c in chans]]
    names = [c.split(" ", 1)[1] if " " in c else c for c in chans]
    loc = pd.read_csv(glob.glob(os.path.join(J, "**", ("Basel" if site == "basel" else "Zurich") + "_SC_locations.csv"), recursive=True)[0])
    lab = glob.glob(os.path.join(J, "**", "Basel_metaclusters.csv" if site == "basel" else "Zurich_matched_metaclusters.csv"), recursive=True)[0]
    cl = pd.read_csv(lab)
    ann = pd.read_csv(glob.glob(os.path.join(J, "**", "Metacluster_annotations.csv"), recursive=True)[0], sep=";")
    ann.columns = [c.strip() for c in ann.columns]
    mc2type = dict(zip(ann["Metacluster"].astype(int), ann["Cell type"].astype(str).str.strip().replace({"Macrohage": "Macrophage"})))
    cells = loc[["id", "core", "Location_Center_X", "Location_Center_Y"]].merge(cl[["id", "cluster"]], on="id")
    cells["cell_type"] = cells["cluster"].astype(int).map(mc2type)
    pos = pos0[seen[pos0.values]]
    cells = cells[cells["id"].isin(pos.index) & cells["cell_type"].notna()].reset_index(drop=True)
    Mc = M[pos.loc[cells["id"]].values]
    meta = glob.glob(os.path.join(J, "**", "Basel_PatientMetadata.csv" if site == "basel" else "Zuri_PatientMetadata.csv"), recursive=True)[0]
    pm = pd.read_csv(meta)
    log(site, "cells", len(cells), "channels", len(chans), "types", cells["cell_type"].nunique())
    return cells, Mc, chans, names, pm


def prep_jackson():
    data = {s: jackson_raw(s) for s in ["basel", "zurich"]}
    chans = data["basel"][2]; names = data["basel"][3]
    keep = [i for i, n in enumerate(names) if not JUNK.search(n) and not JUNK.search(chans[i])]
    log("jackson kept markers", [names[i] for i in keep])
    types = sorted(set(data["basel"][0]["cell_type"]) | set(data["zurich"][0]["cell_type"]))
    ref = None
    for site in ["basel", "zurich"]:
        cells, M, ch, nm, pm = data[site]
        idx = [ch.index(chans[i]) for i in keep]   # align panel by channel name
        Mz, ref = norm_markers(M[:, idx], ref)    # fitted on Basel, applied to Zurich
        tcode = cells["cell_type"].map({t: i for i, t in enumerate(types)}).values
        core2pid = dict(zip(pm["core"].astype(str), pm["PID"].astype(str)))
        cells["pid"] = cells["core"].astype(str).map(core2pid)
        ok = cells["pid"].notna().values
        log(site, "cells with patient", ok.mean())
        cells = cells[ok].reset_index(drop=True)
        images = build_images(cells, "core", "pid", "Location_Center_X", "Location_Center_Y", tcode[ok], Mz[ok], len(types))
        pat = jackson_meta(pm, site)
        meta = dict(types=types, markers=[names[i] for i in keep], covariates=["age", "grade", "size", "ER", "HER2"])
        save_cohort(site, images, pat, meta)


# --------------------------------------------------------------- Schürch CODEX CRC
def prep_codex():
    f = glob.glob(os.path.join(RAW, "codex", "**", "CRC_clusters_neighborhoods_markers*.csv"), recursive=True)[0]
    df = pd.read_csv(f)
    cols = df.columns.tolist(); log("codex cols", cols)
    xc = [c for c in cols if re.fullmatch(r"X(:X)?", c)][0]
    yc = [c for c in cols if re.fullmatch(r"Y(:Y)?", c)][0]
    mk = [c for c in cols if ":Cyc_" in c and not re.search(r"DRAQ5|HOECHST|empty|blank", c, re.I)]
    log("codex markers", len(mk), "x/y", xc, yc)
    M = df[mk].values
    Mz, _ = norm_markers(M)
    types = sorted(df["ClusterName"].astype(str).unique())
    tcode = df["ClusterName"].astype(str).map({t: i for i, t in enumerate(types)}).values
    df["pid"] = df["patients"].astype(int).astype(str)
    images = build_images(df, "File Name", "pid", xc, yc, tcode, Mz, len(types))
    g = df.groupby("pid")["groups"].first()
    pat = pd.DataFrame({"patient_id": g.index.astype(str), "DII": (g.values == 2).astype(float)})
    try:
        x = pd.read_excel(glob.glob(os.path.join(RAW, "codex", "**", "*patient_annotations*.xlsx"), recursive=True)[0], sheet_name=None)
        for k, v in x.items():
            log("xlsx sheet", k, v.shape, v.columns.tolist()); log(v.head(3).to_string())
        ann = list(x.values())[0]
        pc = [c for c in ann.columns if re.search(r"patient", str(c), re.I)]
        osc = [c for c in ann.columns if re.fullmatch(r"\s*OS\s*(\(.*\))?\s*(months)?\s*", str(c), re.I)]
        evc = [c for c in ann.columns if re.search(r"OS.*(cens|event|status)|death|died", str(c), re.I)]
        if pc and osc and evc:
            a = pd.DataFrame({"patient_id": pd.to_numeric(ann[pc[0]], errors="coerce").astype("Int64").astype(str),
                              "os_time": pd.to_numeric(ann[osc[0]], errors="coerce"),
                              "os_event": pd.to_numeric(ann[evc[0]], errors="coerce")})
            log("codex survival from", pc[0], osc[0], evc[0], a["os_event"].value_counts().to_dict())
            if a["os_event"].isin([0, 1]).all() is False:
                pass
            pat = pat.merge(a.groupby("patient_id", as_index=False).first(), on="patient_id", how="left")
        else:
            log("codex: survival columns not found", pc, osc, evc)
    except Exception as e:
        log("codex xlsx failed", e)
    meta = dict(types=types, markers=mk, covariates=[])
    save_cohort("codex", images, pat, meta)


if __name__ == "__main__":
    which = sys.argv[1]
    {"danenberg": prep_danenberg, "jackson": prep_jackson, "codex": prep_codex}[which]()
    log("DONE", which)
