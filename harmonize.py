"""Panel-agnostic, harmonised topology: map each cohort's cell types onto 7 broad classes
(Tumour, T, B, Myeloid, Endothelial, Stroma, Other), then compute composition-calibrated type-level
persistence (TopoNull: indicator fields + cross-type interaction persistence vs a label-permutation null).
The mapping is fixed a priori from the published cell-type definitions.
python harmonize.py <cohort> [workers] -> data/<cohort>/feats_hnull.npz {hnull, image_id}"""
import os, sys, re, json, time
import numpy as np, multiprocessing as mp
import common as C

BROAD = ["Tumour", "T", "B", "Myeloid", "Endothelial", "Stroma", "Other"]
MAP = {
    "danenberg": {"Tumour": ["Basal", "CK8-18^{+} ER^{hi}", "CK8-18^{hi}CXCL12^{hi}", "CK8-18^{hi}ER^{lo}", "CK^{+} CXCL12^{+}",
                             "CK^{lo}ER^{lo}", "CK^{lo}ER^{med}", "CK^{med}ER^{lo}", "ER^{hi}CXCL12^{+}", "Ep CD57^{+}", "Ep Ki67^{+}",
                             "HER2^{+}", "MHC I & II^{hi}", "MHC I^{hi}CD57^{+}", "MHC^{hi}CD15^{+}"],
                  "T": ["CD4^{+} T cells", "CD4^{+} T cells & APCs", "CD8^{+} T cells", "T_{Reg} & T_{Ex}"],
                  "B": ["B cells", "CD38^{+} lymphocytes"],
                  "Myeloid": ["CD15^{+}", "Granulocytes", "Macrophages", "Macrophages & granulocytes"],
                  "Endothelial": ["Endothelial"],
                  "Stroma": ["Fibroblasts", "Fibroblasts FSP1^{+}", "Myofibroblasts", "Myofibroblasts PDPN^{+}"]},
    "basel": {"Tumour": ["Apoptotic", "Basal CK", "CK lo HR hi p53+", "CK lo HR lo", "CK+ HR hi", "CK+ HR lo", "CK+ HR+", "CK7+",
                         "CK7+ CK hi Ecadh hi", "Epithelial low", "Hypoxic", "Myoepithelial", "Proliferative", "p53+ EGFR+"],
              "T": ["T cell", "T & B cells"], "B": ["B cell"], "Myeloid": ["Macrophage"], "Endothelial": ["Endothelial"],
              "Stroma": ["Fibronectin hi", "Large elongated", "SMA hi Vimentin hi", "Small elongated", "Vimentin hi"]},
    "luad": {"Tumour": ["Cancer"], "T": ["T other", "Tc", "Th", "Treg"], "B": ["B cell"],
             "Myeloid": ["Alt MAC", "Cl MAC", "Cl Mo", "DCs cell", "Int Mo", "Mast cell", "Neutrophils", "Non-Cl Mo"],
             "Endothelial": ["Endothelial cell"], "Stroma": []},
    "codex": {"Tumour": ["tumor cells"], "T": ["CD3+ T cells", "CD4+ T cells", "CD4+ T cells CD45RO+", "CD4+ T cells GATA3+", "CD8+ T cells", "Tregs"],
              "B": ["B cells", "plasma cells"],
              "Myeloid": ["CD11b+ monocytes", "CD11b+CD68+ macrophages", "CD11c+ DCs", "CD163+ macrophages", "CD68+ macrophages",
                          "CD68+ macrophages GzmB+", "CD68+CD163+ macrophages", "granulocytes"],
              "Endothelial": ["vasculature", "lymphatics"], "Stroma": ["stroma", "smooth muscle"]},
}
MAP["zurich"] = MAP["basel"]


def broad_index(types, cohort):
    lut = {t: BROAD.index("Other") for t in types}
    for b, lst in MAP[cohort].items():
        for t in lst:
            if t in lut:
                lut[t] = BROAD.index(b)
    return np.array([lut[t] for t in types])


def one(arg):
    im, bidx = arg
    import feat_null as FN
    t = bidx[np.asarray(im["types"]).astype(int)]
    rng = np.random.default_rng(int(np.abs(im["coords"][:20]).sum() * 1000) % (2 ** 32))
    obs = FN._type_part(im, t, len(BROAD))
    nul = np.stack([FN._type_part(im, rng.permutation(t), len(BROAD)) for _ in range(FN.K_NULL)])
    mu, sd = nul.mean(0), nul.std(0)
    z = (obs - mu) / (sd + FN.EPS * (np.abs(mu) + 1e-3) + 1e-6)
    frac = np.bincount(t, minlength=len(BROAD)) / len(t)
    return np.concatenate([np.clip(np.nan_to_num(z), -8, 8), frac]).astype(np.float32)


def names():
    d = ["tp", "n_sig", "wbirth", "pmax", "ent", "top5"]; cn = ["lcc_all", "lcc_set", "ncomp"]
    nm = []
    for b in BROAD:
        nm += [f"hT:{b}|H0|{x}" for x in d] + [f"hT:{b}|H1|{x}" for x in d] + [f"hT:{b}|lvl|{x}" for x in cn]
    xs = ["H0tp", "ncomp@1", "ncomp@2", "nloops", "H1tp", "H1max"]
    for a in range(len(BROAD)):
        for b in range(a, len(BROAD)):
            nm += [f"hX:{BROAD[a]}~{BROAD[b]}|{x}" for x in xs]
    return nm + [f"hfrac:{b}" for b in BROAD]


if __name__ == "__main__":
    c = sys.argv[1]; w = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    imgs, pat, meta = C.load_cohort(c)
    bidx = broad_index(meta["types"], c)
    print(c, {meta["types"][i]: BROAD[bidx[i]] for i in range(len(bidx))}, flush=True)
    keep = ("coords", "types", "edges", "tri_edges", "faces")
    args = [({**{k: im[k] for k in keep}, "scale": meta["scale"]}, bidx) for im in imgs]
    t0 = time.time()
    with mp.get_context("spawn").Pool(w) as p:
        F = p.map(one, args, chunksize=4)
    F = np.stack(F)
    np.savez_compressed(os.path.join(C.cohort_dir(c), "feats_hnull.npz"), hnull=F,
                        image_id=np.array([im["image_id"] for im in imgs]), names=np.array(names()))
    print("DONE hnull", c, F.shape, f"{time.time()-t0:.0f}s", flush=True)
