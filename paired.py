"""Paired significance tests between two fixed methods on identical folds/seeds.
Per task: mean metric difference (ours - prior) with a patient-level bootstrap CI and p-value
(patients resampled within each test fold, B times); across tasks: Wilcoxon signed-rank.
python paired.py  -> writes paired_tests.txt / paired_tests.csv in the project root"""
import os, json, glob, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import common as C

PAIRS = {  # setting: (ours, prior, kind)
    "classification A": ("lgbmsel_mfield2", "lgbm_comp_mm", "bin"),
    "classification B": ("lgbm_bank", "lgbm_comp_mm", "bin"),
    "classification C (tuned prior)": ("lgbmsel_mfield2", "lgbmcv_comp_mm", "bin"),
    "survival, tissue only A": ("estrisk_c_ours", "rsf2cv_comp", "surv"),
    "survival, tissue only B": ("estcvriskE_c_compact", "rsf2cv_comp", "surv"),
    "survival, tissue only C (same learner)": ("estcv_c_compact", "estcv_comp", "surv"),
    "learned-embedding fields (GPU/NPU), classification": ("lgbmsel_emb", "lgbm_comp_mm", "bin"),
    "learned-embedding fields (GPU/NPU), survival": ("estrisk_c_emb", "rsfcv_comp", "surv"),
    "learned-embedding fields (GPU/NPU), survival, same learner": ("estcv_c_emb", "estcv_comp", "surv"),
    "survival + clinical A": ("estpca_k_c_cross", "estpca_k_comp", "surv"),
    "survival + clinical B": ("estrisk_k_c_ours", "estpca_k_comp", "surv"),
    "survival + clinical C (residual topology)": ("estres_k_c_ours", "estpca_k_comp", "surv"),
    "survival + clinical D (residual topology, cross)": ("estres_k_c_cross", "estpca_k_comp", "surv"),
    "survival + clinical E (residual topology, compact)": ("estres_k_c_compact", "estpca_k_comp", "surv"),
    "survival + clinical F (residual topology, mfield2)": ("estres_k_c_mfield2", "estpca_k_comp", "surv"),
    "survival + clinical G (offset Cox)": ("estresL_k_c_ours", "estpca_k_comp", "surv"),
    "survival, tissue only D (residual topology)": ("estres_c_ours", "rsf2cv_comp", "surv"),
    "TopoGate classification: lgbmgate_mfield2 vs lgbmcv_comp_mm": ("lgbmgate_mfield2", "lgbmcv_comp_mm", "bin"),
    "TopoGate classification: lgbmgate_mfield2 vs lgbm_comp_mm": ("lgbmgate_mfield2", "lgbm_comp_mm", "bin"),
    "TopoGate classification: lgbmgate_mfield2 vs lgbm_mm": ("lgbmgate_mfield2", "lgbm_mm", "bin"),
    "TopoGate classification: lgbmgate_mfield2 vs lgbmcv_bank2": ("lgbmgate_mfield2", "lgbmcv_bank2", "bin"),
    "TopoGate classification: lgbmgate_mfield2 vs lgbmsel_mfield2": ("lgbmgate_mfield2", "lgbmsel_mfield2", "bin"),
    "TopoGate classification: lgbmgate_ours vs lgbmcv_comp_mm": ("lgbmgate_ours", "lgbmcv_comp_mm", "bin"),
    "TopoGate classification: lgbmgate_ours vs lgbm_comp_mm": ("lgbmgate_ours", "lgbm_comp_mm", "bin"),
    "TopoGate classification: lgbmgate_ours vs lgbm_mm": ("lgbmgate_ours", "lgbm_mm", "bin"),
    "TopoGate classification: lgbmgate_ours vs lgbmcv_bank2": ("lgbmgate_ours", "lgbmcv_bank2", "bin"),
    "TopoGate classification: lgbmgate_ours vs lgbmsel_mfield2": ("lgbmgate_ours", "lgbmsel_mfield2", "bin"),
    "TopoGate survival tissue: rsf2gate_c_ours vs rsf2cv_comp": ("rsf2gate_c_ours", "rsf2cv_comp", "surv"),
    "TopoGate survival tissue: rsf2gate_c_ours vs rsfcv_comp": ("rsf2gate_c_ours", "rsfcv_comp", "surv"),
    "TopoGate survival tissue: rsf2gate_c_ours vs estcv_comp": ("rsf2gate_c_ours", "estcv_comp", "surv"),
    "TopoGate survival tissue: rsf2gate_c_ours vs estrisk_c_ours": ("rsf2gate_c_ours", "estrisk_c_ours", "surv"),
    "TopoGate survival tissue: rsf2gate_c_compact vs rsf2cv_comp": ("rsf2gate_c_compact", "rsf2cv_comp", "surv"),
    "TopoGate survival tissue: rsf2gate_c_compact vs rsfcv_comp": ("rsf2gate_c_compact", "rsfcv_comp", "surv"),
    "TopoGate survival tissue: rsf2gate_c_compact vs estcv_comp": ("rsf2gate_c_compact", "estcv_comp", "surv"),
    "TopoGate survival tissue: rsf2gate_c_compact vs estrisk_c_ours": ("rsf2gate_c_compact", "estrisk_c_ours", "surv"),
    "TopoGate survival clinical: estgate_k_c_ours vs estpca_k_comp": ("estgate_k_c_ours", "estpca_k_comp", "surv"),
    "TopoGate survival clinical: estgate_k_c_ours vs estcv_k_comp": ("estgate_k_c_ours", "estcv_k_comp", "surv"),
    "TopoGate survival clinical: estgate_k_c_ours vs estpca_k_c_cross": ("estgate_k_c_ours", "estpca_k_c_cross", "surv"),
    "TopoGate-2 classification: lgbmgate2_mfield2 vs lgbmcv_comp_mm": ("lgbmgate2_mfield2", "lgbmcv_comp_mm", "bin"),
    "TopoGate-2 classification: lgbmgate2_mfield2 vs lgbm_comp_mm": ("lgbmgate2_mfield2", "lgbm_comp_mm", "bin"),
    "TopoGate-2 classification: lgbmgate2_mfield2 vs lgbm_mm": ("lgbmgate2_mfield2", "lgbm_mm", "bin"),
    "TopoGate-2 classification: lgbmgate2_mfield2 vs lgbmcv_bank2": ("lgbmgate2_mfield2", "lgbmcv_bank2", "bin"),
    "TopoGate-2 classification: lgbmgate2_mfield2 vs lgbmsel_mfield2": ("lgbmgate2_mfield2", "lgbmsel_mfield2", "bin"),
    "TopoGate-2 classification: lgbmgate2_mfield2 vs lgbmgate_mfield2": ("lgbmgate2_mfield2", "lgbmgate_mfield2", "bin"),
    "TopoGate-2 survival tissue: rsfgate2_c_ours vs rsf2cv_comp": ("rsfgate2_c_ours", "rsf2cv_comp", "surv"),
    "TopoGate-2 survival tissue: rsfgate2_c_ours vs rsfcv_comp": ("rsfgate2_c_ours", "rsfcv_comp", "surv"),
    "TopoGate-2 survival tissue: rsfgate2_c_ours vs estcv_comp": ("rsfgate2_c_ours", "estcv_comp", "surv"),
    "TopoGate-2 survival tissue: rsfgate2_c_ours vs rsf2gate_c_ours": ("rsfgate2_c_ours", "rsf2gate_c_ours", "surv"),
    "TopoGate-2 survival tissue: rsfgate2_c_compact vs rsf2cv_comp": ("rsfgate2_c_compact", "rsf2cv_comp", "surv"),
    "TopoGate-2 survival tissue: rsfgate2_c_compact vs rsfcv_comp": ("rsfgate2_c_compact", "rsfcv_comp", "surv"),
    "TopoGate-2 survival tissue: rsfgate2_c_compact vs estcv_comp": ("rsfgate2_c_compact", "estcv_comp", "surv"),
    "TopoGate-2 survival tissue: rsfgate2_c_compact vs rsf2gate_c_ours": ("rsfgate2_c_compact", "rsf2gate_c_ours", "surv"),
    "SPATIAL cls: lgbmgate2_mfield2 vs lgbm_nbr": ("lgbmgate2_mfield2", "lgbm_nbr", "bin"),
    "SPATIAL cls: lgbmgate2_mfield2 vs lgbmcv_nbr": ("lgbmgate2_mfield2", "lgbmcv_nbr", "bin"),
    "SPATIAL cls: lgbmgate2_mfield2 vs lgbm_cn": ("lgbmgate2_mfield2", "lgbm_cn", "bin"),
    "SPATIAL cls: lgbmgate2_mfield2 vs lgbmcv_cn": ("lgbmgate2_mfield2", "lgbmcv_cn", "bin"),
    "SPATIAL cls: lgbmgate2_znull vs lgbm_nbr": ("lgbmgate2_znull", "lgbm_nbr", "bin"),
    "SPATIAL cls: lgbmgate2_znull vs lgbmcv_nbr": ("lgbmgate2_znull", "lgbmcv_nbr", "bin"),
    "SPATIAL cls: lgbmgate2_znull vs lgbm_cn": ("lgbmgate2_znull", "lgbm_cn", "bin"),
    "SPATIAL cls: lgbmgate2_znull vs lgbmcv_cn": ("lgbmgate2_znull", "lgbmcv_cn", "bin"),
    "SPATIAL cls: lgbmgate2_znull vs lgbmcv_comp_mm": ("lgbmgate2_znull", "lgbmcv_comp_mm", "bin"),
    "SPATIAL cls: lgbmgate2_znull vs lgbm_comp_mm": ("lgbmgate2_znull", "lgbm_comp_mm", "bin"),
    "SPATIAL cls: lgbmgate2_znull vs lgbm_mm": ("lgbmgate2_znull", "lgbm_mm", "bin"),
    "SPATIAL cls: lgbmgate2_zall vs lgbm_nbr": ("lgbmgate2_zall", "lgbm_nbr", "bin"),
    "SPATIAL cls: lgbmgate2_zall vs lgbmcv_nbr": ("lgbmgate2_zall", "lgbmcv_nbr", "bin"),
    "SPATIAL cls: lgbmgate2_zall vs lgbm_cn": ("lgbmgate2_zall", "lgbm_cn", "bin"),
    "SPATIAL cls: lgbmgate2_zall vs lgbmcv_cn": ("lgbmgate2_zall", "lgbmcv_cn", "bin"),
    "SPATIAL cls: lgbmgate2_zall vs lgbmcv_comp_mm": ("lgbmgate2_zall", "lgbmcv_comp_mm", "bin"),
    "SPATIAL cls: lgbmgate2_zall vs lgbm_comp_mm": ("lgbmgate2_zall", "lgbm_comp_mm", "bin"),
    "SPATIAL cls: lgbmgate2_zall vs lgbm_mm": ("lgbmgate2_zall", "lgbm_mm", "bin"),
    "SPATIAL surv: rsf2gate_c_ours vs rsf2cv_c_nbr": ("rsf2gate_c_ours", "rsf2cv_c_nbr", "surv"),
    "SPATIAL surv: rsf2gate_c_ours vs rsf2cv_c_cn": ("rsf2gate_c_ours", "rsf2cv_c_cn", "surv"),
    "SPATIAL surv: rsf2gate_c_znull vs rsf2cv_c_nbr": ("rsf2gate_c_znull", "rsf2cv_c_nbr", "surv"),
    "SPATIAL surv: rsf2gate_c_znull vs rsf2cv_c_cn": ("rsf2gate_c_znull", "rsf2cv_c_cn", "surv"),
    "SPATIAL surv: rsf2gate_c_znull vs rsf2cv_comp": ("rsf2gate_c_znull", "rsf2cv_comp", "surv"),
    "SPATIAL surv: rsf2gate_c_znull vs rsfcv_comp": ("rsf2gate_c_znull", "rsfcv_comp", "surv"),
    "SPATIAL surv: rsf2gate_c_znull vs estcv_comp": ("rsf2gate_c_znull", "estcv_comp", "surv"),
    "SPATIAL clin: estgate_k_c_znull vs estpca_k_comp": ("estgate_k_c_znull", "estpca_k_comp", "surv"),
    "SPATIAL clin: estgate_k_c_znull vs estcv_k_comp": ("estgate_k_c_znull", "estcv_k_comp", "surv"),
    "TopoGate-final cls: lgbmgate4_cnz vs lgbmcv_comp_mm": ("lgbmgate4_cnz", "lgbmcv_comp_mm", "bin"),
    "TopoGate-final cls: lgbmgate4_cnz vs lgbm_comp_mm": ("lgbmgate4_cnz", "lgbm_comp_mm", "bin"),
    "TopoGate-final cls: lgbmgate4_cnz vs lgbm_mm": ("lgbmgate4_cnz", "lgbm_mm", "bin"),
    "TopoGate-final cls: lgbmgate4_cnz vs lgbm_nbr": ("lgbmgate4_cnz", "lgbm_nbr", "bin"),
    "TopoGate-final cls: lgbmgate4_cnz vs lgbmcv_nbr": ("lgbmgate4_cnz", "lgbmcv_nbr", "bin"),
    "TopoGate-final cls: lgbmgate4_cnz vs lgbm_cn": ("lgbmgate4_cnz", "lgbm_cn", "bin"),
    "TopoGate-final cls: lgbmgate4_cnz vs lgbmcv_cn": ("lgbmgate4_cnz", "lgbmcv_cn", "bin"),
    "TopoGate-final cls: lgbmgate4_cnz vs lgbmgate2_znull": ("lgbmgate4_cnz", "lgbmgate2_znull", "bin"),
    "TopoGate-final cls: lgbmgate4_cnz vs gin": ("lgbmgate4_cnz", "gin", "bin"),
    "TopoGate-final cls: lgbmgate4_cnz vs gat": ("lgbmgate4_cnz", "gat", "bin"),
    "TopoGate-final surv: rsfgate4_c_cnz vs rsf2cv_comp": ("rsfgate4_c_cnz", "rsf2cv_comp", "surv"),
    "TopoGate-final surv: rsfgate4_c_cnz vs rsfcv_comp": ("rsfgate4_c_cnz", "rsfcv_comp", "surv"),
    "TopoGate-final surv: rsfgate4_c_cnz vs estcv_comp": ("rsfgate4_c_cnz", "estcv_comp", "surv"),
    "TopoGate-final surv: rsfgate4_c_cnz vs rsf2cv_c_nbr": ("rsfgate4_c_cnz", "rsf2cv_c_nbr", "surv"),
    "TopoGate-final surv: rsfgate4_c_cnz vs rsf2cv_c_cn": ("rsfgate4_c_cnz", "rsf2cv_c_cn", "surv"),
    "TopoGate-final surv: rsfgate4_c_cnz vs rsf2gate_c_ours": ("rsfgate4_c_cnz", "rsf2gate_c_ours", "surv"),
    "TopoGate-final surv: rsfgate4_c_cnz vs gin": ("rsfgate4_c_cnz", "gin", "surv"),
    "TopoGate-final surv: rsfgate4_c_cnz vs gat": ("rsfgate4_c_cnz", "gat", "surv"),
}
B = 1000


def load(cohort, task, method):
    out = {}
    for f in glob.glob(os.path.join(C.RES, cohort, task, method, "f*_s*.json")):
        d = json.load(open(f)); out[(d["fold"], d["seed"])] = d
    return out


def met(kind, d, pred, idx):
    if kind == "bin":
        return C.auroc(np.asarray(d["y"])[idx], pred[idx])
    return C.cindex(np.asarray(d["time"])[idx], np.asarray(d["event"])[idx], pred[idx])


def compare(cohort, task, ours, prior, kind, rng):
    A, P = load(cohort, task, ours), load(cohort, task, prior)
    keys = sorted(set(A) & set(P))
    if len(keys) < 3:
        return None
    obs = []; boot = np.zeros(B)
    for k in keys:
        a, p = A[k], P[k]
        assert a["patients"] == p["patients"], "fold mismatch"
        pa, pp = np.asarray(a["pred"], float), np.asarray(p["pred"], float); n = len(pa)
        full = np.arange(n)
        obs.append(met(kind, a, pa, full) - met(kind, a, pp, full))
        for b in range(B):
            idx = rng.integers(0, n, n)
            da, dp = met(kind, a, pa, idx), met(kind, a, pp, idx)
            boot[b] += (da - dp) if np.isfinite(da) and np.isfinite(dp) else 0.0
    boot /= len(keys); d = float(np.mean(obs))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean()); p = min(max(p, 1.0 / B), 1.0)
    mo = np.mean([met(kind, A[k], np.asarray(A[k]["pred"], float), np.arange(len(A[k]["pred"]))) for k in keys])
    return dict(cohort=cohort, task=task, ours=ours, prior=prior, n_runs=len(keys), ours_mean=mo, prior_mean=mo - d,
                diff=d, ci_lo=lo, ci_hi=hi, p_boot=p)


if __name__ == "__main__":
    rng = np.random.default_rng(0); rows = []; lines = []
    cohorts = [c for c in sorted(os.listdir(C.RES)) if not c.startswith("synth") and not c.endswith("_smoke")]
    for name, (ours, prior, kind) in PAIRS.items():
        sub = []
        for c in cohorts:
            for t in sorted(os.listdir(os.path.join(C.RES, c))):
                base = c[:-4] if c.endswith("_ext") else c
                if C.TASKS.get(base, {}).get(t, ("",))[0] != kind:
                    continue
                r = compare(c, t, ours, prior, kind, rng)
                if r:
                    r["setting"] = name; sub.append(r)
        if not sub:
            continue
        rows += sub; d = np.array([r["diff"] for r in sub])
        w = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else float("nan")
        lines.append(f"\n=== {name}: {ours} vs {prior} | tasks={len(d)} mean diff={d.mean():+.4f} "
                     f"wins={int((d > 0).sum())}/{len(d)} sig.wins(p<.05)={sum(r['diff'] > 0 and r['p_boot'] < .05 for r in sub)} "
                     f"sig.losses={sum(r['diff'] < 0 and r['p_boot'] < .05 for r in sub)} Wilcoxon p={w:.4f}")
        for r in sub:
            lines.append(f"  {r['cohort']:10s} {r['task']:7s} ours={r['ours_mean']:.3f} prior={r['prior_mean']:.3f} "
                         f"diff={r['diff']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p_boot']:.3f} (runs {r['n_runs']})")
        print("\n".join(lines[-len(sub) - 1:]), flush=True)
    open(os.path.join(C.ROOT, "paired_tests.txt"), "w").write("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(C.ROOT, "paired_tests.csv"), index=False)
