#!/usr/bin/env bash
# Full pipeline. Results are resumable (finished runs are skipped).
set -e
PY=${PYTHON:-python}; W=${WORKERS:-8}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL
$PY get_data.py danenberg cbioportal jackson codex
$PY get_luad.py
$PY preprocess.py danenberg; $PY preprocess.py jackson; $PY preprocess.py codex; $PY prep_luad.py
for c in danenberg basel zurich codex luad; do
  for m in feat_compact feat_cross feat_mfield feat_mfield2 feat_null; do $PY featrun.py $m $c $W; done
  $PY feat_cn.py $c
done
CLS=lgbm_comp_mm,lgbmcv_comp_mm,lgbm_mm,lgbm_nbr,lgbmcv_nbr,lgbm_cn,lgbmcv_cn,lgbmsel_mfield2,lgbmcv_bank2,lgbmgate2_mfield2,lgbmgate2_znull
SRV=rsf2cv_comp,rsfcv_comp,estcv_comp,rsf2cv_c_nbr,rsf2cv_c_cn,estrisk_c_ours,rsf2gate_c_ours,rsf2gate_c_znull,estpca_k_comp,estcv_k_comp,estgate_k_c_ours
for c in danenberg basel codex luad; do $PY baselines.py --cohort $c --only $CLS,$SRV --workers $W; done
$PY baselines.py --cohort basel --external --only $CLS --workers $W
$PY paired_fast.py $W
$PY aggregate.py > summary.txt
