# TOPO_TME

Topological descriptors of the tumour microenvironment from single-cell spatial proteomics
(imaging mass cytometry, CODEX), and the learners that use them.

## What is here

| module | purpose |
|---|---|
| `topo_core.py` | exact 0-/1-dimensional persistence of filtrations on Delaunay complexes (union-find + dual union-find, numba) |
| `feat_compact.py` | marker-field and cell-type indicator-field persistence summaries (interpretable scalars) |
| `feat_cross.py` | cross-type interaction persistence for every pair of cell types |
| `bank.py`, `feat_mfield*.py` | multi-scale, two-sided marker-field persistence images |
| `feat_null.py` | **TopoNull**: composition-calibrated topological excess (observed descriptors standardised against a label-/marker-permutation null with fixed cell positions and composition) |
| `feat_cn.py` | non-topological spatial baseline: cellular-neighbourhood frequencies |
| `featrun.py` | parallel feature extraction: `python featrun.py <feat_module> <cohort> [workers]` |
| `baselines.py` | all tabular learners (LightGBM, logistic, Cox, RSF, extra survival trees, gradient-boosted survival, Coxnet), inner-CV tuning, topology screening/PCA/risk-score variants and the **TopoGate** learners |
| `train.py`, `models.py`, `models3.py` | learnable persistence and GNN (GIN/GAT) models (PyTorch) |
| `cellenc.py` | self-supervised cell encoder (GPU training, OpenVINO inference) for learned-embedding fields |
| `paired.py`, `paired_fast.py` | paired patient-level bootstrap tests per task and Wilcoxon tests across tasks |
| `aggregate.py` | average-rank summary over all methods and tasks |
| `interp.py` | which descriptors carry signal beyond composition (conditional likelihood-ratio tests, BH-FDR, cross-validated AUROC gain) |
| `get_data.py`, `get_luad.py`, `preprocess.py`, `prep_luad.py` | download (HTTP range reads into the public archives) and preprocessing of each cohort |
| `synth.py` | synthetic marked point processes with known topological ground truth |

### TopoGate

`lgbmgate2_<fset>` / `rsf2gate_<fset>` combine a tuned composition model with a topology model as
`rank(base) + alpha * rank(topology)`. The base learner and `alpha` (including `alpha = 0`, which
reproduces the composition model exactly) are chosen by inner cross-validation on the training fold only.

## Data

All cohorts are public; `get_data.py` / `get_luad.py` fetch only the files that are needed.

| cohort | technology | source |
|---|---|---|
| METABRIC breast (Danenberg et al., Nat Genet 2022) | IMC | Zenodo 6036188 + cBioPortal clinical data |
| Basel / Zurich breast (Jackson et al., Nature 2020) | IMC | Zenodo 4607374 |
| colorectal (Schuerch et al., Cell 2020) | CODEX | Mendeley Data |
| lung adenocarcinoma (Sorin et al., Nature 2023) | IMC | Zenodo 7760826 |

## Usage

```bash
pip install -r requirements.txt
python get_data.py danenberg cbioportal jackson codex
python get_luad.py
python preprocess.py danenberg ; python preprocess.py jackson ; python preprocess.py codex ; python prep_luad.py
for m in feat_compact feat_cross feat_mfield feat_mfield2 feat_null; do python featrun.py $m luad 8; done
python feat_cn.py luad
python baselines.py --cohort luad --only lgbm_comp_mm,lgbmcv_comp_mm,lgbmgate2_mfield2 --workers 8
python paired_fast.py 8
```

`run_all.sh` runs the full pipeline. Set `TOPO_ROOT` to choose where `raw/`, `data/` and `results/` live
(default: the parent of the code folder). On Intel MKL builds without OpenMP, set
`MKL_THREADING_LAYER=SEQUENTIAL`.
