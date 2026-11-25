# TOPO_TME

Composition-calibrated topology of single-cell spatial maps (imaging mass cytometry, CODEX and
similar platforms).

A spatial descriptor of a tissue, such as the number of T-cell clusters or loops, usually changes
with the number of cells of each type, so an apparent spatial signal often restates composition.
This repository computes persistent homology of cell-type and protein-expression fields on the
Delaunay complex of single cells and standardizes every descriptor against a permutation null that
keeps cell positions and composition fixed (**TopoNull**). The calibrated descriptors (topological
excess, TEx) are centered near 0 when the cells are arranged at random, whatever the composition.
**TopoGate** then adds them to composition models through a cross-validated gate that can switch
topology off when it does not help.

## Quick start: minimum working example (synthetic data, no downloads)

```bash
git clone https://github.com/sarwanpasha/TOPO_TME
cd TOPO_TME
pip install -r requirements.txt
python example.py
```

`example.py` simulates 60 tumor cores of about 1,400 cells each (5 cell types, 8 protein markers).
In half of them (y = 1), 240 T cells form one long connected channel; in the other half (y = 0), the
same random walk is cut into 8 separate chains of 30 cells. Both classes have exactly the same
composition and the same distance between neighboring T cells, so composition and local contact
statistics cannot separate them. The script then

1. builds the Delaunay complex of each core (`preprocess.finish_image`),
2. computes 405 TopoNull descriptors per core (`feat_null.compute`): 15 per cell type, 6 per pair of
   cell types and 30 per marker, each standardized against 8 permutations of the cell labels,
3. prints the composition of both classes and the descriptors that differ most between them,
4. compares 5-fold cross-validated AUROC of a logistic regression on composition alone and on
   composition plus TopoNull (descriptor screening is done inside each training fold).

Expected output (about 1 to 2 minutes on a laptop; the first call also compiles the numba kernels):

```
60 images, 1420 cells per image on average
TopoNull: 405 calibrated descriptors per image in 92.4 s

mean composition (y=0 | y=1):
  tumour      0.254 | 0.254
  T cell      0.169 | 0.169
  stroma      0.493 | 0.493
  B cell      0.028 | 0.028
  macrophage  0.056 | 0.056

top calibrated descriptors (Mann-Whitney p, mean TEx y=0 | y=1):
  zT:T cell|H0|pmax            p=6.0e-09  -2.68 | -4.61
  zT:T cell|H0|top5            p=2.4e-07  -4.53 | -5.69
  zT:T cell|H0|ent             p=1.8e-04  -0.02 | +0.11
  zM:m0|sup|H0|top5            p=1.1e-02  -0.19 | +0.13
  zM:m0|sup|H0|pmax            p=1.1e-02  -0.27 | +0.27

5-fold CV AUROC, composition only:       0.50
5-fold CV AUROC, composition + TopoNull: 0.88
```

How to read it. Composition is identical, so the composition model is at chance (AUROC 0.50). The
leading descriptors are 0-dimensional persistence (connected components, `H0`) of the T-cell
indicator field: `pmax` is the persistence of the most persistent component and `top5` the mean of
the 5 largest. They are far below 0 in both classes because the T cells are much more clustered
than under random placement, and lower still when the T cells form one connected channel. The run
time depends on the CPU; the other numbers are deterministic for a given package version (tested
with Python 3.11, NumPy 2.0, SciPy 1.17, scikit-learn 1.9 and numba 0.68).

### Descriptor names

`zT:<type>|H0|...` and `zT:<type>|H1|...` are components and loops of the indicator field of one cell
type, `zT:<type>|lvl|...` are level-set connectivity summaries, `zX:<a>~<b>|...` describe how 2 cell
types interleave, and `zM:<marker>|sup|...` and `zM:<marker>|sub|...` describe regions of high and low
expression of one marker. The summaries are total persistence (`tp`), number of persistent pairs
(`n_sig`), persistence-weighted birth (`wbirth`), maximum persistence (`pmax`), persistent entropy
(`ent`) and the mean of the 5 largest persistences (`top5`).

## Use on your own data

Each image needs cell positions, an integer cell type per cell and a matrix of standardized marker
intensities (one row per cell).

```python
import numpy as np
from preprocess import finish_image
import feat_null as FN

xy = ...       # (n_cells, 2) positions, any unit
types = ...    # (n_cells,) integers 0 .. n_types-1
markers = ...  # (n_cells, n_markers) standardized intensities, for example z-scores per marker

im = finish_image("img1", "patient1", xy, types, markers, n_types=4)
im["scale"] = im["med_edge"]   # with several images, use the median of im["med_edge"] over all images
tex = FN.compute(im, n_types=4)                       # one vector of calibrated descriptors
names = FN.feature_names(["CD3", "CD8", "CD20", "CD68", "PanCK", "CD31"],
                         ["tumor", "T cell", "stroma", "myeloid"])
```

For 2,000 randomly placed cells with 4 types and 6 markers this returns 300 descriptors with a mean
absolute value of about 0.3, as expected under spatial randomness. Stack the vectors of all images
into a matrix and use it, together with composition, in any classifier or survival model.

## What is here

| module | purpose |
|---|---|
| `example.py` | minimum working example on synthetic tissue (above) |
| `topo_core.py` | exact 0- and 1-dimensional persistence of filtrations on Delaunay complexes (union-find and dual union-find, numba) |
| `feat_compact.py` | marker-field and cell-type indicator-field persistence summaries |
| `feat_cross.py` | cross-type interaction persistence for every pair of cell types |
| `feat_null.py` | **TopoNull**: descriptors standardized against a label and marker permutation null with fixed cell positions and composition |
| `feat_ladder.py`, `ladder_analysis.py` | nested nulls (composition, then local density, then region) and the decomposition of each descriptor into abundance, density, regional, type-layout and micro-organization components |
| `feat_nulld.py`, `feat_null32.py`, `feat_nullpert.py` | density-aware null, 32-permutation null and perturbed-map robustness variants |
| `feat_alttopo.py`, `feat_chrom.py`, `feat_moran.py`, `feat_conn.py` | alternative spatial statistics for comparison (alpha, 2-parameter, Dowker and chromatic persistence, spatial autocorrelation, graph connectivity) |
| `bank.py`, `feat_mfield*.py` | multi-scale, two-sided marker-field persistence images |
| `feat_cn.py` | cellular-neighborhood frequencies (non-topological spatial baseline) |
| `featrun.py` | parallel feature extraction: `python featrun.py <feat_module> <cohort> [workers]` |
| `baselines.py` | tabular learners (LightGBM, logistic, Cox, random survival forests and others), inner-CV tuning and the **TopoGate** learners |
| `train.py`, `models.py`, `models3.py`, `tuned_select.py` | graph neural network baselines (GIN, GAT) and their tuning (PyTorch) |
| `paired.py`, `paired_fast.py`, `ablate_tests.py`, `nb_tests.py`, `cohort_inference.py` | paired patient-level bootstrap tests, Wilcoxon tests across tasks, corrected resampled t-tests and cohort-level random effects |
| `interp.py`, `rephen.py`, `vasc_valid.py`, `tex_null.py` | descriptors that carry signal beyond composition, receptor-blind re-typing, vessel metrics and the empirical null of TEx |
| `surv_oof.py`, `surv_perm.py`, `perm_combine.py` | nested cross-validated topology risk score, its test beyond clinical covariates and permutation calibration of the whole fitting procedure |
| `immucan_ws.py`, `immucan_analysis.py` | whole-slide analysis (virtual cores and whole regions) |
| `get_data.py`, `get_luad.py`, `preprocess.py`, `prep_*.py` | download and preprocessing of each public cohort |
| `synth.py` | synthetic marked point processes with known topological ground truth (`ring`, `interact`, `channel`, `ringmk`) |

### TopoGate

`lgbmgate*_<fset>` and `rsf*gate*_<fset>` combine a tuned composition model with a topology model as
`rank(base) + alpha * rank(topology)`. The base learner and `alpha` (including `alpha = 0`, which
reproduces the base model exactly) are chosen by inner cross-validation on the training fold only.

## Real cohorts

All cohorts are public; `get_data.py` and `get_luad.py` fetch only the files that are needed.

| cohort | technology | source |
|---|---|---|
| METABRIC breast (Danenberg et al., Nat Genet 2022) | IMC | Zenodo 6036188 and cBioPortal clinical data |
| Basel and Zurich breast (Jackson et al., Nature 2020) | IMC | Zenodo 4607374 |
| colorectal (Schuerch et al., Cell 2020) | CODEX | Mendeley Data |
| lung adenocarcinoma (Sorin et al., Nature 2023) | IMC | Zenodo 7760826 |
| non-small cell lung cancer (Cords et al., Cancer Cell 2024) | IMC | Zenodo 7961844 |

```bash
python get_data.py danenberg cbioportal jackson codex
python get_luad.py
python preprocess.py danenberg ; python preprocess.py jackson ; python preprocess.py codex ; python prep_luad.py
for m in feat_compact feat_cross feat_null; do python featrun.py $m luad 8; done
python feat_cn.py luad
python baselines.py --cohort luad --only lgbm_comp_mm,lgbmcv_comp_mm,lgbmgate4_cnz --workers 8
python paired_fast.py 8
```

`run_all.sh` runs the full pipeline. Set `TOPO_ROOT` to choose where `raw/`, `data/` and `results/`
are written (default: the parent of the code folder). On Intel MKL builds without OpenMP, set
`MKL_THREADING_LAYER=SEQUENTIAL`.

## License

See `LICENSE`.
