# apple-proxy-label-evaluation

Analysis code and aggregate result tables for the article

> S. Oh, "Target-Specific Evaluation of Uncertainty in Proxy-Supervised Image Regression: A Tree-Held-Out Study of RGB Apple Sugar Estimation."

The article presents an evaluation protocol for image regression models that are trained on labels produced by another instrument (proxy labels). The protocol reports model–proxy error, model–reference error and proxy–reference discrepancy separately, holds out whole trees, compares the model with constant baselines computed from the fitting labels, and verifies that Monte Carlo dropout inference is stochastic. It is applied to RGB estimation of apple sugar content with near-infrared (NIR) estimates as proxy labels and destructive measurements as the reference.

## Contents

| Path | Contents |
|---|---|
| `tables/` | Aggregate result tables (Data S1–S8 and S10–S16 of the article's supplementary material) |
| `code/` | Source reconstruction, model fitting, verification, aggregate analysis and figure rendering scripts |
| `dependencies/` | Representative-image selection definitions used by the source worker |
| `protocols/` | Dated analysis protocol and amendments |
| `reports/` | Aggregate audit, verification and runtime reports |
| `MANIFEST.sha256` | SHA-256 checksums of the files in this repository |

### Tables

| File | Contents |
|---|---|
| `DataS01_split_inventory.csv` | Record and tree counts by scheme, role and source pool |
| `DataS02_accuracy.csv` | Prediction accuracy for all schemes, cohorts, cultivars, predictors and cluster definitions |
| `DataS03_uncertainty_three_targets.csv` | Uncertainty and disagreement rankings against three error targets |
| `DataS04_interval_coverage.csv` | Interval coverage and width |
| `DataS05_paired_accuracy.csv` | Paired accuracy differences |
| `DataS06_seed_metrics.csv` | Per-seed metrics |
| `DataS07_full_reference_label_audit.csv` | Descriptive NIR–reference comparison in the full reference pool |
| `DataS08_capture_curves.csv` | Retrospective capture curves |
| `DataS10_within_cultivar_scores.csv` | Within-cultivar score–error correlations |
| `DataS11_farm_overlap.csv` | Farm overlap between test and development |
| `DataS12_underfit_three_seed_mean.csv` | Under-fitting sensitivity runs, three-seed means |
| `DataS13_underfit_by_seed.csv` | Under-fitting sensitivity runs by seed |
| `DataS14_underfit_learning_curves.csv` | Training loss and proxy selection MAE by epoch for the sensitivity runs |
| `DataS15_cohort_label_date_summary.csv` | Label and retained-key date summaries for the tree-scheme cohorts |
| `DataS16_checkpoint_selection_sensitivity.csv` | Aggregate check of checkpoint selection after omission of one zero-valued NIR record |

In the tables and code, the scheme code `zone` denotes the cultivar–farm-combination partition, `kind` the cultivar and `pos` the provider's farm position code.

## What is included and what is not

Included: analysis code, protocol documents, aggregate reports and aggregate tables at the level of schemes, cohorts, cultivars and seeds.

Not included: source images, JSON annotations, crops, individual labels, row-level predictions, source keys, split manifests, trained checkpoints, and tree-level tables. Data S9 of the article (per-cluster counts and error sums) is distributed with the article's supplementary material only.

## Running the code

Requirements for the aggregate tools: Python 3.11, NumPy and pandas. Model fitting and inference additionally use PyTorch, torchvision, timm and Pillow; analysis uses SciPy and figure rendering uses Matplotlib. The recorded run environment is in `reports/runtime_environment.json`.

- `code/render_figures_from_tables.py` redraws the result figures of the article from the aggregate tables. Its output files `figure1_…` to `figure4_…` are Figures 2–5 of the article; Figure 1 of the article is a protocol overview.
- `code/reproduce_aggregate_accuracy.py --root .` rebuilds the pooled MAE values and paired cluster-bootstrap intervals of Data S2 and Data S5 from Data S9. Place `DataS09_cluster_error_sums.csv` from the article's supplementary material in `tables/` before running it.
- The remaining scripts (`rebuild_source_crops_local.py`, `train_corrected.py`, `launch_verified_training.sh`, `verify_corrected.py`, `analyze_corrected.py`, `underfit_check.py`, `analyze_underfit.py`, `summarize_cohort_labels.py`, `reproduce_checkpoint_selection_summary.py`) are the executed analysis files. They need the authorized source data and the author-held split manifests and are provided to document the computation. Paths of the form `/path/to/project_root` and `/path/to/AIHub_dataset_490/...` are placeholders. Directory names that contain run labels (for example `submission_20261026_v4`) belong to the executed directory layout.

A full rerun proceeds in this order: source reconstruction, data verification, six model fits, full validation, aggregate analysis, figure rendering.

## Data source and conditions of use

The source images and measurements are AI Hub dataset 490, "Jeonbuk Jangsu Apple Sugar Content Quality Data" (https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=490), constructed through a project of the National Information Society Agency (NIA), Republic of Korea. The data are obtained from AI Hub under its access and use conditions and are not redistributed here. The MIT license of this repository covers the code and the aggregate tables only.

## Citation

Please cite the article; see `CITATION.cff`.
