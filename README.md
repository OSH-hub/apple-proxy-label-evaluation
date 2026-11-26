# apple-proxy-label-evaluation

Analysis code and aggregate result tables for the article

> S. Oh, "Target-Specific Evaluation of Uncertainty in Proxy-Supervised Image Regression: A Tree-Held-Out Study of RGB Apple Sugar Estimation."

The article presents an evaluation protocol for image regression models that are trained on labels produced by another instrument (proxy labels). The protocol reports model–proxy error, model–reference error and proxy–reference discrepancy separately, holds out whole trees, compares the model with constant baselines computed from the fitting labels, and verifies that Monte Carlo dropout inference is stochastic. It is applied to RGB estimation of apple sugar content with near-infrared (NIR) estimates as proxy labels and destructive measurements as the reference.

## Contents

| Path | Contents |
|---|---|
| `tables/` | Aggregate result tables (CSV) |
| `code/` | Source reconstruction, model fitting, verification, aggregate analysis and figure rendering scripts |
| `dependencies/` | Representative-image selection definitions used by the source worker |
| `protocols/` | Dated analysis protocol and amendments, including the under-fitting sensitivity plan |
| `reports/` | Aggregate audit, verification and runtime reports |
| `docs/extended_methods.md` | Extended methods and audit notes (source audit, partitions, model fits, cohorts, ranking definitions, reproduction, sensitivity runs, label and date summaries) |
| `MANIFEST.sha256` | SHA-256 checksums of the files in this repository |

## Tables and the parts of the article they support

Machine-readable aggregate tables:

| File | Contents | Article |
|---|---|---|
| `tables/split_inventory.csv` | Record and tree counts by scheme, role and source pool | Section II-B, Table I |
| `tables/accuracy.csv` | Prediction accuracy for all schemes, cohorts, cultivars, predictors and cluster definitions | Section III-A, Tables II–IV, Figure 2 |
| `tables/paired_accuracy.csv` | Paired accuracy differences with tree-, farm- and combination-cluster intervals | Section III-A, Tables II and IV |
| `tables/reference_pool_label_audit.csv` | Descriptive NIR–reference comparison in the complete reference pool, by cultivar | Section III-A, Table III |
| `tables/seed_metrics.csv` | Per-seed reference MAE, NIR MAE, MC-dropout SD and aleatoric SD | Sections III-A and III-B, note of Table II |
| `tables/uncertainty_three_targets.csv` | Uncertainty and disagreement rankings against three error targets, all scores, cohorts and budgets | Section III-B, Table V, Figure 3 |
| `tables/within_cultivar_scores.csv` | Within-cultivar score–error correlations | Section III-B, Table VI |
| `tables/capture_curves.csv` | Retrospective capture curves for every score | Section III-B, Figure 4 |
| `tables/interval_coverage.csv` | Interval coverage and width for all methods, targets, cohorts and cluster definitions | Section III-C, Tables IV and VII, Figure 5 |
| `tables/cohort_label_date_summary.csv` | Label and retained-key date summaries for the tree-scheme cohorts | Sections III-A and III-C, note of Table VII |
| `tables/underfit_three_seed_mean.csv` | Under-fitting sensitivity runs, three-seed means | Section III-D, Table VIII |
| `tables/underfit_by_seed.csv` | Under-fitting sensitivity runs by seed | Section III-D |
| `tables/underfit_learning_curves.csv` | Training loss and proxy selection MAE by epoch for the sensitivity runs | Section III-D |
| `tables/checkpoint_selection_sensitivity.csv` | Aggregate check of checkpoint selection after omission of one zero-valued NIR record | Section II-B |
| `tables/farm_overlap.csv` | Farm overlap between test and development in the combination partition | Section II-B |
| `tables/cluster_error_sums.csv` | Anonymous per-cluster record counts and absolute-error sums | Sections II-G and III-A; input of `code/reproduce_aggregate_accuracy.py` |

Document-style tables with rounded values as displayed:

| File | Contents | Article |
|---|---|---|
| `tables/combination_partition.csv` | Cultivar–farm-combination partition after same-row reconstruction | Section II-B |
| `tables/model_fit_summary.csv` | Selected epoch, selection MAE, reference MAE, MC-dropout SD and run time of the six model fits | Sections II-C and III-A |
| `tables/cohort_comparison.csv` | Accuracy, paired differences and coverage for the complete and strict cohorts of both partitions | Table IV |
| `tables/underfit_three_seed_mean_display.csv` | Under-fitting sensitivity, three-seed means | Table VIII |
| `tables/underfit_by_seed_display.csv` | Under-fitting sensitivity by seed | Section III-D |
| `tables/reference_test_label_summary.csv` | Reference-test label and key-date summaries by cultivar | Sections III-A and III-C |

In the tables and code, the scheme code `zone` denotes the cultivar–farm-combination partition, `kind` the cultivar and `pos` the provider's farm position code.

## What is included and what is not

Included: analysis code, protocol documents, aggregate reports and aggregate tables at the level of schemes, cohorts, cultivars, seeds and anonymous clusters.

Not included: source images, JSON annotations, crops, individual labels, row-level predictions, source keys, split manifests and trained checkpoints. `tables/cluster_error_sums.csv` identifies clusters only by an index within each scheme and cohort; it does not carry tree, farm or record keys and does not reconstruct individual labels.

## Running the code

Requirements for the aggregate tools: Python 3.11, NumPy and pandas. Model fitting and inference additionally use PyTorch, torchvision, timm and Pillow; analysis uses SciPy and figure rendering uses Matplotlib. The recorded run environment is in `reports/runtime_environment.json`.

- `code/render_figures_from_tables.py` redraws the result figures of the article from the aggregate tables. Its output files `figure1_…` to `figure4_…` are Figures 2–5 of the article; Figure 1 of the article is a protocol overview.
- `code/reproduce_aggregate_accuracy.py --root .` rebuilds the pooled MAE values and paired cluster-bootstrap intervals of `tables/accuracy.csv` and `tables/paired_accuracy.csv` from `tables/cluster_error_sums.csv`.
- The remaining scripts (`rebuild_source_crops_local.py`, `train_corrected.py`, `launch_verified_training.sh`, `verify_corrected.py`, `analyze_corrected.py`, `underfit_check.py`, `analyze_underfit.py`, `summarize_cohort_labels.py`, `reproduce_checkpoint_selection_summary.py`) are the executed analysis files. They need the authorized source data and the author-held split manifests and are provided to document the computation. Paths of the form `/path/to/project_root` and `/path/to/AIHub_dataset_490/...` are placeholders. Directory names that contain run labels belong to the executed directory layout.

A full rerun proceeds in this order: source reconstruction, data verification, six model fits, full validation, aggregate analysis, figure rendering.

## Data source and conditions of use

The source images and measurements are AI Hub dataset 490, "Jeonbuk Jangsu Apple Sugar Content Quality Data" (https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=490), constructed through a project of the National Information Society Agency (NIA), Republic of Korea. The data are obtained from AI Hub under its access and use conditions and are not redistributed here. The MIT license of this repository covers the code and the aggregate tables only.

## Citation

Please cite the article; see `CITATION.cff`.
