# Extended methods and audit notes

These notes accompany the article "Target-Specific Evaluation of Uncertainty in Proxy-Supervised Image Regression: A Tree-Held-Out Study of RGB Apple Sugar Estimation". They give the source audit, partition, model-fit, cohort, ranking, reproduction and sensitivity details behind the tables in `tables/`.

## 1. Source provenance and observation cohorts

All 21,443 selected source records passed normalized filename lookup, JSON image/NIR/reference linkage, source ZIP CRC and decoding, EXIF normalization, exact JSON dimensions, positive contained bounding-box (bbox) and generated JPEG checks. The original manifest has 426,951 image rows but is not the model's training sample size. The experiment pool contains 15,267 proxy-only records and 6,176 reference-pool records. Exactly 314 selected reference rows lack sugar; all remain outside model fitting and reference evaluation. The complete same-row reference cohort has 5,862 records; the strict unique-NIR cohort has 5,364. Whether the provider's H and A filename variants identify the same physical fruit, and whether fruit identity persists across dates, remain unknown.

The .json.jpg/.json.json filename mismatch had caused 1,807 full-frame fallbacks in an earlier image-preparation step. The source worker also applies every supported EXIF transformation, including dimension-preserving ones, before bbox use. 10,463 selected source images have a non-identity EXIF transform. Private provenance records original JPEG SHA-256, raw size, effective EXIF tag, oriented size, original JSON SHA-256, bbox, padded box and final JPEG SHA-256. Effective tag 1 includes an absent-tag identity default; undefined tag 0 (110 records) remains native only when JSON dimensions and bbox checks agree. The visual sample includes every observed tag and all cultivars. Exactly one pair of records shares original JPEG bytes, but it remains in the same partition in both schemes; no regenerated crop bytes are duplicated. The source worker used Pillow 12.3.0 and JPEG 8.0. Final crop hashes are rechecked after transfer to the GPU server. A stratified visual audit examined 20 normalized source/bbox/crop pairs. Raw visual evidence remains private and is excluded from the supplement package.

An earlier image-preparation step had used PIL conversion without EXIF normalization. Partial extraction attempts were preserved and never used for fitting. Source reading and crop construction were performed on a local workstation, and model fitting and analysis on a GPU server; this arrangement did not alter the chosen rows, labels, group assignments or hyperparameters. Files documenting the source, orientation and scheduling amendments accompany the code. Provider label release and independent physical-fruit identity were not established.

## 2. Partitions and leakage controls

The existing group assignment is retained exactly: SHA-256('20261026:'+tree_key) within cultivar allocates rounded 20/10/10% to test/selection/calibration and the remainder to fitting. Same-row exclusions do not reorder or reassign groups. Only proxy fitting records enter optimization/target normalization; proxy selection records select checkpoints. Reference rows assigned to fitting or selection are not used for either purpose.

**`tables/combination_partition.csv`.** Cultivar–farm-combination partition after same-row reconstruction. Test boundaries exclude cultivar–farm combinations, whereas all five test farms recur in development through other cultivars.

| Role | Proxy-only records | Proxy trees | Same-row reference records | Reference trees |
|---|---|---|---|---|
| Fitting | 10,087 | 149 | 3886 | 144 |
| Checkpoint selection | 1344 | 19 | 483 | 18 |
| Calibration | 1294 | 19 | 505 | 19 |
| Test | 2542 | 38 | 988 | 38 |

Independent validation checks the hashes in the input-readiness metadata, all 21,443 regenerated crop hashes and image geometries, unique crop hashes, original group assignments, and exact source-linkage→split image/label/tag agreement. Predictions must agree one-to-one with frozen split keys, labels and group fields. Validation requires all six models and complete prediction records without duplicate rows.

## 3. Six completed fits and stochastic inference

All runs start from ImageNet ResNet-50 weights with dropout 0.2, a 2048-to-2 mean/logvariance head and fitting-only target normalization. Batch size 128, four data workers, AdamW backbone/head rates 10⁻⁵/10⁻⁴, weight decay 10⁻⁴ and cosine scheduling are fixed for 40 epochs. Epochs 1–5 use Huber loss; 6–40 use Gaussian NLL. The lowest proxy-only selection MAE from epochs 6–40 chooses the checkpoint. No reference, calibration or test outcome selects a model. Automatic CUDA mixed precision and gradient scaling were used. The run environment was Python 3.11.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.27, NumPy 2.4.4, CUDA 12.8 and NVIDIA GeForce RTX 5090; server Pillow 12.2.0 decoded the transferred crops. Source JPEG normalization was performed separately with the source-worker Pillow version reported in Section 1.

**`tables/model_fit_summary.csv`.** Summary of the six model fits. Scheme “combination” denotes the cultivar–farm-combination partition. Selected epochs are displayed one-based (stored code uses zero-based indices). MC SD is the mean saved per-record within-seed SD on the primary reference test. All times are actual completed-run measurements.

| Scheme | Seed | Selected epoch | Selection NIR MAE | Reference MAE | Mean MC SD | Seconds |
|---|---|---|---|---|---|---|
| tree | 0 | 37 | 1.434 | 1.432 | 0.137 | 504.0 |
| tree | 1 | 34 | 1.428 | 1.387 | 0.141 | 503.1 |
| tree | 2 | 34 | 1.427 | 1.418 | 0.137 | 516.1 |
| combination | 0 | 39 | 1.293 | 1.536 | 0.140 | 518.7 |
| combination | 1 | 32 | 1.293 | 1.469 | 0.143 | 504.9 |
| combination | 2 | 28 | 1.284 | 1.524 | 0.140 | 514.5 |

Each selected checkpoint passed repeated deterministic prediction, dropout-zero versus evaluation, dropout-zero repeat, nonzero dropout and unchanged-BatchNorm checks. Every saved array contains 20 passes with finite values and nonzero per-record SD. Repeating the writer's per-column mean and population-SD expressions matches the prediction CSV within 10⁻¹² °Brix; float64 means additionally satisfy the standard float32 summation-error bound. An initial vectorized float32 check used a different reduction order and was corrected after a maximum 3.8 × 10⁻⁶ °Brix rounding difference; no saved prediction or model was changed. This numerical validation amendment and its diagnostic are retained. Object order and reference labels are checked independently. Histories contain exactly 40 epochs and the stored checkpoint matches the minimum permitted selection MAE. Three seeds quantify limited fitting variability, not biological replication.

## 4. Complete cohort sensitivities and calibration

The primary cohort uses complete selected-row labels. Strict unique-NIR requires one NIR value across all rows sharing the historical composite key. The same rule is applied separately to reference calibration and test; proxy calibration remains unchanged. Models are not refitted by sensitivity cohort. NIR values are taken directly from the chosen image's JSON row and are never selected by model performance.

**`tables/cohort_comparison.csv`.** Cohort comparisons of accuracy, paired differences and coverage. Paired differences are RGB minus fitting cultivar-mean MAE. The cluster is tree for the tree test and farm for the combination test. Nominal 95% reference intervals are recalibrated within each cohort.

| Scheme | Cohort | Test n | Ref. cal n | RGB MAE | Cultivar MAE | Paired difference (95% interval) | Coverage | Mean width |
|---|---|---|---|---|---|---|---|---|
| tree | same_row_complete | 1191 | 633 | 1.407 | 1.226 | 0.181 (0.044 to 0.312) | 93.3% | 6.601 |
| tree | strict_unique_NIR | 1079 | 587 | 1.402 | 1.195 | 0.207 (0.065 to 0.347) | 93.2% | 6.530 |
| combination | same_row_complete | 988 | 505 | 1.504 | 1.340 | 0.164 (−0.161 to 0.533) | 87.2% | 5.744 |
| combination | strict_unique_NIR | 872 | 450 | 1.489 | 1.265 | 0.223 (−0.058 to 0.629) | 90.4% | 6.101 |

Accuracy, paired contrasts and coverage use 5,000 whole-cluster replicates; rank intervals use 1,000, seed 20261026. Reference coverage intervals hold both predictions and calibration multiplier fixed. They exclude calibration-estimation and refitting variation. Primary tree inference is accompanied by complete farm-cluster MAE, paired-contrast and coverage sensitivities, without selecting the more favorable interval. Primary combination inference resamples five farm codes; seven-combination resampling is separately labeled sensitivity. All five test farms are shared with development, so neither bootstrap creates external validation. Intervals are conditional and exploratory, not adjusted family-wise superiority tests.

The moment-matched total variance is mean conditional variance plus between-seed deterministic-mean variance. MC variance remains a separate diagnostic. Calibration uses rank ceil[(n+1)p], capped at n, of |target−prediction|/total SD. Repeated/clustered records and source-selection differences prevent a standard exchangeable distribution-free claim.

## 5. Ranking definitions and analyses not included

Seven scores are evaluated against all three error targets, all three budgets and both reference cohorts. Spearman uses average ranks; top-budget selection uses floor(fraction×n) and fractional boundary ties. Error events strictly exceed 2 °Brix. Model–NIR disagreement equals model–NIR error and is marked as an identity. Within-cultivar scores are descriptive; pooled mixture effects do not identify an instrument mechanism. Capture curves are retrospective and contain no acquisition/retraining loop.

Leave-one-cultivar-out models from earlier exploratory analyses predate the source quality control described in Section 1 and were not refitted; their results are not used in any result table, figure or conclusion. A complete leave-one-cultivar-out evaluation would require additional fits under a separate recorded protocol. Likewise, the present combination test is not a farm-disjoint external holdout.

## 6. Reproduction and public/private boundary

The scripts are rebuild_source_crops_local.py, train_corrected.py, launch_verified_training.sh, verify_corrected.py, analyze_corrected.py and render_figures_from_tables.py. The source worker needs authorized source images, the selected JSON annotations, the original manifest and the original group assignments. Raw inputs and private linkage are not included publicly. The training launcher requires a completed source audit and independently validated transferred inputs.

Document-style tables of this repository are `tables/combination_partition.csv`, `tables/model_fit_summary.csv`, `tables/cohort_comparison.csv`, `tables/underfit_three_seed_mean_display.csv`, `tables/underfit_by_seed_display.csv` and `tables/reference_test_label_summary.csv`; they hold rounded values as displayed. The machine-readable aggregate files are `tables/split_inventory.csv`, `tables/accuracy.csv`, `tables/uncertainty_three_targets.csv`, `tables/interval_coverage.csv`, `tables/paired_accuracy.csv`, `tables/seed_metrics.csv`, `tables/reference_pool_label_audit.csv`, `tables/capture_curves.csv`, `tables/cluster_error_sums.csv`, `tables/within_cultivar_scores.csv`, `tables/farm_overlap.csv`, `tables/underfit_three_seed_mean.csv`, `tables/underfit_by_seed.csv`, `tables/underfit_learning_curves.csv`, `tables/cohort_label_date_summary.csv`, `tables/checkpoint_selection_sensitivity.csv`. In these files the scheme code zone denotes the cultivar–farm-combination partition.

Anonymous cluster counts and absolute-error sums independently reconstruct pooled MAEs and paired cluster-bootstrap intervals within 10⁻¹². They do not reconstruct individual labels or the full ranking analyses. Figures are generated from aggregate tables as 600 dpi PNG and vector PDF. No raw image, JSON, crop, individual prediction, source filename/key, object-level label or trained checkpoint is part of the public package. AI Hub access and use conditions remain applicable.

## 7. Under-fitting sensitivity runs

A plan for these checks was written on 15 November 2026 before any of their results were computed and is supplied as protocols/underfit_plan.md; its checksum is recorded with the results. The primary results and the saved learning curves of the primary runs were already known, so the plan is a post hoc sensitivity plan and not a preregistration. The tree-scheme manifest, crops, partitions and selection rule of the primary analysis were used unchanged. Reference values and test records were not used to choose any setting, epoch or seed. Every planned run is reported.

Predictions recomputed from the three saved primary checkpoints agreed with the saved primary predictions on the reference test (maximum absolute difference 0.0). The capacity test (512 fitting records, no augmentation, dropout 0, learning rates 10⁻⁴/10⁻³, 200 epochs) reached a training MAE of 0.083 °Brix and a correlation of 0.991 on those records, against 1.051 for their constant mean; this measures training fit, not performance on held-out records.

**`tables/underfit_three_seed_mean_display.csv`.** Under-fitting sensitivity on the tree scheme, three-seed means. NIR columns are MAE in °Brix against the proxy label; the cultivar-mean row is the constant baseline computed from proxy fitting records. Reference columns use the 1,191 reference-test records from 45 trees. The paired difference is ensemble MAE minus cultivar-mean MAE with a 95% tree-bootstrap interval (2,000 resamples, seed 20261115).

| Setting | Fitting NIR MAE | Selection NIR MAE | Proxy-test NIR MAE | Reference MAE | Reference r | Paired difference (95% interval) |
|---|---|---|---|---|---|---|
| Cultivar mean | 1.492 | 1.541 | 1.379 | 1.226 | 0.274 | — |
| Primary (prespecified) | 1.332 | 1.426 | 1.286 | 1.407 | 0.274 | 0.181 (0.049 to 0.314) |
| Learning rate × 10 | 1.086 | 1.332 | 1.211 | 1.321 | 0.326 | 0.095 (−0.020 to 0.206) |
| 120 epochs | 1.236 | 1.379 | 1.242 | 1.384 | 0.309 | 0.158 (0.028 to 0.283) |
| 448-pixel input | 1.310 | 1.396 | 1.266 | 1.378 | 0.289 | 0.152 (0.028 to 0.278) |
| Learning rate × 10, 448-pixel input | 1.174 | 1.334 | 1.211 | 1.293 | 0.327 | 0.068 (−0.045 to 0.175) |

**`tables/underfit_by_seed_display.csv`.** Under-fitting sensitivity by seed. The selected epoch is the epoch with the lowest proxy selection MAE from epoch 6 through the final epoch.

| Setting | Seed | Selected epoch | Fitting NIR MAE | Selection NIR MAE | Reference MAE | Reference r |
|---|---|---|---|---|---|---|
| Primary (prespecified) | 0 | 37 | 1.342 | 1.434 | 1.432 | 0.267 |
| Primary (prespecified) | 1 | 34 | 1.335 | 1.428 | 1.387 | 0.283 |
| Primary (prespecified) | 2 | 34 | 1.331 | 1.427 | 1.418 | 0.270 |
| Learning rate × 10 | 0 | 10 | 1.191 | 1.355 | 1.309 | 0.327 |
| Learning rate × 10 | 1 | 13 | 1.154 | 1.380 | 1.375 | 0.316 |
| Learning rate × 10 | 2 | 22 | 1.033 | 1.355 | 1.365 | 0.311 |
| 120 epochs | 0 | 90 | 1.250 | 1.385 | 1.395 | 0.306 |
| 120 epochs | 1 | 100 | 1.245 | 1.384 | 1.380 | 0.309 |
| 120 epochs | 2 | 88 | 1.244 | 1.390 | 1.404 | 0.304 |
| 448-pixel input | 0 | 36 | 1.319 | 1.396 | 1.400 | 0.289 |
| 448-pixel input | 1 | 37 | 1.314 | 1.402 | 1.351 | 0.295 |
| 448-pixel input | 2 | 32 | 1.313 | 1.405 | 1.396 | 0.278 |
| Learning rate × 10, 448-pixel input | 0 | 10 | 1.183 | 1.336 | 1.358 | 0.322 |
| Learning rate × 10, 448-pixel input | 1 | 9 | 1.208 | 1.361 | 1.253 | 0.338 |
| Learning rate × 10, 448-pixel input | 2 | 10 | 1.202 | 1.365 | 1.342 | 0.305 |

Pearson correlations between the three-seed mean and the reference within cultivar were: primary AR 0.344, HJ 0.077, HR 0.077, SG 0.293; learning rate × 10 AR 0.191, HJ 0.260, HR 0.134, SG 0.417; 120 epochs AR 0.318, HJ 0.195, HR 0.113, SG 0.358; 448-pixel input AR 0.248, HJ 0.141, HR 0.067, SG 0.487; learning rate × 10 with 448-pixel input AR 0.137, HJ 0.271, HR 0.113, SG 0.388. With the higher learning rate the fitting MAE fell well below the selection MAE, so the selected checkpoints were early ones (epochs 9 to 22 of 40) and later epochs over-fitted the fitting trees.

The first 120-epoch run stopped at epoch 116 because prediction data loaders created at every epoch kept their worker processes and exhausted the open-file limit. The loader was changed so that prediction workers end with their loader, which does not alter any computation, and all three 120-epoch runs were then run from the start; the interrupted run is not used. Runs shared GPUs with unrelated jobs, so run times are not reported. These checks cover learning rate, training length and input resolution for one backbone. Uncertainty rankings and interval calibration were not recomputed for the alternative settings. The corresponding machine-readable tables are `tables/underfit_three_seed_mean.csv`, `tables/underfit_by_seed.csv`, `tables/underfit_learning_curves.csv`.

## 8. Post hoc label and date summaries

A descriptive audit on 18 November 2026 used the retained labels in the saved primary and sensitivity prediction files. Row keys and labels agreed across all 15 files. The available cohorts contained 8,919 proxy fitting records, 1,643 proxy selection records, 3,197 proxy-test records and 1,191 reference-test records. These summaries describe the selected tree-scheme cohorts; they do not describe the entire provider dataset. The files do not include reference-calibration records.

**`tables/reference_test_label_summary.csv`.** Reference-test label summaries. Mean ± sample SD and ranges are in °Brix. Dates are the date fields in the retained composite keys, not independently verified instrument timestamps.

| Cultivar | Records / trees | NIR mean ± SD | Reference mean ± SD | Reference range | Key dates in 2021 |
|---|---|---|---|---|---|
| All | 1191 / 45 | 12.591 ± 1.659 | 13.464 ± 1.294 | 9.2–18.0 | 09-12 to 11-06 |
| AR | 112 / 4 | 12.707 ± 0.972 | 12.894 ± 0.780 | 10.6–15.1 | 09-12 |
| HJ | 549 / 21 | 13.423 ± 1.588 | 13.827 ± 1.331 | 9.2–18.0 | 10-24 to 11-06 |
| HR | 457 / 16 | 11.496 ± 1.287 | 13.146 ± 1.254 | 9.9–17.5 | 09-12 |
| SG | 73 / 4 | 13.007 ± 1.135 | 13.612 ± 1.053 | 11.4–18.0 | 10-10 |

AR, Arisu; HJ, Fuji; HR, Hongro; SG, Shinano Gold. `tables/cohort_label_date_summary.csv` provides all cohort/cultivar summaries, including NIR ranges and retained-key date ranges. Proxy fitting and test records spanned 16 August–6 November 2021. The reference-test records were concentrated on 12 September for AR and HR, 10 October for SG, and 24 October–6 November for HJ. This date-composition difference does not identify its contribution to model error.

One retained proxy-only HR record had a numeric NIR value of 0.0 in all 15 original annotation JSONs associated with its retained source key; the corresponding reference fields were null. This establishes the origin of the value, not its measurement meaning. The available source archives did not contain instrument logs or a coding definition that distinguished a measured zero from an unmeasured or failed reading. The provider's meaning remains unknown, and the value was retained under the original finite-value rule without recoding.

The record belongs to the primary proxy test and to checkpoint selection in the combination partition. Excluding it only from the primary proxy-test metric changes ensemble MAE from 1.286457 to 1.283222 °Brix and cultivar-mean MAE from 1.379314 to 1.375995 °Brix (3,197 to 3,196 records), leaving their observed order unchanged. This exclusion does not change primary fitting, checkpoint selection, calibration or reference-test results.

A post hoc sensitivity analysis on 18 November 2026 replayed all 40 training epochs for each combination-split seed using the archived code and inputs. Training losses and selection MAEs matched the archived histories exactly at every epoch. All 320 checkpoint tensors per seed also matched the archived weights. Recalculating selection MAE after omitting the zero-valued record reduced the selection pool from 1,344 to 1,343 records. Seeds 0, 1 and 2 selected epochs 39, 32 and 28, respectively, with and without that record; the selected weights were identical. Thus this exclusion leaves the reported combination-test predictions and results unchanged. `tables/checkpoint_selection_sensitivity.csv` reports the aggregate verification. The source value's meaning remains unresolved despite the absence of an effect on checkpoint selection.
