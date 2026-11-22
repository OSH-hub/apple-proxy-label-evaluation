# Plan: under-fitting checks for the proxy-trained RGB model (P1)

Written on 15 November 2026 before any result of these checks was computed. The runner records the SHA-256 of this file. The primary analysis (six models, tree and cultivar–farm-combination schemes) is not changed, and no alternative setting replaces it. Every check and every seed listed here is reported, including unfavourable outcomes. This is a post hoc sensitivity plan, not a preregistration: the primary results and the saved learning curves of the primary runs were already known.

## Question

The primary RGB ensemble was no more accurate against reference sugar than a cultivar-mean baseline. The manuscript states that under-fitting of the backbone cannot be excluded because no alternative learning rate or training length was evaluated. These checks ask whether the primary setting under-fitted the proxy target, and whether plausible alternative settings change the conclusion.

## Data and partitions

The tree scheme of the primary analysis, unchanged: the same manifest, the same fitting, checkpoint-selection, calibration and test trees, the same regenerated crops. Weights are updated only on proxy-only fitting records; checkpoints are selected only on proxy-only selection records (lowest selection MAE among epochs 6 to the last). Reference values and test records select nothing: no setting, epoch or seed is chosen using them.

## Checks

D1. Learning curves of the three primary tree-scheme runs from their saved histories (training loss and proxy selection MAE by epoch).

D2. Fit of the primary checkpoints to their own fitting records: deterministic predictions (evaluation transform) of the three saved primary checkpoints on the 8,919 proxy fitting records, compared with their selection and proxy-test MAE and with the cultivar-mean baseline on each set.

D3. Capacity test: one model (seed 0) trained on the first 512 proxy fitting records in manifest order, evaluation transform only (no augmentation), dropout 0, backbone/head learning rates 10⁻⁴/10⁻³, 200 epochs, otherwise as primary. Reported: training MAE on those 512 records. This tests whether the pipeline can fit labels at all; it is not a predictive model.

D4. Alternative settings, each with seeds 0, 1 and 2, everything else as in the primary runs (ImageNet ResNet-50 from timm, dropout 0.2, batch 128, AdamW, weight decay 10⁻⁴, cosine schedule, Huber for epochs 1–5 then Gaussian negative log likelihood, mixed precision):

| Code | Change from primary |
|---|---|
| LR10 | backbone/head learning rates 10⁻⁴/10⁻³ (ten times higher) |
| EP120 | 120 epochs |
| RES448 | 448-pixel input (random resized crop 448, scale 0.6–1; evaluation resize 512, centre crop 448) |
| LR10_RES448 | both LR10 and RES448 |

For each run: training loss and selection MAE by epoch, selected epoch, and deterministic predictions of the selected checkpoint on the fitting, selection, proxy-test (3,197) and reference-test (1,191) records. For each setting the three seed predictions are averaged as in the primary analysis.

## Reported quantities

- Proxy target: MAE and Pearson correlation with NIR on fitting, selection and proxy-test records, for single seeds and the three-seed mean, next to the cultivar-mean baseline.
- Reference target: MAE and Pearson correlation with reference sugar on the 1,191 reference-test records for the three-seed mean, next to the primary ensemble (1.407 °Brix, 0.274) and the cultivar-mean baseline (1.226 °Brix, 0.274), with the paired difference to the cultivar-mean baseline and a 95% tree-bootstrap interval (2,000 resamples of test trees, seed 20261115).
- Within-cultivar Pearson correlation of the three-seed mean with reference sugar.

## Interpretation rules

- Under-fitting of the proxy target is indicated if fitting-record MAE is close to the cultivar-mean baseline and falls clearly under an alternative setting together with selection MAE. A large gap between fitting and selection MAE instead indicates limited generalisation across trees.
- An alternative setting is described as changing the conclusion only if its three-seed mean is more accurate than the cultivar-mean baseline on the reference test with a tree-bootstrap interval excluding zero. Otherwise the conclusion is described as unchanged for the settings tried.
- These checks cover learning rate, training length and input resolution for one backbone. Other backbones, other augmentations and other loss functions are not covered.
- Runs share GPUs with unrelated jobs; run times are not reported as benchmarks.
