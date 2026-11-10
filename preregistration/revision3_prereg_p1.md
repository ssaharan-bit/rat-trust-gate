# Pre-registration: Paper 1, replication study 3 (2026-11-05T17:06)
Written before any run on seeds 255–259; a file search of tmc_gaps2 found no outputs for these seeds.

## Design
Seeds 255–259; 16 datasets; QMF, PDF, MNL and BML trained on the training split. RAT-full is the frozen pre-registered method. Cluster bootstrap over datasets, then seeds, B = 5000, paired within (dataset, seed, backbone).

## Confirmatory rules
- **R8 (anchor replication).** The sample-level anchor (β = 1 for all modalities unless max_m v_m exceeds its clean-calibration 90% quantile) reduces the pooled clean cost of RAT-full by ≥ 50% and keeps ≥ 80% of its pooled unseen gain.
- **R9 (RAT-kNN equivalence).** RAT-kNN (law on the surprisal of the raw-input kNN statistic only) minus RAT-full, unseen accuracy, pooled: 90% CI within ±0.3 pt.
- **R10 (cross-fitting).** Two-fold cross-fitted RAT minus RAT-full, unseen accuracy, pooled: 90% CI within ±0.3 pt. Cross-fitting: the calibration split is divided into two stratified halves. Each half is used to fit RRT (probe selection, temperature, MER selection); the law is then fitted on the other half. β is the mean of the two cross-fitted laws.

## Descriptive (no decision rule)
RAT-kNN + anchor; the single-half law (no cross-fitting); a Mahalanobis gate (Ledoit–Wolf covariance, raw features); a confidence-law gate (same isotonic machinery on the probe's 1 − MSP); a combined law ((v + u_JSD)/2, JSD among probe predictions); the distance ratio E[d_k | damaged] / E[d_k | intact].

## Amendment A1 (2026-11-05T17:08), before any run on seeds 255–259
A dev-seed smoke test (ROSMAP, seed 221) showed that a half calibration split (about 12 samples) leaves no bin with 20 samples, and the frozen law then raises an error. For the two half-split fits only, the law falls back to a single CE-optimal temperature; the number of fallbacks is recorded. RAT-full itself is unchanged.

## Amendment A2 (2026-11-05T17:43)
The first launch failed on 100Leaves and PIE, which have one calibration sample per class: per-class halving left one half empty. The job was stopped. No result file was opened. The halves are now formed by alternate assignment along a class-ordered permutation (approximately stratified, never empty). All 80 runs are restarted from scratch into res/rev3b.
