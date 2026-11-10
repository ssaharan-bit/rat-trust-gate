# Pre-registration: Paper 1 revision study (response to external review)
Written 2026-11-05T07:15, before any run on seeds 240–249 (never used before; verified on the server).
Frozen modules: mvbench (HASHES_v2). Drivers: mvdiag/p1_revision.py, p1_sensor.py, p1_severity.py, p1_overhead.py.
Everything here is reported **separately** from the confirmatory plug-in study (seeds 230–239). That study's results, including the failed PH1, are unchanged.

## Study A: validity-estimator ablation, transfer, error prediction, atypical-clean (seeds 240–244; 16 datasets; backbones PDF and MNL, train-only)
β variants inserted into the same gate fused = Σ w_native · β · z:
- RAT (all 3 statistics, isotonic law);
- kNN only; out-of-range only; MER-magnitude only; kNN+range; kNN+MER; range+MER;
- linear-logistic law β = 2σ(a + b·q) fitted by calibration NLL;
- fixed β (one NLL-optimal constant per view);
- oracle two-state β (NLL-optimal temperature of intact / damaged test views, using test labels: a ceiling);
- clean-anchored law: β = 1 for surprisal ≤ 90th percentile of clean-calibration surprisal, RAT law otherwise (post hoc variant motivated by the review);
- reference bank size N ∈ {256, 512, 1024, 2000, 4000} (capped at the training size).
Cross-backbone transfer: the RAT law fitted with PDF heads, applied to the MNL gate, and the reverse.

Decision rules (pooled seed-level mean over 16 datasets, 95% bootstrap CI over seeds):
- **R2 (transfer):** under unseen damage, |transferred-law gate − own-law gate| ≤ 0.3 pt for both directions.
- **R3 (clean-anchored law, post hoc):** relative to RAT, reduces the clean cost (gate − native, clean) by ≥ 50% while retaining ≥ 80% of the unseen-damage gain, for both backbones.
Descriptive only: component ablation, law form, fixed/oracle β, bank size.

Error prediction (descriptive). Population: all (sample, view) pairs in the unseen-damage condition. Target: induced error (the clean twin view is predicted correctly by the backbone head, the damaged view is not). Scores: validity surprisal (RAT), kNN-only surprisal, 1−MSP, entropy, energy (−logsumexp), cross-modal Jensen–Shannon disagreement. Metrics: AUROC, average precision, AURC.

Atypical-clean (descriptive): on clean test data, samples with any view's surprisal above that view's clean-calibration 90th percentile; gate − native accuracy on these against the rest.

## Study B: natural sensor shift (SUN RGB-D, seeds 240–249; backbones QMF, PDF, MNL, BML)
Train and calibrate on Kinect v1/v2 rows only (z-scoring refitted on Kinect train+cal). Test on RealSense + Xtion rows: clean, and with unseen families (B3). Reference: Kinect test rows.
- **R1:** on clean new-sensor data the gate's accuracy change (gate − native) has 95% CI lower bound ≥ −0.3 pt (pooled over backbones), and we report clean β on new sensors vs Kinect.

## Study C (descriptive): severity sweep, overhead, hierarchical statistics
- Severity sweep: banks, RRT experts, seed 240. Per modality × family × severity 1–5: accuracy, MSP, entropy, surprisal and β.
- Overhead: native vs gated fusion latency (ms/1000 samples), peak GPU memory, added parameters, reference-bank memory.
- Hierarchical statistics on the existing confirmatory plug-in data: a dataset-cluster bootstrap (resampling datasets, then seeds), between-dataset and between-seed variance, per-dataset effects.

## Amendment A1 (2026-11-05T07:23), before any run on seeds 240–249
A dev-seed smoke test (seed 220) showed that the frozen RRT ignores the `_external_experts` option: the RAT law is always fitted with RAT's own per-view probes (one temperature-scaled MLP per modality, trained on the training split), never with the backbone heads. This holds for the confirmatory plug-in study (seeds 230–239) as well. The same β is therefore applied to every backbone and the backbone is never queried, so the planned PDF→MNL 'transfer' is identical by construction.
**R2 is replaced by R2′:** a law *recalibrated on each backbone's own heads* (mvdiag RRT with the external-experts hook) does not beat the shared RAT law by more than 0.3 pt under unseen damage (pooled, both backbones). If it does not, a shared, backbone-free β is sufficient.
