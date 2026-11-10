# Pre-registration — confirmatory plug-in study (Paper 1)

Written 2026-11-04T19:16 before any run with seeds 230–239 (verified unused on the server).

## Question (RQ1/RQ4)
Does validity-conditioned trust (RAT β), inserted as a multiplicative gate on a published backbone's own fusion weights,
fused(x) = Σ_v w_v^native(x) · β_v(x) · z_v(x),
preserve clean accuracy and improve accuracy under unseen damage, without modifying or retraining the backbone?

## Fixed design
- Code: frozen mvbench modules (mv_method.py, mv_bench.py, mv_bank.py, mv_baselines2.py; HASHES_v2), driver p1_plugin_conf.py.
- Seeds 230–239 (10). Datasets: the 12 standard sets plus 4 banks (SUN RGB-D, MILK10k, UPMC-Food101, CREMA-D).
- Backbones (official code ports, each fuses as Σ_v w_v z_v): PDF (ICML 2024; confidence-driven weights), MNL (NeurIPS 2025; uniform weights),
  QMF (ICML 2023; energy confidence weights), BML (CVPR 2026; learned reliability router).
- Primary protocol "trainonly": backbone trained on the train split; RAT fitted on the held-out cal split using the backbone's own per-view heads
  as experts (RRT v2 config floor_mode='untempered'). Backbone learning rates: lrs_v2.json / lrs_bank.json (same as the confirmatory v2 run).
- Control protocol "traincal": backbone trained on train+cal, RAT still fitted on cal (contaminated calibration; expected to weaken β).
- Conditions: standard C0 clean, C3 unseen damage, C1 ECML conflict, C4 swap; banks B0 clean, B2 seen, B3 unseen, B3_unseen_all.
- Gate: no renormalisation (non-competitive), exactly as in the dev study (seeds 220–222).

## Hypotheses and decision rules (α = 0.05; seed-level pooled means over the 16 datasets; sign-flip test over seeds; bootstrap 95% CI)
- **PH1 clean non-inferiority**: for each backbone, pooled Δ_clean = gated − native (trainonly) has lower 95% CI bound ≥ −0.3 pt. Pass if 4/4 backbones.
- **PH2 unseen-damage gain**: for each backbone, pooled Δ_unseen (C3/B3) > 0, Holm-adjusted p < 0.05 over the 4 backbones. Pass if ≥ 3/4.
- **PH3 gain follows trust gap**: Spearman ρ between per-dataset Δ_unseen (mean over backbones) and the confirmatory-v2 RAT gain
  (RRT − RRT-noRAT, seeds 190–209; values frozen in rat_gain_predictor.csv) is > 0 with one-sided p < 0.05 (n = 16).
- **PH4 β is a better damage detector than native weights**: for PDF, QMF and BML, the per-dataset AUROC (damaged view vs intact view, unseen
  damage, damaged samples' views vs intact views) of −β exceeds that of −w^native; pooled mean difference > 0 with sign-flip p < 0.05 per backbone
  (Holm over 3). Pass if ≥ 2/3. MNL has constant weights and is excluded.

## Secondary (descriptive, no decision)
Swap/conflict conditions (C1, C4); gated(trainonly) vs native(traincal); traincal gated; all-damaged B3; per-dataset tables; NLL.

## Predictor values frozen now (RAT gain, confirmatory v2)
ds
brca           0.304
caltech20      1.067
handwritten    0.375
landuse        7.476
leaves         4.667
mosi          -1.443
mustard        3.551
outdoor        2.268
pie            1.912
rosmap         1.792
scene15        6.767
urfunny        1.276
crema          2.837
food           0.284
milk           1.180
sun            0.259
