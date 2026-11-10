# Pre-registration: Paper 1, second revision study (2026-11-05T10:59)
Written before any run on the seeds below. A file search of tmc_gaps2 found no outputs for seeds 250–254.

## Design
- Seeds 250–254 (fresh). Datasets: the 12 standard sets plus the 4 banks (16). Backbones QMF, PDF, MNL, BML, trained on the training split only. RAT is fitted exactly as in the frozen code (own probes, calibration split).
- Conditions: clean, seen, unseen (C3/B3), and swap (standard sets only).
- Statistics: dataset-then-seed cluster bootstrap, B = 5000, paired within (dataset, seed, backbone); medians per dataset.

## Confirmatory rules
- **R4 (transfer, equivalence).** The head-refitted law minus the shared probe law, unseen accuracy, pooled over the 4 backbones. Equivalence holds if the 90% cluster CI lies within [−0.3, +0.3] pt (TOST, α = 0.05 per side).
- **R5 (MER restoration).** Restored-input kNN minus raw-input kNN, unseen gain. Holds if the 95% cluster CI is > 0 (pooled over backbones).
- **R6 (complementarity).** The combined surprisal (mean of the validity surprisal and the clean-calibrated JSD surprisal) minus validity alone, induced-error AP. Holds if the 95% cluster CI is > 0 (pooled over PDF and MNL heads, as in Table 4).
- **R7 (sample-level anchor, post hoc law).** β_m = 1 unless max_m v_m exceeds its clean-calibration 90% quantile. Holds if the clean cost is reduced by ≥ 50% relative to RAT while ≥ 80% of the unseen gain is kept (pooled over backbones).

## Descriptive analyses (no decision rule)
- Gate decomposition: NLL of native, T(x)·native, normalised RAT and RAT; numeric check that RAT and normalised RAT have identical accuracy; count of T(x) = 0 (native fallback).
- Validity variants: raw kNN, restored kNN, MER step, range, and combinations; mean vs max vs joint-empirical aggregation.
- Laws: shared probe law, head-refitted law, fusion-loss-calibrated law, oracle two-state; per-bin β (probe vs heads).
- Logit-scale stress test: one modality's logits × (0.5, 2).
- Oracle-approximation error E(C), E(v), E(C, v).
- Empirical curvature, logit bound and class spread.
- Calibration fraction r_cal ∈ (0.04, 0.08, 0.16, 0.24) (12 standard sets, PDF).
- Latency profile; dataset inventory.

## Reporting
All rules are reported whatever the outcome. Earlier pre-registered results (PH1–PH4, R1–R3) are not changed.
