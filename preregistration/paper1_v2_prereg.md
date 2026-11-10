# Paper 1 v2 pre-registration: "Machine metacognition under sensor damage: confidence reports difficulty, validity reports damage"
Written 2026-10-28 before any run on seeds 130–139. Development evidence: seeds 120–124 (p1_diagnosis.md, p1_signals_dev.md, p1_fusion_dev_tests.csv). Datasets SUN RGB-D, MILK10k, UPMC Food-101, CREMA-D (9 modalities). Frozen encoders and banks as in Paper 1 v1. Calibration uses clean data and seen families A, B only. Tests use unseen families C, D.

## Derivation (five-step)
1. **Phenomenon.** Reliability estimators that predict errors well on pooled data still over-trust a damaged modality in fusion, silently and mostly under evidence removal (phenomenon audit; MILK meta −27 pts).
2. **Unknown quantity.** Which error component a reliability score tracks: *intrinsic* (the clean twin is also wrong) or *degradation-induced* (the clean twin is right). Also, which of the two a fusion temperature should encode.
3. **Hypothesis.** Confidence-type scores read the strength of the surviving evidence. This is a positive-evidence bias, so they track difficulty and are blind to, or inverted by, damage. Corruption acts as a systematic bias (rotation of the discriminant evidence), not as random error, so internal-consistency scores are also blind. Only an external criterion, the clean-data class geometry (typicality, "validity"), tracks damage. Logits already carry difficulty, so a fusion temperature driven by confidence double-counts difficulty and over-trusts damaged modalities. A temperature driven by validity does not.
4. **Disproving experiments.** H1–H5 below.
5. **Decision rule.** Assign reliability channels by role. Validity sets the fusion temperature; confidence stays inside the logits. Method: **Role-Assigned Trust (RAT)**.

## Definitions (fixed)
- Confidence channel: MSP. Confidence class: mean AUROC of MSP, negative entropy, negative energy, TMC belief.
- Validity channel (typicality composite): mean of z-scored negative Mahalanobis, kNN similarity and prototype alignment. z-scores use clean cal rows. Validity class: mean AUROC of those three.
- Trust law: 1-D increasing isotonic map from the condition-mean channel score to log β*(condition), fitted on cal conditions (clean + A, B; ≥ 20 rows each).
- law_conf / law_val: fusion temperatures β_m = exp(law(score)), multiplied by τ (2T convention).
- **RAT**: β_m = VRE_m^(1−μ) · law_val,m^μ. μ ∈ {0, .25, .5, .75, 1} is chosen by leave-one-family-out (A↔B) fused NLL on cal instances, shared across modalities of a dataset. μ = 0 falls back to VRE. (VRE's own PCAR law is not refitted inside the folds; it uses cal data only.)
- Fusion test: two draws per test sample, each modality corrupted with probability 0.5 by a C or D variant. Clean test: no corruption. Accuracy in pts.

## Hypotheses (seeds 130–139; exact one-sided sign-flip test over 10 seed-level means; bootstrap 95% CI)
- **H1 (primary 1, double dissociation).** Interaction = [AUROC(conf class) − AUROC(validity class)] on intrinsic error (clean rows) − the same difference on induced error (C/D rows with a correct clean twin). Mean over 9 modalities > 0, SESOI 0.05, p < 0.05, and > 0 in ≥ 6 of 9 modalities. Dev: +0.179, 8/9.
- **H2 (primary 2, channel role in fusion).** law_val − law_conf, unseen-family accuracy pooled over 4 datasets > 0, SESOI 0.5 pt, p < 0.05, and > 0 on ≥ 3 of 4 datasets. Dev: +1.36 (CREMA +2.41, Food +0.58, MILK +1.94, SUN +0.51).
- **H3 (method).** (a) Non-inferiority RAT − VRE ≥ −0.10 pt (lower 95% CI bound). (b) Superiority RAT − 2T > 0, SESOI 0.3 pt. (c) Clean accuracy RAT − 2T ≥ −0.3 pt. Dev expectation is honest: VRE × law_val − VRE = −0.04 (Food −0.47), VRE − 2T = +0.20. A null on RAT − VRE is the expected outcome and will be reported as such.
- **H4 (reliability ≠ validity).** Induced-error AUROC of split-half evidence reliability (SHER, Spearman–Brown) < 0.5, mean over 9 modalities, p < 0.05. Dev: 0.20–0.69 per modality, below 0.5 on 7 of 9.
- **H5 (over-trust mechanism).** On held-out C conditions, signed log-temperature error (law prediction − log β* of that condition) is smaller (less over-trust) for law_val than for law_conf. Pooled over modalities, p < 0.05.

## Kill and fallback rules
- If H1 fails, the "two channels" framing is dropped and Paper 1 reverts to the v1 benchmark framing.
- If H2 fails, the double-counting claim is withdrawn.
- If H3a fails, RAT is not proposed and VRE stays the Paper 1 method.
- No new estimators, channels or datasets are added after this document. Any deviation is listed in the report.
