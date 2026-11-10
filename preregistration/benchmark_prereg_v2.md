# Pre-registration v2: expanded benchmark (12 datasets) and Paper 1 frozen-bank comparison

Written 2026-11-02, after the RRT v2 dev round (rrt_v2_dev.md) and before any confirmatory run.
- Confirmatory seeds: standard benchmark **190–199**; banks **200–209**.
- LR selection seeds: 150–152 (standard benchmark) and 300–301 (banks). None of these are used for testing.
- v1 pre-registration (benchmark_prereg.md) and Amendment 1 still govern everything not changed here.

## 1. Method under test
- **RRT v2** = RRT v1 (frozen defaults, expert grid EG5) with `floor_mode='untempered'`. The contamination is applied to the untempered expert posterior before β·τ tempering.
- No other change. The code hash is recorded in HASHES_v2.txt before launch.
- Fix B (gated/aug experts) was rejected on dev and is not used.

## 2. Part A: standard multi-view benchmark (Paper 2 / thesis)

**Datasets (12).** The selection criterion was fixed before results.
- Original six: HandWritten, Scene15, Caltech101-20, LandUse-21, BRCA, ROSMAP.
- Six added:
  - PIE (3 views, K=68);
  - 100Leaves (3×64, K=100);
  - OutdoorScene (4 views, K=8);
  - CMU-MOSI, UR-FUNNY and MUStARD: MultiBench official splits, with each modality time-pooled to mean+std. MOSI is binarised at label ≥ 0.

**Baselines (official code, V-view adapters, no change to the losses).**
- Earlier set: TMC, ETMC, RCML, QMF, MMDynamics.
- New 2024–26 A*/Q1 methods: TMCEK (ICML 2025), GTMC-HOA (IEEE TMM 2026), TUNED (AAAI 2025), BML (CVPR 2026), PDF (ICML 2024), MNL (NeurIPS 2025).
- Every official baseline also runs as +aug.
- Simple baselines: Concat, LateAvg, PoE, 2T.
- LR is chosen per dataset from base×{1,3,10} by clean calibration NLL on selection seeds; grid edges are extended to ×{30,100}.

**Conditions.** C0–C4 as in v1. The "best baseline" is chosen post hoc per dataset × condition over all baselines and variants.

| id | claim | pass rule |
|---|---|---|
| **H1 (primary)** | C3: RRT v2 − best baseline > 0 | pooled CI lower bound > 0 AND Holm-significant positive on ≥ 8/12 datasets |
| H2 | C2 (mean over σ): RRT v2 − best baseline > 0 | pooled CI lower bound > 0, positive on ≥ 8/12 |
| H3 | C1: RRT v2 − RCML > 0 | pooled CI lower bound > 0 |
| H4 | C0: non-inferior to best baseline | one-sided 95% bound > −0.5 pt on ≥ 10/12 |
| H5 | C3: NLL(RRT v2) < NLL(best-accuracy baseline) | ≥ 8/12 |
| H6 (v2 replication) | τ swap cost = acc(v2 −τ) − acc(v2) on C4 | pooled mean ≤ 50% of v1's (v1 run in the same job) |
| H7 (new SOTA) | C3: RRT v2 − best of {TMCEK, GTMC-HOA, TUNED, BML, PDF, MNL} (+aug) | pooled CI lower bound > 0 |
| B1 (boundary) | C4 | gain in C4 < gain in C3; LateAvg/BML may lead on swaps |
| B2 (v2 cost) | C4 on LandUse and Scene15 | v2 < v1 (predicted from dev) |

**Directional predictions written now.**
- Clean HandWritten and PIE are saturated (ties).
- MUStARD (138 test samples) and ROSMAP are underpowered.
- On time-pooled MOSI, UR-FUNNY and MUStARD, text dominates, so the headroom for trust weighting under C3 depends on whether text is damaged.
- BML (swap augmentation in training) is predicted to be the strongest baseline on C1/C4.

**Ablations (C0–C4).** −MERA, −RAT, −τ, −ε, static τ (no release), softmax attention, and v1 (tempered floor).

## 3. Part B: Paper 1 comparison on frozen-encoder banks with real raw-signal corruptions

**Banks.**
- SUN RGB-D (rgb, depth; K=19)
- MILK10k (clin, derm, meta; K=11)
- UPMC-Food101 (img, text; K=101)
- CREMA-D (audio, video; K=6)

**Corruption families.** These are raw-signal corruptions, applied before the frozen encoder.

| Code | Family | Role |
|---|---|---|
| A | noise | **seen** (calibration / augmentation only) |
| B | blur / compression | **seen** (calibration / augmentation only) |
| C | occlusion / dropout | **unseen**, test only |
| D | environment | **unseen**, test only |

**Protocol (mv_bank.py).**
- Features are z-scored per modality using clean train+cal.
- Test conditions:
  - **B0** clean;
  - **B2** seen families: 50% of samples, a random non-empty proper subset of modalities replaced by a real corrupted variant;
  - **B3** unseen families (C/D), same rule (**primary**);
  - **B3-all** unseen families on 100% of samples.
- RRT calibrates only on the calibration split's real A/B variants.
- Baselines train on train+cal. Their +aug variants add a copy in which 50% of samples have modalities replaced by real variants from train/cal (A/B only, which are the only families present there).
- **Training budget (step-matched).** Banks are 2–12× larger than the standard sets. Every iterative trainer, RRT experts included, uses epochs = max(30, round(E_official · min(1, 2000/n))). This keeps the number of gradient updates of the official schedule at n = 2000. MMDynamics is full-batch and unchanged.

**Comparators.** QMF, PDF, MNL, TMC, ETMC, RCML, TMCEK, GTMC-HOA, TUNED, BML, MMDynamics (+aug each), plus Concat, LateAvg, PoE and 2T.
- **Not run:**
  - UDML: its pipeline is tied to its own encoders.
  - EAU and DEL: no public code.

| id | claim | pass rule |
|---|---|---|
| **P1-H1 (primary)** | B3: RRT v2 − best baseline > 0 | pooled CI lower bound > 0 AND Holm-significant positive on ≥ 3/4 banks |
| P1-H2 | B3-all: RRT v2 − best baseline > 0 | pooled CI lower bound > 0 |
| P1-H3 | B3 NLL(RRT v2) < NLL(best-accuracy baseline) | ≥ 3/4 banks |
| P1-H4 (RQ1) | β flags damaged modalities in B3 | AUROC ≥ 0.85 on ≥ 3/4 banks |
| P1-H5 | B0 non-inferiority | one-sided 95% bound > −1.0 pt on ≥ 3/4 banks |
| P1-B1 (boundary) | B2 seen families | +aug baselines, trained on A/B, are predicted to match or beat RRT; reported as confirmed or refuted |

**Statistics.**
- Paired over seeds, with identical substitutions for all methods within a seed.
- Bootstrap 95% CI, exact one-sided sign-flip p, Holm within each hypothesis.

**Reporting rule.** Every dataset, condition, baseline and hypothesis is reported, pass or fail.
