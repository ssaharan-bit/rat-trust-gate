# RAT: a plug-in trust gate for multimodal logit fusion

Code, pre-registration documents and per-seed results for

> S. Saharan and R. Balasubramanian, *Validity, not confidence: a plug-in trust gate for robust multimodal fusion under unseen corruption*, submitted to Pattern Recognition Letters.

**Role-Assigned Trust (RAT)** sets each modality's logit temperature from its *distributional validity*, that is, from how typical its input features are of clean training data. Confidence stays in the logits. A decreasing isotonic law, fitted once on a labelled calibration split with auxiliary probes, maps a validity score to a temperature $\beta_m$. The gate multiplies each modality's logits by $\beta_m$ inside the backbone's own additive fusion, $\sum_m w_m\beta_m z_m$, and does not retrain the backbone.

## Contents

| Folder | What it holds |
|---|---|
| `rat/` | Frozen method and benchmark code used by every study: `mv_method.py` (RAT/MER, validity statistics, law), `mv_bench.py` (standard datasets, corruptions, training), `mv_bank.py` (frozen-encoder banks), `mv_baselines2.py` (wrappers around the official code of the published backbones). |
| `studies/` | One script per pre-registered study (table below). Each runs one dataset and one seed and writes a JSON file. |
| `features/` | `audit_extract.py`: builds the four frozen-encoder banks (DINOv2-base, BERT-base, wav2vec 2.0-base) from the raw datasets, with the corruption families applied to raw signals. `crema_seq_extract.py`: frame-level CREMA-D features; set `RAT_RAW` to the folder with the raw files. |
| `scripts/` | `setup_external.sh` (clones the official baselines at the exact commits), schedulers, `condense_results.py`, `reproduce_tables.py`, `make_figures.py`. |
| `results/raw/` | Per-dataset, per-seed JSON outputs of every study used in the paper. |
| `results/tables/` | Condensed per-seed rows and the summary tables behind every number in the paper and supplement. |
| `preregistration/` | The time-stamped study documents and `SHA256SUMS`. The hashes match the pre-registration record in Section S10 of the supplement. |

## Installation

Tested on Ubuntu 22.04 with one NVIDIA RTX 5090 (driver 570.133), Python 3.10.18, PyTorch 2.7 (CUDA 12.8), NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.0, pandas 2.2.3 and transformers 5.3.0.

```bash
conda env create -f environment.yml && conda activate rat
bash scripts/setup_external.sh        # official code of QMF, PDF, MNL, BML and the other baselines, pinned commits
```

The official repositories are not redistributed here; they keep their own licences.

## Data

All datasets are public. See `DATA.md` for download links and the expected folder layout.

- **Standard multi-view sets:** Handwritten, Scene15, Caltech101-20, LandUse-21, PIE, 100Leaves and Outdoor Scene (the `.mat` files distributed with the trusted-fusion code); BRCA and ROSMAP (the MOGONET splits); and CMU-MOSI, MUStARD and UR-FUNNY (MultiBench features).
- **Frozen-encoder banks:** SUN RGB-D, MILK10k (the 5,240 public lesions), UPMC Food-101 and CREMA-D. Build them with `python features/audit_extract.py --ds {sun,milk,food,crema}`. The splits are drawn once with seed 0 (`make_splits`).

## Reproducing the paper

**Step 1. Numbers only (no GPU, about 1 min).** The per-seed rows are included:

```bash
python scripts/reproduce_tables.py      # Tables 2-3, Tables S3-S4, damage-detection AUROC, PH3 correlation
python scripts/make_figures.py          # Fig. 2
python scripts/reproduce_table_s5.py    # Table S5 (all official-code methods on the four banks)
```

Means and medians are deterministic and match the paper exactly. Bootstrap intervals use B = 5000 resamples with seed 0; they agree with the paper to Monte Carlo error (about ±0.05 points).

**Step 2. Re-running the studies (GPU).** Each study script takes `--ds DATASET --seed SEED --out FILE.json`. Dataset keys:
- standard sets: `handwritten scene15 caltech20 landuse pie leaves outdoor brca rosmap mosi mustard urfunny`;
- banks: `bank:sun bank:milk bank:food bank:crema`.

| Paper item | Study (pre-registration) | Script | Seeds |
|---|---|---|---|
| Table 2, Fig. 2, PH1–PH4, Table S2, Fig. S1 | Confirmatory plug-in study (`plugin_prereg.md`, amendment 1) | `studies/p1_plugin_conf.py` | 230–239 |
| Table 2 (damaged-sample gain, ΔECE, ΔBrier), Tables S3–S4 | Re-evaluation of the confirmatory runs (same seeds and splits; adds calibration metrics and damaged-subset accuracy) | `studies/p1_plugin_eval.py`, `scripts/sched_eval.sh scripts/jobs_eval_230_239.txt`, `scripts/condense_eval.py` (raw files in `results/raw/ploteval`) | 230–239 |
| §5.4 sensor shift and atypical inputs, Fig. S2, Tables S8, S14, S21, S22, rules R1–R3 | Revision study 1 (`revision_prereg_p1.md`) | `studies/p1_revision.py`, `p1_sensor.py --test new`, `p1_severity.py` | 240–249 |
| Table 3(a), §5.2–5.3, Tables S6, S7, S9–S13, S19, S20, rules R4–R7 | Revision study 2 (`revision2_prereg_p1.md`) | `studies/p1_rev2.py` | 250–254 |
| Calibration size (Table S18) | Revision study 2 | `studies/p1_calsize.py --rcal {0.04,0.08,0.16,0.24}` | 250–254 |
| Cost (§5.5, Table S23) | Revision study 2 | `studies/p1_latency.py`, `studies/p1_enc_time.py` | 250 |
| kNN scalability (§5.5, Table S24) | Benchmark on random data | `studies/bench_knn_scaling.py` (output `results/tables/bench_knn.json`) | — |
| Table 3(b), revision study 3, rules R8–R10, alternative gates, distance ratio (Tables S15–S17) | Revision study 3 (`revision3_prereg_p1.md`) | `studies/p1_rev3.py` | 255–259 |
| Dataset inventory (Table S1) | — | `studies/p1_inventory.py` | 250 |
| Bank comparison with all 11 official-code methods (Table S5, incl. Amendment-2 re-runs), confidence-driven law (Section S5) | `benchmark_prereg_v2.md`, `paper1_v2_prereg.md` | `rat/mv_bench.py`; rows in `results/tables/bank_benchmark_*.json.gz` (`scripts/condense_bankbench.py`); table: `scripts/reproduce_table_s5.py` | 200–209, 130–139 |

Example (run from `studies/`, which is where the scheduler expects the scripts):

```bash
cd studies
export RAT_DATA=/path/to/data RAT_BANKS=/path/to/data/banks
python p1_plugin_conf.py --ds landuse --seed 230 --out res/plugconf/pc_landuse_230.json   # one run
bash ../scripts/sched_plug.sh ../scripts/jobs_plug_230_239.txt                         # all 160 runs of the confirmatory study
cd .. && python scripts/condense_results.py studies/res/plugconf                         # JSON -> one row file
bash ../scripts/sched_eval.sh ../scripts/jobs_eval_230_239.txt                         # re-evaluation with ECE/Brier (run from studies/)
cd .. && python scripts/condense_eval.py studies/res/ploteval                           # JSON -> results/tables/plugin_eval_rows_230_239.csv.gz
```

The 4 % calibration-fraction run for ROSMAP, seed 254, stops by design: no calibration bin reaches 20 samples. It is excluded from Table S18 (59 of 60 runs).

## Verification

Re-running `studies/p1_plugin_conf.py --ds handwritten --seed 230` from this tree on the original machine reproduced the stored result `results/raw/plugconf/pc_handwritten_230.json` exactly: the maximum absolute difference in native and gated accuracy over all backbones and conditions was 0.0. `scripts/reproduce_tables.py` regenerates every mean and median of Tables 2–3 and Tables S3–S4 from the stored rows. The re-evaluation script `studies/p1_plugin_eval.py` reproduced the stored accuracy and NLL of all 2,560 confirmatory rows exactly (maximum difference 0.0) before adding the calibration columns.

## Notes on fidelity

- The code in `rat/` and `studies/` is the frozen code that produced the paper. Only the absolute paths were replaced by paths relative to the repository (`RAT_ROOT`). Internal names differ from the paper: `RRT` is the class that implements RAT, and `mer` is Memory-referenced Evidence Restoration.
- The kNN statistic is the mean distance to the 2nd–11th nearest neighbours. The nearest neighbour is dropped for every query.
- `results/raw/rev3b` holds revision study 3. An earlier, incomplete run with a halving bug (`rev3`) was discarded under amendment A2 and is not included.

## Citation

See `CITATION.cff`. Licence: MIT (code). Datasets and the official baseline code remain under their own licences.
