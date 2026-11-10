# Amendment P-A1 to plugin_prereg.md (compute budget), 2026-11-04T21:28
Written before any confirmatory outcome was analysed. The only result content inspected so far was a file-structure check of one row (ROSMAP seed 230, PDF clean) and per-job runtimes.

Reason: each dataset×seed job takes about 45–65 min with 16 in parallel (GPU at 86%), so 160 jobs need about 9 h. Half of this cost is the descriptive train+cal control.

Change: the train+cal control (secondary, descriptive, no decision rule) runs for seed 230 on all 16 datasets, plus any seed-231 jobs already in flight. Seeds 231–239 run the primary train-only protocol only.

Unchanged: all primary hypotheses PH1–PH4 (train-only protocol), seeds 230–239, datasets, backbones, decision rules and statistics.
