# β ≤ 1 cap — development test (seeds 220–222), decision rule fixed before results
Variant: RRT-cap, β' = min(β, 1) everywhere β enters (τ fit and prediction); frozen RRT otherwise.
Datasets: 12 standard + 4 banks. Reference arms: RRT (v2), RRT-noRAT.
RAT swap cost s = mean over {C1, C4} of (RRT-noRAT − RRT). Recovery r = mean over {C1, C4} of (RRT-cap − RRT) / s (pooled over datasets and seeds).
ADOPT iff r ≥ 0.5 AND (RRT − RRT-cap) ≤ 0.2 pt on C3 (standard) AND ≤ 0.2 pt on B3 (banks) AND ≤ 0.2 pt clean.
Otherwise: report as a negative result; the swap limitation stays in Paper 2.
One smoke run (scene15, seed 220) was seen before this file was written: recovery ≈ 0.10. The rule is unchanged from the approved plan.
