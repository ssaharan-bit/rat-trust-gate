"""Condense the re-evaluation JSON files (studies/p1_plugin_eval.py, seeds 230-239) into one row file.
Usage: python scripts/condense_eval.py [studies/res/ploteval] [results/tables/plugin_eval_rows_230_239.csv.gz]
Adds dece (ECE difference, points), dbri (Brier difference x100) and dnll (NLL difference), each gate minus native.
"""
import json, glob, os, sys
import pandas as pd
src = sys.argv[1] if len(sys.argv) > 1 else "studies/res/ploteval"
out = sys.argv[2] if len(sys.argv) > 2 else "results/tables/plugin_eval_rows_230_239.csv.gz"
rows = []
for f in sorted(glob.glob(os.path.join(src, "pe_*.json"))):
    r = json.load(open(f))
    for key, cs in r["res"].items():
        bb, proto = key.split("|")
        if proto != "trainonly":
            continue
        for c, d in cs.items():
            rows.append(dict(ds=r["ds"], seed=r["seed"], backbone=bb, cond=c, **{k: v for k, v in d.items() if not isinstance(v, (list, dict))}))
df = pd.DataFrame(rows)
df["dece"] = df.ece_gated - df.ece_native
df["dbri"] = (df.brier_gated - df.brier_native) * 100
df["dnll"] = df.nll_gated - df.nll_native
df.to_csv(out, index=False); print(len(df), "rows ->", out)
