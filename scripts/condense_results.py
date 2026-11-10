"""Condense the per-seed JSON files of the confirmatory study into one row file.
Usage: python scripts/condense_results.py [results/raw/plugconf]
"""
import json,glob,gzip,os,sys
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv)>1 else "results/raw/plugconf", "pc_*.json"))):
    r=json.load(open(f))
    for key,cs in r["res"].items():
        bb,proto=key.split("|")
        for c,d in cs.items():
            rows.append(dict(ds=r["ds"],seed=r["seed"],backbone=bb,proto=proto,cond=c,**{k:v for k,v in d.items() if not isinstance(v,(list,dict))}))
gzip.open("results/tables/plugin_rows_raw.json.gz","wt").write(json.dumps(rows)); print(len(rows))
