"""Condense the bank-benchmark JSON files (mv_bench.py, seeds 200-209) into one row file.
Usage: python scripts/condense_bankbench.py RESULT_DIR OUT.json.gz
"""
import json,glob,gzip,sys,os
src=sys.argv[1]; out=sys.argv[2]; rows=[]
for f in sorted(glob.glob(os.path.join(src,'*.json'))):
    r=json.load(open(f))
    for meth,cs in r['results'].items():
        if not isinstance(cs,dict): continue
        for c,d in cs.items():
            if isinstance(d,dict) and 'acc' in d:
                rows.append(dict(ds=r['ds'],seed=r['seed'],method=meth,cond=c,acc=d.get('acc'),nll=d.get('nll'),ece=d.get('ece'),f1=d.get('f1')))
gzip.open(out,'wt').write(json.dumps(rows)); print(len(rows))
