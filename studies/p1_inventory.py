"""Paper 1 rev2: dataset inventory."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1_plugin_conf as PC
import numpy as np, mv_bench as BN
STD = ['handwritten', 'scene15', 'caltech20', 'landuse', 'pie', 'leaves', 'outdoor', 'brca', 'rosmap', 'mosi', 'mustard', 'urfunny']
out = {}
for ds in STD:
    X, y, fixed = BN.load_dataset(ds); tr, cal, te = BN.make_split(y, 250, fixed)
    out[ds] = dict(n=int(len(y)), K=int(y.max() + 1), V=len(X), dims=[int(x.shape[1]) for x in X], n_tr=int(len(tr)), n_cal=int(len(cal)), n_te=int(len(te)),
                   fixed_test=fixed is not None, bank_N=[int(min(2000, len(tr)))] * len(X))
for b in ('sun', 'milk', 'food', 'crema'):
    D = PC.BK.load_bank(b); mods = D['mods']
    out['bank:' + b] = dict(n=int(sum(len(D[s]['y']) for s in ('train', 'cal', 'test'))), K=int(D['K']), V=len(mods), mods=list(mods),
                            dims=[int(D['train'][m].shape[1]) for m in mods], n_tr=int(len(D['train']['y'])), n_cal=int(len(D['cal']['y'])), n_te=int(len(D['test']['y'])),
                            bank_N=[int(min(2000, len(D['train']['y'])))] * len(mods))
json.dump(out, open(sys.argv[1], 'w'), indent=1); print(json.dumps(out))
