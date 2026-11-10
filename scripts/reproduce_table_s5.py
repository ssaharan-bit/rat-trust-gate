"""Regenerate supplement Table S5 (frozen-encoder banks, seeds 200-209, all official-code methods).
Usage: python scripts/reproduce_table_s5.py
For each method and bank the better of the plain and swap-augmented (+aug) variants by B3 accuracy is used,
with clean accuracy, B3 accuracy and B3 NLL taken from that variant. Base = RRT-noRAT, Base + RAT = RRT.
"""
import os, json, gzip, pandas as pd
T = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'tables')
DS = ['sun', 'milk', 'food', 'crema']
METH = ['TMC', 'ETMC', 'RCML', 'MMD', 'TMCEK', 'GTMC-HOA', 'TUNED', 'QMF', 'PDF', 'MNL', 'BML']

def load(fn):
    d = pd.DataFrame(json.loads(gzip.open(os.path.join(T, fn), 'rt').read()))
    return d[d.seed.between(200, 209)].groupby(['ds', 'method', 'cond'])[['acc', 'nll']].mean()

def table(m, methods, base=True):
    out = {}
    for meth in methods:
        row = {}
        for d in DS:
            vs = [v for v in (meth, meth + '+aug') if (d, v, 'B3_unseen') in m.index]
            if not vs: continue
            v = max(vs, key=lambda v: m.loc[(d, v, 'B3_unseen'), 'acc'])
            row.update({f'{d}_clean': m.loc[(d, v, 'B0_clean'), 'acc'], f'{d}_B3': m.loc[(d, v, 'B3_unseen'), 'acc'],
                        f'{d}_NLL': m.loc[(d, v, 'B3_unseen'), 'nll'], f'{d}_aug': v.endswith('+aug')})
        out[meth] = row
    if base:
        for nm, meth in (('Base', 'RRT-noRAT'), ('Base + RAT', 'RRT')):
            out[nm] = {k: v for d in DS for k, v in ((f'{d}_clean', m.loc[(d, meth, 'B0_clean'), 'acc']),
                       (f'{d}_B3', m.loc[(d, meth, 'B3_unseen'), 'acc']), (f'{d}_NLL', m.loc[(d, meth, 'B3_unseen'), 'nll']))}
    return pd.DataFrame(out).T

if __name__ == '__main__':
    pd.set_option('display.width', 250)
    t = table(load('bank_benchmark_rows_200_209.json.gz'), METH)
    print('Table S5 (pre-registered learning-rate protocol)')
    print(t[[c for c in t.columns if not c.endswith('aug')]].astype(float).round(2).to_string())
    a = table(load('bank_benchmark_amendment2_rows.json.gz'), ['TMC', 'ETMC', 'RCML', 'TMCEK', 'TUNED', 'BML', 'MNL'], base=False)
    print('\nAmendment 2 re-runs (accuracy-based learning-rate selection), B3 accuracy')
    print(a[[c for c in a.columns if c.endswith('_B3')]].astype(float).round(1).to_string())
