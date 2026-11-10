"""Paper 1 revision study C: severity sweep on the banks (RRT experts; seed 240)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1_plugin_conf as PC
import numpy as np, torch
import mv_bank as BK
from mv_method import RRT


def main(ds, seed):
    Xtr, ytr, Xcal, ycal, yte, K, rcfg, conds, lrs, ep, bank = PC.setup('bank:' + ds, seed)
    D = BK.load_bank(ds); b = np.load(os.path.join(BK.BANKS, f'{ds}.npz'), allow_pickle=True)
    rx = RRT(dict(rcfg), seed).fit(Xtr, ytr, Xcal, ycal, K); rows = []
    for v, m in enumerate(D['mods']):
        var = D['test'][m + '_var']; fam = D['test'][m + '_fam']; sev = b[f'test_{m}_sev'].astype(int)
        def summ(Xm, yy, f, s):
            st, z = rx._view_stats(v, Xm); q = rx._surprisal(v, st); be = np.clip(rx.laws[v].predict(q), 0, 2)
            p = torch.softmax(torch.tensor(z), 1).numpy()
            rows.append(dict(ds=ds, mod=m, fam=int(f), sev=int(s), n=len(yy), acc=float((z.argmax(1) == yy).mean() * 100),
                             msp=float(p.max(1).mean()), entropy=float(-(p * np.log(p + 1e-12)).sum(1).mean()),
                             surprisal=float(q.mean()), beta=float(be.mean())))
        summ(D['test'][m], yte, 0, 0)
        for f in (1, 2, 3, 4):
            for s in (1, 2, 3, 4, 5):
                ii, jj = np.where((fam == f) & (sev == s))
                if len(ii) >= 30: summ(var[ii, jj], yte[ii], f, s)
    return rows


if __name__ == '__main__':
    out = []
    for ds in ('sun', 'milk', 'food', 'crema'): out += main(ds, 240)
    json.dump(out, open(sys.argv[1], 'w'), indent=1); print(len(out))
