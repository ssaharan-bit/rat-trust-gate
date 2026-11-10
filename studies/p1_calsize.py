"""Paper 1 rev2: calibration-fraction sensitivity (12 standard sets, PDF, seeds 250-254)."""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1_plugin_conf as PC
import numpy as np, torch
from mv_method import RRT
import mv_baselines2 as MB2, mv_bench as BN


def split(y, seed, rcal, fixed_test=None):
    rng = np.random.RandomState(seed); idx = np.arange(len(y))
    te = fixed_test if fixed_test is not None else np.concatenate([rng.permutation(np.where(y == c)[0])[:int(round(0.2 * (y == c).sum()))] for c in np.unique(y)])
    tv = np.setdiff1d(idx, te); frac = rcal / (1.0 - len(te) / len(y))
    cal = np.concatenate([rng.permutation(tv[y[tv] == c])[:max(1, int(round(frac * (y[tv] == c).sum())))] for c in np.unique(y)])
    return rng.permutation(np.setdiff1d(tv, cal)), cal, np.sort(te)


def main(ds, seed, rcal):
    X, y, fixed = BN.load_dataset(ds); K = int(y.max() + 1)
    tr, cal, te = split(y, seed, rcal, fixed); X = BN.minmax(X, np.concatenate([tr, cal]))
    Xtr, Xcal, Xte = [x[tr] for x in X], [x[cal] for x in X], [x[te] for x in X]
    allc = BN.build_conditions(Xte, y[te], X, y, seed, K)
    conds = {k: (v[0], np.asarray(v[1], bool)) for k, v in allc.items() if k in ('C0_clean', 'C3_unseen')}
    lrs = json.load(open(os.path.join(PC.MV, 'lrs_v2.json'))).get(ds, {})
    pr = MB2.train_pdf(Xtr, y[tr], K, seed, lr=lrs.get('PDF', BN.BASE_LR['PDF'])); M = PC.getmodel(pr)
    rx = RRT(dict(floor_mode='untempered'), seed).fit(Xtr, y[tr], Xcal, y[cal], K)
    out = dict(ds=ds, seed=seed, rcal=rcal, n_tr=len(tr), n_cal=len(cal), n_te=len(te), res={})
    for c, (Xc, Dm) in conds.items():
        lg, w = PC.parts('pdf', M, Xc); _, B = rx._evidence(Xc)
        out['res'][c] = dict(native=PC.acc((w[:, :, None] * lg).sum(0), y[te]), rat=PC.acc((w[:, :, None] * B[:, :, None] * lg).sum(0), y[te]),
                             beta_mean=float(B.mean()))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ds'); ap.add_argument('--seed', type=int); ap.add_argument('--rcal', type=float); ap.add_argument('--out')
    a = ap.parse_args(); r = main(a.ds, a.seed, a.rcal); json.dump(r, open(a.out, 'w'), indent=1); print(r)
