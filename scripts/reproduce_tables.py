"""Regenerate the numbers in Tables 2-3 of the paper and Tables S3-S4 of the supplement from the stored per-seed rows.

Usage:  python scripts/reproduce_tables.py [--B 5000]
Means and medians are deterministic. Bootstrap intervals use B resamples with seed 0
and agree with the paper to Monte Carlo error (about +/-0.05 points at B = 5000).
"""
import argparse, os, numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(HERE, '..', 'results', 'tables')
UNSEEN = {'C3_unseen', 'B3_unseen'}; CLEAN = {'C0_clean', 'B0_clean'}

def seed_ci(v, B, rng):
    st = [rng.choice(v, len(v), replace=True).mean() for _ in range(B)]; return np.quantile(st, [0.025, 0.975])

def cluster_ci(per, B, rng, alpha=0.05):
    """per: Series indexed by (ds, seed). Resample datasets, then seeds within each dataset."""
    ds = per.index.get_level_values(0).unique().tolist(); arr = {d: per.loc[d].values for d in ds}
    st = np.empty(B)
    for b in range(B):
        pick = rng.choice(len(ds), len(ds), replace=True)
        st[b] = np.mean([rng.choice(arr[ds[i]], len(arr[ds[i]]), replace=True).mean() for i in pick])
    return np.quantile(st, [alpha / 2, 1 - alpha / 2])

def table1(B):
    pc = pd.read_csv(os.path.join(T, 'plugin_rows_230_239.csv.gz')); t = pc[pc.proto == 'trainonly'].copy()
    t['d'] = t.gated - t.native; rng = np.random.default_rng(0); out = []
    for bb in ['QMF', 'PDF', 'MNL', 'BML']:
        for nm, cs in [('clean', CLEAN), ('unseen', UNSEEN)]:
            x = t[(t.backbone == bb) & t.cond.isin(cs)]
            ps = x.groupby('seed').d.mean(); lo, hi = seed_ci(ps.values, B, rng)
            per = x.groupby(['ds', 'seed']).d.mean(); clo, chi = cluster_ci(per, B, rng)
            out.append(dict(backbone=bb, cond=nm, mean=ps.mean(), seed_lo=lo, seed_hi=hi, cluster_lo=clo, cluster_hi=chi,
                            median_ds=x.groupby('ds').d.mean().median()))
    u = t[t.cond.isin(UNSEEN)]
    auc = u.groupby('backbone')[['auroc_native', 'auroc_beta']].mean()
    gains = pd.read_csv(os.path.join(T, 'trust_gap_190_209.csv'), index_col=0).iloc[:, 0]
    gains.index = [('bank:' + i if i in ('crema', 'food', 'milk', 'sun') else i) for i in gains.index]
    g = u.groupby('ds').d.mean(); gg = pd.concat([g, gains], axis=1, join='inner')
    rho = spearmanr(gg.iloc[:, 0], gg.iloc[:, 1])[0]
    return pd.DataFrame(out), auc, rho

def pooled(df, c, var, B, rng):
    x = df[(df.c == c) & (df['var'] == var)]
    if not len(x): return [np.nan] * 4
    per_ds = x.groupby('ds').g.mean(); per = x.groupby(['ds', 'seed']).g.mean(); lo, hi = cluster_ci(per, B, rng)
    return [per_ds.mean(), lo, hi, per_ds.median()]

def table_rows(fn, variants, B):
    df = pd.read_csv(os.path.join(T, fn)); rng = np.random.default_rng(0); rows = []
    for v in variants:
        r = [v]
        for c in ['clean', 'unseen']: r += pooled(df, c, v, B, rng)
        rows.append(r)
    return pd.DataFrame(rows, columns=['variant', 'clean', 'clean_lo', 'clean_hi', 'clean_med', 'unseen', 'unseen_lo', 'unseen_hi', 'unseen_med'])

def calibration():
    """Table 2 (damaged-sample gain, ECE, Brier) and Tables S3-S4: re-evaluation of the confirmatory runs."""
    ev = pd.read_csv(os.path.join(T, 'plugin_eval_rows_230_239.csv.gz')); out, nd = {}, {}
    for bb in ['QMF', 'PDF', 'MNL', 'BML']:
        raw = ev[(ev.backbone == bb) & ev.cond.isin(UNSEEN)]
        u = raw.groupby('ds').mean(numeric_only=True)
        c = ev[(ev.backbone == bb) & ev.cond.isin(CLEAN)].groupby('ds').mean(numeric_only=True)
        out[bb] = dict(ece_native=u.ece_native.mean(), ece_rat=u.ece_gated.mean(), dECE=u.dece.mean(), dECE_med=u.dece.median(),
                       dBrier_x100=u.dbri.mean(), dBrier_med=u.dbri.median(), dNLL_med=u.dnll.median(),
                       dECE_clean=c.dece.mean(), dBrier_clean=c.dbri.mean(),
                       gain_damaged=(u.acc_dam_gated - u.acc_dam_native).mean(), gain_intact=(u.acc_int_gated - u.acc_int_native).mean())
        row = {}
        for k in range(4):
            x = raw[raw[f'n{k}'] > 0]
            row[f'{k} damaged'] = (x[f'acc_gated_d{k}'] - x[f'acc_native_d{k}']).groupby(x.ds).mean().mean()
        nd[bb] = row
    return pd.DataFrame(out).T, pd.DataFrame(nd).T

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--B', type=int, default=5000); a = ap.parse_args()
    pd.set_option('display.width', 200)
    t1, auc, rho = table1(a.B)
    print('Table 2 (seeds 230-239): gate minus native, accuracy points'); print(t1.round(2).to_string(index=False))
    print('Damage-detection AUROC (native / beta):'); print(auc.round(2).to_string()); print(f'PH3 Spearman rho = {rho:.2f}')
    print('\nTable 3(a) (seeds 250-254, four backbones pooled)')
    print(table_rows('rev2_rows_250_254.csv', ['rat', 'knn_raw', 'knn_res', 'range', 'mer', 'fixed', 'recal', 'fusion', 'anchor_joint', 'oracle'], a.B).round(2).to_string(index=False))
    print('\nTable 3(b), revision study 3 (seeds 255-259)')
    print(table_rows('rev3_rows_255_259.csv', ['rat', 'knn', 'rat_anchor', 'knn_anchor', 'crossfit', 'half', 'maha', 'conf', 'comb'], a.B).round(2).to_string(index=False))
    cal, nd = calibration()
    print('\nTable 2 calibration columns / Table S3 (unseen damage unless marked clean; ECE in points, Brier x100)'); print(cal.round(2).to_string())
    print('\nTable S4: gain by number of damaged modalities'); print(nd.round(2).to_string())
