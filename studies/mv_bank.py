"""Paper 1 benchmark on frozen-encoder banks with real raw-signal corruptions.
Families: 1 A noise, 2 B blur/compression (seen; calibration/augmentation only), 3 C occlusion/dropout,
4 D environment (unseen; test only). python mv_bank.py --ds sun --seed 210 --out res/p1/p1_sun_210.json"""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mv_bench as B
from mv_method import RRT, log_softmax_np
BANKS = os.environ.get('RAT_BANKS', os.path.join(B.ROOT, 'banks'))   # frozen-encoder banks
SEEN_F, UNSEEN_F = (1, 2), (3, 4)


def load_bank(ds):
    b = np.load(os.path.join(BANKS, f'{ds}.npz'), allow_pickle=True)
    mods = [str(m) for m in b['mods']]; K = int(b['classes'])
    D = {'mods': mods, 'K': K}
    for sp in ('train', 'cal', 'test'):
        D[sp] = {'y': b[f'{sp}_y'].astype(int)}
        for m in mods:
            D[sp][m] = b[f'{sp}_{m}_clean'].astype(np.float64)
            D[sp][m + '_var'] = b[f'{sp}_{m}_var'].astype(np.float64)
            D[sp][m + '_fam'] = b[f'{sp}_{m}_fam'].astype(int)
    # z-score per modality, fitted on clean train+cal
    for m in mods:
        ref = np.vstack([D['train'][m], D['cal'][m]]); mu = ref.mean(0); sd = ref.std(0) + 1e-6
        for sp in ('train', 'cal', 'test'):
            D[sp][m] = (D[sp][m] - mu) / sd
            if D[sp][m + '_var'].size:
                D[sp][m + '_var'] = (D[sp][m + '_var'] - mu) / sd
    return D


def substitute(D, sp, fams, rng, frac=0.5):
    """damage a random non-empty proper subset of modalities (one modality if V == 2) for frac of samples,
    using the bank's real corrupted feature for a family drawn from `fams`."""
    mods = D['mods']; V = len(mods); n = len(D[sp]['y'])
    X = [D[sp][m].copy() for m in mods]; Dm = np.zeros((n, V), bool); Fm = np.full(n, -1)
    for i in rng.choice(n, int(frac * n), replace=False):
        f = fams[rng.randint(len(fams))]
        k = rng.randint(1, V) if V > 2 else 1
        for v in rng.choice(V, k, replace=False):
            fv = D[sp][mods[v] + '_fam'][i]
            cols = np.where(fv == f)[0]
            if len(cols) == 0:
                continue
            X[v][i] = D[sp][mods[v] + '_var'][i, cols[rng.randint(len(cols))]]; Dm[i, v] = True
        Fm[i] = f - 1
    return X, Dm, Fm


def run(ds, seed, methods, cfg=None, lrs=None, variants=None):
    D = load_bank(ds); mods = D['mods']; K = D['K']; V = len(mods)
    Xtr = [D['train'][m] for m in mods]; ytr = D['train']['y']
    Xcal = [D['cal'][m] for m in mods]; ycal = D['cal']['y']
    Xte = [D['test'][m] for m in mods]; yte = D['test']['y']
    # RRT calibration damage: real seen families (A/B) on the calibration split only
    cal_dam = []
    for m in mods:
        var, fam = D['cal'][m + '_var'], D['cal'][m + '_fam']
        sel = np.isin(fam, SEEN_F)
        cal_dam.append((var[sel], np.repeat(ycal[:, None], var.shape[1], 1)[sel]))
    mv_seen, _, _ = substitute(D, 'cal', SEEN_F, np.random.RandomState(seed + 7))
    # step-matched training budget: same number of gradient updates as the official schedule on n = 2000
    n_ref = 2000.0
    def ep_scale(E, n): return int(max(30, round(E * min(1.0, n_ref / n))))
    rcfg = dict(cfg or {}, _cal_damaged=cal_dam, _mv_seen=mv_seen)
    rcfg.setdefault('epochs', ep_scale(300, len(ytr)))
    # test conditions (identical across methods within a seed)
    conds = {'B0_clean': ([x.copy() for x in Xte], np.zeros((len(yte), V), bool), np.full(len(yte), -1))}
    conds['B2_seen'] = substitute(D, 'test', SEEN_F, np.random.RandomState(seed * 100 + 2))
    conds['B3_unseen'] = substitute(D, 'test', UNSEEN_F, np.random.RandomState(seed * 100 + 3))
    conds['B3_unseen_all'] = substitute(D, 'test', UNSEEN_F, np.random.RandomState(seed * 100 + 4), frac=1.0)
    res = dict(ds=ds, seed=seed, n=[len(ytr), len(ycal), len(yte)], V=V, K=K, mods=mods, results={}, time={})
    t0 = time.time(); preds = {}
    print('loaded', [x.shape for x in Xtr], len(ycal), len(yte), [d[0].shape for d in cal_dam], flush=True)
    rrt = RRT(rcfg, seed).fit(Xtr, ytr, Xcal, ycal, K); res['rrt'] = rrt.diagnostics()
    print('rrt fit', round(time.time() - t0, 1), flush=True)
    preds['RRT'] = rrt.predict_proba
    for ab, kw in [('RRT-noMERA', dict(ops=['raw'])), ('RRT-noRAT', dict(no_beta=True)), ('RRT-noTau', dict(no_tau=True)),
                   ('RRT-noEps', dict(no_eps=True)), ('RRT-staticTau', dict(release=False))]:
        if 'all' in methods or ab in methods:
            preds[ab] = RRT(dict(rcfg, **kw), seed).fit(Xtr, ytr, Xcal, ycal, K, base=rrt).predict_proba
    for vn, vcfg in (variants or {}).items():
        vc = dict(rcfg, **vcfg); refit = vc.pop('refit', False)
        preds['V:' + vn] = RRT(vc, seed).fit(Xtr, ytr, Xcal, ycal, K, base=None if refit else rrt).predict_proba
    preds['LateAvg'] = lambda Xn: np.exp(B.expert_logp(rrt, Xn)).mean(0)
    preds['PoE'] = lambda Xn: np.exp(log_softmax_np(B.expert_logp(rrt, Xn).sum(0)))
    preds['2T'] = B.TwoT(rrt, Xcal, ycal, seed).predict
    res['time']['rrt'] = time.time() - t0
    # baselines: train on train+cal clean; +aug = same data plus real seen-family substitutions
    Xtv = [np.vstack([a, b]) for a, b in zip(Xtr, Xcal)]; ytv = np.concatenate([ytr, ycal])
    rng = np.random.RandomState(seed + 11); Xaug = [x.copy() for x in Xtv]
    ntr = len(ytr)
    for i in range(len(ytv)):
        if rng.rand() < 0.5:
            sp, j = ('train', i) if i < ntr else ('cal', i - ntr)
            k = rng.randint(1, V) if V > 2 else 1
            for v in rng.choice(V, k, replace=False):
                var = D[sp][mods[v] + '_var']
                if var.size and var.shape[1]:
                    Xaug[v][i] = var[j, rng.randint(var.shape[1])]
    Xtva = [np.vstack([a, b]) for a, b in zip(Xtv, Xaug)]; ytva = np.concatenate([ytv, ytv])
    LR = lrs or {}
    OFF_EP = {'TMC': 500, 'ETMC': 500, 'RCML': 500, 'QMF': 500, 'TMCEK': 500, 'GTMC-HOA': 500, 'TUNED': 500,
              'BML': 200, 'PDF': 500, 'MNL': 500}
    def lk(n):
        d = {'lr': LR[n]} if n in LR else {}
        if n in OFF_EP: d['epochs'] = ep_scale(OFF_EP[n], len(ytv))
        return d
    res['epochs'] = {n: ep_scale(E, len(ytv)) for n, E in OFF_EP.items()}; res['epochs']['RRT_experts'] = rcfg['epochs']
    import mv_baselines2 as MB2
    T = {'Concat': lambda X_, y_: B.train_concat(X_, y_, K, seed),
         'TMC': lambda X_, y_: B.train_tmc(X_, y_, K, seed, **lk('TMC')),
         'ETMC': lambda X_, y_: B.train_tmc(X_, y_, K, seed, etmc=True, **lk('ETMC')),
         'RCML': lambda X_, y_: B.train_rcml(X_, y_, K, seed, **lk('RCML')),
         'QMF': lambda X_, y_: B.train_qmf(X_, y_, K, seed, **lk('QMF')),
         'MMD': lambda X_, y_: B.train_mmd(X_, y_, K, seed, **lk('MMD'))}
    for n_, (f_, _) in MB2.TRAINERS2.items():
        T[n_] = (lambda f, nn_: (lambda X_, y_: f(X_, y_, K, seed, **lk(nn_))))(f_, n_)
    for n_, fn in T.items():
        for aug in ([False] if n_ == 'Concat' else [False, True]):
            nm = n_ + ('+aug' if aug else '')
            if 'all' in methods or nm in methods or n_ in methods:
                t1 = time.time()
                preds[nm] = fn(Xtva, ytva) if aug else fn(Xtv, ytv)
                res['time'][nm] = time.time() - t1; print(nm, round(res['time'][nm], 1), flush=True)
    for nm, f in preds.items():
        res['results'][nm] = {}
        for c, (Xc, Dm, Fm) in conds.items():
            p = f(Xc); r = B.metrics(p, yte, Dm, None)
            if c.startswith('B3') or c.startswith('B2'):
                r['by_family'] = {str(fv + 1): float((p.argmax(1)[Fm == fv] == yte[Fm == fv]).mean() * 100)
                                  for fv in np.unique(Fm) if fv >= 0}
            if nm == 'RRT' and Dm.any() and (~Dm).any():
                from sklearn.metrics import roc_auc_score
                Bw = rrt.last_B; r['beta_auroc_damage'] = float(roc_auc_score(Dm.T.ravel(), -Bw.ravel()))
            res['results'][nm][c] = r
    res['time']['total'] = time.time() - t0
    return res


def select_lr_bank(ds, seed, mults=(1, 3, 10), only=None):
    """same rule as mv_bench.select_lr: train on train split, score clean calibration NLL/acc."""
    import mv_baselines2 as MB2
    D = load_bank(ds); mods = D['mods']; K = D['K']
    Xtr = [D['train'][m] for m in mods]; ytr = D['train']['y']; Xcal = [D['cal'][m] for m in mods]; ycal = D['cal']['y']
    n_tv = len(ytr) + len(ycal)                         # budget computed as in run() (train+cal size)
    def E(n): return int(max(30, round(n * min(1.0, 2000.0 / n_tv))))
    fns = {'TMC': lambda lr: B.train_tmc(Xtr, ytr, K, seed, lr=lr, epochs=E(500)),
           'ETMC': lambda lr: B.train_tmc(Xtr, ytr, K, seed, etmc=True, lr=lr, epochs=E(500)),
           'RCML': lambda lr: B.train_rcml(Xtr, ytr, K, seed, lr=lr, epochs=E(500)),
           'QMF': lambda lr: B.train_qmf(Xtr, ytr, K, seed, lr=lr, epochs=E(500)),
           'MMD': lambda lr: B.train_mmd(Xtr, ytr, K, seed, lr=lr)}
    for n_, (f_, _) in MB2.TRAINERS2.items():
        fns[n_] = (lambda f, e: (lambda lr: f(Xtr, ytr, K, seed, lr=lr, epochs=e)))(f_, E(200 if n_ == 'BML' else 500))
    out = {}
    for n, f in fns.items():
        if only and n not in only:
            continue
        out[n] = {}
        for mlt in mults:
            lr = B.BASE_LR[n] * mlt; p = f(lr)(Xcal)
            out[n][str(lr)] = dict(acc=float((p.argmax(1) == ycal).mean() * 100),
                                   nll=float(-np.log(np.clip(p[np.arange(len(ycal)), ycal], 1e-12, 1)).mean()))
    return dict(ds='bank:' + ds, seed=seed, sel=out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--methods', default='all')
    ap.add_argument('--cfg', default='{}'); ap.add_argument('--lrs', default='{}'); ap.add_argument('--variants', default='{}'); ap.add_argument('--select', action='store_true')
    a = ap.parse_args()
    if a.select:
        c_ = json.loads(a.cfg); r = select_lr_bank(a.ds, a.seed, tuple(c_.get('mults', [1, 3, 10])), c_.get('only'))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(r, open(a.out, 'w'), indent=1); print(r); sys.exit(0)
    lrs = json.load(open(a.lrs)).get('bank:' + a.ds, {}) if os.path.exists(a.lrs) else json.loads(a.lrs)
    r = run(a.ds, a.seed, a.methods.split(','), json.loads(a.cfg), lrs, json.loads(a.variants)); r['lrs'] = lrs
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(r, open(a.out, 'w'), indent=1)
    print({m: round(v['B3_unseen']['acc'], 2) for m, v in r['results'].items()}, round(r['time']['total'], 1))
