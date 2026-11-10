"""Paper 1 replication study 3 (seeds 255-259). Frozen mvbench modules; writes to mvdiag/res/rev3."""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1_plugin_conf as PC
import numpy as np, torch
from sklearn.covariance import LedoitWolf
from mv_method import RRT, DEV, logits_of, seen_damage, SEEN
import mv_bench as BN
import p1_rev2 as R2

def sm(z): return R2.sm(z)

class RRTsafe(RRT):
    """frozen RRT; only for half-calibration cross-fits: single global temperature if no bin reaches 20 samples."""
    fallbacks = 0
    def _fit_law(self, q, z, y, nb=12):
        try:
            return RRT._fit_law(self, q, z, y, nb)
        except ValueError:
            RRTsafe.fallbacks += 1; c0 = R2.ce_grid(z, y)
            return type('C', (), {'predict': staticmethod(lambda qq, c0=c0: np.full(len(qq), c0))})()

def maha_model(Xtr, seed):
    rs = np.random.RandomState(seed); B = Xtr[rs.choice(len(Xtr), min(2000, len(Xtr)), replace=False)]
    lw = LedoitWolf().fit(B); return lw.location_, lw.precision_

def maha(mod, X):
    mu, P = mod; D = X - mu; return np.log(np.sqrt(np.maximum((D @ P * D).sum(1), 0)) + 1e-8)[:, None]

class Law1:
    """isotonic decreasing law of a single statistic's clean-cal surprisal (same machinery as RAT)."""
    def __init__(self, rx, s_c, z_c, yc, s_d, z_d, yd):
        self.ref = np.sort(s_c, 0); qc, qd = self.q(s_c), self.q(s_d); self.qc = qc; self.fallback = False
        q = np.concatenate([qc, qd]); z = np.concatenate([z_c, z_d]); y = np.concatenate([yc, yd])
        try: self.iso = rx._fit_law(q, z, y)
        except ValueError:
            c0 = R2.ce_grid(z, y); self.fallback = True
            self.iso = type('C', (), {'predict': staticmethod(lambda qq, c0=c0: np.full(len(qq), c0))})()
    def q(self, s): return R2.marg(self.ref, s, list(range(s.shape[1]))).mean(1)
    def __call__(self, s): q = self.q(s); return np.clip(self.iso.predict(q), 0, 2), q

def jsd_others(P, m):
    oth = [u for u in range(len(P)) if u != m]; return R2.jsd(P[m], np.mean([P[u] for u in oth], 0))

def cal_damage_index(ds, ycal, V, rx):
    if ds.startswith('bank:'):
        D = PC.BK.load_bank(ds[5:]); out = []
        for m in D['mods']:
            fam = D['cal'][m + '_fam']; sel = np.isin(fam, PC.BK.SEEN_F)
            out.append(np.repeat(np.arange(len(ycal))[:, None], fam.shape[1], 1)[sel])
        return out
    reps = sum(len(v) for v in SEEN.values()); return [np.tile(np.arange(len(ycal)), reps) for _ in range(V)]

def main(ds, seed):
    Xtr, ytr, Xcal, ycal, yte, K, rcfg, conds, lrs, ep, bank = PC.setup(ds, seed)
    V = len(Xtr); out = dict(ds=ds, seed=seed, res={}, ratio={}, fallback={}, time={})
    t0 = time.time()
    rx = RRT(dict(rcfg), seed).fit(Xtr, ytr, Xcal, ycal, K)
    idxd = cal_damage_index(ds, ycal, V, rx)
    S_c, Z_c, S_d, Z_d = [], [], [], []
    for m in range(V):
        a, b = R2.stats4(rx, m, Xcal[m]); S_c.append(a); Z_c.append(b)
        Xd, yd = rx.cal_damaged[m]; a, b = R2.stats4(rx, m, Xd); S_d.append(a); Z_d.append(b)
    yd_all = [rx.cal_damaged[m][1] for m in range(V)]
    L = {}
    for m in range(V):
        L[('rat', m)] = R2.VLaw(rx, S_c[m], Z_c[m], ycal, S_d[m], Z_d[m], yd_all[m], [0, 1, 2], 'mean')
        L[('knn', m)] = R2.VLaw(rx, S_c[m], Z_c[m], ycal, S_d[m], Z_d[m], yd_all[m], [3], 'mean')
    # Mahalanobis and confidence laws
    MM = [maha_model(Xtr[m], seed + m) for m in range(V)]
    for m in range(V):
        Xd = rx.cal_damaged[m][0]
        L[('maha', m)] = Law1(rx, maha(MM[m], Xcal[m]), Z_c[m], ycal, maha(MM[m], Xd), Z_d[m], yd_all[m])
        L[('conf', m)] = Law1(rx, (1 - sm(Z_c[m]).max(1))[:, None], Z_c[m], ycal, (1 - sm(Z_d[m]).max(1))[:, None], Z_d[m], yd_all[m])
    # combined law: (v + u_JSD)/2 with JSD among probe predictions
    Pc = [sm(Z_c[m]) for m in range(V)]
    J_c = [jsd_others(Pc, m)[:, None] for m in range(V)]
    J_d = []
    for m in range(V):
        Pd = [Pc[u][idxd[m]] for u in range(V)]; Pd[m] = sm(Z_d[m]); J_d.append(jsd_others(Pd, m)[:, None])
    class Comb:
        def __init__(s, m):
            s.v = L[('rat', m)]; s.jref = np.sort(J_c[m], 0)
            uj_c = R2.marg(s.jref, J_c[m], [0])[:, 0]; uj_d = R2.marg(s.jref, J_d[m], [0])[:, 0]
            q = np.concatenate([0.5 * (s.v.qc + uj_c), 0.5 * (s.v.q(S_d[m]) + uj_d)])
            z = np.concatenate([Z_c[m], Z_d[m]]); y = np.concatenate([ycal, yd_all[m]])
            try: s.iso = rx._fit_law(q, z, y)
            except ValueError:
                c0 = R2.ce_grid(z, y); s.iso = type('C', (), {'predict': staticmethod(lambda qq, c0=c0: np.full(len(qq), c0))})()
        def __call__(s, st, jj):
            q = 0.5 * (s.v.q(st) + R2.marg(s.jref, jj, [0])[:, 0]); return np.clip(s.iso.predict(q), 0, 2)
    CB = [Comb(m) for m in range(V)]
    # cross-fitting: two stratified halves of the calibration split
    rs = np.random.RandomState(seed + 99); half = np.zeros(len(ycal), bool)
    order = np.concatenate([rs.permutation(np.where(ycal == c)[0]) for c in rs.permutation(np.unique(ycal))])
    half[order[::2]] = True                                  # alternate assignment: stratified, never empty
    H = [np.where(half)[0], np.where(~half)[0]]; cf = []
    for a_, b_ in ((0, 1), (1, 0)):
        ia, ib = H[a_], H[b_]; cfg = dict(rcfg)
        if bank:
            cfg['_cal_damaged'] = [(rcfg['_cal_damaged'][m][0][np.isin(idxd[m], ia)], rcfg['_cal_damaged'][m][1][np.isin(idxd[m], ia)]) for m in range(V)]
            cfg['_mv_seen'] = [x[ia] for x in rcfg['_mv_seen']]
        rxa = RRTsafe(cfg, seed).fit(Xtr, ytr, [x[ia] for x in Xcal], ycal[ia], K)
        laws = []
        for m in range(V):
            sc, zc = R2.stats4(rxa, m, Xcal[m][ib])
            if bank:
                msk = np.isin(idxd[m], ib); Xd, yd = rcfg['_cal_damaged'][m][0][msk], rcfg['_cal_damaged'][m][1][msk]
            else:
                rng = np.random.RandomState(seed + 7 + a_); Xd = np.vstack([seen_damage(Xcal[m][ib], f, s_, rng) for f, sv in SEEN.items() for s_ in sv])
                yd = np.tile(ycal[ib], sum(len(v) for v in SEEN.values()))
            sd, zd = R2.stats4(rxa, m, Xd)
            laws.append(R2.VLaw(rxa, sc, zc, ycal[ib], sd, zd, yd, [0, 1, 2], 'mean'))
        cf.append((rxa, laws))
    out['fallback'] = {k: int(sum(L[(k, m)].fallback for m in range(V))) for k in ('rat', 'knn', 'maha', 'conf')}
    # test-time betas per condition
    BV = {}
    for c, (Xc, Dm) in conds.items():
        st = [R2.stats4(rx, m, Xc[m]) for m in range(V)]
        b = {}
        b['rat'] = np.stack([L[('rat', m)](st[m][0])[0] for m in range(V)]); q_rat = np.stack([L[('rat', m)](st[m][0])[1] for m in range(V)])
        b['knn'] = np.stack([L[('knn', m)](st[m][0])[0] for m in range(V)]); q_knn = np.stack([L[('knn', m)](st[m][0])[1] for m in range(V)])
        b['maha'] = np.stack([L[('maha', m)](maha(MM[m], Xc[m]))[0] for m in range(V)])
        b['conf'] = np.stack([L[('conf', m)]((1 - sm(st[m][1]).max(1))[:, None])[0] for m in range(V)])
        Pt = [sm(st[m][1]) for m in range(V)]
        b['comb'] = np.stack([CB[m](st[m][0], jsd_others(Pt, m)[:, None]) for m in range(V)])
        for nm_, (bb, qq, key) in {'rat_anchor': (b['rat'], q_rat, 'rat'), 'knn_anchor': (b['knn'], q_knn, 'knn')}.items():
            tj = float(np.quantile(np.stack([L[(key, m)].qc for m in range(V)]).max(0), 0.9))
            B2 = bb.copy(); B2[:, qq.max(0) <= tj] = 1.0; b[nm_] = B2
        bcf = []
        for rxa, laws in cf:
            bcf.append(np.stack([laws[m](R2.stats4(rxa, m, Xc[m])[0])[0] for m in range(V)]))
        b['crossfit'] = 0.5 * (bcf[0] + bcf[1]); b['half'] = np.stack([np.clip(cf[0][0]._evidence(Xc)[1][m], 0, 2) for m in range(V)])
        BV[c] = b
        if Dm.any() and (~Dm).any():
            out['ratio'][c] = [float(np.exp(st[m][0][Dm[:, m], 3]).mean() / np.exp(st[m][0][~Dm[:, m], 3]).mean()) if Dm[:, m].any() else float('nan') for m in range(V)]
    out['cf_fallbacks'] = RRTsafe.fallbacks
    out['rat_match'] = float(max(np.abs(rx._evidence(conds[c][0])[1] - BV[c]['rat']).max() for c in conds))
    out['time']['rat'] = time.time() - t0
    for name, (trainer, kind, e0) in R2.TR.items():
        t1 = time.time(); kw = dict(lr=lrs.get(name, BN.BASE_LR[name]))
        if bank: kw['epochs'] = ep(e0)
        pr = trainer(Xtr, ytr, K, seed, **kw); M = PC.getmodel(pr); R_ = {}
        for c, (Xc, Dm) in conds.items():
            lg, w = PC.parts(kind, M, Xc); nat = (w[:, :, None] * lg).sum(0); r = {'native': R2.acc(nat, yte), 'nll_native': R2.nll(nat, yte)}
            for vn, B in BV[c].items():
                f = R2.fused(w, B, lg); r[vn] = R2.acc(f, yte); r['nll_' + vn] = R2.nll(f, yte)
            R_[c] = r
        out['res'][name] = R_; out['time'][name] = time.time() - t1
        del pr, M; torch.cuda.empty_cache()
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ds'); ap.add_argument('--seed', type=int); ap.add_argument('--out')
    a = ap.parse_args(); t0 = time.time(); r = main(a.ds, a.seed); r['time']['total'] = time.time() - t0
    json.dump(r, open(a.out, 'w'), indent=1, default=float)
    print({k: {c: (round(v['native'], 2), round(v['rat'], 2), round(v['knn'], 2), round(v['rat_anchor'], 2), round(v['crossfit'], 2)) for c, v in d.items()} for k, d in r['res'].items()}, 'match', r['rat_match'])
