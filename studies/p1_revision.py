"""Paper 1 revision study A (seeds 240-244). Frozen mvbench modules; writes to mvdiag/res/revision."""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1_plugin_conf as PC                     # inserts frozen mvbench path
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from mv_method import RRT, DEV, logits_of
import mv_baselines2 as MB2, mv_bench as BN
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location('mv_method_diag', os.path.join(HERE, 'mv_method.py')); MD = _ilu.module_from_spec(_sp); _sp.loader.exec_module(MD)

GRID = np.concatenate([np.linspace(0.0, 0.2, 11), np.linspace(0.25, 2.0, 36)])
COMBOS = {'rat': [0, 1, 2], 'knn': [0], 'range': [1], 'mer': [2], 'knn+range': [0, 1], 'knn+mer': [0, 2], 'range+mer': [1, 2]}
BANKS_N = [256, 512, 1024, 2000, 4000]


def ce_grid(z, y):
    zt = torch.tensor(z, dtype=torch.float64); yt = torch.tensor(y)
    return float(GRID[int(np.argmin([float(F.cross_entropy(zt * g, yt)) for g in GRID]))])


def vstats(rx, m, Xraw, bank):
    ref = rx.refs[m]; Xres = ref.apply(Xraw, rx.ops[m])
    Bt = torch.tensor(bank, device=DEV, dtype=torch.float32); Xt = torch.tensor(Xres, device=DEV, dtype=torch.float32)
    k = min(rx.cfg['knn'] + 1, len(bank))
    dk = torch.cdist(Xt, Bt).topk(k, largest=False).values[:, 1:].mean(1).cpu().numpy()
    oor = ((Xraw < ref.lo) | (Xraw > ref.hi)).mean(1); mag = np.linalg.norm(Xraw - Xres, axis=1)
    z = logits_of(rx.experts[m], Xres) / rx.T[m]
    return np.stack([np.log(dk + 1e-8), oor, np.log1p(mag)], 1), z


def surpr(ref_sorted, st, cols):
    n = len(ref_sorted)
    S = np.stack([1.0 - np.searchsorted(ref_sorted[:, j], st[:, j], side='left') / n for j in cols], 1)
    return (-np.log(np.clip(S, 1.0 / (n + 1), 1.0))).mean(1)


class Law:
    """per-view validity law for one statistic subset / bank / law form."""
    def __init__(self, rx, m, Xcal_m, ycal, bank, cols, form='iso'):
        self.rx, self.m, self.bank, self.cols, self.form = rx, m, bank, cols, form
        st_c, z_c = vstats(rx, m, Xcal_m, bank); Xd, yd = rx.cal_damaged[m]; st_d, z_d = vstats(rx, m, Xd, bank)
        self.ref = np.sort(st_c, 0)
        qc, qd = surpr(self.ref, st_c, cols), surpr(self.ref, st_d, cols)
        q = np.concatenate([qc, qd]); z = np.concatenate([z_c, z_d]); y = np.concatenate([ycal, yd])
        self.t90 = float(np.quantile(qc, 0.9))
        self.fallback = False
        if form in ('iso', 'anchored'):
            try:
                self.iso = rx._fit_law(q, z, y)
            except ValueError:                       # all bins < 20 samples (tied statistic): constant law
                c0 = ce_grid(z, y); self.fallback = True
                self.iso = type('C', (), {'predict': staticmethod(lambda qq, c0=c0: np.full(len(qq), c0))})()
        elif form == 'logistic':
            qt = torch.tensor(q, dtype=torch.float64); zt = torch.tensor(z, dtype=torch.float64); yt = torch.tensor(y)
            p = torch.zeros(2, dtype=torch.float64, requires_grad=True); opt = torch.optim.LBFGS([p], max_iter=200, line_search_fn='strong_wolfe')
            def cl():
                opt.zero_grad(); b = 2 * torch.sigmoid(p[0] + p[1] * qt); l = F.cross_entropy(zt * b[:, None], yt); l.backward(); return l
            opt.step(cl); self.p = p.detach().numpy()
        elif form == 'fixed':
            self.b = ce_grid(z, y)
    def __call__(self, Xm):
        st, _ = vstats(self.rx, self.m, Xm, self.bank); q = surpr(self.ref, st, self.cols)
        if self.form == 'iso': return np.clip(self.iso.predict(q), 0, 2), q
        if self.form == 'anchored': return np.where(q <= self.t90, 1.0, np.clip(self.iso.predict(q), 0, 2)), q
        if self.form == 'logistic': return 2 / (1 + np.exp(-(self.p[0] + self.p[1] * q))), q
        return np.full(len(q), self.b), q


def jsd(p, q):
    m = 0.5 * (p + q); kl = lambda a, b: (a * (np.log(a + 1e-12) - np.log(b + 1e-12))).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def aurc(risk, err):
    o = np.argsort(risk); e = err[o].astype(float); cov = np.arange(1, len(e) + 1)
    return float((np.cumsum(e) / cov).mean())


def main(ds, seed):
    Xtr, ytr, Xcal, ycal, yte, K, rcfg, conds, lrs, ep, bank = PC.setup(ds, seed)
    V = len(Xtr); out = dict(ds=ds, seed=seed, res={}, err={}, atyp={}, time={})
    clean_key = 'B0_clean' if bank else 'C0_clean'; un_key = 'B3_unseen' if bank else 'C3_unseen'
    rs = np.random.RandomState(seed)
    fits = {}
    for name, trainer, kind, e0 in [('PDF', MB2.train_pdf, 'pdf', 500), ('MNL', MB2.train_mnl, 'mnl', 500)]:
        t0 = time.time(); kw = dict(lr=lrs.get(name, BN.BASE_LR[name]))
        if bank: kw['epochs'] = ep(e0)
        pr = trainer(Xtr, ytr, K, seed, **kw); M = PC.getmodel(pr)
        heads = [PC.Head(M, v, kind) for v in range(V)]
        rx = RRT(dict(rcfg, _external_experts=heads), seed).fit(Xtr, ytr, Xcal, ycal, K)
        laws = {}
        for v in range(V):
            dflt = rx.refs[v].bank
            for cn, cols in COMBOS.items(): laws[(cn, v)] = Law(rx, v, Xcal[v], ycal, dflt, cols)
            for fm in ('logistic', 'fixed', 'anchored'): laws[(fm, v)] = Law(rx, v, Xcal[v], ycal, dflt, [0, 1, 2], fm)
            for N in BANKS_N:
                bk = Xtr[v][rs.choice(len(Xtr[v]), min(N, len(Xtr[v])), replace=False)]
                laws[(f'bank{N}', v)] = Law(rx, v, Xcal[v], ycal, bk, [0, 1, 2])
        rxd = MD.RRT(dict(rcfg, _external_experts=heads), seed).fit(Xtr, ytr, Xcal, ycal, K)   # law recalibrated on backbone heads
        fits[name] = (M, kind, rx, laws, rxd); out['time'][name] = time.time() - t0
    variants = list(COMBOS) + ['logistic', 'fixed', 'anchored'] + [f'bank{N}' for N in BANKS_N]
    Xclean = conds[clean_key][0]
    for name, (M, kind, rx, laws, rxd) in fits.items():
        other = 'MNL' if name == 'PDF' else 'PDF'
        out['res'][name] = {}
        for c, (Xc, Dm) in conds.items():
            lg, w = PC.parts(kind, M, Xc); nat = (w[:, :, None] * lg).sum(0)
            r = {'native': PC.acc(nat, yte)}
            Bv = {}
            for vn in variants:
                Bw = np.stack([laws[(vn, v)](Xc[v])[0] for v in range(V)]); Bv[vn] = Bw
                r[vn] = PC.acc((w[:, :, None] * Bw[:, :, None] * lg).sum(0), yte)
            # sanity: rebuilt RAT equals frozen RAT
            _, Bref = rx._evidence(Xc); r['rat_match'] = float(np.abs(Bref - Bv['rat']).max())
            # per-backbone recalibrated law (fitted on this backbone's heads)
            _, Bd = rxd._evidence(Xc)
            r['recal'] = PC.acc((w[:, :, None] * Bd[:, :, None] * lg).sum(0), yte); r['beta_shared_vs_recal_maxdiff'] = float(np.abs(Bd - Bv['rat']).max())
            # oracle two-state beta (test labels; ceiling)
            if Dm.any():
                Bo = np.ones_like(Bv['rat'])
                for v in range(V):
                    zv = lg[v]
                    for msk in (Dm[:, v], ~Dm[:, v]):
                        if msk.sum() > 5: Bo[v, msk] = ce_grid(zv[msk], yte[msk])
                r['oracle'] = PC.acc((w[:, :, None] * Bo[:, :, None] * lg).sum(0), yte)
            r['fallbacks'] = [vn for vn in variants if any(laws[(vn, v)].fallback for v in range(V))]
            out['res'][name][c] = r
            # error prediction (unseen condition): view-level induced error
            if c == un_key:
                lg0, _ = PC.parts(kind, M, Xclean); rows = []
                for v in range(V):
                    pv = torch.softmax(torch.tensor(lg[v]), 1).numpy(); pc0 = lg0[v].argmax(1); pd_ = lg[v].argmax(1)
                    induced = (pc0 == yte) & (pd_ != yte)
                    oth = [u for u in range(V) if u != v]
                    pm = np.mean([torch.softmax(torch.tensor(lg[u]), 1).numpy() for u in oth], 0)
                    _, q_rat = laws[('rat', v)](Xc[v]); _, q_knn = laws[('knn', v)](Xc[v])
                    sc = {'validity': q_rat, 'knn_only': q_knn, '1-msp': 1 - pv.max(1), 'entropy': -(pv * np.log(pv + 1e-12)).sum(1),
                          'energy': -torch.logsumexp(torch.tensor(lg[v]), 1).numpy(), 'xmodal_jsd': jsd(pv, pm)}
                    rows.append((induced, sc))
                y_all = np.concatenate([r_[0] for r_ in rows])
                e = {'rate': float(y_all.mean())}
                for k_ in rows[0][1]:
                    s_all = np.concatenate([r_[1][k_] for r_ in rows])
                    e[k_] = dict(auroc=float(roc_auc_score(y_all, s_all)) if 0 < y_all.sum() < len(y_all) else float('nan'),
                                 ap=float(average_precision_score(y_all, s_all)) if y_all.sum() > 0 else float('nan'), aurc=aurc(s_all, y_all))
                out['err'][name] = e
            if c == clean_key:
                _, q = zip(*[laws[('rat', v)](Xc[v]) for v in range(V)])
                flag = np.any(np.stack([q[v] > laws[('rat', v)].t90 for v in range(V)]), 0)
                g = (w[:, :, None] * Bv['rat'][:, :, None] * lg).sum(0)
                out['atyp'][name] = dict(frac=float(flag.mean()), n=int(flag.sum()),
                    native_flag=PC.acc(nat[flag], yte[flag]) if flag.any() else float('nan'), gated_flag=PC.acc(g[flag], yte[flag]) if flag.any() else float('nan'),
                    native_rest=PC.acc(nat[~flag], yte[~flag]), gated_rest=PC.acc(g[~flag], yte[~flag]),
                    beta_flag=float(Bv['rat'][:, flag].mean()) if flag.any() else float('nan'))
        torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ds'); ap.add_argument('--seed', type=int); ap.add_argument('--out')
    a = ap.parse_args(); t0 = time.time(); r = main(a.ds, a.seed); r['time']['total'] = time.time() - t0
    json.dump(r, open(a.out, 'w'), indent=1)
    print({k: {c: (round(v['native'], 2), round(v['rat'], 2), round(v['anchored'], 2), round(v['recal'], 2), v['rat_match']) for c, v in d.items()} for k, d in r['res'].items()})
