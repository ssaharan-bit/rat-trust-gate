"""Paper 1 second revision study (seeds 250-254). Frozen mvbench modules; writes to mvdiag/res/rev2."""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1_plugin_conf as PC                     # inserts frozen mvbench path
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression
from mv_method import RRT, DEV, logits_of
import mv_baselines2 as MB2, mv_bench as BN
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location('mv_method_diag', os.path.join(HERE, 'mv_method.py')); MD = _ilu.module_from_spec(_sp); _sp.loader.exec_module(MD)

GRID = np.concatenate([np.linspace(0.0, 0.2, 11), np.linspace(0.25, 2.0, 36)])
GT = torch.tensor(GRID, dtype=torch.float64)
TR = {'PDF': (MB2.train_pdf, 'pdf', 500), 'MNL': (MB2.train_mnl, 'mnl', 500), 'QMF': (BN.train_qmf, 'qmf', 500), 'BML': (MB2.train_bml, 'bml', 200)}
# statistic columns: 0 kNN(restored), 1 range, 2 MER step, 3 kNN(raw)
VAR = {'rat': ([0, 1, 2], 'mean'), 'knn_res': ([0], 'mean'), 'knn_raw': ([3], 'mean'), 'range': ([1], 'mean'), 'mer': ([2], 'mean'),
       'knnraw+range': ([3, 1], 'mean'), 'knnres+range': ([0, 1], 'mean'), 'rat_rawknn': ([3, 1, 2], 'mean'),
       'rat_max': ([0, 1, 2], 'max'), 'rat_joint': ([0, 1, 2], 'joint')}


def ce_grid(z, y):
    zt = torch.tensor(z, dtype=torch.float64); yt = torch.tensor(y)
    return float(GRID[int(np.argmin([float(F.cross_entropy(zt * g, yt)) for g in GRID]))])


def acc(s, y): return float((np.asarray(s).argmax(1) == y).mean() * 100)
def nll(lg, y):
    lp = torch.log_softmax(torch.tensor(lg, dtype=torch.float64), 1).numpy(); return float(-lp[np.arange(len(y)), y].mean())


def knn_dist(X, bank, k):
    Bt = torch.tensor(bank, device=DEV, dtype=torch.float32); out = []
    for i in range(0, len(X), 4096):
        Xt = torch.tensor(X[i:i + 4096], device=DEV, dtype=torch.float32)
        out.append(torch.cdist(Xt, Bt).topk(k, largest=False).values[:, 1:].mean(1).cpu().numpy())
    return np.concatenate(out)


def stats4(rx, m, Xraw):
    ref = rx.refs[m]; Xres = ref.apply(Xraw, rx.ops[m]); k = min(rx.cfg['knn'] + 1, len(ref.bank))
    d_res = knn_dist(Xres, ref.bank, k); d_raw = knn_dist(Xraw, ref.bank, k)
    oor = ((Xraw < ref.lo) | (Xraw > ref.hi)).mean(1); mag = np.linalg.norm(Xraw - Xres, axis=1)
    z = logits_of(rx.experts[m], Xres) / rx.T[m]
    return np.stack([np.log(d_res + 1e-8), oor, np.log1p(mag), np.log(d_raw + 1e-8)], 1), z


def marg(ref_sorted, st, cols):
    n = len(ref_sorted)
    S = np.stack([1.0 - np.searchsorted(ref_sorted[:, j], st[:, j], side='left') / n for j in cols], 1)
    return -np.log(np.clip(S, 1.0 / (n + 1), 1.0))


def joint(ref, st, cols):
    R = torch.tensor(ref[:, cols], device=DEV); n = len(ref); out = []
    for i in range(0, len(st), 512):
        S = torch.tensor(st[i:i + 512][:, cols], device=DEV)
        cnt = (R[None] >= S[:, None]).all(2).sum(1).double()
        out.append((-torch.log((1 + cnt) / (n + 1))).cpu().numpy())
    return np.concatenate(out)


class VLaw:
    def __init__(self, rx, st_c, z_c, yc, st_d, z_d, yd, cols, agg):
        self.ref = np.sort(st_c, 0); self.refraw = st_c; self.cols, self.agg = cols, agg
        qc, qd = self.q(st_c), self.q(st_d); q = np.concatenate([qc, qd]); z = np.concatenate([z_c, z_d]); y = np.concatenate([yc, yd])
        self.qc = qc; self.fallback = False
        try:
            self.iso = rx._fit_law(q, z, y)
        except ValueError:
            c0 = ce_grid(z, y); self.fallback = True
            self.iso = type('C', (), {'predict': staticmethod(lambda qq, c0=c0: np.full(len(qq), c0))})()
    def q(self, st):
        if self.agg == 'joint': return joint(self.refraw, st, self.cols)
        u = marg(self.ref, st, self.cols)
        return u.max(1) if self.agg == 'max' else u.mean(1)
    def __call__(self, st):
        q = self.q(st); return np.clip(self.iso.predict(q), 0, 2), q


def fused(w, B, lg): return (w[:, :, None] * B[:, :, None] * lg).sum(0)


def fit_fusion_law(w, lg, y, Q, B0, nb=12, sweeps=2):
    """per-modality bins of surprisal Q[m]; coordinate descent on fused CE; isotonic (decreasing) smoothing."""
    V = len(lg); B = B0.copy(); yt = torch.tensor(y)
    edges = [np.unique(np.quantile(Q[m], np.linspace(0, 1, nb + 1))) for m in range(V)]
    bins = [np.clip(np.searchsorted(edges[m], Q[m], side='right') - 1, 0, max(len(edges[m]) - 2, 0)) for m in range(V)]
    for _ in range(sweeps):
        for m in range(V):
            for b in np.unique(bins[m]):
                idx = np.where(bins[m] == b)[0]
                if len(idx) < 20: continue
                rest = torch.tensor(fused(w[:, idx], B[:, idx], lg[:, idx]) - w[m, idx, None] * B[m, idx, None] * lg[m, idx], dtype=torch.float64)
                comp = torch.tensor(w[m, idx, None] * lg[m, idx], dtype=torch.float64)
                ce = [float(F.cross_entropy(rest + g * comp, yt[idx])) for g in GRID]
                B[m, idx] = GRID[int(np.argmin(ce))]
    isos = []
    for m in range(V):
        xs, ys, ws = [], [], []
        for b in np.unique(bins[m]):
            idx = bins[m] == b; xs.append(np.median(Q[m][idx])); ys.append(B[m][idx].mean()); ws.append(idx.sum())
        ir = IsotonicRegression(increasing=False, out_of_bounds='clip', y_min=0, y_max=2)
        ir.fit(np.array(xs), np.array(ys), sample_weight=np.array(ws)); isos.append(ir)
    return isos


def oracle_fusion(w, lg, y, Dm, sweeps=2):
    V = len(lg); B = np.ones((V, len(y))); yt = torch.tensor(y)
    for _ in range(sweeps):
        for m in range(V):
            for msk in (Dm[:, m], ~Dm[:, m]):
                idx = np.where(msk)[0]
                if len(idx) < 6: continue
                rest = torch.tensor(fused(w[:, idx], B[:, idx], lg[:, idx]) - w[m, idx, None] * B[m, idx, None] * lg[m, idx], dtype=torch.float64)
                comp = torch.tensor(w[m, idx, None] * lg[m, idx], dtype=torch.float64)
                B[m, idx] = GRID[int(np.argmin([float(F.cross_entropy(rest + g * comp, yt[idx])) for g in GRID]))]
    return B


def jsd(p, q):
    m = 0.5 * (p + q); kl = lambda a, b: (a * (np.log(a + 1e-12) - np.log(b + 1e-12))).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def aurc(score, err):
    o = np.argsort(score); e = err[o].astype(float); cov = np.arange(1, len(e) + 1)
    return float((np.cumsum(e) / cov).mean())


def sm(z): return torch.softmax(torch.tensor(z, dtype=torch.float64), 1).numpy()


def bin_law(Cc, zc, yc, nb):
    e = np.unique(np.quantile(Cc, np.linspace(0, 1, nb + 1))); g0 = ce_grid(zc, yc); vals = []
    bc = np.clip(np.searchsorted(e, Cc, side='right') - 1, 0, max(len(e) - 2, 0))
    for b in range(max(len(e) - 1, 1)):
        i = bc == b; vals.append(ce_grid(zc[i], yc[i]) if i.sum() >= 20 else g0)
    vals = np.array(vals)
    return lambda C: vals[np.clip(np.searchsorted(e, C, side='right') - 1, 0, len(vals) - 1)]


def curvature(z, C, d, K):
    out = {}
    e = np.unique(np.quantile(C, np.linspace(0, 1, 7))); cb = np.clip(np.searchsorted(e, C, side='right') - 1, 0, max(len(e) - 2, 0))
    zt = torch.tensor(z, dtype=torch.float64); mu = np.inf
    for g in GRID:
        p = torch.softmax(g * zt, 1); var = ((p * zt ** 2).sum(1) - (p * zt).sum(1) ** 2).numpy()
        for b in np.unique(cb):
            for dd in (0, 1):
                i = (cb == b) & (d == dd)
                if i.sum() >= 20: mu = min(mu, float(var[i].mean()))
    spread = z.max(1) - z.min(1); L = float(np.abs(z).max()); s10 = float(np.percentile(spread, 10))
    return dict(mu_hat=mu, L_max=L, L_p99=float(np.percentile(np.abs(z).max(1), 99)), spread_p10=s10, spread_p50=float(np.median(spread)),
                bound=float(0.9 * s10 ** 2 * np.exp(-2 * 2.0 * L) / (2 * K)))


def main(ds, seed):
    Xtr, ytr, Xcal, ycal, yte, K, rcfg, conds, lrs, ep, bank = PC.setup(ds, seed)
    V = len(Xtr); out = dict(ds=ds, seed=seed, res={}, err={}, atyp={}, dec={}, scale={}, betas={}, orac={}, curv={}, time={}, fallback={})
    ck = 'B0_clean' if bank else 'C0_clean'; uk = 'B3_unseen' if bank else 'C3_unseen'
    t0 = time.time()
    rx = RRT(dict(rcfg), seed).fit(Xtr, ytr, Xcal, ycal, K)          # frozen RAT: own probes, never queries a backbone
    rng = np.random.RandomState(seed + 11)
    Xmix = rx._mv_seen(Xcal, rng)                                      # multi-view seen damage on calibration data
    S_c, Z_c, S_d, Z_d = [], [], [], []
    for m in range(V):
        a, b = stats4(rx, m, Xcal[m]); S_c.append(a); Z_c.append(b)
        Xd, yd = rx.cal_damaged[m]; a, b = stats4(rx, m, Xd); S_d.append(a); Z_d.append(b)
    yd_all = [rx.cal_damaged[m][1] for m in range(V)]
    laws = {(vn, m): VLaw(rx, S_c[m], Z_c[m], ycal, S_d[m], Z_d[m], yd_all[m], cols, agg) for vn, (cols, agg) in VAR.items() for m in range(V)}
    fixed = [ce_grid(np.concatenate([Z_c[m], Z_d[m]]), np.concatenate([ycal, yd_all[m]])) for m in range(V)]
    out['fallback'] = {vn: int(sum(laws[(vn, m)].fallback for m in range(V))) for vn in VAR}
    # test statistics once per condition
    ST = {c: [stats4(rx, m, Xc[m]) for m in range(V)] for c, (Xc, Dm) in conds.items()}
    BV = {c: {vn: np.stack([laws[(vn, m)](ST[c][m][0])[0] for m in range(V)]) for vn in VAR} for c in conds}
    QV = {c: np.stack([laws[('rat', m)](ST[c][m][0])[1] for m in range(V)]) for c in conds}
    out['rat_match'] = float(max(np.abs(rx._evidence(conds[c][0])[1] - BV[c]['rat']).max() for c in conds))
    # sample-level anchor (R7): threshold on max_m v_m at its clean-calibration 90% quantile
    Qcal = np.stack([laws[('rat', m)].qc for m in range(V)]); tj = float(np.quantile(Qcal.max(0), 0.9))
    t90 = [float(np.quantile(laws[('rat', m)].qc, 0.9)) for m in range(V)]
    for c in conds:
        B = BV[c]['rat'].copy(); keep = QV[c].max(0) <= tj; B[:, keep] = 1.0; BV[c]['anchor_joint'] = B
        B2 = np.where(QV[c] <= np.array(t90)[:, None], 1.0, BV[c]['rat']); BV[c]['anchored'] = B2
        BV[c]['fixed'] = np.repeat(np.array(fixed)[:, None], len(yte), 1)
    out['time']['rat'] = time.time() - t0
    # ---------- backbone-free analyses: oracle approximation and curvature (probe logits)
    Cc_all = {m: np.concatenate([sm(Z_c[m]).max(1), sm(Z_d[m]).max(1)]) for m in range(V)}
    for m in range(V):
        zc = np.concatenate([Z_c[m], Z_d[m]]); yc = np.concatenate([ycal, yd_all[m]]); Cc = Cc_all[m]
        qc = np.concatenate([laws[('rat', m)].qc, laws[('rat', m)].q(S_d[m])])
        lawC = bin_law(Cc, zc, yc, 12)
        # 2-D (C, v) law: 4x4 quantile cells
        eC = np.unique(np.quantile(Cc, np.linspace(0, 1, 5))); eV = np.unique(np.quantile(qc, np.linspace(0, 1, 5))); g0 = ce_grid(zc, yc)
        def cell(C, q): return (np.clip(np.searchsorted(eC, C, side='right') - 1, 0, len(eC) - 2), np.clip(np.searchsorted(eV, q, side='right') - 1, 0, len(eV) - 2))
        ic, iv = cell(Cc, qc); tab = np.full((len(eC) - 1, len(eV) - 1), g0)
        for a in range(tab.shape[0]):
            for b in range(tab.shape[1]):
                i = (ic == a) & (iv == b)
                if i.sum() >= 20: tab[a, b] = ce_grid(zc[i], yc[i])
        # test pool: clean (d=0) + unseen (d = damage mask)
        st0, z0 = ST[ck][m]; st1, z1 = ST[uk][m]; Dm = conds[uk][1]
        z = np.concatenate([z0, z1]); y = np.concatenate([yte, yte]); d = np.concatenate([np.zeros(len(yte), bool), Dm[:, m]])
        C = sm(z).max(1); q = np.concatenate([laws[('rat', m)](st0)[1], laws[('rat', m)](st1)[1]])
        eT = np.unique(np.quantile(C, np.linspace(0, 1, 7))); cb = np.clip(np.searchsorted(eT, C, side='right') - 1, 0, len(eT) - 2)
        gT = ce_grid(z, y); bstar = np.full(len(y), gT)
        for b in np.unique(cb):
            for dd in (False, True):
                i = (cb == b) & (d == dd)
                if i.sum() >= 10: bstar[i] = ce_grid(z[i], y[i])
        ests = {'C': lawC(C), 'v': np.clip(laws[('rat', m)].iso.predict(q), 0, 2), 'Cv': tab[cell(C, q)]}
        zt = torch.tensor(z, dtype=torch.float64); yt = torch.tensor(y)
        o = {k: dict(mse=float(((b_ - bstar) ** 2).mean()), nll=float(F.cross_entropy(zt * torch.tensor(b_)[:, None], yt))) for k, b_ in ests.items()}
        o['oracle'] = dict(mse=0.0, nll=float(F.cross_entropy(zt * torch.tensor(bstar)[:, None], yt)))
        o['damaged_frac'] = float(d.mean()); out['orac'][m] = o
        dcal = np.concatenate([np.zeros(len(ycal), bool), np.ones(len(yd_all[m]), bool)])
        out['curv'][m] = curvature(zc, Cc, dcal, K)
    # ---------- per backbone
    Xpool = [np.vstack([a, b]) for a, b in zip(Xcal, Xmix)]; ypool = np.concatenate([ycal, ycal])
    Qpool = np.stack([laws[('rat', m)](np.vstack([S_c[m], stats4(rx, m, Xmix[m])[0]]))[1] for m in range(V)])
    Bpool = np.stack([np.clip(laws[('rat', m)].iso.predict(Qpool[m]), 0, 2) for m in range(V)])
    for name, (trainer, kind, e0) in TR.items():
        t1 = time.time(); kw = dict(lr=lrs.get(name, BN.BASE_LR[name]))
        if bank: kw['epochs'] = ep(e0)
        pr = trainer(Xtr, ytr, K, seed, **kw); M = PC.getmodel(pr)
        heads = [PC.Head(M, v, kind) for v in range(V)]
        rxd = MD.RRT(dict(rcfg, _external_experts=heads), seed).fit(Xtr, ytr, Xcal, ycal, K)        # law refitted on this backbone's heads
        lgP, wP = PC.parts(kind, M, Xpool)
        isos = fit_fusion_law(wP, lgP, ypool, Qpool, Bpool)                                         # fusion-loss-calibrated law
        lgC, _ = PC.parts(kind, M, Xcal)
        R, dec, sc, bt = {}, {}, {}, {}
        for c, (Xc, Dm) in conds.items():
            lg, w = PC.parts(kind, M, Xc); nat = (w[:, :, None] * lg).sum(0); r = {'native': acc(nat, yte), 'nll_native': nll(nat, yte)}
            Bs = dict(BV[c]); Bs['recal'] = rxd._evidence(Xc)[1]
            Bs['fusion'] = np.stack([np.clip(isos[m].predict(QV[c][m]), 0, 2) for m in range(V)])
            if Dm.any():
                Bo = np.ones((V, len(yte)))
                for m in range(V):
                    for msk in (Dm[:, m], ~Dm[:, m]):
                        if msk.sum() > 5: Bo[m, msk] = ce_grid(lg[m][msk], yte[msk])
                Bs['oracle'] = Bo; Bs['oracle_fusion'] = oracle_fusion(w, lg, yte, Dm)
            for vn, B in Bs.items():
                f = fused(w, B, lg); r[vn] = acc(f, yte); r['nll_' + vn] = nll(f, yte)
            # decomposition: f_RAT = T(x) * sum_m (w_m beta_m / T) z_m
            B = Bs['rat']; Tx = (w * B).sum(0); f = fused(w, B, lg); t0m = Tx <= 1e-12
            fn = np.where(t0m[:, None], nat, f / np.maximum(Tx, 1e-12)[:, None]); fg = Tx[:, None] * nat
            fr = np.where(t0m[:, None], nat, f)
            dec[c] = dict(acc_rat=acc(f, yte), acc_norm=acc(fn, yte), acc_global=acc(fg, yte), acc_fallback=acc(fr, yte), n_T0=int(t0m.sum()),
                          nll_rat=nll(f, yte), nll_norm=nll(fn, yte), nll_global=nll(fg, yte), nll_native=r['nll_native'], T_mean=float(Tx.mean()),
                          T_p05=float(np.percentile(Tx, 5)), w_sum_mean=float(w.sum(0).mean()))
            # logit-scale stress test (modality 0 scaled by a)
            sc[c] = {}
            for a_ in (0.5, 2.0):
                lg2 = lg.copy(); lg2[0] = lg2[0] * a_; n2 = (w[:, :, None] * lg2).sum(0)
                sc[c][str(a_)] = dict(native=acc(n2, yte), rat=acc(fused(w, Bs['rat'], lg2), yte), recal=acc(fused(w, Bs['recal'], lg2), yte))
            # beta summaries
            msk_d = Dm.T if Dm.any() else np.zeros_like(B, bool)
            bt[c] = {vn: dict(mean=float(Bs[vn].mean()), dam=float(Bs[vn][msk_d].mean()) if msk_d.any() else float('nan'),
                              intact=float(Bs[vn][~msk_d].mean())) for vn in ('rat', 'recal', 'fusion')}
            bt[c]['corr_rat_recal'] = float(np.corrcoef(Bs['rat'].ravel(), Bs['recal'].ravel())[0, 1]) if Bs['recal'].std() > 0 and Bs['rat'].std() > 0 else float('nan')
            R[c] = r
            # error prediction (unseen)
            if c == uk:
                lg0, _ = PC.parts(kind, M, conds[ck][0]); rows = []
                for m in range(V):
                    pv = sm(lg[m]); induced = (lg0[m].argmax(1) == yte) & (lg[m].argmax(1) != yte)
                    oth = [u for u in range(V) if u != m]
                    js = jsd(pv, np.mean([sm(lg[u]) for u in oth], 0))
                    jsc = jsd(sm(lgC[m]), np.mean([sm(lgC[u]) for u in oth], 0)); jref = np.sort(jsc)
                    uj = marg(jref[:, None], js[:, None], [0])[:, 0]
                    q = QV[c][m]
                    rows.append((induced, {'validity': q, 'jsd': js, 'combined': 0.5 * (q + uj), '1-msp': 1 - pv.max(1),
                                           'knn_raw': laws[('knn_raw', m)](ST[c][m][0])[1], 'knn_res': laws[('knn_res', m)](ST[c][m][0])[1]}))
                ya = np.concatenate([r_[0] for r_ in rows]); e = {'rate': float(ya.mean())}
                for k_ in rows[0][1]:
                    sa = np.concatenate([r_[1][k_] for r_ in rows])
                    e[k_] = dict(auroc=float(roc_auc_score(ya, sa)) if 0 < ya.sum() < len(ya) else float('nan'),
                                 ap=float(average_precision_score(ya, sa)) if ya.sum() > 0 else float('nan'), aurc=aurc(sa, ya))
                out['err'][name] = e
            if c == ck:
                flag = (QV[c] > np.array(t90)[:, None]).any(0); flagj = QV[c].max(0) > tj
                out['atyp'][name] = dict(frac=float(flag.mean()), frac_joint=float(flagj.mean()),
                                         d_flag=acc(fused(w, Bs['rat'], lg)[flag], yte[flag]) - acc(nat[flag], yte[flag]) if flag.any() else float('nan'),
                                         d_rest=acc(fused(w, Bs['rat'], lg)[~flag], yte[~flag]) - acc(nat[~flag], yte[~flag]))
        out['res'][name] = R; out['dec'][name] = dec; out['scale'][name] = sc; out['betas'][name] = bt
        out['time'][name] = time.time() - t1
        del pr, M, heads, rxd; torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ds'); ap.add_argument('--seed', type=int); ap.add_argument('--out')
    a = ap.parse_args(); t0 = time.time(); r = main(a.ds, a.seed); r['time']['total'] = time.time() - t0
    json.dump(r, open(a.out, 'w'), indent=1, default=float)
    print({k: {c: (round(v['native'], 2), round(v['rat'], 2), round(v['recal'], 2), round(v['fusion'], 2)) for c, v in d.items()} for k, d in r['res'].items()}, 'match', r['rat_match'])
