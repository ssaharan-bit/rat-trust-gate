"""Confirmatory plug-in study (seeds 230-239). Imports FROZEN modules from mvbench; writes only to mvdiag/res/plugconf."""
import os, sys, json, time, argparse, inspect
MV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rat")   # frozen method code
sys.path.insert(0, MV)
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import mv_bank as BK, mv_bench as BN, mv_baselines2 as MB2
from mv_method import RRT, DEV


def acc(s, y): return float((np.asarray(s).argmax(1) == y).mean() * 100)
def nll(lg, y):
    lp = torch.log_softmax(torch.tensor(lg, dtype=torch.float64), 1).numpy(); return float(-lp[np.arange(len(y)), y].mean())
def T(X): return [torch.tensor(x, dtype=torch.float32, device=DEV) for x in X]
def getmodel(pr):
    m = getattr(pr, 'model', None)
    if m is not None: return m
    f = pr
    while True:
        nl = inspect.getclosurevars(f).nonlocals
        if 'model' in nl: return nl['model']
        f = getattr(f, '__wrapped__', None)
        if f is None: raise KeyError('model')


class Head(torch.nn.Module):
    def __init__(self, model, m, kind):
        super().__init__(); self.mdl, self.m, self.kind = model, m, kind; self.mdl.eval()
    def forward(self, x):
        with torch.no_grad():
            M = self.mdl
            if self.kind == 'pdf': return M.clf[self.m](M.enc[self.m](x))
            if self.kind == 'bml': return M.classifier_list[self.m](M.encoder_list[self.m](x))
            return M.clf[self.m](x)                                   # mnl, qmf


@torch.no_grad()
def parts(kind, M, X):
    """per-view logits (V,n,K) and the backbone's native fusion weights (V,n): fused_native = sum_v w_v z_v."""
    M.eval(); Xb = T(X)
    if kind == 'pdf':
        f = [e(x) for e, x in zip(M.enc, Xb)]; lg = [c(z) for c, z in zip(M.clf, f)]
        tcp = [cn(z) for cn, z in zip(M.conf, f)]
        lt = torch.cat([torch.log(t.clamp_min(1e-8)) for t in tcp], 1); tot = lt.sum(1, keepdim=True)
        holo = (tot - lt) / (tot + 1e-8); cb = torch.cat(tcp, 1) + holo
        du = torch.stack([torch.mean(torch.abs(F.softmax(z, 1) - 1.0 / z.shape[1]), 1) for z in lg], 1)
        cb = cb * (du / du.max(1, keepdim=True).values.clamp_min(1e-8)); w = torch.softmax(cb, 1).T
    elif kind == 'mnl':
        lg = [c(x) for c, x in zip(M.clf, Xb)]; w = torch.full((len(lg), Xb[0].shape[0]), 1.0 / len(lg), device=DEV)
    elif kind == 'qmf':
        fused, lg, conf = M(Xb); w = torch.cat(conf, 1).T
    elif kind == 'bml':
        lg, fused, rel = M(Xb); w = rel.T
    return torch.stack(lg).cpu().numpy().astype(np.float64), w.cpu().numpy().astype(np.float64)


def setup(ds, seed):
    if ds.startswith('bank:'):
        b = ds[5:]; D = BK.load_bank(b); mods = D['mods']; K = D['K']
        Xtr = [D['train'][m] for m in mods]; ytr = D['train']['y']; Xcal = [D['cal'][m] for m in mods]; ycal = D['cal']['y']
        Xte = [D['test'][m] for m in mods]; yte = D['test']['y']
        cal_dam = []
        for m in mods:
            var, fam = D['cal'][m + '_var'], D['cal'][m + '_fam']; sel = np.isin(fam, BK.SEEN_F)
            cal_dam.append((var[sel], np.repeat(ycal[:, None], var.shape[1], 1)[sel]))
        mv_seen, _, _ = BK.substitute(D, 'cal', BK.SEEN_F, np.random.RandomState(seed + 7))
        rcfg = dict(floor_mode='untempered', _cal_damaged=cal_dam, _mv_seen=mv_seen)
        n = len(yte); V = len(mods); z = np.zeros((n, V), bool)
        conds = {'B0_clean': ([x.copy() for x in Xte], z)}
        for c, fams, s, fr in [('B2_seen', BK.SEEN_F, 2, 0.5), ('B3_unseen', BK.UNSEEN_F, 3, 0.5), ('B3_unseen_all', BK.UNSEEN_F, 4, 1.0)]:
            Xc, Dm = BK.substitute(D, 'test', fams, np.random.RandomState(seed * 100 + s), frac=fr)[:2]; conds[c] = (Xc, np.asarray(Dm, bool))
        lrs = json.load(open(os.path.join(MV, 'lrs_bank.json'))).get(ds, {})
        ntv = len(ytr) + len(ycal); ep = lambda e: int(max(30, round(e * min(1.0, 2000.0 / ntv))))
        return Xtr, ytr, Xcal, ycal, yte, K, rcfg, conds, lrs, ep, True
    X, y, fixed = BN.load_dataset(ds); K = int(y.max() + 1)
    tr, cal, te = BN.make_split(y, seed, fixed); X = BN.minmax(X, np.concatenate([tr, cal]))
    Xtr, Xcal, Xte = [x[tr] for x in X], [x[cal] for x in X], [x[te] for x in X]
    allc = BN.build_conditions(Xte, y[te], X, y, seed, K)
    conds = {k: (v[0], np.asarray(v[1], bool)) for k, v in allc.items() if k in ('C0_clean', 'C1_rcml', 'C3_unseen', 'C4_swap')}
    lrs = json.load(open(os.path.join(MV, 'lrs_v2.json'))).get(ds, {})
    return Xtr, y[tr], Xcal, y[cal], y[te], K, dict(floor_mode='untempered'), conds, lrs, None, False


def main(ds, seed):
    Xtr, ytr, Xcal, ycal, yte, K, rcfg, conds, lrs, ep, bank = setup(ds, seed)
    V = len(Xtr); Xtv = [np.vstack([p, q]) for p, q in zip(Xtr, Xcal)]; ytv = np.concatenate([ytr, ycal])
    TR = {'PDF': (MB2.train_pdf, 'pdf', 500), 'MNL': (MB2.train_mnl, 'mnl', 500), 'QMF': (BN.train_qmf, 'qmf', 500),
          'BML': (MB2.train_bml, 'bml', 200)}
    out = dict(ds=ds, seed=seed, res={}, time={})
    for name, (trainer, kind, e0) in TR.items():
        protos = {'trainonly': (Xtr, ytr), 'traincal': (Xtv, ytv)} if seed == 230 else {'trainonly': (Xtr, ytr)}   # Amendment P-A1
        for proto, (Xa, ya) in protos.items():
            t0 = time.time(); kw = dict(lr=lrs.get(name, BN.BASE_LR[name]))
            if bank: kw['epochs'] = ep(e0)
            pr = trainer(Xa, ya, K, seed, **kw); M = getmodel(pr)
            heads = [Head(M, v, kind) for v in range(V)]
            rx = RRT(dict(rcfg, _external_experts=heads), seed).fit(Xtr, ytr, Xcal, ycal, K)
            key = f'{name}|{proto}'; out['res'][key] = {}
            for c, (Xc, Dm) in conds.items():
                lg, w = parts(kind, M, Xc); _, Bw = rx._evidence(Xc)
                nat = (w[:, :, None] * lg).sum(0); gated = (w[:, :, None] * Bw[:, :, None] * lg).sum(0)
                r = {'native': acc(nat, yte), 'gated': acc(gated, yte), 'native_predict': acc(pr(Xc), yte),
                     'nll_native': nll(nat, yte), 'nll_gated': nll(gated, yte), 'rrt_on_heads': acc(rx.predict_proba(Xc), yte)}
                if Dm.any() and (~Dm).any():
                    yv = Dm.ravel(); r['auroc_beta'] = float(roc_auc_score(yv, -Bw.T.ravel()))
                    r['auroc_native'] = float(roc_auc_score(yv, -w.T.ravel())) if kind != 'mnl' else float('nan')
                    r['beta_damaged'] = float(Bw.T[Dm].mean()); r['beta_intact'] = float(Bw.T[~Dm].mean())
                    dmg = Dm.any(1); r['acc_dam_native'] = acc(nat[dmg], yte[dmg]); r['acc_dam_gated'] = acc(gated[dmg], yte[dmg])
                else:
                    r['beta_mean'] = float(Bw.mean()); r['beta_p05'] = float(np.percentile(Bw, 5))
                out['res'][key][c] = r
            out['time'][key] = time.time() - t0
            del pr, M, heads, rx; torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ds'); ap.add_argument('--seed', type=int); ap.add_argument('--out')
    a = ap.parse_args(); t0 = time.time(); r = main(a.ds, a.seed); r['time']['total'] = time.time() - t0
    json.dump(r, open(a.out, 'w'), indent=1)
    print({k: {c: (round(v[c]['native'], 2), round(v[c]['gated'], 2)) for c in v} for k, v in r['res'].items()})
