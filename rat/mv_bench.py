"""Standard trusted multi-view benchmark harness (official baseline model code).

python mv_bench.py --ds handwritten --seed 160 --out res/mvb_handwritten_160.json
"""
import os, sys, json, time, argparse, importlib.util, copy
import numpy as np, scipy.io as sio, scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('RAT_DATA', os.path.join(os.path.dirname(HERE), 'data'))   # raw datasets
EXT = os.path.join(HERE, 'ext')
sys.path.insert(0, HERE)
import mv_method as MM
from mv_method import DEV, SEEN, RRT, Expert, train_mlp, logits_of, fit_temp, log_softmax_np


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


tmc_mod = _load('tmc_official', os.path.join(EXT, 'TMC', 'TMC ICLR', 'model.py'))
rcml_model = _load('rcml_model', os.path.join(EXT, 'RCML', 'model.py'))
rcml_loss = _load('rcml_loss', os.path.join(EXT, 'RCML', 'loss_function.py'))
mmd_mod = _load('mmd_model', os.path.join(EXT, 'mmdynamics', 'model.py'))
crl = _load('qmf_crl', os.path.join(EXT, 'QMF', 'RGBD-scene-recognition', 'utils', 'crl_utils.py'))

# ----------------------------------------------------------------- data
MATS = {'handwritten': 'HandWritten.mat', 'scene15': 'Scene15.mat', 'caltech20': 'Caltech101-20.mat',
        'landuse': 'LandUse_21.mat', 'leaves': '100Leaves.mat', 'outdoor': 'OutdoorScene.mat'}


MB = {'mosi': ('mosi-20260616T063357Z-3-001/mosi_data.pkl', 'pos'),
      'urfunny': ('humor-20260616T063616Z-3-001/humor/humor.pkl', 'bin'),
      'mustard': ('sarcasm-20260616T063252Z-3-001/sarcasm.pkl', 'bin')}


def load_multibench(name):
    import pickle
    path, kind = MB[name]
    D = pickle.load(open(os.path.join(ROOT, path), 'rb'))
    Xs, ys, sizes = [[], [], []], [], {}
    for sp in ('train', 'valid', 'test'):
        d = D[sp]
        for i, m in enumerate(('text', 'audio', 'vision')):
            a = np.nan_to_num(np.asarray(d[m], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            Xs[i].append(np.concatenate([a.mean(1), a.std(1)], 1))          # time-pooled (mean, std)
        lab = np.asarray(d['labels'], dtype=np.float64).reshape(len(a), -1)[:, 0]
        ys.append((lab >= 0).astype(int) if kind == 'pos' else (lab > 0.5).astype(int))
        sizes[sp] = len(lab)
    X = [np.vstack(v) for v in Xs]; y = np.concatenate(ys)
    n_tv = sizes['train'] + sizes['valid']
    return X, y, np.arange(n_tv, len(y))


def load_dataset(name):
    if name in MB:
        return load_multibench(name)
    if name == 'pie':
        d = sio.loadmat(os.path.join(EXT, 'RCML', 'data', 'PIE_face_10.mat'))
        y = np.asarray(d['gt']).ravel().astype(int)
        X = [np.asarray(v, dtype=np.float64) for v in d['X'].ravel()]
        X = [v.T if v.shape[0] != len(y) else v for v in X]
        return X, y - y.min(), None
    if name in ('brca', 'rosmap'):
        d = os.path.join(EXT, 'MOGONET', name.upper())
        Xtr = [np.loadtxt(f'{d}/{i}_tr.csv', delimiter=',') for i in (1, 2, 3)]
        Xte = [np.loadtxt(f'{d}/{i}_te.csv', delimiter=',') for i in (1, 2, 3)]
        ytr = np.loadtxt(f'{d}/labels_tr.csv', delimiter=',').astype(int)
        yte = np.loadtxt(f'{d}/labels_te.csv', delimiter=',').astype(int)
        X = [np.vstack([a, b]) for a, b in zip(Xtr, Xte)]
        y = np.concatenate([ytr, yte]); return X, y - y.min(), np.arange(len(ytr), len(y))
    d = sio.loadmat(os.path.join(ROOT, MATS[name]))
    ykey = [k for k in ('Y', 'y', 'gt') if k in d][0]
    y = np.asarray(d[ykey]).ravel().astype(int)
    X = []
    for v in d['X'].ravel():
        v = v.toarray() if sp.issparse(v) else np.asarray(v, dtype=np.float64)
        if v.shape[0] != len(y):
            v = v.T
        X.append(v.astype(np.float64))
    return X, y - y.min(), None


def make_split(y, seed, fixed_test=None):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    if fixed_test is None:
        te = np.concatenate([rng.permutation(np.where(y == c)[0])[:int(round(0.2 * (y == c).sum()))]
                             for c in np.unique(y)])
    else:
        te = fixed_test
    tv = np.setdiff1d(idx, te)
    cal = np.concatenate([rng.permutation(tv[y[tv] == c])[:max(1, int(round(0.1 * (y[tv] == c).sum())))]
                          for c in np.unique(y)])
    tr = np.setdiff1d(tv, cal)
    return rng.permutation(tr), cal, np.sort(te)


def minmax(X, fit_idx):
    out = []
    for x in X:
        lo = x[fit_idx].min(0); hi = x[fit_idx].max(0)
        out.append((x - lo) / np.maximum(hi - lo, 1e-10))
    return out


# ----------------------------------------------------------------- test corruptions (all unseen by RRT)
UNSEEN = {'dropout': [0.3, 0.5, 0.7], 'saltpepper': [0.1, 0.2, 0.4],
          'drift': [(1.5, 0.25), (2.0, 0.5), (3.0, 1.0)], 'laplace': [0.1, 0.3, 0.6]}
FAMS = list(UNSEEN)


def unseen_damage(x, fam, sev, rng):
    if fam == 'gauss':
        return x + rng.normal(0, sev, x.shape)
    if fam == 'dropout':
        y = x.copy(); y[rng.rand(*x.shape) < sev] = 0.0; return y
    if fam == 'saltpepper':
        y = x.copy(); m = rng.rand(*x.shape) < sev; y[m] = rng.randint(0, 2, m.sum()); return y
    if fam == 'drift':
        g, b = sev; return x * g * (1 + rng.uniform(-0.1, 0.1)) + b * (1 + rng.uniform(-0.1, 0.1))
    if fam == 'laplace':
        return x + rng.laplace(0, sev, x.shape)
    raise ValueError(fam)


def build_conditions(Xte, yte, Xall, yall, seed, K):
    """returns {cond: (X list, damaged-mask (n,V), family-code (n,))}"""
    V, n = len(Xte), len(yte)
    conds = {}
    first = {c: np.where(yall == c)[0][0] for c in range(K)}          # RCML exemplar records
    def fresh(): return [x.copy() for x in Xte], np.zeros((n, V), bool), np.full(n, -1)
    # C0
    conds['C0_clean'] = fresh()
    # C1 RCML conflictive protocol (official data.py semantics)
    rng = np.random.RandomState(seed * 100 + 1); X, D, Fm = fresh()
    for i in rng.choice(n, int(0.1 * n), replace=False):
        for v in rng.choice(V, size=rng.randint(V), replace=False):
            X[v][i] = rng.normal(X[v][i], 0.5); D[i, v] = True; Fm[i] = 10
    for i in rng.choice(n, int(0.4 * n), replace=False):
        v = rng.randint(V); X[v][i] = Xall[v][first[(yte[i] + 1) % K]]; D[i, v] = True; Fm[i] = 11
    conds['C1_rcml'] = (X, D, Fm)
    # C2 Gaussian sweep
    for j, sig in enumerate([0.1, 0.3, 0.5, 1.0]):
        rng = np.random.RandomState(seed * 100 + 2 + j); X, D, Fm = fresh()
        for i in rng.choice(n, int(0.5 * n), replace=False):
            for v in rng.choice(V, size=rng.randint(1, V), replace=False):
                X[v][i] = unseen_damage(X[v][i], 'gauss', sig, rng); D[i, v] = True
            Fm[i] = 20 + j
        conds[f'C2_gauss{sig}'] = (X, D, Fm)
    # C3 unseen sensor damage
    rng = np.random.RandomState(seed * 100 + 7); X, D, Fm = fresh()
    for i in rng.choice(n, int(0.5 * n), replace=False):
        f = rng.randint(len(FAMS)); s = UNSEEN[FAMS[f]][rng.randint(3)]
        for v in rng.choice(V, size=rng.randint(1, V), replace=False):
            X[v][i] = unseen_damage(X[v][i], FAMS[f], s, rng); D[i, v] = True
        Fm[i] = f
    conds['C3_unseen'] = (X, D, Fm)
    # C4 pure swap
    rng = np.random.RandomState(seed * 100 + 8); X, D, Fm = fresh()
    for i in rng.choice(n, int(0.4 * n), replace=False):
        v = rng.randint(V); X[v][i] = Xall[v][first[(yte[i] + 1) % K]]; D[i, v] = True; Fm[i] = 11
    conds['C4_swap'] = (X, D, Fm)
    return conds


# ----------------------------------------------------------------- training-time augmentation (seen families only)
def aug_batch(Xb, gen):
    B, V = Xb[0].shape[0], len(Xb)
    dam = torch.rand(B, generator=gen, device=DEV) < 0.5
    k = torch.randint(1, max(V, 2), (B,), generator=gen, device=DEV)
    rk = torch.rand(B, V, generator=gen, device=DEV).argsort(1).argsort(1)
    mask = (rk < k[:, None]) & dam[:, None]
    fam = torch.randint(0, 2, (B,), generator=gen, device=DEV)
    si = torch.randint(0, 3, (B,), generator=gen, device=DEV)
    a = torch.where(fam == 0, torch.tensor(SEEN['uniform'], device=DEV)[si],
                    torch.tensor(SEEN['speckle'], device=DEV)[si])
    out = []
    for v, x in enumerate(Xb):
        u = (torch.rand(x.shape, generator=gen, device=DEV) * 2 - 1) * a[:, None]
        xn = torch.where((fam == 0)[:, None], x + u, x * (1 + u))
        out.append(torch.where(mask[:, v:v + 1], xn, x))
    return out


def T(X):
    return [torch.tensor(x, dtype=torch.float32, device=DEV) for x in X]


def batches(n, bs, gen):
    perm = torch.randperm(n, generator=gen, device=DEV)
    for i in range(0, n, bs):
        yield perm[i:i + bs]


# ----------------------------------------------------------------- official-code baselines
def train_tmc(X, y, K, seed, aug=False, etmc=False, epochs=500, lr=3e-4, bs=200):
    torch.manual_seed(seed); gen = torch.Generator(device=DEV).manual_seed(seed)
    Xt = T(X); yt = torch.tensor(y, device=DEV)
    dims = [[x.shape[1]] for x in X] + ([[sum(x.shape[1] for x in X)]] if etmc else [])
    model = tmc_mod.TMC(K, len(dims), dims, lambda_epochs=50).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    def views(Xb): return {i: v for i, v in enumerate(Xb + ([torch.cat(Xb, 1)] if etmc else []))}
    for ep in range(1, epochs + 1):
        model.train()
        for b in batches(len(y), bs, gen):
            Xb = [x[b] for x in Xt]
            if aug: Xb = aug_batch(Xb, gen)
            _, _, loss = model(views(Xb), yt[b], ep)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    @torch.no_grad()
    def predict(Xn):
        Xb = T(Xn); ev = model.infer(views(Xb))
        alpha = model.DS_Combin({i: e + 1 for i, e in ev.items()})
        return (alpha / alpha.sum(1, keepdim=True)).cpu().numpy()
    return predict


def train_rcml(X, y, K, seed, aug=False, epochs=500, lr=3e-3, bs=200):
    torch.manual_seed(seed); gen = torch.Generator(device=DEV).manual_seed(seed)
    Xt = T(X); yt = torch.tensor(y, device=DEV)
    model = rcml_model.RCML(len(X), np.array([[x.shape[1]] for x in X]), K).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(1, epochs + 1):
        model.train()
        for b in batches(len(y), bs, gen):
            Xb = [x[b] for x in Xt]
            if aug: Xb = aug_batch(Xb, gen)
            ev, ev_a = model({i: v for i, v in enumerate(Xb)})
            loss = rcml_loss.get_loss(ev, ev_a, yt[b], ep, K, 50, 1, DEV)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    @torch.no_grad()
    def predict(Xn):
        _, ev_a = model({i: v for i, v in enumerate(T(Xn))})
        a = ev_a + 1; return (a / a.sum(1, keepdim=True)).cpu().numpy()
    return predict


class QMFv(nn.Module):
    """QMF (Zhang et al. ICML 2023) generalised to V views: energy confidence, dynamic late fusion."""
    def __init__(self, dims, K, hidden=512, p=0.1):
        super().__init__()
        self.clf = nn.ModuleList([nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, K))
                                  for d in dims])
    def forward(self, Xb):
        logits = [c(x) for c, x in zip(self.clf, Xb)]
        conf = [-0.1 * (-torch.logsumexp(z, 1, keepdim=True)) for z in logits]
        return sum(z * c for z, c in zip(logits, conf)), logits, conf


def qmf_rank_loss(confidence, idx, history):
    confidence = confidence.squeeze()
    rank_input1 = confidence; rank_input2 = torch.roll(confidence, -1); idx2 = torch.roll(idx, -1)
    rank_target, rank_margin = history.get_target_margin(idx, idx2)
    rtn = rank_target.clone(); rtn[rtn == 0] = 1
    rank_input2 = rank_input2 + rank_margin / rtn
    return nn.MarginRankingLoss(margin=0.0)(rank_input1, rank_input2, -rank_target)


def train_qmf(X, y, K, seed, aug=False, epochs=500, lr=1e-4, bs=200, lamb=1.0):
    torch.manual_seed(seed); gen = torch.Generator(device=DEV).manual_seed(seed)
    Xt = T(X); yt = torch.tensor(y, device=DEV)
    model = QMFv([x.shape[1] for x in X], K).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    hist = [crl.History(len(y)) for _ in X]
    for ep in range(1, epochs + 1):
        model.train()
        for b in batches(len(y), bs, gen):
            Xb = [x[b] for x in Xt]
            if aug: Xb = aug_batch(Xb, gen)
            fused, logits, conf = model(Xb)
            loss = F.cross_entropy(fused, yt[b]) + sum(F.cross_entropy(z, yt[b]) for z in logits)
            if len(b) > 1:
                rl = 0; bc = b.cpu()
                for v in range(len(X)):
                    rl = rl + qmf_rank_loss(conf[v], bc, hist[v])
                    hist[v].correctness_update(bc, F.cross_entropy(logits[v], yt[b], reduction='none').detach(),
                                               conf[v].squeeze())
                loss = loss + lamb * rl
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    @torch.no_grad()
    def predict(Xn):
        return F.softmax(model(T(Xn))[0], 1).cpu().numpy()
    return predict


def train_mmd(X, y, K, seed, aug=False, ds='', lr=1e-4):
    torch.manual_seed(seed); gen = torch.Generator(device=DEV).manual_seed(seed)
    epochs = {'brca': 2500, 'rosmap': 1000}.get(ds, 1000)
    hidden = [1000] if ds in ('brca', 'rosmap') else [500]
    Xt = T(X); yt = torch.tensor(y, device=DEV)
    model = mmd_mod.MMDynamic([x.shape[1] for x in X], hidden, K, dropout=0.5).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.2)
    for ep in range(epochs + 1):
        model.train()
        Xb = aug_batch(Xt, gen) if aug else Xt
        loss, _ = model(Xb, yt)
        opt.zero_grad(); torch.mean(loss).backward(); opt.step(); sch.step()
    model.eval()
    @torch.no_grad()
    def predict(Xn):
        return F.softmax(model.infer(T(Xn)), 1).cpu().numpy()
    return predict


def train_concat(X, y, K, seed):
    torch.manual_seed(seed)
    Xc = np.hstack(X)
    m = train_mlp(nn.Sequential(nn.Linear(Xc.shape[1], 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, K)),
                  Xc, y, 300, 1e-3, 1e-4, 128, seed)
    return lambda Xn: np.exp(log_softmax_np(logits_of(m, np.hstack(Xn))))


# ----------------------------------------------------------------- expert-based simple baselines (share RRT experts)
def expert_logp(rrt, Xn, restore=False):
    out = []
    for m in range(rrt.V):
        x = rrt.refs[m].apply(Xn[m], rrt.ops[m]) if restore else Xn[m]
        out.append(log_softmax_np(logits_of(rrt.experts[m], x) / rrt.T[m]))
    return np.stack(out)


class TwoT:
    """2T (thesis Paper 2): tau (clean) x beta-net on confidence statistics fitted on seen damage."""
    def __init__(self, rrt, Xcal, ycal, seed):
        self.r = rrt; rng = np.random.RandomState(seed + 3)
        self.nets = []
        for m in range(rrt.V):
            Xd, yd = rrt.cal_damaged[m]
            Xp = np.vstack([Xcal[m], Xd]); yp = np.concatenate([ycal, yd])
            z = logits_of(rrt.experts[m], Xp) / rrt.T[m]
            f = torch.tensor(self._feat(z), dtype=torch.float32); zt = torch.tensor(z, dtype=torch.float32)
            yt = torch.tensor(yp)
            torch.manual_seed(seed + m)
            net = nn.Sequential(nn.Linear(f.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
            opt = torch.optim.Adam(net.parameters(), lr=1e-2)
            for _ in range(300):
                b = F.softplus(net(f)); l = F.cross_entropy(zt * b, yt); opt.zero_grad(); l.backward(); opt.step()
            self.nets.append(net)
        L, B = self._ev(Xcal)
        Lt, Bt, yt = torch.tensor(L), torch.tensor(B), torch.tensor(ycal)
        r = torch.zeros(rrt.V, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([r], max_iter=200, line_search_fn='strong_wolfe')
        def cl():
            opt.zero_grad(); w = F.softplus(r + 0.5413)[:, None] * Bt
            l = F.cross_entropy((w[:, :, None] * Lt).sum(0), yt); l.backward(); return l
        opt.step(cl); self.tau = F.softplus(r + 0.5413).detach().numpy()

    @staticmethod
    def _feat(z):
        p = np.exp(log_softmax_np(z))
        return np.stack([-(p * np.log(p + 1e-12)).sum(1), p.max(1),
                         np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1)], 1)

    @torch.no_grad()
    def _ev(self, Xn):
        L, B = [], []
        for m in range(self.r.V):
            z = logits_of(self.r.experts[m], Xn[m]) / self.r.T[m]
            L.append(log_softmax_np(z))
            B.append(F.softplus(self.nets[m](torch.tensor(self._feat(z), dtype=torch.float32))).numpy().ravel())
        return np.stack(L), np.stack(B).astype(np.float64)

    def predict(self, Xn):
        L, B = self._ev(Xn)
        s = ((self.tau[:, None] * B)[:, :, None] * L).sum(0)
        return np.exp(log_softmax_np(s))


class SoftmaxAttn:
    """ablation: learned softmax attention over (restored) views, scores from validity stats + entropy."""
    def __init__(self, rrt, Xcal, ycal, seed):
        self.r = rrt; rng = np.random.RandomState(seed + 5)
        Xmix = rrt._mv_seen(Xcal, rng)
        Xp = [np.vstack([a, b]) for a, b in zip(Xcal, Xmix)]; yp = np.concatenate([ycal, ycal])
        L, Fe = self._ev(Xp)
        Lt, Ft, yt = torch.tensor(L, dtype=torch.float32), torch.tensor(Fe, dtype=torch.float32), torch.tensor(yp)
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(Ft.shape[2], 32), nn.ReLU(), nn.Linear(32, 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-2)
        for _ in range(300):
            w = torch.softmax(self.net(Ft).squeeze(-1), 0) * rrt.V
            l = F.cross_entropy((w[:, :, None] * Lt).sum(0), yt); opt.zero_grad(); l.backward(); opt.step()

    def _ev(self, Xn):
        L, Fe = [], []
        for m in range(self.r.V):
            st, z = self.r._view_stats(m, Xn[m]); lp = log_softmax_np(z)
            L.append(lp); Fe.append(np.concatenate([st, -(np.exp(lp) * lp).sum(1, keepdims=True)], 1))
        return np.stack(L), np.stack(Fe)

    @torch.no_grad()
    def predict(self, Xn):
        L, Fe = self._ev(Xn)
        w = torch.softmax(self.net(torch.tensor(Fe, dtype=torch.float32)).squeeze(-1), 0).numpy() * self.r.V
        return np.exp(log_softmax_np((w[:, :, None] * L).sum(0)))


# ----------------------------------------------------------------- metrics
def metrics(p, y, D=None, Fm=None):
    pred = p.argmax(1); acc = float((pred == y).mean() * 100)
    nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
    conf = p.max(1); bins = np.linspace(0, 1, 16); ece = 0.0
    for a, b in zip(bins[:-1], bins[1:]):
        mk = (conf > a) & (conf <= b)
        if mk.any(): ece += mk.mean() * abs((pred[mk] == y[mk]).mean() - conf[mk].mean())
    out = dict(acc=acc, f1=float(f1_score(y, pred, average='macro') * 100), nll=nll, ece=float(ece * 100))
    if D is not None and D.any(1).any():
        dm = D.any(1); out['acc_damaged'] = float((pred[dm] == y[dm]).mean() * 100)
        out['acc_intact'] = float((pred[~dm] == y[~dm]).mean() * 100) if (~dm).any() else None
    if Fm is not None and Fm.max() >= 0 and Fm.max() < len(FAMS):
        out['by_family'] = {FAMS[f]: float((pred[Fm == f] == y[Fm == f]).mean() * 100) for f in range(len(FAMS))
                            if (Fm == f).any()}
    return out


# ----------------------------------------------------------------- runner
def run(ds, seed, methods, cfg=None, quick=False, variants=None, lrs=None):
    X, y, fixed = load_dataset(ds)
    K = int(y.max() + 1)
    tr, cal, te = make_split(y, seed, fixed)
    X = minmax(X, np.concatenate([tr, cal]))
    Xtr, Xcal, Xte = [x[tr] for x in X], [x[cal] for x in X], [x[te] for x in X]
    Xtv = [np.vstack([a, b]) for a, b in zip(Xtr, Xcal)]; ytv = np.concatenate([y[tr], y[cal]])
    conds = build_conditions(Xte, y[te], X, y, seed, K)
    res = dict(ds=ds, seed=seed, n=[len(tr), len(cal), len(te)], V=len(X), K=K, dims=[x.shape[1] for x in X],
               results={}, time={})
    ep = 50 if quick else None
    preds = {}
    t0 = time.time()
    rrt = RRT(cfg, seed).fit(Xtr, y[tr], Xcal, y[cal], K)
    res['time']['rrt_fit'] = time.time() - t0
    res['rrt'] = rrt.diagnostics()
    def add(name, f):
        t1 = time.time(); preds[name] = f; res['time'][name] = time.time() - t1
    # ours + ablations (share fitted components)
    add('RRT', rrt.predict_proba)
    for ab, kw in [('RRT-noMERA', dict(ops=['raw'])), ('RRT-noRAT', dict(no_beta=True)),
                   ('RRT-noTau', dict(no_tau=True)), ('RRT-noEps', dict(no_eps=True)),
                   ('RRT-staticTau', dict(release=False)), ('RRT-epsEM', dict(eps_mode='em'))]:
        if ab in methods or 'all' in methods:
            r2 = RRT(dict(cfg or {}, **kw), seed).fit(Xtr, y[tr], Xcal, y[cal], K, base=rrt)
            add(ab, r2.predict_proba); res.setdefault('ablation_diag', {})[ab] = r2.diagnostics()
    for vn, vcfg in (variants or {}).items():                      # dev-only RRT variants
        vc = dict(cfg or {}, **vcfg); refit = vc.pop('refit', False)
        r2 = RRT(vc, seed).fit(Xtr, y[tr], Xcal, y[cal], K, base=None if refit else rrt)
        add('V:' + vn, r2.predict_proba); res.setdefault('variant_diag', {})[vn] = r2.diagnostics()
    if 'RRT-softmax' in methods or 'all' in methods:
        add('RRT-softmax', SoftmaxAttn(rrt, Xcal, y[cal], seed).predict)
    # simple baselines on the same experts
    add('LateAvg', lambda Xn: np.exp(expert_logp(rrt, Xn)).mean(0))
    add('PoE', lambda Xn: np.exp(log_softmax_np(expert_logp(rrt, Xn).sum(0))))
    add('2T', TwoT(rrt, Xcal, y[cal], seed).predict)
    # trained baselines on train+cal (80%)
    kw = dict(epochs=ep) if quick else {}
    LR = lrs or {}
    def lk(n): return {'lr': LR[n]} if n in LR else {}
    trainers = {'Concat': lambda a: train_concat(Xtv, ytv, K, seed),
                'TMC': lambda a: train_tmc(Xtv, ytv, K, seed, aug=a, **kw, **lk('TMC')),
                'ETMC': lambda a: train_tmc(Xtv, ytv, K, seed, aug=a, etmc=True, **kw, **lk('ETMC')),
                'RCML': lambda a: train_rcml(Xtv, ytv, K, seed, aug=a, **kw, **lk('RCML')),
                'QMF': lambda a: train_qmf(Xtv, ytv, K, seed, aug=a, **kw, **lk('QMF')),
                'MMD': lambda a: train_mmd(Xtv, ytv, K, seed, aug=a, ds=ds, **lk('MMD'))}
    if any(m_ in methods for m_ in NEW_BASE) or 'all' in methods or 'new' in methods:
        import mv_baselines2 as MB2
        for nm_, (fn_, _) in MB2.TRAINERS2.items():
            trainers[nm_] = (lambda f_, n_: (lambda a: f_(Xtv, ytv, K, seed, aug=a, aug_fn=aug_batch,
                                                          **({'epochs': ep} if quick else {}), **lk(n_))))(fn_, nm_)
    for name, fn in trainers.items():
        for a in ([False] if name == 'Concat' else [False, True]):
            nm = name + ('+aug' if a else '')
            if 'all' in methods or nm in methods or (name in methods):
                t1 = time.time(); preds[nm] = fn(a); res['time'][nm] = time.time() - t1
    # evaluate
    for nm, f in preds.items():
        res['results'][nm] = {}
        for c, (Xc, D, Fm) in conds.items():
            p = f(Xc)
            res['results'][nm][c] = metrics(p, y[te], D, Fm)
            if hasattr(getattr(f, '__self__', None), 'last_eps'):
                res['results'][nm][c]['eps_used'] = float(f.__self__.last_eps)
            if nm == 'RRT':
                B = rrt.last_B                                       # (V,n) trust weights
                if D.any() and (~D).any():
                    from sklearn.metrics import roc_auc_score
                    res['results'][nm][c]['beta_auroc_damage'] = float(roc_auc_score(D.T.ravel(), -B.ravel()))
                    res['results'][nm][c]['beta_mean_damaged'] = float(B[D.T].mean())
                    res['results'][nm][c]['beta_mean_intact'] = float(B[~D.T].mean())
    res['time']['total'] = time.time() - t0
    return res


BASE_LR = {'TMC': 3e-4, 'ETMC': 3e-4, 'RCML': 3e-3, 'QMF': 1e-4, 'MMD': 1e-4,
           'TMCEK': 3e-3, 'GTMC-HOA': 3e-3, 'TUNED': 3e-3, 'BML': 2e-3, 'PDF': 1e-4, 'MNL': 1e-4}
NEW_BASE = ['TMCEK', 'GTMC-HOA', 'TUNED', 'BML', 'PDF', 'MNL']


def select_lr(ds, seed, mults=(1, 3, 10), only=None):
    """hyper-parameter selection for official baselines: train on train split, score on clean calibration split."""
    X, y, fixed = load_dataset(ds); K = int(y.max() + 1)
    tr, cal, te = make_split(y, seed, fixed)
    X = minmax(X, np.concatenate([tr, cal]))
    Xtr, Xcal = [x[tr] for x in X], [x[cal] for x in X]
    fns = {'TMC': lambda lr: train_tmc(Xtr, y[tr], K, seed, lr=lr),
           'ETMC': lambda lr: train_tmc(Xtr, y[tr], K, seed, etmc=True, lr=lr),
           'RCML': lambda lr: train_rcml(Xtr, y[tr], K, seed, lr=lr),
           'QMF': lambda lr: train_qmf(Xtr, y[tr], K, seed, lr=lr),
           'MMD': lambda lr: train_mmd(Xtr, y[tr], K, seed, ds=ds, lr=lr)}
    if only and any(n in NEW_BASE for n in only):
        import mv_baselines2 as MB2
        for n_, (f_, _) in MB2.TRAINERS2.items():
            fns[n_] = (lambda f: (lambda lr: f(Xtr, y[tr], K, seed, lr=lr)))(f_)
    out = {}
    for n, f in fns.items():
        if only and n not in only:
            continue
        out[n] = {}
        for mlt in mults:
            lr = BASE_LR[n] * mlt
            p = f(lr)(Xcal)
            out[n][str(lr)] = dict(acc=float((p.argmax(1) == y[cal]).mean() * 100),
                                   nll=float(-np.log(np.clip(p[np.arange(len(cal)), y[cal]], 1e-12, 1)).mean()))
    return dict(ds=ds, seed=seed, sel=out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--methods', default='all')
    ap.add_argument('--cfg', default='{}'); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--variants', default='{}')
    ap.add_argument('--select', action='store_true'); ap.add_argument('--lrs', default='{}')
    a = ap.parse_args()
    if a.select:
        r = select_lr(a.ds, a.seed, tuple(json.loads(a.cfg).get('mults', [1, 3, 10])), json.loads(a.cfg).get('only')); json.dump(r, open(a.out, 'w'), indent=1); print(r); sys.exit(0)
    lrs = json.load(open(a.lrs)).get(a.ds, {}) if os.path.exists(a.lrs) else json.loads(a.lrs)
    r = run(a.ds, a.seed, a.methods.split(','), json.loads(a.cfg), a.quick, json.loads(a.variants), lrs)
    r['lrs'] = lrs
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(r, open(a.out, 'w'), indent=1)
    acc = {m: round(v['C0_clean']['acc'], 2) for m, v in r['results'].items()}
    acc3 = {m: round(v['C3_unseen']['acc'], 2) for m, v in r['results'].items()}
    print('C0', acc); print('C3', acc3); print('time', round(r['time']['total'], 1), 'diag', r['rrt'])
