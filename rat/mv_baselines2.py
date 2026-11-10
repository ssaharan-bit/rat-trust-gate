"""Adapters for 2024-26 official baselines (code imported unchanged from the cloned repos):
TMCEK (ICML'25), GTMC-HOA (IEEE TMM'26), TUNED (AAAI'25), BML (CVPR'26)."""
import os, sys, importlib, importlib.util, types
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, 'ext')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def _isolated_import(paths, modnames):
    """import repo-local modules that use bare intra-repo imports, then scrub them from sys.modules"""
    before = set(sys.modules); old = list(sys.path)
    for p in reversed(paths):
        sys.path.insert(0, p)
    try:
        mods = {n: importlib.import_module(n) for n in modnames}
    finally:
        sys.path[:] = old
        for k in set(sys.modules) - before:
            if k.split('.')[0] in ('model', 'utilities', 'utils', 'loss_function', 'data', 'EvidenceCollector',
                                   'GTMC_HOA', 'View_Block', 'RCML', 'RCML_View', 'Attention', 'Dot_Attention'):
                sys.modules.pop(k, None)
    return mods


_TMCEK = os.path.join(EXT, 'jie019_TMCEK_ICML2025', 'TMCEK', 'Muti-view classification')
tmcek_model = _by_path('tmcek_model', os.path.join(_TMCEK, 'model.py'))
tmcek_loss = _by_path('tmcek_loss', os.path.join(_TMCEK, 'loss_function.py'))
_G = os.path.join(EXT, 'lshi91_GTMC-HOA')
_g = _isolated_import([_G, os.path.join(_G, 'model')], ['GTMC_HOA', 'View_Block'])
GTMC_HOA, ViewBlock = _g['GTMC_HOA'].GTMC_HOA, _g['View_Block'].ViewBlock
_T = os.path.join(EXT, 'JethroJames_TUNED')
_t = _isolated_import([_T], ['model.handwritten', 'utils'])
TUNED, tuned_utils = _t['model.handwritten'].TUNED, _t['utils']
tuned_loss = _by_path('tuned_loss', os.path.join(_T, 'loss_function.py'))
bml = _by_path('bml_mv', os.path.join(EXT, 'XLearning-SCU_2026-CVPR-BML', 'multi_view.py'))


def _T_(X):
    return [torch.tensor(x, dtype=torch.float32, device=DEV) for x in X]


def _batches(n, bs, gen):
    perm = torch.randperm(n, generator=gen, device=DEV)
    for i in range(0, n, bs):
        yield perm[i:i + bs]


# ------------------------------------------------------------------ TMCEK (ICML 2025)
def train_tmcek(X, y, K, seed, aug=False, aug_fn=None, epochs=500, lr=3e-3, bs=200):
    torch.manual_seed(seed); gen = torch.Generator(device=DEV).manual_seed(seed)
    Xt = _T_(X); yt = torch.tensor(y, device=DEV)
    model = tmcek_model.TMCEK(len(X), np.array([[x.shape[1]] for x in X]), K).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(1, epochs + 1):
        model.train()
        for b in _batches(len(y), bs, gen):
            Xb = [x[b] for x in Xt]
            if aug: Xb = aug_fn(Xb, gen)
            ev, ev_a = model({i: v for i, v in enumerate(Xb)})
            loss = tmcek_loss.get_loss(ev, ev_a, yt[b], ep, K, 50, 1, DEV)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    @torch.no_grad()
    def predict(Xn):
        _, ev_a = model({i: v for i, v in enumerate(_T_(Xn))})
        _, g = tmcek_loss.compute_gini(ev_a, K)
        a = ev_a + (g + 1) / 2
        return (a / a.sum(1, keepdim=True)).cpu().numpy()
    return predict


# ------------------------------------------------------------------ GTMC-HOA (IEEE TMM 2026)
def train_gtmc(X, y, K, seed, aug=False, aug_fn=None, epochs=500, lr=3e-3, bs=400, comm=64):
    torch.manual_seed(seed); gen = torch.Generator(device=DEV).manual_seed(seed)
    Xt = _T_(X); yt = torch.tensor(y, device=DEV)
    args = types.SimpleNamespace(gamma=1.0, delta=1.0, eta=0.001)
    blocks = [ViewBlock(v, X[v].shape[1], comm) for v in range(len(X))]
    model = GTMC_HOA(blocks, comm, K, 50, args, DEV).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(1, epochs + 1):
        model.train()
        for b in _batches(len(y), bs, gen):
            Xb = [x[b] for x in Xt]
            if aug: Xb = aug_fn(Xb, gen)
            opt.zero_grad()
            _, ev_a, loss = model(Xb, yt[b], ep)
            loss.backward(); opt.step()
    model.eval()
    @torch.no_grad()
    def predict(Xn):
        Xb = _T_(Xn); out = []
        for i in range(0, len(Xn[0]), 1024):
            sl = [x[i:i + 1024] for x in Xb]
            _, ev_a, _ = model(sl, torch.zeros(len(sl[0]), dtype=torch.long, device=DEV), epochs + 1)
            a = ev_a + 1; out.append((a / a.sum(1, keepdim=True)).cpu().numpy())
        return np.vstack(out)
    return predict


# ------------------------------------------------------------------ TUNED (AAAI 2025)
def _graphs(Xb, neighbor):
    A = []
    for x in Xb:
        Av, _ = tuned_utils.build_CAN(x.detach().cpu().T, neighbor)
        A.append(Av.to(DEV).float())
    return A


def train_tuned(X, y, K, seed, aug=False, aug_fn=None, epochs=500, lr=3e-3, bs=64, neighbor=20, psi=0.7,
                annealing=50):
    torch.manual_seed(seed); gen = torch.Generator(device=DEV).manual_seed(seed)
    Xt = _T_(X); yt = torch.tensor(y, device=DEV)
    model = TUNED(len(X), [[x.shape[1]] for x in X], K, psi=psi).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(1, epochs + 1):
        model.train(); ce = min(1.0, ep / epochs)
        for b in _batches(len(y), bs, gen):
            if len(b) <= neighbor + 1:
                continue
            Xb = [x[b] for x in Xt]
            if aug: Xb = aug_fn(Xb, gen)
            ev, ev_a = model(Xb, _graphs(Xb, neighbor), ce_scale=ce)
            loss = tuned_loss.get_loss(ev, ev_a, yt[b], ep, K, annealing, 1.0, DEV)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    @torch.no_grad()
    def predict(Xn):
        # official evaluation uses a shuffled test order (random batches); class-sorted batches would leak labels
        n = len(Xn[0]); perm = np.random.RandomState(seed + 991).permutation(n); inv = np.argsort(perm)
        Xb = _T_([x[perm] for x in Xn]); out = []
        for i in range(0, n, bs):
            sl = [x[i:i + bs] for x in Xb]
            nb = min(neighbor, len(sl[0]) - 2)
            _, ev_a = model(sl, _graphs(sl, max(nb, 1)), ce_scale=0)
            a = ev_a + 1; out.append((a / a.sum(1, keepdim=True)).cpu().numpy())
        return np.vstack(out)[inv]
    return predict


# ------------------------------------------------------------------ BML (CVPR 2026)
def train_bml(X, y, K, seed, aug=False, aug_fn=None, epochs=200, lr=2e-3, bs=2048, feat_dim=128,
              augment_ratio=0.5, lambda_w=1.0):
    bml.set_seed(seed); gen = torch.Generator(device=DEV).manual_seed(seed)
    args = types.SimpleNamespace(dims_list=[x.shape[1] for x in X], feat_dim=feat_dim, num_classes=K, eps=1e-8)
    model = bml.MultiViewBackbone(args).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-8)
    Xf = [x.astype(np.float32) for x in X]; idx = np.arange(len(y))
    for ep in range(epochs):
        model.train()
        ds = bml.NC_MultiViewDataset(Xf, y.astype(np.int64), idx, noise_seed=ep + 1000 * seed, noise_ratio=augment_ratio)
        perm = torch.randperm(len(ds.labels), generator=torch.Generator().manual_seed(seed * 7919 + ep))
        D = [torch.tensor(d, device=DEV) for d in ds.data_list]
        L = torch.tensor(ds.labels, device=DEV); I = torch.tensor(ds.indicators, device=DEV).float()
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs].to(DEV)
            if len(b) < 2:
                continue
            Xb = [d[b] for d in D]
            if aug: Xb = aug_fn(Xb, gen)
            _, fused, rel = model(Xb)
            loss = F.cross_entropy(fused, L[b]) + lambda_w * F.binary_cross_entropy(rel, I[b])
            loss.backward(); opt.step(); opt.zero_grad()
        sch.step()
    model.eval()
    @torch.no_grad()
    def predict(Xn):
        _, fused, _ = model(_T_(Xn)); return F.softmax(fused, 1).cpu().numpy()
    return predict


TRAINERS2 = {'TMCEK': (train_tmcek, 3e-3), 'GTMC-HOA': (train_gtmc, 3e-3), 'TUNED': (train_tuned, 3e-3),
             'BML': (train_bml, 2e-3)}


# ------------------------------------------------------------------ PDF (ICML 2024): predictive dynamic fusion, V-view port
class _PDFv(nn.Module):
    def __init__(self, dims, K, h=512, p=0.1):
        super().__init__()
        self.enc = nn.ModuleList([nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)) for d in dims])
        self.clf = nn.ModuleList([nn.Linear(h, K) for _ in dims])
        self.conf = nn.ModuleList([nn.Sequential(nn.Linear(h, 2 * h), nn.Linear(2 * h, h), nn.Linear(h, 1), nn.Sigmoid())
                                   for _ in dims])

    def forward(self, Xb, test=False):
        f = [e(x) for e, x in zip(self.enc, Xb)]
        lg = [c(z) for c, z in zip(self.clf, f)]
        tcp = [cn(z if test else z.detach()) for cn, z in zip(self.conf, f)]
        lt = torch.cat([torch.log(t.clamp_min(1e-8)) for t in tcp], 1)          # (B,V)
        tot = lt.sum(1, keepdim=True)
        holo = (tot - lt) / (tot + 1e-8)                                         # sum_{j!=m} log tcp_j / sum_j log tcp_j
        cb = torch.cat(tcp, 1) + holo
        if test:                                                                 # relative calibration
            du = torch.stack([torch.mean(torch.abs(F.softmax(z, 1) - 1.0 / z.shape[1]), 1) for z in lg], 1)
            cb = cb * (du / du.max(1, keepdim=True).values.clamp_min(1e-8))
        else:
            cb = cb.detach()
        w = torch.softmax(cb, 1)
        fused = sum(w[:, m:m + 1] * lg[m] for m in range(len(lg)))
        return fused, lg, tcp


def train_pdf(X, y, K, seed, aug=False, aug_fn=None, epochs=500, lr=1e-4, bs=200):
    torch.manual_seed(seed); gen = torch.Generator(device=DEV).manual_seed(seed)
    Xt = _T_(X); yt = torch.tensor(y, device=DEV)
    model = _PDFv([x.shape[1] for x in X], K).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(1, epochs + 1):
        model.train()
        for b in _batches(len(y), bs, gen):
            Xb = [x[b] for x in Xt]
            if aug: Xb = aug_fn(Xb, gen)
            fused, lg, tcp = model(Xb)
            loss = F.cross_entropy(fused, yt[b]) + sum(F.cross_entropy(z, yt[b]) for z in lg)
            for z, t in zip(lg, tcp):
                p_true = F.softmax(z, 1).gather(1, yt[b][:, None]).detach()
                loss = loss + F.l1_loss(t, p_true)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    @torch.no_grad()
    def predict(Xn):
        return F.softmax(model(_T_(Xn), test=True)[0], 1).cpu().numpy()
    return predict


# ------------------------------------------------------------------ MNL (NeurIPS 2025): multimodal negative learning, V-view port
class _LFv(nn.Module):
    def __init__(self, dims, K, h=512, p=0.1):
        super().__init__()
        self.clf = nn.ModuleList([nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, K)) for d in dims])

    def forward(self, Xb):
        lg = [c(x) for c, x in zip(self.clf, Xb)]
        return sum(lg) / len(lg), lg


def train_mnl(X, y, K, seed, aug=False, aug_fn=None, epochs=500, lr=1e-4, bs=128, w1=0.7):
    torch.manual_seed(seed); gen = torch.Generator(device=DEV).manual_seed(seed)
    Xt = _T_(X); yt = torch.tensor(y, device=DEV)
    model = _LFv([x.shape[1] for x in X], K).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    ar = None
    for ep in range(1, epochs + 1):
        model.train()
        for b in _batches(len(y), bs, gen):
            Xb = [x[b] for x in Xt]
            if aug: Xb = aug_fn(Xb, gen)
            out, lg = model(Xb); t = yt[b]; ar = torch.arange(len(t), device=DEV)
            label = F.one_hot(t, K).float()
            loss = F.cross_entropy(out, t) + sum(F.cross_entropy(z, t) for z in lg)
            po = F.softmax(out, 1).clone(); po[ar, t] = -1; j = po.argmax(1)
            xi = torch.stack([z[ar, t] - z[ar, j] for z in lg], 1)                     # (B,V) target margin
            pt = torch.stack([F.softmax(z, 1)[ar, t] for z in lg], 1)                 # (B,V)
            teach = pt.argmax(1)                                                       # strongest view per sample
            for m in range(len(lg)):
                tl = torch.stack(lg, 1)[ar, teach].detach().softmax(1)
                valid = ((xi[ar, teach] > xi[:, m]) & (pt[ar, teach] > pt[:, m]) & (teach != m)).float()[:, None]
                if valid.any():
                    ls = lg[m].log_softmax(1)
                    boost = -(label * ls * valid) - w1 * (tl * ls * valid) + w1 * (label * tl * ls * valid)
                    loss = loss + boost.mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    @torch.no_grad()
    def predict(Xn):
        return F.softmax(model(_T_(Xn))[0], 1).cpu().numpy()
    return predict


TRAINERS2.update({'PDF': (train_pdf, 1e-4), 'MNL': (train_mnl, 1e-4)})
