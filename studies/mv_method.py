"""RRT: Restore-then-Reweight Trust for vector multi-view data.

Components (thesis): MERA restoration (outlier-only memory restoration / James-Stein
common-mode rejection, chosen per view on calibration data with a clean guard),
RAT validity trust beta_m(x) (clean-referenced typicality -> isotonic law fitted on
seen damage), clean-fitted utility/dependence temperature tau_m (2T), and
non-competitive log-linear evidence attention with a contamination floor eps.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEN = {'uniform': [0.2, 0.5, 1.0], 'speckle': [0.5, 1.0, 2.0]}
CFG = dict(hidden=256, drop=0.3, epochs=300, lr=1e-3, wd=1e-4, bs=128,
           n_mem=64, alpha=0.05, margin=0.02, knn=10, guard=0.5, tau_l2=0.1, eps_mode='cal',
           eps_em_grid=[0.0, 1e-3, 3e-3, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
           eps_grid=[0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1],
           release=True,                                   # frozen after dev (Amendment 1)
           expert_grid=[dict(hidden=256, drop=0.3, wd=1e-4, lr=1e-3), dict(hidden=512, drop=0.5, wd=1e-3, lr=1e-3),
                        dict(hidden=256, drop=0.5, wd=1e-2, lr=1e-3), dict(hidden=512, drop=0.3, wd=1e-4, lr=3e-3),
                        dict(hidden=512, drop=0.5, wd=1e-3, lr=1e-2)])


# ----------------------------------------------------------------- experts
class Expert(nn.Module):
    def __init__(self, d, K, h, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, K))

    def forward(self, x):
        return self.net(x)


class GatedExpert(nn.Module):
    """feature-informativeness gating (MMDynamics-style sigmoid gate with L1) in front of an MLP expert"""
    def __init__(self, d, K, h, p):
        super().__init__()
        self.gate = nn.Linear(d, d); self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, K))
        self.last_gate = None

    def forward(self, x):
        g = torch.sigmoid(self.gate(x)); self.last_gate = g
        return self.net(x * g)


def _seen_aug_torch(x, gen):
    B = x.shape[0]
    dam = torch.rand(B, 1, generator=gen, device=x.device) < 0.5
    fam = torch.rand(B, 1, generator=gen, device=x.device) < 0.5
    si = torch.randint(0, 3, (B,), generator=gen, device=x.device)
    a = torch.where(fam[:, 0], torch.tensor(SEEN['uniform'], device=x.device)[si],
                    torch.tensor(SEEN['speckle'], device=x.device)[si])[:, None]
    u = (torch.rand(x.shape, generator=gen, device=x.device) * 2 - 1) * a
    return torch.where(dam, torch.where(fam, x + u, x * (1 + u)), x)


def train_mlp(model, X, y, epochs, lr, wd, bs, seed, aug=False, l1=0.0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    ga = torch.Generator(device=DEV).manual_seed(seed + 17)
    model.to(DEV).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV); yt = torch.tensor(y, device=DEV)
    n = len(y)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(DEV)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            xb = _seen_aug_torch(Xt[b], ga) if aug else Xt[b]
            loss = F.cross_entropy(model(xb), yt[b])
            if l1 > 0 and getattr(model, 'last_gate', None) is not None:
                loss = loss + l1 * model.last_gate.mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def logits_of(model, X):
    return model(torch.tensor(X, dtype=torch.float32, device=DEV)).cpu().numpy()


def fit_temp(z, y):
    zt = torch.tensor(z, dtype=torch.float64); yt = torch.tensor(y)
    lt = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([lt], max_iter=100, line_search_fn='strong_wolfe')
    def cl():
        opt.zero_grad(); l = F.cross_entropy(zt / lt.exp(), yt); l.backward(); return l
    opt.step(cl)
    return float(lt.exp())


def log_softmax_np(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


# ----------------------------------------------------------------- damage (seen, for calibration)
def seen_damage(x, fam, sev, rng):
    u = rng.uniform(-sev, sev, x.shape)
    return x + u if fam == 'uniform' else x * (1 + u)


# ----------------------------------------------------------------- MERA for vectors
def kmeans(X, k, seed, iters=50):
    rng = np.random.RandomState(seed)
    k = min(k, max(2, len(X) // 10))
    C = X[rng.choice(len(X), k, replace=False)].copy()
    Xt = torch.tensor(X, device=DEV, dtype=torch.float32)
    Ct = torch.tensor(C, device=DEV, dtype=torch.float32)
    for _ in range(iters):
        a = torch.cdist(Xt, Ct).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                Ct[j] = Xt[m].mean(0)
    return Ct.cpu().numpy()


class ViewRef:
    """Clean reference geometry for one view (fitted on clean train features)."""

    def __init__(self, Xtr, seed, cfg):
        self.cfg = cfg
        self.lo = np.quantile(Xtr, 0.005, axis=0); self.hi = np.quantile(Xtr, 0.995, axis=0)
        rngw = np.maximum(self.hi - self.lo, 1e-6)
        self.lo = self.lo - cfg['margin'] * rngw; self.hi = self.hi + cfg['margin'] * rngw
        self.M = kmeans(Xtr, cfg['n_mem'], seed)
        rs = np.random.RandomState(seed)
        self.bank = Xtr[rs.choice(len(Xtr), min(2000, len(Xtr)), replace=False)]
        # clean residual quantiles per dim (for survival)
        a = self._nearest(Xtr, np.ones_like(Xtr, bool))
        r = np.abs(Xtr - self.M[a])
        self.qs = np.linspace(0, 1, 201)
        self.rq = np.quantile(r, self.qs, axis=0)            # (201, d)
        g, b = self._gainoff_batch(Xtr)
        self.th_tr = float(((g - 1.0) ** 2 + b ** 2).mean())

    def _nearest(self, X, mask):
        Xt = torch.tensor(X, device=DEV, dtype=torch.float32)
        Mt = torch.tensor(self.M, device=DEV, dtype=torch.float32)
        mt = torch.tensor(mask, device=DEV, dtype=torch.float32)
        # masked squared distance / number of used dims
        d = (mt ** 2 * Xt ** 2).sum(1, keepdim=True) - 2 * (mt * Xt) @ Mt.T + mt @ (Mt ** 2).T
        d = d / mt.sum(1, keepdim=True).clamp_min(1)
        return d.argmin(1).cpu().numpy()

    def _gainoff_batch(self, X):
        """affine common-mode fit x ~ g*a + b, a = clean atom with max Pearson correlation
        (affine-invariant matching, so the match is not corrupted by the shift itself)."""
        Xc = X - X.mean(1, keepdims=True); Mc = self.M - self.M.mean(1, keepdims=True)
        Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
        Mn = Mc / (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-12)
        a = (Xn @ Mn.T).argmax(1)
        A = self.M[a]; Ac = A - A.mean(1, keepdims=True)
        g = (Xc * Ac).sum(1) / np.maximum((Ac ** 2).sum(1), 1e-12)
        b = X.mean(1) - g * A.mean(1)
        return g, b

    def survival(self, r):
        # P_clean(R >= r) per dim, by interpolation in the clean residual quantiles
        S = np.empty_like(r)
        for d in range(r.shape[1]):
            S[:, d] = 1.0 - np.interp(r[:, d], self.rq[:, d], self.qs, left=0.0, right=1.0)
        return S

    def restore_mr(self, X):
        inr = (X >= self.lo) & (X <= self.hi)
        a = self._nearest(X, inr)
        A = self.M[a]
        S = self.survival(np.abs(X - A))
        v = np.minimum(1.0, S / self.cfg['alpha'])
        return v * X + (1 - v) * A, float((v < 1).mean())

    def restore_cmr(self, X):
        g, b = self._gainoff_batch(X)
        n2 = (g - 1.0) ** 2 + b ** 2
        s = np.maximum(0.0, 1.0 - self.th_tr / np.maximum(n2, 1e-12))   # James-Stein shrinkage
        g2, b2 = 1.0 + s * (g - 1.0), s * b
        return (X - b2[:, None]) / np.maximum(g2, 0.1)[:, None]

    def apply(self, X, op):
        if op == 'raw':
            return X
        if op == 'mr':
            return self.restore_mr(X)[0]
        if op == 'cmr':
            return self.restore_cmr(X)
        if op == 'cmr+mr':
            return self.restore_mr(self.restore_cmr(X))[0]
        raise ValueError(op)

    def validity_stats(self, Xraw, Xres):
        """clean-referenced typicality stats (higher = less valid)."""
        Bt = torch.tensor(self.bank, device=DEV, dtype=torch.float32)
        Xt = torch.tensor(Xres, device=DEV, dtype=torch.float32)
        dk = torch.cdist(Xt, Bt).topk(self.cfg['knn'] + 1, largest=False).values[:, 1:].mean(1).cpu().numpy()
        oor = ((Xraw < self.lo) | (Xraw > self.hi)).mean(1)
        mag = np.linalg.norm(Xraw - Xres, axis=1)
        return np.stack([np.log(dk + 1e-8), oor, np.log1p(mag)], 1)


# ----------------------------------------------------------------- RRT
class RRT:
    def __init__(self, cfg=None, seed=0):
        self.cfg = dict(CFG, **(cfg or {})); self.seed = seed

    # ---- fit ----
    def fit(self, Xtr, ytr, Xcal, ycal, K, base=None):
        cfg, seed = self.cfg, self.seed
        V = len(Xtr); self.V, self.K = V, K
        rng = np.random.RandomState(seed + 7)
        self.logpi = np.log(np.bincount(ytr, minlength=K) / len(ytr) + 1e-12)
        self._Xcal_ref = Xcal
        self.experts, self.T, self.refs, self.ops = [], [], [], []
        if base is not None:                      # ablations reuse the same experts / references
            self.experts, self.T, self.refs = base.experts, base.T, base.refs
        self.expert_choice = []
        ext = cfg.get('_external_experts')
        if ext is not None and base is None:      # plug-in mode: use a backbone's per-view heads as experts
            for m in range(V):
                zc = logits_of(ext[m], Xcal[m]); self.experts.append(ext[m]); self.T.append(fit_temp(zc, ycal))
                self.expert_choice.append(-1); self.refs.append(ViewRef(Xtr[m], seed + m, cfg))
        for m in range(V if (base is None and ext is None) else 0):
            grid = cfg.get('expert_grid') or [dict(hidden=cfg['hidden'], drop=cfg['drop'], wd=cfg['wd'])]
            best = None
            for gi, g in enumerate(grid):                     # per-view selection by clean calibration NLL
                torch.manual_seed(seed * 31 + m)
                Arch = GatedExpert if g.get('gate') else Expert
                _Xe, _ye = cfg['_expert_train'][m] if cfg.get('_expert_train') is not None else (Xtr[m], ytr)
                e = train_mlp(Arch(Xtr[m].shape[1], K, g['hidden'], g['drop']), _Xe, _ye,
                              g.get('epochs', cfg['epochs']), g.get('lr', cfg['lr']), g['wd'], cfg['bs'], seed * 31 + m,
                              aug=g.get('aug', False), l1=g.get('l1', 0.0))
                zc = logits_of(e, Xcal[m]); Tm = fit_temp(zc, ycal)
                nll = float(-log_softmax_np(zc / Tm)[np.arange(len(ycal)), ycal].mean())
                if best is None or nll < best[0]:
                    best = (nll, e, Tm, gi)
            self.experts.append(best[1]); self.T.append(best[2]); self.expert_choice.append(best[3])
            self.refs.append(ViewRef(Xtr[m], seed + m, cfg))
        # seen-damage calibration pool per view (synthetic seen families, or supplied real seen variants)
        self.cal_damaged = []
        sup = cfg.get('_cal_damaged')                      # list per view of (Xd, yd), e.g. bank families A/B
        for m in range(V):
            if sup is not None:
                self.cal_damaged.append(sup[m]); continue
            Xd, yd = [], []
            for fam, sevs in SEEN.items():
                for s in sevs:
                    Xd.append(seen_damage(Xcal[m], fam, s, rng)); yd.append(ycal)
            self.cal_damaged.append((np.vstack(Xd), np.concatenate(yd)))
        # (1) operator selection with clean guard
        for m in range(V):
            best, best_acc = 'raw', -1
            raw_clean = (logits_of(self.experts[m], Xcal[m]).argmax(1) == ycal).mean()
            for op in self.cfg.get('ops', ['raw', 'mr', 'cmr', 'cmr+mr']):
                cl = (logits_of(self.experts[m], self.refs[m].apply(Xcal[m], op)).argmax(1) == ycal).mean()
                Xd, yd = self.cal_damaged[m]
                da = (logits_of(self.experts[m], self.refs[m].apply(Xd, op)).argmax(1) == yd).mean()
                if cl >= raw_clean - cfg['guard'] / 100 and da > best_acc + 1e-9:
                    best, best_acc = op, da
            self.ops.append(best)
        # (2) validity -> log beta law (RAT)
        self.stat_ref, self.laws = [], []
        for m in range(V):
            st_c, lp_c = self._view_stats(m, Xcal[m])
            Xd, yd = self.cal_damaged[m]
            st_d, lp_d = self._view_stats(m, Xd)
            self.stat_ref.append(np.sort(st_c, 0))                 # clean-cal reference distribution
            q = np.concatenate([self._surprisal(m, st_c), self._surprisal(m, st_d)])
            z = np.concatenate([lp_c, lp_d]); yy = np.concatenate([ycal, yd])
            self.laws.append(self._fit_law(q, z, yy))
        # (3) tau (log-linear, beta applied): 'clean' | 'mixed' (clean + seen damage) | L2 prior on log tau
        Lc, Bc = self._evidence(Xcal)
        Xmix = self._mv_seen(Xcal, rng)
        Lm, Bm = self._evidence(Xmix, eps=None)
        if cfg.get('tau_fit', 'clean') == 'mixed':
            self.tau = self._fit_tau(np.concatenate([Lc, Lm], 1), np.concatenate([Bc, Bm], 1),
                                     np.concatenate([ycal, ycal]))
        else:
            self.tau = self._fit_tau(Lc, Bc, ycal)
        # (4) eps on clean + seen multi-view damage
        best = None
        for eps in cfg['eps_grid']:
            nll = 0.0
            for (L, B, yy) in ((Lc, Bc, ycal), (Lm, Bm, ycal)):
                s = self._score(L, B, self.tau, eps)
                nll += -log_softmax_np(s)[np.arange(len(yy)), yy].mean()
            if best is None or nll < best[0]:
                best = (nll, eps)
        self.eps = best[1]
        return self

    def _mv_seen(self, X, rng):
        if self.cfg.get('_mv_seen') is not None and X is self._Xcal_ref:
            return self.cfg['_mv_seen']                    # supplied multi-view seen-damage calibration set
        n, V = len(X[0]), len(X)
        out = [x.copy() for x in X]
        for i in range(n):
            if rng.rand() < 0.5:
                k = rng.randint(1, V) if V > 1 else 1
                fam = list(SEEN)[rng.randint(2)]; s = SEEN[fam][rng.randint(3)]
                for v in rng.choice(V, k, replace=False):
                    out[v][i] = seen_damage(X[v][i:i + 1], fam, s, rng)[0]
        return out

    def _view_stats(self, m, Xraw):
        Xres = self.refs[m].apply(Xraw, self.ops[m])
        z = logits_of(self.experts[m], Xres) / self.T[m]
        return self.refs[m].validity_stats(Xraw, Xres), z

    def _surprisal(self, m, st):
        ref = self.stat_ref[m]; n = len(ref)
        S = np.stack([1.0 - np.searchsorted(ref[:, j], st[:, j], side='left') / n for j in range(st.shape[1])], 1)
        return (-np.log(np.clip(S, 1.0 / (n + 1), 1.0))).mean(1)

    def _fit_law(self, q, z, y, nb=12):
        edges = np.unique(np.quantile(q, np.linspace(0, 1, nb + 1)))
        cs, bs, ws = [], [], []
        lsz = torch.tensor(z, dtype=torch.float64); yt = torch.tensor(y)
        grid = np.concatenate([np.linspace(0.0, 0.2, 11), np.linspace(0.25, 2.0, 36)])
        for a, b in zip(edges[:-1], edges[1:]):
            mk = (q >= a) & (q <= b)
            if mk.sum() < 20:
                continue
            zz, yy = lsz[mk], yt[mk]
            nl = [float(F.cross_entropy(zz * g, yy)) for g in grid]
            cs.append(float(np.median(q[mk]))); bs.append(grid[int(np.argmin(nl))]); ws.append(mk.sum())
        iso = IsotonicRegression(increasing=False, out_of_bounds='clip').fit(cs, bs, sample_weight=ws)
        return iso

    def _evidence(self, X, eps=None):
        L, B = [], []
        for m in range(self.V):
            st, z = self._view_stats(m, X[m])
            L.append(log_softmax_np(z))
            if self.cfg.get('no_beta'):
                B.append(np.ones(len(z)))
            else:
                B.append(np.clip(self.laws[m].predict(self._surprisal(m, st)), 0.0, 2.0))
        return np.stack(L), np.stack(B)                            # (V,n,K), (V,n)

    def _score(self, L, B, tau, eps):
        """Bayes contamination form (theory T8b): s(y) = log pi(y) + sum_m log[(1-eps) r_m(y) + eps],
        r_m = tempered posterior / prior, tempered posterior ∝ q_m^{tau_m beta_m(x)}."""
        if self.cfg.get('release') and self.V > 1:
            # dependence release (theory T2-iii, first order): tau_m^(mean partner trust)
            Bc = np.minimum(B, 1.0)
            part = (Bc.sum(0, keepdims=True) - Bc) / (self.V - 1)
            w = np.power(tau[:, None], part) * B
        else:
            w = tau[:, None] * B                                   # non-competitive trust weights
        if self.cfg.get('floor_mode') == 'untempered':
            # T8c fix: contaminate the untempered posterior toward the prior, then temper. Each view's
            # contribution (incl. the floor) scales with its own weight, so the floor is tau-invariant.
            Lc = L if eps <= 0 else np.logaddexp(np.log1p(-eps) + L, np.log(eps) + self.logpi[None, None, :])
            lr = w[:, :, None] * (Lc - self.logpi[None, None, :])
            return self.logpi[None, :] + lr.sum(0)
        Z = w[:, :, None] * L
        Zm = Z.max(2, keepdims=True)
        Lt = Z - Zm - np.log(np.exp(Z - Zm).sum(2, keepdims=True))
        lr = Lt - self.logpi[None, None, :]
        if eps > 0:
            lr = np.logaddexp(np.log1p(-eps) + lr, np.log(eps))
        return self.logpi[None, :] + lr.sum(0)

    def _em_eps(self, L, B):
        """label-free ML estimate of the contamination rate on an unlabelled batch:
        argmax_eps mean_i log sum_y pi(y) prod_m [(1-eps) r_m(y) + eps]."""
        best = None
        for eps in self.cfg['eps_em_grid']:
            s = self._score(L, B, self.tau, eps)
            sm = s.max(1, keepdims=True)
            ll = float((sm.ravel() + np.log(np.exp(s - sm).sum(1))).mean())
            if best is None or ll > best[0]:
                best = (ll, eps)
        return best[1]

    def _fit_tau(self, L, B, y):
        if self.cfg.get('no_tau'):
            return np.ones(self.V)
        Lt = torch.tensor(L); Bt = torch.tensor(B); yt = torch.tensor(y)
        r = torch.zeros(self.V, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([r], max_iter=200, line_search_fn='strong_wolfe')
        lam = float(self.cfg.get('tau_l2', 0.0))
        def cl():
            opt.zero_grad()
            t = F.softplus(r + 0.5413)
            w = t[:, None] * Bt
            s = (w[:, :, None] * Lt).sum(0)
            l = F.cross_entropy(s, yt) + lam * (torch.log(t) ** 2).sum(); l.backward(); return l
        opt.step(cl)
        return F.softplus(r + 0.5413).detach().numpy()

    # ---- predict ----
    def predict_proba(self, X, eps=None):
        L, B = self._evidence(X)
        e = self.eps if eps is None else eps
        if self.cfg.get('eps_mode') == 'em' and eps is None:
            e = self._em_eps(L, B)
        if self.cfg.get('no_eps'):
            e = 0.0
        self.last_eps = e
        s = self._score(L, B, self.tau, e)
        p = np.exp(log_softmax_np(s))
        self.last_B = B
        return p

    def diagnostics(self):
        return dict(ops=self.ops, tau=[float(t) for t in self.tau], eps=float(self.eps), T=self.T,
                    expert_choice=getattr(self, 'expert_choice', None))
