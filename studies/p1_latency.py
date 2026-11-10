"""Paper 1 rev2: latency profile (SUN RGB-D, PDF, seed 250) and encoder cost."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1_plugin_conf as PC
import numpy as np, torch
import mv_baselines2 as MB2
from mv_method import RRT, DEV
import p1_rev2 as R2

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def tm(f, reps):
    ts = []
    for _ in range(3): f()
    for _ in range(reps):
        sync(); t0 = time.perf_counter(); f(); sync(); ts.append((time.perf_counter() - t0) * 1000)
    return dict(p50=float(np.percentile(ts, 50)), p95=float(np.percentile(ts, 95)), mean=float(np.mean(ts)))

Xtr, ytr, Xcal, ycal, yte, K, rcfg, conds, lrs, ep, bank = PC.setup('bank:sun', 250); V = len(Xtr)
pr = MB2.train_pdf(Xtr, ytr, K, 250, lr=lrs.get('PDF', 1e-4), epochs=ep(500)); M = PC.getmodel(pr)
rx = RRT(dict(rcfg), 250).fit(Xtr, ytr, Xcal, ycal, K)
Xc = conds['B0_clean'][0]
res = {'cpu': os.popen("lscpu | grep 'Model name'").read().strip(), 'gpu': torch.cuda.get_device_name(0), 'torch_threads': torch.get_num_threads()}

def beta_only(X):
    B = []
    for m in range(V):
        ref = rx.refs[m]; Xres = ref.apply(X[m], rx.ops[m]); st = ref.validity_stats(X[m], Xres)
        B.append(np.clip(rx.laws[m].predict(rx._surprisal(m, st)), 0, 2))
    return np.stack(B)

def stages(X):
    t = {}
    for m in range(V):
        ref = rx.refs[m]
        sync(); a = time.perf_counter(); Xres = ref.apply(X[m], rx.ops[m]); sync(); b = time.perf_counter()
        st = ref.validity_stats(X[m], Xres); sync(); c = time.perf_counter()
        q = rx._surprisal(m, st); d = time.perf_counter(); rx.laws[m].predict(q); e = time.perf_counter()
        for k, v in (('restore', b - a), ('stats_knn', c - b), ('surprisal', d - c), ('isotonic', e - d)): t[k] = t.get(k, 0) + v * 1000
    return t

for nm, X in (('batch1000', [x[:1000] for x in Xc]), ('single', [x[:1] for x in Xc])):
    reps = 20 if nm == 'batch1000' else 200
    res[nm] = dict(native=tm(lambda: (lambda lw: (lw[1][:, :, None] * lw[0]).sum(0))(PC.parts('pdf', M, X)), reps),
                   beta_only=tm(lambda: beta_only(X), reps), evidence_full=tm(lambda: rx._evidence(X), reps),
                   ops=list(rx.ops))
    st = [stages(X) for _ in range(reps)]; res[nm]['stages_ms'] = {k: float(np.mean([s[k] for s in st])) for k in st[0]}
# bank size scaling (batch 1000)
X = [x[:1000] for x in Xc]; full = [r.bank for r in rx.refs]; res['bank'] = {}
for N in (256, 512, 1000, 2000):
    for m in range(V): rx.refs[m].bank = full[m][:N]
    res['bank'][N] = tm(lambda: beta_only(X), 20)
for m in range(V): rx.refs[m].bank = full[m]
# encoders
os.environ['HF_HUB_OFFLINE'] = '1'
try:
    from transformers import AutoModel
    enc = {}
    dm = AutoModel.from_pretrained('facebook/dinov2-base').to(DEV).eval()
    for bs in (1, 64):
        x = torch.randn(bs, 3, 224, 224, device=DEV)
        with torch.no_grad(): enc[f'dinov2_base_bs{bs}_ms'] = tm(lambda: dm(pixel_values=x), 20 if bs == 64 else 100)
    del dm; torch.cuda.empty_cache()
    bm = AutoModel.from_pretrained('bert-base-uncased').to(DEV).eval()
    for bs in (1, 64):
        ids = torch.randint(1000, 5000, (bs, 64), device=DEV)
        with torch.no_grad(): enc[f'bert_base_bs{bs}_ms'] = tm(lambda: bm(input_ids=ids), 20 if bs == 64 else 100)
    res['encoders'] = enc
except Exception as e:
    res['encoders'] = {'error': repr(e)}
json.dump(res, open(sys.argv[1], 'w'), indent=1); print(json.dumps(res, indent=1))
