"""Paper 1 revision study C: overhead of the gate (SUN RGB-D bank, PDF backbone, seed 240)."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1_plugin_conf as PC
import numpy as np, torch
import mv_baselines2 as MB2, mv_bench as BN
from mv_method import RRT

ds = sys.argv[1] if len(sys.argv) > 2 else 'bank:sun'
Xtr, ytr, Xcal, ycal, yte, K, rcfg, conds, lrs, ep, bank = PC.setup(ds, 240); V = len(Xtr)
pr = MB2.train_pdf(Xtr, ytr, K, 240, lr=lrs.get('PDF', 1e-4), epochs=ep(500)); M = PC.getmodel(pr)
heads = [PC.Head(M, v, 'pdf') for v in range(V)]
rx = RRT(dict(rcfg, _external_experts=heads), 240).fit(Xtr, ytr, Xcal, ycal, K)
Xc = [x[:1000] for x in conds['B0_clean'][0]]
def tnat():
    lg, w = PC.parts('pdf', M, Xc); return (w[:, :, None] * lg).sum(0)
def tgate():
    lg, w = PC.parts('pdf', M, Xc); _, B = rx._evidence(Xc); return (w[:, :, None] * B[:, :, None] * lg).sum(0)
res = {}
for nm, f in [('native', tnat), ('gated', tgate)]:
    for _ in range(3): f()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    for _ in range(20): f()
    torch.cuda.synchronize(); res[nm] = dict(ms_per_1000=(time.time() - t0) / 20 * 1000, peak_mb=torch.cuda.max_memory_allocated() / 2**20)
res['bank_mb'] = float(sum(r.bank.nbytes + r.M.nbytes + r.rq.nbytes for r in rx.refs) / 2**20)
res['law_params'] = int(sum(len(l.X_thresholds_) * 2 for l in rx.laws))
res['backbone_params'] = int(sum(p.numel() for p in M.parameters()))
res['throughput_native'] = 1000 / (res['native']['ms_per_1000'] / 1000); res['throughput_gated'] = 1000 / (res['gated']['ms_per_1000'] / 1000)
json.dump(res, open(sys.argv[-1], 'w'), indent=1); print(res)
