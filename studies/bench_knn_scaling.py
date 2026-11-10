"""Scalability of the kNN validity statistic (the step that takes ~92% of RAT's time).
Times the exact operation used by RAT: Euclidean distances from a query batch to a clean bank on the GPU,
followed by the 11 smallest (the nearest is dropped and the next 10 are averaged).
Usage: python bench_knn_scaling.py out.json"""
import sys, json, time, numpy as np, torch
dev = torch.device('cuda'); torch.manual_seed(0)
def tm(f, reps):
    for _ in range(3): f()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))
def knn(X, B):
    d = torch.cdist(X, B); return d.topk(11, largest=False).values[:, 1:].mean(1)
res = {'gpu': torch.cuda.get_device_name(0), 'd': 768, 'rows': []}
for N in (2000, 10000, 50000, 100000):
    B = torch.randn(N, 768, device=dev)
    torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    for bs in (1, 1000):
        X = torch.randn(bs, 768, device=dev)
        for V in (2, 4, 6):
            t = tm(lambda: [knn(X, B) for _ in range(V)], 50 if bs == 1000 else 200)
            res['rows'].append(dict(bank=N, batch=bs, modalities=V, ms=t))
    res['rows'].append(dict(bank=N, bank_MB=B.numel() * 4 / 2**20, peak_extra_MB=(torch.cuda.max_memory_allocated() - base) / 2**20))
    del B; torch.cuda.empty_cache()
json.dump(res, open(sys.argv[1], 'w'), indent=1); print(json.dumps(res))
