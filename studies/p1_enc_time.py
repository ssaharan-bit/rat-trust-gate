import json, sys, time, os, numpy as np, torch
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoConfig, AutoModel, Wav2Vec2Config, Wav2Vec2Model
dev = 'cuda'
def tm(f, reps):
    for _ in range(3): f()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    return dict(p50=float(np.percentile(ts, 50)), p95=float(np.percentile(ts, 95)))
res = {'note': 'architectures built from configs (random weights); timing does not depend on weight values; fp16'}
mk = {'dinov2-base': lambda: AutoModel.from_config(AutoConfig.from_pretrained('facebook/dinov2-base')),
      'bert-base-uncased': lambda: AutoModel.from_config(AutoConfig.from_pretrained('bert-base-uncased')),
      'wav2vec2-base': lambda: Wav2Vec2Model(Wav2Vec2Config())}
for name, f in mk.items():
    try:
        m = f().half().to(dev).eval()
        for bs in (1, 64):
            if name.startswith('dino'): x = dict(pixel_values=torch.randn(bs, 3, 224, 224, device=dev, dtype=torch.float16))
            elif name.startswith('bert'): x = dict(input_ids=torch.randint(1000, 5000, (bs, 64), device=dev))
            else: x = dict(input_values=torch.randn(bs, 16000 * 3, device=dev, dtype=torch.float16))
            with torch.no_grad(): res[f'{name}|bs{bs}'] = tm(lambda: m(**x), 100 if bs == 1 else 20)
        del m; torch.cuda.empty_cache()
    except Exception as e:
        res[name] = repr(e)[:300]
json.dump(res, open(sys.argv[1], 'w'), indent=1); print(res)
