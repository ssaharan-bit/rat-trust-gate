"""Clip-level CREMA-D bank in the standard audit format + segment-token bank, derived from banks/crema_seq.npz.
python crema_clip_bank.py  ->  banks/crema.npz (train/cal/test clean (N,d) = segment mean; var (N,K,d); fam/sub/sev), banks/crema_tok.npz (segment tokens)
Each variant corrupts the whole clip at one family x severity (sub = 0). cal: K=2 variants from families 1,2; test: one variant per family 1..4.
Severities 1..3 of the sequence bank are kept as sev 1..3."""
import numpy as np
z = np.load('banks/crema_seq.npz'); bank = dict(classes=np.array(6), mods=np.array(['audio', 'video'])); tok = {}
for split in ('train', 'cal', 'test'):
    y = z[f'{split}_y']; N = len(y); bank[f'{split}_y'] = y; bank[f'{split}_idx'] = np.arange(N)
    K = 0 if split == 'train' else (4 if split == 'test' else 2)
    for mi, m in enumerate(('audio', 'video')):
        rng = np.random.default_rng(1000 * mi + {'train': 0, 'cal': 1, 'test': 2}[split])
        cl = z[f'{split}_{m}_clean']; bank[f'{split}_{m}_clean'] = cl.astype(np.float32).mean(1).astype(np.float16); tok[f'{split}_{m}_tok_clean'] = cl
        if K == 0:
            bank[f'{split}_{m}_var'] = np.zeros((N, 0, cl.shape[-1]), np.float16)
            for f_ in ('fam', 'sub', 'sev'): bank[f'{split}_{m}_{f_}'] = np.zeros((N, 0), np.int16)
            continue
        fam = np.stack([rng.permutation([1, 2])[:2] for _ in range(N)]) if K == 2 else np.tile([1, 2, 3, 4], (N, 1))
        sev = rng.integers(1, 4, (N, K)); idx = (fam - 1) * 3 + sev - 1
        V = z[f'{split}_{m}_var'][np.arange(N)[:, None], idx]          # (N,K,T,d)
        bank[f'{split}_{m}_var'] = V.astype(np.float32).mean(2).astype(np.float16); tok[f'{split}_{m}_tok_var'] = V
        bank[f'{split}_{m}_fam'] = fam.astype(np.int16); bank[f'{split}_{m}_sub'] = np.zeros_like(fam, np.int16); bank[f'{split}_{m}_sev'] = sev.astype(np.int16)
        for f_ in ('fam', 'sub', 'sev'): tok[f'{split}_{m}_tok_{f_}'] = bank[f'{split}_{m}_{f_}']
    bank[f'{split}_meta_actor'] = z[f'{split}_actor']
np.savez('banks/crema.npz', **bank); np.savez('banks/crema_tok.npz', **tok); print('saved', {k: v.shape for k, v in bank.items() if k.endswith('_var')})
