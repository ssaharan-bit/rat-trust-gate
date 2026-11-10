"""CREMA-D temporal banks for RQ5 (plan step 2). Raw-signal corruptions, per-segment frozen features on a T=8 timeline.
python crema_seq_extract.py --out banks/crema_seq.npz [--limit 40]
audio: wav2vec2-base last hidden state, adaptive-avg-pooled to 8 windows; video: DINOv2-base CLS of 8 uniformly sampled frames.
Each clip is encoded clean and fully corrupted for 4 families x 3 severities per modality; time-varying corruption is assembled
segment-wise from these banks (segment t takes the variant at severity s_t). fam 1 A noise, 2 B clipping/blur (seen), 3 C dropout/random occlusion,
4 D environment: low-pass / haze (unseen). Actor-disjoint split (seed 0): 55 train / 18 cal / 18 test actors."""
import os, glob, time, argparse, numpy as np, torch, cv2, torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt
R1 = os.environ.get('RAT_RAW', '/path/to/raw_data')  # folder with the raw CREMA-D files
WAV = f'{R1}/paper1_dynasense_lite/data/crema_d/data/AudioWAV'; VID = f'{R1}/prl_v2/data/crema_d_video'
EMO = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']; T = 8; dev = 'cuda'
# severities calibrated on 300 cal-actor clips (crema_sev_probe.py) to ~80/50/25% of clean above-chance accuracy
A_LV = {1: [30, 25, 20], 2: [20, 50, 120], 3: [0.6, 0.4, 0.25], 4: [2000, 1500, 1000]}
V_LV = {1: [2, 4, 12], 2: [2, 2.5, 3], 3: [.05, .1, .2], 4: [.2, .1, .05]}

def load_wav(p):
    sr, w = wavfile.read(p); w = w.astype(np.float32); w = w.mean(1) if w.ndim > 1 else w
    w = w / 32768.0 if np.abs(w).max() > 1.5 else w; assert sr == 16000; return w

def load_frames(p):
    cap = cv2.VideoCapture(p); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    cap.release()
    if len(fr) < T: return None
    out = []
    for i in np.linspace(0, len(fr) - 1, T).round().astype(int):
        f = cv2.cvtColor(fr[i], cv2.COLOR_BGR2RGB); h, w = f.shape[:2]; s = min(h, w)
        out.append(cv2.resize(f[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s], (224, 224)))
    return np.stack(out).astype(np.float32) / 255.0

def audio_variants(w, rng):
    out = []; p = np.mean(w ** 2) + 1e-12
    for fam in (1, 2, 3, 4):
        for lv in A_LV[fam]:
            if fam == 1: x = w + rng.normal(0, np.sqrt(p / 10 ** (lv / 10)), w.shape)
            elif fam == 2: x = np.clip(w * lv, -1, 1)
            elif fam == 3: x = w * lv + rng.normal(0, 1e-3, w.shape)
            else: x = sosfiltfilt(butter(4, lv, fs=16000, output='sos'), w)
            out.append(x.astype(np.float32))
    return out

def blur(x, s):
    k = int(2 * round(3 * s) + 1); g = torch.exp(-(torch.arange(k, device=dev) - k // 2) ** 2 / (2 * s * s)); g = (g / g.sum()).float()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode='reflect'), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode='reflect'), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)

def video_variants(v, gen):
    out = []
    for fam in (1, 2, 3, 4):
        for lv in V_LV[fam]:
            if fam == 1: x = (v + torch.randn(v.shape, generator=gen, device=dev) * lv / 255).clamp(0, 1)
            elif fam == 2: x = blur(v, lv)
            elif fam == 3:
                x = v.clone(); a = int(224 * np.sqrt(lv)); r_ = torch.randint(0, 224 - a + 1, (2,), generator=gen, device=dev).tolist(); x[:, :, r_[0]:r_[0] + a, r_[1]:r_[1] + a] = 0
            else: x = (v * lv + (1 - lv) * 0.6).clamp(0, 1)  # haze / fog: contrast reduction toward grey
            out.append(x)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out'); ap.add_argument('--limit', type=int, default=0); a = ap.parse_args()
    from transformers import AutoModel, Wav2Vec2Model
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{WAV}/*.wav'))
    ids = [i for i in ids if os.path.exists(f'{VID}/{i}.flv') and os.path.getsize(f'{VID}/{i}.flv') > 20000]
    actors = sorted({i[:4] for i in ids}); perm = np.random.default_rng(0).permutation(actors)
    split_of = {ac: ('train' if k < 55 else 'cal' if k < 73 else 'test') for k, ac in enumerate(perm)}
    if a.limit: ids = [i for sp in ('train', 'cal', 'test') for i in [j for j in ids if split_of[j[:4]] == sp][:a.limit]]
    vit = AutoModel.from_pretrained('facebook/dinov2-base', torch_dtype=torch.float16).to(dev).eval()
    w2v = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base', cache_dir=f'{R1}/prl_v2/hf_cache/hub').to(dev).eval()
    mu = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]; sd = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    D = {sp: dict(a0=[], av=[], v0=[], vv=[], y=[], actor=[], dur=[]) for sp in ('train', 'cal', 'test')}; t0 = time.time(); skipped = 0
    with torch.no_grad():
        for n, i in enumerate(ids):
            fr = load_frames(f'{VID}/{i}.flv')
            if fr is None: skipped += 1; continue
            w = load_wav(f'{WAV}/{i}.wav'); rng = np.random.default_rng(n); gen = torch.Generator(device=dev).manual_seed(n)
            ws = np.stack([w] + audio_variants(w, rng)); ws = (ws - ws.mean(1, keepdims=True)) / (ws.std(1, keepdims=True) + 1e-7)
            h = w2v(torch.from_numpy(ws).float().to(dev)).last_hidden_state.transpose(1, 2); A = F.adaptive_avg_pool1d(h, T).transpose(1, 2).half().cpu().numpy()
            v = torch.from_numpy(fr).permute(0, 3, 1, 2).to(dev); V = torch.stack([v] + video_variants(v, gen))  # (13,T,3,224,224)
            V = ((V.flatten(0, 1) - mu) / sd).half(); Vf = vit(pixel_values=V).pooler_output.view(13, T, -1).cpu().numpy()
            d = D[split_of[i[:4]]]; d['a0'].append(A[0]); d['av'].append(A[1:]); d['v0'].append(Vf[0]); d['vv'].append(Vf[1:])
            d['y'].append(EMO.index(i.split('_')[2])); d['actor'].append(int(i[:4])); d['dur'].append(len(w) / 16000)
            if n % 500 == 0: print(n, len(ids), round(time.time() - t0), flush=True)
    bank = dict(classes=6, mods=np.array(['audio', 'video']), T=T, fam=np.repeat([1, 2, 3, 4], 3), sev=np.tile([1, 2, 3], 4))
    for sp, d in D.items():
        bank[f'{sp}_audio_clean'] = np.stack(d['a0']); bank[f'{sp}_audio_var'] = np.stack(d['av']); bank[f'{sp}_video_clean'] = np.stack(d['v0'])
        bank[f'{sp}_video_var'] = np.stack(d['vv']); bank[f'{sp}_y'] = np.array(d['y']); bank[f'{sp}_actor'] = np.array(d['actor']); bank[f'{sp}_dur'] = np.array(d['dur'])
        print(sp, bank[f'{sp}_audio_var'].shape, bank[f'{sp}_video_var'].shape, np.bincount(bank[f'{sp}_y']))
    np.savez(a.out, **bank); print('skipped', skipped, 'sec', round(time.time() - t0), 'saved', a.out)
