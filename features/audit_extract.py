"""Signal-audit feature banks: frozen encoders on clean + raw-signal-corrupted inputs.
python audit_extract.py --ds sun|milk|food --root <tmc_gaps2> [--smoke]
Bank layout (npz): {split}_y, {split}_meta_*, {split}_{mod}_clean (N,D), {split}_{mod}_var (N,K,D), {split}_{mod}_fam/sub/sev (N,K)
fam: 1..4 = A noise, B blur/compression, C occlusion/dropout, D environment. train/cal use A,B only; test has one variant per family."""
import argparse, os, glob, io, json, html, re, zlib, time, numpy as np, torch
from PIL import Image
from scipy import ndimage

SUN19 = ['bathroom', 'bedroom', 'classroom', 'computer_room', 'conference_room', 'corridor', 'dining_area', 'dining_room', 'discussion_area',
         'furniture_store', 'home_office', 'kitchen', 'lab', 'lecture_theatre', 'library', 'living_room', 'office', 'rest_space', 'study_space']
IM = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def rng_for(*key): return np.random.default_rng(zlib.crc32('|'.join(map(str, key)).encode()))

# ------------------------------------------------------------ image loading
def load_rgb(path):
    im = Image.open(path).convert('RGB'); w, h = im.size; s = 256 / min(w, h)
    im = im.resize((max(IM, round(w * s)), max(IM, round(h * s))), Image.BILINEAR); w, h = im.size
    l, t = (w - IM) // 2, (h - IM) // 2; return np.asarray(im.crop((l, t, l + IM, t + IM)), np.float32) / 255.

def load_sun_depth(path):
    raw = np.asarray(Image.open(path)).astype(np.uint16); d = ((raw >> 3) | (raw << 13)).astype(np.float32) / 1000.
    d = np.clip(d / 8.0, 0, 1); im = Image.fromarray(d); w, h = im.size; s = 256 / min(w, h)
    im = im.resize((max(IM, round(w * s)), max(IM, round(h * s))), Image.NEAREST); w, h = im.size
    l, t = (w - IM) // 2, (h - IM) // 2; return np.asarray(im.crop((l, t, l + IM, t + IM)), np.float32)

def smooth_mask(rng, frac, scale=12):
    f = ndimage.gaussian_filter(rng.random((IM, IM)), scale); return f > np.quantile(f, 1 - frac)

# ------------------------------------------------------------ corruptions (applied at 224x224, raw signal)
def corrupt_rgb(x, fam, sub, sev, rng):
    s = sev - 1
    if fam == 1:
        if sub == 0: y = x + rng.normal(0, [.04, .08, .12, .18, .26][s], x.shape)
        else:
            p = [.01, .03, .06, .1, .17][s]; u = rng.random(x.shape[:2]); y = x.copy(); y[u < p / 2] = 0; y[(u >= p / 2) & (u < p)] = 1
    elif fam == 2:
        if sub == 0: sg = [1, 2, 3, 4.5, 6][s]; y = ndimage.gaussian_filter(x, (sg, sg, 0))
        else:
            b = io.BytesIO(); Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).save(b, 'JPEG', quality=[25, 15, 10, 6, 3][s])
            y = np.asarray(Image.open(b), np.float32) / 255.
    elif fam == 3:
        y = x.copy()
        if sub == 0: y[smooth_mask(rng, [.05, .1, .2, .3, .45][s], 20)] = 0.5
        else:
            rows = np.repeat(rng.random(IM // 8) < [.1, .2, .35, .5, .65][s], 8); y[rows] = 0
    else:
        if sub == 0:
            c = [.6, .45, .3, .2, .12][s]; haze = 0.55 + 0.25 * ndimage.gaussian_filter(rng.random((IM, IM)), 25)[..., None] / 0.5
            y = c * x + (1 - c) * np.clip(haze, 0, 1)
        else:
            a = [.5, .35, .22, .14, .08][s]; y = x * a + rng.normal(0, 0.01 * (s + 1), x.shape)
    return np.clip(y, 0, 1).astype(np.float32)

def corrupt_depth(d, fam, sub, sev, rng):
    s = sev - 1; valid = d > 0; y = d.copy()
    if fam == 1:
        if sub == 0: y = d + rng.normal(0, [.01, .02, .04, .07, .1][s], d.shape) * valid
        else: y = d * (1 + rng.normal(0, [.02, .05, .1, .15, .25][s], d.shape))
    elif fam == 2:
        if sub == 0: lv = [64, 32, 16, 8, 4][s]; y = np.round(d * lv) / lv
        else: sg = [1, 2, 3, 4.5, 6][s]; y = ndimage.gaussian_filter(d, sg)
    elif fam == 3:
        if sub == 0: y[smooth_mask(rng, [.05, .1, .2, .3, .45][s], 10)] = 0
        else:
            cols = np.repeat(rng.random(IM // 8) < [.1, .2, .35, .5, .65][s], 8); y[:, cols] = 0
    else:
        if sub == 0:
            f = rng.uniform(0.05, 0.15); ph = rng.uniform(0, 6.28); xx = np.arange(IM)[None, :]
            y = d + [.02, .05, .1, .15, .2][s] * np.sin(2 * np.pi * f * xx + ph) * valid
        else: y[d > [.6, .45, .35, .25, .15][s]] = 0
    y[~valid] = 0; return np.clip(y, 0, 1).astype(np.float32)

def depth_to_img(d): return np.repeat(d[..., None], 3, -1)

def to_tensor(x): return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy())

CONF = {'o': '0', 'l': '1', 'i': '1', 'e': 'c', 'a': '@', 's': '5', 'b': '6', 'g': '9', 't': '7', 'm': 'rn', 'n': 'h'}
def corrupt_text(t, fam, sub, sev, rng, vocab, foreign):
    s = sev - 1; w = t.split()
    if fam == 1:
        r = [.03, .06, .1, .15, .25][s] if sub == 0 else [.1, .2, .3, .45, .6][s]; out = []
        for ch in t:
            if rng.random() < r and ch.isalpha():
                if sub == 0:
                    op = rng.integers(3)
                    out.append('' if op == 0 else ch + chr(97 + rng.integers(26)) if op == 1 else chr(97 + rng.integers(26)))
                else: out.append(CONF.get(ch.lower(), chr(97 + rng.integers(26))))
            else: out.append(ch)
        return ''.join(out)
    if fam == 2:
        r = [.1, .2, .3, .45, .6][s]
        if sub == 0: w = [vocab[rng.integers(len(vocab))] if rng.random() < r else x for x in w]
        else: w = [re.sub('[aeiou]', '', x) if rng.random() < r and len(x) > 3 else x for x in w]
        return ' '.join(w)
    if fam == 3:
        n = len(w); f = [.1, .2, .35, .5, .7][s] if sub == 0 else [.15, .3, .45, .6, .75][s]; k = max(1, int(round(f * n)))
        if sub == 0: st = rng.integers(0, max(1, n - k + 1)); w = w[:st] + w[st + k:]
        else: w = w[:max(1, n - k)]
        return ' '.join(w) if w else '.'
    if sub == 0:
        w = w[:]; nsw = int(round([.1, .25, .5, .75, 1.0][s] * len(w)))
        for _ in range(nsw):
            i, j = rng.integers(0, len(w), 2); w[i], w[j] = w[j], w[i]
        return ' '.join(w)
    k = [1, 2, 3, 5, 8][s]; ins = ' '.join(foreign[rng.integers(len(foreign))].split()[0] for _ in range(k))
    return ins + ' ' + t if rng.random() < .5 else t + ' ' + ins

# ------------------------------------------------------------ MILK metadata modality
SITES = None
def meta_vec(age, sex, site, tone, miss):
    v = np.zeros(4 + 3 + len(SITES) + 1 + 7, np.float32)
    if not miss.get('age') and age == age: v[0] = age / 100.
    else: v[1] = 1
    v[2 + {'male': 0, 'female': 1}.get(sex if not miss.get('sex') else '', 2)] = 1
    o = 5; v[o + (SITES.index(site) if (not miss.get('site') and site in SITES) else len(SITES))] = 1
    o = 5 + len(SITES) + 1; v[o + (int(tone) if (not miss.get('tone') and tone == tone and 0 <= int(tone) <= 5) else 6)] = 1
    return v

def corrupt_meta(rec, fam, sub, sev, rng):
    s = sev - 1; age, sex, site, tone = rec; miss = {}
    if fam == 1:
        if sub == 0: age = age + rng.normal(0, [5, 10, 15, 25, 35][s]) if age == age else age
        elif rng.random() < [.1, .2, .35, .5, .7][s] and tone == tone: tone = int(np.clip(tone + rng.choice([-2, -1, 1, 2]), 0, 5))
    elif fam == 2:
        if sub == 0 and age == age: bw = [10, 20, 30, 50, 100][s]; age = (age // bw) * bw + bw / 2
        elif sub == 1 and rng.random() < [.2, .35, .5, .7, .9][s]: miss['site'] = True
    elif fam == 3:
        k = [1, 1, 2, 3, 4][s] if sub == 0 else 3
        for f in rng.permutation(['age', 'sex', 'site', 'tone'])[:k]: miss[f] = True
    else:
        p = [.1, .2, .35, .5, .7][s]
        if rng.random() < p: sex = {'male': 'female', 'female': 'male'}.get(sex, sex)
        if rng.random() < p: site = SITES[rng.integers(len(SITES))]
    return meta_vec(age, sex, site, tone, miss)

# ------------------------------------------------------------ dataset indices
def index_sun(root, smoke):
    items = []
    for sc in sorted(glob.glob(f'{root}/SUNRGBD/**/scene.txt', recursive=True)):
        lab = open(sc).read().strip()
        if lab not in SUN19: continue
        d = os.path.dirname(sc); im = glob.glob(d + '/image/*.jpg') + glob.glob(d + '/image/*.png'); dp = glob.glob(d + '/depth/*.png')
        if im and dp: items.append(dict(rgb=im[0], depth=dp[0], y=SUN19.index(lab), sensor=sc.split('SUNRGBD/')[1].split('/')[0]))
    return items, ['rgb', 'depth'], 19

def index_milk(root, smoke):
    import csv
    global SITES
    meta = list(csv.DictReader(open(f'{root}/MILK10k/MILK10k_Training_Metadata.csv', encoding='utf-8-sig')))
    gt = {r['lesion_id']: r for r in csv.DictReader(open(f'{root}/MILK10k/MILK10k_Training_GroundTruth.csv', encoding='utf-8-sig'))}
    classes = [k for k in next(iter(gt.values())).keys() if k != 'lesion_id']
    SITES = sorted({r['site'] for r in meta if r['site']}); by = {}
    for r in meta: by.setdefault(r['lesion_id'], {})['clin' if r['image_type'].startswith('clinical') else 'derm'] = r
    items = []
    for lid, d in sorted(by.items()):
        if 'clin' not in d or 'derm' not in d or lid not in gt: continue
        r = d['derm']; age = float(r['age_approx']) if r['age_approx'] else float('nan'); tone = float(r['skin_tone_class']) if r['skin_tone_class'] else float('nan')
        items.append(dict(clin=f"{root}/MILK10k/MILK10k_Training_Input/{lid}/{d['clin']['isic_id']}.jpg", derm=f"{root}/MILK10k/MILK10k_Training_Input/{lid}/{r['isic_id']}.jpg",
                          meta=(age, r['sex'], r['site'], tone), y=int(np.argmax([float(gt[lid][c]) for c in classes])), tone=tone))
    return items, ['clin', 'derm', 'meta'], len(classes)

def index_food(root, smoke):
    import csv
    out = {}
    for sp in ('train', 'test'):
        rows = list(csv.reader(open(f'{root}/UPMC-Food101/texts/{sp}_titles.csv', encoding='utf-8', errors='ignore')))
        out[sp] = [dict(img=f'{root}/UPMC-Food101/images/{sp}/{r[2]}/{r[0]}', text=html.unescape(r[1]), lab=r[2]) for r in rows if len(r) >= 3]
    classes = sorted({r['lab'] for r in out['train']}); items = []
    for sp in ('train', 'test'):
        for r in out[sp]:
            if os.path.exists(r['img']): items.append(dict(img=r['img'], text=r['text'], y=classes.index(r['lab']), official=sp))
    return items, ['img', 'text'], len(classes)

def make_splits(ds, items, smoke):
    rng = np.random.default_rng(0); y = np.array([it['y'] for it in items]); idx = {'train': [], 'cal': [], 'test': []}
    if ds == 'food':
        off = np.array([it['official'] for it in items]); per = (6, 2, 3) if smoke else (200, 50, 100)
        for c in np.unique(y):
            tr = rng.permutation(np.where((y == c) & (off == 'train'))[0]); te = rng.permutation(np.where((y == c) & (off == 'test'))[0])
            idx['train'] += list(tr[:per[0]]); idx['cal'] += list(tr[per[0]:per[0] + per[1]]); idx['test'] += list(te[:per[2]])
    else:
        for c in np.unique(y):
            ii = rng.permutation(np.where(y == c)[0]); n = len(ii); a, b = int(.6 * n), int(.7 * n)
            idx['train'] += list(ii[:a]); idx['cal'] += list(ii[a:b]); idx['test'] += list(ii[b:])
        if smoke: idx = {k: list(rng.permutation(v)[:{'train': 600, 'cal': 150, 'test': 300}[k]]) for k, v in idx.items()}
    return {k: np.array(sorted(v)) for k, v in idx.items()}

# ------------------------------------------------------------ worker dataset
class DS(torch.utils.data.Dataset):
    def __init__(s, ds, items, ids, split, mods, K, fams, vocab=None, foreign=None):
        s.__dict__.update(ds=ds, items=items, ids=ids, split=split, mods=mods, K=K, fams=fams, vocab=vocab, foreign=foreign)
    def __len__(s): return len(s.ids)
    def __getitem__(s, j):
        i = int(s.ids[j]); it = s.items[i]; out = {}
        for m in s.mods:
            rng = rng_for(s.ds, s.split, i, m); st = [(int(rng.choice(s.fams)), int(rng.integers(2)), int(rng.integers(1, 6))) for _ in range(s.K)]
            if s.split == 'test': st = [(f, int(rng.integers(2)), int(rng.integers(1, 6))) for f in (1, 2, 3, 4)]
            if m in ('rgb', 'clin', 'derm', 'img'):
                x = load_rgb(it[m]); xs = [x] + [corrupt_rgb(x, *a, rng) for a in st]; out[m] = torch.stack([to_tensor(v) for v in xs])
            elif m == 'depth':
                d = load_sun_depth(it['depth']); ds_ = [d] + [corrupt_depth(d, *a, rng) for a in st]; out[m] = torch.stack([to_tensor(depth_to_img(v)) for v in ds_])
            elif m == 'text':
                out[m] = [it['text']] + [corrupt_text(it['text'], *a, rng, s.vocab, s.foreign) for a in st]
            elif m == 'meta':
                out[m] = torch.tensor(np.stack([meta_vec(*it['meta'], {})] + [corrupt_meta(it['meta'], *a, rng) for a in st]))
            out[m + '_st'] = np.array(st, np.int16).reshape(-1, 3)
        out['y'] = it['y']; return out

def collate(b): return b

@torch.no_grad()
def run(ds, root, smoke):
    from transformers import AutoModel, AutoTokenizer
    t0 = time.time(); items, mods, C = {'sun': index_sun, 'milk': index_milk, 'food': index_food}[ds](root, smoke)
    sp = make_splits(ds, items, smoke); print(ds, len(items), {k: len(v) for k, v in sp.items()}, 'classes', C, round(time.time() - t0, 1), flush=True)
    dev = 'cuda'; vit = AutoModel.from_pretrained('facebook/dinov2-base', torch_dtype=torch.float16).to(dev).eval()
    vocab = foreign = None; bert = tok = None
    if 'text' in mods:
        bert = AutoModel.from_pretrained('bert-base-uncased', torch_dtype=torch.float16).to(dev).eval(); tok = AutoTokenizer.from_pretrained('bert-base-uncased')
        vocab = sorted({w for i in sp['train'] for w in items[i]['text'].split()}); foreign = [items[i]['text'] for i in sp['train']]
    bank = dict(classes=np.array(C), mods=np.array(mods))
    for split in ('train', 'cal', 'test'):
        K = 4 if split == 'test' else 2
        dl = torch.utils.data.DataLoader(DS(ds, items, sp[split], split, mods, K, [1, 2], vocab, foreign), batch_size=16, num_workers=24, collate_fn=collate)
        acc = {m: [] for m in mods}; st = {m: [] for m in mods}; ys = []
        for b in dl:
            ys += [r['y'] for r in b]
            for m in mods:
                if m == 'text':
                    txt = [t for r in b for t in r[m]]; e = tok(txt, padding=True, truncation=True, max_length=40, return_tensors='pt').to(dev)
                    h = bert(**e).last_hidden_state.float(); mk = e['attention_mask'][..., None].float(); z = (h * mk).sum(1) / mk.sum(1)
                elif m == 'meta': z = torch.cat([r[m] for r in b])
                else:
                    x = torch.cat([r[m] for r in b]).to(dev, torch.float16); z = vit(pixel_values=x).pooler_output.float()
                acc[m].append(z.view(len(b), K + 1, -1).cpu().numpy().astype(np.float16)); st[m] += [r[m + '_st'] for r in b]
        bank[f'{split}_y'] = np.array(ys); bank[f'{split}_idx'] = sp[split]
        for m in mods:
            Z = np.concatenate(acc[m]); S = np.stack(st[m]); bank[f'{split}_{m}_clean'] = Z[:, 0]; bank[f'{split}_{m}_var'] = Z[:, 1:]
            bank[f'{split}_{m}_fam'], bank[f'{split}_{m}_sub'], bank[f'{split}_{m}_sev'] = S[..., 0], S[..., 1], S[..., 2]
        if ds == 'sun': bank[f'{split}_meta_sensor'] = np.array([items[i]['sensor'] for i in sp[split]])
        if ds == 'milk': bank[f'{split}_meta_tone'] = np.array([items[i]['tone'] for i in sp[split]])
        print(split, len(ys), round(time.time() - t0, 1), flush=True)
    os.makedirs('banks', exist_ok=True); np.savez_compressed(f'banks/{ds}{"_smoke" if smoke else ""}.npz', **bank); print('saved', ds, round(time.time() - t0, 1))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ds'); ap.add_argument('--root'); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system'); run(a.ds, a.root, a.smoke)
