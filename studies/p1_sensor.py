"""Paper 1 revision study B: natural sensor shift on SUN RGB-D (train/cal on Kinect v1/v2; test on RealSense/Xtion)."""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p1_plugin_conf as PC
import numpy as np
import mv_bank as BK

SRC = ('kv1', 'kv2')


def load_sun_sensor(test_mode):
    b = np.load(os.path.join(BK.BANKS, 'sun.npz'), allow_pickle=True)
    mods = [str(m) for m in b['mods']]; K = int(b['classes']); D = {'mods': mods, 'K': K}
    for sp in ('train', 'cal', 'test'):
        sen = b[f'{sp}_meta_sensor'].astype(str)
        keep = np.isin(sen, SRC) if (sp != 'test' or test_mode == 'kinect') else ~np.isin(sen, SRC)
        D[sp] = {'y': b[f'{sp}_y'].astype(int)[keep]}
        for m in mods:
            D[sp][m] = b[f'{sp}_{m}_clean'].astype(np.float64)[keep]
            D[sp][m + '_var'] = b[f'{sp}_{m}_var'].astype(np.float64)[keep]
            D[sp][m + '_fam'] = b[f'{sp}_{m}_fam'].astype(int)[keep]
    for m in mods:                                   # z-score on Kinect train+cal only
        ref = np.vstack([D['train'][m], D['cal'][m]]); mu = ref.mean(0); sd = ref.std(0) + 1e-6
        for sp in ('train', 'cal', 'test'):
            D[sp][m] = (D[sp][m] - mu) / sd
            if D[sp][m + '_var'].size: D[sp][m + '_var'] = (D[sp][m + '_var'] - mu) / sd
    return D


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int); ap.add_argument('--test', default='new'); ap.add_argument('--out')
    a = ap.parse_args()
    BK.load_bank = lambda ds: load_sun_sensor(a.test)
    t0 = time.time(); r = PC.main('bank:sun', a.seed); r['time']['total'] = time.time() - t0; r['test_mode'] = a.test
    json.dump(r, open(a.out, 'w'), indent=1)
    print({k: {c: (round(v['native'], 2), round(v['gated'], 2), round(v.get('beta_mean', v.get('beta_intact', 0)), 2)) for c, v in d.items()} for k, d in r['res'].items()})
