"""Re-draw Fig. 2 (results) of the paper from the stored per-seed rows.

Usage:  python scripts/make_figures.py   ->  figures/fig_p1_plugin.{pdf,png}
"""
import os, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(HERE, '..', 'results', 'tables'); OUT = os.path.join(HERE, '..', 'figures')
os.makedirs(OUT, exist_ok=True)
mpl.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
UNSEEN = {'C3_unseen', 'B3_unseen'}; BLUE = '#1f5fa8'
BBC = {'QMF': '#4c72b0', 'PDF': '#dd8452', 'MNL': '#55a868', 'BML': '#8172b3'}
pc = pd.read_csv(os.path.join(T, 'plugin_rows_230_239.csv.gz')); t = pc[pc.proto == 'trainonly'].copy(); t['delta'] = t.gated - t.native
PC = pd.read_csv(os.path.join(T, 'plugin_confirm_230_239.csv')).set_index(['backbone', 'condition'])
gains = pd.read_csv(os.path.join(T, 'trust_gap_190_209.csv'), index_col=0).iloc[:, 0]
gains.index = [('bank:' + i if i in ('crema', 'food', 'milk', 'sun') else i) for i in gains.index]
g = t[t.cond.isin(UNSEEN)].groupby('ds').delta.mean(); gg = pd.concat([g, gains], axis=1, join='inner'); gg.columns = ['delta', 'gap']
rho = spearmanr(gg.delta, gg.gap)[0]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.15), gridspec_kw=dict(width_ratios=[1, 1, 1.15], wspace=0.45))
ax = axs[0]; bbs = ['QMF', 'PDF', 'MNL', 'BML']; x = np.arange(4)
for nm, mk, off in [('clean', 'o', -0.15), ('unseen', 's', 0.15)]:
    d = PC.loc[[(b, nm) for b in bbs]]
    ax.errorbar(x + off, d['diff'], yerr=[d['diff'] - d['lo'], d['hi'] - d['diff']], fmt=mk, ms=4.5, capsize=0, lw=1,
                color='#777' if nm == 'clean' else BLUE, mfc='white' if nm == 'clean' else BLUE, label='clean' if nm == 'clean' else 'unseen damage')
ax.axhline(0, color='k', lw=0.6); ax.set_xticks(x); ax.set_xticklabels(bbs); ax.set_ylabel('gate − native (points)')
ax.set_title('Gain without retraining', loc='left'); ax.legend(frameon=False, loc='center left', bbox_to_anchor=(0, 0.55)); ax.margins(x=0.1); ax.set_ylim(-1.2, 3.9)
ax = axs[1]; b3 = ['QMF', 'PDF', 'BML']; a = t[t.cond.isin(UNSEEN) & t.backbone.isin(b3)]
for i, b in enumerate(b3):
    q = a[a.backbone == b].groupby('ds')[['auroc_native', 'auroc_beta']].mean()
    for _, r in q.iterrows(): ax.plot([i - 0.18, i + 0.18], [r.auroc_native, r.auroc_beta], color='#bbb', lw=0.5, zorder=1)
    ax.scatter(np.full(len(q), i - 0.18), q.auroc_native, s=10, color=BBC[b], facecolors='white', zorder=2)
    ax.scatter(np.full(len(q), i + 0.18), q.auroc_beta, s=10, color=BBC[b], zorder=2)
ax.axhline(0.5, color='#b0612b', lw=0.7, ls=':'); ax.set_xticks(range(3)); ax.set_xticklabels(b3); ax.set_ylabel('damage-detection AUROC')
ax.set_title('Native weights vs validity β', loc='left'); ax.set_ylim(0, 1.05)
ax.text(0.98, 0.03, 'open: native, filled: β', transform=ax.transAxes, ha='right', color='#444')
ax = axs[2]; isb = np.array([i.startswith('bank:') for i in gg.index])
ax.scatter(gg.gap[~isb], gg.delta[~isb], s=16, color=BLUE, label='standard set')
ax.scatter(gg.gap[isb], gg.delta[isb], s=16, marker='s', color='#6a3d9a', label='bank')
ax.axhline(0, color='k', lw=0.6); ax.set_xlabel('trust gap, independent study (pt)'); ax.set_ylabel('plug-in gain (points)')
ax.set_title('Gain follows the trust gap', loc='left'); ax.text(0.04, 0.9, f'Spearman ρ = {rho:.2f}', transform=ax.transAxes)
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 0.86), handletextpad=0.3); ax.margins(0.06)
for a_, l in zip(axs, 'abc'): a_.text(-0.18, 1.08, l, transform=a_.transAxes, fontweight='bold', fontsize=9)
fig.subplots_adjust(left=0.075, right=0.975, bottom=0.21, top=0.87)
fig.savefig(os.path.join(OUT, 'fig_p1_plugin.pdf')); fig.savefig(os.path.join(OUT, 'fig_p1_plugin.png'), dpi=300)
print('saved', OUT, 'rho =', round(rho, 2))
