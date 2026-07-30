"""Generate main-text Figures 2 and 3 from the frozen JSONs.

Figure 1 is built by gen_fig1.py. Figure 2 is the novelty sweep and Figure 3 is the
per-cell method gain. Styling matches gen_fig1.py so the three read as one set.

    python gen_main_figures.py   ->  fig2_frontier.png, fig3_forest.png
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'drug_discovery', 'theranostics_current', 'outputs', 'cwm_v1')
OUT = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(CW, f)))
T = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']
LAB = {'scd1': 'SCD-1', 'fads': 'FADS', 'nk1r': 'NK1R', 'drd2': 'DRD2', 'drd3': 'DRD3'}
BLUE, ORANGE, GREEN, VERM = '#0072B2', '#E69F00', '#009E73', '#D55E00'
DARK = '#333333'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ------------------------------------------------------------------ Fig 2: novelty sweep
fr = L('frontier_v2_results.json')['results']
rows = [r for rs in fr.values() for r in rs]
nv = np.array([r['novelty'] for r in rows])
series = [(np.array([r['d_train'] for r in rows]), 'Distance to training set', BLUE),
          (np.array([r['sigma'] for r in rows]), 'Ensemble disagreement $\\sigma_T$', ORANGE),
          (np.array([r['potency'] for r in rows]), 'Predicted potency (pIC$_{50}$)', GREEN)]
bins = np.linspace(nv.min(), nv.max(), 9)
bc = 0.5 * (bins[:-1] + bins[1:])
idx = np.clip(np.digitize(nv, bins) - 1, 0, len(bc) - 1)

# Bin occupancy is very uneven (the lowest-novelty bin holds a handful of runs while the
# highest holds a couple of hundred), so marker area is scaled to the number of runs behind
# each point. A reader can then see which points carry weight instead of reading eight
# equally prominent dots.
nb = np.array([(idx == b).sum() for b in range(len(bc))])

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, (y, lab, col), tag in zip(axes, series, 'abc'):
    m = np.array([y[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(bc))])
    s = np.array([y[idx == b].std(ddof=1) / np.sqrt((idx == b).sum()) if (idx == b).sum() > 1
                  else 0.0 for b in range(len(bc))])
    ax.errorbar(bc, m, yerr=s, fmt='-', color=col, lw=1.4, capsize=2, zorder=2)
    ax.scatter(bc, m, s=8 + 42 * np.sqrt(nb / nb.max()), color=col, zorder=3,
               edgecolor='white', lw=0.5)
    ax.set_xlabel('Generated-set novelty')
    ax.set_ylabel(lab, fontsize=8)
    ax.grid(alpha=0.22, lw=0.6)
    ax.set_title(tag, loc='left', pad=5, fontsize=10.5, fontweight='bold')

# the disagreement score turns over at the highest novelty: mark it rather than let the
# eye read panel b as monotone
_pk = int(np.nanargmax([series[1][0][idx == b].mean() for b in range(len(bc))]))
axes[1].annotate('turns over', xy=(bc[_pk], series[1][0][idx == _pk].mean()),
                 xytext=(bc[_pk] - 0.30, series[1][0][idx == _pk].mean() + 0.02),
                 fontsize=6.5, color=VERM, fontweight='bold', ha='right',
                 arrowprops=dict(arrowstyle='->', color=VERM, lw=0.9))
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_frontier.png'), dpi=400)
plt.close(fig)
print(f'Fig 2  {len(rows)} runs, novelty {nv.min():.2f}-{nv.max():.2f}; bin n from {nb.min()} to {nb.max()}')

# ------------------------------------------------------------------ Fig 3: method gain
mv = L('methods_v2_results.json')['results']
labels, means, cis = [], [], []
for r in mv:
    d = np.array([c['stga_ecfp'] - c['graphga'] for c in r['per_seed']])
    labels.append(f"{LAB[r['target']]}  $k$={r['k']}")
    means.append(d.mean())
    cis.append(1.96 * d.std(ddof=1) / np.sqrt(len(d)))
tgt = {}
for r in mv:
    tgt.setdefault(r['target'], []).append(
        np.mean([c['stga_ecfp'] - c['graphga'] for c in r['per_seed']]))
tm = np.array([np.mean(tgt[t]) for t in T])
tl, tci = tm.mean(), 1.96 * tm.std(ddof=1) / np.sqrt(len(tm))

fig, ax = plt.subplots(figsize=(4.4, 4.6))
y = np.arange(len(labels))[::-1]
ax.axvline(0, color=DARK, lw=0.9, ls='--')
ax.errorbar(means, y, xerr=cis, fmt='o', color=BLUE, ms=4.5, capsize=2, lw=1.0)
ax.errorbar([tl], [-1.6], xerr=[tci], fmt='D', color=VERM, ms=8, capsize=3, lw=1.2)
ax.set_yticks(list(y) + [-1.6])
ax.set_yticklabels(labels + ['Target-level ($n$=5)'], fontsize=7.5)
ax.get_yticklabels()[-1].set_fontweight('bold')
ax.set_ylim(-2.6, len(labels) - 0.4)
ax.set_xlabel('Top-molecule reward gain over Graph GA')
ax.grid(alpha=0.22, axis='x', lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_forest.png'), dpi=400)
plt.close(fig)
print(f'Fig 3  target-level gain {tl:+.4f} +/- {tci:.4f}; cells positive '
      f'{sum(m > 0 for m in means)}/{len(means)}')
