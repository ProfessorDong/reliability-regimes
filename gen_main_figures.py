"""Generate main-text Figures 2 and 3 from the frozen JSONs.

Figure 1 is built by gen_fig1.py. Figure 2 is the novelty sweep and Figure 3 is the
per-cell method gain. Styling matches gen_fig1.py so the three read as one set.

    python gen_main_figures.py   ->  fig2_frontier.png, fig3_forest.png
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy import stats as sps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _data_dir():
    """Locate the frozen outputs, whether this file sits in the repository or the workspace."""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = (here, os.path.join(os.path.dirname(here), 'drug_discovery', 'theranostics_current'))
    for root in roots:
        out = os.path.join(root, 'outputs')
        if not os.path.isdir(out):
            continue
        for sub in sorted(os.listdir(out)):
            cand = os.path.join(out, sub)
            if os.path.isfile(os.path.join(cand, 'reliability_v2_analysis.json')):
                return cand
    raise SystemExit('frozen outputs not found beside this file or in the workspace')


CW = _data_dir()
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
rows = [dict(r, target=t) for t, rs in fr.items() for r in rs]
nv = np.array([r['novelty'] for r in rows])
# The 1,200 runs are 300 search trajectories, each contributing its four novelty settings, so
# a standard error over runs treats repeated measures as independent. Intervals come instead
# from a cluster bootstrap that resamples whole trajectories, keeping their four settings
# together, and resamples targets first so between-target variation is carried too.
# One support set per target and seed, reused across optimizers, penalties and novelty
# weights, so a block is a target-seed pair carrying all 16 conditions.
BLOCK = np.array([f"{r['target']}|{r['seed']}" for r in rows])
TGT = np.array([r['target'] for r in rows])
_uT = sorted(set(TGT))
_blocks_by_t = {t: sorted(set(BLOCK[TGT == t])) for t in _uT}
_idx_by_block = {b: np.flatnonzero(BLOCK == b) for b in set(BLOCK)}
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

def cluster_ci(y, n_boot=2000, seed=5):
    """Percentile interval per novelty bin, resampling targets then whole trajectories."""
    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, len(bc)), np.nan)
    for it in range(n_boot):
        take = []
        for t in rng.choice(_uT, len(_uT), replace=True):
            bl = _blocks_by_t[t]
            for b in rng.choice(bl, len(bl), replace=True):
                take.append(_idx_by_block[b])
        sel = np.concatenate(take)
        yi, ii = y[sel], idx[sel]
        for b in range(len(bc)):
            m = ii == b
            if m.any():
                draws[it, b] = yi[m].mean()
    lo = np.nanpercentile(draws, 2.5, axis=0)
    hi = np.nanpercentile(draws, 97.5, axis=0)
    return lo, hi


fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, (y, lab, col), tag in zip(axes, series, 'abc'):
    m = np.array([y[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(bc))])
    lo, hi = cluster_ci(y)
    ax.fill_between(bc, lo, hi, color=col, alpha=0.18, lw=0, zorder=1)
    ax.plot(bc, m, '-', color=col, lw=1.4, zorder=2)
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
# Intervals are Student t, not the normal approximation. It matters most for the
# target-level estimate, where n=5 gives t=2.78 against z=1.96, and a normal interval would
# be 29% too narrow and would disagree with the interval the manuscript quotes.
def ci95(v):
    v = np.asarray(v, float)
    return float(sps.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


labels, means, cis = [], [], []
for r in mv:
    d = np.array([c['stga_ecfp'] - c['graphga'] for c in r['per_seed']])
    labels.append(f"{LAB[r['target']]}  $k$={r['k']}")
    means.append(d.mean())
    cis.append(ci95(d))
tgt = {}
for r in mv:
    tgt.setdefault(r['target'], []).append(
        np.mean([c['stga_ecfp'] - c['graphga'] for c in r['per_seed']]))
tm = np.array([np.mean(tgt[t]) for t in T])
tl, tci = tm.mean(), ci95(tm)

fig, ax = plt.subplots(figsize=(4.4, 4.6))
y = np.arange(len(labels))[::-1]
ax.axvline(0, color=DARK, lw=0.9, ls='--')
ax.errorbar(means, y, xerr=cis, fmt='o', color=BLUE, ms=4.5, capsize=2, lw=1.0)
ax.errorbar([tl], [-1.6], xerr=[tci], fmt='D', color=VERM, ms=8, capsize=3, lw=1.2)
ax.set_yticks(list(y) + [-1.6])
ax.set_yticklabels(labels + ['Target-level ($n$=5)'], fontsize=7.5)
ax.get_yticklabels()[-1].set_fontweight('bold')
ax.set_ylim(-2.6, len(labels) - 0.4)
ax.set_xlabel('Top-10 reward gain over Graph GA')
ax.grid(alpha=0.22, axis='x', lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_forest.png'), dpi=400)
plt.close(fig)
print(f'Fig 3  target-level gain {tl:+.4f} 95% CI [{tl-tci:+.4f}, {tl+tci:+.4f}]; cells positive '
      f'{sum(m > 0 for m in means)}/{len(means)}')
