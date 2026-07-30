"""Build main Figure 1 from the frozen outputs.

Four panels carrying the whole argument:
  (a) schematic defining the three quantities the paper separates. No numeric axes:
      position stands for chemical space and only the labeled distances carry meaning.
  (b) Regime 1, in distribution: risk-coverage per target, with conformal coverage inset.
  (c) Regime 2, under temporal shift: the same two quantities beside their size-matched
      random controls, which is what shows the loss is shift and not sample size.
  (d) Acquisition against real measured activity: enrichment over random selection.

Every number is read from the frozen outputs.
    python gen_fig1.py   ->  fig1_overview.png
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch

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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig1_overview.png')
L = lambda f: json.load(open(os.path.join(CW, f)))
T = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']
LAB = {'scd1': 'SCD-1', 'fads': 'FADS', 'nk1r': 'NK1R', 'drd2': 'DRD2', 'drd3': 'DRD3'}
# Okabe-Ito, colorblind safe, no red/green contrast
BLUE, ORANGE, GREEN, PINK, VERM, SKY, YELL = ('#0072B2', '#E69F00', '#009E73',
                                              '#CC79A7', '#D55E00', '#56B4E9', '#F0E442')
GREY, DARK = '#9A9A9A', '#333333'

rel = L('reliability_v2_analysis.json')
con = L('conformal_analysis.json')
tmp = L('temporal_analysis.json')
pool = L('poolopt_analysis.json')['summary']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
})

fig = plt.figure(figsize=(7.2, 6.2))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32,
                      left=0.085, right=0.975, top=0.92, bottom=0.10)


def panel_label(ax, s, dx=-0.16, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10.5, fontweight='bold', va='top')


# ---------------------------------------------------------------- (a) schematic
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 10); ax.set_ylim(-0.2, 7.6); ax.axis('off')
panel_label(ax, 'a', dx=-0.07)
ax.set_title('Three distances, measured separately', pad=6, loc='left')

# training set cloud
ax.add_patch(Ellipse((4.0, 4.35), 7.0, 4.0, facecolor=SKY, alpha=0.16,
                     edgecolor=SKY, lw=1.1))
ax.text(0.75, 6.35, 'training set\n(all measured compounds)', fontsize=7, color='#2b6a8f',
        ha='left', va='center', linespacing=1.25)
# scattered training compounds
rng = np.random.default_rng(3)
tx = rng.normal(4.0, 1.55, 46); ty = rng.normal(4.35, 0.88, 46)
keep = ((tx - 4.0) / 3.4) ** 2 + ((ty - 4.35) / 1.9) ** 2 < 0.92
ax.scatter(tx[keep], ty[keep], s=11, color=SKY, alpha=0.85, lw=0, zorder=2)
# the k starting compounds
sx, sy = np.array([2.30, 2.75, 2.50, 2.95, 2.62]), np.array([3.95, 4.25, 4.52, 3.75, 4.20])

# The panel's job is to define the two distances, so the arrows are drawn to the actual
# nearest neighbours rather than placed by hand: a reader who measures them must find the
# definition the caption states. The starting compounds are part of the training set, which
# is what forces d_train <= nu.
TRAIN = np.vstack([np.c_[tx[keep], ty[keep]], np.c_[sx, sy]])
START = np.c_[sx, sy]


def nearest(pt, pool):
    pt = np.asarray(pt, float)
    i = int(np.argmin(np.hypot(*(pool - pt).T)))
    return pool[i], float(np.hypot(*(pool[i] - pt)))
ax.scatter(sx, sy, s=30, color=BLUE, edgecolor='white', lw=0.7, zorder=4)
ax.text(2.55, 3.25, 'starting\ncompounds', fontsize=7, color=BLUE, ha='center',
        va='top', fontweight='bold', linespacing=1.2)

# query molecule 1: novel vs seeds, still inside the training set
q1 = (6.96, 4.60)
ax.scatter(*q1, s=62, marker='D', color=GREEN, edgecolor='white', lw=0.8, zorder=5)
q1_s, q1_nu = nearest(q1, START)
q1_t, q1_dt = nearest(q1, TRAIN)
ax.annotate('', xy=q1, xytext=tuple(q1_s),
            arrowprops=dict(arrowstyle='<->', color=BLUE, lw=1.2, shrinkA=3, shrinkB=5))
ax.annotate('', xy=q1, xytext=tuple(q1_t),
            arrowprops=dict(arrowstyle='<->', color=VERM, lw=1.2, shrinkA=2, shrinkB=5))
ax.text(6.96, 3.62, 'novel, still\nsupported', fontsize=7, color=GREEN,
        fontweight='bold', ha='center', linespacing=1.2)

# query molecule 2: outside the training set
q2 = (8.90, 6.15)
ax.scatter(*q2, s=62, marker='D', color=ORANGE, edgecolor='white', lw=0.8, zorder=5)
q2_t, q2_dt = nearest(q2, TRAIN)
q2_s, q2_nu = nearest(q2, START)
ax.annotate('', xy=q2, xytext=tuple(q2_t),
            arrowprops=dict(arrowstyle='<->', color=VERM, lw=1.2, shrinkA=2, shrinkB=5))
ax.text(8.95, 7.15, 'novel and\nunsupported', fontsize=7, color='#a8560a',
        fontweight='bold', ha='center', linespacing=1.2)

# distance legend, stacked with clear spacing
ax.text(0.30, 1.62, r'$\nu$', fontsize=8.5, color=BLUE, fontweight='bold')
ax.text(1.50, 1.62, 'distance to nearest starting compound', fontsize=7, color=BLUE)
ax.text(0.30, 0.98, r'$d_{\mathrm{train}}$', fontsize=8.5, color=VERM, fontweight='bold')
ax.text(1.50, 0.98, 'distance to nearest training compound', fontsize=7, color=VERM)
ax.text(0.30, 0.34, r'$d_{\mathrm{train}}\leq\nu$ always; the gap is novelty relative to the leads alone',
        fontsize=6.6, color=DARK)
ax.text(0.30, 7.45, 'schematic', fontsize=6.2, color=GREY, style='italic', va='top')

# ---------------------------------------------------------------- (b) Regime 1
ax = fig.add_subplot(gs[0, 1])
panel_label(ax, 'b')
ax.set_title('Regime 1  in distribution', pad=6, loc='left')
cov = [0.2, 0.4, 0.6, 0.8, 1.0]
for i, t in enumerate(T):
    rc = rel[t]['risk_coverage_rmse']
    ax.plot(cov, [rc[f'{c:.1f}'] for c in cov], '-o', ms=3, lw=1.1,
            color=[BLUE, ORANGE, GREEN, PINK, SKY][i], label=LAB[t], alpha=0.85)
mi = rel['pooled']['risk_coverage_micro']
ax.plot(cov, [mi[f'{c:.1f}'] for c in cov], '--', lw=2.0, color=DARK, label='Pooled', zorder=5)
ax.set_xlabel('Fraction retained, most confident first')
ax.set_ylabel('RMSE (pIC$_{50}$)')
ax.set_xlim(0.14, 1.06); ax.grid(alpha=0.22, lw=0.6)
ax.legend(fontsize=6.3, frameon=False, ncol=2, loc='upper left', handlelength=1.5,
          columnspacing=0.9, borderpad=0.2)
pc = con['pooled']['alpha0.1']
ax.text(0.97, 0.06, f"90% intervals cover {pc['adaptive_coverage']:.3f}",
        transform=ax.transAxes, ha='right', fontsize=6.8, color=GREEN, fontweight='bold')

# ---------------------------------------------------------------- (c) Regime 2
ax = fig.add_subplot(gs[1, 0])
panel_label(ax, 'c')
ax.set_title('Regime 2  applied to later chemistry', pad=6, loc='left')
tt = [t for t in ['scd1', 'nk1r', 'drd2', 'drd3'] if t in tmp]
y = np.arange(len(tt))[::-1]
K = 'conformal_coverage_adaptive'
for i, t in enumerate(tt):
    d, c = tmp[t], tmp[t]['control_random_same_size']
    a, b = c[K], d[K]
    ax.annotate('', xy=(b, y[i]), xytext=(a, y[i]),
                arrowprops=dict(arrowstyle='-|>', color=VERM, lw=1.6, shrinkA=0, shrinkB=0))
    # 95% intervals: bootstrap over the evaluation compounds for the single temporal split,
    # t interval across replicates for the control
    for v, ci, col, dy in ((a, c.get(K + '_ci95'), BLUE, 0.17), (b, d.get(K + '_ci95'), VERM, -0.17)):
        if ci:
            ax.plot(ci, [y[i] + dy] * 2, color=col, lw=1.1, solid_capstyle='butt', zorder=4)
            for e in ci:
                ax.plot([e, e], [y[i] + dy - 0.06, y[i] + dy + 0.06], color=col, lw=1.1, zorder=4)
    ax.scatter([a], [y[i]], s=34, color=BLUE, zorder=5, edgecolor='white', lw=0.6)
    ax.scatter([b], [y[i]], s=34, color=VERM, zorder=5, edgecolor='white', lw=0.6)
ax.axvline(0.90, color=DARK, ls=':', lw=1.1)
ax.text(0.906, len(tt) - 0.62, 'nominal 0.90', fontsize=6.4, color=DARK, ha='left')
ax.set_yticks(y); ax.set_yticklabels([LAB[t] for t in tt]); ax.set_ylim(-0.75, len(tt) - 0.35)
ax.set_xlabel('Coverage of 90% prediction intervals')
ax.set_xlim(0.70, 0.96); ax.grid(alpha=0.22, axis='x', lw=0.6)
ax.scatter([], [], s=34, color=BLUE, label='size-matched random split')
ax.scatter([], [], s=34, color=VERM, label='temporal split')
ax.legend(fontsize=6.4, frameon=False, loc='upper left', handletextpad=0.3,
          borderpad=0.2, bbox_to_anchor=(-0.01, 0.30))
tp = tmp['pooled']
# the panel shows coverage, so annotate coverage. The pooled error-ranking contrast is
# not stated here: it is a target-dependent effect and a single pooled number misreads it.
_sep = sum(tmp[t][K + '_ci95'][1]
           < tmp[t]['control_random_same_size'][K + '_ci95'][0] for t in tt)
ax.text(0.035, 0.055, f"intervals separate on {_sep} of {len(tt)}; pooled "
                      f"{tmp['pooled'][K]:.3f}",
        transform=ax.transAxes, fontsize=6.8, color=VERM, fontweight='bold')

# ---------------------------------------------------------------- (d) acquisition
ax = fig.add_subplot(gs[1, 1])
panel_label(ax, 'd')
ax.set_title('Acquiring measured activities', pad=6, loc='left')
meths = ['random', 'greedy', 'ucb', 'lcb', 'conformal']
names = ['Random', 'Predicted\nmean', 'Optimistic\n$\\mu+\\sigma$',
         'Cautious\n$\\mu-\\sigma$', 'Scaled lower\nscore']
vals = [np.mean([pool[t][m]['enrichment_vs_random'] for t in T]) for m in meths]
sds = [np.std([pool[t][m]['enrichment_vs_random'] for t in T], ddof=1) / np.sqrt(5) for m in meths]
cols = [GREY, BLUE, SKY, ORANGE, VERM]
bars = ax.bar(range(5), vals, yerr=sds, capsize=2.5, color=cols, width=0.66,
              edgecolor='white', lw=0.6, error_kw=dict(lw=0.9, ecolor=DARK))
for i, (v, s) in enumerate(zip(vals, sds)):
    ax.text(i, v + s + 0.10, f'{v:.1f}×', ha='center', fontsize=7.0, fontweight='bold')
ax.axhline(1.0, color=DARK, ls=':', lw=1.0)
ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=6.6, linespacing=1.15)
ax.set_ylabel('Top-percentile compounds found\n(relative to random)')
ax.set_ylim(0, max(vals) + 1.45); ax.grid(alpha=0.22, axis='y', lw=0.6)
ax.text(0.5, 0.995, 'penalizing uncertainty finds fewer', transform=ax.transAxes,
        ha='center', va='top', fontsize=6.8, color=VERM, fontweight='bold')

# the schematic must obey the relation it is drawn to explain
_ins = lambda p: ((p[0] - 4.0) / 3.5) ** 2 + ((p[1] - 4.35) / 2.0) ** 2
assert q1_dt <= q1_nu and q2_dt <= q2_nu, 'panel a violates d_train <= nu'
assert _ins(q1) < 1.0 < _ins(q2), 'panel a: q1 must sit inside the cloud and q2 outside'

fig.savefig(OUT, dpi=400, bbox_inches='tight', facecolor='white')
print('wrote', OUT)
print(f'  (a) q1 nu={q1_nu:.2f} d_train={q1_dt:.2f} | q2 nu={q2_nu:.2f} d_train={q2_dt:.2f} '
      f'(arrows drawn to true nearest neighbours)')
print(f"  (b) pooled RC {mi['0.2']:.2f}->{mi['1.0']:.2f}; conformal {pc['adaptive_coverage']:.3f}")
print(f"  (c) coverage temporal {tp['conformal_coverage_adaptive']:.3f} vs controls "
      f"{[round(tmp[t]['control_random_same_size']['conformal_coverage_adaptive'],3) for t in tt]}")
print(f"  (d) enrichment " + ', '.join(f'{m}={v:.2f}' for m, v in zip(meths, vals)))
