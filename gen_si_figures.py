"""Generate the Supplementary figures from the frozen JSONs.

Each Supplementary figure disaggregates a pooled main-text result to the level of the
individual target; none of them repeats a main-text image. Main Figure 1 (gen_fig1.py)
carries the pooled risk-coverage curve, the pooled coverage under temporal shift and the
mean acquisition enrichment, so the Supplementary figures show, respectively, the
calibration of the intervals at every nominal level and conditional on the disagreement
score (S1), the temporal degradation per target on the two quantities Figure 1c does not
show (S2), and the acquisition outcome per target (S3).

All values are read from outputs/cwm_v1/*.json. The reliability source is the
structure-level analysis (reliability_v2_analysis.json), the same file that make_numbers.py
reads; the superseded raw-record file is not used anywhere.

    python gen_si_figures.py   ->  figS1_calibration.png, figS2_temporal.png, figS3_acquisition.png
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
BLUE, ORANGE, GREEN, PINK, VERM, SKY = ('#0072B2', '#E69F00', '#009E73',
                                        '#CC79A7', '#D55E00', '#56B4E9')
COL = [BLUE, ORANGE, GREEN, PINK, SKY]
DARK = '#333333'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
})

con = L('conformal_analysis.json')
tmp = L('temporal_analysis.json')
_po = L('poolopt_analysis.json')
pool, NSEED = _po['summary'], _po['config']['seeds']

# =====================================================================  S1 calibration
# (a) empirical coverage against nominal, per target, for the disagreement-normalised
#     intervals. (b) coverage of the same intervals within the low- and high-disagreement
#     halves, which is what shows the guarantee is marginal and not conditional.
ALPHA = [('alpha0.2', 0.80), ('alpha0.1', 0.90), ('alpha0.05', 0.95)]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

ax = axes[0]
nom = [n for _, n in ALPHA]
ax.plot([0.78, 0.97], [0.78, 0.97], ls=':', color=DARK, lw=1.0, zorder=1)
ax.text(0.955, 0.945, 'nominal', fontsize=7, color=DARK, rotation=38, ha='right')
for i, t in enumerate(T):
    ax.plot(nom, [con[t][a]['adaptive_coverage'] for a, _ in ALPHA], '-o',
            color=COL[i], ms=4, lw=1.3, label=LAB[t], alpha=0.9)
ax.plot(nom, [con['pooled'][a]['adaptive_coverage'] for a, _ in ALPHA], '--',
        color=DARK, lw=2.0, label='Pooled', zorder=5)
ax.set_xlabel('Nominal coverage'); ax.set_ylabel('Empirical coverage')
ax.set_xticks(nom); ax.set_xlim(0.775, 0.975); ax.grid(alpha=0.22, lw=0.6)
ax.legend(fontsize=7, frameon=False, ncol=2, loc='upper left', handlelength=1.5,
          columnspacing=0.9, borderpad=0.2)
ax.set_title('a  Marginal coverage, all nominal levels', loc='left', pad=6)

ax = axes[1]
x = np.arange(len(T)); w = 0.36
lo = [con[t]['alpha0.1']['adaptive_coverage_low_sigma'] for t in T]
hi = [con[t]['alpha0.1']['adaptive_coverage_high_sigma'] for t in T]
ax.bar(x - w / 2, lo, w, color=SKY, label='low $\\sigma_T$ half')
ax.bar(x + w / 2, hi, w, color=VERM, label='high $\\sigma_T$ half')
ax.axhline(0.90, color=DARK, ls=':', lw=1.1, label='nominal 0.90')
ax.set_xticks(x); ax.set_xticklabels([LAB[t] for t in T])
ax.set_ylabel('Empirical coverage'); ax.set_ylim(0.6, 1.15)
ax.grid(alpha=0.22, axis='y', lw=0.6)
ax.legend(fontsize=7, frameon=False, loc='upper left', ncol=3, borderpad=0.2,
          handlelength=1.4, columnspacing=1.0)
ax.set_title('b  Coverage conditional on disagreement', loc='left', pad=6)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figS1_calibration.png'), dpi=400)
plt.close(fig)
print('S1  pooled adaptive coverage:',
      [f"{con['pooled'][a]['adaptive_coverage']:.3f}" for a, _ in ALPHA])

# =====================================================================  S2 temporal
# Main Figure 1c shows interval coverage only. Here: the error and the error ranking,
# each against its size-matched random control, plus the per-target risk-coverage curve
# under the shifted protocol.
tt = [t for t in T if t in tmp]
fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
x = np.arange(len(tt)); w = 0.36

ax = axes[0]
ax.bar(x - w / 2, [tmp[t]['control_random_same_size']['rmse'] for t in tt], w,
       color=BLUE, label='size-matched random split')
ax.bar(x + w / 2, [tmp[t]['rmse_test'] for t in tt], w, color=VERM, label='temporal split')
ax.set_xticks(x); ax.set_xticklabels([LAB[t] for t in tt])
ax.set_ylabel('RMSE (pIC$_{50}$)'); ax.grid(alpha=0.22, axis='y', lw=0.6)
ax.set_ylim(0, max(tmp[t]['rmse_test'] for t in tt) * 1.32)
ax.legend(fontsize=7, frameon=False, loc='upper left', borderpad=0.2)
ax.set_title('a  Error', loc='left', pad=6)

ax = axes[1]
ax.bar(x - w / 2, [tmp[t]['control_random_same_size']['spearman_sigma_err'] for t in tt], w,
       color=BLUE)
ax.bar(x + w / 2, [tmp[t]['spearman_sigma_err'] for t in tt], w, color=VERM)
ax.axhline(0, color=DARK, lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([LAB[t] for t in tt])
ax.set_ylabel('Spearman $\\rho$($\\sigma_T$, absolute error)')
ax.grid(alpha=0.22, axis='y', lw=0.6)
ax.set_title('b  Error ranking', loc='left', pad=6)

ax = axes[2]
cov = [0.2, 0.4, 0.6, 0.8, 1.0]
for i, t in enumerate(tt):
    rc = tmp[t]['risk_coverage_rmse']
    ax.plot(cov, [rc[f'{c:.1f}'] for c in cov], '-o', ms=3.5, lw=1.3,
            color=COL[T.index(t)], label=LAB[t], alpha=0.9)
rc = tmp['pooled']['risk_coverage_rmse']
ax.plot(cov, [rc[f'{c:.1f}'] for c in cov], '--', color=DARK, lw=2.0, label='Pooled')
ax.set_xlabel('Fraction retained, most confident first')
ax.set_ylabel('RMSE (pIC$_{50}$)'); ax.set_xlim(0.15, 1.05)
ax.grid(alpha=0.22, lw=0.6)
ax.legend(fontsize=7, frameon=False, ncol=2, loc='upper left', handlelength=1.5,
          columnspacing=0.9, borderpad=0.2)
ax.set_title('c  Risk\u2013coverage after the shift', loc='left', pad=6)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figS2_temporal.png'), dpi=400)
plt.close(fig)
print('S2  temporal rho per target:', {t: round(tmp[t]['spearman_sigma_err'], 3) for t in tt})

# =====================================================================  S3 acquisition
# Main Figure 1d shows the mean over targets. Here: every target separately, so that the
# ordering of the strategies can be read per target rather than only on average.
STRAT = [('random', 'Random', '#9A9A9A'), ('greedy', 'Predicted mean', BLUE),
         ('ucb', 'Optimistic $\\mu+\\sigma$', SKY), ('lcb', 'Cautious $\\mu-\\sigma$', ORANGE),
         ('conformal', 'Conformal bound', VERM)]
pt = [t for t in T if t in pool]
fig, axes = plt.subplots(1, len(pt), figsize=(9.6, 3.1))
for j, t in enumerate(pt):
    ax = axes[j]
    for i, (k, lab, c) in enumerate(STRAT):
        h = pool[t][k]['hits']
        se = pool[t][k]['hits_sd'] / np.sqrt(NSEED)
        ax.bar(i, h, 0.72, color=c, yerr=se, capsize=2,
               error_kw=dict(lw=0.9, ecolor=DARK), label=lab if j == 0 else None)
    ax.set_xticks(range(len(STRAT)))
    ax.set_xticklabels([s[1] for s in STRAT], rotation=55, ha='right', fontsize=7)
    ax.set_title(LAB[t], loc='left', pad=5)
    ax.grid(alpha=0.22, axis='y', lw=0.6)
    ax.set_ylim(0, max(pool[t][k]['hits'] for k, _, _ in STRAT) * 1.30)
    if j == 0:
        ax.set_ylabel('Top-percentile compounds found', fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'figS3_acquisition.png'), dpi=400)
plt.close(fig)
print('S3  hits per target:',
      {t: {k: pool[t][k]['hits'] for k, _, _ in STRAT} for t in pt})

for f in ['figS1_calibration.png', 'figS2_temporal.png', 'figS3_acquisition.png']:
    print(f'wrote {f}  {os.path.getsize(os.path.join(OUT, f))} B')
