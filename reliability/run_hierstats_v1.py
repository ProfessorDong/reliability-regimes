"""Hierarchical / target-level statistics for the DUAL-ENCODER search against Graph GA.

This script differences wmga_results.json, the dual-encoder (latent-surrogate) search, against
graphga_results.json. It is the ablation reported as the "Dual encoder" row of Supplementary
Table S13, not the headline fingerprint-surrogate result, which is computed from
methods_v2_results.json instead. The docstring used to say "ST-GA vs Graph GA", which names the
fingerprint comparison and points a reader at the wrong file.

The pooled paired t-test over 375 nested runs treats runs as independent
(pseudoreplication). Here we respect the hierarchy: runs are nested within
target-by-k cells within five targets. We report, for the paired difference
d = top10(dual-encoder search) - top10(Graph GA):
  * per-cell mean, paired t-test p, effect size dz, fraction of seeds > 0
  * per-target mean (over 3 k x 25 seeds)
  * TARGET-LEVEL test: one-sample t on the 5 per-target means (df=4) -- the honest
    "generalize to a new target" analysis (effective n = 5 targets)
  * equal-cell-weighted mean (mean of 15 cell means)
  * leave-one-target-out means
  * (if statsmodels available) a random-intercept mixed model d ~ 1 + (1|target/k)
Writes hierstats_analysis.json.

    python -m reliability.run_hierstats_v1
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'outputs', 'frozen', 'hierstats_analysis.json')
TARGETS = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']


def load_paired():
    W = json.load(open(os.path.join(BASE, 'outputs/frozen/wmga_results.json')))['results']
    G = json.load(open(os.path.join(BASE, 'outputs/frozen/graphga_results.json')))['results']
    gmap = {(c['target'], c['k']): c['per_seed'] for c in G}
    cells = {}
    for c in W:
        key = (c['target'], c['k'])
        gp = gmap[key]
        n = min(len(c['per_seed']), len(gp))
        d = np.array([c['per_seed'][i]['wm']['top10'] - gp[i]['top10'] for i in range(n)])
        cells[key] = d
    return cells


def main():
    cells = load_paired()
    out = {'per_cell': {}, 'per_target': {}}
    for (t, k), d in sorted(cells.items()):
        dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float('nan')
        tt, p = stats.ttest_rel(d, np.zeros_like(d)) if len(d) > 1 else (np.nan, np.nan)
        out['per_cell'][f'{t}_k{k}'] = dict(n=len(d), mean=float(d.mean()),
                                            p=float(p), dz=dz, frac_pos=float(np.mean(d > 0)))
    # per-target mean over all its runs
    tgt_means = {}
    for t in TARGETS:
        allt = np.concatenate([d for (tt, k), d in cells.items() if tt == t])
        tgt_means[t] = float(allt.mean())
        out['per_target'][t] = dict(n=len(allt), mean=tgt_means[t],
                                    frac_pos=float(np.mean(allt > 0)))
    tm = np.array([tgt_means[t] for t in TARGETS])
    # TARGET-LEVEL one-sample t (df=4): the honest generalization test
    t_stat, t_p = stats.ttest_1samp(tm, 0.0)
    se = tm.std(ddof=1) / np.sqrt(len(tm))
    ci = stats.t.interval(0.95, len(tm) - 1, loc=tm.mean(), scale=se)
    out['target_level'] = dict(
        target_means=tgt_means, mean=float(tm.mean()), sd_across_targets=float(tm.std(ddof=1)),
        se=float(se), ci95=[float(ci[0]), float(ci[1])], t=float(t_stat), p=float(t_p), n_targets=len(tm))
    # equal-cell-weighted mean (15 cells)
    cell_means = np.array([d.mean() for d in cells.values()])
    out['equal_cell_weighted_mean'] = float(cell_means.mean())
    # leave-one-target-out (mean of remaining target means)
    out['loto'] = {t: float(np.mean([tgt_means[u] for u in TARGETS if u != t])) for t in TARGETS}
    # naive pooled (for contrast)
    alld = np.concatenate(list(cells.values()))
    tt, pp = stats.ttest_rel(alld, np.zeros_like(alld))
    out['naive_pooled'] = dict(n=len(alld), mean=float(alld.mean()), p=float(pp))

    # optional: random-intercept mixed model
    try:
        import pandas as pd, statsmodels.formula.api as smf
        recs = []
        for (t, k), d in cells.items():
            for i, v in enumerate(d):
                recs.append(dict(d=v, target=t, cell=f'{t}_{k}'))
        df = pd.DataFrame(recs)
        m = smf.mixedlm('d ~ 1', df, groups=df['target'], re_formula='1').fit(reml=True)
        out['mixed_model'] = dict(intercept=float(m.params['Intercept']),
                                  se=float(m.bse['Intercept']),
                                  p=float(m.pvalues['Intercept']),
                                  group_var=float(m.cov_re.iloc[0, 0]))
    except Exception as e:
        out['mixed_model'] = f'unavailable: {type(e).__name__}: {e}'

    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    tl = out['target_level']
    print(f"naive pooled: mean={out['naive_pooled']['mean']:+.4f}, p={out['naive_pooled']['p']:.1e} (n=375, PSEUDOREP)")
    print(f"TARGET-LEVEL (n=5): mean={tl['mean']:+.4f}  SE={tl['se']:.4f}  "
          f"95% CI [{tl['ci95'][0]:+.4f},{tl['ci95'][1]:+.4f}]  t={tl['t']:.2f}  p={tl['p']:.3f}")
    print(f"  per-target means: " + ' '.join(f"{t}={tgt_means[t]:+.3f}" for t in TARGETS))
    print(f"  equal-cell-weighted mean: {out['equal_cell_weighted_mean']:+.4f}")
    if isinstance(out['mixed_model'], dict):
        mm = out['mixed_model']
        print(f"  mixed model (random target intercept): {mm['intercept']:+.4f} (SE {mm['se']:.4f}, p={mm['p']:.3f})")
    print('wrote', OUT)


if __name__ == '__main__':
    main()
