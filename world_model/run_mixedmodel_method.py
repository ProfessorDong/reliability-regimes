"""Random-intercept model for the primary method gain (ECFP-surrogate triage vs Graph GA).

The primary test is a one-sample t-test on the five per-target mean differences, which
treats targets as the units of generalisation. The Methods also state that a random-intercept
model gives the same point estimate; this script is what that statement rests on, so the
coefficient, its standard error, the interval and the variance components are reported
rather than asserted.

Model, on the seed-level paired difference d = top10(ST-GA-ECFP) - top10(Graph GA):

    d_{ijk} = beta_0 + u_i + w_{ij} + eps_{ijk}

with u_i a random intercept for target i and w_{ij} a random intercept for the
target-by-support-size cell j nested within it. Runs are paired by seed, so d is already
a within-seed contrast and no fixed effects are needed.

Writes mixedmodel_method.json.

    python -m world_model.run_mixedmodel_method
"""
from __future__ import annotations
import json, os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sps

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, 'outputs', 'cwm_v1', 'methods_v2_results.json')
OUT = os.path.join(BASE, 'outputs', 'cwm_v1', 'mixedmodel_method.json')
T = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']


def main():
    res = json.load(open(SRC))['results']
    rows = []
    for r in res:
        for i, c in enumerate(r['per_seed']):
            rows.append(dict(target=r['target'], k=int(r['k']), seed=i,
                             cell=f"{r['target']}_k{r['k']}",
                             d=c['stga_ecfp'] - c['graphga']))
    df = pd.DataFrame(rows)

    # target-level primary test, repeated here so the two appear side by side
    tmeans = df.groupby('target')['d'].mean().reindex(T)
    t, p = sps.ttest_1samp(tmeans.values, 0.0)
    se = tmeans.std(ddof=1) / np.sqrt(len(tmeans))
    ci = sps.t.interval(0.95, len(tmeans) - 1, loc=tmeans.mean(), scale=se)

    # random intercept for target, with cell nested inside it
    m = smf.mixedlm('d ~ 1', df, groups=df['target'],
                    re_formula='1', vc_formula={'cell': '0 + C(cell)'}).fit(reml=True)
    b = float(m.params['Intercept'])
    bse = float(m.bse['Intercept'])
    out = dict(
        n_runs=int(len(df)), n_targets=int(df.target.nunique()),
        n_cells=int(df.cell.nunique()),
        target_level=dict(mean=float(tmeans.mean()), se=float(se), t=float(t), p=float(p),
                          ci95=[float(ci[0]), float(ci[1])],
                          per_target={k: float(v) for k, v in tmeans.items()}),
        mixed_model=dict(
            formula='d ~ 1 + (1|target) + (1|target:cell), REML',
            intercept=b, se=bse, z=b / bse,
            p=float(2 * sps.norm.sf(abs(b / bse))),
            ci95=[b - 1.96 * bse, b + 1.96 * bse],
            group_var_target=float(m.cov_re.iloc[0, 0]),
            vcomp_cell=[float(v) for v in np.atleast_1d(m.vcomp)],
            scale_residual=float(m.scale)),
    )
    json.dump(out, open(OUT, 'w'), indent=2)
    tl, mm = out['target_level'], out['mixed_model']
    print(f"target-level : {tl['mean']:+.4f}  se {tl['se']:.4f}  "
          f"t={tl['t']:.2f}  P={tl['p']:.4f}  95% CI [{tl['ci95'][0]:+.4f}, {tl['ci95'][1]:+.4f}]")
    print(f"mixed model  : {mm['intercept']:+.4f}  se {mm['se']:.4f}  "
          f"P={mm['p']:.2e}  95% CI [{mm['ci95'][0]:+.4f}, {mm['ci95'][1]:+.4f}]")
    print(f"  var(target)={mm['group_var_target']:.5f}  vcomp(cell)={mm['vcomp_cell']}  "
          f"residual={mm['scale_residual']:.5f}")
    print('wrote', OUT)


if __name__ == '__main__':
    main()
