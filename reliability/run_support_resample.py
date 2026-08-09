#!/usr/bin/env python3
"""Does the seed-novelty conclusion survive a different choice of starting compounds?

Seed novelty is measured against ten active compounds drawn from the fold's training set:

    nu(x) = 1 - max_{z in M_0} Tanimoto(x, z)

and the whole in-domain gap g(x) = nu(x) - d_train(x) inherits that draw. run_reliability_v2.py
uses one draw per fold, seeded at 0. The paper's claim that seed novelty carries little
information about error once nearest-training distance is known therefore rests, as written, on
one arbitrary set of ten leads, and a different ten could in principle change it.

This script resamples the support set. Nothing about the model depends on the draw: the folds,
the forest, the out-of-fold predictions, the errors and d_train are all identical across draws
and are computed once per target. Only nu and g are recomputed, which is a Tanimoto lookup
against k compounds, so many draws are cheap.

For each target, each support size k in {5, 10, 20} and each of R draws it records the Spearman
correlation of nu with absolute error, the partial correlation given d_train, and whether the
high-novelty nearer-training group beats the farther group on RMSE. Reports the distribution
across draws, so the conclusion can be stated over plausible starting sets rather than over one.

Writes outputs/frozen/support_resample.json.

    python -m reliability.run_support_resample [--draws 100]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability.oracle import TARGETS, featurize  # noqa: E402
from reliability.run_applicability_v1 import max_tanimoto_to_set, partial_spearman  # noqa: E402
from reliability.run_reliability_v2 import load_deduplicated, RF_KW  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'outputs', 'frozen', 'support_resample.json')
KS = (5, 10, 20)


def pct(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    return dict(median=float(np.median(v)),
                lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)),
                mean=float(v.mean()), n=int(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=100)
    args = ap.parse_args()
    R = args.draws

    res = {'protocol': 'support set resampled; folds, forest, predictions, errors and d_train '
                       'are identical across draws and computed once per target',
           'n_draws': R, 'k_values': list(KS)}
    for tgt in sorted(TARGETS):
        smiles, y, thr, _ = load_deduplicated(tgt)
        X = featurize(smiles)
        y = np.asarray(y, float)
        n = len(y)
        pred = np.full(n, np.nan)
        dtr = np.full(n, np.nan)
        folds = []
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            rf = RandomForestRegressor(**RF_KW).fit(X[tr], y[tr])
            pred[te] = rf.predict(X[te])
            dtr[te] = 1.0 - max_tanimoto_to_set(X[te], X[tr])
            folds.append((tr, te, tr[y[tr] >= thr]))
        err = np.abs(pred - y)

        res[tgt] = {}
        for k in KS:
            rho_nov, rho_par, near_wins, gap_pos = [], [], [], []
            for rep in range(R):
                nov = np.full(n, np.nan)
                for fi, (tr, te, tr_act) in enumerate(folds):
                    if len(tr_act) == 0:
                        continue
                    rng = np.random.default_rng(1000 * rep + fi)
                    take = min(k, len(tr_act))
                    sup = tr_act[rng.choice(len(tr_act), take, replace=False)]
                    nov[te] = 1.0 - max_tanimoto_to_set(X[te], X[sup])
                ok = np.isfinite(nov) & np.isfinite(err)
                rho_nov.append(stats.spearmanr(nov[ok], err[ok])[0])
                rho_par.append(partial_spearman(err[ok], nov[ok], dtr[ok])[0])
                gap_pos.append(float(np.mean((nov[ok] - dtr[ok]) >= 0)))
                # the in-domain comparison, recomputed on this draw
                hi = nov >= np.nanquantile(nov, 2 / 3)
                med = np.nanmedian(dtr[hi])
                a, b = hi & (dtr <= med), hi & (dtr > med)
                if a.sum() > 1 and b.sum() > 1:
                    near_wins.append(float(np.sqrt(np.mean(err[a] ** 2))
                                           < np.sqrt(np.mean(err[b] ** 2))))
            res[tgt][f'k{k}'] = dict(
                spearman_nov_err=pct(rho_nov),
                partial_nov_err_given_dtr=pct(rho_par),
                gap_frac_positive=pct(gap_pos),
                near_beats_far_frac=float(np.mean(near_wins)) if near_wins else None)
            p = res[tgt][f'k{k}']
            print('  %-6s k=%-3d rho(nu,err) %6.3f [%6.3f, %6.3f]   partial %6.3f [%6.3f, %6.3f]'
                  '   near beats far %s'
                  % (tgt, k, p['spearman_nov_err']['median'], p['spearman_nov_err']['lo'],
                     p['spearman_nov_err']['hi'], p['partial_nov_err_given_dtr']['median'],
                     p['partial_nov_err_given_dtr']['lo'], p['partial_nov_err_given_dtr']['hi'],
                     ('%.0f%%' % (100 * p['near_beats_far_frac'])) if p['near_beats_far_frac'] is not None else '-'))

    # The headline the manuscript needs: across every target, k and draw, how large does the
    # partial correlation ever get? If the upper tail stays small the conclusion is not an
    # artifact of one support set.
    allhi = [res[t][f'k{k}']['partial_nov_err_given_dtr']['hi']
             for t in sorted(TARGETS) for k in KS
             if res[t][f'k{k}']['partial_nov_err_given_dtr']]
    allmed = [res[t][f'k{k}']['partial_nov_err_given_dtr']['median']
              for t in sorted(TARGETS) for k in KS
              if res[t][f'k{k}']['partial_nov_err_given_dtr']]
    res['summary'] = dict(max_upper_partial=float(np.max(allhi)),
                          max_abs_median_partial=float(np.max(np.abs(allmed))),
                          n_cells=len(allmed))
    print('\n  across all %d target-by-k cells: largest 97.5%% partial = %.3f, '
          'largest |median partial| = %.3f'
          % (res['summary']['n_cells'], res['summary']['max_upper_partial'],
             res['summary']['max_abs_median_partial']))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
