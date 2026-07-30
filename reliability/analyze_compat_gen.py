"""Does C_nn govern generation transfer? (Section V-D analysis.)

Regresses the warm-start generation gain on the source-target compatibility C_nn
(reused from the prediction paper), to test whether the same law that governs
few-shot PREDICTION transfer also governs few-shot GENERATION transfer.

Writes outputs/frozen/compat_gen_analysis.json.

    python -m reliability.analyze_compat_gen
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, 'outputs', 'frozen', 'compat_gen_results.json')
OUT = os.path.join(BASE, 'outputs', 'frozen', 'compat_gen_analysis.json')


def _ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = len(y) - X1.shape[1]
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = [2 * stats.t.sf(abs(t), dof) for t in tvals]
    return beta, pvals


def main():
    d = json.load(open(SRC))['results']
    cnn = np.array([r['C_nn'] for r in d])
    gain = np.array([r['gain_mean'] for r in d])
    logn = np.log10(np.array([r['n_target'] for r in d]))

    pr = stats.pearsonr(cnn, gain); sr = stats.spearmanr(cnn, gain)
    beta, pv = _ols(np.column_stack([cnn, logn]), gain)

    # per-seed pooled (more power): each pair contributes its per-seed gains at its C_nn
    pcnn, pgain = [], []
    for r in d:
        for g in r['per_seed_gain']:
            pcnn.append(r['C_nn']); pgain.append(g)
    pcnn = np.array(pcnn); pgain = np.array(pgain)
    pr_pool = stats.pearsonr(pcnn, pgain)

    res = dict(
        n_pairs=len(d), n_seedpoints=len(pgain),
        pair_pearson_r=float(pr.statistic), pair_pearson_p=float(pr.pvalue),
        pair_spearman=float(sr.correlation), pair_spearman_p=float(sr.pvalue),
        ols_intercept=float(beta[0]), ols_C_nn=float(beta[1]), ols_C_nn_p=float(pv[1]),
        ols_logn=float(beta[2]), ols_logn_p=float(pv[2]),
        pooled_pearson_r=float(pr_pool.statistic), pooled_pearson_p=float(pr_pool.pvalue),
        mean_gain=float(gain.mean()),
        gain_highCnn=float(gain[cnn > 0.5].mean()) if (cnn > 0.5).any() else float('nan'),
        gain_lowCnn=float(gain[cnn < 0.35].mean()) if (cnn < 0.35).any() else float('nan'))
    with open(OUT, 'w') as f:
        json.dump({'summary': res, 'rows': d}, f, indent=2)

    print(f"n_pairs={res['n_pairs']}  mean gain={res['mean_gain']:+.4f}")
    print(f"  gain | C_nn>0.5 = {res['gain_highCnn']:+.4f} ; C_nn<0.35 = {res['gain_lowCnn']:+.4f}")
    print(f"  pair-level Pearson(gain,C_nn) = {res['pair_pearson_r']:+.3f} (p={res['pair_pearson_p']:.3g}); "
          f"Spearman={res['pair_spearman']:+.3f} (p={res['pair_spearman_p']:.3g})")
    print(f"  OLS gain ~ C_nn + log10(n_target): C_nn coef={res['ols_C_nn']:+.3f} "
          f"(p={res['ols_C_nn_p']:.3g}), logn coef={res['ols_logn']:+.3f} (p={res['ols_logn_p']:.3g})")
    print(f"  pooled per-seed Pearson = {res['pooled_pearson_r']:+.3f} (p={res['pooled_pearson_p']:.3g}, "
          f"n={res['n_seedpoints']})")
    verdict = (res['pair_pearson_p'] < 0.05 and res['pair_pearson_r'] > 0)
    print(f"\nC_nn governs generation transfer: {'YES (significant positive)' if verdict else 'NO/WEAK (not significant)'}")


if __name__ == '__main__':
    main()
