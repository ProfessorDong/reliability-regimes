"""Uncertainty calibration check (reviewer point: is sigma_T calibrated?).

For each target, run 5-fold CV with the deployed oracle configuration, collect
out-of-fold predictions, the per-tree ensemble std sigma_T, and the true pIC50.
Report whether sigma_T tracks the actual absolute error: Spearman(sigma_T, |err|),
and RMSE within each sigma_T quintile (should rise monotonically if sigma_T is a
usable reliability signal). Writes calibration_analysis.json.

    python -m reliability.analyze_calibration
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from reliability.oracle import TARGETS, load_target_data, featurize

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'outputs', 'frozen', 'calibration_analysis.json')
RF_KW = dict(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)


def main():
    out = {}
    pooled_sig, pooled_err = [], []
    for tgt in TARGETS:
        smiles, acts, thr = load_target_data(tgt)
        X = featurize(smiles); y = np.asarray(acts, float)
        oof_pred = np.full(len(y), np.nan); oof_sig = np.full(len(y), np.nan)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            rf = RandomForestRegressor(**RF_KW).fit(X[tr], y[tr])
            P = np.stack([t.predict(X[te]) for t in rf.estimators_], 0)
            oof_pred[te] = P.mean(0); oof_sig[te] = P.std(0)
        err = np.abs(oof_pred - y)
        rho, p = stats.spearmanr(oof_sig, err)
        q = np.quantile(oof_sig, [0.2, 0.4, 0.6, 0.8])
        binr = np.digitize(oof_sig, q)
        rmse_by_q = [float(np.sqrt(np.mean((oof_pred[binr == b] - y[binr == b]) ** 2)))
                     for b in range(5)]
        out[tgt] = dict(n=len(y), spearman_sigma_err=float(rho), p=float(p),
                        rmse_low_unc=rmse_by_q[0], rmse_high_unc=rmse_by_q[4],
                        rmse_by_quintile=rmse_by_q)
        pooled_sig.append(oof_sig); pooled_err.append(err)
        print(f"{tgt}: rho(sigma,|err|)={rho:+.3f} (p={p:.1e})  "
              f"RMSE Q1={rmse_by_q[0]:.2f} -> Q5={rmse_by_q[4]:.2f}")
    ps = np.concatenate(pooled_sig); pe = np.concatenate(pooled_err)
    rho, p = stats.spearmanr(ps, pe)
    out['pooled'] = dict(n=len(ps), spearman_sigma_err=float(rho), p=float(p))
    print(f"POOLED: rho(sigma,|err|)={rho:+.3f} (p={p:.1e}), n={len(ps)}")
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
