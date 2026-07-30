"""Measured reliability vs novelty, and risk-coverage (reviewer sections 3, 4).

Converts the centerpiece from a proxy (novelty vs sigma_T) into a MEASURED result:
using leakage-free out-of-fold predictions on real labeled compounds, we test whether
the oracle's ABSOLUTE ERROR (not merely its ensemble disagreement) rises with
support-novelty, and we separate support-novelty (distance to a few sampled seed
actives, as in generation) from d_train (distance to the full training fold).

For each target we run 5-fold CV with the deployed RF oracle. For every out-of-fold
compound we record: predicted potency, measured potency, |error|, sigma_T (tree
dispersion), d_train, and support_novelty (1 - max Tanimoto to k=10 active seeds drawn
from the training fold). We report:
  * Spearman(support_novelty, |error|)          -- measured error rises with novelty?
  * Spearman(d_train, |error|)
  * partial Spearman(|error|, support_novelty | d_train)
  * partial Spearman(|error|, sigma_T | support_novelty, d_train)  -- sigma_T beyond both?
  * risk-coverage: RMSE of the lowest-sigma_T fraction, and error-retention AUC
  * calibration: mean |error| by sigma_T quintile
  * mean |error| by support_novelty quintile (nonparametric)
Writes reliability_analysis.json.

    python -m reliability.run_reliability_v1
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability.oracle import TARGETS, load_target_data, featurize  # noqa: E402
from reliability.run_applicability_v1 import max_tanimoto_to_set, partial_spearman  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'outputs', 'frozen', 'reliability_analysis.json')
RF_KW = dict(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)
K_SUPPORT = 10


def partial2(w, x, y, z):
    """Partial Spearman of w,x controlling for BOTH y and z (2nd-order, rank-based)."""
    R = np.vstack([stats.rankdata(a) for a in (w, x, y, z)])
    C = np.corrcoef(R)
    P = np.linalg.pinv(C)                       # precision matrix
    r = -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])
    n = len(w)
    if np.isfinite(r) and abs(r) < 1:
        t = r * np.sqrt((n - 4) / (1 - r ** 2)); p = 2 * stats.t.sf(abs(t), df=n - 4)
    else:
        p = float('nan')
    return float(r), float(p)


def risk_coverage(sig, err):
    """RMSE of the lowest-sigma fraction; error-retention AUC (mean RMSE over coverage)."""
    order = np.argsort(sig)
    e = err[order]
    covs = [0.2, 0.4, 0.6, 0.8, 1.0]
    rmse = {}
    for c in covs:
        m = max(1, int(round(c * len(e))))
        rmse[f'{c:.1f}'] = float(np.sqrt(np.mean(e[:m] ** 2)))
    auc = float(np.mean(list(rmse.values())))
    return rmse, auc


def quintile_mean(key, err):
    order = stats.rankdata(key, method='ordinal') - 1
    b = (order * 5 // len(key)).astype(int)
    return [float(np.mean(err[b == i])) for i in range(5)]


def main():
    out = {}
    P = {k: [] for k in ('err', 'sig', 'dtr', 'nov')}
    for tgt in TARGETS:
        smiles, acts, thr = load_target_data(tgt)
        X = featurize(smiles); y = np.asarray(acts, float)
        oof = {k: np.full(len(y), np.nan) for k in ('pred', 'sig', 'dtr', 'nov')}
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            rf = RandomForestRegressor(**RF_KW).fit(X[tr], y[tr])
            Pr = np.stack([t.predict(X[te]) for t in rf.estimators_], 0)
            oof['pred'][te] = Pr.mean(0); oof['sig'][te] = Pr.std(0)
            oof['dtr'][te] = 1.0 - max_tanimoto_to_set(X[te], X[tr])
            # support = k active seeds drawn from the TRAINING fold (leakage-free)
            tr_act = tr[y[tr] >= thr]
            rng = np.random.default_rng(0)
            sup = tr_act[rng.choice(len(tr_act), min(K_SUPPORT, len(tr_act)), replace=False)]
            oof['nov'][te] = 1.0 - max_tanimoto_to_set(X[te], X[sup])
        err = np.abs(oof['pred'] - y)
        rc, rc_auc = risk_coverage(oof['sig'], err)
        out[tgt] = dict(
            n=int(len(y)),
            spearman_nov_err=float(stats.spearmanr(oof['nov'], err)[0]),
            p_nov_err=float(stats.spearmanr(oof['nov'], err)[1]),
            spearman_dtr_err=float(stats.spearmanr(oof['dtr'], err)[0]),
            partial_err_nov_given_dtr=partial_spearman(err, oof['nov'], oof['dtr'])[0],
            partial_err_sig_given_nov_dtr=partial2(err, oof['sig'], oof['nov'], oof['dtr'])[0],
            risk_coverage_rmse=rc, error_retention_auc=rc_auc,
            err_by_sigma_quintile=quintile_mean(oof['sig'], err),
            err_by_novelty_quintile=quintile_mean(oof['nov'], err),
        )
        for k, src in (('err', err), ('sig', oof['sig']), ('dtr', oof['dtr']), ('nov', oof['nov'])):
            P[k].append(src)
        print(f"{tgt}: nov->err {out[tgt]['spearman_nov_err']:+.3f}  "
              f"partial(err,nov|dtr) {out[tgt]['partial_err_nov_given_dtr']:+.3f}  "
              f"partial(err,sig|nov,dtr) {out[tgt]['partial_err_sig_given_nov_dtr']:+.3f}  "
              f"RMSE@0.2={out[tgt]['risk_coverage_rmse']['0.2']:.2f} vs @1.0={out[tgt]['risk_coverage_rmse']['1.0']:.2f}",
              flush=True)
    err = np.concatenate(P['err']); sig = np.concatenate(P['sig'])
    dtr = np.concatenate(P['dtr']); nov = np.concatenate(P['nov'])
    rc, rc_auc = risk_coverage(sig, err)
    out['pooled'] = dict(
        n=int(len(err)),
        spearman_nov_err=float(stats.spearmanr(nov, err)[0]),
        spearman_dtr_err=float(stats.spearmanr(dtr, err)[0]),
        spearman_nov_dtr=float(stats.spearmanr(nov, dtr)[0]),
        partial_err_nov_given_dtr=partial_spearman(err, nov, dtr)[0],
        partial_err_sig_given_nov_dtr=partial2(err, sig, nov, dtr)[0],
        risk_coverage_rmse=rc, error_retention_auc=rc_auc,
        err_by_sigma_quintile=quintile_mean(sig, err),
        err_by_novelty_quintile=quintile_mean(nov, err),
    )
    po = out['pooled']
    print(f"POOLED (n={po['n']}): nov->err {po['spearman_nov_err']:+.3f}  "
          f"dtr->err {po['spearman_dtr_err']:+.3f}  nov<->dtr {po['spearman_nov_dtr']:+.3f}  "
          f"partial(err,nov|dtr) {po['partial_err_nov_given_dtr']:+.3f}  "
          f"partial(err,sig|nov,dtr) {po['partial_err_sig_given_nov_dtr']:+.3f}")
    print("  RMSE by coverage:", {k: round(v, 2) for k, v in rc.items()})
    print("  |err| by novelty quintile:", [round(v, 2) for v in po['err_by_novelty_quintile']])
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
