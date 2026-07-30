"""Calibrated prediction intervals via split conformal regression (Tier-2 item 1).

The disagreement score sigma_T ranks error but is not a calibrated interval. Split
conformal regression converts it into intervals with a guaranteed marginal coverage
under exchangeability, and lets us test whether sigma_T carries usable information:

  * STANDARD (unnormalised) conformal: score s_i = |y_i - yhat_i|; every compound gets
    the SAME interval width q. This ignores sigma_T.
  * ADAPTIVE (normalised) conformal:   score s_i = |y_i - yhat_i| / (sigma_i + eps);
    width is q * (sigma_i + eps), so molecules the ensemble disagrees on get wider
    intervals.

If sigma_T is informative, adaptive intervals attain the same coverage with a smaller
median width, and their width tracks actual error. Both are evaluated on a test fold
never used for fitting or calibration, with three disjoint roles per split
(proper-train / calibration / test), on the structure-deduplicated data.

Reports, per target and pooled, at alpha in {0.2, 0.1, 0.05}:
  empirical coverage, mean and median interval width, and the conditional coverage in
  the lowest and highest sigma_T quintile (a check on whether coverage is uniform).
Writes conformal_analysis.json.

    python -m world_model.run_conformal_v1
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.oracle import TARGETS, featurize  # noqa: E402
from world_model.run_reliability_v2 import load_deduplicated  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'outputs', 'cwm_v1', 'conformal_analysis.json')
RF_KW = dict(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)
ALPHAS = [0.2, 0.1, 0.05]
EPS = 1e-3


def conformal_quantile(scores, alpha):
    """Finite-sample split-conformal quantile: ceil((n+1)(1-alpha))/n empirical quantile."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:                      # not enough calibration points for this alpha
        return float('inf')
    return float(np.sort(scores)[k - 1])


def main():
    out, pooled = {}, {a: {'cov_s': [], 'w_s': [], 'cov_a': [], 'w_a': []} for a in ALPHAS}
    for tgt in TARGETS:
        smiles, y, thr, dup = load_deduplicated(tgt)
        X = featurize(smiles)
        rec = {a: dict(cov_s=[], w_s=[], cov_a=[], w_a=[], cov_a_lo=[], cov_a_hi=[]) for a in ALPHAS}
        for tr_all, te in KFold(5, shuffle=True, random_state=0).split(X):
            rng = np.random.default_rng(0)
            perm = rng.permutation(len(tr_all))
            n_cal = max(50, len(tr_all) // 4)                 # 25% of train for calibration
            cal, ptr = tr_all[perm[:n_cal]], tr_all[perm[n_cal:]]
            rf = RandomForestRegressor(**RF_KW).fit(X[ptr], y[ptr])   # proper-train only
            def pred(idx):
                P = np.stack([t.predict(X[idx]) for t in rf.estimators_], 0)
                return P.mean(0), P.std(0)
            yc, sc = pred(cal); yt, st_ = pred(te)
            s_std = np.abs(y[cal] - yc)                        # unnormalised score
            s_ada = s_std / (sc + EPS)                         # sigma-normalised score
            err_t = np.abs(y[te] - yt)
            q_lo, q_hi = np.quantile(st_, [0.2, 0.8])
            for a in ALPHAS:
                qs, qa = conformal_quantile(s_std, a), conformal_quantile(s_ada, a)
                w_s = np.full(len(te), 2 * qs)
                w_a = 2 * qa * (st_ + EPS)
                cov_s = err_t <= qs
                cov_a = err_t <= qa * (st_ + EPS)
                rec[a]['cov_s'].append(cov_s); rec[a]['w_s'].append(w_s)
                rec[a]['cov_a'].append(cov_a); rec[a]['w_a'].append(w_a)
                rec[a]['cov_a_lo'].append(cov_a[st_ <= q_lo])
                rec[a]['cov_a_hi'].append(cov_a[st_ >= q_hi])
        out[tgt] = {}
        for a in ALPHAS:
            cs = np.concatenate(rec[a]['cov_s']); ca = np.concatenate(rec[a]['cov_a'])
            ws = np.concatenate(rec[a]['w_s']); wa = np.concatenate(rec[a]['w_a'])
            out[tgt][f'alpha{a}'] = dict(
                target_coverage=1 - a,
                standard_coverage=float(cs.mean()), standard_width_median=float(np.median(ws)),
                adaptive_coverage=float(ca.mean()), adaptive_width_median=float(np.median(wa)),
                adaptive_width_mean=float(np.mean(wa)),
                adaptive_coverage_low_sigma=float(np.concatenate(rec[a]['cov_a_lo']).mean()),
                adaptive_coverage_high_sigma=float(np.concatenate(rec[a]['cov_a_hi']).mean()),
                n=int(len(cs)))
            pooled[a]['cov_s'].append(cs); pooled[a]['w_s'].append(ws)
            pooled[a]['cov_a'].append(ca); pooled[a]['w_a'].append(wa)
        r = out[tgt]['alpha0.1']
        print(f"{tgt}: 90% target | standard cov={r['standard_coverage']:.3f} w={r['standard_width_median']:.2f}"
              f" | adaptive cov={r['adaptive_coverage']:.3f} w={r['adaptive_width_median']:.2f}"
              f" | adaptive cov lo-sigma={r['adaptive_coverage_low_sigma']:.3f}"
              f" hi-sigma={r['adaptive_coverage_high_sigma']:.3f}", flush=True)
    out['pooled'] = {}
    for a in ALPHAS:
        cs = np.concatenate(pooled[a]['cov_s']); ca = np.concatenate(pooled[a]['cov_a'])
        ws = np.concatenate(pooled[a]['w_s']); wa = np.concatenate(pooled[a]['w_a'])
        out['pooled'][f'alpha{a}'] = dict(
            target_coverage=1 - a,
            standard_coverage=float(cs.mean()), standard_width_median=float(np.median(ws)),
            adaptive_coverage=float(ca.mean()), adaptive_width_median=float(np.median(wa)),
            width_ratio_adaptive_over_standard=float(np.median(wa) / np.median(ws)),
            n=int(len(cs)))
        p = out['pooled'][f'alpha{a}']
        print(f"POOLED alpha={a}: standard cov={p['standard_coverage']:.3f} w={p['standard_width_median']:.2f}"
              f" | adaptive cov={p['adaptive_coverage']:.3f} w={p['adaptive_width_median']:.2f}"
              f" | width ratio={p['width_ratio_adaptive_over_standard']:.2f}")
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
