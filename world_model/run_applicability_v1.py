"""Applicability-domain analysis (reviewer points: shared-fingerprint confound;
"is uncertainty just measuring distance?").

For each target, run 5-fold CV with the deployed oracle configuration (identical
to analyze_calibration.py), and for every out-of-fold (OOF) compound collect:
  - predicted pIC50  (ensemble mean)
  - measured pIC50   (ground truth)  -> absolute error
  - sigma_T          (per-tree ensemble std)
  - d_train          (1 - max Tanimoto to that fold's TRAINING compounds),
                     the distance from the model's applicability domain.

We then test, per target and pooled:
  (a) Spearman(d_train, sigma_T)   does uncertainty rise with distance?
  (b) Spearman(d_train, error)     does oracle error rise with distance?
  (c) Spearman(sigma_T, error)     baseline (reproduces calibration result)
  (d) partial Spearman(error, sigma_T | d_train)
                                   does sigma_T predict error BEYOND distance?
and a d_train-quintile table of mean sigma_T and RMSE.

All ranking is leak-free: d_train and OOF predictions are computed against the
SAME training fold. Writes applicability_analysis.json.

    python -m world_model.run_applicability_v1
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from world_model.oracle import TARGETS, load_target_data, featurize

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'outputs', 'cwm_v1', 'applicability_analysis.json')
RF_KW = dict(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)


def max_tanimoto_to_set(Xq, Xref, block=512):
    """For each row of binary Xq, max Tanimoto to any row of binary Xref.

    Tanimoto(a,b) = |a&b| / (|a|+|b|-|a&b|). Vectorized via dot products.
    """
    Xq = Xq.astype(np.float32); Xref = Xref.astype(np.float32)
    aref = Xref.sum(1)                       # |b| for each reference
    out = np.empty(len(Xq), dtype=np.float32)
    for i in range(0, len(Xq), block):
        q = Xq[i:i + block]
        inter = q @ Xref.T                   # |a&b|
        aq = q.sum(1)[:, None]               # |a|
        union = aq + aref[None, :] - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            tan = np.where(union > 0, inter / union, 0.0)
        out[i:i + block] = tan.max(1)
    return out


def partial_spearman(x, y, z):
    """Partial Spearman corr of x,y controlling for z (rank-based partial corr)."""
    rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    r = (rxy - rxz * ryz) / denom if denom > 0 else np.nan
    n = len(x)
    # t-test for partial correlation, df = n - 3
    if np.isfinite(r) and abs(r) < 1:
        t = r * np.sqrt((n - 3) / (1 - r ** 2))
        p = 2 * stats.t.sf(abs(t), df=n - 3)
    else:
        p = np.nan
    return float(r), float(p)


def quintile_table(key, sig, err):
    """Mean sigma_T and RMSE within quintiles of `key` (e.g. d_train).

    Uses rank-based equal-count bins so ties in `key` cannot empty a bin.
    """
    order = stats.rankdata(key, method='ordinal') - 1
    b = (order * 5 // len(key)).astype(int)
    mean_sig, rmse = [], []
    for i in range(5):
        m = b == i
        mean_sig.append(float(np.mean(sig[m])) if m.any() else None)
        rmse.append(float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else None)
    return mean_sig, rmse


def main():
    out = {}
    P_err, P_sig, P_dtr = [], [], []
    for tgt in TARGETS:
        smiles, acts, thr = load_target_data(tgt)
        X = featurize(smiles); y = np.asarray(acts, float)
        oof_pred = np.full(len(y), np.nan)
        oof_sig = np.full(len(y), np.nan)
        oof_dtr = np.full(len(y), np.nan)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            rf = RandomForestRegressor(**RF_KW).fit(X[tr], y[tr])
            Pr = np.stack([t.predict(X[te]) for t in rf.estimators_], 0)
            oof_pred[te] = Pr.mean(0); oof_sig[te] = Pr.std(0)
            sim = max_tanimoto_to_set(X[te], X[tr])
            oof_dtr[te] = 1.0 - sim
        err = np.abs(oof_pred - y)
        rho_ds, p_ds = stats.spearmanr(oof_dtr, oof_sig)
        rho_de, p_de = stats.spearmanr(oof_dtr, err)
        rho_se, p_se = stats.spearmanr(oof_sig, err)
        pr, pp = partial_spearman(err, oof_sig, oof_dtr)
        msig, rmse = quintile_table(oof_dtr, oof_sig, err)
        out[tgt] = dict(
            n=int(len(y)),
            spearman_dtrain_sigma=float(rho_ds), p_dtrain_sigma=float(p_ds),
            spearman_dtrain_err=float(rho_de), p_dtrain_err=float(p_de),
            spearman_sigma_err=float(rho_se), p_sigma_err=float(p_se),
            partial_spearman_err_sigma_given_dtrain=pr,
            partial_p=pp,
            dtrain_quintile_mean_sigma=msig,
            dtrain_quintile_rmse=rmse,
        )
        P_err.append(err); P_sig.append(oof_sig); P_dtr.append(oof_dtr)
        print(f"{tgt}: d->sig {rho_ds:+.3f}  d->err {rho_de:+.3f}  "
              f"sig->err {rho_se:+.3f}  partial(err,sig|d) {pr:+.3f} (p={pp:.1e})")
    pe = np.concatenate(P_err); ps = np.concatenate(P_sig); pd = np.concatenate(P_dtr)
    rho_ds, p_ds = stats.spearmanr(pd, ps)
    rho_de, p_de = stats.spearmanr(pd, pe)
    rho_se, p_se = stats.spearmanr(ps, pe)
    pr, pp = partial_spearman(pe, ps, pd)
    msig, rmse = quintile_table(pd, ps, pe)
    out['pooled'] = dict(
        n=int(len(pe)),
        spearman_dtrain_sigma=float(rho_ds), p_dtrain_sigma=float(p_ds),
        spearman_dtrain_err=float(rho_de), p_dtrain_err=float(p_de),
        spearman_sigma_err=float(rho_se), p_sigma_err=float(p_se),
        partial_spearman_err_sigma_given_dtrain=pr, partial_p=pp,
        dtrain_quintile_mean_sigma=msig, dtrain_quintile_rmse=rmse,
    )
    print(f"POOLED (n={len(pe)}): d->sig {rho_ds:+.3f}  d->err {rho_de:+.3f}  "
          f"sig->err {rho_se:+.3f}  partial(err,sig|d) {pr:+.3f} (p={pp:.1e})")
    print("d_train quintile mean sigma:", [f"{v:.3f}" for v in msig])
    print("d_train quintile RMSE:      ", [f"{v:.3f}" for v in rmse])
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
