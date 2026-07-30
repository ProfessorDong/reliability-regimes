"""Build and rigorously sanity-gate the Chemistry World Model reward oracles.

Motivation: the v4 per-target neural MLPs collapse to the training mean and are
unusable as a reward. We use field-standard QSAR oracles (tree ensembles on
1024-bit ECFP4) and select per target by repeated cross-validation.

Rigor (per user: correctness over speed):
  * Repeated K-fold CV over N_SEEDS random seeds -> mean +/- std for every metric.
  * Model bake-off: RandomForest vs ExtraTrees vs HistGradientBoosting.
  * Deployed oracle = best by mean CV Spearman among ENSEMBLE models that expose
    per-estimator predictions (RF / ExtraTrees), so the reward can use epistemic
    uncertainty (R = mean - lambda * std).
  * Scaffold split reported alongside random CV: the in-distribution -> OOD gap
    quantifies the applicability domain and motivates the actionability constraint.

Artifacts (outputs/frozen/): oracle_{target}.pkl (deployed model + metadata),
oracle_metrics.json (full bake-off, mean+/-std, scaffold gap). Exit 1 on gate fail.

Run:
    python -m reliability.oracle_sanity_gate
    python -m reliability.oracle_sanity_gate --targets drd3 --seeds 5
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.chemistry import scaffold_split  # noqa: E402
from reliability.oracle import (  # noqa: E402
    TARGETS, OUT_DIR, featurize, load_target_data,
)

from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor)
from sklearn.model_selection import KFold, cross_val_predict
from scipy.stats import spearmanr
import joblib

# Candidate models. `ensemble` flags those that expose per-estimator predictions
# (eligible to be the deployed oracle, since the reward needs uncertainty).
def _models():
    return {
        'rf': (RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     n_jobs=-1, random_state=0), True),
        'extratrees': (ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                                           n_jobs=-1, random_state=0), True),
        'histgb': (HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                 random_state=0), False),
    }

N_SEEDS = 5
MIN_CV_R2, MIN_CV_SPEARMAN, MIN_CV_AUC = 0.30, 0.50, 0.70


def _auc(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels).astype(int)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(scores); ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def _r2(y, p):
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1 - np.sum((y - p) ** 2) / ss_tot) if ss_tot > 0 else float('nan')


def repeated_cv(make_model, X, y, thr, seeds):
    r2s, sprs, aucs = [], [], []
    for s in range(seeds):
        cv = KFold(5, shuffle=True, random_state=s)
        p = cross_val_predict(make_model(), X, y, cv=cv, n_jobs=-1)
        r2s.append(_r2(y, p)); sprs.append(float(spearmanr(p, y).correlation))
        aucs.append(_auc(p, y >= thr))
    agg = lambda a: (float(np.mean(a)), float(np.std(a)))
    return dict(zip(['r2', 'spearman', 'auc'], [agg(r2s), agg(sprs), agg(aucs)]))


def evaluate_target(target, seeds):
    smiles, acts, thr = load_target_data(target)
    X = featurize(smiles)

    bakeoff = {}
    for name, (proto, is_ens) in _models().items():
        params = proto.get_params()
        make = lambda p=params, c=type(proto): c(**p)
        m = repeated_cv(make, X, acts, thr, seeds)
        m['ensemble'] = is_ens
        bakeoff[name] = m

    # deployed = best mean CV Spearman among ensemble models (uncertainty-capable)
    ens = {k: v for k, v in bakeoff.items() if v['ensemble']}
    selected = max(ens, key=lambda k: ens[k]['spearman'][0])

    # scaffold split (applicability-domain boundary), using the selected model
    tr, va, te = scaffold_split(smiles, frac_train=0.8, frac_val=0.1)
    proto = _models()[selected][0]
    sc_model = type(proto)(**proto.get_params()).fit(X[[i for i in tr + va]],
                                                     acts[[i for i in tr + va]])
    te_p = sc_model.predict(X[te]); te_y = acts[te]
    scaffold = dict(r2=_r2(te_y, te_p),
                    spearman=float(spearmanr(te_p, te_y).correlation),
                    n_active_test=int((te_y >= thr).sum()))

    # fit deployed oracle on ALL data and persist
    final = type(proto)(**proto.get_params()).fit(X, acts)
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump({'model': final, 'model_name': selected, 'n_bits': 1024,
                 'radius': 2, 'threshold': thr, 'target': target,
                 'n_train': len(smiles)},
                os.path.join(OUT_DIR, f'oracle_{target}.pkl'))

    sel = bakeoff[selected]
    res = dict(target=target, n=len(smiles), threshold=thr,
               frac_active=float(np.mean(acts >= thr)),
               selected_model=selected, bakeoff=bakeoff, scaffold=scaffold,
               cv_r2=sel['r2'], cv_spearman=sel['spearman'], cv_auc=sel['auc'])
    res['pass'] = bool(sel['r2'][0] >= MIN_CV_R2 and sel['spearman'][0] >= MIN_CV_SPEARMAN
                       and sel['auc'][0] >= MIN_CV_AUC)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=list(TARGETS))
    ap.add_argument('--seeds', type=int, default=N_SEEDS)
    args = ap.parse_args()

    rows = []
    for t in args.targets:
        print(f"[{t}] repeated {args.seeds}-seed CV bake-off ...", flush=True)
        rows.append(evaluate_target(t, args.seeds))

    print(f"\n{'target':6} {'n':>5} {'model':>10} {'CV_R2':>13} {'CV_Spr':>13} "
          f"{'CV_AUC':>13} {'Sc_Spr':>7} {'pass':>5}")
    print('-' * 86)
    for r in rows:
        f = lambda m: f"{m[0]:.3f}+/-{m[1]:.3f}"
        print(f"{r['target']:6} {r['n']:>5} {r['selected_model']:>10} "
              f"{f(r['cv_r2']):>13} {f(r['cv_spearman']):>13} {f(r['cv_auc']):>13} "
              f"{r['scaffold']['spearman']:>7.3f} {'PASS' if r['pass'] else 'FAIL':>5}")

    with open(os.path.join(OUT_DIR, 'oracle_metrics.json'), 'w') as fh:
        json.dump({'n_seeds': args.seeds,
                   'gate': dict(cv_r2=MIN_CV_R2, cv_spearman=MIN_CV_SPEARMAN, cv_auc=MIN_CV_AUC),
                   'results': rows}, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'oracle_metrics.json')} and oracle_*.pkl")

    failed = [r['target'] for r in rows if not r['pass']]
    if failed:
        print(f"GATE FAILED (in-distribution) for: {', '.join(failed)}")
        sys.exit(1)
    print("GATE PASSED (in-distribution) for all targets.")


if __name__ == '__main__':
    main()
