"""Leakage-controlled reliability analysis with macro-averaging and the in-domain
novelty gap (Stage-1 items 2, 3, 4).

Three additions over run_reliability_v1.py:

(1) STRUCTURE-DISJOINT SPLITS. Replicate records are aggregated to one value per
    canonical structure (median pIC50) BEFORE cross-validation, so no structure can
    appear in both a training and a test fold. Only SCD-1 contains replicates
    (136 of 762 rows, up to 7 per compound); the other four targets are already one
    row per structure, so their numbers are unchanged by construction. We also report
    the within-compound spread of the replicate measurements, which bounds the error
    any model can achieve on that target.

(2) MACRO-AVERAGED RISK-COVERAGE. The pooled (micro) curve is dominated by the two
    largest targets (DRD2 and DRD3 supply 16,168 of 21,173 predictions). We therefore
    also compute risk-coverage within each target, using a within-target disagreement
    ranking, and average the five curves with equal weight.

(3) IN-DOMAIN NOVELTY GAP. Because the support set is drawn from the training
    compounds, d_train(x) <= nu_support(x) for every molecule. The gap
        g(x) = nu_support(x) - d_train(x) >= 0
    identifies molecules that are novel with respect to the few starting compounds but
    still close to some other training compound. We test whether high-gap molecules
    retain low error, that is, whether novelty can be obtained without leaving the
    region the model was fitted on.

Writes reliability_v2_analysis.json.

    python -m world_model.run_reliability_v2
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem, RDLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.oracle import TARGETS, featurize  # noqa: E402
from world_model.run_applicability_v1 import max_tanimoto_to_set, partial_spearman  # noqa: E402
from world_model.run_reliability_v1 import partial2, risk_coverage, quintile_mean  # noqa: E402

RDLogger.DisableLog('rdApp.*')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'outputs', 'cwm_v1', 'reliability_v2_analysis.json')
RF_KW = dict(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)
K_SUPPORT = 10
COVS = [0.2, 0.4, 0.6, 0.8, 1.0]


def load_deduplicated(target):
    """One row per STANDARDIZED PARENT (InChIKey), median activity.

    Delegates to world_model.standardize so that every analysis in the study uses the
    same grouping. Canonical-SMILES grouping leaves salt and charge variants of the same
    compound in different folds; parent-InChIKey grouping removes them.
    """
    from world_model.standardize import load_standardized
    smiles, y, thr, st = load_standardized(target)
    stats_d = dict(n_rows=st['n_rows_raw'], n_unique=st['n_unique_parents'],
                   n_duplicate_rows=st['n_duplicate_rows'],
                   n_compounds_with_replicates=st['n_compounds_with_replicates'],
                   mean_within_compound_sd=st['mean_within_compound_sd'],
                   max_replicates=st['max_replicates'])
    return smiles, y, thr, stats_d


def rmse_quintile(key, err):
    """RMSE within equal-count quintiles of `key` (Q1 = lowest)."""
    order = stats.rankdata(key, method='ordinal') - 1
    b = (order * 5 // len(key)).astype(int)
    return [float(np.sqrt(np.mean(err[b == i] ** 2))) for i in range(5)]


def main():
    out = {'protocol': 'structure-disjoint 5-fold CV on canonical structures '
                       '(replicates aggregated by median before splitting)'}
    P = {k: [] for k in ('err', 'sig', 'dtr', 'nov', 'gap')}
    per_target_rc = {}
    for tgt in TARGETS:
        smiles, y, thr, dup = load_deduplicated(tgt)
        X = featurize(smiles)
        oof = {k: np.full(len(y), np.nan) for k in ('pred', 'sig', 'dtr', 'nov')}
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            rf = RandomForestRegressor(**RF_KW).fit(X[tr], y[tr])
            Pr = np.stack([t.predict(X[te]) for t in rf.estimators_], 0)
            oof['pred'][te] = Pr.mean(0); oof['sig'][te] = Pr.std(0)
            oof['dtr'][te] = 1.0 - max_tanimoto_to_set(X[te], X[tr])
            tr_act = tr[y[tr] >= thr]
            rng = np.random.default_rng(0)
            sup = tr_act[rng.choice(len(tr_act), min(K_SUPPORT, len(tr_act)), replace=False)]
            oof['nov'][te] = 1.0 - max_tanimoto_to_set(X[te], X[sup])
        err = np.abs(oof['pred'] - y)
        gap = oof['nov'] - oof['dtr']                      # >= 0 by nesting
        rc, rc_auc = risk_coverage(oof['sig'], err)
        per_target_rc[tgt] = rc
        # in-domain novelty: novel vs seeds (top tercile of nu) split by low/high d_train
        hi_nov = oof['nov'] >= np.quantile(oof['nov'], 2 / 3)
        med_d = np.median(oof['dtr'][hi_nov])
        in_dom = hi_nov & (oof['dtr'] <= med_d)            # novel but still supported
        out_dom = hi_nov & (oof['dtr'] > med_d)            # novel and unsupported
        out[tgt] = dict(
            duplicates=dup, n=int(len(y)),
            spearman_sigma_err=float(stats.spearmanr(oof['sig'], err)[0]),
            spearman_nov_err=float(stats.spearmanr(oof['nov'], err)[0]),
            spearman_dtr_err=float(stats.spearmanr(oof['dtr'], err)[0]),
            partial_err_nov_given_dtr=partial_spearman(err, oof['nov'], oof['dtr'])[0],
            partial_err_sig_given_nov_dtr=partial2(err, oof['sig'], oof['nov'], oof['dtr'])[0],
            risk_coverage_rmse=rc, error_retention_auc=rc_auc,
            rmse_by_sigma_quintile=rmse_quintile(oof['sig'], err),
            gap_mean=float(np.nanmean(gap)), gap_frac_positive=float(np.mean(gap > 1e-9)),
            novel_in_domain_rmse=float(np.sqrt(np.mean(err[in_dom] ** 2))),
            novel_out_domain_rmse=float(np.sqrt(np.mean(err[out_dom] ** 2))),
            novel_in_domain_sigma=float(np.mean(oof['sig'][in_dom])),
            novel_out_domain_sigma=float(np.mean(oof['sig'][out_dom])),
            n_novel_in_domain=int(in_dom.sum()), n_novel_out_domain=int(out_dom.sum()),
        )
        for k, src in (('err', err), ('sig', oof['sig']), ('dtr', oof['dtr']),
                       ('nov', oof['nov']), ('gap', gap)):
            P[k].append(src)
        o = out[tgt]
        print(f"{tgt}: n={o['n']} (dups removed {dup['n_duplicate_rows']})  "
              f"sig->err {o['spearman_sigma_err']:+.3f}  nov->err {o['spearman_nov_err']:+.3f}  "
              f"partial(err,sig|nov,d) {o['partial_err_sig_given_nov_dtr']:+.3f}  |  "
              f"novel RMSE in-domain {o['novel_in_domain_rmse']:.2f} vs out {o['novel_out_domain_rmse']:.2f}",
              flush=True)

    err = np.concatenate(P['err']); sig = np.concatenate(P['sig'])
    dtr = np.concatenate(P['dtr']); nov = np.concatenate(P['nov']); gap = np.concatenate(P['gap'])
    rc_micro, _ = risk_coverage(sig, err)
    rc_macro = {f'{c:.1f}': float(np.mean([per_target_rc[t][f'{c:.1f}'] for t in TARGETS]))
                for c in COVS}
    out['pooled'] = dict(
        n=int(len(err)),
        spearman_sigma_err=float(stats.spearmanr(sig, err)[0]),
        spearman_nov_err=float(stats.spearmanr(nov, err)[0]),
        spearman_dtr_err=float(stats.spearmanr(dtr, err)[0]),
        partial_err_nov_given_dtr=partial_spearman(err, nov, dtr)[0],
        partial_err_sig_given_nov_dtr=partial2(err, sig, nov, dtr)[0],
        risk_coverage_micro=rc_micro, risk_coverage_macro=rc_macro,
        rmse_by_sigma_quintile=rmse_quintile(sig, err),
        gap_mean=float(np.nanmean(gap)), gap_frac_positive=float(np.mean(gap > 1e-9)),
        err_by_gap_quintile=quintile_mean(gap, err),
        sigma_by_gap_quintile=quintile_mean(gap, sig),
    )
    po = out['pooled']
    print(f"\nPOOLED (n={po['n']}): sig->err {po['spearman_sigma_err']:+.3f}  "
          f"nov->err {po['spearman_nov_err']:+.3f}  d->err {po['spearman_dtr_err']:+.3f}  "
          f"partial(err,nov|d) {po['partial_err_nov_given_dtr']:+.3f}  "
          f"partial(err,sig|nov,d) {po['partial_err_sig_given_nov_dtr']:+.3f}")
    print("  risk-coverage MICRO:", {k: round(v, 2) for k, v in rc_micro.items()})
    print("  risk-coverage MACRO:", {k: round(v, 2) for k, v in rc_macro.items()})
    print(f"  novelty gap g: mean={po['gap_mean']:.3f}, positive in {100*po['gap_frac_positive']:.0f}% of compounds")
    print("  |err| by gap quintile:", [round(v, 3) for v in po['err_by_gap_quintile']])
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
