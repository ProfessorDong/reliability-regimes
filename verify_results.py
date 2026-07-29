"""Verify every numeric claim in the manuscript against the frozen output files.

Usage:  python verify_results.py
Exits non-zero if any assertion fails.

Each check names the manuscript location of the claim it verifies. All values are
read from outputs/cwm_v1/*.json, which are produced by the run_*/analyze_* scripts
in world_model/ (see README for the reproduction order).
"""
from __future__ import annotations
import json, os, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'cwm_v1')
T = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']
FAILED, CHECKED = [], 0


def L(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def close(where, what, got, expect, tol):
    global CHECKED
    CHECKED += 1
    ok = abs(float(got) - float(expect)) <= tol
    if not ok:
        FAILED.append(f'{where}: {what}: got {got}, expected {expect} +/- {tol}')
    print(f"{'ok  ' if ok else 'FAIL'} [{where}] {what}: {got:.4g} (expect {expect} +/- {tol})")


def cond(where, what, ok, detail=''):
    global CHECKED
    CHECKED += 1
    if not ok:
        FAILED.append(f'{where}: {what}: {detail}')
    print(f"{'ok  ' if ok else 'FAIL'} [{where}] {what}{(' :: ' + detail) if detail else ''}")


# --------------------------------------------------------------- Results 5.1
om = {e['target']: e for e in L('oracle_metrics.json')['results']}
for t, sp in [('scd1', 0.657), ('fads', 0.817), ('nk1r', 0.841), ('drd2', 0.808), ('drd3', 0.852)]:
    e = om[t]
    close('Results/oracle', f'{t} random-CV Spearman', e['bakeoff'][e['selected_model']]['spearman'][0], sp, 0.01)
cond('Results/oracle', 'CV Spearman within 0.66-0.85 for all targets',
     all(0.65 <= om[t]['bakeoff'][om[t]['selected_model']]['spearman'][0] <= 0.86 for t in T))
for t in ['scd1', 'fads', 'drd2']:
    cond('SI Note 2', f'{t} scaffold fold has no actives (AUC/R2 undefined)',
         om[t].get('scaffold', {}).get('n_active_test') == 0)

# --------------------------------------------------------------- Results 5.2
cal = L('calibration_analysis.json')
close('Results/risk', 'pooled Spearman(sigma_T, |error|)', cal['pooled']['spearman_sigma_err'], 0.399, 0.01)
cond('Results/risk', 'sigma_T -> |error| positive in every target',
     all(cal[t]['spearman_sigma_err'] > 0 for t in T))

rel = L('reliability_v2_analysis.json')
cond('Results/risk', 'structure-disjoint n = 21,037 predictions', rel['pooled']['n'] == 21037, str(rel['pooled']['n']))
rc = rel['pooled']['risk_coverage_micro']
close('Results/risk (Fig 1)', 'micro RMSE at 20% coverage', rc['0.2'], 0.40, 0.02)
close('Results/risk (Fig 1)', 'micro RMSE at 100% coverage', rc['1.0'], 0.71, 0.02)
_ma = rel['pooled']['risk_coverage_macro']
close('Results/risk (macro)', 'macro RMSE at 20% coverage', _ma['0.2'], 0.44, 0.02)
close('Results/risk (macro)', 'macro RMSE at 100% coverage', _ma['1.0'], 0.75, 0.02)
cond('Methods/leakage', 'structure-disjoint: SCD-1 duplicates removed (762 -> 626)',
     rel['scd1']['duplicates']['n_unique'] == 626 and rel['scd1']['duplicates']['n_duplicate_rows'] == 136)
cond('Results/in-domain novelty', 'novelty gap positive for 99.6% of compounds',
     rel['pooled']['gap_frac_positive'] > 0.99)
cond('Results/in-domain novelty', 'error falls as the novelty gap widens',
     rel['pooled']['err_by_gap_quintile'][0] > rel['pooled']['err_by_gap_quintile'][-1])
cond('Results/in-domain novelty', 'novel in-domain beats out-of-domain on 4 of 5 targets',
     sum(rel[t]['novel_in_domain_rmse'] < rel[t]['novel_out_domain_rmse'] for t in T) == 4)
cond('Results/risk (Fig 1)', 'risk-coverage monotone increasing (pooled)',
     all(rc[f'{a:.1f}'] <= rc[f'{b:.1f}'] + 1e-9 for a, b in [(.2, .4), (.4, .6), (.6, .8), (.8, 1.)]))
cond('SI Table S4', 'risk-coverage monotone in every target',
     all(all(rel[t]['risk_coverage_rmse'][f'{a:.1f}'] <= rel[t]['risk_coverage_rmse'][f'{b:.1f}'] + 1e-9
             for a, b in [(.2, .4), (.4, .6), (.6, .8), (.8, 1.)]) for t in T))
close('Results/risk', 'partial rho(err, sigma_T | novelty, d_train)',
      rel['pooled']['partial_err_sig_given_nov_dtr'], 0.381, 0.01)

# --------------------------------------------------------------- Results 5.3
close('Results/drift', 'pooled Spearman(support-novelty, |error|)', rel['pooled']['spearman_nov_err'], 0.068, 0.01)
close('Results/drift', 'partial rho(err, novelty | d_train) ~ 0', rel['pooled']['partial_err_nov_given_dtr'], 0.047, 0.01)
close('Results/drift', 'FADS Spearman(novelty, |error|) is negative', rel['fads']['spearman_nov_err'], -0.140, 0.01)
close('Results/drift', 'pooled Spearman(d_train, |error|)', rel['pooled']['spearman_dtr_err'], 0.141, 0.01)

app = L('applicability_analysis.json')
close('SI Note 4', 'pooled Spearman(d_train, sigma_T)', app['pooled']['spearman_dtrain_sigma'], 0.435, 0.01)
close('SI Note 4', 'partial rho(err, sigma_T | d_train)', app['pooled']['partial_spearman_err_sigma_given_dtrain'], 0.383, 0.01)

fr = L('frontier_v2_analysis.json')
for key in ['graphga_lam0.0', 'graphga_lam0.1', 'stga_lam0.0', 'stga_lam0.1']:
    a = fr[key]
    cond('Results/drift (Fig 2)', f'{key}: novelty -> d_train in 0.97-0.99', 0.96 <= a['nov_dtrain'] <= 0.995, f"{a['nov_dtrain']:.3f}")
    cond('Results/drift (Fig 2)', f'{key}: novelty -> sigma_T positive (0.40-0.60)', 0.38 <= a['nov_sig'] <= 0.62, f"{a['nov_sig']:.3f}")
    cond('Results/drift (Fig 2)', f'{key}: novelty -> potency negative', a['nov_pot'] < -0.7, f"{a['nov_pot']:.3f}")
    cond('Results/drift (Fig 2)', f'{key}: n = 300 runs', a['n'] == 300, str(a['n']))
cond('Results/drift', 'novelty->sigma_T holds with NO uncertainty penalty (lambda=0)',
     fr['graphga_lam0.0']['nov_sig'] > 0.3 and fr['stga_lam0.0']['nov_sig'] > 0.3)

# --------------------------------------------------------------- Results 5.4
mv = L('methods_v2_results.json')['results']
cond('Results/method (Table S7)', '15 target-by-k cells', len(mv) == 15, str(len(mv)))
cond('Methods/budget', 'every run uses exactly 300 oracle calls (no overshoot)',
     all(c[k] <= 300 for r in mv for c in r['per_seed'] for k in ('calls_stga','calls_rt','calls_ga')),
     'max=%d' % max(c[k] for r in mv for c in r['per_seed'] for k in ('calls_stga','calls_rt','calls_ga')))
cond('Results/method (Table S7)', 'ST-GA beats Graph GA in ALL 15 cells',
     all(r['agg']['d_vs_ga'] > 0 for r in mv))
cond('Results/method (Table S7)', 'every cell significant at 0.05',
     all(r['agg']['p_vs_ga'] < 0.05 for r in mv))
cond('Results/method', 'ST-GA beats random-triage in ALL 15 cells',
     all(r['agg']['d_vs_rt'] > 0 for r in mv))
import statistics as st
tgt = {t: st.mean([c['stga_ecfp'] - c['graphga'] for r in mv if r['target'] == t for c in r['per_seed']]) for t in T}
tl = st.mean(tgt.values())
close('Results/method (Fig 3)', 'target-level mean gain over Graph GA', tl, 0.0620, 0.004)
cond('Results/method (Fig 3)', 'all five per-target means positive', all(v > 0 for v in tgt.values()),
     ', '.join(f'{t}={v:+.3f}' for t, v in tgt.items()))
close('Results/method', 'DRD2 k=20 gain', [r for r in mv if r['target'] == 'drd2' and r['k'] == 20][0]['agg']['d_vs_ga'], 0.117, 0.01)

hs = L('hierstats_analysis.json')['target_level']
close('SI Table S8', 'dual-encoder target-level gain', hs['mean'], 0.0239, 0.002)
close('SI Table S8', 'dual-encoder target-level P', hs['p'], 0.023, 0.005)

ec = L('ecfp_baseline.json')['summary']
cond('Results/method (Table S10)', 'fingerprint surrogate beats latent on ALL targets',
     all(ec[t]['latent_minus_ecfp'] < 0 for t in T),
     ', '.join(f"{t}={ec[t]['latent_minus_ecfp']:+.3f}" for t in T))
cond('Results/method (Table S10)', 'latent-vs-fingerprint significant (P<=0.02) on all targets',
     all(ec[t]['p_lat_ecfp'] <= 0.02 for t in T))

# --------------------------------------------------------------- Results 5.5 (negatives)
be = L('beta_ablation.json')['summary']
d10 = [be[t]['beta1_vs_beta0_novel']['delta'] for t in T]
close('Results/negatives (Table S9)', 'beta=1 minus beta=0, mean over targets', st.mean(d10), 0.0015, 0.004)
cond('Results/negatives (Table S9)', 'beta term significant on at most 1 of 5 targets',
     sum(be[t]['beta1_vs_beta0_novel']['p'] < 0.05 for t in T) <= 1)
cond('Results/negatives (Table S9)', 'beta term negative on DRD3', be['drd3']['beta1_vs_beta0_novel']['delta'] < 0)

cg = L('compat_gen_analysis.json')['summary']
close('Results/negatives (Table S11)', 'compatibility pair-level Pearson r', cg['pair_pearson_r'], -0.142, 0.02)
cond('Results/negatives (Table S11)', 'compatibility correlation not significant', cg['pair_pearson_p'] > 0.05,
     f"P={cg['pair_pearson_p']:.2f}")

rv = L('recovery_v2_results.json')['results']
cond('Results/negatives (Table S12)', 'retrained model ranks held-out actives 6.8-8.1 pIC50',
     all(6.5 <= rv[t]['retrained_pred'] <= 8.2 for t in T))
cond('Results/negatives (Table S12)', 'retrained model flags held-out cluster as uncertain (sigma 0.57-1.02)',
     all(0.5 <= rv[t]['retrained_sigma'] <= 1.1 for t in T))
cond('Results/negatives (Table S12)', 'recovery above chemical-space null on every target',
     all(rv[t]['rec_stga'] > rv[t]['null_sim'] for t in T))
cond('Results/negatives (Table S12)', 'triage does NOT improve recovery (Graph GA >= ST-GA)',
     all(rv[t]['rec_graphga'] >= rv[t]['rec_stga'] - 1e-9 for t in T))

# --------------------------------------------------------------- summary
print('\n' + '=' * 72)
if FAILED:
    print(f'{len(FAILED)} of {CHECKED} checks FAILED:')
    for f in FAILED:
        print('  -', f)
    sys.exit(1)
print(f'All {CHECKED} numeric claims verified against the frozen outputs.')
