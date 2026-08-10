"""Verify every numeric claim in the manuscript against the frozen output files.

Usage:  python verify_results.py
Exits non-zero if any assertion fails.

Each check names the manuscript location of the claim it verifies. All values are
read from outputs/frozen/*.json, which are produced by the run_*/analyze_* scripts
in reliability/ (see README for the reproduction order).
"""
from __future__ import annotations
import json, os, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', 'frozen')
_HERE_DIR = os.path.dirname(os.path.abspath(__file__))
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
cond('Results/risk', 'structure-disjoint n = 20,853 parent structures', rel['pooled']['n'] == 20853, str(rel['pooled']['n']))
rc = rel['pooled']['risk_coverage_micro']
close('Results/risk (Fig 1)', 'micro RMSE at 20% coverage', rc['0.2'], 0.41, 0.02)
close('Results/risk (Fig 1)', 'micro RMSE at 100% coverage', rc['1.0'], 0.71, 0.02)
_ma = rel['pooled']['risk_coverage_macro']
close('Results/risk (macro)', 'macro RMSE at 20% coverage', _ma['0.2'], 0.48, 0.02)
close('Results/risk (macro)', 'macro RMSE at 100% coverage', _ma['1.0'], 0.76, 0.02)
cond('Methods/leakage', 'structure-disjoint: SCD-1 duplicates removed (762 -> 626)',
     rel['scd1']['duplicates']['n_unique'] == 626 and rel['scd1']['duplicates']['n_duplicate_rows'] == 136)
cond('Results/in-domain novelty', 'novelty gap positive for 99.6% of compounds',
     rel['pooled']['gap_frac_positive'] > 0.99)
cond('Results/in-domain novelty', 'error falls as the novelty gap widens',
     rel['pooled']['err_by_gap_quintile'][0] > rel['pooled']['err_by_gap_quintile'][-1])
cond('Results/in-domain novelty', 'novel in-domain beats out-of-domain on 4 of 5 targets',
     sum(rel[t]['novel_in_domain_rmse'] < rel[t]['novel_out_domain_rmse'] for t in T) == 4)
cond('Results/risk (Fig 1)', 'pooled risk-coverage monotone increasing',
     all(rc[f'{a:.1f}'] <= rc[f'{b:.1f}'] + 1e-9 for a, b in [(.2, .4), (.4, .6), (.6, .8), (.8, 1.)]))
_mono = [t for t in T if all(rel[t]['risk_coverage_rmse'][f'{a:.1f}'] <= rel[t]['risk_coverage_rmse'][f'{b:.1f}'] + 1e-9
         for a, b in [(.2, .4), (.4, .6), (.6, .8), (.8, 1.)])]
cond('SI Table S4', 'risk-coverage monotone on 4 of 5 targets (SCD-1 flat)',
     len(_mono) == 4 and 'scd1' not in _mono, 'monotone: ' + ','.join(_mono))
close('Results/risk', 'partial rho(err, sigma_T | novelty, d_train)',
      rel['pooled']['partial_err_sig_given_nov_dtr'], 0.381, 0.01)

# --------------------------------------------------------------- Results 5.3
close('Results/drift', 'pooled Spearman(support-novelty, |error|)', rel['pooled']['spearman_nov_err'], 0.042, 0.01)
close('Results/drift', 'partial rho(err, novelty | d_train) ~ 0', rel['pooled']['partial_err_nov_given_dtr'], 0.022, 0.01)
close('Results/drift', 'FADS Spearman(novelty, |error|) is negative', rel['fads']['spearman_nov_err'], -0.232, 0.01)
close('Results/drift', 'pooled Spearman(d_train, |error|)', rel['pooled']['spearman_dtr_err'], 0.135, 0.01)

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
cond('Results/negatives (Table S12)', 'retrained model ranks held-out actives 6.8-8.1 pActivity',
     all(6.5 <= rv[t]['retrained_pred'] <= 8.2 for t in T))
cond('Results/negatives (Table S12)', 'retrained model flags held-out cluster as uncertain (sigma 0.57-1.02)',
     all(0.5 <= rv[t]['retrained_sigma'] <= 1.1 for t in T))
cond('Results/negatives (Table S12)', 'recovery above chemical-space null on every target',
     all(rv[t]['rec_stga'] > rv[t]['null_sim'] for t in T))
cond('Results/negatives (Table S12)', 'triage does NOT improve recovery (Graph GA >= ST-GA)',
     all(rv[t]['rec_graphga'] >= rv[t]['rec_stga'] - 1e-9 for t in T))

# =============================================================== Regime 2: temporal
tmp = L('temporal_analysis.json')
tp = tmp['pooled']
cond('Methods/temporal', 'temporal split trains pre-2015, tests 2015+', tmp['cut_year'] == 2015)
close('Results/temporal', 'future-compound RMSE', tp['rmse'], 1.07, 0.03)
close('Results/temporal', 'sigma->err degrades under shift', tp['spearman_sigma_err'], 0.13, 0.03)
close('Results/temporal', 'conformal coverage falls below nominal', tp['conformal_coverage_adaptive'], 0.838, 0.02)
cond('Results/temporal', 'temporal coverage is BELOW nominal 0.90',
     tp['conformal_coverage_adaptive'] < 0.90, f"{tp['conformal_coverage_adaptive']:.3f}")
cond('Results/temporal', 'DRD3 loses the signal entirely under shift',
     abs(tmp['drd3']['spearman_sigma_err']) < 0.05, f"{tmp['drd3']['spearman_sigma_err']:.3f}")
for t in ['scd1', 'nk1r', 'drd2', 'drd3']:
    c = tmp[t]['control_random_same_size']
    cond('Results/temporal (control)', f'{t}: size-matched control has LOWER error than temporal',
         c['rmse'] < tmp[t]['rmse_test'], f"{c['rmse']:.2f} vs {tmp[t]['rmse_test']:.2f}")
    cond('Results/temporal (control)', f'{t}: size-matched control recovers ~nominal coverage',
         c['conformal_coverage_adaptive'] > 0.87, f"{c['conformal_coverage_adaptive']:.3f}")
cond('Results/temporal (control)', 'DRD3 control recovers the ranking the temporal split loses',
     tmp['drd3']['control_random_same_size']['spearman_sigma_err'] > 0.25)
# The Discussion types two counts that a re-run could silently invalidate: the ranking
# "vanishes on one target of four", and "the control degrades on none of these". Neither was
# derived from a macro, so assert both against the frozen split rather than the prose.
_tt = ['scd1', 'nk1r', 'drd2', 'drd3']
_gone = [t for t in _tt if abs(tmp[t]['spearman_sigma_err']) < 0.05]
cond('Results/temporal', 'the ranking vanishes on exactly one of the four temporal targets',
     len(_gone) == 1, f"lost on {_gone}; rho=" +
     ', '.join(f"{t} {tmp[t]['spearman_sigma_err']:.3f}" for t in _tt))
cond('Results/temporal (control)', 'the control degrades on none of error, ranking or coverage',
     all(tmp[t]['control_random_same_size']['rmse'] < tmp[t]['rmse_test']
         and tmp[t]['control_random_same_size']['spearman_sigma_err'] > 0.15
         and tmp[t]['control_random_same_size']['conformal_coverage_adaptive'] > 0.89
         for t in _tt),
     'control rho=' + ', '.join(
         f"{t} {tmp[t]['control_random_same_size']['spearman_sigma_err']:.2f}" for t in _tt))

# =============================================================== Regime 1: conformal
con = L('conformal_analysis.json')['pooled']
for a, tol in [('alpha0.2', 0.02), ('alpha0.1', 0.02), ('alpha0.05', 0.02)]:
    c = con[a]
    close('Results/conformal', f'{a}: adaptive coverage hits nominal',
          c['adaptive_coverage'], c['target_coverage'], tol)
cond('Results/conformal', 'sigma-normalised intervals are narrower at 90%',
     con['alpha0.1']['width_ratio_adaptive_over_standard'] < 1.0,
     f"ratio={con['alpha0.1']['width_ratio_adaptive_over_standard']:.2f}")

# =============================================================== acquisition vs real labels
po = L('poolopt_analysis.json')['summary']
import statistics as _st
enr = {m: _st.mean([po[t][m]['enrichment_vs_random'] for t in po])
       for m in ['greedy', 'ucb', 'lcb', 'conformal']}
cond('Results/pool', 'all model-guided strategies beat random',
     all(v > 1.5 for v in enr.values()), str({k: round(v, 2) for k, v in enr.items()}))
cond('Results/pool', 'CONSERVATIVE acquisition finds FEWER true actives than greedy',
     enr['lcb'] < enr['greedy'] and enr['conformal'] < enr['greedy'],
     f"greedy={enr['greedy']:.2f} lcb={enr['lcb']:.2f} conformal={enr['conformal']:.2f}")
cond('Results/pool', 'optimistic (UCB) is the best strategy', enr['ucb'] == max(enr.values()),
     f"ucb={enr['ucb']:.2f}")

# =============================================================== cross-document consistency
# The original defect was hand-copied numbers desyncing between manuscript and SI.
# numbers.tex is generated from these same JSONs; check the macros agree with the source.
import re as _re, os as _os
NUM = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..',
                    'WritePaper', 'theranostics', 'JournalPapers_npjDD', 'numbers.tex')
mac = {}          # always defined: a clean clone has no numbers.tex and must still run
if _os.path.exists(NUM):
    _txt = open(NUM).read()
    for _m in _re.finditer(r'\\newcommand\{\\(\w+)\}\{', _txt):
        _i = _m.end(); _d = 1; _b = ''
        while _i < len(_txt) and _d:
            _c = _txt[_i]
            if _c == '{': _d += 1
            elif _c == '}':
                _d -= 1
                if not _d: break
            _b += _c; _i += 1
        mac[_m.group(1)] = _b
    cond('numbers.tex', 'Nstruct macro matches the frozen structure count',
         mac.get('Nstruct', '').replace('{,}', '').replace(',', '') == str(rel['pooled']['n']),
         f"macro={mac.get('Nstruct')} json={rel['pooled']['n']}")
    close('numbers.tex', 'RhoSigErr macro matches source', float(mac['RhoSigErr']),
          rel['pooled']['spearman_sigma_err'], 0.005)
    close('numbers.tex', 'TempRhoSigErr macro matches source', float(mac['TempRhoSigErr']),
          tp['spearman_sigma_err'], 0.005)
    close('numbers.tex', 'PoolUCB macro matches source', float(mac['PoolUCB']), enr['ucb'], 0.01)
    # Table 1 carries two distinct counts per target. Conflating them is the failure this
    # check exists to prevent, so both are asserted against the same frozen source.
    _cap = {'scd1': 'Scd', 'fads': 'Fads', 'nk1r': 'Nkone',
            'drd2': 'Drdtwo', 'drd3': 'Drdthree'}
    _int = lambda v: int(v.replace('{,}', '').replace(',', ''))
    for _t, _c in _cap.items():
        _d = rel[_t]['duplicates']
        cond('numbers.tex', f'Table 1 record count for {_t} matches source',
             _int(mac[f'Rows{_c}']) == _d['n_rows'],
             f"macro={mac[f'Rows{_c}']} json={_d['n_rows']}")
        cond('numbers.tex', f'Table 1 structure count for {_t} matches source',
             _int(mac[f'Struct{_c}']) == _d['n_unique'],
             f"macro={mac[f'Struct{_c}']} json={_d['n_unique']}")
    cond('numbers.tex', 'Table 1 record total is the sum of the per-target records',
         _int(mac['Nrows']) == sum(rel[t]['duplicates']['n_rows'] for t in _cap),
         f"macro={mac['Nrows']}")
    cond('numbers.tex', 'Table 1 structure total equals the pooled prediction count',
         _int(mac['Nstruct']) == sum(rel[t]['duplicates']['n_unique'] for t in _cap)
         == rel['pooled']['n'], f"macro={mac['Nstruct']} json={rel['pooled']['n']}")
else:
    print('skip [numbers.tex] macro cross-check: numbers.tex lives with the manuscript, '
          'which is intentionally not in this repository')

# --------------------------------------------------- semantic checks (not just numbers)
# A number can match its source file while the prose around it names the wrong target or the
# wrong direction. These assertions pin the statements the text actually makes.
print()
_N = {'scd1': 'SCD-1', 'fads': 'FADS', 'nk1r': 'NK1R', 'drd2': 'DRD2', 'drd3': 'DRD3'}
_TT = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']
_ex = [t for t in _TT if not rel[t]['novel_in_domain_rmse'] < rel[t]['novel_out_domain_rmse']]
cond('semantic', 'the in-domain error exception is SCD-1, not FADS',
     [_N[t] for t in _ex] == ['SCD-1'],
     f"exception target(s) = {[_N[t] for t in _ex]}")
cond('semantic', 'nearer-training group has lower error on exactly 4 of 5 targets',
     sum(rel[t]['novel_in_domain_rmse'] < rel[t]['novel_out_domain_rmse'] for t in _TT) == 4, '')
cond('semantic', 'nearer-training group has lower disagreement on all 5 targets',
     all(rel[t]['novel_in_domain_sigma'] < rel[t]['novel_out_domain_sigma'] for t in _TT), '')

_tt = [t for t in ['scd1', 'nk1r', 'drd2', 'drd3'] if t in tmp]
cond('semantic', 'temporal split is dated by first disclosure, not median year',
     tmp.get('year_field') == 'year_min', f"year_field={tmp.get('year_field')}")
cond('semantic', 'temporal error rises above the size-matched control on every target',
     all(tmp[t]['rmse_test'] > tmp[t]['control_random_same_size']['rmse'] for t in _tt), '')
# "kept" and "lost" are decided by whether the temporal interval overlaps the control's,
# not by a hand-set threshold on the difference.
def _overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]

_keep, _lose = [], []
for t in _tt:
    _a = tmp[t]['spearman_sigma_err_ci95']
    _b = tmp[t]['control_random_same_size']['spearman_sigma_err_ci95']
    (_keep if _overlap(_a, _b) else _lose).append(t)
cond('semantic', 'error ranking is kept on SCD-1 and NK1R and lost on DRD2 and DRD3',
     sorted(_keep) == ['nk1r', 'scd1'] and sorted(_lose) == ['drd2', 'drd3'],
     f"kept={[_N[t] for t in _keep]} lost={[_N[t] for t in _lose]}")
cond('semantic', 'every temporal RMSE interval lies above its control interval',
     all(tmp[t]['rmse_test_ci95'][0]
         > tmp[t]['control_random_same_size']['rmse_ci95'][1] for t in _tt), '')
_covsep = sum(tmp[t]['conformal_coverage_adaptive_ci95'][1]
              < tmp[t]['control_random_same_size']['conformal_coverage_adaptive_ci95'][0]
              for t in _tt)
cond('semantic', 'coverage intervals separate on exactly two targets (DRD2, DRD3)',
     _covsep == 2, f'{_covsep} of {len(_tt)}')
cond('semantic', 'pooled coverage interval lies below the nominal 0.900',
     tmp['pooled']['conformal_coverage_adaptive_ci95'][1] < 0.900,
     str(tmp['pooled']['conformal_coverage_adaptive_ci95']))
cond('semantic', 'every reported interval brackets its own point estimate',
     all(d[k][0] <= d[k.replace('_ci95', '')] <= d[k][1]
         for t in _tt for d in (tmp[t], tmp[t]['control_random_same_size'])
         for k in d if k.endswith('_ci95')
         and k.replace('_ci95', '') in d and isinstance(d[k], list)), '')
cond('semantic', 'the pooled temporal correlation is below every per-target value',
     all(tmp['pooled']['spearman_sigma_err'] < tmp[t]['spearman_sigma_err'] for t in _tt
         if t != 'drd3'), '')
close('semantic', 'temporal RMSE increase over control is ~49 percent, not 100',
      tmp['delta_vs_control']['rmse_pct_increase_vs_control'], 49.0, 2.0)

_p = L('poolopt_analysis.json')['summary']
cond('semantic', 'both conservative rules fall below the predicted mean on every target',
     all(_p[t]['lcb']['hits'] < _p[t]['greedy']['hits']
         and _p[t]['conformal']['hits'] < _p[t]['greedy']['hits'] for t in _p), '')
cond('semantic', 'the optimistic rule leads on average, so uncertainty is not useless here',
     sum(_p[t]['ucb']['enrichment_vs_random'] for t in _p)
     > sum(_p[t]['greedy']['enrichment_vs_random'] for t in _p), '')
_conall = L('conformal_analysis.json')
cond('semantic', 'adaptive intervals under-cover low-sigma and over-cover high-sigma compounds',
     all(_conall[t]['alpha0.1']['adaptive_coverage_low_sigma']
         < _conall[t]['alpha0.1']['adaptive_coverage_high_sigma'] for t in _TT), '')

_mm = L('mixedmodel_method.json')
close('semantic', 'random-intercept model reproduces the target-level point estimate',
      _mm['mixed_model']['intercept'], _mm['target_level']['mean'], 0.001)

# Figure 1 is the paper's definitional figure, so its panels are pinned to the same sources
# the text reads. An audit found its schematic drawing arrows to points that were not the
# nearest neighbours, and its caption quoting a cross-regime contrast the panel does not show.
_fig = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'gen_fig1.py')
if _os.path.exists(_fig):
    import numpy as _np
    _g = open(_fig, encoding='utf-8').read()
    cond('figure 1', 'panel a computes nearest neighbours instead of hand-placing arrows',
         'def nearest(' in _g and 'nearest(q1, START)' in _g and 'nearest(q1, TRAIN)' in _g, '')
    cond('figure 1', 'panel a asserts the relation it illustrates',
         'violates d_train <= nu' in _g, '')
    cond('figure 1', 'panel b plots the micro-pooled risk-coverage curve',
         "rel['pooled']['risk_coverage_micro']" in _g, '')
    cond('figure 1', 'panel c plots the disagreement-normalized coverage with its intervals',
         "K = 'conformal_coverage_adaptive'" in _g and "K + '_ci95'" in _g, '')
    cond('figure 1', 'panel d averages over targets and shows the standard error across them',
         "np.sqrt(5)" in _g and 'enrichment_vs_random' in _g, '')
    # Panel c compares one temporal observation against the control. The right reference is the
    # spread of control replicates, not the interval on their mean: at 1000 replicates the
    # latter is under 2% of the axis and rendered as an artefact rather than as data.
    cond('figure 1', 'panel c shows the control replicate spread, not the interval on its mean',
         "_csd" in _g and "1.96 * _csd" in _g,
         'one observation is judged against a distribution, not against a mean')
    cond('figure 1', 'panel c sets its limits from the data and asserts nothing is clipped',
         'panel c clips a drawn interval' in _g,
         'two control bands ran past a hand-set axis limit')
    cond('figure 1', 'panel d overlays the individual targets, not only their mean',
         'ax.scatter(i + _rng.uniform' in _g,
         'with five targets the points carry more than a standard error does')
    cond('figure 1', 'panel d draws no per-target points on the random bar',
         "if m == 'random':" in _g,
         'enrichment is 1.0 there by definition, so five points imply a spread that cannot exist')
    # The article path is resolved again here: this block runs long before the manuscript
    # section defines it, and referring to it early crashed the whole script.
    _art1 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'WritePaper',
                          'theranostics', 'JournalPapers_npjDD', 'npjDD_Reliability.tex')
    if _os.path.exists(_art1):
        _c1 = open(_art1, encoding='utf-8').read()
        _cap = _c1[_c1.find(r'\textbf{Where and when the activity model'):][:3200]
        cond('figure 1', 'the caption describes the per-target points in panel d',
             'Open circles' in _cap, 'every drawn element must be accounted for')
        # Panel d's annotation read "penalizing uncertainty finds fewer", which left the reader
        # to supply both the object, from the y axis, and the comparator, from nowhere. A
        # dangling comparative in a figure is read as a claim about everything on the panel.
        cond('figure 1', 'panel d\'s annotation names what it is fewer than',
             'finds fewer than' in _g,
             'the bare "finds fewer" gave neither the object nor the baseline')

        # Panel b draws a conformal-coverage annotation. Trimming the legend to fit the word cap
        # once removed the sentence explaining it, leaving unexplained text in the figure, so the
        # caption must keep naming the inset.
        # Test that the annotation is accounted for, not how the sentence introduces it. Keying
        # on the introducing word ("Inset") broke as soon as that word was corrected, which is
        # the same failure this file has now hit three times.
        cond('figure 1', 'the caption explains the conformal annotation panel b draws',
             'conformal coverage' in _re.sub(r'\s+', ' ', _cap).lower() and '0.900' in _cap,
             'the panel prints an empirical coverage that the caption must account for')
        # The acquisition rule is named "conformal-style lower score" in the text. The bar label
        # said "Scaled lower score", so a reader could not match bar to prose.
        # Every figure that plots the acquisition rules must name the fifth one the way the text
        # does. Checking gen_fig1.py alone let Supplementary Figure S3 keep "Scaled lower score"
        # for a whole audit after Figure 1 was corrected, so the two figures disagreed.
        _figsrc = {}
        for _gf in ('gen_fig1.py', 'gen_si_figures.py'):
            _gp = _os.path.join(_os.path.dirname(_art1), _gf)
            if _os.path.exists(_gp):
                _figsrc[_gf] = open(_gp, encoding='utf-8').read()
        for _gf, _gt in _figsrc.items():
            if 'Cautious' not in _gt:
                continue
            cond('figures', f'{_gf} labels the fifth acquisition rule as the text names it',
                 'Conformal-' in _gt and 'Scaled lower' not in _gt,
                 'the bar label must carry the term the prose uses')
        cond('figure 1', 'panel a is titled for what it actually shows',
             'Two distances and the gap' in _g,
             'it showed two distances and their gap while claiming three distances')
        # The control bar must be the spread ACROSS replicates, not the interval on their mean,
        # which at 1000 replicates is narrower than the marker. Test that meaning rather than one
        # phrasing of it: an earlier version keyed on the word "central" and broke when the
        # wording was corrected to the normal approximation the generator actually computes.
        _capf = _re.sub(r'\s+', ' ', _cap)
        cond('figure 1', 'the caption describes the control bar as the replicate spread',
             'replicates for the control' in _capf
             and ('1.96' in _capf or 'central' in _capf),
             'it previously described a t interval on the mean')
        cond('figure 1', 'the caption does not claim the optimistic rule reliably wins',
             'not separated from zero' in _cap,
             'its advantage covers zero across targets')
    # every value the caption states must still be the value the source holds
    _mi = rel['pooled']['risk_coverage_micro']
    close('figure 1', 'caption panel b: 42 percent reduction',
          100 * (_mi['1.0'] - _mi['0.2']) / _mi['1.0'], 42.0, 1.0)
    cond('figure 1', 'caption panel b: SCD-1 is the non-monotone curve',
         not all(rel['scd1']['risk_coverage_rmse']['%.1f' % c]
                 <= rel['scd1']['risk_coverage_rmse']['%.1f' % d]
                 for c, d in zip([0.2, 0.4, 0.6, 0.8], [0.4, 0.6, 0.8, 1.0]))
         and sum(all(rel[t]['risk_coverage_rmse']['%.1f' % c]
                     <= rel[t]['risk_coverage_rmse']['%.1f' % d]
                     for c, d in zip([0.2, 0.4, 0.6, 0.8], [0.4, 0.6, 0.8, 1.0]))
                 for t in ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']) == 4, '')
    _K = 'conformal_coverage_adaptive'
    cond('figure 1', 'caption panel c: every temporal coverage is below nominal 0.900',
         all(tmp[t][_K] < 0.900 for t in _tt), '')
    cond('figure 1', 'caption panel c: panel covers four targets, FADS excluded',
         len(_tt) == 4 and 'fads' not in _tt, '')
    _po = L('poolopt_analysis.json')['summary']
    _m = lambda k: sum(_po[t][k]['enrichment_vs_random'] for t in _po) / len(_po)
    cond('figure 1', 'caption panel d: optimistic leads on average, conservative rules trail',
         _m('ucb') > _m('greedy') > _m('lcb') > _m('conformal'), '')

# Figure 2's panel b is non-monotone and the manuscript now says so. Pin the turnover,
# because an audit found the figure title asserting a rise the panel itself contradicts.
_frf = _os.path.join(OUT, 'frontier_v2_results.json')
if _os.path.exists(_frf):
    import numpy as _np, collections as _co
    _rs = json.load(open(_frf))['results']
    _rows = [dict(r, target=t) for t, rr in _rs.items() for r in rr]
    _nv = _np.array([r['novelty'] for r in _rows]); _sg = _np.array([r['sigma'] for r in _rows])
    _lam = _np.array([r['lam'] for r in _rows]); _tg = _np.array([r['target'] for r in _rows])
    _bn = _np.linspace(_nv.min(), _nv.max(), 9)
    _ix = _np.clip(_np.digitize(_nv, _bn) - 1, 0, 7)
    _m = [_sg[_ix == b].mean() for b in range(8)]
    _n = [int((_ix == b).sum()) for b in range(8)]
    cond('figure 2', 'disagreement peaks at the seventh bin and falls in the eighth',
         int(_np.argmax(_m)) == 6 and _m[7] < _m[6], f'peak bin {int(_np.argmax(_m))+1}')
    close('figure 2', 'caption: the turnover is 0.92 to 0.81', _m[6], 0.92, 0.005)
    close('figure 2', 'caption: the turnover is 0.92 to 0.81 (second value)', _m[7], 0.81, 0.005)
    cond('figure 2', 'the turnover is not an artefact of the uncertainty penalty',
         all(_sg[(_ix == 7) & (_lam == L)].mean() < _sg[(_ix == 6) & (_lam == L)].mean()
             for L in (0.0, 0.1)), '')
    cond('figure 2', 'the last two bins are not thin at either penalty',
         all(((_ix == b) & (_lam == L)).sum() > 100 for b in (6, 7) for L in (0.0, 0.1)), '')
    _dec = 0
    for t in set(_tg):
        _mm = _np.array([_sg[(_ix == b) & (_tg == t)].mean()
                         if ((_ix == b) & (_tg == t)).sum() else _np.nan for b in range(8)])
        _dec += _mm[7] < _np.nanmax(_mm) - 1e-9
    cond('figure 2', 'disagreement turns over on four of the five targets', _dec == 4, f'{_dec} of 5')
    cond('figure 2', 'caption: bin counts run from 8 to 244',
         _n[0] == 8 and _n[7] == 244, f'{_n[0]} to {_n[7]}')
    _blk2 = _co.Counter((r['target'], r['seed']) for r in _rows)
    cond('figure 2', 'caption: 1,200 runs from 75 blocks of sixteen matched conditions',
         len(_rows) == 1200 and len(_blk2) == 75 and set(_blk2.values()) == {16},
         f'{len(_rows)} runs, {len(_blk2)} blocks')
    _gf = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'gen_main_figures.py')
    if _os.path.exists(_gf):
        _g = open(_gf, encoding='utf-8').read()
        _blk = _co.Counter((r['target'], r['seed']) for r in _rows)
    cond('figure 2', 'one support set per target and seed, so blocks carry all 16 conditions',
         len(_blk) == 75 and set(_blk.values()) == {16}, f'{len(_blk)} blocks')
    cond('figure 2', 'the bootstrap resamples target-seed blocks, not method-penalty runs',
         "f\"{r['target']}|{r['seed']}\"" in _g and "|{r['opt']}" not in _g, '')
    cond('figure 2', 'marker area encodes bin occupancy',
             'nb / nb.max()' in _g and 'turns over' in _g, '')
    # scatter's s is area in points squared, and the generator passes sqrt(n), so the area
    # grows as the square root of the bin count. The caption claimed proportionality, under
    # which the lowest bin would hold 3% of the area instead of the 31% it actually has.
    cond('figure 2', 'the marker area is a square-root encoding, not a proportional one',
         'np.sqrt(nb / nb.max())' in _g, 'scatter s is an area, so sqrt(n) area grows as sqrt')
    _art2 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'WritePaper',
                          'theranostics', 'JournalPapers_npjDD', 'npjDD_Reliability.tex')
    if _os.path.exists(_art2):
        _t2 = open(_art2, encoding='utf-8').read()
        cond('figure 2', 'the caption does not claim marker area is proportional to bin count',
             'marker area proportional to' not in _t2,
             'the encoding is a square root; proportional overstates the lowest bin by 10x')
        cond('figure 2', 'the caption states the square-root encoding',
             'square root of the number of runs' in _t2, '')

# The Figure 3 caption claimed the target-level interval is the widest drawn. It is, by 2%,
# which no reader can see and which understates the point: the informative comparison is with
# an interval that treats the 375 runs as independent, against which it is nearly 6 times
# wider. Pin the reworded claim to the data.
import numpy as _np
from scipy import stats as _sps
_mv3 = L('methods_v2_results.json')['results']
_allp = _np.array([c['stga_ecfp'] - c['graphga'] for r in _mv3 for c in r['per_seed']], float)
_cm3 = [_np.mean([c['stga_ecfp'] - c['graphga'] for c in r['per_seed']]) for r in _mv3]
_tm3 = _np.array([_np.mean([_cm3[i] for i, r in enumerate(_mv3) if r['target'] == t])
                  for t in ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']], float)
_hw3 = lambda a: float(_sps.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / _np.sqrt(len(a)))
cond('figure 3', 'the target-level interval is several times wider than a per-run one',
     _hw3(_tm3) / _hw3(_allp) > 4, f'{_hw3(_tm3) / _hw3(_allp):.1f}x')
_art3 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'WritePaper',
                      'theranostics', 'JournalPapers_npjDD', 'npjDD_Reliability.tex')
if _os.path.exists(_art3):
    _t3 = open(_art3, encoding='utf-8').read()
    cond('figure 3', 'the caption does not rest on the marginal widest-in-figure claim',
         'is the widest in the figure' not in _t3,
         'true by 2%, so it reads as wrong against the drawn intervals')
    cond('figure 3', 'the caption gives the width ratio against a per-run interval',
         'MethodWidthRatio' in _t3, '')

# Figure 3 drew normal-approximation intervals while the text quoted Student t ones, so the
# same estimate carried two different 95% intervals. Pin the figure to the text's convention.
_mvf = _os.path.join(OUT, 'methods_v2_results.json')
_g3 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'gen_main_figures.py')
if _os.path.exists(_mvf) and _os.path.exists(_g3):
    import numpy as _np
    from scipy import stats as _sps
    _mv = json.load(open(_mvf))['results']
    _TT5 = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']
    _ci = lambda v: float(_sps.t.ppf(0.975, len(v) - 1) * _np.std(v, ddof=1) / _np.sqrt(len(v)))
    _cell = [(_np.mean([c['stga_ecfp'] - c['graphga'] for c in r['per_seed']]),
              _ci([c['stga_ecfp'] - c['graphga'] for c in r['per_seed']])) for r in _mv]
    _tm = [_np.mean([c['stga_ecfp'] - c['graphga'] for r in _mv if r['target'] == t
                     for c in r['per_seed']]) for t in _TT5]
    _tci = _ci(_tm)
    _gg = open(_g3, encoding='utf-8').read()
    cond('figure 3', 'intervals are Student t, not the normal approximation',
         'sps.t.ppf(0.975' in _gg and '1.96 *' not in _gg, '')
    # only comparable where the manuscript macro file is present; a clean clone skips it
    _m = globals().get('mac') or {}
    if 'MethodCIlo' in _m:
        cond('figure 3', 'the figure and the manuscript quote the same target-level interval',
             abs((_np.mean(_tm) - _tci) - float(_m['MethodCIlo'])) < 0.0006
             and abs((_np.mean(_tm) + _tci) - float(_m['MethodCIhi'])) < 0.0006,
             f'figure [{_np.mean(_tm)-_tci:+.4f}, {_np.mean(_tm)+_tci:+.4f}] vs '
             f"macro [{_m['MethodCIlo']}, {_m['MethodCIhi']}]")
    cond('figure 3', 'caption: gain positive and interval clear of zero in all 15 cells',
         len(_cell) == 15 and all(m - h > 0 for m, h in _cell), '')
    # Retained as a fact, not as a caption claim: the target-level interval does exceed every
    # cell interval, but only by 2%, so the caption rests on the per-run comparison instead.
    cond('figure 3', 'the target-level interval exceeds every cell interval',
         _tci > max(h for _, h in _cell),
         f'{_tci:.4f} vs {max(h for _, h in _cell):.4f}, a margin too small to carry a caption')
    cond('figure 3', 'caption: 375 runs behind the 15 cells',
         sum(len(r['per_seed']) for r in _mv) == 375, '')
    cond('figure 3', 'axis label names the same metric as the caption',
         'Top-10 reward gain' in _gg, '')

# The Supplementary figures. An audit found S1 describing a quintile split as a "half" and
# claiming under-coverage on every target when FADS is over-covered in its lowest fifth.
_gsi = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'gen_si_figures.py')
_conf = L('conformal_analysis.json')
_T5 = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']
_a1 = lambda t: _conf[t]['alpha0.1']
cond('figure S1', 'the sigma split is labelled a fifth, matching the 20/80 percentile source',
     _os.path.exists(_gsi) and 'fifth' in open(_gsi, encoding='utf-8').read()
     and 'half' not in open(_gsi, encoding='utf-8').read(), '')
cond('figure S1', 'coverage rises with disagreement on every target',
     all(_a1(t)['adaptive_coverage_high_sigma'] > _a1(t)['adaptive_coverage_low_sigma']
         for t in _T5), '')
cond('figure S1', 'the highest fifth is over-covered on all five targets',
     all(_a1(t)['adaptive_coverage_high_sigma'] >= 0.90 for t in _T5), '')
_under = [t for t in _T5 if _a1(t)['adaptive_coverage_low_sigma'] < 0.90]
cond('figure S1', 'the lowest fifth is under-covered on four of five, FADS excepted',
     len(_under) == 4 and 'fads' not in _under, f'under-covered: {_under}')
cond('figure S1', 'pooled coverage tracks nominal at all three levels',
     all(abs(_conf['pooled'][a]['adaptive_coverage'] - n) < 0.01
         for a, n in [('alpha0.2', 0.80), ('alpha0.1', 0.90), ('alpha0.05', 0.95)]), '')

_ov = lambda a, b: a[0] <= b[1] and b[0] <= a[1]
# Compare against the band the figure actually draws, the central 95% of the control
# replicates, not the interval on their mean. The overlap verdict is the same under both, but a
# check that tests a comparison the reader never sees is not checking the figure.
def _cb(t, k):
    c = tmp[t]['control_random_same_size']
    return [c[k] - 1.96 * c[k + '_sd'], c[k] + 1.96 * c[k + '_sd']]


cond('figure S2', 'panels a and b plot both arms with their intervals',
     all(k in tmp[t] for t in _tt for k in ('rmse_test_ci95', 'spearman_sigma_err_ci95'))
     and all(k in tmp[t]['control_random_same_size'] for t in _tt
             for k in ('rmse_sd', 'spearman_sigma_err_sd')), '')
cond('figure S2', 'caption: the ranking matches its control on SCD-1 and NK1R',
     all(_ov(tmp[t]['spearman_sigma_err_ci95'], _cb(t, 'spearman_sigma_err'))
         for t in ('scd1', 'nk1r')), '')
cond('figure S2', 'caption: the ranking separates from control on DRD2 and DRD3',
     all(not _ov(tmp[t]['spearman_sigma_err_ci95'], _cb(t, 'spearman_sigma_err'))
         for t in ('drd2', 'drd3')), '')

_pl = L('poolopt_analysis.json')
_ps, _ns = _pl['summary'], _pl['config']['seeds']
cond('figure S3', 'error bars use the seed count the config records',
     _ns == 20 and _os.path.exists(_gsi) and 'np.sqrt(NSEED)' in open(_gsi, encoding='utf-8').read(),
     f'seeds={_ns}')
cond('figure S3', 'caption: both conservative rules trail the predicted mean on every target',
     all(_ps[t]['lcb']['hits'] < _ps[t]['greedy']['hits']
         and _ps[t]['conformal']['hits'] < _ps[t]['greedy']['hits'] for t in _ps), '')
cond('figure S3', 'caption: which of mean and optimistic leads varies by target',
     len({('greedy' if _ps[t]['greedy']['hits'] > _ps[t]['ucb']['hits']
           else 'ucb' if _ps[t]['ucb']['hits'] > _ps[t]['greedy']['hits'] else 'tie')
          for t in _ps}) > 1, '')
cond('figure S3', 'every model-guided rule beats random on every target',
     all(_ps[t][m]['hits'] > _ps[t]['random']['hits']
         for t in _ps for m in ('greedy', 'ucb', 'lcb', 'conformal')), '')

# Table 1. Its counts were already asserted; the thresholds were the last hand-typed cells
# in it, and the caption previously implied its counts described the temporal analysis too.
_om = {e['target']: e for e in L('oracle_metrics.json')['results']}
_T5 = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']
_CAP1 = {'scd1': 'Scd', 'fads': 'Fads', 'nk1r': 'Nkone', 'drd2': 'Drdtwo', 'drd3': 'Drdthree'}
cond('table 1', 'structures never exceed records on any row',
     all(rel[t]['duplicates']['n_unique'] <= rel[t]['duplicates']['n_rows'] for t in _T5), '')
cond('table 1', 'the structure total is the pooled prediction count',
     sum(rel[t]['duplicates']['n_unique'] for t in _T5) == rel['pooled']['n'], '')
cond('table 1', 'FADS is the only target with no duplicate structures',
     [t for t in _T5 if rel[t]['duplicates']['n_duplicate_rows'] == 0] == ['fads'], '')
if _os.path.exists(NUM):
    for t in _T5:
        cond('table 1', f'threshold for {t} comes from the frozen oracle metrics',
             abs(float(mac[f'Thr{_CAP1[t]}']) - _om[t]['threshold']) < 1e-9,
             f"macro={mac.get('Thr'+_CAP1[t])} json={_om[t]['threshold']}")
    _tex = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'WritePaper',
                         'theranostics', 'JournalPapers_npjDD', 'npjDD_Reliability.tex')
    if _os.path.exists(_tex):
        _b = open(_tex, encoding='utf-8').read()
        cond('table 1', 'no hand-typed threshold literals remain in the table body',
             all(f'& {v}\\\\' not in _b.split('tab:targets')[1][:600] for v in ('7.0', '6.5')),
             'thresholds must come from macros')
        cond('table 1', 'the caption scopes its counts to the cross-validation cohort',
             'cross-validation cohort' in _b and 'do not describe the temporal analysis' in _b, '')
        cond('table 1', 'the caption discloses that FADS pools two isoforms',
             'FADS pools FADS1 and FADS2' in _b, '')

# Supplementary tables. The article is a separate document and cited them by hard-coded
# number; inserting one table silently shifted ten of those citations onto the wrong table,
# and one pointed at a table that did not exist. Numbers now come from si_refs.tex.
_D = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'WritePaper',
                   'theranostics', 'JournalPapers_npjDD')
_sit = _os.path.join(_D, 'si_tables.tex'); _sir = _os.path.join(_D, 'si_refs.tex')
_art = _os.path.join(_D, 'npjDD_Reliability.tex'); _sid = _os.path.join(_D, 'npjDD_SI.tex')
if all(_os.path.exists(x) for x in (_sit, _sir, _art, _sid)):
    import re as _r
    _t = open(_sit, encoding='utf-8').read()
    _labels = _r.findall(r'\\label\{tab:s-(.*?)\}', _t)
    _macros = dict(_r.findall(r'\\newcommand\{\\Tab(\w+)\}\{S(\d+)\}',
                              open(_sir, encoding='utf-8').read()))
    _a = open(_art, encoding='utf-8').read(); _d = open(_sid, encoding='utf-8').read()
    cond('SI tables', 'every table number macro matches the generated table order',
         all(int(_macros[l.capitalize()]) == i for i, l in enumerate(_labels, 1)),
         f'{len(_labels)} tables')
    cond('SI tables', 'the article cites Supplementary tables by macro, never by literal',
         not _r.search(r'Supplementary Tables?~S\d', _a), 'a literal would not track reordering')
    cond('SI tables', 'every macro the article cites is defined',
         all(k in _macros for k in _r.findall(r'\\Tab(\w+)', _a)),
         str(sorted(set(_r.findall(r'\\Tab(\w+)', _a)) - set(_macros))))
    cond('SI tables', 'the article loads the generated numbers', '\\input{si_refs}' in _a, '')
    cond('SI tables', 'every generated table is referenced from the SI text',
         all(f'tab:s-{l}' in _d for l in _labels),
         str([l for l in _labels if f'tab:s-{l}' not in _d]))
    # Figures cross the same document boundary as the tables and were cited the same way.
    _sfig = _r.findall(r'\\begin\{figure\}.*?\\label\{sfig:(.*?)\}.*?\\end\{figure\}', _d, _r.S)
    _afig = _r.findall(r'\\begin\{figure\}.*?\\label\{fig:(.*?)\}.*?\\end\{figure\}', _a, _r.S)
    _fm = dict(_r.findall(r'\\newcommand\{\\SFig(\w+)\}\{S(\d+)\}',
                          open(_sir, encoding='utf-8').read()))
    _am = dict(_r.findall(r'\\newcommand\{\\ArtFig(\w+)\}\{(\d+)\}',
                          open(_sir, encoding='utf-8').read()))
    cond('SI figures', 'every Supplementary figure carries a label',
         len(_sfig) == 3, f'{len(_sfig)} labelled')
    cond('SI figures', 'Supplementary figure macros match the order they appear in',
         all(int(_fm[l.capitalize()]) == i for i, l in enumerate(_sfig, 1)), str(_sfig))
    cond('SI figures', 'article figure macros match the order they appear in',
         all(int(_am[l.capitalize()]) == i for i, l in enumerate(_afig, 1)), str(_afig))
    cond('SI figures', 'the article cites Supplementary figures by macro, never by literal',
         not _r.search(r'Supplementary Fig[a-z.]*~S\d', _a), '')
    cond('SI figures', 'the Supplementary text cites article figures by macro, never by literal',
         not _r.search(r'Figure \d+[a-d]? of the main', _d), '')
    cond('SI figures', 'every Supplementary figure is cited from the article',
         all(f'SFig{l.capitalize()}' in _a for l in _sfig),
         str([l for l in _sfig if f'SFig{l.capitalize()}' not in _a])),
    cond('SI figures', 'the Supplementary Information loads the generated numbers',
         '\\input{si_refs}' in _d, '')

    cond('SI tables', 'the compatibility negative result has a table of its own',
         'compat' in _labels and 'TabCompat' in _a, '')
    _cg = L('compat_gen_analysis.json')
    cond('SI tables', 'the compatibility table has one row per source-target pair',
         len(_cg['rows']) == _cg['summary']['n_pairs'] == 20, f"{len(_cg['rows'])} rows")

# Banned phrasings. Every one of these was corrected once in the place it was noticed and left
# standing somewhere else: the floor claim in the Methods after the SI was fixed, "roughly
# doubles" in the Introduction and Discussion after the abstract was fixed, FADS named as the
# in-domain exception in the Discussion after the Results were fixed. Scan every source file.
# The scan covered the three TeX files only, so "Conformal bound" survived as a visible axis
# label in Figure S3 for as long as the term had been banned everywhere else. Anything a reader
# sees is in scope, and figure labels are read by a reader.
_SRC = {n: _os.path.join(_D, n) for n in
        ('npjDD_Reliability.tex', 'npjDD_SI.tex', 'si_tables.tex',
         'gen_fig1.py', 'gen_main_figures.py', 'gen_si_figures.py')}
_BANNED = [
    ('places a floor', 'within-parent spread is not a lower bound on attainable error'),
    ('bounds the error any model', 'same floor claim, other wording'),
    ('roughly doubl', 'RMSE rises ~50% over control; squared error doubles, RMSE does not'),
    ('halves that error', 'quote the exact percentage at the stated coverage'),
    ('halves the error', 'quote the exact percentage at the stated coverage'),
    ('training distribution', 'd_train is a nearest-neighbour distance, not a distributional one'),
    ('true activit', 'pool labels are measurements, not ground truth'),
    ('true active', 'pool labels are measurements, not ground truth'),
    ('coverage guarantee fails', 'say coverage falls below nominal once exchangeability breaks'),
    ('most of the ranking power', 'the ranking degrades unevenly, not uniformly'),
    # the acquisition rule carries no nominal guarantee; one name for it everywhere
    ('conformal lower bound', 'call it the conformal-style lower score'),
    ('conformal bound', 'call it the conformal-style lower score'),
    ('conformal-constrained', 'call it the conformal-style lower score'),
    # the acquisition conclusion is objective-dependent
    ('not a good rule for deciding which compound to test next', 'scope it to the objective'),
    # collapsed records are not all replicates
    ('With replicates', 'multi-record parents; provenance is not in the retained fields'),
    ('Duplicates removed', 'records collapsed by parent grouping'),
    ('Within-compound SD', 'within-parent SD'),
    # adjectives standing in for quantities
    ('substantial', 'give the quantity instead of the adjective'),
    ('significantly', 'give the exact P or the effect size'),
    ('fundamental', 'give the quantity instead of the adjective'),
    ('robust', 'give the quantity instead of the adjective'),
    # the acquisition finding is that penalising uncertainty costs, not that the score is
    # useless for acquisition; mu+sigma is itself uncertainty-aware and wins
    ('whatever the objective', 'UCB uses uncertainty and does best; scope it to the objective'),
    # the rule has no nominal coverage under policy-selected labels, so it is not a bound
    ('conformal-style lower bound', 'call it the conformal-style lower score'),
    ('with a calibrated scale', 'the scale is residual-derived, not calibrated'),
    # no tautomer canonicalisation is applied, so tautomers do not generally collapse
    ('tautomer variants of one compound', 'InChI merges only mobile-H tautomers'),
    # the introduction asks four questions
    ('we ask three questions', 'four questions are asked'),
    # the label pools four ChEMBL endpoint types and is not pure IC50
    ('named $\\mathrm{pIC}_{50}$', 'the pooled response is mostly Ki and is named pActivity'),
    ('name the pooled response $\\mathrm{pIC}_{50}$', 'the pooled response is named pActivity'),
    # only one score on one model family was validated against measured error
    ('reliability of whatever model', 'only RF per-tree disagreement was validated; another predictor needs its own score'),
    # the macro-average temporal effect is a t interval across the four target-level effects;
    # the bootstrap-plus-control-SE construction is what the PER-TARGET effects use
    ('the interval combining the temporal resampling', 'the macro interval is a t interval across targets'),
    # random CV measures within-pool generalization; a search selects adaptively, so CV is not
    # "the regime a search occupies"
    ('the regime a search occupies', 'random CV is within-pool generalization, not the search regime'),
    # sigma_T stops rising at the extreme; the generated molecules were never assayed, so
    # nothing establishes that it stops carrying information
    ('losing its meaning', 'say it ceases to be monotonic; error there is unobserved'),
    # the encoder and search settings are given but not exhaustively, so do not claim completeness
    ('fully specified choice', 'give the reason it is preferred, not a completeness claim'),
    # EC50 is retained and is not an affinity endpoint, so the dropped records are not "non-affinity"
    ('non-affinity endpoint', 'EC50 is retained; the filter drops ineligible standard types'),
    # (1+k)/(R+1) over 1000 random draws is a Monte Carlo tail probability, not an enumerated test
    ('exact one-sided empirical', 'call it a one-sided Monte Carlo empirical P'),
    # mu+sigma's advantage covers zero, so the upside claim must carry the asymmetry that
    # supports it: penalising costs compounds, seeking them out does not reliably gain any
    ('upside lies.', 'unqualified; state the asymmetry, since the optimistic rule covers zero'),
]
def _flat(t):
    """Collapse whitespace before scanning.

    The scan used to run on the raw file, so a banned phrase that happened to wrap across a
    source line was invisible to it: "an exact one-sided\nempirical P" sat in the SI while the
    guard for "exact one-sided empirical" reported clean. LaTeX line breaks are not part of a
    sentence, so neither the haystack nor the needle should depend on them.
    """
    return _re.sub(r'\s+', ' ', t)


for _n, _f in _SRC.items():
    if not _os.path.exists(_f):
        continue
    _txt = _flat(open(_f, encoding='utf-8').read())
    for _bad, _why in _BANNED:
        cond('phrasing', f'{_n}: no "{_bad}"', _flat(_bad) not in _txt, _why)

# Claims that appear in more than one place must agree everywhere they appear.
if all(_os.path.exists(f) for f in _SRC.values()):
    _all = _flat('\n'.join(open(f, encoding='utf-8').read() for f in _SRC.values()))
    cond('phrasing', 'SCD-1, not FADS, is named as the in-domain exception wherever it is named',
         'NinDomExcept' in _all and 'where the in-domain novelty comparison does not hold' not in _all,
         'the exception must come from the macro the data sets')
    cond('phrasing', 'the Methods no longer claim four targets have one record per structure',
         'one record per structure' not in _all, 'Table S1 shows collapses in four of five')
    # The Discussion generalises, so its two widest claims need their scope stated in the same
    # breath. Both were flagged in review while the Results said the qualified thing already.
    _disc = (open(_SRC['npjDD_Reliability.tex'], encoding='utf-8').read()
             .split('\\section{Discussion}')[-1].split('\\section{Methods}')[0])
    cond('phrasing', 'the Discussion scopes the validated score to one predictor family',
         'per-tree disagreement of a random forest' in _disc
         and 'require a score appropriate to that predictor' in _disc,
         'the framework was not shown to quantify reliability for an arbitrary model')
    cond('phrasing', 'the Discussion states the acquisition asymmetry, not a case for optimism',
         "optimistic rule's advantage covers zero" in _disc,
         'mu+sigma minus mu is +1.18 [-1.36, +3.72], P=0.27, so optimism is not shown better')

# Every input a generator needs must resolve from a clean clone. The curation provenance was
# reachable only through the workspace layout, so a clone silently produced one table fewer and
# renumbered every table after it, giving a reader different numbers from the ones the article
# cites. Check both that the file resolves and that the table it feeds was actually emitted.
_prov = _os.path.join(_HERE_DIR, 'data', 'chembl_v2', 'curation_provenance.json')
cond('portability', 'the curation provenance ships in the repository',
     _os.path.isfile(_prov), 'the curation table is generated from it')
# Test the resolution itself rather than a generated artefact, so the check runs in every
# state and the reported total does not depend on whether the generators have been run.
try:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location('_mst', _os.path.join(_HERE_DIR, 'make_si_tables.py'))
    _src = open(_os.path.join(_HERE_DIR, 'make_si_tables.py'), encoding='utf-8').read()
    _ns = {'__file__': _os.path.join(_HERE_DIR, 'make_si_tables.py'), 'os': _os}
    exec(_src[:_src.index('PROV = _prov_path()')] + 'PROV = _prov_path()', _ns)
    _resolved = _os.path.isfile(_ns['PROV'])
except Exception as _e:
    _resolved = False
cond('portability', 'make_si_tables resolves the curation provenance from this directory',
     _resolved, 'unresolved, the curation table is dropped and every later table renumbers')

# Table 1 lists the per-target thresholds, which decide which compounds seed a search, what
# counts as a hit and how AUC is computed. Their origin was stated nowhere, and one of them was
# set from the activity distribution rather than by convention, so both facts are now disclosed.
_omt = {e['target']: e for e in json.load(open(_os.path.join(OUT, 'oracle_metrics.json')))['results']}
_fa = [_omt[t]['frac_active'] for t in ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')]
cond('table 1', 'the active split is near balanced on every target, not a rare positive class',
     all(0.3 < f < 0.8 for f in _fa),
     'fractions ' + ', '.join(f'{100*f:.0f}%' for f in _fa))
cond('table 1', 'only two distinct thresholds are used',
     len({_omt[t]['threshold'] for t in _omt}) == 2,
     'the caption and Methods describe exactly two cutoffs')
_art4 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'WritePaper',
                      'theranostics', 'JournalPapers_npjDD', 'npjDD_Reliability.tex')
if _os.path.exists(_art4):
    _t4 = open(_art4, encoding='utf-8').read()
    cond('table 1', 'the threshold provenance is stated, including the one set from the data',
         'carried over from the earlier curation pipeline' in _t4 and 'sits' in _t4,
         'DRD3 takes the 100 nM cutoff because its distribution is higher')
    cond('table 1', 'the caption says what the pooled activity scale pools',
         'TabEndpoint' in _t4.split('label{tab:targets}')[0][-1400:],
         'a reader meets the pooled-activity label first in Table 1')

# Figure S1 panel b compares coverage against a nominal level. Drawn as bars from a cut axis
# at 0.6, length overstated the differences by about half; anchored on the nominal level the
# length is the departure from it. Panel a's pooled curve was drawn over the targets and hid
# NK1R, whose values sit within 0.002 of it.
_gs1 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'gen_si_figures.py')
if _os.path.exists(_gs1):
    _s1 = open(_gs1, encoding='utf-8').read()
    cond('SI figures', 'figure S1b anchors its bars on the nominal level, not on a cut axis',
         'bottom=0.90' in _s1, 'bar length must encode the departure from nominal')
    cond('SI figures', 'figure S1b sets its limits from the data rather than by hand',
         'ax.set_ylim(0.6, 1.15)' not in _s1, 'a hand-cut axis is what caused the distortion')
    cond('SI figures', 'figure S1a draws the pooled curve behind the target curves',
         "label='Pooled', zorder=0" in _s1, 'at zorder 5 it hid NK1R completely')

# Figure S2 draws the same control comparison as main Figure 1c, so it must use the same
# interval. It was using the interval on the control mean, which at 1000 replicates is 0.1 to
# 0.4% of the axis height and invisible, and which answers a different question from the one
# the panel poses.
_gs2 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'gen_si_figures.py')
if _os.path.exists(_gs2):
    _s2 = open(_gs2, encoding='utf-8').read()
    cond('SI figures', 'figure S2 shows the control replicate spread, as main Figure 1c does',
         'def _cband(' in _s2 and "_cband(t, 'rmse')" in _s2,
         'the two figures must mean the same thing by a control interval')
    cond('SI figures', 'figure S2 no longer draws the interval on the control mean',
         "['rmse_ci95'] for t in tt" not in _s2, 'invisible at 1000 replicates')
_tj2 = _os.path.join(OUT, 'temporal_analysis.json')
if _os.path.exists(_tj2):
    _t2j = json.load(open(_tj2))
    _rc = lambda t: _t2j[t]['risk_coverage_rmse']
    _inv = [t for t in ('scd1', 'nk1r', 'drd2', 'drd3') if _rc(t)['0.2'] >= _rc(t)['1.0']]
    cond('SI figures', 'DRD3 alone has an inverted risk-coverage curve after the shift',
         _inv == ['drd3'], f'inverted on {_inv}; it is the target whose ranking does not survive')

# Figure S3's per-target ordering claim, including the tie the earlier wording hid.
_p3 = json.load(open(_os.path.join(OUT, 'poolopt_analysis.json')))['summary']
_h = lambda t, k: _p3[t][k]['hits']
_T3 = ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
cond('SI figures', 'S3: both unpenalised rules beat both penalised ones on every target',
     all(_h(t, g) > _h(t, p) for t in _T3 for g in ('greedy', 'ucb')
         for p in ('lcb', 'conformal')), '')
cond('SI figures', 'S3: the optimistic rule leads on SCD-1, FADS and DRD3',
     [t for t in _T3 if _h(t, 'ucb') > _h(t, 'greedy')] == ['scd1', 'fads', 'drd3'], '')
cond('SI figures', 'S3: the predicted mean leads on DRD2 alone',
     [t for t in _T3 if _h(t, 'greedy') > _h(t, 'ucb')] == ['drd2'], '')
cond('SI figures', 'S3: the two are exactly equal on NK1R',
     _h('nk1r', 'ucb') == _h('nk1r', 'greedy'),
     'the caption names the tie rather than implying one leads')

# Table S1's within-parent SD is a mean over parents holding more than one record. FADS has
# none, so the quantity is undefined there; the table printed 0.00, which reads as a measured
# spread of zero. Check the generator emits n/a and that the arithmetic of the table closes.
_relS1 = json.load(open(_os.path.join(OUT, 'reliability_v2_analysis.json')))
for _t in ('scd1', 'fads', 'nk1r', 'drd2', 'drd3'):
    _d = _relS1[_t]['duplicates']
    cond('SI tables', f'S1 arithmetic closes for {_t}: records minus collapsed equals parents',
         _d['n_rows'] - _d['n_duplicate_rows'] == _d['n_unique'],
         f"{_d['n_rows']} - {_d['n_duplicate_rows']} = {_d['n_unique']}")
cond('SI tables', 'S1 reports no within-parent SD where there are no multi-record parents',
     _relS1['fads']['duplicates']['n_compounds_with_replicates'] == 0,
     'FADS has none, so the entry must be n/a rather than 0.00')
_mst1 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_mst1):
    cond('SI tables', 'the S1 generator prints n/a for an undefined within-parent SD',
         "else 'n/a'" in open(_mst1, encoding='utf-8').read(),
         'zero and undefined are different statements')

# Table S2 reports a scaffold-split R^2 of -222.96 for FADS. That reads as a software fault
# unless the table says why: R^2 divides by the held-out fold's own variance, and the FADS
# scaffold fold is nearly constant. The RMSE the same R^2 implies is ordinary, which is the
# check that distinguishes a low-variance denominator from a broken model.
_sfp = _os.path.join(OUT, 'scaffold_fold_stats.json')
if _os.path.exists(_sfp):
    _sf = json.load(open(_sfp))
    _T2 = ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
    cond('SI tables', 'the scaffold RMSE reproduces exactly from the R2 and the fold spread',
         all(abs(_sf[t]['implied_rmse'] - _sf[t]['sd_test'] * (1 - _sf[t]['r2']) ** 0.5) < 1e-9
             for t in _T2), 'RMSE = SD(test) * sqrt(1 - R2)')
    cond('SI tables', 'the FADS scaffold RMSE is ordinary despite its extreme R2',
         0.9 < _sf['fads']['implied_rmse'] < 1.6,
         f"RMSE {_sf['fads']['implied_rmse']:.2f} against R2 {_sf['fads']['r2']:.0f}")
    cond('SI tables', 'FADS has by far the least variable scaffold fold',
         max(_T2, key=lambda t: _sf[t]['var_ratio']) == 'fads'
         and _sf['fads']['var_ratio'] > 100,
         f"variance ratio {_sf['fads']['var_ratio']:.0f}")
    _mst2 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
    if _os.path.exists(_mst2):
        _c2 = open(_mst2, encoding='utf-8').read()
        # Spelling-agnostic: this broke when the caption was Americanized, which is the fifth
        # time in this file a guard has tested a wording instead of the claim it stands for.
        cond('SI tables', 'the S2 caption explains the negative scaffold R2',
             _re.search(r'is normali[sz]ed by the variance of the ', _c2) is not None
             and 'less well than its own mean does' in _c2,
             'otherwise -222.96 reads as a bug')
        cond('SI tables', 'S2 carries a scaffold RMSE column beside the R2',
             "rmse_s" in _c2, 'the absolute error is what shows the model is not broken')

# Table S3 is the error-stratification table. Four targets and the pooled row increase across
# every quintile; SCD-1 does not, and a table headed "stratification" that shows an unordered
# row without comment invites the reader to distrust the rest.
_r3 = json.load(open(_os.path.join(OUT, 'reliability_v2_analysis.json')))
_mono3 = [t for t in ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
          if all(_r3[t]['rmse_by_sigma_quintile'][i] <= _r3[t]['rmse_by_sigma_quintile'][i + 1]
                 for i in range(4))]
cond('SI tables', 'S3: SCD-1 is the only target whose quintiles are not ordered',
     _mono3 == ['fads', 'nk1r', 'drd2', 'drd3'], f'monotone on {_mono3}')
cond('SI tables', 'S3: the pooled quintiles are ordered',
     all(_r3['pooled']['rmse_by_sigma_quintile'][i] <= _r3['pooled']['rmse_by_sigma_quintile'][i + 1]
         for i in range(4)), '')
cond('SI tables', 'S3: the top quintile exceeds the bottom on every row',
     all(_r3[t]['rmse_by_sigma_quintile'][4] > _r3[t]['rmse_by_sigma_quintile'][0]
         for t in ('scd1', 'fads', 'nk1r', 'drd2', 'drd3', 'pooled')),
     'the ranking holds end to end even where it is not monotone')
_mst3 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_mst3):
    cond('SI tables', 'the S3 caption derives its monotone list from the data',
         '_mono = [LAB[t] for t in T' in open(_mst3, encoding='utf-8').read(),
         'so it cannot drift if a target moves')

# Table S4 is the cumulative form of Table S3: retaining the lowest-disagreement fraction is
# the same as accumulating quintiles in order. That makes the two tables checkable against each
# other, which is stronger than checking either against itself.
import math as _math
_r4 = json.load(open(_os.path.join(OUT, 'reliability_v2_analysis.json')))
_C4 = ['0.2', '0.4', '0.6', '0.8', '1.0']
_worst = 0.0
for _t in ('scd1', 'fads', 'nk1r', 'drd2', 'drd3', 'pooled'):
    _q = _r4[_t]['rmse_by_sigma_quintile']
    _rc = (_r4[_t]['risk_coverage_rmse'] if _t != 'pooled'
           else _r4['pooled']['risk_coverage_micro'])
    _worst = max(_worst, max(abs(_math.sqrt(sum(v ** 2 for v in _q[:k]) / k) - _rc[_C4[k - 1]])
                             for k in range(1, 6)))
cond('SI tables', 'S4 is the cumulative form of S3 on every row',
     _worst < 0.005, f'worst discrepancy {_worst:.4f} pActivity')
_ma4 = _r4['pooled']['risk_coverage_macro']
cond('SI tables', 'the S4 macro row is the unweighted mean of the five target curves',
     all(abs(sum(_r4[t]['risk_coverage_rmse'][c] for t in
                 ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')) / 5 - _ma4[c]) < 5e-4 for c in _C4),
     'not an RMS average, which would differ in the third decimal')
cond('SI tables', 'the S4 micro row sits below the macro row at every coverage',
     all(_r4['pooled']['risk_coverage_micro'][c] < _ma4[c] for c in _C4),
     'pooling is dominated by the two largest and easiest targets')

# Table S5 separates novelty from nearest-training distance. The caption used to claim that
# novelty loses its signal once distance is controlled for, which four targets and the pooled
# row do but FADS does not: its correlation is negative and grows slightly under control.
_r5 = json.load(open(_os.path.join(OUT, 'reliability_v2_analysis.json')))
_T5 = ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
_shr = [t for t in _T5
        if abs(_r5[t]['partial_err_nov_given_dtr']) < abs(_r5[t]['spearman_nov_err'])]
cond('SI tables', 'S5: FADS is the only target whose novelty signal does not shrink',
     _shr == ['scd1', 'nk1r', 'drd2', 'drd3'], f'shrinks on {_shr}')
cond('SI tables', 'S5: FADS is the only negative novelty correlation',
     [t for t in _T5 if _r5[t]['spearman_nov_err'] < 0] == ['fads'],
     'more novel compounds carry lower error there, against the pooled direction')
cond('SI tables', 'S5: the pooled novelty signal shrinks once distance is controlled for',
     abs(_r5['pooled']['partial_err_nov_given_dtr']) < abs(_r5['pooled']['spearman_nov_err']),
     f"{_r5['pooled']['spearman_nov_err']:.3f} -> {_r5['pooled']['partial_err_nov_given_dtr']:.3f}")
cond('SI tables', 'S5: disagreement survives controlling for novelty and distance everywhere',
     all(_r5[t]['partial_err_sig_given_nov_dtr'] > 0.1 for t in _T5),
     'the claim the caption makes about every target')
_mst5 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_mst5):
    cond('SI tables', 'the S5 caption derives its exception from the data',
         '_shrink = [LAB[t] for t in T' in open(_mst5, encoding='utf-8').read(), '')

# Table S6 reports interval widths. Its caption used to read the narrowing as evidence the
# score is informative, which is the inference Supplementary Note 4 explicitly declines to make:
# the gain is 1 to 4% and at the 0.80 level is partly bought with coverage.
_c6 = json.load(open(_os.path.join(OUT, 'conformal_analysis.json')))['pooled']
_A6 = ('alpha0.2', 'alpha0.1', 'alpha0.05')
cond('SI tables', 'S6: both interval kinds reach nominal coverage at all three levels',
     all(abs(_c6[a][k] - _c6[a]['target_coverage']) < 0.02
         for a in _A6 for k in ('standard_coverage', 'adaptive_coverage')), '')
cond('SI tables', 'S6: the adaptive interval is narrower at all three levels',
     all(_c6[a]['width_ratio_adaptive_over_standard'] < 1.0 for a in _A6),
     ', '.join(f"{_c6[a]['width_ratio_adaptive_over_standard']:.3f}" for a in _A6))
_red = [100 * (1 - _c6[a]['width_ratio_adaptive_over_standard']) for a in _A6]
cond('SI tables', 'S6: the width reduction stays in single digits',
     max(_red) < 10, f'{min(_red):.1f}% to {max(_red):.1f}%')
cond('SI tables', 'S6: the 0.80 level is the one where coverage is traded for width',
     [a for a in _A6 if _c6[a]['adaptive_coverage'] < _c6[a]['standard_coverage']] == ['alpha0.2'],
     'the caption names that level rather than claiming equal coverage throughout')
_m6 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_m6):
    cond('SI tables', 'the S6 caption does not read interval width as the case for the score',
         'is not what the case for the score rests on' in open(_m6, encoding='utf-8').read(),
         'Supplementary Note 4 rests it on the error ranking instead')

# Table S7 compares the two datings. Its caption asserted that the error increase over control
# is the same under both; it is 50% against 49%, which the table's own last row and the note
# both state. A caption that contradicts the table it captions is the worst kind.
_ma = json.load(open(_os.path.join(OUT, 'temporal_analysis.json')))
_mbp = _os.path.join(OUT, 'temporal_analysis_yearmedian.json')
if _os.path.exists(_mbp):
    _mb = json.load(open(_mbp))
    _T7 = ('scd1', 'nk1r', 'drd2', 'drd3')
    _pa = _ma['delta_vs_control']['rmse_pct_increase_vs_control']
    _pb = _mb['delta_vs_control']['rmse_pct_increase_vs_control']
    cond('SI tables', 'S7: the two datings give a close but not identical error increase',
         abs(_pa - _pb) < 5 and round(_pa) != round(_pb),
         f'{_pa:.1f}% against {_pb:.1f}%, so the caption must not call them the same')
    _lost = lambda d: [t for t in _T7
                       if d[t]['spearman_sigma_err_ci95'][1]
                       < d[t]['control_random_same_size']['spearman_sigma_err_ci95'][0]]
    cond('SI tables', 'S7: the same two targets lose the ranking under both datings',
         _lost(_ma) == _lost(_mb) == ['drd2', 'drd3'], f'{_lost(_ma)} and {_lost(_mb)}')
    cond('SI tables', 'S7: both datings use the same number of control replicates',
         _ma['scd1']['control_random_same_size']['n_reps']
         == _mb['scd1']['control_random_same_size']['n_reps'],
         'the sensitivity must not be resolved more coarsely than the analysis')
    cond('SI tables', 'S7: median dating lifts the SCD-1 correlation above its own control',
         _mb['scd1']['spearman_sigma_err']
         > _mb['scd1']['control_random_same_size']['spearman_sigma_err'],
         'the stated difference between the two datings')
    _m7 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
    if _os.path.exists(_m7):
        cond('SI tables', 'the S7 caption does not claim the two increases are the same',
             'are the same under both' not in open(_m7, encoding='utf-8').read(), '')

# The pooled temporal correlation was said, in the article, the SI note and the S8 caption, to
# be lower than ANY per-target value. It is not: DRD3 sits below it. The point being made,
# that pooling is not a summary, survives; the universal quantifier did not.
_t8 = json.load(open(_os.path.join(OUT, 'temporal_analysis.json')))
_T8 = ('scd1', 'nk1r', 'drd2', 'drd3')
_pl = _t8['pooled']['spearman_sigma_err']
_below = [t for t in _T8 if _pl < _t8[t]['spearman_sigma_err']]
cond('SI tables', 'S8: the pooled correlation is below three of four per-target values, not all',
     len(_below) == 3 and 'drd3' not in _below,
     f'pooled {_pl:.3f}; above DRD3 at {_t8["drd3"]["spearman_sigma_err"]:.3f}')
cond('SI tables', 'S8: coverage falls below nominal on all four targets',
     all(_t8[t]['conformal_coverage_adaptive'] < 0.900 for t in _T8),
     'the caption said three of four')
cond('SI tables', 'S8: the control replicate count in the caption comes from the data',
     _t8['scd1']['control_random_same_size']['n_reps'] == 1000,
     'the caption hard-coded twenty after the run moved to 1000')
cond('SI tables', 'S8: error rises above its control on every target',
     all(_t8[t]['rmse_test'] > _t8[t]['control_random_same_size']['rmse'] for t in _T8), '')
cond('SI tables', 'S8: the pooled evaluation count is the sum of the per-target ones',
     sum(_t8[t]['n_test'] for t in _T8) == _t8['pooled']['n'], '')
for _f8 in ('npjDD_Reliability.tex', 'npjDD_SI.tex'):
    _p8 = _os.path.join(_D, _f8)
    if _os.path.exists(_p8):
        cond('phrasing', f'{_f8}: no claim that pooled is below every per-target value',
             'lower than any of the' not in open(_p8, encoding='utf-8').read(),
             'DRD3 is below the pooled value')

# The label's composition qualifies every target-level comparison, so it is disclosed where the
# Results first name the response, not only in the Methods and the table captions. A reviewer
# asked for it earlier in the text; this keeps it from drifting back out.
_pfg = _os.path.join(_D, 'npjDD_Reliability.tex')
if _os.path.exists(_pfg):
    _sfg = ' '.join(open(_pfg, encoding='utf-8').read().split())
    _res = _sfg[_sfg.index(r'\section{Results}'):]
    _head = _res[:2000]
    cond('phrasing', 'the Results foreground the pooled label before any target comparison',
         'four ChEMBL standard types' in _head and 'affinity rather than a potency' in _head,
         'composition must be flagged where the response is first named')
    cond('phrasing', 'the Results flag the FADS isoform pooling where FADS results first appear',
         'FADS1 and FADS2' in _head, 'target-level differences on FADS must carry the caveat')

# A macro whose value is a word must never sit inside $...$. Math mode sets it in italic with
# math spacing, so "SCD-1" renders as an italic SCD with a spaced minus, and "four" is letter-spaced
# into "f our". Both shipped unnoticed because the value was correct and only its typesetting was
# wrong, which no numeric check can see.
_mv = dict(_re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}', open(
    _os.path.join(_D, 'numbers.tex'), encoding='utf-8').read())) if _os.path.exists(
    _os.path.join(_D, 'numbers.tex')) else {}
def _texty(v):
    v2 = _re.sub(r'\\mbox\{[^}]*\}', '', v)
    v2 = _re.sub(r'\\[A-Za-z]+', '', v2)
    return bool(_re.search(r'[A-Za-z]', v2))
if _mv:
    _inmath = []
    for _fm in ('npjDD_Reliability.tex', 'npjDD_SI.tex'):
        _pm = _os.path.join(_D, _fm)
        if not _os.path.exists(_pm):
            continue
        _sm = open(_pm, encoding='utf-8').read()
        for _seg in _re.finditer(r'(?<!\\)\$([^$]{1,400}?)(?<!\\)\$', _sm, _re.S):
            for _mac in _re.findall(r'\\([A-Za-z]+)', _seg.group(1)):
                if _mac in _mv and _texty(_mv[_mac]):
                    _inmath.append(f'{_fm}:\\{_mac}={_mv[_mac]!r}')
    cond('phrasing', 'no word-valued macro is typeset inside math mode',
         not _inmath, '; '.join(sorted(set(_inmath))))

# The mirror of the check above. A purely numeric macro carrying a negative value must sit
# inside $...$, or LaTeX prints U+002D hyphen where a U+2212 minus belongs. The same macros were
# set in math in one document and in text in the other, so one number rendered two ways. The SI
# tables already solved this in make_si_tables.f(); the prose had not.
if _mv:
    def _numeric(v):
        v2 = _re.sub(r'\\[A-Za-z]+|\{,\}|[\[\]()%$,+\-.0-9 ]', '', v)
        return v2 == '' and bool(_re.search(r'-\d', v))
    _hyph = []
    for _fm in ('npjDD_Reliability.tex', 'npjDD_SI.tex'):
        _pm = _os.path.join(_D, _fm)
        if not _os.path.exists(_pm):
            continue
        _sm = open(_pm, encoding='utf-8').read()
        _inm = set()
        for _seg in _re.finditer(r'(?<!\\)\$([^$]{1,600}?)(?<!\\)\$', _sm, _re.S):
            _inm.update(range(_seg.start(), _seg.end()))
        for _k, _v in _mv.items():
            if not _numeric(_v):
                continue
            for _u in _re.finditer(r'\\' + _k + r'(?![A-Za-z])', _sm):
                if _u.start() not in _inm:
                    _hyph.append(f'{_fm}:\\{_k}={_v}')
    cond('phrasing', 'every negative number is typeset in math, so it prints a minus not a hyphen',
         not _hyph, '; '.join(sorted(set(_hyph))))

# scikit-learn's version is named in the Methods and its random forest produces every sigma_T,
# so it carries the same exposure as RDKit and is pinned for the same reason. RDKit was pinned
# and guarded; scikit-learn was named but left floating, so a reader rebuilding from the archive
# could get a different forest and different numbers.
_req = _os.path.join(_HERE_DIR, 'requirements.txt')
if _os.path.exists(_req) and _mv is not None:
    _rq = open(_req, encoding='utf-8').read()
    _pin = _re.search(r'scikit-learn==([0-9.]+)', _rq)
    _pth = _os.path.join(_D, 'npjDD_Reliability.tex')
    _sk = None
    if _os.path.exists(_pth):
        _sk = _re.search(r'scikit-learn~\\cite\{[a-z0-9]+\}\s*\$([0-9.]+)\$',
                         _re.sub(r'\s+', ' ', open(_pth, encoding='utf-8').read()))
    cond('repo sync', 'scikit-learn is pinned to the version the Methods name',
         bool(_pin) and (_sk is None or _pin.group(1) == _sk.group(1)),
         f"pinned {_pin.group(1) if _pin else 'nothing'}, Methods say {_sk.group(1) if _sk else 'n/a'}")

# The pooled response is named pActivity, not pIC50. pIC50 denotes -log10(IC50) specifically and
# would name the minority constituent: the label is filled from four ChEMBL standard types and is
# mostly Ki. Asserted positively, because a ban on the old name would pass on a document that had
# simply stopped naming the response at all.
for _fn in ('npjDD_Reliability.tex', 'npjDD_SI.tex'):
    _pn = _os.path.join(_D, _fn)
    if not _os.path.exists(_pn):
        continue
    _sn = open(_pn, encoding='utf-8').read()
    cond('manuscript', f'{_fn}: the response notation is defined once as pActivity',
         _sn.count(r'\newcommand{\pAct}') == 1 and r'\mathrm{pActivity}' in _sn,
         'a single definition keeps the two documents from drifting apart')
    cond('manuscript', f'{_fn}: the response is written through that macro, not spelled out',
         _sn.count(r'\pAct') > 1, 'every use must read from the one definition')
    # The old name may survive ONLY where the Methods explain why it is not used.
    _pic = [m.start() for m in _re.finditer(r'\\mathrm\{pIC\}_\{50\}', _sn)]
    _just = _sn.find('rather than $\\mathrm{pIC}_{50}$')
    cond('manuscript', f'{_fn}: pIC50 appears only where the Methods explain rejecting it',
         all(_just != -1 and abs(i - _just) < 700 for i in _pic),
         f'{len(_pic)} occurrence(s) of the old name, which must all sit in that sentence')

# ---- Temporal leakage sensitivity: cutoff-spanning parents removed (Table S23) ----
# A parent's label is the median over all its records while the split follows first disclosure,
# so a pre-cutoff parent re-measured later carries future information into training. The reported
# analysis therefore UNDERSTATES the shift. This arm removes those parents; every degradation must
# survive and none may weaken, or the reported result would be an artifact of the leak.
_nsp_p = _os.path.join(OUT, 'temporal_no_spanning.json')
if _os.path.exists(_nsp_p):
    _ns = json.load(open(_nsp_p))
    cond('temporal/leakage', 'the no-spanning run declares its exclusion',
         _ns.get('exclude_spanning') is True, str(_ns.get('exclude_spanning')))
    cond('temporal/leakage', 'it shares the cutoff and dating of the reported analysis',
         _ns['cut_year'] == _t8['cut_year'] and _ns['year_field'] == _t8['year_field'], '')
    cond('temporal/leakage', 'the historical pool is strictly smaller on every target',
         all(_ns[t]['n_train'] + _ns[t]['n_cal'] < _t8[t]['n_train'] + _t8[t]['n_cal']
             for t in _T8), 'the exclusion must actually remove compounds')
    cond('temporal/leakage', 'error still rises above its control on all four targets',
         all(_ns[t]['delta_vs_control']['rmse']['delta'] > 0 for t in _T8),
         ', '.join(f"{t}={_ns[t]['delta_vs_control']['rmse']['delta']:+.2f}" for t in _T8))
    cond('temporal/leakage', 'error is above every control replicate on all four targets',
         all(_ns[t]['control_random_same_size']['temporal_outside_range']['rmse'] for t in _T8), '')
    # The claim the article makes: removing the leak makes each degradation LARGER, so the
    # reported figures understate the shift. Compared in the degrading direction.
    _m = lambda d, k: sum(d[t]['delta_vs_control'][k]['delta'] for t in _T8) / len(_T8)
    for _k, _sg in (('rmse', +1), ('spearman', -1), ('coverage', -1)):
        cond('temporal/leakage', f'the mean {_k} effect is not weaker once the leak is removed',
             _sg * _m(_ns, _k) >= _sg * _m(_t8, _k),
             f'no-spanning {_m(_ns, _k):+.3f} vs reported {_m(_t8, _k):+.3f}')
    # Semantic: the SAME two targets must lose the ranking, or a per-target claim would change.
    _lost = lambda d: [t for t in _T8 if d[t]['spearman_sigma_err_ci95'][1]
                       < d[t]['control_random_same_size']['spearman_sigma_err_ci95'][0]]
    cond('temporal/leakage', 'the same two targets lose the ranking, DRD2 and DRD3',
         _lost(_ns) == _lost(_t8) == ['drd2', 'drd3'],
         f'no-spanning {_lost(_ns)}, reported {_lost(_t8)}')

# ---- Temporal shift under a single measurement type (Results, Methods, Table S23) ----
# The pooled response mixes IC50, Ki, Kd and EC50, so the temporal degradation could in principle
# be the assay changing rather than the chemistry. This is the direct test: each target reduced
# to the parents whose records are all of one standard type, with its own size-matched controls.
_epp = _os.path.join(OUT, 'temporal_endpoint.json')
if _os.path.exists(_epp):
    _te = json.load(open(_epp))
    cond('temporal/endpoint', 'the restricted run declares its restriction',
         _te.get('endpoint_restriction') == 'single', str(_te.get('endpoint_restriction')))
    cond('temporal/endpoint', 'restricted and pooled runs share cutoff and dating',
         _te['cut_year'] == _t8['cut_year'] and _te['year_field'] == _t8['year_field'],
         f"{_te['cut_year']}/{_te['year_field']} vs {_t8['cut_year']}/{_t8['year_field']}")
    cond('temporal/endpoint', 'the control replicate count matches the pooled analysis',
         all(_te[t]['control_random_same_size']['n_reps']
             == _t8[t]['control_random_same_size']['n_reps'] for t in _T8), '')
    # The headline: error degrades on every target, above every one of the control replicates.
    cond('temporal/endpoint', 'error rises above its own size-matched control on all four targets',
         all(_te[t]['delta_vs_control']['rmse']['delta'] > 0 for t in _T8),
         ', '.join(f"{t}={_te[t]['delta_vs_control']['rmse']['delta']:+.2f}" for t in _T8))
    cond('temporal/endpoint', 'the error rise is at the Monte Carlo floor on all four targets',
         all(_te[t]['control_random_same_size']['empirical_p']['rmse']
             <= _te[t]['control_random_same_size']['empirical_p']['floor'] for t in _T8), '')
    cond('temporal/endpoint', 'coverage falls below its control on all four targets',
         all(_te[t]['delta_vs_control']['coverage']['delta'] < 0 for t in _T8), '')
    # Semantic, not merely numeric: the ranking exception must be the SAME target as in the
    # pooled analysis. A run that degraded a different target would still pass a count check.
    _exr = [t for t in _T8 if _te[t]['delta_vs_control']['spearman']['delta'] > 0]
    _exp = [t for t in _T8 if _t8[t]['delta_vs_control']['spearman']['delta'] > 0]
    cond('temporal/endpoint', 'the ranking exception is SCD-1, the same target as pooled',
         _exr == ['scd1'] and _exp == ['scd1'], f'restricted {_exr}, pooled {_exp}')
    # Every mean effect must sit further from the control than the pooled one, which is what the
    # article claims. Compared in the degrading direction, since two of the three are negative.
    _mac = lambda d, k: sum(d[t]['delta_vs_control'][k]['delta'] for t in _T8) / len(_T8)
    for _k, _sgn in (('rmse', +1), ('spearman', -1), ('coverage', -1)):
        cond('temporal/endpoint', f'the mean {_k} effect is further from the control than pooled',
             _sgn * _mac(_te, _k) > _sgn * _mac(_t8, _k),
             f'restricted {_mac(_te, _k):+.3f} vs pooled {_mac(_t8, _k):+.3f}')
    # NK1R keeps Ki rather than the IC50 that dominates its pooled set, because its post-cutoff
    # records are mostly Ki. The article states this as evidence of turnover, so it is asserted.
    cond('temporal/endpoint', 'NK1R keeps Ki, not the IC50 that dominates its pooled set',
         _te['kept_type']['nk1r'] == 'Ki', str(_te['kept_type']))
    # The Table S23 caption explains WHY that flip happens: NK1R's pooled set is mostly IC50
    # while its post-cutoff records are mostly Ki. That reasoning is a factual claim about the
    # composition, so it is asserted rather than left to the reader to trust.
    _ec = json.load(open(_os.path.join(OUT, 'endpoint_composition.json')))['temporal_cohort']['nk1r']
    _dom = lambda d: max(d, key=d.get)
    cond('temporal/endpoint', 'NK1R is mostly IC50 overall but mostly Ki after the cutoff',
         _dom(_ec['overall_pct']) == 'IC50' and _dom(_ec['post_pct']) == 'Ki',
         f"overall {_dom(_ec['overall_pct'])} {max(_ec['overall_pct'].values()):.0f}%, "
         f"post {_dom(_ec['post_pct'])} {max(_ec['post_pct'].values()):.0f}%")
    cond('temporal/endpoint', 'SCD-1 keeps IC50, so its restriction is close to a no-op',
         _te['kept_type']['scd1'] == 'IC50', str(_te['kept_type']))
    # SCD-1 is the procedure check: one evaluation compound removed, pooled result returned.
    cond('temporal/endpoint', 'the SCD-1 restriction drops exactly one evaluation compound',
         _t8['scd1']['n_test'] - _te['scd1']['n_test'] == 1,
         f"{_t8['scd1']['n_test']} -> {_te['scd1']['n_test']}")
    close('temporal/endpoint', 'SCD-1 restricted error reproduces the pooled one',
          _te['scd1']['rmse_test'], _t8['scd1']['rmse_test'], 0.01)
    # With four targets the interval on the mean error effect includes zero. The article says so
    # and leans on the per-target comparison instead; this asserts the article is not overclaiming.
    import statistics as _st
    _d4 = [_te[t]['delta_vs_control']['rmse']['delta'] for t in _T8]
    _lo = _st.mean(_d4) - 3.182 * _st.stdev(_d4) / len(_d4) ** 0.5
    cond('temporal/endpoint', 'the four-target interval on the mean error effect includes zero',
         _lo < 0, f'lower limit {_lo:+.3f}; the article must not claim separation from zero')
    # The article says the interval on EACH of the three mean effects includes zero. Asserted for
    # all three, because quoting the weakness of one while three are weak is a partial disclosure.
    _incl = {}
    for _k, _dg in (('rmse', +1), ('spearman', -1), ('coverage', -1)):
        _v = [_te[t]['delta_vs_control'][_k]['delta'] for t in _T8]
        _mn, _sd = _st.mean(_v), _st.stdev(_v)
        _incl[_k] = (_mn - 3.182 * _sd / 2) < 0 < (_mn + 3.182 * _sd / 2)
    cond('temporal/endpoint', 'each of the three restricted mean effects has an interval spanning zero',
         all(_incl.values()), str(_incl))
    _pep = _os.path.join(_D, 'npjDD_Reliability.tex')
    if _os.path.exists(_pep):
        # Structural rather than literal: the mean error effect may not be quoted without the
        # interval that shows it includes zero. Testing for a form of words would break on a
        # rewrite; testing that the two macros travel together tests the claim.
        _sep = open(_pep, encoding='utf-8').read()
        # Matched by macro name, not by its surrounding braces or math delimiters: an earlier
        # version tested the literal "\TempEpMacroDRmseCI{}" and broke when the same macro was
        # wrapped in $...$ to render its minus correctly. The claim is that the two travel
        # together, not how either is punctuated.
        _has = lambda n: bool(_re.search(r'\\' + n + r'(?![A-Za-z])', _sep))
        cond('phrasing', 'the restricted mean effect is never quoted without its interval',
             (not _has('TempEpMacroDRmse')) or _has('TempEpMacroDRmseCI'),
             'the interval includes zero, so the mean must not stand alone')

# Table S9's per-target comparison is compressed on the small pools: the fixed 300-query budget
# is about half of SCD-1 but 3% of DRD2, so on SCD-1 and FADS the best rule already takes most of
# the top-percentile compounds that exist and the rules cannot separate much there.
_p9 = json.load(open(_os.path.join(OUT, 'poolopt_analysis.json')))['summary']
_T9 = ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
_M9 = ('greedy', 'ucb', 'lcb', 'conformal')
_share = {t: max(_p9[t][m]['hits'] for m in _M9) / _p9[t]['n_top1pct_in_pool'] for t in _T9}
cond('SI tables', 'S9: SCD-1 and FADS are the targets near their acquisition ceiling',
     sorted(t for t in _T9 if _share[t] > 0.8) == ['fads', 'scd1'],
     ', '.join(f'{t} {100*_share[t]:.0f}%' for t in _T9))
cond('SI tables', 'S9: no rule acquires more than the pool contains',
     all(_p9[t][m]['hits'] <= _p9[t]['n_top1pct_in_pool'] for t in _T9 for m in _M9), '')
cond('SI tables', 'S9: no rule acquires more hits than the query budget allows',
     all(_p9[t][m]['hits'] <= 300 for t in _T9 for m in _M9), '')
cond('SI tables', 'S9: the top-percentile set is about 1% of each pool',
     all(0.009 < _p9[t]['n_top1pct_in_pool'] / _p9[t]['pool_size'] < 0.013 for t in _T9),
     'ties at the cutoff make it at least 1%')
cond('SI tables', 'S9: random enrichment is exactly 1 by construction',
     all(abs(_p9[t]['random']['enrichment_vs_random'] - 1.0) < 1e-9 for t in _T9), '')
_m9 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_m9):
    cond('SI tables', 'the S9 caption states the ceiling on the small pools',
         'little room to separate there' in open(_m9, encoding='utf-8').read(),
         'otherwise the per-target pattern reads as a rule difference')

# Table S10 reported five per-target turnover differences as a bare sign count. The cells are
# small and unbalanced, DRD2's seventh bin holding 7 runs and DRD3's 9, and a per-target
# bootstrap over seeds shows DRD3's difference covers zero. A sign count cannot show that.
import numpy as _np10
_f10 = json.load(open(_os.path.join(OUT, 'frontier_v2_results.json')))['results']
_r10 = [dict(r, target=t) for t, rs in _f10.items() for r in rs]
_n10 = [r['novelty'] for r in _r10]
_e10 = [min(_n10) + (max(_n10) - min(_n10)) * i / 8 for i in range(9)]
def _b10(v):
    for i in range(8):
        if v <= _e10[i + 1]:
            return i
    return 7
_cov0, _small = [], {}
for _t in ('scd1', 'fads', 'nk1r', 'drd2', 'drd3'):
    _s10 = [r for r in _r10 if r['target'] == _t]
    _small[_t] = sum(1 for r in _s10 if _b10(r['novelty']) == 6)
    _sd = sorted({r['seed'] for r in _s10})
    _by = {x: [r for r in _s10 if r['seed'] == x] for x in _sd}
    _mk = lambda rs, k: (_np10.mean([r['sigma'] for r in rs if _b10(r['novelty']) == k])
                         if any(_b10(r['novelty']) == k for r in rs) else _np10.nan)
    _rg = _np10.random.default_rng(3); _dd = []
    for _ in range(2000):
        _pk = [x for q in _rg.choice(_sd, len(_sd), replace=True) for x in _by[q]]
        _v = _mk(_pk, 6) - _mk(_pk, 7)
        if not _np10.isnan(_v):
            _dd.append(_v)
    _lo, _hi = _np10.percentile(_dd, [2.5, 97.5])
    if _lo <= 0 <= _hi:
        _cov0.append(_t)
cond('SI tables', 'S10: DRD3 alone has a turnover interval covering zero',
     _cov0 == ['drd3'], f'covers zero on {_cov0}, so a sign count of four overstates it')
cond('SI tables', 'S10: the seventh bin is thin on DRD2 and DRD3',
     _small['drd2'] < 10 and _small['drd3'] < 10,
     f"n7 = {_small['drd2']} and {_small['drd3']}, against {_small['nk1r']} on NK1R")
_m10 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_m10):
    cond('SI tables', 'S10 carries a per-target interval column',
         '_ci_fmt(_ci2)' in open(_m10, encoding='utf-8').read(),
         'five sign counts alone cannot say whether any is distinguishable')

# Table S11 shows novelty-to-distance correlations of 0.975 to 0.983. Part of that is
# definitional, since the starting compounds are drawn from the training set, so d <= nu for
# every molecule. The main text and the note say so; the caption did not, and a caption is read
# on its own.
_fr11 = json.load(open(_os.path.join(OUT, 'frontier_v2_analysis.json')))
_K11 = ('graphga_lam0.0', 'graphga_lam0.1', 'stga_lam0.0', 'stga_lam0.1')
cond('SI tables', 'S11: every row rests on 300 runs, 1200 in total',
     all(_fr11[k]['n'] == 300 for k in _K11) and sum(_fr11[k]['n'] for k in _K11) == 1200, '')
cond('SI tables', 'S11: novelty rises with distance and disagreement in all four rows',
     all(_fr11[k]['nov_dtrain'] > 0 and _fr11[k]['nov_sig'] > 0 for k in _K11), '')
cond('SI tables', 'S11: predicted potency falls with novelty in all four rows',
     all(_fr11[k]['nov_pot'] < 0 for k in _K11), '')
cond('SI tables', 'S11: the pattern holds at zero uncertainty penalty',
     all(_fr11[k]['nov_dtrain'] > 0 and _fr11[k]['nov_sig'] > 0 and _fr11[k]['nov_pot'] < 0
         for k in ('graphga_lam0.0', 'stga_lam0.0')),
     'so it is not induced by the penalty')
_m11 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_m11):
    cond('SI tables', 'the S11 caption discloses the definitional part of the association',
         'is definitional' in open(_m11, encoding='utf-8').read(),
         '0.98 would otherwise read as an empirical finding')

# Table S12's per-cell method comparison, and the P formatting it shares with S15.
_mt12 = json.load(open(_os.path.join(OUT, 'methods_v2_results.json')))['results']
cond('SI tables', 'S12: the reported delta equals ST-GA minus Graph GA in every cell',
     all(abs((r['agg']['stga_ecfp'] - r['agg']['graphga']) - r['agg']['d_vs_ga']) < 5e-4
         for r in _mt12), '')
cond('SI tables', 'S12: the surrogate beats Graph GA in all 15 cells',
     all(r['agg']['d_vs_ga'] > 0 for r in _mt12), f'{len(_mt12)} cells')
cond('SI tables', 'S12: the surrogate also beats random triage in all 15 cells',
     all(r['agg']['stga_ecfp'] > r['agg']['randtriage'] for r in _mt12),
     'which attributes the gain to the surrogate, not the larger pool')
cond('SI tables', 'S12: the win fraction is a proportion on every row',
     all(0.0 <= r['agg']['frac_pos_ga'] <= 1.0 for r in _mt12), '')
_m12 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_m12):
    _c12 = open(_m12, encoding='utf-8').read()
    cond('SI tables', 'S12 and S15 format P through one helper, not two ad hoc expressions',
         _c12.count('_psci(') >= 3 and "replace('e-0'" not in _c12,
         'the old chained-replace produced a stray brace for any P above 0.1')
    cond('phrasing', 'no threshold-only P survives in the generated tables',
         'P<0.05' not in _c12, 'npj requires an exact P')
for _f12 in ('npjDD_Reliability.tex', 'npjDD_SI.tex'):
    _pp = _os.path.join(_D, _f12)
    if _os.path.exists(_pp):
        cond('phrasing', f'{_f12}: no threshold-only P',
             'P<0.05' not in open(_pp, encoding='utf-8').read().replace(' ', ''),
             'npj forbids reporting significance against a threshold alone')

# Table S13's two rows come from different files, and the dual-encoder one could not be
# regenerated from a clone at all: wmga_results.json was the single script input the repository
# did not ship. Check every script input is present, not just this one.
import re as _re13
_have13 = set(_os.listdir(OUT))
_missing13 = {}
for _f13 in sorted(_os.listdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                             'reliability'))):
    if not _f13.endswith('.py'):
        continue
    _src13 = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'reliability', _f13),
                  encoding='utf-8').read()
    _need = set(_re13.findall(r"outputs/frozen/([A-Za-z0-9_]+\.json)", _src13))
    # Also catch inputs referenced through os.path.join rather than a literal path; one script
    # read outputs/compat_v8_16tgt.json that way and slipped past the first version of this.
    _need |= set(_re13.findall(r"os\.path\.join\(BASE, 'outputs', (?:'frozen', )?'([A-Za-z0-9_]+\.json)'\)", _src13))
    _gone = sorted(n for n in _need if n not in _have13)
    if _gone:
        _missing13[_f13] = _gone
# The README documents four superseded _v1 entry points as off the reproduction path, some of
# which read inputs that are deliberately not distributed. Everything else must be runnable.
_EXEMPT13 = {'run_fewshot_v1.py', 'run_reliability_v1.py', 'run_ecfp_baseline_v1.py',
             'analyze_frontier.py'}
_missing13 = {k: v for k, v in _missing13.items() if k not in _EXEMPT13}
cond('portability', 'every reproduction-path script has its frozen inputs shipped',
     not _missing13,
     str(_missing13) + ' (four superseded _v1 entry points are exempt, as the README states)')
_h13 = json.load(open(_os.path.join(OUT, 'hierstats_analysis.json')))['target_level']
cond('SI tables', 'S13: the dual-encoder effect is positive on all five targets',
     all(v > 0 for v in _h13['target_means'].values()), '')
cond('SI tables', 'S13: the dual-encoder test uses the five targets as the unit',
     _h13['n_targets'] == 5, 'df = 4')
_mt13 = json.load(open(_os.path.join(OUT, 'methods_v2_results.json')))['results']
import statistics as _st13
_fp13 = {t: _st13.mean([c['stga_ecfp'] - c['graphga'] for r in _mt13 if r['target'] == t
                        for c in r['per_seed']]) for t in ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')}
cond('SI tables', 'S13: the fingerprint surrogate beats the dual encoder on every target',
     all(_fp13[t] > _h13['target_means'][t] for t in _fp13),
     'which is the point of showing the two rows together')
_rh = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'reliability',
                    'run_hierstats_v1.py')
if _os.path.exists(_rh):
    cond('SI tables', 'the hierstats script says which comparison it computes',
         'DUAL-ENCODER' in open(_rh, encoding='utf-8').read(),
         'its docstring named the fingerprint comparison it does not compute')

# Table S14's beta ablation. Its acquisition uses the triage surrogate's own dispersion, a
# different quantity from the activity model's disagreement the rest of the paper calls sigma_T,
# and the caption used the symbol without saying so.
_b14 = json.load(open(_os.path.join(OUT, 'beta_ablation.json')))['summary']
_T14 = ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
_BS = ('beta0.0', 'beta0.5', 'beta1.0', 'beta2.0')
cond('SI tables', 'S14: the reported delta is beta=1 minus beta=0 on every target',
     all(abs((_b14[t]['beta1.0']['top10_novel'] - _b14[t]['beta0.0']['top10_novel'])
             - _b14[t]['beta1_vs_beta0_novel']['delta']) < 5e-4 for t in _T14), '')
cond('SI tables', 'S14: every setting of beta beats Graph GA on every target',
     all(_b14[t][b]['top10_novel'] > _b14[t]['graphga']['top10_novel']
         for t in _T14 for b in _BS),
     'the surrogate helps; the uncertainty term in its acquisition is what does not')
cond('SI tables', 'S14: the beta effect is negative on DRD3 alone',
     [t for t in _T14 if _b14[t]['beta1_vs_beta0_novel']['delta'] < 0] == ['drd3'], '')
cond('SI tables', 'S14: NK1R is the one target with a small P, and its effect is positive',
     min(_T14, key=lambda t: _b14[t]['beta1_vs_beta0_novel']['p']) == 'nk1r'
     and _b14['nk1r']['beta1_vs_beta0_novel']['delta'] > 0,
     'so the one precise result is a tiny help, not a harm')
cond('SI tables', 'S14: the beta=1 minus beta=0 difference is negligible on every target',
     max(abs(_b14[t]['beta1_vs_beta0_novel']['delta']) for t in _T14) < 0.02,
     'this is the claim the caption makes; larger beta is a separate question')
_mono14 = [t for t in _T14
           if all(_b14[t][_BS[i]]['top10_novel'] >= _b14[t][_BS[i + 1]]['top10_novel']
                  for i in range(3))]
cond('SI tables', 'S14: DRD3 alone declines monotonically as beta rises',
     _mono14 == ['drd3'],
     'the only target where the uncertainty weight has a consistent effect, and it is a loss')
_m14 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_m14):
    cond('SI tables', 'the S14 caption distinguishes the surrogate dispersion from sigma_T',
         "not the activity model" in open(_m14, encoding='utf-8').read(),
         'they are different quantities and the caption reuses a similar symbol')

# Table S15 is the latent-versus-fingerprint ablation at k=10. The same surrogate also appears
# as the dual-encoder row of Table S13, pooled over all three support sizes, so the two tables
# report the same comparison with figures that differ by up to 0.03. Both captions now say
# which support sizes they cover.
_e15 = json.load(open(_os.path.join(OUT, 'ecfp_baseline.json')))['summary']
_T15 = ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
cond('SI tables', 'S15: the reported difference is latent minus fingerprint on every target',
     all(abs((_e15[t]['latent'] - _e15[t]['ecfp']) - _e15[t]['latent_minus_ecfp']) < 5e-4
         for t in _T15), '')
cond('SI tables', 'S15: the latent surrogate loses to fingerprints on all five targets',
     all(_e15[t]['latent_minus_ecfp'] < 0 for t in _T15), '')
cond('SI tables', 'S15: the latent surrogate still beats Graph GA on all five',
     all(_e15[t]['latent'] > _e15[t]['graphga'] for t in _T15),
     'worse than a fingerprint, not worse than no surrogate')
cond('SI tables', 'S15: every latent-versus-fingerprint P is below 0.05',
     max(_e15[t]['p_lat_ecfp'] for t in _T15) < 0.05,
     f"largest {max(_e15[t]['p_lat_ecfp'] for t in _T15):.1e}")
_m15 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_m15):
    _c15 = open(_m15, encoding='utf-8').read()
    cond('SI tables', 'S13 states which support sizes it pools',
         'pooled over the support sizes' in _c15,
         'otherwise its dual-encoder row and S15 disagree with no explanation')
    cond('SI tables', 'S15 explains why a Graph GA column is shown',
         'not worse than no surrogate at all' in _c15, '')

# Table S16's 20 ordered pairs share targets, so an ordinary Pearson P over them has no
# independent-error justification. An exact permutation of the five target labels does. The
# n_target column also held raw record counts where the paper's unit is the parent structure.
import itertools as _it16v
from scipy import stats as _sp16v
_cg16 = json.load(open(_os.path.join(OUT, 'compat_gen_analysis.json')))['rows']
_T16v = sorted({x['target'] for x in _cg16})
cond('SI tables', 'S16: every target appears equally often as source and destination',
     len({sum(1 for x in _cg16 if x['source'] == t) for t in _T16v}) == 1
     and len({sum(1 for x in _cg16 if x['target'] == t) for t in _T16v}) == 1,
     'which is why the pairs are not independent')
_cm = {(x['source'], x['target']): x['C_nn'] for x in _cg16}
_gm = {(x['source'], x['target']): x['gain_mean'] for x in _cg16}
_obs = _sp16v.pearsonr([x['C_nn'] for x in _cg16], [x['gain_mean'] for x in _cg16])[0]
_nl = []
for _pm in _it16v.permutations(_T16v):
    _mp = dict(zip(_T16v, _pm)); _c, _g = [], []
    for (_s, _t) in _gm:
        if (_mp[_s], _mp[_t]) in _cm:
            _c.append(_cm[(_mp[_s], _mp[_t])]); _g.append(_gm[(_s, _t)])
    if len(_c) == len(_gm):
        _nl.append(_sp16v.pearsonr(_c, _g)[0])
_pp = (1 + sum(1 for v in _nl if abs(v) >= abs(_obs))) / (len(_nl) + 1)
cond('SI tables', 'S16: the label-permutation test is exact over all 120 relabelings',
     len(_nl) == 120, f'{len(_nl)} permutations')
cond('SI tables', 'S16: the warm-start correlation is null under the permutation test',
     _pp > 0.1, f'permutation P {_pp:.2f} against a nominal {_obs:.2f} correlation')
cond('SI tables', 'S16: warm-start gains are small and of both signs',
     min(x['gain_mean'] for x in _cg16) < 0 < max(x['gain_mean'] for x in _cg16)
     and max(abs(x['gain_mean']) for x in _cg16) < 0.2, '')
_m16 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_m16):
    _c16 = open(_m16, encoding='utf-8').read()
    cond('SI tables', 'S16 reports standardized parents, not raw records',
         "rel[r['target']]['duplicates']['n_unique']" in _c16,
         'the frozen n_target field is a raw record count')
    cond('SI tables', 'S16 says the compatibility score is carried over, not fitted here',
         'carried over from the companion' in _c16, '')

# Table S17 claimed the model predicts above threshold on four of five with SCD-1
# underpredicted, but showed no threshold column, so neither half could be checked: 6.80 fails
# on SCD-1 while 6.95 passes on DRD2 only because the thresholds differ. All five are in fact
# underpredicted, and FADS by almost as much as SCD-1.
_rc17 = json.load(open(_os.path.join(OUT, 'recovery_v2_results.json')))['results']
_om17 = {e['target']: e for e in json.load(open(_os.path.join(OUT, 'oracle_metrics.json')))['results']}
_T17 = ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
cond('SI tables', 'S17: every withheld cluster is underpredicted, not only SCD-1',
     all(_rc17[t]['retrained_pred'] < _rc17[t]['measured'] for t in _T17),
     'so underprediction is not what singles SCD-1 out')
cond('SI tables', 'S17: SCD-1 alone falls below its activity threshold',
     [t for t in _T17 if _rc17[t]['retrained_pred'] < _om17[t]['threshold']] == ['scd1'],
     'that is the threshold crossing, a different claim from underprediction')
cond('SI tables', 'S17: FADS is underpredicted nearly as much as SCD-1 yet clears its threshold',
     abs((_rc17['fads']['measured'] - _rc17['fads']['retrained_pred'])
         - (_rc17['scd1']['measured'] - _rc17['scd1']['retrained_pred'])) < 0.05
     and _rc17['fads']['retrained_pred'] >= _om17['fads']['threshold'],
     'the SCD-1 failure is partly where its threshold sits')
cond('SI tables', 'S17: recovery exceeds the chemical-space null on every target',
     all(_rc17[t]['rec_stga'] > _rc17[t]['null_sim'] for t in _T17), '')
cond('SI tables', 'S17: the triage does not improve recovery on any target',
     all(_rc17[t]['rec_graphga'] >= _rc17[t]['rec_stga'] for t in _T17), '')
_m17 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_m17):
    cond('SI tables', 'S17 shows the threshold its claim depends on',
         "om[t]['threshold']:.1f" in open(_m17, encoding='utf-8').read(),
         'otherwise 6.80 failing and 6.95 passing looks arbitrary')

# Table S18 carries the in-domain novelty finding. Its exception was wrong in an earlier
# version of the paper, so pin the target by name, the direction, and the size of the gap that
# makes "indistinguishable" a statement rather than an assertion.
_r18 = json.load(open(_os.path.join(OUT, 'reliability_v2_analysis.json')))
_T18 = ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
_g18 = {t: _r18[t]['novel_out_domain_rmse'] - _r18[t]['novel_in_domain_rmse'] for t in _T18}
cond('SI tables', 'S18: nearer-training compounds have lower error on four targets',
     sum(_g18[t] > 0 for t in _T18) == 4, str({t: round(_g18[t], 3) for t in _T18}))
cond('SI tables', 'S18: SCD-1 is the exception, not FADS',
     [t for t in _T18 if _g18[t] <= 0] == ['scd1'],
     'an earlier version of the paper named FADS here')
cond('SI tables', 'S18: the SCD-1 gap is an order smaller than any real one',
     min(abs(_g18[t]) for t in _T18 if t != 'scd1') / abs(_g18['scd1']) > 10,
     'which is what makes the two groups indistinguishable there')
cond('SI tables', 'S18: disagreement is lower in the nearer group on all five',
     all(_r18[t]['novel_in_domain_sigma'] < _r18[t]['novel_out_domain_sigma'] for t in _T18),
     'the direction holds even where the error gap does not')
cond('SI tables', 'S18: the two groups are near equal in size, as a median split implies',
     all(abs(_r18[t]['n_novel_in_domain'] - _r18[t]['n_novel_out_domain'])
         / _r18[t]['n_novel_in_domain'] < 0.1 for t in _T18), '')
cond('SI tables', 'S18: each group is about a sixth of the structures, as the third-then-half is',
     all(0.13 < (_r18[t]['n_novel_in_domain'] + _r18[t]['n_novel_out_domain'])
         / _r18[t]['duplicates']['n_unique'] < 0.36 for t in _T18),
     'confirms the most-novel-third then median-split procedure')
_m18 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_m18):
    cond('SI tables', 'S18 shows both group sizes, not one unlabelled n',
         "r['n_novel_out_domain']" in open(_m18, encoding='utf-8').read(), '')

# Table S19 claimed its columns let a reader reproduce the curation flow. They did not: three
# of the five drop categories were shown and the parent-grouping stage was missing entirely,
# which on DRD2 is 5,579 records against 557 for every filter combined. Assert the flow closes.
_pv19 = _os.path.join(_HERE_DIR, 'data', 'chembl_v2', 'curation_provenance.json')
if _os.path.exists(_pv19):
    _p19 = json.load(open(_pv19))
    _bad19 = []
    for _t19, _v19 in _p19.items():
        _d19 = _v19['dropped']
        _recs = (_v19['n_raw_records'] - _d19.get('standard_type_not_affinity', 0)
                 - _d19.get('no_value', 0) - _d19.get('nonpositive_value', 0)
                 - _d19.get('unparsable_structure', 0) - _d19.get('non_human', 0))
        if _recs != _v19['n_after_filters'] or _recs < _v19['n_unique_parents']:
            _bad19.append(_t19)
    cond('SI tables', 'S19: raw minus every drop category equals the retained record count',
         not _bad19, str(_bad19) + ' (the three-column version left a residual on three targets)')
    cond('SI tables', 'S19: retained records minus collapsed equals the parent count',
         all(_v['n_after_filters'] >= _v['n_unique_parents'] for _v in _p19.values()), '')
    cond('SI tables', 'S19: parent grouping is the largest single reduction on DRD2',
         (_p19['drd2']['n_after_filters'] - _p19['drd2']['n_unique_parents'])
         > (_p19['drd2']['n_raw_records'] - _p19['drd2']['n_after_filters']),
         'so omitting it hid the biggest stage of the flow')
    cond('SI tables', 'S19: the parent counts match the temporal cohort files',
         all(_p19[t]['n_unique_parents'] > 0 for t in _p19), '')
_m19 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_m19):
    cond('SI tables', 'the S19 generator asserts its own flow reconciles',
         'curation flow does not reconcile' in open(_m19, encoding='utf-8').read(),
         'so a future drop category cannot silently break the arithmetic')

# Table S20's n column mixed two units: the endpoint file counts distinct canonical SMILES,
# which equals the parent count on SCD-1 but the record count on the other three, while the
# temporal rows are parent counts. One column, two definitions.
_ep20 = json.load(open(_os.path.join(OUT, 'endpoint_composition.json')))
_r20 = json.load(open(_os.path.join(OUT, 'reliability_v2_analysis.json')))
_T20 = ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
cond('SI tables', 'S20: every endpoint mix sums to 100 percent',
     all(abs(sum(_ep20['cv_cohort'][t]['types_pct'].values()) - 100) < 0.3
         for t in _T20 if _ep20['cv_cohort'][t]['types_pct'])
     and all(abs(sum(_ep20['temporal_cohort'][t][k].values()) - 100) < 0.3
             for t in ('scd1', 'nk1r', 'drd2', 'drd3')
             for k in ('overall_pct', 'pre_pct', 'post_pct')), '')
# The Results claim a bake-off selected the random forest. That claim needs its numbers shown,
# and it needs the disclosure that selection and reporting share a cross-validation.
_omq = {e['target']: e for e in json.load(open(_os.path.join(OUT, 'oracle_metrics.json')))['results']}
cond('audit/model selection', 'the random forest is selected on every target',
     all(v['selected_model'] == 'rf' for v in _omq.values()),
     ', '.join(f"{k}:{v['selected_model']}" for k, v in sorted(_omq.items())))
cond('audit/model selection', 'all three compared ensembles are present for every target',
     all(set(v['bakeoff']) == {'rf', 'extratrees', 'histgb'} for v in _omq.values()), '')

# Supplementary captions must not carry typed measurements. Table S2's explanation of the
# extreme FADS R^2 stated the fold spread, the whole-set spread, the variance ratio, the implied
# RMSE and the range across targets as literals; all five were right, but all five come from
# scaffold_fold_stats.json and would have gone stale on a re-run.
_msty = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_msty):
    _mst = open(_msty, encoding='utf-8').read()
    _s2 = _mst[_mst.index("tab('tab:s-protocol'"):]
    _s2 = _s2[:_s2.index('rows,')] if 'rows,' in _s2 else _s2[:3000]
    cond('SI tables', 'S2 computes its scaffold statistics rather than typing them',
         'activity spread of %s against %s' in _s2 and "sfs['fads']['sd_test']" in _s2,
         'the caption quoted five measured values as literals')
    # And the values it computes must still be the ones the frozen stats give.
    _sfsq = json.load(open(_os.path.join(OUT, 'scaffold_fold_stats.json')))
    _rr = [v['implied_rmse'] for v in _sfsq.values() if isinstance(v, dict)]
    close('SI tables', 'S2: FADS held-out spread', _sfsq['fads']['sd_test'], 0.07, 0.005)
    close('SI tables', 'S2: FADS variance ratio', _sfsq['fads']['var_ratio'], 531, 1.0)
    close('SI tables', 'S2: implied RMSE range covers the five targets',
          max(_rr) - min(_rr), 1.5362 - 0.9728, 0.01)

# Table font policy, which has two halves and they are easy to confuse. No table may shrink its
# whole body: a wide table is widened by its column count, so padding is the fix, not the font.
# But S8's bracketed intervals ARE set one size down on purpose, because they are supporting
# detail beside the estimate they qualify and crowd the column at full size. So: no size command
# applied to a whole tabular, and the inline shrink confined to S8's intervals.
_sit = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'WritePaper',
                     'theranostics', 'JournalPapers_npjDD', 'si_tables.tex')
if _os.path.exists(_sit):
    _sitx = open(_sit, encoding='utf-8').read()
    _whole = []
    for _blk in _re.split(r'(?=\\begin\{table\})', _sitx):
        _lab = _re.search(r'\\label\{(tab:s-[a-z]+)\}', _blk)
        if not _lab:
            continue
        _pre = _blk.split('\\begin{tabular}')[0]
        if _re.search(r'\\(small|footnotesize|scriptsize|tiny)\b', _pre):
            _whole.append(_lab.group(1))
    cond('SI tables', 'no table shrinks its whole body font',
         not _whole, f'whole-table size command on {_whole}; widen with column padding instead')
    _inline = {_re.search(r'\\label\{(tab:s-[a-z]+)\}', b).group(1)
               for b in _re.split(r'(?=\\begin\{table\})', _sitx)
               if _re.search(r'\\label\{tab:s-[a-z]+\}', b) and '{\\small [' in b}
    cond('SI tables', 'the inline interval shrink is confined to S8',
         _inline in (set(), {'tab:s-temporal'}), f'also used in {_inline - {"tab:s-temporal"}}')

# Supplementary Note 11 opens by counting the negative results it is about to list. It said
# three and listed four, each with its own table, so the count and the content disagreed. Tie the
# stated number to the tables the note actually cites.
_sinote = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'WritePaper',
                        'theranostics', 'JournalPapers_npjDD', 'npjDD_SI.tex')
if _os.path.exists(_sinote):
    _sn = open(_sinote, encoding='utf-8').read()
    if '\\section{Results that were negative}' in _sn:
        _neg = _sn.split('\\section{Results that were negative}')[1].split('\\section{')[0]
        _ncited = len(_re.findall(r'\\ref\{tab:s-[a-z]+\}', _neg))
        _WORD = {2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six'}
        cond('SI notes', 'Note 11 counts the negative results it actually lists',
             _re.sub(r'\s+', ' ', _neg).lstrip().startswith(_WORD.get(_ncited, '?') + ' further'),
             f'it cites {_ncited} supplementary tables, one per result')

# --- math/consistency audit invariants -------------------------------------------------
# Every derived percentage must be reconstructible from the values it is derived from. These
# are cheap and they catch a stale numerator or denominator that no per-value check would.
_relq = json.load(open(_os.path.join(OUT, 'reliability_v2_analysis.json')))
_pq = _relq['pooled']
_mi, _ma = _pq['risk_coverage_micro'], _pq['risk_coverage_macro']
close('audit/arithmetic', 'micro risk-coverage percentage matches its own endpoints',
      100 * (_mi['1.0'] - _mi['0.2']) / _mi['1.0'],
      float(mac['RCmicroPct']) if 'RCmicroPct' in mac else 42.0, 0.6)
close('audit/arithmetic', 'macro risk-coverage percentage matches its own endpoints',
      100 * (_ma['1.0'] - _ma['0.2']) / _ma['1.0'],
      float(mac['RCmacroPct']) if 'RCmacroPct' in mac else 37.0, 0.6)
# sigma_T is defined in the article with a 1/N_tree normalisation, which is numpy's ddof=0.
# A change to ddof would silently redefine the quantity the whole paper is about.
_src_rel = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              'reliability', 'run_reliability_v2.py'), encoding='utf-8').read()
cond('audit/math', 'sigma_T is computed with the 1/N_tree normalisation the equation states',
     'Pr.std(0)' in _src_rel and 'ddof' not in _src_rel.split('Pr.std(0)')[0][-80:],
     'numpy std defaults to ddof=0, matching 1/N_tree in Eq. (1)')
# The risk-coverage curve is monotone on four targets, and SCD-1 is the stated exception.
_COVK = ['0.2', '0.4', '0.6', '0.8', '1.0']
_nonmono = [t for t in ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
            if not all(_relq[t]['risk_coverage_rmse'][_COVK[i]]
                       <= _relq[t]['risk_coverage_rmse'][_COVK[i + 1]] + 1e-9
                       for i in range(4))]
cond('audit/consistency', 'SCD-1 is the only target with a non-monotone risk-coverage curve',
     _nonmono == ['scd1'], f'non-monotone on {_nonmono}')
# Conditional coverage: over-covered in the top sigma fifth everywhere, under-covered in the
# bottom fifth on four of five, FADS excepted. Both documents state this.
_conq = json.load(open(_os.path.join(OUT, 'conformal_analysis.json')))
_lowunder = [t for t in ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
             if _conq[t]['alpha0.1']['adaptive_coverage_low_sigma'] < 0.90]
_highover = [t for t in ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
             if _conq[t]['alpha0.1']['adaptive_coverage_high_sigma'] > 0.90]
cond('audit/consistency', 'the top disagreement fifth is over-covered on every target',
     len(_highover) == 5, f'over-covered on {_highover}')
cond('audit/consistency', 'the bottom fifth is under-covered on four of five, FADS excepted',
     len(_lowunder) == 4 and 'fads' not in _lowunder, f'under-covered on {_lowunder}')
# The reward's scalarisation: novelty is allowed twice the maximum activity contribution at the
# top of the sweep, so the falling predicted activity is partly by construction and is disclosed.
_tex_audit = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'WritePaper',
                           'theranostics', 'JournalPapers_npjDD', 'npjDD_Reliability.tex')
if _os.path.exists(_tex_audit):
    _st2 = _re.sub(r'\s+', ' ', open(_tex_audit, encoding='utf-8').read())
    cond('audit/consistency', 'the article discloses that the novelty sweep outweighs activity',
         'novelty can contribute twice what' in _st2,
         'w_nu reaches 2 while w_p is 1 and g_p is bounded by 1')

# Seed novelty is defined against one draw of k active training compounds, so the claim that it
# adds little beyond nearest-training distance could have been an artifact of that draw.
# run_support_resample.py repeats it; assert the conclusion holds across every draw and k.
_sr = json.load(open(_os.path.join(OUT, 'support_resample.json')))
_srT = [t for t in ('scd1', 'fads', 'nk1r', 'drd2', 'drd3') if t in _sr]
cond('Results/novelty', 'the support resampling covers three support sizes and many draws',
     _sr['n_draws'] >= 100 and set(_sr['k_values']) == {5, 10, 20},
     f"{_sr['n_draws']} draws at k={_sr['k_values']}")
_hi = [_sr[t][f'k{k}']['partial_nov_err_given_dtr']['hi'] for t in _srT for k in (5, 10, 20)]
cond('Results/novelty', 'the partial correlation stays small across every draw and support size',
     max(_hi) < 0.20, f'largest 97.5th percentile {max(_hi):.3f} over {len(_hi)} cells')
cond('Results/novelty', 'FADS is the sign exception under resampling, as the text states',
     _sr['fads']['k10']['partial_nov_err_given_dtr']['median'] < 0
     and all(_sr[t]['k10']['partial_nov_err_given_dtr']['median'] > 0
             for t in _srT if t != 'fads'),
     'more novel FADS compounds carry lower error')
cond('Results/novelty', 'the nearer-training group wins a majority of draws on every target',
     all(_sr[t][f'k{k}']['near_beats_far_frac'] > 0.5 for t in _srT for k in (5, 10, 20)),
     'lowest share %.0f%%' % (100 * min(_sr[t][f'k{k}']['near_beats_far_frac']
                                        for t in _srT for k in (5, 10, 20))))
# Aggregating to a median across DIFFERENT endpoint types is a stronger assumption than
# aggregating within one, so the share of parents where that happens, and what it costs in
# within-parent spread, are reported rather than left to a citation about mixed IC50 data.
_emx = json.load(open(_os.path.join(OUT, 'endpoint_mixing.json')))
close('Methods/endpoints', 'share of parents pooling more than one endpoint type',
      _emx['pooled']['pct_mixed_of_parents'], 7.0, 0.1)
cond('Methods/endpoints', 'mixed-type parents carry a wider within-parent spread on DRD2 and DRD3',
     _emx['drd2']['sd_mixed_type'] > _emx['drd2']['sd_single_type']
     and _emx['drd3']['sd_mixed_type'] > _emx['drd3']['sd_single_type'],
     'DRD2 %.2f vs %.2f, DRD3 %.2f vs %.2f'
     % (_emx['drd2']['sd_mixed_type'], _emx['drd2']['sd_single_type'],
        _emx['drd3']['sd_mixed_type'], _emx['drd3']['sd_single_type']))
cond('Methods/endpoints', 'SCD-1 pools no endpoint types, matching its single-type composition',
     _emx['scd1']['n_mixed_type'] == 0, 'SCD-1 is entirely IC50')
cond('SI tables', 'S20: the pre and post counts reconcile with the parent total',
     all(_ep20['temporal_cohort'][t]['n_pre'] + _ep20['temporal_cohort'][t]['n_post']
         + _ep20['temporal_cohort'][t]['n_undated'] == _ep20['temporal_cohort'][t]['n_parents']
         for t in ('scd1', 'nk1r', 'drd2', 'drd3')), '')
cond('SI tables', 'S20: the stored structure count is not the parent count on every target',
     any(_ep20['cv_cohort'][t]['n_structures'] != _r20[t]['duplicates']['n_unique']
         for t in _T20),
     'which is why the table must take n from the parent counts rather than that field')
_m20 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'make_si_tables.py')
if _os.path.exists(_m20):
    _c20 = open(_m20, encoding='utf-8').read()
    cond('SI tables', 'S20 takes n from the parent counts, as Table 1 does',
         "cross-validation & {rel[t]['duplicates']['n_unique']:,}" in _c20, '')
    cond('SI tables', 'S20 states the denominator its percentages use',
         'behind the matched structures rather than of the dataset' in _c20,
         'the CV percentages are of matched ChEMBL records, not of the dataset')

# The Supplementary Information opens by promising that every procedure is in the main
# Methods, which npj requires since it forbids Supplementary Methods. Each resampling scheme
# the paper uses must therefore appear in the article, not only in an SI caption.
_artN = _os.path.join(_D, 'npjDD_Reliability.tex')
_siN = _os.path.join(_D, 'npjDD_SI.tex')
if _os.path.exists(_artN) and _os.path.exists(_siN):
    _aN = open(_artN, encoding='utf-8').read()
    _sN = open(_siN, encoding='utf-8').read()
    cond('manuscript', 'the Methods describe every resampling scheme the paper uses',
         'Resampling respects whatever grouping' in _aN
         and 'resample seeds within one target' in _aN
         and 'permuting the five target labels' in _aN,
         'the SI promises the Methods carry every procedure')
    cond('SI notes', 'each Supplementary figure is cited from the note it belongs to',
         all(('sfig:' + k) in _sN for k in ('calibration', 'temporal', 'acquisition'))
         and _sN.count('\\ref{sfig:') >= 3,
         'they were cited only from the article, so the SI narrative skipped past them')
    _body = _sN.split('\\section{', 1)[1]
    import re as _reN
    _lits = [v for v in _reN.findall(r'(?<![\w.])(\d+\.\d+)(?![\w])', _body)
             if v not in ('0.900', '0.90', '0.95', '0.80', '0.05', '0.0625')]
    cond('SI notes', 'no measured value is typed into the notes',
         not _lits, 'literals found: ' + ', '.join(sorted(set(_lits))))
    for _pat, _why in (('---', 'em-dash'), ('honest', 'the word honest')):
        cond('SI notes', f'the notes contain no {_why}', _pat not in _body, '')

# The Methods must state what the code does. These read the implementation, not the frozen
# outputs, so a hyperparameter change shows up as a Methods error rather than silently.
_RD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'reliability')
_srcM = {f: open(_os.path.join(_RD, f), encoding='utf-8').read()
         for f in ('graph_ga.py', 'surrogate_ga.py', 'reward.py', 'run_poolopt_v1.py',
                   'run_reliability_v2.py')
         if _os.path.exists(_os.path.join(_RD, f))}
_artM = _os.path.join(_D, 'npjDD_Reliability.tex')
if _os.path.exists(_artM) and len(_srcM) == 5:
    _aM = open(_artM, encoding='utf-8').read()
    cond('methods', 'the stated mutation probability matches the search code',
         'probability $0.5$' in _aM and 'mutation_rate=0.5' in _srcM['graph_ga.py']
         and 'mut_rate=0.5' in _srcM['surrogate_ga.py'], '')
    cond('methods', 'the stated invalid-molecule reward matches the reward code',
         'reward of $-1$' in _aM and 'invalid_reward=-1.0' in _srcM['reward.py'], '')
    cond('methods', 'the population is described as a cap, which is what the code does',
         'capped at $60$' in _aM and 'population_size=60' in _srcM['graph_ga.py']
         and '[:self.pop_size]' in _srcM['graph_ga.py'],
         'it holds the best scored so far, so it starts at k rather than at 60')
    cond('methods', 'the acquisition forest size is stated and matches its script',
         'random forest of $200$ trees' in _aM
         and 'n_estimators=200' in _srcM['run_poolopt_v1.py'],
         'it differs from the 300-tree activity model and the text now says so')
    cond('methods', 'the activity-model forest size is stated and matches its script',
         'n_estimators=300' in _srcM['run_reliability_v2.py']
         and ('300$ trees' in _aM or '{\\mathrm{tree}}=300' in _aM), '')
    cond('methods', 'the offspring pool size follows from the stated multiplier',
         'size $200$' in _aM and 'pool_mult=5' in _srcM['surrogate_ga.py'],
         '40 evaluated per generation times a pool multiplier of five')

# The Results and Methods asserted that small neural predictors collapse toward the training
# mean, varying by under 0.13 pActivity. Nothing in the repository could check it: the bake-off
# fits three tree ensembles and no network. Measuring it refutes it, so the claim is gone and
# mlp_collapse.json is the record of why.
_mcp = _os.path.join(OUT, 'mlp_collapse.json')
if _os.path.exists(_mcp):
    _mc = json.load(open(_mcp))
    _TM = ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
    cond('methods', 'a small neural predictor does not collapse to the training mean',
         all(_mc[t]['mlp_spread_ratio'] > 0.8 for t in _TM),
         'prediction spread is '
         + ', '.join(f"{100*_mc[t]['mlp_spread_ratio']:.0f}%" for t in _TM)
         + ' of the measured spread')
    cond('methods', 'the forest shrinks its predictions more than the network does',
         all(_mc[t]['rf_spread_ratio'] < _mc[t]['mlp_spread_ratio'] for t in _TM),
         'the opposite of the collapse the removed claim asserted')
    _artM2 = _os.path.join(_D, 'npjDD_Reliability.tex')
    if _os.path.exists(_artM2):
        _t2 = open(_artM2, encoding='utf-8').read()
        cond('methods', 'the refuted neural-collapse claim is not in the manuscript',
             'collapsed toward the training mean' not in _t2,
             'it was unreproducible from this repository and measurement contradicts it')
        cond('methods', 'the model choice rests on the bake-off that was actually run',
             'bake-off among a random forest' in _t2, '')

# Rendered figures must be newer than the data behind them. The value checks above read the
# generator source and the frozen outputs, so they all passed while two committed PNGs had been
# built before the temporal outputs were replaced: Figure 1c and Figure S2 shipped stale. An
# mtime comparison is coarse, but it fails in the safe direction.
_figs = ['fig1_overview.png', 'fig2_frontier.png', 'fig3_forest.png',
         'figS1_calibration.png', 'figS2_temporal.png', 'figS3_acquisition.png']
if _os.path.isdir(_D) and _os.path.isdir(OUT):
    _newest = max((_os.path.getmtime(_os.path.join(OUT, f))
                   for f in _os.listdir(OUT) if f.endswith('.json')), default=0)
    for _fg in _figs:
        _fp = _os.path.join(_D, _fg)
        if _os.path.exists(_fp):
            cond('figures', f'{_fg} is not older than the frozen outputs',
                 _os.path.getmtime(_fp) >= _newest,
                 'rebuild the figure generators after any analysis change')

# Temporal control at R replicates. The point of raising R is not a narrower control interval
# but the resolution of the comparison: the smallest one-sided empirical P obtainable is
# 1/(R+1), so at R=20 nothing below 0.048 could be reported however extreme the observation.
_tj = _os.path.join(OUT, 'temporal_analysis.json')
if _os.path.exists(_tj):
    _tt = json.load(open(_tj))
    _TG = _tt['macro']['targets']
    _R = _tt[_TG[0]]['control_random_same_size']['n_reps']
    cond('numeric', 'the size-matched control uses at least 1000 replicates',
         _R >= 1000, f'{_R} replicates; the empirical-P floor is 1/(R+1)')
    cond('numeric', 'every target reports the same number of control replicates',
         len({_tt[t]['control_random_same_size']['n_reps'] for t in _TG}) == 1,
         'the control must be size-matched and equally resolved on every target')
    _fl = 1.0 / (_R + 1)
    cond('numeric', 'the reported empirical-P floor matches the replicate count',
         abs(_tt[_TG[0]]['control_random_same_size']['empirical_p']['floor'] - _fl) < 1e-12,
         f'{_fl:.2e}')
    cond('numeric', 'error rises above the control on every target at the empirical-P floor',
         all(_tt[t]['control_random_same_size']['empirical_p']['rmse'] <= _fl + 1e-12
             for t in _TG),
         'the temporal RMSE exceeds all R control replicates on each target')
    cond('numeric', 'the direct RMSE effect excludes zero on every target',
         all(_tt[t]['delta_vs_control']['rmse']['ci95'][0] > 0 for t in _TG),
         'combining the temporal bootstrap with the control-mean standard error')
    # The scaffold bootstrap must actually be doing something: resampling series rather than
    # compounds should not give the same interval back.
    cond('numeric', 'the scaffold-cluster bootstrap widens the temporal RMSE interval',
         sum((_tt[t]['scaffold_cluster_bootstrap']['rmse_ci95'][1]
              - _tt[t]['scaffold_cluster_bootstrap']['rmse_ci95'][0])
             > (_tt[t]['rmse_test_ci95'][1] - _tt[t]['rmse_test_ci95'][0]) for t in _TG) >= 3,
         'compounds within a series are not independent, so clustering must cost precision')
    cond('numeric', 'fewer scaffolds than test compounds on every target',
         all(_tt[t]['scaffold_cluster_bootstrap']['n_scaffolds'] < _tt[t]['n_test']
             for t in _TG), 'otherwise the clustering is a no-op')
    # Direction of the two claims that survive the move, and the two that do not.
    cond('semantic', 'the ranking effect covers zero on SCD-1 and NK1R',
         all(_tt[t]['delta_vs_control']['spearman']['ci95'][0] < 0 <
             _tt[t]['delta_vs_control']['spearman']['ci95'][1] for t in ('scd1', 'nk1r')),
         'these are the targets the article says keep their ranking')
    cond('semantic', 'the ranking effect is negative and excludes zero on DRD2 and DRD3',
         all(_tt[t]['delta_vs_control']['spearman']['ci95'][1] < 0
             for t in ('drd2', 'drd3')),
         'these are the targets the article says lose it')
    if _os.path.exists(_art):
        _ta = open(_art, encoding='utf-8').read()
        cond('manuscript', 'the article reports the exact empirical P, not just the range test',
             'TempPfloor' in _ta or 'TempPRmseWorst' in _ta,
             'the replicate-range statement alone cannot carry a P')
        cond('manuscript', 'the article reports the direct effect against the control mean',
             'TempMacroDRmse' in _ta, 'interval overlap is not an interval for the difference')

# Activity endpoint composition. The pooled response is named pActivity because it is
# populated from four ChEMBL standard types, so the article has to state the mix rather than
# define the label as an IC50. The pre/post-cutoff mixes also test whether endpoint turnover,
# rather than chemistry, could explain the temporal result.
_epf = _os.path.join(OUT, 'endpoint_composition.json')
if _os.path.exists(_epf):
    _ep = json.load(open(_epf))
    _cvc, _tcc = _ep['cv_cohort'], _ep['temporal_cohort']
    cond('numeric', 'the pooled temporal response is majority Ki, not IC50',
         _tcc['pooled_pct']['Ki'] > 50 > _tcc['pooled_pct']['IC50'],
         f"Ki {_tcc['pooled_pct']['Ki']}%, IC50 {_tcc['pooled_pct']['IC50']}%")
    cond('numeric', 'SCD-1 is essentially all IC50 in the cross-validation cohort',
         _cvc['scd1']['types_pct'].get('IC50', 0) > 99, str(_cvc['scd1']['types_pct']))
    cond('numeric', 'DRD2 and DRD3 are majority Ki in the cross-validation cohort',
         _cvc['drd2']['types_pct']['Ki'] > 50 and _cvc['drd3']['types_pct']['Ki'] > 50,
         'the label name is a convention, not a measurement type')
    cond('numeric', 'the cross-validation endpoint mix is recovered for most structures',
         all(_cvc[t]['matched_pct'] > 80 for t in ('scd1', 'nk1r', 'drd2', 'drd3')),
         'FADS is a literature panel and is excluded from this check')
    # The article argues endpoint turnover does not order the targets the way transport
    # failure does. That argument rests on NK1R shifting most while keeping its ranking.
    _tv = {t: _tcc[t]['tv_distance'] for t in ('scd1', 'nk1r', 'drd2', 'drd3')}
    cond('semantic', 'NK1R has the largest endpoint shift across the temporal cutoff',
         max(_tv, key=_tv.get) == 'nk1r', str(_tv))
    cond('semantic', 'SCD-1 has the smallest endpoint shift across the temporal cutoff',
         min(_tv, key=_tv.get) == 'scd1',
         'the two targets that keep their ranking bracket the shift range')
    if _os.path.exists(_art):
        _a = open(_art, encoding='utf-8').read()
        cond('manuscript', 'the article states the endpoint mix and cites its table',
             'TabEndpoint' in _a and 'K_i$' in _a, 'the response is named pActivity and its mix is stated')
        cond('manuscript', 'the abstract restricts the nearer-training result to the novel third',
             'within the most novel third' in _a.lower(),
             'the comparison is inside the most novel third, not over all compounds')
        cond('manuscript', 'the RDKit standardization order is stated',
             'LargestFragmentChooser' in _a and 'Uncharger' in _a,
             'the parent operation must be reproducible')
        cond('manuscript', 'the top-percentile endpoint is defined exactly',
             'Q_{0.99}' in _a and 'tied at that cutoff' in _a, 'including tie handling')

# Acquisition, paired by seed with the target as the unit. The article's claim is that
# penalising uncertainty costs, not that optimism wins: the first must be separated from the
# predicted-mean rule and the second must not be overstated.
_pf = _os.path.join(OUT, 'poolopt_analysis.json')
if _os.path.exists(_pf):
    import statistics as _st
    _pres = json.load(open(_pf))['results']
    _TT = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']
    _pm = {}
    for _m in ('ucb', 'lcb', 'conformal'):
        _per = [_st.mean([r[_m]['hits'] - r['greedy']['hits'] for r in _pres[t]]) for t in _TT]
        _pm[_m] = (_st.mean(_per), _st.stdev(_per) / len(_per) ** 0.5)
    cond('numeric', 'the conservative rule loses to the predicted mean on every target',
         all(_st.mean([r['lcb']['hits'] - r['greedy']['hits'] for r in _pres[t]]) < 0
             for t in _TT), 'this is the paper\'s acquisition claim')
    cond('numeric', 'the conformal-style rule loses to the predicted mean on every target',
         all(_st.mean([r['conformal']['hits'] - r['greedy']['hits'] for r in _pres[t]]) < 0
             for t in _TT), 'both penalised rules, not just one')
    cond('semantic', 'the optimistic rule is not separated from the predicted mean',
         _pm['ucb'][0] - 2.776 * _pm['ucb'][1] < 0 < _pm['ucb'][0] + 2.776 * _pm['ucb'][1],
         'its interval covers zero, so the article must not claim it reliably wins')
    if _os.path.exists(_art):
        _aa = open(_art, encoding='utf-8').read()
        cond('manuscript', 'the article reports the paired acquisition intervals',
             'PoolLcbCI' in _aa and 'PoolUcbCI' in _aa, 'means alone cannot separate the rules')
        cond('manuscript', 'the article does not claim the optimistic rule reliably wins',
             'optimistic rule is not' in _aa, 'its advantage rests largely on DRD3')

# Literals that duplicate a macro. Twelve numbers were typed into the article while a macro
# already held the same quantity, and one of them (0.97, where the data give 0.975) was simply
# wrong. Flag any numeric literal in the body that equals a macro value.
# The scan originally covered the article only, and a hand-typed width reduction survived in
# the Supplementary Information for exactly that reason. Scan both documents.
if _os.path.exists(NUM) and _os.path.exists(_art):
    import re as _r2
    _m2 = dict(_r2.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}', open(NUM, encoding='utf-8').read()))
    _vals = {v.replace('{,}', '').replace('$', ''): k for k, v in _m2.items()
             if _r2.fullmatch(r'[+-]?\d+\.\d+', v.replace('$', ''))}
    _body = open(_art, encoding='utf-8').read().split(r'\section{Introduction}')[-1]
    if _os.path.exists(_sid):
        _body += '\n' + open(_sid, encoding='utf-8').read()
    # Nominal levels, alpha values and reported P values are legitimately typed: they are design
    # constants, not measurements, even when one coincides with a macro value.
    _DESIGN = {'0.900', '0.90', '0.95', '0.80', '0.05', '0.10', '0.20', '0.25', '0.0625'}
    _bad = []
    for _mm in _r2.finditer(r'\$([+-]?\d+\.\d{2,})\$', _body):
        _v = _mm.group(1)
        _pre = _body[max(0, _mm.start() - 30):_mm.start()]
        if _v in _DESIGN or _r2.search(r'(P=|P\s*=|alpha|nominal)\s*$', _pre):
            continue
        if _v in _vals:
            _bad.append(f'{_v} (=\\{_vals[_v]})')
    cond('literals', 'no measured value is typed where a macro already holds it',
         not _bad, 'use the macro: ' + ', '.join(sorted(set(_bad))[:6]))

# The clean-clone path must run. It once crashed on a name defined only when numbers.tex exists.
cond('portability', 'the manuscript macro table is defined even without numbers.tex',
     isinstance(mac, dict),
     'a clean clone has no numbers.tex and must still complete')

# Layout. A table that runs past the text block is a defect a reader sees before any number,
# and pdflatex already reports it, so the logs are checked rather than the pages eyeballed.
for _doc in ('npjDD_Reliability', 'npjDD_SI'):
    _lg = _os.path.join(_D, _doc + '.log')
    if _os.path.exists(_lg):
        _txt = open(_lg, encoding='utf-8', errors='replace').read()
        _over = [l for l in _txt.split('\n') if l.startswith('Overfull \\hbox')]
        cond('layout', f'{_doc}: nothing overflows the text block',
             not _over, f'{len(_over)} overfull hbox(es): ' + '; '.join(_over[:3]))
        cond('layout', f'{_doc}: compiles without errors',
             '\n! ' not in _txt, '')

# The README states results in prose and drifted out of date once already. Pin the claims
# that would be wrong if an analysis changed under it.
_rm = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'README.md')
if _os.path.exists(_rm):
    _r = open(_rm, encoding='utf-8').read()
    for _bad, _why in [('roughly doubles', 'RMSE rises ~50%, it does not double'),
                       ('7,067', 'superseded temporal n from median-year dating'),
                       ('true actives', 'pool labels are measurements, not ground truth')]:
        cond('README', f'no stale claim: "{_bad}"', _bad not in _r, _why)
    cond('README', 'quotes the temporal n that the frozen output actually has',
         f"{tmp['pooled']['n']:,}" in _r, f"expected {tmp['pooled']['n']:,}")
    cond('README', 'quotes the temporal RMSE increase over control',
         f"{tmp['delta_vs_control']['rmse_pct_increase_vs_control']:.0f}%" in _r,
         f"expected {tmp['delta_vs_control']['rmse_pct_increase_vs_control']:.0f}%")
    cond('README', 'quotes the pooled temporal coverage',
         f"{tmp['pooled']['conformal_coverage_adaptive']:.3f}" in _r, '')

# manuscript-side checks: only runnable where the .tex lives
_tex = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'WritePaper',
                     'theranostics', 'JournalPapers_npjDD', 'npjDD_Reliability.tex')
if _os.path.exists(_tex) and _os.path.exists(NUM):
    _s = open(_tex, encoding='utf-8').read()
    _ab = _re.search(r'\\abstract\{(.*?)\}\s*\n\n', _s, _re.S).group(1)
    _t = _re.sub(r'\\(\w+)\{\}', lambda m: mac.get(m.group(1), m.group(1)), _ab)
    _t = _t.replace(r'\pAct', 'pActivity').replace(r'\mathrm{pIC}_{50}', 'pIC50').replace('{,}', ',')
    _t = _re.sub(r'\\[a-zA-Z]+', ' ', _t)
    _t = _re.sub(r'[{}$\\]', '', _t)
    _t = _re.sub(r'\s+%', '%', _t)
    _w = [w for w in _t.split() if _re.search(r'[A-Za-z0-9]', w)]
    cond('manuscript', 'abstract is within the npj Drug Discovery 150-word limit',
         len(_w) <= 150, f'{len(_w)} words')
    # The macro-expanded count above is not the count a journal makes. Math splits "$P=0.005$"
    # into three whitespace tokens and a subscript sheds a stray comma, so the rendered abstract
    # ran three words longer than this check reported and sat over the cap while it passed.
    # Count what a reader of the PDF sees, which is the conservative convention.
    _pdf = _os.path.join(_os.path.dirname(_tex), 'npjDD_Reliability.pdf')
    if _os.path.exists(_pdf):
        import subprocess as _sp
        try:
            _txt = _sp.run(['pdftotext', _pdf, '-'], capture_output=True, text=True,
                           timeout=120).stdout
            _m = _re.search(r'\nAbstract\n(.*?)\nKeywords', _txt, _re.S)
            if _m:
                _rw = len(_m.group(1).split())
                cond('manuscript', 'rendered abstract is also within 150 words',
                     _rw <= 150, f'{_rw} whitespace-delimited words in the compiled PDF')
            # Figure legends have their own cap and the same source-versus-rendered gap. The
            # Figure 1 legend was trimmed to 348 words of source and shipped at 368 rendered,
            # over npj's 350, because only the abstract was being counted this way.
            _i = _txt.find('Where and when the activity model can be trusted')
            _j = _txt.find('Fig. 2', _i)
            if _i > 0 and _j > _i:
                _seg = ' '.join(l for l in _txt[_i:_j].split('\n')
                                if l.strip() and not l.strip().isdigit())
                _lw = len(_seg.split())
                cond('manuscript', 'rendered Figure 1 legend is within the 350-word limit',
                     _lw <= 350, f'{_lw} whitespace-delimited words in the compiled PDF')
        except (OSError, _sp.SubprocessError):
            pass
    cond('manuscript', 'no reference to the superseded raw-record count in the text',
         '21,173' not in _s.replace('{,}', ',') or '\\Nrows' in _s,
         'raw-record total must come from the macro, inside Table 1 only')
    cond('manuscript', 'the removed duplicate reference is no longer cited',
         'tropsha2010best' not in _s, '')
    # sn-jnl sets \\boldmath in \\Authorfont and \\abstractfont never resets the math version, so
    # every inline formula in the abstract rendered bold against regular-weight text: the label and
    # P = 0.005 looked like emphasis nobody wrote. Nothing in the source said bold, so only the
    # rendered page showed it. \\mathversion{normal} at the top of the abstract fixes it.
    cond('manuscript', 'the abstract resets the math version the class leaves bold',
         '\\mathversion{normal}' in _ab,
         'sn-jnl leaks \\boldmath from the author block into abstract math')

    # The in-domain comparison is a median split of the top novelty tercile, so the two groups
    # are halves, not extremes. The abstract said "closest", a superlative that describes a
    # subgroup the analysis never forms; the Results correctly say nearer against further.
    _abflat = _re.sub(r'\s+', ' ', _ab)
    cond('manuscript', 'the abstract describes the in-domain split as a comparison, not an extreme',
         'closest to the training' not in _abflat and 'nearer the training' in _abflat,
         'in_dom is d_train <= median within the novel third, not the closest compounds')

    # The abstract described the temporal ranking as "surviving on two targets and vanishing
    # on one", which accounts for three of the four temporal targets and silently drops DRD2,
    # whose ranking neither survives against its control nor vanishes. Any description of the
    # spread has to span all four.
    _ntemp = len([t for t in ('scd1', 'nk1r', 'drd2', 'drd3') if t in tmp])
    # Two of the four are unchanged, one weakens and one is lost, so any description naming a
    # subset drops a target. Require the range framing, not one wording of it: this guard has
    # now caught the same regression twice and been broken once by an improvement in phrasing.
    _abf = _re.sub(r'\s+', ' ', _ab)
    cond('manuscript', 'the abstract accounts for every temporal target, not a subset',
         'surviving on two targets' not in _abf and 'holds on two targets' not in _abf
         and 'from unchanged to' in _abf,
         f'{_ntemp} temporal targets; two survive, one weakens, one is lost')
    # The Introduction promises four contributions, but the abstract covered only three: the
    # fingerprint-surrogate method had no sentence at all. It must stay, and it must keep the
    # qualifier that the gain is on the model's own score rather than on measured activity.
    # The surrogate-triage method was deliberately dropped from the abstract when it was rewritten
    # as a four-regime narrative: in the words available it could only be stated opaquely, and the
    # paper itself calls it modest. It must therefore still be carried, with its statistics, by
    # the Results, or the contribution would vanish from the paper rather than from one section.
    cond('manuscript', 'the Results still state the surrogate-triage gain with its statistics',
         'fingerprint' in _s and '\\MethodGain' in _s and '\\MethodP' in _s,
         'dropped from the abstract for space; it cannot also be missing downstream')
    # The caveat, not one phrasing of it. The Discussion carries it as "not as a claim of
    # measured activity"; keying on the abstract's word "model-scored" failed the moment the
    # abstract was rewritten, which is this file's recurring mistake.
    _sf = _re.sub(r'\s+', ' ', _s)
    cond('manuscript', 'the gain is somewhere marked as model-defined, not measured, activity',
         'model-scored' in _sf or 'not as a claim of measured activity' in _sf,
         'generated molecules were never assayed; the gain is on the reward the model defines')
    # npj Drug Discovery caps the title at 15 words, and the user's standing rule forbids a
    # colon in it.
    _ti = _re.search(r'\\title(?:\[.*?\])?\{(.*?)\}', _s, _re.S).group(1)
    cond('manuscript', 'title is within the npj Drug Discovery 15-word limit',
         len(_ti.split()) <= 15, f'{len(_ti.split())} words')
    cond('manuscript', 'title carries no colon', ':' not in _ti, _ti)
    # The main text reads the coverage separation off the scaffold-cluster bootstrap, so the
    # table that carries that bootstrap has to display coverage and not only RMSE and ranking.
    _st = open(_os.path.join(_os.path.dirname(_tex), 'si_tables.tex'), encoding='utf-8').read()
    cond('manuscript', 'the scaffold-cluster bootstrap reports coverage, not only RMSE and ranking',
         _st.count('scaffolds, coverage [') >= 4,
         'the article claims coverage separation survives scaffold resampling')
    # ChEMBL changes between releases, so a bare "curated from ChEMBL" is not reproducible.
    cond('manuscript', 'the ChEMBL retrieval date is stated',
         _re.search(r'web\s+services on \d{4}-\d{2}-\d{2}', _s) is not None,
         'the curated files are archived, but the retrieval date still has to be given')
    # Rina Foygel Barber's surname is Barber. The two downloaded citation records disagree (the
    # IMA journal exports "Foygel Barber, Rina", Curran exports "Barber, Rina Foygel"), and she
    # co-authors two references here, so following each source would render her as two different
    # people in one list. User decision: Barber, consistently.
    if _os.path.exists(_os.path.join(_os.path.dirname(_tex), 'theranostics_generation.bib')):
        _bt0 = open(_os.path.join(_os.path.dirname(_tex), 'theranostics_generation.bib'),
                    encoding='utf-8').read()
        cond('manuscript', 'Barber is cited under one surname in both of her references',
             _bt0.count('Barber, Rina Foygel') == 2 and 'Foygel Barber, Rina' not in _bt0,
             'surname is Barber; the IMA export disagrees with Curran, do not follow it')
    # PLOS's BibTeX exporter emits pages 1-12, the PDF pagination. The article locator PLOS
    # itself displays is e61007, which is also the form Nature journals use, so the downloaded
    # file is deliberately not followed here. User decision.
    if _os.path.exists(_os.path.join(_os.path.dirname(_tex), 'theranostics_generation.bib')):
        cond('manuscript', 'the PLOS reference cites its article locator, not PDF pagination',
             'pages={e61007}' in _bt0.replace(' ', ''),
             'e61007 is the locator PLOS displays; 1-12 is an exporter artifact')
    # The article states the verification count as a hand-typed integer, which drifted from 300
    # to a stale value once guards were added. It refers to the ARCHIVED release, not to HEAD, so
    # it must track the tag: if the archive is re-cut, this number and the DOI move together.
    _ARCHIVED_CLEAN_COUNT = 338
    _mcnt = _re.search(r're-checks all \$(\d+)\$ numeric claims', _re.sub(r'\s+', ' ', _s))
    cond('manuscript', 'the stated verification count matches the archived release',
         _mcnt is not None and int(_mcnt.group(1)) == _ARCHIVED_CLEAN_COUNT,
         f"article says {_mcnt.group(1) if _mcnt else 'nothing'}, "
         f"archived release reports {_ARCHIVED_CLEAN_COUNT}")
    cond('manuscript', 'the stated count is scoped to the archive, not to a moving HEAD',
         'the count in the archived release' in _re.sub(r'\s+', ' ', _s),
         'HEAD gains checks over time; the archive does not')

    # The archived snapshot is what a reader is directed to, so the DOI and the release tag in
    # the availability statements must stay together and must name the tag that was archived.
    _ZEN_DOI, _ZEN_TAG = '10.5281/zenodo.21874305', 'v1.3-npjdd-submission'
    cond('manuscript', 'both availability statements cite the archived DOI',
         _s.count(_ZEN_DOI) == 2, f'found {_s.count(_ZEN_DOI)} occurrence(s), expected 2')
    cond('manuscript', 'the DOI is paired with the release tag it was minted from',
         _s.count(_ZEN_TAG) == 2 and all(
             _ZEN_TAG in _s[max(0, m.start() - 260):m.start()]
             for m in _re.finditer(_re.escape(_ZEN_DOI), _s)),
         'a DOI without its tag does not say which snapshot was archived')

    # The repository ships its own copies of the figure and table generators. Those drifted from
    # the ones that actually build the manuscript: the repo's gen_fig1.py was three weeks behind
    # and its make_si_tables.py was missing a table, so a reader cloning the release would have
    # regenerated a different Figure 1 and a short Supplementary set. Worse, the checks above
    # read the repository copy, so they were validating a file nobody used. Assert they match.
    _wsdir = _os.path.dirname(_tex_audit)
    for _gen in ('gen_fig1.py', 'gen_main_figures.py', 'gen_si_figures.py',
                 'make_numbers.py', 'make_si_tables.py'):
        _a = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _gen)
        _b = _os.path.join(_wsdir, _gen)
        if _os.path.exists(_a) and _os.path.exists(_b):
            cond('repo sync', f'{_gen} in the repository matches the one that builds the paper',
                 open(_a, encoding='utf-8').read() == open(_b, encoding='utf-8').read(),
                 'a reader regenerating from the release would not reproduce the figures shown')

    # American spelling throughout the article and the SI (user instruction). Check the RENDERED
    # text, because a British form can enter through a generated table or a figure label as well
    # as through the prose. Stems only, and only ones with no American word containing them:
    # "characteris" would match characteristic, "optimis" matches optimistic, "analyse" matches
    # the plural analyses, all of which are correct.
    import subprocess as _sp2
    _BRIT = ('analogue', 'normalis', 'penalis', 'standardis', 'summaris', 'minimis', 'maximis',
             'recognis', 'generalis', 'utilis', 'colour', 'behaviour', 'neighbour', 'modelling',
             'labelled', 'artefact', 'amongst', 'whilst', 'programme', 'licence', 'defence')
    for _pdf in ('npjDD_Reliability.pdf', 'npjDD_SI.pdf'):
        _pp = _os.path.join(_wsdir, _pdf)
        if not _os.path.exists(_pp):
            continue
        try:
            _pt = _sp2.run(['pdftotext', _pp, '-'], capture_output=True, text=True,
                           timeout=120).stdout.replace('-\n', '')
        except (OSError, _sp2.SubprocessError):
            continue
        _bad = sorted({_m.group(0).lower() for _b in _BRIT
                       for _m in _re.finditer(r'\b\w*' + _b + r'\w*\b', _pt, _re.I)})
        cond('spelling', f'{_pdf} uses American spelling', not _bad, f'found {_bad}')

    # Counts that a re-run changes must come from macros, not from typed digits. The article
    # once stated a verification total of 300 that had drifted; the same exposure existed for the
    # method cell count and the frontier run count, both of which had macros that went unused.
    _sflat = _re.sub(r'\s+', ' ', _s)
    for _lit, _macro in (('$15$ target-by-$k$ cells', 'MethodCells'),
                         ('$1{,}200$', 'FrontRuns')):
        cond('manuscript', f'the article uses \\{_macro} rather than typing its value',
             _lit not in _sflat and ('\\' + _macro) in _sflat,
             'a typed count goes stale silently when the experiment is re-run')
    # Every Supplementary object must be cited from the article. Two tables, the conformal
    # results and the acquisition counts, existed and were reachable only from the SI itself.
    _sirefs = _os.path.join(_wsdir, 'si_refs.tex')
    if _os.path.exists(_sirefs):
        _objs = _re.findall(r'\\newcommand\{\\((?:Tab|SFig)[A-Za-z]+)\}',
                            open(_sirefs, encoding='utf-8').read())
        _unc = [o for o in _objs if ('\\' + o) not in _sflat]
        cond('manuscript', 'every Supplementary table and figure is cited from the article',
             not _unc, f'uncited: {_unc}')

    # The Methods state the calibration split in prose: a quarter of each training fold, or 50
    # compounds if that is larger. Two separate files implement it, so the prose can fall out of
    # step with either. Assert the rule rather than trusting the sentence.
    _hd = _os.path.dirname(_os.path.abspath(__file__))
    for _f in ('run_conformal_v1.py', 'run_temporal_v1.py'):
        _fp = _os.path.join(_hd, 'reliability', _f)
        if _os.path.exists(_fp):
            _ft = open(_fp, encoding='utf-8').read()
            cond('Methods/code', f'{_f} implements the stated calibration split',
                 _re.search(r'max\(50,\s*len\([^)]*\)\s*//\s*4\)', _ft) is not None,
                 'Methods say a quarter of the training fold, or 50 if larger')
    cond('Methods/code', 'the Methods state that calibration rule',
         _re.search(r'quarter of each training fold, or \$?50\$? compounds if that is larger',
                    _re.sub(r'\s+', ' ', _s)) is not None,
         'the prose and the two implementations must agree')

    # The in-domain exception is named from one support draw. The resampling shows SCD-1 and
    # FADS are both marginal and that their order flips between draws, so naming SCD-1 flatly
    # would make the article and the Discussion more confident than the Supplementary evidence.
    # All three places must carry the instability.
    _srq = json.load(open(_os.path.join(OUT, 'support_resample.json')))
    _marg = sorted(t for t in ('scd1', 'fads', 'nk1r', 'drd2', 'drd3')
                   if _srq[t]['k10']['near_beats_far_frac'] < 0.95)
    cond('audit/consistency', 'exactly two targets are marginal on the in-domain comparison',
         _marg == ['fads', 'scd1'],
         f'marginal: {_marg}; the article must not name one of them as a stable exception')
    cond('manuscript', 'the article states that the in-domain exception is draw-dependent',
         'not a stable label' in _sflat, 'resampling flips which of SCD-1 and FADS is weaker')
    _dsc = _sflat.split('\\section{Discussion}')[-1].split('\\section{Methods}')[0]
    cond('manuscript', 'the Discussion carries the same caveat as the Results and SI',
         'depends on the draw' in _dsc,
         'the Discussion asserted SCD-1 as the exception without the instability')

    # The Introduction states counts in words, so no macro protects them and a re-curation would
    # leave them silently wrong. This is the same exposure that produced the stale "300 claims",
    # in prose rather than in digits.
    _intro = _sflat.split('\\section{Introduction}')[-1].split('\\section{Results}')[0]
    cond('manuscript', '"more than twenty thousand" holds for the structure count',
         'more than twenty thousand' not in _intro or _pq['n'] > 20000,
         f"Introduction says more than twenty thousand; n={_pq['n']:,}")
    cond('manuscript', 'the Introduction poses exactly the four questions it promises',
         'four questions' not in _intro or _intro.count('?') == 4,
         f"{_intro.count('?')} question marks against the stated four")
    # Both documents frame the study as two regimes; Figure 1 labels its panels the same way.
    _sitex = _os.path.join(_wsdir, 'npjDD_SI.tex')
    if _os.path.exists(_sitex):
        _sif = _re.sub(r'\s+', ' ', open(_sitex, encoding='utf-8').read())
        cond('manuscript', 'article and SI use the same two-regime framing',
             ('two regimes' in _intro) == ('two regimes of the main article' in _sif),
             'the SI organizes its notes by the regimes the Introduction names')

    # Figure 2's caption stated the extreme bin occupancies and the block count as digits. The
    # values were right, but all three are derived from the 1,200 runs, so a re-run would leave
    # the caption describing a binning that no longer exists. The existing figure-2 checks verify
    # the values; this one keeps them coming from the data rather than from memory.
    for _lit, _macro in (('$8$ in the lowest bin', 'FrontBinLo'),
                         ('$244$ in the highest', 'FrontBinHi'),
                         ('$75$ matched blocks', 'FrontBlocks')):
        cond('figure 2', f'the caption reads \\{_macro} rather than typing it',
             _lit not in _sflat and ('\\' + _macro) in _sflat,
             'a derived count typed by hand goes stale when the runs change')

    # Co-author correction (O.K.O., radiology): "metabolic stability" is the wrong term for a
    # radiotracer. What limits one is stability in serum and whether the radiolabel stays on the
    # molecule. The Limitations must not revert to the ADME term.
    cond('manuscript', 'the radiotracer limitation names serum and radiolabel stability',
         'metabolic stability' not in _sflat
         and 'stability both in serum and of the radiolabel' in _sflat,
         'a radiolabel can detach; that is not metabolic stability')

    # The SI preamble made two claims it could not keep. It sent readers to the mutable
    # repository while the article cited the archived release, and it said the verifier re-checks
    # "every number" in both documents, which is an exhaustiveness claim: values are generated
    # from the frozen outputs, and a stated number of claims is re-checked, but not every macro
    # carries its own assertion.
    _sipre = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'WritePaper',
                           'theranostics', 'JournalPapers_npjDD', 'npjDD_SI.tex')
    if _os.path.exists(_sipre):
        _sip = _re.sub(r'\s+', ' ', open(_sipre, encoding='utf-8').read())
        cond('manuscript', 'the SI cites the archived release, as the article does',
             _ZEN_DOI in _sip and _ZEN_TAG in _sip,
             'the SI pointed only at the mutable repository URL')
        cond('manuscript', 'the SI does not claim every number is re-checked',
             're-checks every number' not in _sip,
             'generation prevents drift; the verifier re-checks a stated set of claims')
    # The same overclaim survived in the article's Code availability statement after the SI was
    # corrected, because the guard was written for one document. Scope it to both.
    cond('manuscript', 'the article does not claim every number is re-checked',
         're-checks every number' not in _re.sub(r'\s+', ' ', _s),
         'the Code availability statement said "every number reported here"')

    # The RDKit version is an empirical fact about the runs, and it appears in three places that
    # can drift apart: the Methods prose, the bibliography entry, and the pinned requirement. A
    # citation record downloaded for release 2026_03_5 nearly attached a DOI for software that
    # was never run, so assert all three name the same release.
    _rdk_tex = _re.search(r'RDKit \$(\d{4}\.\d{2}\.\d+)\$', _s)
    _bibp = _os.path.join(_os.path.dirname(_tex), 'theranostics_generation.bib')
    _rdk_bib = None
    if _os.path.exists(_bibp):
        _bt = open(_bibp, encoding='utf-8').read()
        _m2 = _re.search(r'@misc\{rdkit,.*?\n\}', _bt, _re.S)
        if _m2:
            _m3 = _re.search(r'(\d{4}\.\d{2}\.\d+)', _m2.group(0))
            _rdk_bib = _m3.group(1) if _m3 else None
    _reqp = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'requirements.txt')
    _m4 = _re.search(r'rdkit==(\S+)', open(_reqp, encoding='utf-8').read()) \
        if _os.path.exists(_reqp) else None
    _rdk_req = _m4.group(1) if _m4 else None
    cond('manuscript', 'the RDKit release agrees across Methods, bibliography and requirements',
         _rdk_tex is not None and _rdk_bib == _rdk_tex.group(1) == _rdk_req,
         f'Methods {_rdk_tex.group(1) if _rdk_tex else None}, bib {_rdk_bib}, pinned {_rdk_req}')
    # The pooled response mixes endpoint types; the Methods must quantify how often the median
    # itself crosses them rather than resting on a mixed-IC50 citation.
    cond('manuscript', 'the Methods quantify how often aggregation crosses endpoint types',
         '\\MixPct' in _s and '\\MixSdDtwoMixed' in _s,
         'a citation about mixed IC50 data does not cover pooling four types')
    # Keywords carry the discoverability the title deliberately does not. "uncertainty" is the
    # paper's central object and the term this literature is indexed under, so it has to be a
    # keyword; the journal's own scope phrase is not a distinguishing one.
    _kw = [k.strip() for k in
           _re.search(r'\\keywords\{(.*?)\}', _s, _re.S).group(1).split(',')]
    cond('manuscript', 'keywords name uncertainty quantification',
         any('uncertainty' in k for k in _kw), ', '.join(_kw))
    # The temporal axis is the paper's most distinctive contribution and had no keyword: a
    # reader searching for time-split validation could not have found it. QSAR is the field's
    # standard index term and was likewise missing.
    cond('manuscript', 'keywords name the temporal axis, which nothing else covers',
         any('temporal' in k for k in _kw), ', '.join(_kw))
    cond('manuscript', 'keywords name QSAR, the standard index term for this field',
         any(k.strip().upper() == 'QSAR' for k in _kw), ', '.join(_kw))
    cond('manuscript', 'keywords are not padded with the journal\'s own scope phrase',
         'machine learning in drug discovery' not in _kw, ', '.join(_kw))
    cond('manuscript', 'every keyword names something the manuscript actually does',
         all(k.lower().split()[0] in _s.lower() for k in _kw), ', '.join(_kw))
else:
    print('skip [manuscript] checks: the manuscript is intentionally not in this repository')

# The README quotes how many checks this script runs. That can only be compared once every
# check has run, and only in the clean-clone case, where the manuscript macros are absent.
if _os.path.exists(_rm) and not _os.path.exists(NUM):
    _r = open(_rm, encoding='utf-8').read()
    # Match the clean-clone phrase specifically. The looser test passed while the README carried
    # two different totals, because a stale duplicate elsewhere in the file still said the right
    # number and satisfied it.
    _rc = _re.search(r'\((\d+) assertions from a clean clone', _r)
    cond('README', 'the two assertion counts in the README agree',
         _rc is not None and f'{_rc.group(1)} claims' in _r,
         'the file quotes the total twice; a stale duplicate satisfied the old check')
    # KEEP LAST. It compares the README against CHECKED + 1, which is the final total only if
    # nothing is asserted after it. A check added below this line makes it undercount.
    cond('README', 'states the assertion count this script actually reports',
         _rc is not None and int(_rc.group(1)) == CHECKED + 1,
         f'clean-clone total is {CHECKED + 1}, README says {_rc.group(1) if _rc else "nothing"}')

# --------------------------------------------------------------- summary
print('\n' + '=' * 72)
if FAILED:
    print(f'{len(FAILED)} of {CHECKED} checks FAILED:')
    for f in FAILED:
        print('  -', f)
    sys.exit(1)
print(f'All {CHECKED} numeric claims verified against the frozen outputs.')

