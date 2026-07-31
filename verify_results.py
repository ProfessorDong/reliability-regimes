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
cond('Results/negatives (Table S12)', 'retrained model ranks held-out actives 6.8-8.1 pIC50',
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
        cond('figure 1', 'the caption describes the control bar as the replicate spread',
             'central $95\\%$ of its' in _cap or 'central 95' in _cap,
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
    ('Activities were pooled to $\\mathrm{pIC}_{50}=', 'state the endpoint mix; the pooled response is mostly Ki'),
]
for _n, _f in _SRC.items():
    if not _os.path.exists(_f):
        continue
    _txt = open(_f, encoding='utf-8').read()
    for _bad, _why in _BANNED:
        cond('phrasing', f'{_n}: no "{_bad}"', _bad not in _txt, _why)

# Claims that appear in more than one place must agree everywhere they appear.
if all(_os.path.exists(f) for f in _SRC.values()):
    _all = '\n'.join(open(f, encoding='utf-8').read() for f in _SRC.values())
    cond('phrasing', 'SCD-1, not FADS, is named as the in-domain exception wherever it is named',
         'NinDomExcept' in _all and 'where the in-domain novelty comparison does not hold' not in _all,
         'the exception must come from the macro the data sets')
    cond('phrasing', 'the Methods no longer claim four targets have one record per structure',
         'one record per structure' not in _all, 'Table S1 shows collapses in four of five')

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
         'a reader meets the label pIC50 first in Table 1')

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

# Activity endpoint composition. The pooled response is named pIC50 by convention but is
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
             'TabEndpoint' in _a and 'K_i$' in _a, 'pIC50 is named as a convention')
        cond('manuscript', 'the abstract restricts the nearer-training result to the novel third',
             'within the most novel third' in _a,
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
    _t = _t.replace(r'\mathrm{pIC}_{50}', 'pIC50').replace('{,}', ',')
    _t = _re.sub(r'\\[a-zA-Z]+', ' ', _t)
    _t = _re.sub(r'[{}$\\]', '', _t)
    _t = _re.sub(r'\s+%', '%', _t)
    _w = [w for w in _t.split() if _re.search(r'[A-Za-z0-9]', w)]
    cond('manuscript', 'abstract is within the npj Drug Discovery 150-word limit',
         len(_w) <= 150, f'{len(_w)} words')
    cond('manuscript', 'no reference to the superseded raw-record count in the text',
         '21,173' not in _s.replace('{,}', ',') or '\\Nrows' in _s,
         'raw-record total must come from the macro, inside Table 1 only')
    cond('manuscript', 'the removed duplicate reference is no longer cited',
         'tropsha2010best' not in _s, '')
else:
    print('skip [manuscript] checks: the manuscript is intentionally not in this repository')

# The README quotes how many checks this script runs. That can only be compared once every
# check has run, and only in the clean-clone case, where the manuscript macros are absent.
if _os.path.exists(_rm) and not _os.path.exists(NUM):
    _r = open(_rm, encoding='utf-8').read()
    cond('README', 'states the assertion count this script actually reports',
         f'{CHECKED + 1} claims' in _r or f'{CHECKED + 1} assertions' in _r,
         f'clean-clone total is {CHECKED + 1}')

# --------------------------------------------------------------- summary
print('\n' + '=' * 72)
if FAILED:
    print(f'{len(FAILED)} of {CHECKED} checks FAILED:')
    for f in FAILED:
        print('  -', f)
    sys.exit(1)
print(f'All {CHECKED} numeric claims verified against the frozen outputs.')

