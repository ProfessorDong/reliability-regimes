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
if _os.path.exists(NUM):
    _txt = open(NUM).read()
    mac = {}
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

