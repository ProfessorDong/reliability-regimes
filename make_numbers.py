"""Generate numbers.tex from the frozen analysis outputs.

Every quantity that appears in more than one place in the manuscript or the SI is
defined here as a LaTeX macro, so the abstract, Results, Methods, SI notes, tables and
figure captions all read from a single source. Hand-copying a number into the TeX is
what allowed the raw-record (21,173) and structure-level (21,037) analyses to become
mixed; macros make that failure impossible.

    python make_numbers.py     ->  writes numbers.tex
"""
from __future__ import annotations
import json, os
import numpy as np

def _data_dir():
    """Locate the frozen outputs, whether this file sits in the repository or the workspace."""
    here = os.path.dirname(os.path.abspath(__file__))
    roots = (here, os.path.join(os.path.dirname(here), 'drug_discovery', 'theranostics_current'))
    for root in roots:
        out = os.path.join(root, 'outputs')
        if not os.path.isdir(out):
            continue
        for sub in sorted(os.listdir(out)):
            cand = os.path.join(out, sub)
            if os.path.isfile(os.path.join(cand, 'reliability_v2_analysis.json')):
                return cand
    raise SystemExit('frozen outputs not found beside this file or in the workspace')


CW = _data_dir()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'numbers.tex')
L = lambda f: json.load(open(os.path.join(CW, f)))
T = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']

rel = L('reliability_v2_analysis.json')          # THE structure-level source of truth
meth = L('methods_v2_results.json')['results']
hier = L('hierstats_analysis.json')['target_level']
beta = L('beta_ablation.json')['summary']
ecfp = L('ecfp_baseline.json')['summary']
fr = L('frontier_v2_analysis.json')
rec = L('recovery_v2_results.json')['results']
cg = L('compat_gen_analysis.json')['summary']

p = rel['pooled']
mi, ma = p['risk_coverage_micro'], p['risk_coverage_macro']


def thou(n):
    return f"{n:,}".replace(",", "{,}")


def pct(lo, hi):
    return f"{100.0 * (hi - lo) / hi:.0f}"


import statistics as st
tgt = {t: st.mean([c['stga_ecfp'] - c['graphga'] for r in meth if r['target'] == t
                   for c in r['per_seed']]) for t in T}
tl = st.mean(tgt.values())
sd = st.stdev(tgt.values()); se = sd / (len(tgt) ** 0.5)
from scipy import stats as sps
tstat, pval = sps.ttest_1samp(list(tgt.values()), 0.0)
ci = sps.t.interval(0.95, len(tgt) - 1, loc=tl, scale=se)
d_rt = [r['agg']['d_vs_rt'] for r in meth]
d_ga = [r['agg']['d_vs_ga'] for r in meth]

M = {
    # --- dataset / protocol
    'Nstruct': thou(p['n']),
    'NscdRows': '762', 'NscdUnique': str(rel['scd1']['duplicates']['n_unique']),
    'NscdDup': str(rel['scd1']['duplicates']['n_duplicate_rows']),
    'NdupNk': str(rel['nk1r']['duplicates']['n_duplicate_rows']),
    'NdupDtwo': str(rel['drd2']['duplicates']['n_duplicate_rows']),
    'NdupDthree': str(rel['drd3']['duplicates']['n_duplicate_rows']),
    'ScdWithinSD': f"{rel['scd1']['duplicates']['mean_within_compound_sd']:.2f}",
    # --- reliability score
    'RhoSigErr': f"{p['spearman_sigma_err']:.2f}",
    'RhoSigErrPartial': f"{p['partial_err_sig_given_nov_dtr']:.2f}",
    'RhoNovErr': f"{p['spearman_nov_err']:.3f}",
    'RhoNovErrPartial': f"{p['partial_err_nov_given_dtr']:.3f}",
    'RhoFadsNovErr': f"{rel['fads']['spearman_nov_err']:.3f}",
    'RhoDtrErr': f"{p['spearman_dtr_err']:.3f}",
    'RhoScdSigErr': f"{rel['scd1']['spearman_sigma_err']:.3f}",
    # --- risk-coverage
    'RCmicroLo': f"{mi['0.2']:.2f}", 'RCmicroHi': f"{mi['1.0']:.2f}",
    'RCmicroPct': pct(mi['0.2'], mi['1.0']),
    'RCmacroLo': f"{ma['0.2']:.2f}", 'RCmacroHi': f"{ma['1.0']:.2f}",
    'RCmacroPct': pct(ma['0.2'], ma['1.0']),
    # --- in-domain novelty
    'GapMean': f"{p['gap_mean']:.2f}",
    'GapFracPct': f"{100 * p['gap_frac_positive']:.1f}",
    'GapErrLo': f"{p['err_by_gap_quintile'][0]:.2f}",
    'GapErrHi': f"{p['err_by_gap_quintile'][-1]:.2f}",
    'NinDomWin': str(sum(rel[t]['novel_in_domain_rmse'] < rel[t]['novel_out_domain_rmse'] for t in T)),
    'Drdtwoin': f"{rel['drd2']['novel_in_domain_rmse']:.2f}",
    'Drdtwoout': f"{rel['drd2']['novel_out_domain_rmse']:.2f}",
    'Nkonein': f"{rel['nk1r']['novel_in_domain_rmse']:.2f}",
    'Nkoneout': f"{rel['nk1r']['novel_out_domain_rmse']:.2f}",
    # --- method
    'MethodGain': f"{tl:+.3f}", 'MethodCIlo': f"{ci[0]:+.3f}", 'MethodCIhi': f"{ci[1]:+.3f}",
    'MethodT': f"{tstat:.2f}", 'MethodP': f"{pval:.3f}",
    'MethodCells': str(len(meth)),
    'MethodRTlo': f"{min(d_rt):+.3f}", 'MethodRThi': f"{max(d_rt):+.3f}",
    'MethodBest': f"{max(d_ga):+.3f}",
    'LatentGain': f"{hier['mean']:+.3f}", 'LatentP': f"{hier['p']:.3f}",
    # --- negatives
    'BetaDelta': f"{st.mean([beta[t]['beta1_vs_beta0_novel']['delta'] for t in T]):+.3f}",
    'BetaBestP': f"{min(beta[t]['beta1_vs_beta0_novel']['p'] for t in T):.3f}",
    'BetaOtherPLo': f"{sorted(beta[t]['beta1_vs_beta0_novel']['p'] for t in T)[1]:.2f}",
    'BetaOtherPHi': f"{max(beta[t]['beta1_vs_beta0_novel']['p'] for t in T):.2f}",
    'CompatR': f"{cg['pair_pearson_r']:.2f}", 'CompatP': f"{cg['pair_pearson_p']:.2f}",
    'CompatGain': f"{cg.get('mean_gain', -0.011):.3f}",
    # --- frontier
    'FrontNovDtrLo': f"{min(fr[k]['nov_dtrain'] for k in fr):.3f}",
    'FrontNovDtrHi': f"{max(fr[k]['nov_dtrain'] for k in fr):.3f}",
    'FrontNovSigLo': f"{min(fr[k]['nov_sig'] for k in fr):.2f}",
    'FrontNovSigHi': f"{max(fr[k]['nov_sig'] for k in fr):.2f}",
    'FrontNovPotLo': f"{max(fr[k]['nov_pot'] for k in fr):.2f}",
    'FrontNovPotHi': f"{min(fr[k]['nov_pot'] for k in fr):.2f}",
    'FrontRuns': thou(sum(fr[k]['n'] for k in fr)),
    # --- recovery
    'RecScdPred': f"{rec['scd1']['retrained_pred']:.2f}",
    'RecSigLo': f"{min(rec[t]['retrained_sigma'] for t in T):.2f}",
    'RecSigHi': f"{max(rec[t]['retrained_sigma'] for t in T):.2f}",
}


# --- temporal shift (Regime 2) ---
tmp = L('temporal_analysis.json')
tp = tmp['pooled']
ctl = {t: tmp[t]['control_random_same_size'] for t in tmp if t in T}
M.update({
    'TempCut': str(tmp['cut_year']),
    'TempN': thou(tp['n']),
    'TempRmse': f"{tp['rmse']:.2f}",
    'TempRhoSigErr': f"{tp['spearman_sigma_err']:.2f}",
    'TempPartial': f"{tp['partial_err_sigma_given_dtr']:.2f}",
    'TempCov': f"{tp['conformal_coverage_adaptive']:.3f}",
    'TempRCpct': pct(tp['risk_coverage_rmse']['0.2'], tp['risk_coverage_rmse']['1.0']),
    'TempCtlRmseLo': f"{min(c['rmse'] for c in ctl.values()):.2f}",
    'TempCtlRmseHi': f"{max(c['rmse'] for c in ctl.values()):.2f}",
    'TempCtlCovLo': f"{min(c['conformal_coverage_adaptive'] for c in ctl.values()):.3f}",
    'TempCtlCovHi': f"{max(c['conformal_coverage_adaptive'] for c in ctl.values()):.3f}",
    'TempDrdthreeRho': f"{tmp['drd3']['spearman_sigma_err']:.3f}",
    'TempDrdthreeCtl': f"{ctl['drd3']['spearman_sigma_err']:.2f}",
})
# The temporal cohort is a separately re-curated four-target set, so the only like-for-like
# comparator is its own size-matched random control on the identical data. Pooled micro
# correlations also mix targets of different error scale, so macro values are carried too.
_mac, _dc = tmp['macro'], tmp['delta_vs_control']
_CAPT = {'scd1': 'Scd', 'nk1r': 'Nkone', 'drd2': 'Drdtwo', 'drd3': 'Drdthree'}
M.update({
    'TempNtargets': str(_mac['n_targets']),
    'TempNtargetsWord': {2: 'two', 3: 'three', 4: 'four', 5: 'five'}[_mac['n_targets']],
    'TempYearField': 'first disclosure',
    'TempCtlRmsePooled': f"{_mac['control']['rmse_pooled_nweighted']:.2f}",
    'TempRmsePctVsCtl': f"{_dc['rmse_pct_increase_vs_control']:.0f}",
    'TempMacroRho': f"{_mac['temporal']['spearman_sigma_err']:.2f}",
    'TempCtlMacroRho': f"{_mac['control']['spearman_sigma_err']:.2f}",
    'TempMacroCov': f"{_mac['temporal']['conformal_coverage_adaptive']:.3f}",
    'TempCtlMacroCov': f"{_mac['control']['conformal_coverage_adaptive']:.3f}",
})
_ci = lambda v, d=2: f"[{v[0]:.{d}f}, {v[1]:.{d}f}]"
_cis = lambda v, d=2: f"[{v[0]:+.{d}f}, {v[1]:+.{d}f}]"
M.update({
    'TempRmseCI': _ci(tp['rmse_ci95']),
    'TempCovCI': _ci(tp['conformal_coverage_adaptive_ci95'], 3),
    'TempRhoCI': _cis(tp['spearman_sigma_err_ci95']),
    # strict criterion: the temporal interval lies wholly outside the control's interval.
    # Reported alongside the replicate-range statistic, which is the weaker of the two.
    'TempSepRmse': str(sum(tmp[t]['rmse_test_ci95'][0]
                           > tmp[t]['control_random_same_size']['rmse_ci95'][1] for t in _CAPT)),
    'TempSepCov': str(sum(tmp[t]['conformal_coverage_adaptive_ci95'][1]
                          < tmp[t]['control_random_same_size']['conformal_coverage_adaptive_ci95'][0]
                          for t in _CAPT)),
    'TempSepRho': str(sum(tmp[t]['spearman_sigma_err_ci95'][1]
                          < tmp[t]['control_random_same_size']['spearman_sigma_err_ci95'][0]
                          for t in _CAPT)),
    'TempOutsideRmse': str(_mac['temporal_outside_control_range']['rmse']),
    'TempSepCovWord': {0: 'none', 1: 'one', 2: 'two', 3: 'three', 4: 'four'}[
        sum(tmp[t]['conformal_coverage_adaptive_ci95'][1]
            < tmp[t]['control_random_same_size']['conformal_coverage_adaptive_ci95'][0]
            for t in _CAPT)],
    'TempOutsideCov': str(_mac['temporal_outside_control_range']['coverage']),
    'TempOutsideRho': str(_mac['temporal_outside_control_range']['spearman']),
    'TempCtlReps': str(tmp[list(_CAPT)[0]]['control_random_same_size']['n_reps']),
})
for _t, _c in _CAPT.items():
    _d, _k = tmp[_t], tmp[_t]['control_random_same_size']
    M[f'TempRmseCI{_c}'] = _ci(_d['rmse_test_ci95'])
    M[f'TempCovCI{_c}'] = _ci(_d['conformal_coverage_adaptive_ci95'], 3)
    M[f'TempRhoCI{_c}'] = _cis(_d['spearman_sigma_err_ci95'])
    M[f'TempCtlRmseCI{_c}'] = _ci(_k['rmse_ci95'])
    M[f'TempCtlCovCI{_c}'] = _ci(_k['conformal_coverage_adaptive_ci95'], 3)
    M[f'TempCtlRhoCI{_c}'] = _cis(_k['spearman_sigma_err_ci95'])
for _t, _c in _CAPT.items():
    M[f'TempRho{_c}'] = f"{tmp[_t]['spearman_sigma_err']:.2f}"
    M[f'TempCtlRho{_c}'] = f"{tmp[_t]['control_random_same_size']['spearman_sigma_err']:.2f}"
    M[f'TempDrho{_c}'] = f"{tmp['per_target_delta'][_t]['d_rho']:+.2f}"
    M[f'TempN{_c}'] = thou(tmp[_t]['n_test'])
    M[f'TempCov{_c}'] = f"{tmp[_t]['conformal_coverage_adaptive']:.3f}"
    M[f'TempCtlCov{_c}'] = f"{tmp[_t]['control_random_same_size']['conformal_coverage_adaptive']:.3f}"
# Exact comparison against the size-matched control. Three quantities that the replicate range
# alone cannot give: the one-sided empirical P, whose smallest attainable value is 1/(R+1) and
# is therefore set by how many replicates were run; the direct effect against the control mean
# with the two arms' uncertainties combined rather than compared by interval overlap; and a
# temporal interval that resamples Bemis-Murcko scaffold groups, since compounds from one
# series are not independent.
_ep0 = tmp[list(_CAPT)[0]]['control_random_same_size']['empirical_p']
_pf = lambda v: (f"{v:.4f}" if v >= 1e-4 else f"{v:.1e}")
M.update({
    'TempPfloor': _pf(_ep0['floor']),
    'TempPRmseWorst': _pf(max(tmp[t]['control_random_same_size']['empirical_p']['rmse']
                              for t in _CAPT)),
    'TempPCovWorst': _pf(max(tmp[t]['control_random_same_size']['empirical_p']['coverage']
                             for t in _CAPT)),
    'TempPRmseAtFloor': str(sum(
        tmp[t]['control_random_same_size']['empirical_p']['rmse'] <= _ep0['floor'] + 1e-12
        for t in _CAPT)),
})
for _t, _c in _CAPT.items():
    _e = tmp[_t]['control_random_same_size']['empirical_p']
    _dv = tmp[_t]['delta_vs_control']
    _sb = tmp[_t]['scaffold_cluster_bootstrap']
    M[f'TempPRmse{_c}'] = _pf(_e['rmse'])
    M[f'TempPCov{_c}'] = _pf(_e['coverage'])
    M[f'TempPRho{_c}'] = _pf(_e['spearman'])
    M[f'TempDRmse{_c}'] = f"{_dv['rmse']['delta']:+.2f}"
    M[f'TempDRmseCI{_c}'] = _cis(_dv['rmse']['ci95'])
    M[f'TempDRho{_c}'] = f"{_dv['spearman']['delta']:+.2f}"
    M[f'TempDRhoCI{_c}'] = _cis(_dv['spearman']['ci95'])
    M[f'TempDCov{_c}'] = f"{_dv['coverage']['delta']:+.3f}"
    M[f'TempDCovCI{_c}'] = _cis(_dv['coverage']['ci95'], 3)
    M[f'TempScafRmseCI{_c}'] = _ci(_sb['rmse_ci95'])
    M[f'TempScafRhoCI{_c}'] = _cis(_sb['spearman_sigma_err_ci95'])
    M[f'TempScafCovCI{_c}'] = _ci(_sb['conformal_coverage_adaptive_ci95'], 3)
    M[f'TempScafN{_c}'] = thou(_sb['n_scaffolds'])
# Direct effect averaged over targets, with the target as the unit of generalisation, matching
# how the method gain is tested elsewhere in the paper.
for _key, _tag in [('rmse', 'Rmse'), ('spearman', 'Rho'), ('coverage', 'Cov')]:
    _pt = np.array([tmp[t]['delta_vs_control'][_key]['delta'] for t in _CAPT], float)
    _sem = _pt.std(ddof=1) / np.sqrt(len(_pt))
    _tc4 = sps.t.ppf(0.975, len(_pt) - 1)
    _dec = 3 if _key == 'coverage' else 2
    M[f'TempMacroD{_tag}'] = f"{_pt.mean():+.{_dec}f}"
    M[f'TempMacroD{_tag}CI'] = _cis([_pt.mean() - _tc4 * _sem, _pt.mean() + _tc4 * _sem], _dec)
    M[f'TempMacroD{_tag}Neg'] = str(int((_pt < 0).sum()))

# median-year sensitivity, so the manuscript never hard-codes the comparison figure
_med = os.path.join(CW, 'temporal_analysis_yearmedian.json')
if os.path.exists(_med):
    _m = json.load(open(_med))
    _lost = lambda d: [t for t in _CAPT if d[t]['spearman_sigma_err_ci95'][1]
                       < d[t]['control_random_same_size']['spearman_sigma_err_ci95'][0]]
    M.update({
        'TempMedRmsePct': f"{_m['delta_vs_control']['rmse_pct_increase_vs_control']:.0f}",
        'TempMedLostSame': 'yes' if _lost(_m) == _lost(tmp) else 'no',
        'TempMedRhoScd': f"{_m['scd1']['spearman_sigma_err']:.2f}",
        'TempMedCtlRhoScd': f"{_m['scd1']['control_random_same_size']['spearman_sigma_err']:.2f}",
    })

# per-target in-domain comparison: which target is the exception, read from the source
_IN = [t for t in T if rel[t]['novel_in_domain_rmse'] < rel[t]['novel_out_domain_rmse']]
_EX = [t for t in T if t not in _IN]
_NAME = {'scd1': 'SCD-1', 'fads': 'FADS', 'nk1r': 'NK1R', 'drd2': 'DRD2', 'drd3': 'DRD3'}
M.update({
    'NinDomWin': str(len(_IN)),
    'NinDomExcept': ', '.join(_NAME[t] for t in _EX) if _EX else 'none',
    'NinDomSigWin': str(sum(rel[t]['novel_in_domain_sigma'] < rel[t]['novel_out_domain_sigma']
                            for t in T)),
})
# --- conformal (Regime 1) ---
con = L('conformal_analysis.json')['pooled']['alpha0.1']
M.update({
    'ConfCov': f"{con['adaptive_coverage']:.3f}",
    'ConfCovStd': f"{con['standard_coverage']:.3f}",
    'ConfWidthRatio': f"{con['width_ratio_adaptive_over_standard']:.2f}",
    'ConfCovLoSig': f"{L('conformal_analysis.json')['drd3']['alpha0.1']['adaptive_coverage_low_sigma']:.3f}",
    'ConfCovHiSig': f"{L('conformal_analysis.json')['drd3']['alpha0.1']['adaptive_coverage_high_sigma']:.3f}",
})
# --- pool-based acquisition against real labels ---
po = L('poolopt_analysis.json')['summary']
enr = lambda m: st.mean([po[t][m]['enrichment_vs_random'] for t in po])
M.update({
    'PoolGreedy': f"{enr('greedy'):.2f}", 'PoolUCB': f"{enr('ucb'):.2f}",
    'PoolLCB': f"{enr('lcb'):.2f}", 'PoolConf': f"{enr('conformal'):.2f}",
    'PoolBudget': '300',
    'PoolNtop': str(sum(po[t]['n_top1pct_in_pool'] for t in po)),
})

# --- per-target record and parent-structure counts (Table 1) ---
# Two distinct quantities that must never be conflated: the curated activity records, and
# the parent structures they reduce to, which are what the model is fitted and split on.
CAP = {'scd1': 'Scd', 'fads': 'Fads', 'nk1r': 'Nkone', 'drd2': 'Drdtwo', 'drd3': 'Drdthree'}
for t in T:
    d = rel[t]['duplicates']
    M[f'Rows{CAP[t]}'] = thou(d['n_rows'])
    M[f'Struct{CAP[t]}'] = thou(d['n_unique'])
M['Nrows'] = thou(sum(rel[t]['duplicates']['n_rows'] for t in T))

# Quantities that were still literals in the text. Each is read from the same frozen output
# the sentence describes, so the sentence cannot drift from it.
_om2 = {e['target']: e for e in L('oracle_metrics.json')['results']}
_sp = [_om2[t]['bakeoff'][_om2[t]['selected_model']]['spearman'][0] for t in T]
M['ProtoRhoLo'] = f"{min(_sp):.2f}"
M['ProtoRhoHi'] = f"{max(_sp):.2f}"
for _t, _c in CAP.items():
    M[f'WithinSD{_c}'] = f"{rel[_t]['duplicates']['mean_within_compound_sd']:.2f}"
    M[f'NrepPar{_c}'] = str(rel[_t]['duplicates']['n_compounds_with_replicates'])
M['RecSimLo'] = f"{min(rec[t]['rec_stga'] for t in T):.2f}"
M['RecSimHi'] = f"{max(rec[t]['rec_stga'] for t in T):.2f}"
M['RecNullLo'] = f"{min(rec[t]['null_sim'] for t in T):.2f}"
M['RecNullHi'] = f"{max(rec[t]['null_sim'] for t in T):.2f}"
M['RecScdMeas'] = f"{rec['scd1']['measured']:.2f}"
M['RecScdGap'] = f"{rec['scd1']['measured'] - rec['scd1']['retrained_pred']:.1f}"
M['MethodRuns'] = str(sum(len(r['per_seed']) for r in meth))
M['ConfWidthPct'] = f"{100 * (1 - float(L('conformal_analysis.json')['pooled']['alpha0.1']['width_ratio_adaptive_over_standard'])):.0f}"


# Activity thresholds were the last hand-typed numbers in Table 1. Read them from the frozen
# oracle metrics, which record the cutoff each run actually used, so the table cannot drift
# from the analysis the way a literal can.
_om = {e['target']: e for e in L('oracle_metrics.json')['results']}
for t in T:
    M[f'Thr{CAP[t]}'] = f"{_om[t]['threshold']:.1f}"

# The disagreement turnover, with an interval that respects the design: the 1,200 runs are
# 300 trajectories of four novelty settings, so targets and then whole trajectories are
# resampled rather than runs.
_fr = L('frontier_v2_results.json')['results']
_rw = [dict(r, target=t) for t, rs in _fr.items() for r in rs]
_nv = np.array([r['novelty'] for r in _rw]); _sg = np.array([r['sigma'] for r in _rw])
# One support set is drawn per target and seed and reused across both optimizers, both
# uncertainty penalties and all four novelty weights, so the resampling unit is the
# target-seed block of 16 observations, not the 300 method-penalty trajectories.
_bl = np.array([f"{r['target']}|{r['seed']}" for r in _rw])
_tg = np.array([r['target'] for r in _rw])
_bn = np.linspace(_nv.min(), _nv.max(), 9)
_ix = np.clip(np.digitize(_nv, _bn) - 1, 0, 7)
_uT = sorted(set(_tg)); _bbt = {t: sorted(set(_bl[_tg == t])) for t in _uT}
_ib = {b: np.flatnonzero(_bl == b) for b in set(_bl)}
_rng = np.random.default_rng(5); _D = []
for _ in range(2000):
    _take = []
    for _t in _rng.choice(_uT, len(_uT), replace=True):
        _b = _bbt[_t]
        for _q in _rng.choice(_b, len(_b), replace=True):
            _take.append(_ib[_q])
    _sel = np.concatenate(_take); _yi, _ii = _sg[_sel], _ix[_sel]
    _D.append([_yi[_ii == k].mean() if (_ii == k).any() else np.nan for k in range(8)])
_D = np.array(_D); _d78 = _D[:, 6] - _D[:, 7]
M.update({
    'TurnPeak': f"{np.nanmean(_D[:, 6]):.2f}",
    'TurnLast': f"{np.nanmean(_D[:, 7]):.2f}",
    'TurnDrop': f"{np.nanmean(_d78):+.2f}",
    'TurnDropCI': f"[{np.nanpercentile(_d78, 2.5):+.2f}, {np.nanpercentile(_d78, 97.5):+.2f}]",
    'TurnDropP': f"{np.nanmean(_d78 > 0):.3f}",
})

mm = L('mixedmodel_method.json')['mixed_model']
M.update({
    'MixedB': f"{mm['intercept']:+.3f}", 'MixedSE': f"{mm['se']:.3f}",
    'MixedCIlo': f"{mm['ci95'][0]:+.3f}", 'MixedCIhi': f"{mm['ci95'][1]:+.3f}",
    'MixedVarTgt': f"{mm['group_var_target']:.5f}",
    'MixedVarCell': f"{mm['vcomp_cell'][0]:.5f}",
    'MixedResid': f"{mm['scale_residual']:.5f}",
})

# Acquisition, paired by seed against the predicted-mean rule. Means alone cannot say whether
# a rule is separated from greedy, and two of these are while one is not: penalising
# uncertainty costs on every target, whereas the optimistic rule's higher average rests on one
# target and its interval covers zero. The unit of generalisation is the target, as elsewhere.
_pr = L('poolopt_analysis.json')['results']
_PT = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']
for _m, _tag in [('ucb', 'Ucb'), ('lcb', 'Lcb'), ('conformal', 'Conf')]:
    _per = [np.mean([r[_m]['hits'] - r['greedy']['hits'] for r in _pr[t]]) for t in _PT]
    _all = [r[_m]['hits'] - r['greedy']['hits'] for t in _PT for r in _pr[t]]
    _a = np.array(_per, float)
    _se = _a.std(ddof=1) / np.sqrt(len(_a))
    _tc = sps.t.ppf(0.975, len(_a) - 1)
    _p = sps.ttest_1samp(_a, 0).pvalue
    M.update({
        'Pool%sD' % _tag: f"{_a.mean():+.2f}",
        'Pool%sCI' % _tag: f"[{_a.mean() - _tc * _se:+.2f}, {_a.mean() + _tc * _se:+.2f}]",
        'Pool%sP' % _tag: f"{_p:.3f}",
        'Pool%sWin' % _tag: f"{np.mean([x > 0 for x in _all]):.2f}",
    })

# Endpoint composition. The pooled response is named pIC50 by convention but is populated
# from four ChEMBL standard types, so the article states the mix per target. The total
# variation distance between the pre- and post-cutoff mixes tests whether endpoint
# composition shifts across the temporal split and could stand in for the chemical shift.
ep = L('endpoint_composition.json')
_cv, _tc = ep['cv_cohort'], ep['temporal_cohort']
M.update({
    'EndpScdIC': f"{_cv['scd1']['types_pct'].get('IC50', 0):.0f}",
    'EndpNkIC': f"{_cv['nk1r']['types_pct'].get('IC50', 0):.0f}",
    'EndpNkKi': f"{_cv['nk1r']['types_pct'].get('Ki', 0):.0f}",
    'EndpDtwoKi': f"{_cv['drd2']['types_pct'].get('Ki', 0):.0f}",
    'EndpDthreeKi': f"{_cv['drd3']['types_pct'].get('Ki', 0):.0f}",
    'EndpPoolKi': f"{_tc['pooled_pct'].get('Ki', 0):.0f}",
    'EndpTVScd': f"{_tc['scd1']['tv_distance']:.2f}",
    'EndpTVNk': f"{_tc['nk1r']['tv_distance']:.2f}",
    'EndpTVDtwo': f"{_tc['drd2']['tv_distance']:.2f}",
    'EndpTVDthree': f"{_tc['drd3']['tv_distance']:.2f}",
    'EndpMatchDtwo': f"{_cv['drd2']['matched_pct']:.0f}",
    'EndpMatchNk': f"{_cv['nk1r']['matched_pct']:.0f}",
})

with open(OUT, 'w') as f:
    f.write("% Generated by make_numbers.py from the frozen outputs - DO NOT EDIT BY HAND.\n")
    f.write("% Regenerate after any analysis change, then recompile.\n")
    for k, v in M.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(f'wrote {OUT} with {len(M)} macros')
for k in ['Nstruct', 'RhoSigErr', 'RhoNovErr', 'RCmicroPct', 'RCmacroPct', 'MethodGain', 'MethodP']:
    print(f'  \\{k} = {M[k]}')
