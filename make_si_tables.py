"""Generate si_tables.tex from the frozen analysis outputs.

Same principle as make_numbers.py: no Supplementary table is hand-typed. Every value is
read from the frozen outputs (or data/chembl_v2/curation_provenance.json), so a table
cannot drift out of step with the analysis it reports. Regenerate after any analysis
change, then recompile the SI.

    python make_si_tables.py    ->  writes si_tables.tex
"""
from __future__ import annotations
import json, os, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
def _prov_path():
    """Curation provenance, whether this file sits in the repository or the workspace.

    The repository ships data/chembl_v2/curation_provenance.json beside this script. Resolving
    only against the workspace layout made the curation table vanish from a clean clone, which
    also renumbered every table after it, so a reader reproducing the Supplementary Information
    got different numbers from the ones the article cites.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, 'data', 'chembl_v2', 'curation_provenance.json'),
                 os.path.join(ROOT, 'drug_discovery', 'theranostics_current', 'data',
                              'chembl_v2', 'curation_provenance.json')):
        if os.path.isfile(cand):
            return cand
    return cand


PROV = _prov_path()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'si_tables.tex')
L = lambda f: json.load(open(os.path.join(CW, f)))
T = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']
LAB = {'scd1': 'SCD-1', 'fads': 'FADS', 'nk1r': 'NK1R', 'drd2': 'DRD2', 'drd3': 'DRD3'}
COV = ['0.2', '0.4', '0.6', '0.8', '1.0']

rel = L('reliability_v2_analysis.json')
con = L('conformal_analysis.json')
tmp = L('temporal_analysis.json')
pool = L('poolopt_analysis.json')['summary']
meth = L('methods_v2_results.json')['results']
hier = L('hierstats_analysis.json')
beta = L('beta_ablation.json')['summary']
ecfp = L('ecfp_baseline.json')['summary']
fr = L('frontier_v2_analysis.json')
rec = L('recovery_v2_results.json')['results']
cg = L('compat_gen_analysis.json')['summary']
om = {e['target']: e for e in L('oracle_metrics.json')['results']}
sfs = L('scaffold_fold_stats.json') if os.path.exists(os.path.join(CW, 'scaffold_fold_stats.json')) else {}
prov = json.load(open(PROV)) if os.path.exists(PROV) else {}

OUTL = []


def tab(label, caption, header, rows, spec, note='', size='', tight=False):
    """Emit one Supplementary table.

    `size` shrinks a table that would otherwise run past the text block: pass 'small' or
    'footnotesize'. Column padding is tightened with it, since a wide table is usually wide
    because of the number of columns rather than the font. Any table left overflowing is
    caught by the overfull-hbox check in verify_results.py rather than by eye.
    """
    OUTL.append(r'\begin{table}[htbp]\centering\caption{%s}\label{%s}' % (caption, label))
    if size:
        OUTL.append(r'{\%s\setlength{\tabcolsep}{3pt}' % size)
    elif tight:
        # Narrow the column padding but keep the body at full size, for a table that is wide
        # because of its column count rather than its font.
        OUTL.append(r'{\setlength{\tabcolsep}{3.5pt}')
    OUTL.append(r'\begin{tabular}{@{}%s@{}}\toprule' % spec)
    OUTL.append(header + r'\\\midrule')
    OUTL.extend(r + r'\\' for r in rows[:-1])
    OUTL.append(rows[-1] + r'\\\bottomrule')
    OUTL.append(r'\end{tabular}' + (('\n\\par\\smallskip\\footnotesize ' + note) if note else ''))
    if size or tight:
        OUTL.append('}')
    OUTL.append(r'\end{table}' + '\n')


def _psci(pv):
    """A P value in scientific notation, set in math mode.

    The previous inline version chained two string replacements on '%.1e' output and produced a
    stray brace for any P with a positive exponent. Every P here is far below 1, so it never
    fired, but formatting should not depend on that.
    """
    m, e = ('%.1e' % pv).split('e')
    return '$%s\\times10^{%d}$' % (m, int(e))


def _ci_fmt(ci):
    """A signed interval in math mode, so negatives get a real minus rather than a hyphen."""
    return '$[%+.2f, %+.2f]$' % (ci[0], ci[1])


def _andlist(xs):
    """Join names as English prose: 'a, b and c'."""
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' and ' + xs[-1]


def f(x, d=2):
    """Fixed-point, with negatives set in math so they get a real minus rather than a hyphen."""
    t = ('%.' + str(d) + 'f') % x
    return ('$%s$' % t) if t.startswith('-') else t


def sg(x, d=3):
    return ('$%+.' + str(d) + 'f$') % x


# ---------------------------------------------------------------- S1 datasets
rows = []
for t in T:
    d = rel[t]['duplicates']
    # With no multi-record parent the within-parent SD is undefined, not zero. FADS printed
    # 0.00, which reads as a measured spread of zero rather than as nothing to measure.
    _sd = f(d['mean_within_compound_sd']) if d['n_compounds_with_replicates'] else 'n/a'
    rows.append(f"{LAB[t]} & {d['n_rows']:,} & {d['n_unique']:,} & {d['n_duplicate_rows']} & "
                f"{d['n_compounds_with_replicates']} & {_sd}".replace(',', '{,}'))
tab('tab:s-data',
    'Dataset composition after parent standardization. Records sharing a standardized parent '
    'InChIKey are aggregated to a median activity before splitting. Several source records '
    'for one parent may be alternative molecular representations, repeated measurements, or '
    'measurements made in different assay contexts, and the retained fields do not separate '
    'these, so they are reported as multi-record parents rather than uniformly called '
    'replicates. Within-parent SD is computed among parents holding more than one record, is undefined and shown as n/a where a target has none, and '
    'describes label heterogeneity; because models are scored against the median aggregate '
    'rather than against a fresh measurement, it is not a lower bound on attainable error.',
    r'Target & Records & \shortstack[r]{Parent\\structures} & \shortstack[r]{Records\\collapsed} & \shortstack[r]{Multi-record\\parents} & \shortstack[r]{Within-parent\\SD}',
    rows, 'lrrrrr')

# ---------------------------------------------------------------- S2 protocols
rows = []
for t in T:
    e = om[t]; bo = e['bakeoff'][e['selected_model']]; sc = e.get('scaffold', {})
    r2 = sc.get('r2')
    r2s = f(r2) if isinstance(r2, (int, float)) else 'n/a'
    # R^2 divides by the held-out fold's own variance, so a nearly constant fold sends it far
    # negative at ordinary absolute error. Carry the RMSE beside it or FADS at -222.96 reads
    # as a software fault rather than as a low-variance denominator.
    _sfs = sfs.get(t, {})
    rmse_s = f(_sfs['implied_rmse']) if _sfs.get('implied_rmse') is not None else 'n/a'
    tt = tmp.get(t)
    tcell = f(tt['rmse_test']) if tt else 'n/a'
    rows.append(f"{LAB[t]} & {f(bo['spearman'][0],3)} & {f(bo['r2'][0],3)} & {f(bo['auc'][0],3)} & "
                f"{f(sc.get('spearman', float('nan')),3)} & {r2s} & {rmse_s} & {tcell}")
tab('tab:s-protocol',
    'Activity-model performance under three evaluation protocols. Random-CV is five-fold '
    'cross-validation on parent structures. Scaffold is a Bemis--Murcko split used as a '
    'structural-shift stress test; where the held-out fold contains no active compounds the '
    'classification AUC is undefined while the continuous metrics remain defined, so no '
    'scaffold AUC column is shown. $R^2$ here is the coefficient of determination, '
    '$1-\\sum(y-\\hat{y})^2/\\sum(y-\\bar{y})^2$, which equals a squared correlation only for an '
    'in-sample fit and is unbounded below out of sample; a negative value means the model '
    'predicts that fold less well than its own mean does. It is normalised by the variance of the '
    'held-out fold itself, so a fold whose response barely varies drives it far negative even '
    'at ordinary absolute error: the FADS fold has an activity spread of 0.07 against 1.54 '
    'for the whole dataset, a 531-fold difference in variance, and its RMSE of 1.00 sits '
    'inside the 0.97 to 1.54 range spanned by the five targets. The RMSE column is given for '
    'that reason. Temporal '
    'RMSE is on compounds published from 2015 onwards after training on earlier ones, and is '
    'available only for the four targets re-curated from ChEMBL.',
    r'Target & \multicolumn{3}{c}{Random-CV} & \multicolumn{3}{c}{Scaffold} & Temporal\\'
    r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}'
    r' & Spearman & $R^2$ & AUC & Spearman & $R^2$ & RMSE & RMSE',
    rows, 'lrrrrrrr', tight=True)

# ---------------------------------------------------------------- S3 error stratification
rows = []
for t in T:
    q = rel[t].get('rmse_by_sigma_quintile', [float('nan')] * 5)
    rows.append(f"{LAB[t]} & {f(rel[t]['spearman_sigma_err'],3)} & " + ' & '.join(f(v) for v in q))
p = rel['pooled']
rows.append(r'\midrule Pooled & ' + f(p['spearman_sigma_err'], 3) + ' & ' +
            ' & '.join(f(v) for v in p.get('rmse_by_sigma_quintile', [float('nan')] * 5)))
# Which rows actually increase across the quintiles, worked out from the data rather than
# asserted, so the caption cannot drift if a target moves.
_mono = [LAB[t] for t in T
         if all(rel[t]['rmse_by_sigma_quintile'][i] <= rel[t]['rmse_by_sigma_quintile'][i + 1]
                for i in range(4))]
_nonmono = [LAB[t] for t in T if LAB[t] not in _mono]
tab('tab:s-strat',
    'Error stratification by ensemble disagreement, in distribution. Spearman correlation '
    'between the disagreement score and the absolute out-of-fold error, and RMSE within '
    'quintiles of the disagreement score (Q1 lowest). This is a ranking result, not a '
    'calibration; calibrated coverage is reported separately. Error increases across every '
    'quintile on %s and on the pooled row. %s is the exception: its quintiles are not ordered, '
    'and at a correlation of %s it carries little ranking information, the same weakness its '
    'risk--coverage curve shows in Supplementary Table~\\ref{tab:s-riskcov}.'
    % (_andlist(_mono), _andlist(_nonmono),
       f(rel['scd1']['spearman_sigma_err'], 3)),
    r'Target & $\rho(\sigma_T,|e|)$ & Q1 & Q2 & Q3 & Q4 & Q5', rows, 'lrrrrrr')

# ---------------------------------------------------------------- S4 risk-coverage
rows = []
for t in T:
    rc = rel[t]['risk_coverage_rmse']
    rows.append(f"{LAB[t]} & " + ' & '.join(f(rc[c]) for c in COV))
mi, ma = p['risk_coverage_micro'], p['risk_coverage_macro']
rows.append(r'\midrule Pooled (micro) & ' + ' & '.join(f(mi[c]) for c in COV))
rows.append('Equal weight (macro) & ' + ' & '.join(f(ma[c]) for c in COV))
tab('tab:s-riskcov',
    'Risk--coverage in distribution. RMSE of the retained predictions when the fraction with '
    'the lowest disagreement is kept. The micro row pools all structures and is dominated by '
    'the two largest targets; the macro row averages the five per-target curves with equal '
    'weight. Error rises monotonically with coverage on %s; on %s the curve is flat, varying by %s across the '
    'whole range, which is the same weak ranking Supplementary Table~\\ref{tab:s-strat} shows.'
    % (_andlist(_mono), _andlist(_nonmono),
       f(max(rel['scd1']['risk_coverage_rmse'][c] for c in COV)
         - min(rel['scd1']['risk_coverage_rmse'][c] for c in COV))),
    r'Target & 20\% & 40\% & 60\% & 80\% & 100\%', rows, 'lrrrrr')

# ---------------------------------------------------------------- S5 novelty/distance/error
rows = []
for t in T:
    r = rel[t]
    rows.append(f"{LAB[t]} & {sg(r['spearman_nov_err'])} & {sg(r['partial_err_nov_given_dtr'])} & "
                f"{sg(r['spearman_dtr_err'])} & {f(r['spearman_sigma_err'],3)} & "
                f"{f(r['partial_err_sig_given_nov_dtr'],3)}")
rows.append(r'\midrule Pooled & ' + sg(p['spearman_nov_err']) + ' & ' + sg(p['partial_err_nov_given_dtr'])
            + ' & ' + sg(p['spearman_dtr_err']) + ' & ' + f(p['spearman_sigma_err'], 3)
            + ' & ' + f(p['partial_err_sig_given_nov_dtr'], 3))
# Which targets actually lose their novelty signal when distance is controlled for. FADS does
# not: its correlation is negative and grows slightly under control, so a blanket claim is
# contradicted by a row the reader can see.
_shrink = [LAB[t] for t in T
           if abs(rel[t]['partial_err_nov_given_dtr']) < abs(rel[t]['spearman_nov_err'])]
_grow = [LAB[t] for t in T if LAB[t] not in _shrink]
tab('tab:s-novelty',
    'Novelty, chemical distance and measured error. $\\nu$ is novelty relative to the ten '
    'starting compounds and $d$ the distance to the nearest training compound. Novelty is a '
    'weak predictor of error, and on %s it loses what little signal it has once distance is '
    'controlled for, as it does pooled. %s is the exception on both counts: its correlation is '
    'negative, so more novel compounds carry lower error there, and controlling for distance '
    'does not shrink it. Disagreement predicts error after controlling for both on every '
    'target.'
    % (_andlist(_shrink), _andlist(_grow)),
    r'Target & $\rho(\nu,e)$ & $\rho(e,\nu\mid d)$ & $\rho(d,e)$ & $\rho(\sigma_T,e)$ & $\rho(e,\sigma_T\mid \nu,d)$',
    rows, 'lrrrrr')

# ---------------------------------------------------------------- S6 conformal
rows = []
for a in ['alpha0.2', 'alpha0.1', 'alpha0.05']:
    c = con['pooled'][a]
    rows.append(f"{f(c['target_coverage'],2)} & {f(c['standard_coverage'],3)} & {f(c['standard_width_median'])} & "
                f"{f(c['adaptive_coverage'],3)} & {f(c['adaptive_width_median'])} & "
                f"{f(c['width_ratio_adaptive_over_standard'])}")
# The width gain is small, and at the 0.80 level it is partly bought with coverage, so the
# caption must not read as evidence the score is informative. Supplementary Note 4 already says
# the evidence rests on the error ranking instead; the caption used to imply the opposite.
_wr = [100 * (1 - con['pooled'][a_]['width_ratio_adaptive_over_standard'])
       for a_ in ('alpha0.2', 'alpha0.1', 'alpha0.05')]
_traded = [f(con['pooled'][a_]['target_coverage'], 2)
           for a_ in ('alpha0.2', 'alpha0.1', 'alpha0.05')
           if con['pooled'][a_]['adaptive_coverage'] < con['pooled'][a_]['standard_coverage']]
tab('tab:s-conformal',
    'Split conformal intervals in distribution. Standard intervals use the absolute residual '
    'and give every compound the same width; adaptive intervals normalize the residual by the '
    'disagreement score. Both reach their nominal coverage at all three levels. The adaptive '
    'intervals are narrower, but only by %s to %s\\%%, and at the nominal %s level they also '
    'cover slightly less, so part of that narrowing is bought with coverage rather than won. '
    'The efficiency gain is small and is not what the case for the score rests on; that case '
    'is the error ranking and the risk--coverage relationship of Supplementary '
    'Tables~\\ref{tab:s-strat} and~\\ref{tab:s-riskcov}.'
    % (f(min(_wr), 0), f(max(_wr), 0), _andlist(_traded)),
    r'Nominal & \multicolumn{2}{c}{Standard} & \multicolumn{2}{c}{Adaptive} & Width ratio\\'
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}'
    r' & Coverage & Median width & Coverage & Median width & (adaptive/standard)',
    rows, 'rrrrrr')

# --------------------------------------------------- S17 TEMPORAL YEAR-FIELD SENSITIVITY
_sens = os.path.join(CW, 'temporal_analysis_yearmedian.json')
if os.path.exists(_sens):
    _md = json.load(open(_sens))
    rows = []
    for t in ['scd1', 'nk1r', 'drd2', 'drd3']:
        if t not in _md or t not in tmp:
            continue
        a_, b_ = tmp[t], _md[t]
        rows.append(f"{LAB[t]} & {a_['n_test']:,} & {f(a_['rmse_test'])} & "
                    f"{sg(a_['spearman_sigma_err'],2)} & {f(a_['conformal_coverage_adaptive'],3)} & "
                    f"{b_['n_test']:,} & {f(b_['rmse_test'])} & "
                    f"{sg(b_['spearman_sigma_err'],2)} & "
                    f"{f(b_['conformal_coverage_adaptive'],3)}".replace(',', '{,}'))
    _da, _db = tmp['delta_vs_control'], _md['delta_vs_control']
    rows.append(r'\midrule Increase over control & \multicolumn{4}{c}{' +
                f"{_da['rmse_pct_increase_vs_control']:.0f}\\%" + r'} & \multicolumn{4}{c}{' +
                f"{_db['rmse_pct_increase_vs_control']:.0f}\\%" + r'}')
    tab('tab:s-tempsens',
        'Sensitivity of the temporal analysis to how a compound is dated. First disclosure, the '
        'earliest publication year among a compound\'s records, is used throughout the main '
        'article: it is the quantity a prospective split requires, and it guarantees that no '
        'evaluation compound carries a pre-cutoff record. The median year of a compound\'s '
        'records is shown for comparison. The error increase over the size-matched control is '
        'close under the two datings, %s\\%% against %s\\%%, and the same two targets lose the '
        'error ranking, so the conclusion does not depend on the choice. The datings differ on '
        'SCD-1, where median dating lifts the temporal correlation above its own control, '
        'because compounds disclosed before the cutoff are dated into the future set. '
        'Correlations are given to two decimals, so the DRD3 value under median dating shows as '
        '$-0.00$: it is negative and smaller in magnitude than $0.005$.'
        % (f(_da['rmse_pct_increase_vs_control'], 0), f(_db['rmse_pct_increase_vs_control'], 0)),
        r'Target & \multicolumn{4}{c}{First disclosure (used)} & \multicolumn{4}{c}{Median year}\\'
        r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}'
        r' & $n_{\mathrm{test}}$ & RMSE & $\rho(\sigma_T,e)$ & Coverage'
        r' & $n_{\mathrm{test}}$ & RMSE & $\rho(\sigma_T,e)$ & Coverage',
        rows, 'rrrrrrrrr')

# ---------------------------------------------------------------- S7 TEMPORAL (key)
rows = []


def _n(v):
    """Thousands separator, applied to one integer rather than to a whole row."""
    return f"{v:,}".replace(',', '{,}')


def _vc(v, ci, dec=2, signed=False):
    """A value with its 95% interval, for one cell."""
    txt = (('$%+.' + str(dec) + 'f$') % v) if signed else (('%.' + str(dec) + 'f') % v)
    if not ci:
        return txt
    if signed:
        lo, hi = ('%+.*f' % (dec, ci[0])), ('%+.*f' % (dec, ci[1]))
    else:
        lo, hi = ('%.*f' % (dec, ci[0])), ('%.*f' % (dec, ci[1]))
    return txt + ' [' + lo + ', ' + hi + ']'


_TT = [t for t in ['scd1', 'nk1r', 'drd2', 'drd3'] if t in tmp]
for t in ['scd1', 'nk1r', 'drd2', 'drd3']:
    if t not in tmp:
        continue
    d = tmp[t]; c = d['control_random_same_size']
    rows.append(f"{LAB[t]} & Temporal & {_n(d['n_train'])} & {_n(d['n_cal'])} & {_n(d['n_test'])} & "
                f"{_vc(d['rmse_test'], d.get('rmse_test_ci95'))} & "
                f"{_vc(d['spearman_sigma_err'], d.get('spearman_sigma_err_ci95'), signed=True)} & "
                f"{_vc(d['conformal_coverage_adaptive'], d.get('conformal_coverage_adaptive_ci95'), 3)}")
    rows.append(" & Control & & & & "
                f"{_vc(c['rmse'], c.get('rmse_ci95'))} & "
                f"{_vc(c['spearman_sigma_err'], c.get('spearman_sigma_err_ci95'), signed=True)} & "
                f"{_vc(c['conformal_coverage_adaptive'], c.get('conformal_coverage_adaptive_ci95'), 3)}")
tp = tmp['pooled']
rows.append(r'\midrule Pooled & Temporal & --- & --- & ' + _n(tp['n']) + ' & ' +
            _vc(tp['rmse'], tp.get('rmse_ci95')) + ' & ' +
            _vc(tp['spearman_sigma_err'], tp.get('spearman_sigma_err_ci95'), signed=True) + ' & ' +
            _vc(tp['conformal_coverage_adaptive'], tp.get('conformal_coverage_adaptive_ci95'), 3))
_ep = {t: tmp[t]['control_random_same_size']['empirical_p'] for t in _TT}
_dv = {t: tmp[t]['delta_vs_control'] for t in _TT}
_R = tmp[_TT[0]]['control_random_same_size']['n_reps']
_pfmt = lambda v: (('%.4f' % v) if v >= 1e-4 else ('%.1e' % v))
_tnote = (r'One-sided empirical $P$ against the %d size-matched control replicates, in the '
          r'direction that counts as degradation for each measure, with smallest attainable '
          r'value $1/(R+1)=%s$: ' % (_R, _pfmt(1.0 / (_R + 1)))
          + '; '.join('%s RMSE %s, coverage %s, ranking %s'
                      % (LAB[t], _pfmt(_ep[t]['rmse']), _pfmt(_ep[t]['coverage']),
                         _pfmt(_ep[t]['spearman'])) for t in _TT)
          + r'. Direct effect against the control mean, combining the temporal bootstrap with '
            r'the standard error of that mean rather than comparing two separate intervals: '
          + '; '.join('%s RMSE %+.2f %s, ranking %+.2f %s'
                      % (LAB[t], _dv[t]['rmse']['delta'],
                         '[%+.2f, %+.2f]' % tuple(_dv[t]['rmse']['ci95']),
                         _dv[t]['spearman']['delta'],
                         '[%+.2f, %+.2f]' % tuple(_dv[t]['spearman']['ci95'])) for t in _TT)
          + r'. Compounds published together are usually analogues, so the temporal interval is '
            r'also given from a bootstrap resampling Bemis-Murcko scaffold groups rather than '
            r'compounds, which is the weaker basis and the one any separation should be read '
            r'against: '
          # Coverage belongs here too: the main text reads the coverage separation off this
          # bootstrap, so leaving it out of the note left that claim without displayed support.
          + '; '.join('%s RMSE [%.2f, %.2f] over %d scaffolds, coverage [%.3f, %.3f], '
                      'ranking [%+.2f, %+.2f]'
                      % (LAB[t], tmp[t]['scaffold_cluster_bootstrap']['rmse_ci95'][0],
                         tmp[t]['scaffold_cluster_bootstrap']['rmse_ci95'][1],
                         tmp[t]['scaffold_cluster_bootstrap']['n_scaffolds'],
                         tmp[t]['scaffold_cluster_bootstrap']['conformal_coverage_adaptive_ci95'][0],
                         tmp[t]['scaffold_cluster_bootstrap']['conformal_coverage_adaptive_ci95'][1],
                         tmp[t]['scaffold_cluster_bootstrap']['spearman_sigma_err_ci95'][0],
                         tmp[t]['scaffold_cluster_bootstrap']['spearman_sigma_err_ci95'][1])
                      for t in _TT)
          + '.')
tab('tab:s-temporal',
    'Temporal shift and its size-matched control, one row per arm. Models are trained on '
    'compounds first published before 2015 and evaluated on those first published later. Each '
    'target has one temporal split; the control repeats %s random splits with proper-training, '
    'calibration and test sets of identical size drawn from the same pool, so it shares the sizes '
    'shown on the temporal row above it; that control is the only like-for-like comparator, '
    'because this cohort is re-curated independently of the five datasets of Table S1. Error '
    'rises on every target, coverage falls below the nominal 0.900 on all %s and its interval '
    'clears the control on %s of them, and the error ranking degrades on DRD2 and DRD3 while '
    'staying level on SCD-1 and NK1R. Brackets give 95\\%% intervals: a percentile bootstrap '
    'over the evaluation compounds for the temporal split, which happens once, and for the '
    'control a $t$ interval on its mean, which is how precisely that mean is estimated. '
    'Figure~\\ArtFigOverview{}c and Supplementary Figure~\\ref{sfig:temporal} instead draw the '
    'spread of the replicates themselves, the wider quantity a single temporal observation is '
    'judged extreme against. The pooled correlation sits below %s of the %s per-target '
    'values and above %s, because '
    'pooling mixes targets whose errors differ in scale.'
    % (_R, len(_TT),
       sum(tmp[t]['conformal_coverage_adaptive_ci95'][1]
           < tmp[t]['control_random_same_size']['conformal_coverage_adaptive_ci95'][0]
           for t in _TT),
       sum(tmp['pooled']['spearman_sigma_err'] < tmp[t]['spearman_sigma_err'] for t in _TT),
       len(_TT),
       next(LAB[t] for t in _TT
            if tmp['pooled']['spearman_sigma_err'] >= tmp[t]['spearman_sigma_err'])),
    r'Target & Split & $n_{\mathrm{train}}$ & $n_{\mathrm{cal}}$ & $n_{\mathrm{test}}$ & RMSE & '
    r'$\rho(\sigma_T,e)$ & Coverage at 0.900',
    rows, 'llrrrrrr', tight=True, note=_tnote)

# ---------------------------------------------------------------- S8 pool acquisition (key)
rows = []
for t in T:
    a = pool[t]
    rows.append(f"{LAB[t]} & {a['pool_size']:,} & {a['n_top1pct_in_pool']} & "
                + ' & '.join(f(a[m]['hits'], 1) for m in ['random', 'greedy', 'ucb', 'lcb', 'conformal'])
                .replace(',', '{,}'))
mean_enr = {m: st.mean([pool[t][m]['enrichment_vs_random'] for t in T])
            for m in ['random', 'greedy', 'ucb', 'lcb', 'conformal']}
rows.append(r'\midrule Enrichment & --- & --- & ' +
            ' & '.join(f(mean_enr[m]) + r'$\times$' for m in ['random', 'greedy', 'ucb', 'lcb', 'conformal']))
# Paired against the predicted-mean rule, seed by seed, with the target as the unit of
# generalisation. A mean cannot say whether a rule is separated from greedy; these do.
from scipy import stats as sps
_pres = json.load(open(os.path.join(CW, 'poolopt_analysis.json')))['results']
_pd = {}
for _m in ['random', 'ucb', 'lcb', 'conformal']:
    _per = [st.mean([r[_m]['hits'] - r['greedy']['hits'] for r in _pres[t]]) for t in T]
    _n = len(_per); _se = st.stdev(_per) / (_n ** 0.5)
    _tc = sps.t.ppf(0.975, _n - 1); _mu = st.mean(_per)
    _pd[_m] = '%+.2f [%+.2f, %+.2f]' % (_mu, _mu - _tc * _se, _mu + _tc * _se)
_note = (r'Paired against the predicted-mean rule, seed by seed, as the mean per-target '
         r'difference in compounds acquired with a 95\% $t$ interval across the five targets: '
         r'$\mu+\sigma$ ' + _pd['ucb'] + r', $\mu-\sigma$ ' + _pd['lcb'] + r', lower score ' +
         _pd['conformal'] + r', random ' + _pd['random'] + r'. Both penalized rules are '
         r'separated from the predicted mean; the optimistic rule is not.')
tab('tab:s-pool',
    'Acquisition against measured activity. All labels in the pool are hidden except ten known '
    'actives; each strategy spends 300 queries and every query reveals a compound\'s '
    'measured activity. Entries are the number of top-percentile compounds acquired, '
    'averaged over twenty seeds, and the last row is the mean enrichment relative to random '
    'selection. Penalizing uncertainty, whether by a lower-confidence rule or a conformal-style lower '
    'score, finds fewer top-percentile compounds than selecting on the predicted mean. The '
    'paired comparisons against the predicted-mean rule are given below the table. The Top '
    '1\\%% column is how many such compounds the pool contains, and on %s the best rule '
    'already takes most of them, %s, so the rules have little room to separate there; on '
    '%s the best rule takes %s of those available.'
    % (_andlist([LAB[t] for t in T
                 if max(pool[t][m]['hits'] for m in ('greedy', 'ucb', 'lcb', 'conformal'))
                 / pool[t]['n_top1pct_in_pool'] > 0.8]),
       _andlist(['%s of %d on %s' % (f(max(pool[t][m]['hits'] for m in
                                           ('greedy', 'ucb', 'lcb', 'conformal')), 1),
                                     pool[t]['n_top1pct_in_pool'], LAB[t])
                 for t in T
                 if max(pool[t][m]['hits'] for m in ('greedy', 'ucb', 'lcb', 'conformal'))
                 / pool[t]['n_top1pct_in_pool'] > 0.8]),
       LAB['drd2'],
       '%s of %d' % (f(max(pool['drd2'][m]['hits'] for m in
                          ('greedy', 'ucb', 'lcb', 'conformal')), 1),
                     pool['drd2']['n_top1pct_in_pool'])),
    r'Target & Pool & Top 1\% & Random & $\mu$ & $\mu+\sigma$ & $\mu-\sigma$ & Lower score', rows, 'lrrrrrrr',
    note=_note)

# ---------------------------------------------------------------- S9 frontier
rows = []
for k in ['graphga_lam0.0', 'graphga_lam0.1', 'stga_lam0.0', 'stga_lam0.1']:
    a = fr[k]
    opt, lam = ('Graph GA' if 'graphga' in k else 'Surrogate-triaged'), k.split('lam')[1]
    rows.append(f"{opt} & {lam} & {f(a['nov_dtrain'],3)} & {f(a['nov_sig'],3)} & "
                f"{sg(a['nov_pot'],3)} & {f(a['dtrain_sig'],3)} & {a['n']}")
# ------------------------------------------------- TURNOVER of disagreement, per target
_frr = L('frontier_v2_results.json')['results']
_rw2 = [dict(r, target=t) for t, rs in _frr.items() for r in rs]
_nv2 = [r['novelty'] for r in _rw2]
_lo2, _hi2 = min(_nv2), max(_nv2)
_edge = [_lo2 + (_hi2 - _lo2) * i / 8 for i in range(9)]


def _bin_of(v):
    for i in range(8):
        if v <= _edge[i + 1]:
            return i
    return 7


# The per-target cells are small and unbalanced: DRD2's seventh bin holds 7 runs and DRD3's
# holds 9, and those two carry the largest differences. A sign count over five targets says
# nothing about whether any of them is distinguishable, so each row carries its own interval
# from a bootstrap that resamples seeds within the target, the same block unit Figure 2 uses.
import numpy as _np2
_turn_rows, _turn_ci, _turn_diff = [], {}, {}
for _t in T:
    _sel = [r for r in _rw2 if r['target'] == _t]
    _mm, _nn = [], []
    for _k in (6, 7):
        _v = [r['sigma'] for r in _sel if _bin_of(r['novelty']) == _k]
        _nn.append(len(_v)); _mm.append(sum(_v) / len(_v) if _v else float('nan'))
    _sd2 = sorted({r['seed'] for r in _sel})
    _by2 = {_s: [r for r in _sel if r['seed'] == _s] for _s in _sd2}
    _mean_k = lambda rs, k: (_np2.mean([r['sigma'] for r in rs if _bin_of(r['novelty']) == k])
                             if any(_bin_of(r['novelty']) == k for r in rs) else _np2.nan)
    _rng2 = _np2.random.default_rng(3)
    _draws = []
    for _ in range(2000):
        _pick = [x for _s in _rng2.choice(_sd2, len(_sd2), replace=True) for x in _by2[_s]]
        _d2 = _mean_k(_pick, 6) - _mean_k(_pick, 7)
        if not _np2.isnan(_d2):
            _draws.append(_d2)
    _ci2 = [float(_np2.percentile(_draws, 2.5)), float(_np2.percentile(_draws, 97.5))]
    _turn_ci[_t] = _ci2
    _turn_diff[_t] = _mm[0] - _mm[1]
    _turn_rows.append(f"{LAB[_t]} & {_nn[0]} & {f(_mm[0])} & {_nn[1]} & {f(_mm[1])} & "
                      f"{sg(_mm[0] - _mm[1], 2)} & {_ci_fmt(_ci2)}")
tab('tab:s-turnover',
    'Disagreement in the two highest novelty bins, per target. Runs are grouped into eight '
    'equal-width bins of achieved generated-set novelty; the table gives the mean disagreement '
    'in the seventh and eighth bins and their difference. A positive difference means '
    'disagreement falls at the highest novelty rather than continuing to rise. '
    'It does so on %s, and rises instead on %s. The cells are small and unbalanced, so each '
    'row carries a 95\\%% interval from a bootstrap resampling seeds within the target: the '
    'difference is separated from zero on %s, and on %s it is not, which a sign count over five '
    'targets would not show. The pooled difference and its interval, from a bootstrap over '
    'target-seed blocks, are given in the main article.'
    % (_andlist([LAB[t] for t in T if _turn_diff[t] > 0]),
       _andlist([LAB[t] for t in T if _turn_diff[t] <= 0]),
       _andlist([LAB[t] for t in T if _turn_ci[t][0] > 0 or _turn_ci[t][1] < 0]),
       _andlist([LAB[t] for t in T if _turn_ci[t][0] <= 0 <= _turn_ci[t][1]])),
    r'Target & $n_7$ & Bin 7 & $n_8$ & Bin 8 & Difference & 95\% interval',
    _turn_rows, 'lrrrrrr')

tab('tab:s-frontier',
    'Novelty-driven shift during optimization. Correlations across runs between the achieved '
    'novelty of the generated set and its distance to the training compounds, disagreement, '
    'and predicted potency, for two search procedures at two uncertainty penalties, with the '
    'distance-to-disagreement correlation in the last column. Novelty rises with distance and '
    'with disagreement, and falls with predicted potency, in all four rows including the two '
    'with the uncertainty penalty at zero, so none of it is induced by that penalty. Part of '
    'the novelty-to-distance association is definitional: the starting compounds are drawn '
    'from the training set, so $d\\leq\\nu$ for every molecule and the two cannot be '
    'independent. What the column adds is that a search pushed toward novelty leaves the '
    'support of the whole training set, not only of its own seeds.',
    r'Optimizer & $\lambda$ & $\rho(\nu,d)$ & $\rho(\nu,\sigma_T)$ & $\rho(\nu,\hat y)$ & $\rho(d,\sigma_T)$ & $n$',
    rows, 'llrrrrr')

# ---------------------------------------------------------------- S10 method per-cell
rows = []
for r in meth:
    a = r['agg']
    rows.append(f"{LAB[r['target']]} & {r['k']} & {f(a['stga_ecfp'],3)} & {f(a['randtriage'],3)} & "
                f"{f(a['graphga'],3)} & {sg(a['d_vs_ga'])} & {_psci(a['p_vs_ga'])} & "
                f"{f(a['frac_pos_ga'],2)}")
tab('tab:s-method',
    'Method comparison per target and support size. Mean reward of the top ten generated '
    'molecules, excluding the support compounds and any structure in the activity model\'s '
    'training set, over 25 seeds at an exactly fixed budget of 300 evaluations. $\\Delta$ and '
    'the two-sided paired $P$ are the surrogate-triaged search minus vanilla Graph GA; win is '
    'the fraction of seeds in which it is ahead.',
    r'Target & $k$ & ST-GA & Random triage & Graph GA & $\Delta$ & $P$ & Win', rows, 'llrrrrrr')

# ---------------------------------------------------------------- S11 target-level
tgt = {t: st.mean([c['stga_ecfp'] - c['graphga'] for r in meth if r['target'] == t
                   for c in r['per_seed']]) for t in T}
tl = st.mean(tgt.values()); se = st.stdev(tgt.values()) / (5 ** 0.5)
from scipy import stats as sps
tst, pv = sps.ttest_1samp(list(tgt.values()), 0.0)
ci = sps.t.interval(0.95, 4, loc=tl, scale=se)
h = hier['target_level']
rows = [
    'Fingerprint & ' + ' & '.join(f(tgt[t], 3) for t in T) +
    f" & ${tl:+.3f}$ [{ci[0]:+.3f}, {ci[1]:+.3f}] & {f(pv,3)}",
    'Dual encoder & ' + ' & '.join(f(h['target_means'][t], 3) for t in T) +
    f" & ${h['mean']:+.3f}$ [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}] & {f(h['p'],3)}",
]
tab('tab:s-targetlevel',
    ('Target-level summary of the method effect. Per-target mean paired difference against '
     'Graph GA, pooled over the support sizes $k$ of %s, and the two-sided one-sample $t$-test '
     'on the five per-target means ($\\mathrm{df}=4$). Supplementary Table~\\ref{tab:s-latent} '
     'reports the same dual-encoder surrogate at $k=10$ alone, so its figures differ from the '
     'dual-encoder row here. With five targets an assumption-free sign test cannot fall below '
     '$P=0.0625$ even when every target is positive, so the effect is best described as '
     'positive in all five targets studied.'
     % _andlist(['$%d$' % k for k in sorted({r['k'] for r in meth})])),
    r'Surrogate & ' + ' & '.join(LAB[t] for t in T) + r' & Mean [95\% CI] & $P$',
    rows, 'lrrrrrrr')

# ---------------------------------------------------------------- S12 negatives
rows = []
for t in T:
    b = beta[t]; d = b['beta1_vs_beta0_novel']
    rows.append(f"{LAB[t]} & " + ' & '.join(f(b[f'beta{x}']['top10_novel'], 3)
                for x in ['0.0', '0.5', '1.0', '2.0'])
                + f" & {f(b['graphga']['top10_novel'],3)} & {sg(d['delta'])} & {f(d['p'],3)}")
tab('tab:s-beta',
    'Uncertainty in the acquisition does not help. Mean top-ten reward at $k=10$ over 25 seeds '
    'for the acquisition $\\mu+\\beta\\varsigma$, where $\\varsigma$ is the triage surrogate\'s own '
    'dispersion and not the activity model\'s disagreement $\\sigma_T$ used elsewhere. The paired '
    'difference between $\\beta=1$ and $\\beta=0$ is negligible everywhere: it is %s on %s, the '
    'only target where it reaches an exact $P$ below %s, and %s to %s on the other four. It is '
    'negative on DRD3, the one target where the reward falls monotonically as $\\beta$ rises, '
    'by %s at $\\beta=2$. Every setting of $\\beta$ beats Graph GA, so the surrogate helps and the '
    'uncertainty term in its acquisition does not.'
    % (sg(beta[min(T, key=lambda t: beta[t]['beta1_vs_beta0_novel']['p'])]
             ['beta1_vs_beta0_novel']['delta']),
       LAB[min(T, key=lambda t: beta[t]['beta1_vs_beta0_novel']['p'])],
       f(sorted(beta[t]['beta1_vs_beta0_novel']['p'] for t in T)[1], 2),
       f(sorted(beta[t]['beta1_vs_beta0_novel']['p'] for t in T)[1], 2),
       f(max(beta[t]['beta1_vs_beta0_novel']['p'] for t in T), 2),
       sg(beta['drd3']['beta2.0']['top10_novel'] - beta['drd3']['beta0.0']['top10_novel'])),
    r'Target & $\beta=0$ & $\beta=0.5$ & $\beta=1$ & $\beta=2$ & Graph GA & $\Delta_{\beta=1-0}$ & $P$',
    rows, 'lrrrrrrr')

rows = []
for t in T:
    e = ecfp[t]
    rows.append(f"{LAB[t]} & {f(e['latent'],3)} & {f(e['ecfp'],3)} & {f(e['graphga'],3)} & "
                f"{sg(e['latent_minus_ecfp'])} & {_psci(e['p_lat_ecfp'])}")
tab('tab:s-latent',
    'A learned latent representation underperforms fingerprints. Mean top-ten reward at $k=10$ '
    'over 25 seeds, with the two-sided paired $P$ for the latent surrogate minus the '
    'fingerprint surrogate. The latent surrogate loses to fingerprints on all five targets, but '
    'it still beats Graph GA on all five, which is why that column is shown: the learned '
    'representation is worse than a fingerprint, not worse than no surrogate at all.',
    r'Target & Latent & Fingerprint & Graph GA & Latent $-$ Fingerprint & $P$', rows, 'lrrrrr')

# ------------------------------------------------- COMPATIBILITY warm start (negative)
_cgr = L('compat_gen_analysis.json')['rows']
rows = []
for r in sorted(_cgr, key=lambda x: (x['target'], -x['C_nn'])):
    # n_target in the frozen file is a raw record count. The unit everywhere else in the paper
    # is the standardized parent, and Table 1 keeps the two apart, so report parents here too.
    rows.append(f"{LAB[r['source']]} & {LAB[r['target']]} & {f(r['C_nn'], 3)} & "
                f"{rel[r['target']]['duplicates']['n_unique']:,} & {sg(r['gain_mean'], 3)} & "
                f"{f(r['gain_std'], 3)}".replace(',', '{,}'))
# The 20 ordered pairs share targets: each of the five appears four times as source and four
# times as destination, so an ordinary Pearson P has no independent-error justification. Permute
# the five target labels instead and recompute, which is exact over all 120 relabelings.
import itertools as _it16
from scipy import stats as _sp16
_cm16 = {(x['source'], x['target']): x['C_nn'] for x in _cgr}
_gm16 = {(x['source'], x['target']): x['gain_mean'] for x in _cgr}
_T16 = sorted({x['target'] for x in _cgr})
_obs16 = _sp16.pearsonr([x['C_nn'] for x in _cgr], [x['gain_mean'] for x in _cgr])[0]
_null16 = []
for _pm in _it16.permutations(_T16):
    _mp = dict(zip(_T16, _pm))
    _c, _g = [], []
    for (_s16, _t16) in _gm16:
        _key = (_mp[_s16], _mp[_t16])
        if _key in _cm16:
            _c.append(_cm16[_key]); _g.append(_gm16[(_s16, _t16)])
    if len(_c) == len(_gm16):
        _null16.append(_sp16.pearsonr(_c, _g)[0])
_pperm16 = (1 + sum(1 for v in _null16 if abs(v) >= abs(_obs16))) / (len(_null16) + 1)

tab('tab:s-compat',
    'Warm-starting the surrogate on a source target does not help. Each row is a source-target '
    'pair: the surrogate is pre-trained on the source and then used on the destination, and the '
    'gain is the change in top-ten reward against the same search started cold, over 25 seeds. '
    '$C_{\\mathrm{nn}}$ is the structural compatibility of the pair. The gains are small and of '
    'both signs, and across the %d pairs they are uncorrelated with compatibility '
    '($r=%s$). The pairs share targets, each of the five appearing four times as source and '
    'four times as destination, so an ordinary $P$ would have no independent-error '
    'justification; permuting the five target labels and recomputing, which is exact over all '
    '%d relabelings, gives $P=%s$. $C_{\\mathrm{nn}}$ is carried over from the companion '
    'prediction study rather than fitted here, and $n_{\\mathrm{target}}$ counts standardized '
    'parents, as in Table 1.'
    % (len(_cgr), '%.2f' % _obs16, len(_null16), '%.2f' % _pperm16),
    r'Source & Destination & $C_{\mathrm{nn}}$ & $n_{\mathrm{target}}$ & Mean gain & SD',
    rows, 'llrrrr')

rows = []
for t in T:
    d = rec[t]
    # Without the threshold a reader cannot see why 6.80 fails on SCD-1 while 6.95 passes on
    # DRD2: the thresholds differ. The caption's central claim is uncheckable without it.
    rows.append(f"{LAB[t]} & {f(d['retrained_pred'])} & {om[t]['threshold']:.1f} & "
                f"{f(d['measured'])} & {f(d['retrained_sigma'])} & "
                f"{f(d['rec_stga'])} & {f(d['rec_graphga'])} & {f(d['null_sim'])}")
tab('tab:s-recovery',
    'Similarity to withheld scaffold clusters. An entire Bemis--Murcko cluster of actives is '
    'removed from both the seeds and the training set and the model retrained. Every target is '
    'underpredicted against the withheld cluster\'s measured mean, by %s to %s '
    '$\\mathrm{pIC}_{50}$, and all five carry elevated disagreement. What separates them is the '
    'threshold: the prediction still clears it on four, and falls below on %s alone. %s is '
    'underpredicted by almost as much, %s against %s, but clears a lower threshold, so the '
    'failure on %s is partly a matter of where its threshold sits. Recovery is the mean closest '
    'Tanimoto similarity of any generated molecule to the withheld cluster, which is not exact '
    'scaffold recovery, and is not improved by the triage.'
    % (f(min(rec[t]['measured'] - rec[t]['retrained_pred'] for t in T)),
       f(max(rec[t]['measured'] - rec[t]['retrained_pred'] for t in T)),
       _andlist([LAB[t] for t in T if rec[t]['retrained_pred'] < om[t]['threshold']]),
       LAB[sorted((t for t in T if rec[t]['retrained_pred'] >= om[t]['threshold']),
                  key=lambda t: -(rec[t]['measured'] - rec[t]['retrained_pred']))[0]],
       f(max(rec[t]['measured'] - rec[t]['retrained_pred'] for t in T
             if rec[t]['retrained_pred'] >= om[t]['threshold'])),
       f(max(rec[t]['measured'] - rec[t]['retrained_pred'] for t in T
             if rec[t]['retrained_pred'] < om[t]['threshold'])),
       _andlist([LAB[t] for t in T if rec[t]['retrained_pred'] < om[t]['threshold']])),
    r'Target & Predicted & Threshold & Measured & $\sigma_T$ & ST-GA & Graph GA & Null',
    rows, 'lrrrrrrr')

# ---------------------------------------------------------------- S13 in-domain novelty
rows = []
for t in T:
    r = rel[t]
    # Both group sizes, not one unlabelled n: the split is at the median so they are close,
    # and showing both lets a reader confirm that rather than assume it.
    rows.append(f"{LAB[t]} & {f(r['novel_in_domain_rmse'])} & {f(r['novel_out_domain_rmse'])} & "
                f"{f(r['novel_in_domain_sigma'])} & {f(r['novel_out_domain_sigma'])} & "
                f"{r['n_novel_in_domain']:,} & {r['n_novel_out_domain']:,}".replace(',', '{,}'))
tab('tab:s-indomain',
    'Nearer- and farther-training compounds at high novelty. Within the most novel third of '
    'each target, compounds are split at the median distance to the nearest training compound. '
    'Those nearer the training compounds have lower RMSE on four of the five targets, SCD-1 '
    'being the exception, where the gap is %s against %s to %s on the other four, some %d times '
    'smaller than the least of them, and lower mean '
    'disagreement on all five. The split restricts rather than matches on novelty, so residual '
    'differences in novelty between the two groups are not controlled, and it is a relative '
    'grouping within that third rather than a validated domain boundary.'
    % (sg(rel['scd1']['novel_out_domain_rmse'] - rel['scd1']['novel_in_domain_rmse'], 3),
       sg(min(rel[t]['novel_out_domain_rmse'] - rel[t]['novel_in_domain_rmse']
              for t in T if t != 'scd1'), 3),
       sg(max(rel[t]['novel_out_domain_rmse'] - rel[t]['novel_in_domain_rmse']
              for t in T if t != 'scd1'), 3),
       round(min(abs(rel[t]['novel_out_domain_rmse'] - rel[t]['novel_in_domain_rmse'])
                 for t in T if t != 'scd1')
             / abs(rel['scd1']['novel_out_domain_rmse']
                   - rel['scd1']['novel_in_domain_rmse']))),
    r'Target & \multicolumn{2}{c}{RMSE} & \multicolumn{2}{c}{$\sigma_T$} & \multicolumn{2}{c}{$n$}\\'
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}'
    r' & Nearer & Farther & Nearer & Farther & Nearer & Farther',
    rows, 'lrrrrrr')

# ---------------------------------------------------------------- S14 curation provenance
if prov:
    rows = []
    # Every stage the curation script records, so raw minus the drops minus the records
    # collapsed by parent grouping equals the parent count exactly. The previous version
    # showed three of the five drop categories and omitted parent grouping altogether, which
    # is the largest reduction of all: 5,579 records on DRD2.
    for t, v in prov.items():
        d = v['dropped']
        _na = d.get('standard_type_not_affinity', 0)
        _inv = d.get('no_value', 0) + d.get('nonpositive_value', 0)
        _up = d.get('unparsable_structure', 0)
        _recs = v['n_raw_records'] - _na - _inv - _up - d.get('non_human', 0)
        _coll = _recs - v['n_unique_parents']
        assert _recs == v['n_after_filters'], 'curation flow does not reconcile for ' + t
        rows.append(f"{LAB.get(t,t)} & {v['chembl_id']} & {v['n_raw_records']:,} & "
                    f"{_na:,} & {_inv} & {_up} & {_recs:,} & {_coll:,} & "
                    f"{v['n_unique_parents']:,} & "
                    f"{v['year_range'][0]}--{v['year_range'][1]}".replace(',', '{,}'))
    tab('tab:s-curation',
        'ChEMBL curation flow for the re-curated targets. The query returns only records with '
        'an exact relation, nanomolar units and a human target, so those filters remove nothing '
        'here and no species column is shown. Of what returns, records are dropped for a '
        'standard type outside the four eligible ones, for a missing or non-positive value, '
        'and for a structure RDKit '
        'cannot parse; the remainder are then grouped on the standardized parent InChIKey. '
        'Every stage is listed, so raw minus the three drop columns gives Records, and Records '
        'minus Collapsed gives Parents, on each row. Parent grouping is the largest reduction: '
        'it removes %s records on DRD2 against %s for all the filters together. These files '
        'carry publication years and support the temporal analysis; FADS is not included '
        'because ChEMBL holds only two activity records for it.'
        % ('{:,}'.format(prov['drd2']['n_after_filters'] - prov['drd2']['n_unique_parents']).replace(',', '{,}'),
           '{:,}'.format(prov['drd2']['n_raw_records'] - prov['drd2']['n_after_filters']).replace(',', '{,}')),
        r'Target & ChEMBL & Raw & \shortstack[r]{Ineli-\\gible} & Invalid & '
        r'\shortstack[r]{Unpar-\\sable} & Records & \shortstack[r]{Col-\\lapsed} & '
        r'Parents & Years',
        rows, 'llrrrrrrrr', tight=True)

# Endpoint composition of the pooled response, for both cohorts, and its shift at the
# temporal cutoff. The cross-validation files retain no endpoint field, so their mix is
# recovered by matching structures back to ChEMBL; the matched share is reported with it.
_ep = json.load(open(os.path.join(CW, 'endpoint_composition.json')))
NICE = {'scd1': 'SCD-1', 'fads': 'FADS', 'nk1r': 'NK1R', 'drd2': 'DRD2', 'drd3': 'DRD3'}
ORDER = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']
TYPES = ['IC50', 'Ki', 'Kd', 'EC50']


def _mix(d):
    return ' & '.join(('--' if not d else f"{d.get(k, 0.0):.1f}") for k in TYPES)


rows = []
for t in ORDER:
    v = _ep['cv_cohort'][t]
    # n must mean one thing down the column. The endpoint file counts distinct canonical
    # SMILES, which lands on the parent count for SCD-1 and on the record count for the
    # other three, while the temporal rows below are parent counts. Use parents throughout.
    matched = '--' if not v['matched'] else f"{v['matched_pct']:.0f}"
    rows.append(f"{NICE[t]} & cross-validation & {rel[t]['duplicates']['n_unique']:,} & "
                f"{matched} & {_mix(v['types_pct'])}".replace(',', '{,}'))
for t in ['scd1', 'nk1r', 'drd2', 'drd3']:
    v = _ep['temporal_cohort'][t]
    for era, key, nkey in [('temporal, pre-%d' % _ep['cutoff'], 'pre_pct', 'n_pre'),
                           ('temporal, %d on' % _ep['cutoff'], 'post_pct', 'n_post')]:
        rows.append(f"{NICE[t]} & {era} & {v[nkey]:,} & -- & "
                    f"{_mix(v[key])}".replace(',', '{,}'))
tab('tab:s-endpoint',
    'Activity endpoint composition of the pooled response. Every activity is reported on one '
    'negative-logarithmic molar scale named $\\mathrm{pIC}_{50}$ by convention, but that scale '
    'is populated from four ChEMBL standard types and most of it is an affinity rather than a '
    'potency measurement. $n$ counts standardized parent structures throughout, as in '
    'Table 1. The cross-validation files keep only structure and activity, so their '
    'composition is recovered by matching those structures back to ChEMBL; Matched is the '
    'share of them that match, and the percentages on those rows are of the ChEMBL records '
    'behind the matched structures rather than of the dataset as a whole. FADS is a '
    'literature panel and is not recoverable this way. On the temporal rows the percentages '
    'are of the records behind each parent, and the split at the cutoff shows whether '
    'endpoint composition itself shifts across it.',
    r'Target & Cohort & $n$ & Matched (\%) & IC$_{50}$ & $K_i$ & $K_d$ & EC$_{50}$',
    rows, 'llrrrrrr')

# The main article is a separate document, so it cannot \\ref{} into this one and must cite
# Supplementary tables by number. Emit those numbers as macros, in the order the tables are
# actually written, so inserting or reordering a table updates the article automatically.
import re as _re
_here = os.path.dirname(os.path.abspath(__file__))
# ---------------------------------------------------------------- model bake-off
MODLAB = {'rf': 'Random forest', 'extratrees': 'Extremely randomized trees',
          'histgb': 'Histogram gradient boosting'}
bk_rows = []
for _t in T:
    e = om[_t]
    for _m in ('rf', 'extratrees', 'histgb'):
        b = e['bakeoff'][_m]
        star = r'$^{*}$' if _m == e['selected_model'] else ''
        bk_rows.append('%s & %s%s & %.3f (%.3f) & %.3f (%.3f)'
                       % (LAB[_t] if _m == 'rf' else '', MODLAB[_m], star,
                          b['spearman'][0], b['spearman'][1], b['r2'][0], b['r2'][1]))
tab('tab:s-bakeoff',
    'Model comparison behind the activity model. Three tree ensembles are compared on each '
    'target by five-fold cross-validation repeated over five seeds; entries are the mean across '
    'seeds with its standard deviation in parentheses, and $^{*}$ marks the model selected. The '
    'random forest is selected on every target. Two qualifications belong with this table. The '
    'comparison shares its cross-validation with the analyses reported elsewhere, which use the '
    'first of these five seeds, so the selected model carries the mild optimism of having been '
    'chosen on data it is then scored on. And the choice is constrained independently of '
    'accuracy: the quantity this paper studies is the dispersion across ensemble members, which '
    'a forest and extremely randomized trees expose directly and a boosted ensemble, whose '
    'members are fitted to residuals rather than independently, does not supply in a comparable '
    'form.',
    r'Target & Model & Spearman & $R^2$',
    bk_rows, 'llrr')

# ---------------------------------------------------------------- support resampling
sr = L('support_resample.json')
sr_rows = []
for _t in T:
    if _t not in sr:
        continue
    for _k in (5, 10, 20):
        c = sr[_t][f'k{_k}']
        pn, pp = c['spearman_nov_err'], c['partial_nov_err_given_dtr']
        sr_rows.append('%s & %d & %s & %s & %s'
                    % (LAB[_t], _k,
                       '%s $[%+.2f,%+.2f]$' % (sg(pn['median'], 2), pn['lo'], pn['hi']),
                       '%s $[%+.2f,%+.2f]$' % (sg(pp['median'], 2), pp['lo'], pp['hi']),
                       '%.0f' % (100 * c['near_beats_far_frac'])))
tab('tab:s-support',
    'Seed novelty under resampled starting compounds. Novelty is measured against a draw of $k$ '
    'active training compounds, so the single draw used elsewhere fixes it. Here the draw is '
    'repeated %d times at each $k$, with the folds, the fitted forest, the predictions and '
    '$d_{\\mathrm{train}}$ held identical across draws, so only novelty and the gap are '
    'recomputed. Entries are the median over draws with the central $95\\%%$ of them. The '
    'partial correlation is small in every cell, so the conclusion that seed novelty adds little '
    'once nearest-training distance is known does not depend on which compounds start the '
    'search. FADS is the exception in sign: more novel compounds carry lower error there. The '
    'last column is the share of draws in which the high-novelty nearer-training group has the '
    'lower RMSE; it is decisive on DRD2, DRD3 and NK1R and closer to even on SCD-1 and FADS, so '
    'which of those two is the weaker case is itself draw-dependent.' % sr['n_draws'],
    r'Target & $k$ & $\rho(\nu,|e|)$ & Partial $\mid d_{\mathrm{train}}$ & Nearer (\%)',
    sr_rows, 'lrrrr', tight=True)

_labels = _re.findall(r'\\label\{tab:s-(.*?)\}', '\n'.join(OUTL))


def _figure_order(path, prefix):
    """Labels of the float environments in `path`, in the order LaTeX will number them.

    Figures are hand-placed rather than generated, so their numbers are read back out of the
    source. Both documents cite the other's floats and neither can cross-reference into it.
    """
    if not os.path.exists(path):
        return []
    src = open(path, encoding='utf-8').read()
    out = []
    for blk in _re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', src, _re.S):
        m = _re.search(r'\\label\{' + prefix + r'(.*?)\}', blk)
        if m:
            out.append(m.group(1))
    return out


_sfigs = _figure_order(os.path.join(_here, 'npjDD_SI.tex'), 'sfig:')
_afigs = _figure_order(os.path.join(_here, 'npjDD_Reliability.tex'), 'fig:')
_refs = os.path.join(_here, 'si_refs.tex')
with open(_refs, 'w') as _rf:
    _rf.write('% Generated by make_si_tables.py - DO NOT EDIT BY HAND.\n')
    _rf.write('% Cross-document numbers. The article and the Supplementary Information are\n')
    _rf.write('% separate documents, so neither can \\ref{} into the other and both cite by\n')
    _rf.write('% number. Regenerated from the actual order, so reordering cannot desync them.\n')
    for _i, _lab in enumerate(_labels, 1):
        _rf.write('\\newcommand{\\Tab%s}{S%d}\n' % (_lab.capitalize(), _i))
    for _i, _lab in enumerate(_sfigs, 1):
        _rf.write('\\newcommand{\\SFig%s}{S%d}\n' % (_lab.capitalize(), _i))
    for _i, _lab in enumerate(_afigs, 1):
        _rf.write('\\newcommand{\\ArtFig%s}{%d}\n' % (_lab.capitalize(), _i))
print('wrote %s with %d table, %d Supplementary-figure and %d article-figure macros'
      % (_refs, len(_labels), len(_sfigs), len(_afigs)))
with open(OUT, 'w') as fh:
    fh.write('% Generated by make_si_tables.py from the frozen outputs - DO NOT EDIT BY HAND.\n')
    fh.write('\n'.join(OUTL))
print(f'wrote {OUT} with {sum(1 for l in OUTL if l.startswith(chr(92)+"begin{table}"))} tables')

