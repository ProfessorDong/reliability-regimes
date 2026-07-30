"""Generate si_tables.tex from the frozen analysis outputs.

Same principle as make_numbers.py: no Supplementary table is hand-typed. Every value is
read from outputs/cwm_v1/*.json (or data/chembl_v2/curation_provenance.json), so a table
cannot drift out of step with the analysis it reports. Regenerate after any analysis
change, then recompile the SI.

    python make_si_tables.py    ->  writes si_tables.tex
"""
from __future__ import annotations
import json, os, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CW = os.path.join(ROOT, 'drug_discovery', 'theranostics_current', 'outputs', 'cwm_v1')
PROV = os.path.join(ROOT, 'drug_discovery', 'theranostics_current', 'data', 'chembl_v2',
                    'curation_provenance.json')
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
prov = json.load(open(PROV)) if os.path.exists(PROV) else {}

OUTL = []


def tab(label, caption, header, rows, spec, note=''):
    OUTL.append(r'\begin{table}[htbp]\centering\caption{%s}\label{%s}' % (caption, label))
    OUTL.append(r'\begin{tabular}{@{}%s@{}}\toprule' % spec)
    OUTL.append(header + r'\\\midrule')
    OUTL.extend(r + r'\\' for r in rows[:-1])
    OUTL.append(rows[-1] + r'\\\bottomrule')
    OUTL.append(r'\end{tabular}' + (('\n\\par\\smallskip\\footnotesize ' + note) if note else ''))
    OUTL.append(r'\end{table}' + '\n')


def f(x, d=2):
    return ('%.' + str(d) + 'f') % x


def sg(x, d=3):
    return ('$%+.' + str(d) + 'f$') % x


# ---------------------------------------------------------------- S1 datasets
rows = []
for t in T:
    d = rel[t]['duplicates']
    rows.append(f"{LAB[t]} & {d['n_rows']:,} & {d['n_unique']:,} & {d['n_duplicate_rows']} & "
                f"{d['n_compounds_with_replicates']} & {f(d['mean_within_compound_sd'])}".replace(',', '{,}'))
tab('tab:s-data',
    'Dataset composition after standardization. Records are grouped on the standardized '
    'parent InChIKey, so salts, charge states and tautomer variants of one compound collapse '
    'to a single structure. Within-compound spread is the mean standard deviation of the '
    'replicate measurements of compounds that have more than one record, and bounds the error '
    'any model can reach on that target.',
    r'Target & Records & Parent structures & Duplicates removed & With replicates & Within-compound SD',
    rows, 'lrrrrr')

# ---------------------------------------------------------------- S2 protocols
rows = []
for t in T:
    e = om[t]; bo = e['bakeoff'][e['selected_model']]; sc = e.get('scaffold', {})
    r2 = sc.get('r2'); nact = sc.get('n_active_test')
    r2s = f(r2) if isinstance(r2, (int, float)) else 'n/a'
    auc = 'n/a' if not nact else '---'
    tt = tmp.get(t)
    tcell = f(tt['rmse_test']) if tt else 'n/a'
    rows.append(f"{LAB[t]} & {f(bo['spearman'][0],3)} & {f(bo['r2'][0],3)} & {f(bo['auc'][0],3)} & "
                f"{f(sc.get('spearman', float('nan')),3)} & {r2s} & {tcell}")
tab('tab:s-protocol',
    'Activity-model performance under three evaluation protocols. Random-CV is five-fold '
    'cross-validation on parent structures. Scaffold is a Bemis--Murcko split used as a '
    'structural-shift stress test; where the held-out fold contains no active compounds the '
    'classification AUC is undefined while the continuous metrics remain defined. Temporal '
    'RMSE is on compounds published from 2015 onwards after training on earlier ones, and is '
    'available only for the four targets re-curated from ChEMBL.',
    r'Target & \multicolumn{3}{c}{Random-CV} & \multicolumn{2}{c}{Scaffold} & Temporal\\'
    r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}'
    r' & Spearman & $R^2$ & AUC & Spearman & $R^2$ & RMSE',
    rows, 'lrrrrrr')

# ---------------------------------------------------------------- S3 error stratification
rows = []
for t in T:
    q = rel[t].get('rmse_by_sigma_quintile', [float('nan')] * 5)
    rows.append(f"{LAB[t]} & {f(rel[t]['spearman_sigma_err'],3)} & " + ' & '.join(f(v) for v in q))
p = rel['pooled']
rows.append(r'\midrule Pooled & ' + f(p['spearman_sigma_err'], 3) + ' & ' +
            ' & '.join(f(v) for v in p.get('rmse_by_sigma_quintile', [float('nan')] * 5)))
tab('tab:s-strat',
    'Error stratification by ensemble disagreement, in distribution. Spearman correlation '
    'between the disagreement score and the absolute out-of-fold error, and RMSE within '
    'quintiles of the disagreement score (Q1 lowest). This is a ranking result, not a '
    'calibration; calibrated coverage is reported separately.',
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
    'weight. The curve is monotone on four targets; on SCD-1 it is flat.',
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
tab('tab:s-novelty',
    'Novelty, chemical distance and measured error. $\\nu$ is novelty relative to the ten '
    'starting compounds and $d$ the distance to the nearest training compound. Novelty is a '
    'weak predictor of error and loses what little signal it has once distance is controlled '
    'for, whereas disagreement predicts error after controlling for both.',
    r'Target & $\rho(\nu,e)$ & $\rho(e,\nu\mid d)$ & $\rho(d,e)$ & $\rho(\sigma_T,e)$ & $\rho(e,\sigma_T\mid \nu,d)$',
    rows, 'lrrrrr')

# ---------------------------------------------------------------- S6 conformal
rows = []
for a in ['alpha0.2', 'alpha0.1', 'alpha0.05']:
    c = con['pooled'][a]
    rows.append(f"{f(c['target_coverage'],2)} & {f(c['standard_coverage'],3)} & {f(c['standard_width_median'])} & "
                f"{f(c['adaptive_coverage'],3)} & {f(c['adaptive_width_median'])} & "
                f"{f(c['width_ratio_adaptive_over_standard'])}")
tab('tab:s-conformal',
    'Split conformal intervals in distribution. Standard intervals use the absolute residual '
    'and give every compound the same width; adaptive intervals normalise the residual by the '
    'disagreement score. Both reach their nominal coverage, and the adaptive intervals are '
    'narrower at equal coverage, so the score carries usable information.',
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
        'records is shown for comparison. The error increase over the size-matched control, the '
        'set of targets that retain the error ranking and the set that lose it are the same '
        'under both.',
        r'Target & \multicolumn{4}{c}{First disclosure (used)} & \multicolumn{4}{c}{Median year}\\'
        r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}'
        r' & $n_{\mathrm{test}}$ & RMSE & $\rho(\sigma_T,e)$ & Coverage'
        r' & $n_{\mathrm{test}}$ & RMSE & $\rho(\sigma_T,e)$ & Coverage',
        rows, 'rrrrrrrrr')

# ---------------------------------------------------------------- S7 TEMPORAL (key)
rows = []
for t in ['scd1', 'nk1r', 'drd2', 'drd3']:
    if t not in tmp:
        continue
    d = tmp[t]; c = d['control_random_same_size']
    ci = d.get('spearman_sigma_err_ci95')
    rho = sg(d['spearman_sigma_err'], 2) + (
        f" [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else '')
    rows.append(f"{LAB[t]} & {d['n_train']:,} & {d['n_test']:,} & {f(d['rmse_test'])} & {f(c['rmse'])} & "
                f"{rho} & {sg(c['spearman_sigma_err'],2)} & "
                f"{f(d['conformal_coverage_adaptive'],3)} & {f(c['conformal_coverage_adaptive'],3)}".replace(',', '{,}'))
tp = tmp['pooled']
rows.append(r'\midrule Pooled & --- & ' + f"{tp['n']:,}".replace(',', '{,}') + ' & ' + f(tp['rmse']) +
            ' & --- & ' + sg(tp['spearman_sigma_err'], 2) + ' & --- & ' +
            f(tp['conformal_coverage_adaptive'], 3) + ' & ---')
tab('tab:s-temporal',
    'Temporal shift and its size-matched control. Models are trained on compounds published '
    'before 2015 and evaluated on those published later. Each temporal split is repeated five '
    'times with training, calibration and test sets of identical size drawn at random from the '
    'same pool, which separates distribution shift from the smaller training set a temporal '
    'split implies, and is the only like-for-like comparator because this cohort is re-curated '
    'independently of the five datasets of Table S1. Error rises on every target and coverage '
    'falls below nominal on three of four, whereas the error ranking degrades on DRD2 and DRD3 '
    'and is unchanged on SCD-1 and NK1R. Brackets give the 95\\% percentile bootstrap interval on '
    'the temporal rank correlation. The pooled correlation is lower than any per-target value '
    'because pooling mixes targets whose errors differ in scale.',
    r'Target & $n_{\mathrm{train}}$ & $n_{\mathrm{test}}$ & \multicolumn{2}{c}{RMSE} & '
    r'\multicolumn{2}{c}{$\rho(\sigma_T,e)$} & \multicolumn{2}{c}{Coverage at 0.900}\\'
    r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}'
    r' & & & Temporal & Control & Temporal & Control & Temporal & Control',
    rows, 'lrrrrrrrr')

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
tab('tab:s-pool',
    'Acquisition against measured activity. All labels in the pool are hidden except ten known '
    'actives; each strategy spends 300 queries and every query reveals a compound\'s true '
    'measured activity. Entries are the number of genuine top-percentile compounds acquired, '
    'averaged over twenty seeds, and the last row is the mean enrichment relative to random '
    'selection. Penalising uncertainty, whether by a lower-confidence rule or a conformal '
    'bound, finds fewer true actives than selecting on the predicted mean.',
    r'Target & Pool & True top 1\% & Random & Greedy & UCB & LCB & Conformal', rows, 'lrrrrrrr')

# ---------------------------------------------------------------- S9 frontier
rows = []
for k in ['graphga_lam0.0', 'graphga_lam0.1', 'stga_lam0.0', 'stga_lam0.1']:
    a = fr[k]
    opt, lam = ('Graph GA' if 'graphga' in k else 'Surrogate-triaged'), k.split('lam')[1]
    rows.append(f"{opt} & {lam} & {f(a['nov_dtrain'],3)} & {f(a['nov_sig'],3)} & "
                f"{sg(a['nov_pot'],3)} & {f(a['dtrain_sig'],3)} & {a['n']}")
tab('tab:s-frontier',
    'Novelty-driven shift during optimization. Correlations across runs between the achieved '
    'novelty of the generated set and its distance to the training compounds, disagreement, '
    'and predicted potency, for two search procedures at two uncertainty penalties. The '
    'relationship holds with the penalty set to zero.',
    r'Optimizer & $\lambda$ & $\rho(\nu,d)$ & $\rho(\nu,\sigma_T)$ & $\rho(\nu,\hat y)$ & $\rho(d,\sigma_T)$ & $n$',
    rows, 'llrrrrr')

# ---------------------------------------------------------------- S10 method per-cell
rows = []
for r in meth:
    a = r['agg']; pv = a['p_vs_ga']
    ps = ('%.1e' % pv).replace('e-0', r'\times10^{-').replace('e-', r'\times10^{-') + '}'
    rows.append(f"{LAB[r['target']]} & {r['k']} & {f(a['stga_ecfp'],3)} & {f(a['randtriage'],3)} & "
                f"{f(a['graphga'],3)} & {sg(a['d_vs_ga'])} & ${ps}$ & {f(a['frac_pos_ga'],2)}")
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
    'Fingerprint (primary) & ' + ' & '.join(f(tgt[t], 3) for t in T) +
    f" & ${tl:+.3f}$ [{ci[0]:+.3f}, {ci[1]:+.3f}] & {f(pv,3)}",
    'Dual-encoder latent & ' + ' & '.join(f(h['target_means'][t], 3) for t in T) +
    f" & ${h['mean']:+.3f}$ [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}] & {f(h['p'],3)}",
]
tab('tab:s-targetlevel',
    'Target-level summary of the method effect. Per-target mean paired difference against '
    'Graph GA, and the two-sided one-sample $t$-test on the five per-target means '
    '($\\mathrm{df}=4$). With five targets an assumption-free sign test cannot fall below '
    '$P=0.0625$ even when every target is positive, so the effect is best described as '
    'positive in all five targets studied.',
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
    'for the acquisition $\\mu+\\beta\\varsigma$. The paired difference between $\\beta=1$ and '
    '$\\beta=0$ is negligible, reaches $P<0.05$ on one target only, and is negative on DRD3.',
    r'Target & $\beta=0$ & $\beta=0.5$ & $\beta=1$ & $\beta=2$ & Graph GA & $\Delta_{\beta=1-0}$ & $P$',
    rows, 'lrrrrrrr')

rows = []
for t in T:
    e = ecfp[t]
    rows.append(f"{LAB[t]} & {f(e['latent'],3)} & {f(e['ecfp'],3)} & {f(e['graphga'],3)} & "
                f"{sg(e['latent_minus_ecfp'])} & {('%.1e' % e['p_lat_ecfp'])}")
tab('tab:s-latent',
    'A learned latent representation underperforms fingerprints. Mean top-ten reward at $k=10$ '
    'over 25 seeds, with the two-sided paired $P$ for the latent surrogate minus the '
    'fingerprint surrogate.',
    r'Target & Latent & Fingerprint & Graph GA & Latent $-$ Fingerprint & $P$', rows, 'lrrrrr')

rows = []
for t in T:
    d = rec[t]
    rows.append(f"{LAB[t]} & {f(d['retrained_pred'])} & {f(d['measured'])} & {f(d['retrained_sigma'])} & "
                f"{f(d['rec_stga'])} & {f(d['rec_graphga'])} & {f(d['null_sim'])}")
tab('tab:s-recovery',
    'Similarity to withheld scaffold clusters. An entire Bemis--Murcko cluster of actives is '
    'removed from both the seeds and the training set and the model retrained. The retrained '
    'model still ranks the withheld actives highly on four of five targets, SCD-1 being '
    'underpredicted, and flags them with elevated disagreement. Recovery is the mean closest '
    'Tanimoto similarity of any generated molecule to the withheld cluster, and is not improved '
    'by the triage.',
    r'Target & Predicted & Measured & $\sigma_T$ & ST-GA & Graph GA & Null', rows, 'lrrrrrr')

# ---------------------------------------------------------------- S13 in-domain novelty
rows = []
for t in T:
    r = rel[t]
    rows.append(f"{LAB[t]} & {f(r['novel_in_domain_rmse'])} & {f(r['novel_out_domain_rmse'])} & "
                f"{f(r['novel_in_domain_sigma'])} & {f(r['novel_out_domain_sigma'])} & "
                f"{r['n_novel_in_domain']:,}".replace(',', '{,}'))
tab('tab:s-indomain',
    'Nearer- and farther-training compounds at high novelty. Within the most novel third of '
    'each target, compounds are split at the median distance to the training set. Those nearer '
    'the training compounds have lower error and lower disagreement on four of the five '
    'targets. The split is a relative grouping within that third, not a validated domain '
    'boundary.',
    r'Target & \multicolumn{2}{c}{RMSE} & \multicolumn{2}{c}{$\sigma_T$} & $n$\\'
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}'
    r' & Nearer & Farther & Nearer & Farther & ',
    rows, 'lrrrrr')

# ---------------------------------------------------------------- S14 curation provenance
if prov:
    rows = []
    for t, v in prov.items():
        d = v['dropped']
        rows.append(f"{LAB.get(t,t)} & {v['chembl_id']} & {v['n_raw_records']:,} & "
                    f"{d.get('standard_type_not_affinity',0):,} & {d.get('non_human',0):,} & "
                    f"{d.get('unparsable_structure',0)} & {v['n_unique_parents']:,} & "
                    f"{v['year_range'][0]}--{v['year_range'][1]}".replace(',', '{,}'))
    tab('tab:s-curation',
        'ChEMBL curation flow for the re-curated targets. Activities were retained with an '
        'exact relation, nanomolar units, a human target and an affinity endpoint, then grouped '
        'on the standardized parent InChIKey. Counts removed at each stage are given so the '
        'flow can be reproduced. These files carry publication years and support the temporal '
        'analysis; FADS is not included because ChEMBL holds only two activity records for it.',
        r'Target & ChEMBL ID & Raw & Non-affinity & Non-human & Unparsable & Parents & Years',
        rows, 'llrrrrrr')

with open(OUT, 'w') as fh:
    fh.write('% Generated by make_si_tables.py from the frozen outputs - DO NOT EDIT BY HAND.\n')
    fh.write('\n'.join(OUTL))
print(f'wrote {OUT} with {sum(1 for l in OUTL if l.startswith(chr(92)+"begin{table}"))} tables')
