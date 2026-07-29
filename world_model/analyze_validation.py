"""Independent-oracle validation analysis.

Tests whether WM-GA's advantage over vanilla Graph GA, measured on the RF-on-ECFP
search oracle, survives re-scoring with an independent oracle (gradient boosting
on RDKit descriptors) that the search never optimized. Reports, per target and
pooled, the paired WM-GA minus vanilla difference under (a) the search oracle and
(b) the independent oracle. Writes validation_analysis.json.

    python -m world_model.analyze_validation
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, 'outputs', 'cwm_v1', 'validation_results.json')
OUT = os.path.join(BASE, 'outputs', 'cwm_v1', 'validation_analysis.json')


def main():
    d = json.load(open(SRC)); res = d['results']
    out = {}
    pooled = {'rf': {'wm': [], 'va': []}, 'ind': {'wm': [], 'va': []}}
    for tgt, blk in res.items():
        wm, va = blk['rows']['wmga'], blk['rows']['vanilla']
        n = min(len(wm), len(va))
        rf_wm = np.array([wm[i]['rf_reward'] for i in range(n)])
        rf_va = np.array([va[i]['rf_reward'] for i in range(n)])
        ind_wm = np.array([wm[i]['indep_potency'] for i in range(n)])
        ind_va = np.array([va[i]['indep_potency'] for i in range(n)])
        out[tgt] = dict(
            n=n,
            rf_delta=float((rf_wm - rf_va).mean()), rf_p=float(stats.ttest_rel(rf_wm, rf_va).pvalue),
            indep_delta=float((ind_wm - ind_va).mean()), indep_p=float(stats.ttest_rel(ind_wm, ind_va).pvalue),
            indep_win=float((ind_wm > ind_va).mean()))
        pooled['rf']['wm'] += list(rf_wm); pooled['rf']['va'] += list(rf_va)
        pooled['ind']['wm'] += list(ind_wm); pooled['ind']['va'] += list(ind_va)
        print(f"{tgt} (n={n}): RF Delta={out[tgt]['rf_delta']:+.3f} (p={out[tgt]['rf_p']:.2e}) | "
              f"INDEP Delta={out[tgt]['indep_delta']:+.3f} (p={out[tgt]['indep_p']:.2e}, "
              f"win {out[tgt]['indep_win']*100:.0f}%)")
    for key, label in [('rf', 'search oracle'), ('ind', 'independent oracle')]:
        w = np.array(pooled[key]['wm']); v = np.array(pooled[key]['va'])
        delta = float((w - v).mean()); p = float(stats.ttest_rel(w, v).pvalue); win = float((w > v).mean())
        out[f'pooled_{key}'] = dict(delta=delta, p=p, win=win, n=len(w))
        print(f"POOLED {label}: Delta={delta:+.4f} (p={p:.2e}, win {win*100:.0f}%, n={len(w)})")
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
