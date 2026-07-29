"""Three-way comparison: Chemistry World Model vs Graph GA vs random ablation.

All three optimize the SAME reward from the SAME k seed actives under the SAME
oracle budget (paired by seed index). CWM and random come from the few-shot
campaign; Graph GA from run_baselines_v1. Reports paired final-reward (top10)
comparisons and each method's novelty, establishing whether the learned world
model beats a strong standard baseline and whether all methods sit in the same
(low-novelty, high-reliability) region of the novelty-reliability frontier.

Writes outputs/cwm_v1/threeway_analysis.json.

    python -m world_model.analyze_threeway
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWM = os.path.join(BASE, 'outputs', 'cwm_v1')
ORDER = {'scd1': 0, 'fads': 1, 'nk1r': 2, 'drd2': 3, 'drd3': 4}


def _load(fn):
    return json.load(open(os.path.join(CWM, fn)))['results']


def main():
    fs = _load('fewshot_v1_results.json') + _load('fewshot_v1_extra.json')
    ga = {(r['target'], r['k']): r for r in _load('graphga_results.json')}
    fs = sorted(fs, key=lambda r: (ORDER[r['target']], r['k']))

    rows = []
    pool = {'cwm': [], 'ga': [], 'rnd': [], 'nov_cwm': [], 'nov_ga': [], 'nov_rnd': [], 'unc_ga': []}
    for r in fs:
        key = (r['target'], r['k'])
        if key not in ga:
            continue
        cs = r['per_seed']; gs = ga[key]['per_seed']
        n = min(len(cs), len(gs))
        cwm = np.array([cs[i]['wm']['top10'] for i in range(n)])
        rnd = np.array([cs[i]['random']['top10'] for i in range(n)])
        gat = np.array([gs[i]['top10'] for i in range(n)])
        pool['cwm'] += list(cwm); pool['ga'] += list(gat); pool['rnd'] += list(rnd)
        pool['nov_cwm'] += [cs[i]['wm']['novelty_frac'] for i in range(n)]
        pool['nov_ga'] += [gs[i]['novelty_frac'] for i in range(n)]
        pool['nov_rnd'] += [cs[i]['random']['novelty_frac'] for i in range(n)]
        pool['unc_ga'] += [gs[i]['uncertainty'] for i in range(n)]

        def pt(a, b):
            return float(stats.ttest_rel(a, b).pvalue)
        rows.append(dict(target=r['target'], k=r['k'], n=n,
                         cwm=float(cwm.mean()), ga=float(gat.mean()), rnd=float(rnd.mean()),
                         p_cwm_ga=pt(cwm, gat), p_cwm_rnd=pt(cwm, rnd), p_ga_rnd=pt(gat, rnd)))

    cwm = np.array(pool['cwm']); ga = np.array(pool['ga']); rnd = np.array(pool['rnd'])
    pooled = dict(
        n=len(cwm),
        mean_cwm=float(cwm.mean()), mean_ga=float(ga.mean()), mean_rnd=float(rnd.mean()),
        cwm_minus_ga=float((cwm - ga).mean()), p_cwm_ga=float(stats.ttest_rel(cwm, ga).pvalue),
        cwm_win_vs_ga=float(np.mean(cwm > ga)),
        cwm_minus_rnd=float((cwm - rnd).mean()), p_cwm_rnd=float(stats.ttest_rel(cwm, rnd).pvalue),
        ga_minus_rnd=float((ga - rnd).mean()), p_ga_rnd=float(stats.ttest_rel(ga, rnd).pvalue),
        novelty_cwm=float(np.mean(pool['nov_cwm'])), novelty_ga=float(np.mean(pool['nov_ga'])),
        novelty_rnd=float(np.mean(pool['nov_rnd'])), uncertainty_ga=float(np.mean(pool['unc_ga'])))

    with open(os.path.join(CWM, 'threeway_analysis.json'), 'w') as f:
        json.dump(dict(pooled=pooled, per_cell=rows), f, indent=2)

    print(f"{'target':6}{'k':>3}{'CWM':>8}{'GraphGA':>9}{'random':>8}  {'p(CWM-GA)':>10}")
    for r in rows:
        mark = '' if r['p_cwm_ga'] >= 0.05 else ('CWM>' if r['cwm'] > r['ga'] else 'GA>')
        print(f"{r['target']:6}{r['k']:>3}{r['cwm']:>8.3f}{r['ga']:>9.3f}{r['rnd']:>8.3f}  "
              f"{r['p_cwm_ga']:>10.4f} {mark}")
    print(f"\nPOOLED n={pooled['n']}:")
    print(f"  mean reward: CWM {pooled['mean_cwm']:.3f} | GraphGA {pooled['mean_ga']:.3f} | "
          f"random {pooled['mean_rnd']:.3f}")
    print(f"  CWM - GraphGA = {pooled['cwm_minus_ga']:+.4f} (p={pooled['p_cwm_ga']:.2e}, "
          f"CWM wins {pooled['cwm_win_vs_ga']*100:.0f}%)")
    print(f"  CWM - random  = {pooled['cwm_minus_rnd']:+.4f} (p={pooled['p_cwm_rnd']:.2e})")
    print(f"  GraphGA - random = {pooled['ga_minus_rnd']:+.4f} (p={pooled['p_ga_rnd']:.2e})")
    print(f"  novelty: CWM {pooled['novelty_cwm']:.2f} | GraphGA {pooled['novelty_ga']:.2f} | "
          f"random {pooled['novelty_rnd']:.2f}  (all low -> same frontier region; "
          f"GraphGA out-unc {pooled['uncertainty_ga']:.3f})")


if __name__ == '__main__':
    main()
