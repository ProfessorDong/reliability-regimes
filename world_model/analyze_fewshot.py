"""Analyze the world-model vs random-ablation few-shot campaign (Section V-B).

Combines the two result files (drd3+scd1 and fads+nk1r+drd2) into one all-5-target
PAIRED analysis (wm and random share the same k seed molecules per seed-index).
Reports two distinct claims that the data separates:
  * final molecule quality (top10 reward) at a fixed budget, and
  * search speed (trajectory AUC).

Writes outputs/cwm_v1/fewshot_analysis.json (frozen numbers for the manuscript).

    python -m world_model.analyze_fewshot
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWM = os.path.join(BASE, 'outputs', 'cwm_v1')
ORDER = {'scd1': 0, 'fads': 1, 'nk1r': 2, 'drd2': 3, 'drd3': 4}


def main():
    res = []
    for f in ['fewshot_v1_results.json', 'fewshot_v1_extra.json']:
        res += json.load(open(os.path.join(CWM, f)))['results']
    res = sorted(res, key=lambda r: (ORDER[r['target']], r['k']))

    rows = []
    all_d, all_auc = [], []
    for r in res:
        cs = r['per_seed']
        wm = np.array([c['wm']['top10'] for c in cs]); rd = np.array([c['random']['top10'] for c in cs])
        wa = np.array([c['wm']['auc'] for c in cs]); ra = np.array([c['random']['auc'] for c in cs])
        nov = np.array([c['wm']['novelty_frac'] for c in cs])
        diff = wm - rd; dauc = wa - ra
        all_d += list(diff); all_auc += list(dauc)
        _, p = stats.ttest_rel(wm, rd); _, pa = stats.ttest_rel(wa, ra)
        rows.append(dict(target=r['target'], k=r['k'], n_seeds=len(cs),
                         wm_top10=float(wm.mean()), random_top10=float(rd.mean()),
                         paired_delta=float(diff.mean()), win_rate=float(np.mean(diff > 0)),
                         p_value=float(p), dAUC=float(dauc.mean()), p_AUC=float(pa),
                         wm_novelty=float(nov.mean())))

    all_d = np.array(all_d); all_auc = np.array(all_auc)
    _, p = stats.ttest_1samp(all_d, 0); _, pa = stats.ttest_1samp(all_auc, 0)
    pooled = dict(n=int(len(all_d)),
                  final_reward_delta=float(all_d.mean()), win_rate=float(np.mean(all_d > 0)),
                  p_final=float(p), auc_delta=float(all_auc.mean()), p_auc=float(pa))

    out = dict(pooled=pooled, per_cell=rows)
    with open(os.path.join(CWM, 'fewshot_analysis.json'), 'w') as f:
        json.dump(out, f, indent=2)

    print(f"{'tgt':5}{'k':>3}{'pairedDelta':>13}{'win%':>6}{'p':>9}{'dAUC':>9}")
    for r in rows:
        s = '*' if r['p_value'] < 0.05 else ''
        print(f"{r['target']:5}{r['k']:>3}{r['paired_delta']:>+13.4f}{r['win_rate']*100:>5.0f}%"
              f"{r['p_value']:>9.4f}{s}{r['dAUC']:>+9.4f}")
    print(f"\nPOOLED n={pooled['n']}: final-reward delta={pooled['final_reward_delta']:+.4f} "
          f"(win {pooled['win_rate']*100:.0f}%, p={pooled['p_final']:.2e})  |  "
          f"AUC delta={pooled['auc_delta']:+.4f} (p={pooled['p_auc']:.3f})")
    n_sig = sum(r['p_value'] < 0.05 for r in rows)
    print(f"final-quality WM>random significant in {n_sig}/{len(rows)} cells; "
          f"AUC (speed): {'no gain' if pooled['auc_delta'] <= 0 else 'gain'}")


if __name__ == '__main__':
    main()
