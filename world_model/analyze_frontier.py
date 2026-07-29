"""Analyze the novelty-reliability frontier (Section V-C headline).

Loads outputs/cwm_v1/frontier_v1_results.json (the 5-target x 25-seed x novelty
sweep) and quantifies the trade-off: pushing novelty lowers predicted potency and
raises the oracle's epistemic uncertainty, in every target. Oracle uncertainty is
the indicator of when novelty has left the reliable domain.

Writes outputs/cwm_v1/frontier_analysis.json (frozen numbers for the manuscript).

    python -m world_model.analyze_frontier
"""
from __future__ import annotations
import json, os
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, 'outputs', 'cwm_v1', 'frontier_v1_results.json')
OUT = os.path.join(BASE, 'outputs', 'cwm_v1', 'frontier_analysis.json')
ORDER = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']


def main():
    d = json.load(open(SRC))
    by = defaultdict(lambda: defaultdict(list))
    for r in d['results']:
        t = r['target']
        for c in r['per_seed']:
            by[t]['nov'].append(c['novelty_frac'])
            by[t]['pot'].append(c['potency'])
            by[t]['unc'].append(c['uncertainty'])

    per_target = {}
    allnov, allunc, allpot = [], [], []
    for t in ORDER:
        nov = np.array(by[t]['nov']); pot = np.array(by[t]['pot']); unc = np.array(by[t]['unc'])
        allnov += list(nov); allunc += list(unc); allpot += list(pot)
        lo, hi = nov < 0.33, nov > 0.66
        per_target[t] = dict(
            n=len(nov),
            rho_nov_unc=float(spearmanr(nov, unc).correlation),
            rho_nov_pot=float(spearmanr(nov, pot).correlation),
            unc_lowNov=float(unc[lo].mean()), unc_hiNov=float(unc[hi].mean()),
            pot_lowNov=float(pot[lo].mean()), pot_hiNov=float(pot[hi].mean()))

    allnov = np.array(allnov); allunc = np.array(allunc); allpot = np.array(allpot)
    lo, hi = allnov < 0.33, allnov > 0.66
    r_nu = spearmanr(allnov, allunc); r_np = spearmanr(allnov, allpot)
    pooled = dict(
        n=int(len(allnov)),
        rho_nov_unc=float(r_nu.correlation), p_nov_unc=float(r_nu.pvalue),
        rho_nov_pot=float(r_np.correlation), p_nov_pot=float(r_np.pvalue),
        unc_lowNov=float(allunc[lo].mean()), unc_hiNov=float(allunc[hi].mean()),
        unc_ratio=float(allunc[hi].mean() / allunc[lo].mean()),
        pot_lowNov=float(allpot[lo].mean()), pot_hiNov=float(allpot[hi].mean()),
        pot_drop=float(allpot[hi].mean() - allpot[lo].mean()))

    res = dict(pooled=pooled, per_target=per_target)
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=2)

    print(f"POOLED n={pooled['n']}: rho(nov,unc)={pooled['rho_nov_unc']:+.3f} "
          f"(p={pooled['p_nov_unc']:.1e})  rho(nov,pot)={pooled['rho_nov_pot']:+.3f} "
          f"(p={pooled['p_nov_pot']:.1e})")
    print(f"  uncertainty {pooled['unc_lowNov']:.3f} -> {pooled['unc_hiNov']:.3f} "
          f"({pooled['unc_ratio']:.2f}x); potency {pooled['pot_lowNov']:.3f} -> "
          f"{pooled['pot_hiNov']:.3f} ({pooled['pot_drop']:+.2f})")
    for t in ORDER:
        pt = per_target[t]
        print(f"  {t:5} rho(nov,unc)={pt['rho_nov_unc']:+.3f} rho(nov,pot)={pt['rho_nov_pot']:+.3f}")
    print('wrote', OUT)
    # the trade-off must hold (positive nov->unc, negative nov->pot) in every target
    ok = all(per_target[t]['rho_nov_unc'] > 0 and per_target[t]['rho_nov_pot'] < 0 for t in ORDER)
    print('FRONTIER UNIVERSAL across all targets:', ok)


if __name__ == '__main__':
    main()
