"""Cluster-bootstrap confidence interval for the WM-GA gain.

Runs are nested in target-by-support-size cells, so pooled p-values overstate
independence. This computes a cluster bootstrap over the 15 cells: resample cells
with replacement, recompute the mean paired (WM-GA minus vanilla Graph GA)
reward, and report the 95 percent interval. Also reports how many cells are
positive and how many are individually significant. Writes wmga_bootstrap.json.

    python -m world_model.analyze_wmga_bootstrap
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWM = os.path.join(BASE, 'outputs', 'cwm_v1')
OUT = os.path.join(CWM, 'wmga_bootstrap.json')


def main():
    wmga = {(r['target'], r['k']): r for r in json.load(open(os.path.join(CWM, 'wmga_results.json')))['results']}
    ga = {(r['target'], r['k']): r for r in json.load(open(os.path.join(CWM, 'graphga_results.json')))['results']}
    cell_diffs = []   # list of per-cell arrays of per-seed paired diffs
    n_pos = n_sig = 0
    for key in wmga:
        ws, gs = wmga[key]['per_seed'], ga[key]['per_seed']
        n = min(len(ws), len(gs))
        d = np.array([ws[i]['wm']['top10'] - gs[i]['top10'] for i in range(n)])
        cell_diffs.append(d)
        if d.mean() > 0:
            n_pos += 1
        if stats.ttest_rel([ws[i]['wm']['top10'] for i in range(n)],
                           [gs[i]['top10'] for i in range(n)]).pvalue < 0.05:
            n_sig += 1
    all_diffs = np.concatenate(cell_diffs)
    point = float(all_diffs.mean())
    cell_means = np.array([d.mean() for d in cell_diffs])
    rng = np.random.default_rng(0)
    B, ncell = 20000, len(cell_diffs)
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, ncell, ncell)
        boot[b] = np.concatenate([cell_diffs[i] for i in idx]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out = dict(point_mean=point, cell_median=float(np.median(cell_means)),
               ci95_lo=float(lo), ci95_hi=float(hi),
               n_cells=ncell, n_positive=n_pos, n_significant=n_sig, n_boot=B)
    json.dump(out, open(OUT, 'w'), indent=2)
    print(f"WM-GA - vanilla: mean {point:+.4f}, cell-median {out['cell_median']:+.4f}, "
          f"cluster-bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"cells positive: {n_pos}/{ncell}; individually significant: {n_sig}/{ncell}")
    print('wrote', OUT)


if __name__ == '__main__':
    main()
