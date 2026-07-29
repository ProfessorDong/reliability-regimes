"""Apply the pre-registered decision rule: does WM-guided GA beat vanilla Graph GA?

Merges all methods, paired by (target, k, seed):
  wm-ga, ga-randtriage           <- wmga_results.json
  vanilla graph-ga               <- graphga_results.json
  cwm, random                    <- fewshot_v1_results.json + fewshot_v1_extra.json

Decision rule (PREREGISTRATION_wmga.md): keep the world model as a headline method
ONLY IF pooled WM-GA > vanilla GraphGA (p<0.05) AND WM-GA >= GA-random-triage.

Writes outputs/cwm_v1/wmga_analysis.json.

    python -m world_model.analyze_wmga
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
    wmga = {(r['target'], r['k']): r for r in _load('wmga_results.json')}
    ga = {(r['target'], r['k']): r for r in _load('graphga_results.json')}
    fs = {(r['target'], r['k']): r for r in _load('fewshot_v1_results.json') + _load('fewshot_v1_extra.json')}

    keys = sorted(wmga, key=lambda kk: (ORDER[kk[0]], kk[1]))
    P = {m: [] for m in ['wmga', 'randtriage', 'vanilla', 'cwm', 'random']}
    rows = []
    for key in keys:
        ws = wmga[key]['per_seed']; gs = ga[key]['per_seed']; fsc = fs[key]['per_seed']
        n = min(len(ws), len(gs), len(fsc))
        wmga_v = np.array([ws[i]['wm']['top10'] for i in range(n)])
        rand_v = np.array([ws[i]['randtriage']['top10'] for i in range(n)])
        van_v = np.array([gs[i]['top10'] for i in range(n)])
        cwm_v = np.array([fsc[i]['wm']['top10'] for i in range(n)])
        rnd_v = np.array([fsc[i]['random']['top10'] for i in range(n)])
        P['wmga'] += list(wmga_v); P['randtriage'] += list(rand_v); P['vanilla'] += list(van_v)
        P['cwm'] += list(cwm_v); P['random'] += list(rnd_v)
        rows.append(dict(target=key[0], k=key[1], n=n,
                         wmga=float(wmga_v.mean()), vanilla=float(van_v.mean()),
                         randtriage=float(rand_v.mean()), cwm=float(cwm_v.mean()),
                         random=float(rnd_v.mean()),
                         p_wmga_vanilla=float(stats.ttest_rel(wmga_v, van_v).pvalue),
                         p_wmga_randtriage=float(stats.ttest_rel(wmga_v, rand_v).pvalue)))

    A = {m: np.array(v) for m, v in P.items()}
    def cmp(a, b):
        d = A[a] - A[b]
        return dict(delta=float(d.mean()), p=float(stats.ttest_rel(A[a], A[b]).pvalue),
                    win=float(np.mean(d > 0)))
    pooled = dict(n=len(A['wmga']),
                  means={m: float(A[m].mean()) for m in A},
                  wmga_vs_vanilla=cmp('wmga', 'vanilla'),
                  wmga_vs_randtriage=cmp('wmga', 'randtriage'),
                  wmga_vs_cwm=cmp('wmga', 'cwm'),
                  vanilla_vs_random=cmp('vanilla', 'random'))

    r1 = pooled['wmga_vs_vanilla']; r2 = pooled['wmga_vs_randtriage']
    decision = bool(r1['delta'] > 0 and r1['p'] < 0.05 and r2['delta'] >= 0)
    pooled['PREREGISTERED_PASS'] = decision

    with open(os.path.join(CWM, 'wmga_analysis.json'), 'w') as f:
        json.dump(dict(pooled=pooled, per_cell=rows), f, indent=2)

    print(f"{'target':6}{'k':>3}{'WM-GA':>8}{'vanilla':>9}{'randtri':>9}{'CWM':>8}{'rand':>8}"
          f"{'p(WMGA-van)':>12}")
    for r in rows:
        print(f"{r['target']:6}{r['k']:>3}{r['wmga']:>8.3f}{r['vanilla']:>9.3f}"
              f"{r['randtriage']:>9.3f}{r['cwm']:>8.3f}{r['random']:>8.3f}{r['p_wmga_vanilla']:>12.4f}")
    print(f"\nPOOLED n={pooled['n']} mean reward: " +
          ' | '.join(f"{m} {pooled['means'][m]:.3f}" for m in ['wmga', 'vanilla', 'randtriage', 'cwm', 'random']))
    for name in ['wmga_vs_vanilla', 'wmga_vs_randtriage', 'wmga_vs_cwm']:
        c = pooled[name]
        print(f"  {name}: Δ={c['delta']:+.4f}  p={c['p']:.2e}  win={c['win']*100:.0f}%")
    print(f"\nPRE-REGISTERED DECISION (WM-GA>vanilla p<.05 AND WM-GA>=randtriage): "
          f"{'PASS — keep world-model headline' if decision else 'FAIL — revert to empirical-study framing'}")


if __name__ == '__main__':
    main()
