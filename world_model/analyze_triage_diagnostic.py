"""Pool the triage-quality diagnostic into summary statistics.

Reports, pooled and per target, how well the surrogate ranks a fully evaluated
offspring pool: Spearman of acquisition (mu+beta*sigma), of mu alone, and of
sigma alone against true reward, plus enrichment of the true top-40 by each
policy and by random selection. Writes triage_diagnostic_analysis.json.

    python -m world_model.analyze_triage_diagnostic
"""
from __future__ import annotations
import json, os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CWM = os.path.join(BASE, 'outputs', 'cwm_v1')
KEYS = ['rho_acq', 'rho_mean', 'rho_std', 'enr_acq', 'enr_mean', 'enr_std', 'enr_rand']


def main():
    d = json.load(open(os.path.join(CWM, 'triage_diagnostic.json')))['results']
    per_target, pool = {}, {k: [] for k in KEYS}
    for t, rows in d.items():
        if not rows:
            continue
        per_target[t] = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
        for k in KEYS:
            pool[k] += [r[k] for r in rows]
    summary = {k: float(np.mean(v)) for k, v in pool.items()}
    summary['n_runs'] = len(pool['rho_acq'])
    out = {'summary': summary, 'per_target': per_target}
    json.dump(out, open(os.path.join(CWM, 'triage_diagnostic_analysis.json'), 'w'), indent=2)
    s = summary
    print(f"pooled (n={s['n_runs']}): Spearman acq={s['rho_acq']:.3f} mean={s['rho_mean']:.3f} "
          f"std={s['rho_std']:.3f}")
    print(f"enrichment@40: acq={s['enr_acq']:.3f} mean={s['enr_mean']:.3f} "
          f"std={s['enr_std']:.3f} random={s['enr_rand']:.3f}")
    print('per-target rho_acq:', {t: round(v['rho_acq'], 2) for t, v in per_target.items()})


if __name__ == '__main__':
    main()
