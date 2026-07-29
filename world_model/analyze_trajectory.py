"""Sample-efficiency analysis (Section V-C, Fig. budget).

Loads trajectory_results.json and computes, per target, the area under the
best-reward-versus-oracle-call curve for WM-GA, vanilla Graph GA, and the
random-triage control, with paired t-tests. Writes trajectory_analysis.json.

    python -m world_model.analyze_trajectory
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, 'outputs', 'cwm_v1', 'trajectory_results.json')
OUT = os.path.join(BASE, 'outputs', 'cwm_v1', 'trajectory_analysis.json')


def auc(traj, B):
    c = np.array([x[0] for x in traj], float); b = np.array([x[1] for x in traj], float)
    c = np.concatenate([c, [B]]); b = np.concatenate([b, [b[-1]]])
    return float(np.trapz(b, c) / B)


def main():
    d = json.load(open(SRC)); B = d['budget']; res = d['results']
    out = {}
    for tgt, methods in res.items():
        wm = np.array([auc(t, B) for t in methods['wmga']])
        va = np.array([auc(t, B) for t in methods['vanilla']])
        rt = np.array([auc(t, B) for t in methods['randtriage']])
        out[tgt] = dict(
            auc_wmga=float(wm.mean()), auc_vanilla=float(va.mean()), auc_randtriage=float(rt.mean()),
            wmga_minus_vanilla=float((wm - va).mean()), p_wmga_vanilla=float(stats.ttest_rel(wm, va).pvalue),
            wmga_minus_randtriage=float((wm - rt).mean()), p_wmga_randtriage=float(stats.ttest_rel(wm, rt).pvalue),
            n_seeds=len(wm))
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    for tgt, r in out.items():
        print(f"{tgt}: AUC WM-GA={r['auc_wmga']:.4f} vanilla={r['auc_vanilla']:.4f} "
              f"randtri={r['auc_randtriage']:.4f} | WM-GA-vanilla={r['wmga_minus_vanilla']:+.4f} "
              f"(p={r['p_wmga_vanilla']:.2e})")
    print('wrote', OUT)


if __name__ == '__main__':
    main()
