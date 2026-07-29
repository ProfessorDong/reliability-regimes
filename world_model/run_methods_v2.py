"""Definitive method comparison with the ECFP-RF surrogate (the reshaped primary method).

The ECFP surrogate beat the dual-encoder latent surrogate on all targets, so the primary
method becomes ST-GA with an online ECFP random-forest surrogate (mean-surrogate triage,
beta=0). Here we re-run the full method comparison across k in {5,10,20}, 25 seeds, five
targets, on the LEAKAGE-CLEAN metric (top-10 excluding seeds and exact training
compounds):
  * ST-GA-ECFP  : ECFP-RF surrogate triage over the Graph GA offspring pool
  * random-triage: same pool, random selection (attribution control)
  * Graph GA     : vanilla, evaluates its offspring directly
Fully CPU (no dual encoder). Writes methods_v2_results.json.

    python -m world_model.run_methods_v2 --seeds 25
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.oracle import load_target_data  # noqa: E402
from world_model.reward import CWMReward  # noqa: E402
from world_model.wm_guided_ga import WMGuidedGA  # noqa: E402
from world_model.graph_ga import GraphGA, _csmi  # noqa: E402
from world_model.run_ecfp_baseline_v1 import ECFPSurrogate, top10_novel  # noqa: E402
from world_model.run_fewshot_v1 import REWARD_CFG, DEFAULT_CFG, OUT_DIR  # noqa: E402

BUDGET = 300
LAMBDA_UNC = 0.1


def run_cell(target, k, seed, actives, train_canon, cfg):
    rng = np.random.default_rng(1000 * seed + k)
    seeds = [actives[i] for i in rng.choice(len(actives), min(k, len(actives)), replace=False)]
    sc = set(c for c in (_csmi(s) for s in seeds) if c)
    out = {}
    # ST-GA with ECFP-RF surrogate (mean triage)
    rw = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'])
    g = WMGuidedGA(target, rw, world_model=ECFPSurrogate(seed=seed), mode='wm', beta=0.0, seed=seed)
    g.run(seeds, BUDGET); out['stga_ecfp'] = top10_novel(g, sc, train_canon)
    # random triage (same pool, random selection; no surrogate)
    rw = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'])
    g = WMGuidedGA(target, rw, world_model=None, mode='randtriage', seed=seed)
    g.run(seeds, BUDGET); out['randtriage'] = top10_novel(g, sc, train_canon)
    # vanilla Graph GA
    rw = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'])
    g = GraphGA(target, rw, seed=seed); g.run(seeds, BUDGET); out['graphga'] = top10_novel(g, sc, train_canon)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['scd1', 'fads', 'nk1r', 'drd2', 'drd3'])
    ap.add_argument('--ks', nargs='+', type=int, default=[5, 10, 20])
    ap.add_argument('--seeds', type=int, default=25)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'methods_v2_results.json'))
    args = ap.parse_args()
    from scipy import stats
    results = []
    for target in args.targets:
        smiles, acts, thr = load_target_data(target)
        actives = [s for s, a in zip(smiles, acts) if a >= thr]
        train_canon = set(c for c in (_csmi(s) for s in smiles) if c)
        cfg = REWARD_CFG.get(target, DEFAULT_CFG)
        for k in args.ks:
            cells = [run_cell(target, k, s, actives, train_canon, cfg) for s in range(args.seeds)]
            st = np.array([c['stga_ecfp'] for c in cells]); ga = np.array([c['graphga'] for c in cells])
            rt = np.array([c['randtriage'] for c in cells])
            agg = dict(stga_ecfp=float(st.mean()), randtriage=float(rt.mean()), graphga=float(ga.mean()),
                       d_vs_ga=float((st - ga).mean()), p_vs_ga=float(stats.ttest_rel(st, ga)[1]),
                       d_vs_rt=float((st - rt).mean()), p_vs_rt=float(stats.ttest_rel(st, rt)[1]),
                       frac_pos_ga=float(np.mean(st > ga)))
            results.append(dict(target=target, k=k, n_seeds=args.seeds, agg=agg, per_seed=cells))
            print(f"== {target} k={k}: stga={agg['stga_ecfp']:.3f} rt={agg['randtriage']:.3f} "
                  f"GA={agg['graphga']:.3f} | d_vs_GA={agg['d_vs_ga']:+.3f} (p={agg['p_vs_ga']:.3f}) "
                  f"d_vs_rt={agg['d_vs_rt']:+.3f}", flush=True)
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump({'config': vars(args), 'results': results}, f, indent=2)
    # target-level hierarchical summary (n=5) for ST-GA-ECFP vs Graph GA
    tgt = {}
    for target in args.targets:
        d = np.concatenate([np.array([c['stga_ecfp'] - c['graphga'] for c in r['per_seed']])
                            for r in results if r['target'] == target])
        tgt[target] = float(d.mean())
    tm = np.array([tgt[t] for t in args.targets])
    t_stat, p = stats.ttest_1samp(tm, 0.0)
    se = tm.std(ddof=1) / np.sqrt(len(tm)); ci = stats.t.interval(0.95, len(tm) - 1, loc=tm.mean(), scale=se)
    print(f"TARGET-LEVEL ST-GA-ECFP vs GraphGA (n=5): mean={tm.mean():+.4f} "
          f"95% CI [{ci[0]:+.4f},{ci[1]:+.4f}] t={t_stat:.2f} p={p:.3f}")
    print(f"  per-target: " + ' '.join(f"{t}={tgt[t]:+.3f}" for t in args.targets))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
