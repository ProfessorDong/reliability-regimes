"""Decompose the ST-GA vs vanilla Graph GA reward gain into chemical terms
(reviewer/Oz point 4: "what does +0.024 reward units mean chemically?").

Reproduces the head-to-head at k=10 with the frozen campaign configuration, then
re-scores the top-10 molecules of each method into reward COMPONENTS
(predicted pIC50, QED, synthetic accessibility, oracle uncertainty) so the reward
gain Delta can be attributed:
    Delta R = w_p*Delta(pIC50_norm) + w_s*Delta(sel) + w_q*Delta(QED)
              + w_a*Delta(SA_norm) - lambda*Delta(sigma).
Also reports the raw pIC50 difference and the fraction of top-10 molecules above a
high-potency threshold for each method. Writes decomposition_analysis.json.

    python -m world_model.run_decomposition_v1 --seeds 15
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.oracle import load_target_data  # noqa: E402
from world_model.reward import CWMReward  # noqa: E402
from world_model.dynamics import LatentEmbedder, LatentWorldModel  # noqa: E402
from world_model.wm_guided_ga import WMGuidedGA  # noqa: E402
from world_model.graph_ga import GraphGA  # noqa: E402
from world_model.run_fewshot_v1 import REWARD_CFG, DEFAULT_CFG, OUT_DIR  # noqa: E402

BUDGET = 300
LAMBDA_UNC = 0.1
K = 10
COMPS = ['pic50', 'pic50_norm', 'qed', 'sa_norm', 'uncertainty', 'selectivity']


def score_top(scorer, smiles):
    """Mean reward components over a list of (top) molecules."""
    c = scorer.components(smiles)
    out = {k: float(np.nanmean(c[k])) for k in COMPS}
    out['hi_potency_frac'] = float(np.mean(np.nan_to_num(c['pic50']) >= 8.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['scd1', 'fads', 'nk1r', 'drd2', 'drd3'])
    ap.add_argument('--seeds', type=int, default=15)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'decomposition_analysis.json'))
    args = ap.parse_args()
    embedder = LatentEmbedder(device='cuda')
    per_target = {}
    diffs = {k: [] for k in COMPS + ['reward', 'hi_potency_frac']}
    for target in args.targets:
        smiles, acts, thr = load_target_data(target)
        actives = [s for s, a in zip(smiles, acts) if a >= thr]
        cfg = REWARD_CFG.get(target, DEFAULT_CFG)
        w = dict({'potency': 1.0, 'selectivity': 0.0, 'qed': 0.3, 'sa': 0.3}, **cfg['weights'])
        # one scorer per target reused to re-score both methods' molecules
        scorer = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'])
        tdiff = {k: [] for k in COMPS + ['reward', 'hi_potency_frac']}
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 * seed + K)
            seeds = [actives[i] for i in rng.choice(len(actives), min(K, len(actives)), replace=False)]
            rw_wm = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'])
            wm = LatentWorldModel(embedder, n_ensemble=5, device='cuda', seed=seed)
            top_wm = WMGuidedGA(target, rw_wm, world_model=wm, mode='wm', seed=seed
                                ).run(seeds, budget=BUDGET)['top_smiles'][:10]
            rw_va = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'])
            top_va = GraphGA(target, rw_va, seed=seed).run(seeds, budget=BUDGET)['top_smiles'][:10]
            s_wm = score_top(scorer, top_wm); s_va = score_top(scorer, top_va)
            # reconstruct reward from components with the campaign weights
            def rew(s):
                return (w['potency'] * s['pic50_norm'] + w.get('selectivity', 0.0) * s['selectivity']
                        + w['qed'] * s['qed'] + w['sa'] * s['sa_norm'] - LAMBDA_UNC * s['uncertainty'])
            s_wm['reward'] = rew(s_wm); s_va['reward'] = rew(s_va)
            for k in COMPS + ['reward', 'hi_potency_frac']:
                tdiff[k].append(s_wm[k] - s_va[k]); diffs[k].append(s_wm[k] - s_va[k])
        per_target[target] = {k: float(np.mean(v)) for k, v in tdiff.items()}
        per_target[target]['weights'] = w
        print(f"== {target}: dReward={per_target[target]['reward']:+.3f} "
              f"dpIC50={per_target[target]['pic50']:+.2f} dQED={per_target[target]['qed']:+.3f} "
              f"dSA={per_target[target]['sa_norm']:+.3f} dUnc={per_target[target]['uncertainty']:+.3f} "
              f"dHiPot={per_target[target]['hi_potency_frac']:+.2f}", flush=True)
    pooled = {k: float(np.mean(v)) for k, v in diffs.items()}
    # weighted-term contributions to the pooled reward gain (potency uses w_p=1)
    pooled_terms = dict(
        potency=float(np.mean(diffs['pic50_norm'])) * 1.0,
        qed=float(np.mean(diffs['qed'])) * 0.3,
        sa=float(np.mean(diffs['sa_norm'])) * 0.3,
        uncertainty=-LAMBDA_UNC * float(np.mean(diffs['uncertainty'])),
    )
    out = dict(config=vars(args), per_target=per_target, pooled_diff=pooled,
               pooled_term_contributions=pooled_terms)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print("POOLED dReward", round(pooled['reward'], 3),
          "| dpIC50", round(pooled['pic50'], 2),
          "| term contributions", {k: round(v, 4) for k, v in pooled_terms.items()})
    print('wrote', args.out)


if __name__ == '__main__':
    main()
