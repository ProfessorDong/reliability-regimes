"""Collect best-reward-vs-oracle-call trajectories for the sample-efficiency figure.

For two representative targets we run WM-GA, vanilla Graph GA, and GA random-triage
on the same support draws and record the (oracle_calls, best_reward) trajectory of
each run. The three are all genetic-algorithm based and differ only in how the
offspring pool is triaged, so the curves isolate the world model's contribution.

    python -m world_model.run_trajectory_v1 --targets drd2 scd1 --seeds 25
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
K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['drd2', 'scd1'])
    ap.add_argument('--seeds', type=int, default=25)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'trajectory_results.json'))
    args = ap.parse_args()
    embedder = LatentEmbedder(device='cuda')

    results = {}
    for target in args.targets:
        smiles, acts, thr = load_target_data(target)
        actives = [s for s, a in zip(smiles, acts) if a >= thr]
        cfg = REWARD_CFG.get(target, DEFAULT_CFG)
        results[target] = {'wmga': [], 'vanilla': [], 'randtriage': []}
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 * s + K)
            seeds = [actives[i] for i in rng.choice(len(actives), K, replace=False)]

            rw = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=0.1, weights=cfg['weights'])
            wm = LatentWorldModel(embedder, n_ensemble=5, device='cuda', seed=s)
            tr = WMGuidedGA(target, rw, world_model=wm, mode='wm', seed=s).run(seeds, BUDGET)['trajectory']
            results[target]['wmga'].append(tr)

            rw = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=0.1, weights=cfg['weights'])
            tr = GraphGA(target, rw, seed=s).run(seeds, BUDGET)['trajectory']
            results[target]['vanilla'].append(tr)

            rw = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=0.1, weights=cfg['weights'])
            tr = WMGuidedGA(target, rw, world_model=None, mode='randtriage', seed=s).run(seeds, BUDGET)['trajectory']
            results[target]['randtriage'].append(tr)
            print(f"  [{target} seed {s+1}/{args.seeds}] done", flush=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'budget': BUDGET, 'k': K, 'results': results}, f)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
