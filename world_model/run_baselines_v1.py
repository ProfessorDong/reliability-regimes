"""Graph GA baseline run, matched to the CWM/random few-shot campaign (Section V-B).

For each (target, k, seed) we draw the SAME k seed actives (same RNG) and run Graph
GA under the SAME reward config and oracle budget as run_fewshot_v1, recording the
same metrics (top-10 reward, novelty vs seeds, oracle uncertainty). Merged offline
with fewshot_v1_results.json + fewshot_v1_extra.json (which hold cwm and random) to
compare CWM vs Graph GA vs random on quality and on the novelty-reliability frontier.

    python -m world_model.run_baselines_v1 --targets scd1 fads nk1r drd2 drd3 --ks 5 10 20 --seeds 25
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.oracle import load_target_data, load_oracle, predict_pic50  # noqa: E402
from world_model.reward import CWMReward  # noqa: E402
from world_model.graph_ga import GraphGA  # noqa: E402
from world_model.run_fewshot_v1 import novelty, REWARD_CFG, DEFAULT_CFG, OUT_DIR  # noqa: E402

BUDGET = 300
LAMBDA_UNC = 0.1  # match the campaign hp


def run_cell(target, k, seed, oracle):
    smiles, acts, thr = load_target_data(target)
    actives = [s for s, a in zip(smiles, acts) if a >= thr]
    rng = np.random.default_rng(1000 * seed + k)          # SAME draw as run_fewshot_v1
    seeds = [actives[i] for i in rng.choice(len(actives), min(k, len(actives)), replace=False)]
    cfg = REWARD_CFG.get(target, DEFAULT_CFG)
    rw = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'])
    ga = GraphGA(target, rw, seed=seed)
    res = ga.run(seeds, budget=BUDGET)
    top = res['top_smiles']
    nov_frac, nov_sim = novelty(top, seeds)
    _, pstd = predict_pic50(oracle, top, return_std=True) if top else (None, np.array([np.nan]))
    return dict(top10=res['top10_mean'], best=res['best_reward'], n_unique=res['n_unique'],
                novelty_frac=float(nov_frac), uncertainty=float(np.mean(pstd)),
                calls=res['oracle_calls'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['scd1', 'fads', 'nk1r', 'drd2', 'drd3'])
    ap.add_argument('--ks', nargs='+', type=int, default=[5, 10, 20])
    ap.add_argument('--seeds', type=int, default=25)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'graphga_results.json'))
    args = ap.parse_args()

    keys = ['top10', 'best', 'novelty_frac', 'uncertainty', 'n_unique']
    results = []
    for target in args.targets:
        oracle = load_oracle(target)
        for k in args.ks:
            cells = [run_cell(target, k, s, oracle) for s in range(args.seeds)]
            agg = {key: [float(np.mean([c[key] for c in cells])),
                         float(np.std([c[key] for c in cells]))] for key in keys}
            results.append(dict(target=target, k=k, n_seeds=args.seeds, budget=BUDGET,
                                agg=agg, per_seed=cells))
            print(f"== graphga {target} k={k}: top10={agg['top10'][0]:.3f} "
                  f"novelty={agg['novelty_frac'][0]:.2f} unc={agg['uncertainty'][0]:.3f}", flush=True)
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump({'config': vars(args), 'results': results}, f, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
