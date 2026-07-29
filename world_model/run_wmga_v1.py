"""Run WM-guided GA and GA-random-triage, matched to the GraphGA/CWM campaign.

Same k-seed draw, reward config, and oracle budget as run_baselines_v1 / run_fewshot_v1,
so all methods are paired by (target, k, seed). Vanilla Graph GA already lives in
graphga_results.json. Merge offline for the pre-registered comparison.

    python -m world_model.run_wmga_v1 --targets scd1 fads nk1r drd2 drd3 --ks 5 10 20 --seeds 25
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.oracle import load_target_data, load_oracle, predict_pic50  # noqa: E402
from world_model.reward import CWMReward  # noqa: E402
from world_model.dynamics import LatentEmbedder, LatentWorldModel  # noqa: E402
from world_model.wm_guided_ga import WMGuidedGA  # noqa: E402
from world_model.run_fewshot_v1 import novelty, REWARD_CFG, DEFAULT_CFG, OUT_DIR  # noqa: E402

BUDGET = 300
LAMBDA_UNC = 0.1


def run_cell(target, k, seed, oracle, embedder, modes):
    smiles, acts, thr = load_target_data(target)
    actives = [s for s, a in zip(smiles, acts) if a >= thr]
    rng = np.random.default_rng(1000 * seed + k)        # SAME draw as the campaign
    seeds = [actives[i] for i in rng.choice(len(actives), min(k, len(actives)), replace=False)]
    cfg = REWARD_CFG.get(target, DEFAULT_CFG)
    out = {}
    for mode in modes:
        rw = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'])
        wm = LatentWorldModel(embedder, n_ensemble=5, device='cuda', seed=seed) if mode == 'wm' else None
        ga = WMGuidedGA(target, rw, world_model=wm, mode=mode, seed=seed)
        res = ga.run(seeds, budget=BUDGET)
        top = res['top_smiles']
        nov_frac, _ = novelty(top, seeds)
        _, pstd = predict_pic50(oracle, top, return_std=True) if top else (None, np.array([np.nan]))
        out[mode] = dict(top10=res['top10_mean'], best=res['best_reward'],
                         n_unique=res['n_unique'], novelty_frac=float(nov_frac),
                         uncertainty=float(np.mean(pstd)), calls=res['oracle_calls'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['scd1', 'fads', 'nk1r', 'drd2', 'drd3'])
    ap.add_argument('--ks', nargs='+', type=int, default=[5, 10, 20])
    ap.add_argument('--seeds', type=int, default=25)
    ap.add_argument('--modes', nargs='+', default=['wm', 'randtriage'])
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'wmga_results.json'))
    args = ap.parse_args()

    embedder = LatentEmbedder(device='cuda')
    keys = ['top10', 'best', 'novelty_frac', 'uncertainty', 'n_unique']
    results = []
    for target in args.targets:
        oracle = load_oracle(target)
        for k in args.ks:
            cells = [run_cell(target, k, s, oracle, embedder, args.modes) for s in range(args.seeds)]
            agg = {m: {key: [float(np.mean([c[m][key] for c in cells])),
                             float(np.std([c[m][key] for c in cells]))] for key in keys}
                   for m in args.modes}
            results.append(dict(target=target, k=k, n_seeds=args.seeds, budget=BUDGET,
                                agg=agg, per_seed=cells))
            line = ' '.join(f"{m}.top10={agg[m]['top10'][0]:.3f}" for m in args.modes)
            print(f"== {target} k={k}: {line}", flush=True)
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump({'config': vars(args), 'results': results}, f, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
