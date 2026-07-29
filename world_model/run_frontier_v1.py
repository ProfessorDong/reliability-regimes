"""Novelty-reliability frontier sweep (Section V-C, conceptual centerpiece).

The reward's oracle-uncertainty penalty lambda_unc is swept. For each value we run
the few-shot search and characterize the TOP generated molecules by their
*unpenalized* properties: predicted potency (oracle mean), novelty vs the k seeds,
and oracle epistemic uncertainty (ensemble std). The hypothesis: as lambda_unc
rises, the search is pushed away from uncertain (OOD, novel) chemistry toward
confident (in-distribution, near-seed) chemistry, tracing a navigable
novelty <-> reliability frontier. Uncertainty is the control knob.

Run:
    python -m world_model.run_frontier_v1 --targets drd3 scd1 --seeds 25 --k 10
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.oracle import load_target_data, load_oracle, predict_pic50  # noqa: E402
from world_model.reward import CWMReward  # noqa: E402
from world_model.actions import ActionSpace, AdmissibilityFilter  # noqa: E402
from world_model.dynamics import LatentEmbedder, LatentWorldModel  # noqa: E402
from world_model.planner import Planner  # noqa: E402
from world_model.run_fewshot_v1 import (  # noqa: E402
    build_chembl_pool, _library_for, novelty, REWARD_CFG, DEFAULT_CFG, OUT_DIR)

# Sweep the novelty incentive w_novelty (the trade-off driver); hold lambda_unc fixed.
WNOV = [0.0, 0.25, 0.5, 1.0, 2.0]
LAMBDA_UNC = 0.1
HP = dict(adm_thr=0.2, expand_top=30, neighbors_per=40, eval_per_round=12, beta=1.5)


def run_cell(target, wnov, seed, k, budget, pool, embedder, oracle):
    smiles, acts, thr = load_target_data(target)
    actives = [s for s, a in zip(smiles, acts) if a >= thr]
    rng = np.random.default_rng(1000 * seed + k)
    seeds = [actives[i] for i in rng.choice(len(actives), min(k, len(actives)), replace=False)]
    lib = _library_for(seeds, pool)
    cfg = REWARD_CFG.get(target, DEFAULT_CFG)

    rw = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'],
                   novelty_anchors=seeds, w_novelty=wnov)
    wm = LatentWorldModel(embedder, n_ensemble=5, device='cuda', seed=seed)
    sp = ActionSpace(lib, max_neighbors=HP['neighbors_per'], seed=seed)
    adm = AdmissibilityFilter(seeds, threshold=HP['adm_thr'])
    pl = Planner(target, rw, sp, wm, adm, mode='wm', beta=HP['beta'],
                 expand_top=HP['expand_top'], neighbors_per=HP['neighbors_per'],
                 eval_per_round=HP['eval_per_round'], grow_anchors=True, seed=seed)
    res = pl.run(seeds, budget=budget, warmup=min(k, 20))

    top = res['top_smiles'][:20]
    if not top:
        return None
    pmean, pstd = predict_pic50(oracle, top, return_std=True)  # raw oracle, not reward
    nov_frac, nov_sim = novelty(top, seeds)
    return dict(potency=float(np.mean(pmean)), uncertainty=float(np.mean(pstd)),
                novelty_frac=float(nov_frac), sim_to_seed=float(nov_sim),
                oracle_calls=res['oracle_calls'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['drd3', 'scd1', 'fads', 'nk1r', 'drd2'])
    ap.add_argument('--wnov', nargs='+', type=float, default=WNOV)
    ap.add_argument('--seeds', type=int, default=25)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--budget', type=int, default=300)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'frontier_v1_results.json'))
    args = ap.parse_args()

    pool = build_chembl_pool()
    embedder = LatentEmbedder(device='cuda')
    keys = ['potency', 'uncertainty', 'novelty_frac', 'sim_to_seed']
    results = []
    for target in args.targets:
        oracle = load_oracle(target)
        for wnov in args.wnov:
            cells = []
            for s in range(args.seeds):
                c = run_cell(target, wnov, s, args.k, args.budget, pool, embedder, oracle)
                if c:
                    cells.append(c)
            agg = {key: [float(np.mean([c[key] for c in cells])),
                         float(np.std([c[key] for c in cells]))] for key in keys}
            results.append(dict(target=target, w_novelty=wnov, lambda_unc=LAMBDA_UNC, k=args.k,
                                n_seeds=len(cells), budget=args.budget, agg=agg, per_seed=cells))
            print(f"== {target} w_nov={wnov}: potency={agg['potency'][0]:.3f} "
                  f"uncertainty={agg['uncertainty'][0]:.3f} novelty={agg['novelty_frac'][0]:.2f} "
                  f"sim_to_seed={agg['sim_to_seed'][0]:.3f}", flush=True)
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump({'config': vars(args), 'hp': HP, 'results': results}, f, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
