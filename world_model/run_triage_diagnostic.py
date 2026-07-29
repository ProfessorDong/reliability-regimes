"""Triage-quality diagnostic (does the surrogate rank offspring by true reward?).

For each target we run WM-GA to a realistic mid-search state (warm-up budget),
then generate a fresh offspring pool, read the surrogate's predictions for the
whole pool, and FULLY evaluate the pool with the oracle. We then measure how well
the surrogate's acquisition a=mu+beta*sigma ranks the pool against true reward,
and the enrichment of the true top-40 captured by a, by mu only, by sigma only,
and by random selection. This isolates the triage mechanism that WM-GA relies on.

    python -m world_model.run_triage_diagnostic --seeds 8
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.oracle import load_target_data  # noqa: E402
from world_model.reward import CWMReward  # noqa: E402
from world_model.dynamics import LatentEmbedder, LatentWorldModel  # noqa: E402
from world_model.wm_guided_ga import WMGuidedGA  # noqa: E402
from world_model.run_fewshot_v1 import REWARD_CFG, DEFAULT_CFG, OUT_DIR  # noqa: E402

TARGETS = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']
K, WARMUP, POOL, TOPN = 10, 150, 200, 40


def enrichment(score, true_top, n):
    sel = set(np.argsort(-score)[:n].tolist())
    return len(sel & true_top) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'triage_diagnostic.json'))
    args = ap.parse_args()
    embedder = LatentEmbedder(device='cuda')
    rng0 = np.random.default_rng(0)
    results = {}
    for target in TARGETS:
        smiles, acts, thr = load_target_data(target)
        actives = [s for s, a in zip(smiles, acts) if a >= thr]
        cfg = REWARD_CFG.get(target, DEFAULT_CFG)
        rows = []
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 * s + K)
            seeds = [actives[i] for i in rng.choice(len(actives), K, replace=False)]
            rw = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=0.1, weights=cfg['weights'])
            wm = LatentWorldModel(embedder, n_ensemble=5, device='cuda', seed=s)
            ga = WMGuidedGA(target, rw, world_model=wm, mode='wm', seed=s)
            ga.run(seeds, budget=WARMUP)                      # warm the surrogate
            pop = sorted(ga.scores, key=ga.scores.get, reverse=True)[:ga.pop_size]
            fit = [ga.scores[p] for p in pop]
            pool = ga._make_pool(pop, fit, POOL)
            pool = list(dict.fromkeys(pool))                  # unique
            if len(pool) < TOPN * 2 or not ga.wm.fitted:
                continue
            mean, std = ga.wm.predict(pool)
            acq = mean + ga.beta * std
            R = np.array([rw(p) for p in pool])               # full oracle evaluation
            true_top = set(np.argsort(-R)[:TOPN].tolist())
            e_rand = float(np.mean([enrichment(rng0.permutation(len(pool)).astype(float), true_top, TOPN)
                                    for _ in range(50)]))
            rows.append(dict(
                n_pool=len(pool),
                rho_acq=float(spearmanr(acq, R).correlation),
                rho_mean=float(spearmanr(mean, R).correlation),
                rho_std=float(spearmanr(std, R).correlation),
                enr_acq=enrichment(acq, true_top, TOPN), enr_mean=enrichment(mean, true_top, TOPN),
                enr_std=enrichment(std, true_top, TOPN), enr_rand=e_rand))
            print(f"  [{target} seed {s+1}/{args.seeds}] rho_acq={rows[-1]['rho_acq']:.2f} "
                  f"enr_acq={rows[-1]['enr_acq']:.2f} enr_rand={rows[-1]['enr_rand']:.2f}", flush=True)
        results[target] = rows
        os.makedirs(OUT_DIR, exist_ok=True)
        json.dump({'k': K, 'warmup': WARMUP, 'pool': POOL, 'topn': TOPN, 'results': results},
                  open(args.out, 'w'))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
