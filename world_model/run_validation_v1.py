"""Independent-oracle validation (reviewer point: evaluation is oracle-bound).

The search optimizes an RF-on-ECFP oracle. To test whether WM-GA's gain is a
genuine improvement rather than oracle-hacking, we re-score the generated
molecules with an INDEPENDENT oracle that the search never saw: a gradient-
boosting model on RDKit physicochemical descriptors (different model family and
different representation from the RF-on-ECFP search oracle), trained on the same
per-target data. If WM-GA still ranks above vanilla Graph GA under the
independent oracle, the gain transfers beyond the optimized scorer.

    python -m world_model.run_validation_v1 --seeds 15
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.oracle import load_target_data, predict_pic50  # noqa: E402
from world_model.reward import CWMReward  # noqa: E402
from world_model.dynamics import LatentEmbedder, LatentWorldModel  # noqa: E402
from world_model.wm_guided_ga import WMGuidedGA  # noqa: E402
from world_model.graph_ga import GraphGA  # noqa: E402
from world_model.run_fewshot_v1 import REWARD_CFG, DEFAULT_CFG, OUT_DIR  # noqa: E402

BUDGET, K = 300, 10
TARGETS = ['scd1', 'fads', 'nk1r', 'drd2', 'drd3']
_NAMES = [d[0] for d in Descriptors._descList]
_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(_NAMES)


def desc(smiles):
    X = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        X.append(_CALC.CalcDescriptors(m) if m is not None else [0.0] * len(_NAMES))
    return np.nan_to_num(np.asarray(X, float), posinf=0.0, neginf=0.0)


def indep_oracle(target):
    """GBM on RDKit descriptors, trained on full per-target data."""
    smiles, acts, thr = load_target_data(target)
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                          random_state=0).fit(desc(smiles), np.asarray(acts, float))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=15)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'validation_results.json'))
    args = ap.parse_args()
    embedder = LatentEmbedder(device='cuda')
    results = {}
    for target in TARGETS:
        smiles, acts, thr = load_target_data(target)
        actives = [s for s, a in zip(smiles, acts) if a >= thr]
        cfg = REWARD_CFG.get(target, DEFAULT_CFG)
        ind = indep_oracle(target)
        rows = {'wmga': [], 'vanilla': []}
        examples = []
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 * s + K)
            seeds = [actives[i] for i in rng.choice(len(actives), K, replace=False)]
            for method in ('wmga', 'vanilla'):
                rw = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=0.1, weights=cfg['weights'])
                if method == 'wmga':
                    wm = LatentWorldModel(embedder, n_ensemble=5, device='cuda', seed=s)
                    res = WMGuidedGA(target, rw, world_model=wm, mode='wm', seed=s).run(seeds, BUDGET)
                else:
                    res = GraphGA(target, rw, seed=s).run(seeds, BUDGET)
                top = res['top_smiles'][:10]
                rf_pot, _ = predict_pic50(target, top, return_std=True)
                ind_pot = ind.predict(desc(top))
                rows[method].append(dict(rf_reward=res['top10_mean'],
                                         rf_potency=float(np.mean(rf_pot)),
                                         indep_potency=float(np.mean(ind_pot))))
                if s == 0 and method == 'wmga':
                    examples = [dict(smiles=t, rf_pot=float(p), indep_pot=float(q))
                                for t, p, q in zip(top[:5], rf_pot[:5], ind_pot[:5])]
            print(f"  [{target} seed {s+1}/{args.seeds}] done", flush=True)
        results[target] = dict(rows=rows, examples=examples)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'budget': BUDGET, 'k': K, 'seeds': args.seeds, 'results': results}, f)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
