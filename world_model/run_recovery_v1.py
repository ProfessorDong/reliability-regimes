"""Retrospective known-ligand recovery (reviewer/Oz point: show the methodology
identifies and evaluates a known active ligand for a target).

For each target we take the R most potent measured actives as held-out "known
reference ligands". For each reference ligand L:
  * the oracle EVALUATES L (predicted pIC50 vs measured) -> identification check;
  * we seed ST-GA with k OTHER actives (L and its near-duplicates removed from the
    seed pool), run the surrogate-triaged search to the standard oracle budget, and
    measure the maximum Tanimoto between any generated molecule and L (recovery).
We compare against (a) the seed set alone (max Tanimoto from the k seeds to L: does
the SEARCH, not just the seeds, close the gap to L?) and (b) the random-triage
control (isolates the surrogate's contribution).

Identical reward / budget / oracle configuration to run_wmga_v1.py (the frozen
campaign). Writes recovery_results.json.

    python -m world_model.run_recovery_v1 --targets scd1 fads nk1r drd2 drd3 \
        --R 20 --seeds 3 --k 10
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.oracle import load_target_data, load_oracle, predict_pic50  # noqa: E402
from world_model.reward import CWMReward  # noqa: E402
from world_model.dynamics import LatentEmbedder, LatentWorldModel  # noqa: E402
from world_model.wm_guided_ga import WMGuidedGA  # noqa: E402
from world_model.run_fewshot_v1 import REWARD_CFG, DEFAULT_CFG, OUT_DIR  # noqa: E402

RDLogger.DisableLog('rdApp.*')
BUDGET = 300
LAMBDA_UNC = 0.1
DUP_THR = 0.99          # a seed within this Tanimoto of L counts as the same molecule
THETAS = [0.4, 0.5, 0.6]


def _fp(s):
    m = Chem.MolFromSmiles(str(s))
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) if m else None


def max_sim(smiles_list, fpL):
    best = 0.0
    for s in smiles_list:
        fp = _fp(s)
        if fp is None:
            continue
        t = DataStructs.TanimotoSimilarity(fp, fpL)
        if t > best:
            best = t
    return float(best)


def run_cell(target, L, L_idx, seed, actives, oracle, embedder, cfg):
    """One (reference ligand, seed) trial: returns dict of recovery sims."""
    fpL = _fp(L)
    # seed pool = actives excluding L and its near-duplicates
    pool = [s for s in actives if s != L]
    pool = [s for s in pool if (_fp(s) is None) or
            DataStructs.TanimotoSimilarity(_fp(s), fpL) < DUP_THR]
    rng = np.random.default_rng(1000 * seed + L_idx)
    k = min(10, len(pool))
    seeds = [pool[i] for i in rng.choice(len(pool), k, replace=False)]
    seed_sim = max_sim(seeds, fpL)
    out = {'L': L, 'L_idx': int(L_idx), 'seed': int(seed), 'seed_sim': seed_sim}
    for mode in ('wm', 'randtriage'):
        rw = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'])
        wm = LatentWorldModel(embedder, n_ensemble=5, device='cuda', seed=seed) if mode == 'wm' else None
        ga = WMGuidedGA(target, rw, world_model=wm, mode=mode, seed=seed)
        ga.run(seeds, budget=BUDGET)
        gen = [s for s in ga.scores if s not in set(seeds)]   # SEARCH products only
        out[mode + '_sim'] = max_sim(gen, fpL)
        out[mode + '_n'] = len(gen)
    return out


def aggregate(rows):
    seed_sim = np.array([r['seed_sim'] for r in rows])
    wm = np.array([r['wm_sim'] for r in rows])
    rd = np.array([r['randtriage_sim'] for r in rows])
    from scipy import stats
    def rec(a):
        return {f'rec@{t}': float(np.mean(a >= t)) for t in THETAS}
    agg = dict(n=len(rows),
               mean_seed_sim=float(seed_sim.mean()),
               mean_wm_sim=float(wm.mean()), mean_rand_sim=float(rd.mean()),
               wm_recovery=rec(wm), rand_recovery=rec(rd), seed_recovery=rec(seed_sim))
    # paired tests: does the search beat the seeds? does wm beat randtriage?
    t1, p1 = stats.ttest_rel(wm, seed_sim)
    t2, p2 = stats.ttest_rel(wm, rd)
    agg['wm_minus_seed'] = float((wm - seed_sim).mean()); agg['p_wm_vs_seed'] = float(p1)
    agg['wm_minus_rand'] = float((wm - rd).mean()); agg['p_wm_vs_rand'] = float(p2)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['scd1', 'fads', 'nk1r', 'drd2', 'drd3'])
    ap.add_argument('--R', type=int, default=20)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'recovery_results.json'))
    args = ap.parse_args()
    embedder = LatentEmbedder(device='cuda')
    results = {}
    for target in args.targets:
        smiles, acts, thr = load_target_data(target)
        oracle = load_oracle(target)
        pairs = sorted(zip(smiles, acts), key=lambda kv: kv[1], reverse=True)
        # unique canonical actives, most-potent first
        seen, ref = set(), []
        for s, a in pairs:
            cs = Chem.MolToSmiles(Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s)) else None
            if cs and cs not in seen and a >= thr:
                seen.add(cs); ref.append((cs, float(a)))
            if len(ref) >= args.R:
                break
        actives = [s for s, a in zip(smiles, acts) if a >= thr]
        cfg = REWARD_CFG.get(target, DEFAULT_CFG)
        # oracle evaluation (identification) check on the reference ligands
        ref_smiles = [s for s, _ in ref]; ref_meas = np.array([a for _, a in ref])
        ref_pred = predict_pic50(oracle, ref_smiles)
        from scipy import stats
        ev_r, ev_p = stats.pearsonr(ref_pred, ref_meas) if len(ref) > 2 else (float('nan'), float('nan'))
        rows = []
        for L_idx, (L, _a) in enumerate(ref):
            for seed in range(args.seeds):
                rows.append(run_cell(target, L, L_idx, seed, actives, oracle, embedder, cfg))
        agg = aggregate(rows)
        agg['eval_mean_pred'] = float(ref_pred.mean())
        agg['eval_mean_meas'] = float(ref_meas.mean())
        agg['eval_pearson_pred_meas'] = float(ev_r)
        agg['eval_p'] = float(ev_p)
        results[target] = dict(agg=agg, rows=rows)
        print(f"== {target}: wm rec@0.4={agg['wm_recovery']['rec@0.4']:.2f} "
              f"rand={agg['rand_recovery']['rec@0.4']:.2f} seed={agg['seed_recovery']['rec@0.4']:.2f} | "
              f"wm_sim={agg['mean_wm_sim']:.3f} seed_sim={agg['mean_seed_sim']:.3f} "
              f"(wm-seed p={agg['p_wm_vs_seed']:.1e}) | eval r={ev_r:.2f} "
              f"pred={ref_pred.mean():.2f}/meas={ref_meas.mean():.2f}", flush=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'config': vars(args), 'results': results}, f, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
