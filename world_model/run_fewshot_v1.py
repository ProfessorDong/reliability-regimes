"""Few-shot sample-efficiency experiment (Section V-B).

The world-model claim must be tested where selection matters: only k known actives
are given, the search has a large actionable space (seed fragments + a generic
ChEMBL fragment pool), and the real-oracle budget is tight. We compare:
  * wm     : world-model UCB acquisition (imagined rollouts prioritize oracle calls)
  * random : identical budget, random selection (no world model)  [controlled ablation]

For each (target, k, seed) we draw k random actives, run both modes to a fixed
oracle budget, and record best-reward / top-k / sample-efficiency-AUC / novelty.
Aggregated over many seeds (default 25, mirroring the prediction paper's rigor).

Pilot:   python -m world_model.run_fewshot_v1 --targets drd3 --ks 10 --seeds 3 --budget 250
Full:    python -m world_model.run_fewshot_v1 --targets drd3 scd1 --ks 5 10 20 --seeds 25 --budget 300
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.oracle import load_target_data, GPCR_DOCKING  # noqa: E402
from world_model.reward import CWMReward  # noqa: E402
from world_model.actions import FragmentLibrary, ActionSpace, AdmissibilityFilter  # noqa: E402
from world_model.dynamics import LatentEmbedder, LatentWorldModel  # noqa: E402
from world_model.planner import Planner  # noqa: E402
import joblib  # noqa: E402

RDLogger.DisableLog('rdApp.*')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'cwm_v1')
POOL_CACHE = os.path.join(OUT_DIR, 'chembl_fragment_pool.pkl')

# DRD2-vs-DRD3 selectivity reward; potency+drug-likeness elsewhere.
REWARD_CFG = {
    'drd3': dict(anti=('drd2',), weights={'potency': 1.0, 'selectivity': 0.5, 'qed': 0.3, 'sa': 0.3}),
    'drd2': dict(anti=('drd3',), weights={'potency': 1.0, 'selectivity': 0.5, 'qed': 0.3, 'sa': 0.3}),
}
DEFAULT_CFG = dict(anti=(), weights={'potency': 1.0, 'selectivity': 0.0, 'qed': 0.3, 'sa': 0.3})


def build_chembl_pool(n=2000, seed=0):
    if os.path.exists(POOL_CACHE):
        return joblib.load(POOL_CACHE)
    import pandas as pd
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'chembl_pretrain_smiles.csv'))
    col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
    rng = np.random.default_rng(seed)
    sample = [str(df[col].iloc[i]) for i in rng.choice(len(df), n, replace=False)]
    pool = FragmentLibrary().add_smiles(sample)
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(pool, POOL_CACHE)
    return pool


def _fp(s):
    m = Chem.MolFromSmiles(str(s))
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) if m else None


def _library_for(seed_smiles, chembl_pool):
    lib = FragmentLibrary()
    for lbl, frags in chembl_pool.by_label.items():
        lib.by_label[lbl] |= set(frags)
    lib.add_smiles(seed_smiles)
    return lib


def novelty(top_smiles, seed_smiles, thr=0.4):
    """Fraction of generated molecules substantially new vs the k seeds (max Tanimoto < thr)."""
    seeds = [fp for fp in (_fp(s) for s in seed_smiles) if fp is not None]
    gen = [s for s in top_smiles if s not in set(seed_smiles)]
    if not gen or not seeds:
        return float('nan'), float('nan')
    sims = []
    for s in gen:
        fp = _fp(s)
        sims.append(max(DataStructs.BulkTanimotoSimilarity(fp, seeds)) if fp else 1.0)
    sims = np.asarray(sims)
    return float(np.mean(sims < thr)), float(sims.mean())


def auc_efficiency(trajectory, budget):
    """Normalized area under best-reward-vs-oracle-calls (higher = faster)."""
    if not trajectory:
        return float('nan')
    calls = np.array([c for c, _ in trajectory], float)
    best = np.array([b for _, b in trajectory], float)
    calls = np.concatenate([calls, [budget]]); best = np.concatenate([best, [best[-1]]])
    return float(np.trapz(best, calls) / (budget * max(best.max(), 1e-9)))


def run_cell(target, k, seed, budget, chembl_pool, embedder, modes, hp):
    smiles, acts, thr = load_target_data(target)
    actives = [s for s, a in zip(smiles, acts) if a >= thr]
    rng = np.random.default_rng(1000 * seed + k)
    seeds = [actives[i] for i in rng.choice(len(actives), min(k, len(actives)), replace=False)]
    lib = _library_for(seeds, chembl_pool)
    cfg = REWARD_CFG.get(target, DEFAULT_CFG)

    cell = {}
    for mode in modes:
        rw = CWMReward(target, anti_targets=cfg['anti'], lambda_unc=hp['lambda_unc'], weights=cfg['weights'])
        wm = LatentWorldModel(embedder, n_ensemble=5, device='cuda', seed=seed) if mode == 'wm' else None
        sp = ActionSpace(lib, max_neighbors=hp['neighbors_per'], seed=seed)
        adm = AdmissibilityFilter(seeds, threshold=hp['adm_thr'])
        pl = Planner(target, rw, sp, wm, adm, mode=mode, beta=hp['beta'],
                     expand_top=hp['expand_top'], neighbors_per=hp['neighbors_per'],
                     eval_per_round=hp['eval_per_round'], grow_anchors=True, seed=seed)
        res = pl.run(seeds, budget=budget, warmup=min(k, 20))
        nov_frac, nov_sim = novelty(res['top_smiles'], seeds)
        cell[mode] = dict(best=res['best_reward'], top10=res['top10_mean'],
                          top100=res['top100_mean'], n_unique=res['n_unique'],
                          auc=auc_efficiency(res['trajectory'], budget),
                          novelty_frac=nov_frac, mean_sim_to_seed=nov_sim,
                          calls=res['oracle_calls'])
    return cell


def aggregate(cells, modes, keys):
    out = {}
    for mode in modes:
        out[mode] = {}
        for key in keys:
            vals = np.array([c[mode][key] for c in cells if np.isfinite(c[mode][key])])
            out[mode][key] = [float(vals.mean()), float(vals.std())] if len(vals) else [float('nan')] * 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['drd3'])
    ap.add_argument('--ks', nargs='+', type=int, default=[10])
    ap.add_argument('--seeds', type=int, default=25)
    ap.add_argument('--budget', type=int, default=300)
    ap.add_argument('--adm_thr', type=float, default=0.3)
    ap.add_argument('--expand_top', type=int, default=20)
    ap.add_argument('--neighbors_per', type=int, default=24)
    ap.add_argument('--eval_per_round', type=int, default=20)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--lambda_unc', type=float, default=0.05)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    hp = dict(adm_thr=args.adm_thr, expand_top=args.expand_top,
              neighbors_per=args.neighbors_per, eval_per_round=args.eval_per_round,
              beta=args.beta, lambda_unc=args.lambda_unc)

    pool = build_chembl_pool()
    print(f"ChEMBL fragment pool: {pool.size()} fragments / {len(pool.by_label)} labels", flush=True)
    embedder = LatentEmbedder(device='cuda')
    modes = ['wm', 'random']
    keys = ['best', 'top10', 'top100', 'auc', 'novelty_frac', 'mean_sim_to_seed', 'n_unique']

    results = []
    for target in args.targets:
        for k in args.ks:
            cells = []
            for s in range(args.seeds):
                cells.append(run_cell(target, k, s, args.budget, pool, embedder, modes, hp))
                print(f"  [{target} k={k} seed={s+1}/{args.seeds}] "
                      f"wm.top10={cells[-1]['wm']['top10']:.3f} "
                      f"rand.top10={cells[-1]['random']['top10']:.3f}", flush=True)
            agg = aggregate(cells, modes, keys)
            row = dict(target=target, k=k, n_seeds=args.seeds, budget=args.budget, agg=agg,
                       hp=hp, per_seed=cells)
            results.append(row)
            d = agg['wm']['top10'][0] - agg['random']['top10'][0]
            print(f"== {target} k={k}: wm.top10={agg['wm']['top10'][0]:.3f}+/-{agg['wm']['top10'][1]:.3f} "
                  f"random.top10={agg['random']['top10'][0]:.3f}+/-{agg['random']['top10'][1]:.3f} "
                  f"| wm advantage {d:+.3f} | wm.novelty={agg['wm']['novelty_frac'][0]:.2f}", flush=True)
            # incremental save so a long unattended run survives interruption
            os.makedirs(OUT_DIR, exist_ok=True)
            out = args.out or os.path.join(OUT_DIR, 'fewshot_v1_results.json')
            with open(out, 'w') as f:
                json.dump({'config': vars(args), 'hp': hp, 'results': results}, f, indent=2)

    print('wrote', out)


if __name__ == '__main__':
    main()
