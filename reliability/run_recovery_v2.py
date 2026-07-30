"""Leakage-free retrospective recovery (reviewer section 12).

The v1 recovery withheld ligands only from the seeds; they stayed in the oracle
training set, so "the oracle identifies them as potent" was circular. Here we remove an
entire Bemis-Murcko scaffold cluster of actives from BOTH the seeds AND the oracle
training set, RETRAIN the oracle on the remainder, and then ask:
  (a) generalization: does the RETRAINED oracle (which never saw the cluster) still rank
      the held-out actives as potent, and does it flag them as uncertain (higher sigma_T)?
  (b) recovery: seeded from other-scaffold actives, does ST-GA-ECFP (and Graph GA /
      random-triage) generate molecules resembling the held-out cluster, above a
      chemical-space null?
Uses the ECFP-RF surrogate primary method; the reward oracle is the RETRAINED RF. Writes
recovery_v2_results.json.

    python -m reliability.run_recovery_v2 --seeds 3
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability.oracle import load_target_data, featurize, _ensemble_preds  # noqa: E402
from reliability.reward import TargetReward  # noqa: E402
from reliability.surrogate_ga import SurrogateTriagedGA  # noqa: E402
from reliability.graph_ga import GraphGA, _csmi  # noqa: E402
from reliability.run_ecfp_baseline_v1 import ECFPSurrogate  # noqa: E402
from reliability.run_fewshot_v1 import OUT_DIR  # noqa: E402

RDLogger.DisableLog('rdApp.*')
BUDGET = 300
K = 10
THETAS = [0.4, 0.5, 0.6]
RF_KW = dict(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)


def canon(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToSmiles(m) if m else None


def scaffold(s):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(str(s)))
    except Exception:
        return ''


def fp(s):
    m = Chem.MolFromSmiles(str(s)); return AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) if m else None


def max_sim_to(smis, fps_ref):
    best = 0.0
    for s in smis:
        f = fp(s)
        if f is None:
            continue
        t = max(DataStructs.BulkTanimotoSimilarity(f, fps_ref)) if fps_ref else 0.0
        best = max(best, t)
    return float(best)


def run_search(method, target, reward, seeds):
    if method == 'graphga':
        g = GraphGA(target, reward, seed=0)
    elif method == 'randtriage':
        g = SurrogateTriagedGA(target, reward, surrogate=None, mode='randtriage', seed=0)
    else:
        g = SurrogateTriagedGA(target, reward, surrogate=ECFPSurrogate(seed=0), mode='surrogate', beta=0.0, seed=0)
    g.run(seeds, BUDGET)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['scd1', 'fads', 'nk1r', 'drd2', 'drd3'])
    ap.add_argument('--clusters', type=int, default=3)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'recovery_v2_results.json'))
    args = ap.parse_args()
    from scipy import stats
    results = {}
    for target in args.targets:
        smiles, acts, thr = load_target_data(target)
        y = np.asarray(acts, float)
        cs = [canon(s) for s in smiles]
        X = featurize(smiles)
        act_idx = [i for i in range(len(smiles)) if y[i] >= thr and cs[i]]
        # group actives by scaffold, largest clusters first
        groups = {}
        for i in act_idx:
            groups.setdefault(scaffold(smiles[i]), []).append(i)
        clusters = sorted([g for sc, g in groups.items() if sc and len(g) >= 3],
                          key=len, reverse=True)[:args.clusters]
        cfg_w = {'potency': 1.0, 'selectivity': 0.0, 'qed': 0.3, 'sa': 0.3}
        rows = []
        for ci, cl in enumerate(clusters):
            held_canon = set(cs[i] for i in cl)
            held_fps = [fp(smiles[i]) for i in cl]; held_fps = [f for f in held_fps if f]
            tr = [i for i in range(len(smiles)) if cs[i] not in held_canon]
            rf = RandomForestRegressor(**RF_KW).fit(X[tr], y[tr])          # RETRAINED, leakage-free
            Ph = _ensemble_preds(rf, X[cl]); pred_h = Ph.mean(0); sig_h = Ph.std(0)
            # null: chemical-space similarity of random training mols to the held cluster
            rngn = np.random.default_rng(0)
            null_idx = rngn.choice(tr, min(200, len(tr)), replace=False)
            null_sim = float(np.mean([max(DataStructs.BulkTanimotoSimilarity(fp(smiles[i]), held_fps))
                                      for i in null_idx if fp(smiles[i])]))
            seed_pool = [smiles[i] for i in act_idx if cs[i] not in held_canon]
            rec = {m: [] for m in ('stga_ecfp', 'graphga', 'randtriage')}
            for seed in range(args.seeds):
                rng = np.random.default_rng(1000 * seed + ci)
                seeds = [seed_pool[i] for i in rng.choice(len(seed_pool), min(K, len(seed_pool)), replace=False)]
                sc = set(c for c in (_csmi(s) for s in seeds) if c)
                for m in rec:
                    rw = TargetReward(target, anti_targets=(), lambda_unc=0.1, weights=cfg_w)
                    rw.model = rf                                            # inject retrained oracle
                    g = run_search(m, target, rw, seeds)
                    gen = [s for s in g.scores if s not in sc]
                    rec[m].append(max_sim_to(gen, held_fps))
            rows.append(dict(
                cluster_size=len(cl), n_train_removed=len(cl),
                retrained_pred_mean=float(pred_h.mean()), measured_mean=float(y[cl].mean()),
                retrained_sigma_mean=float(sig_h.mean()), null_sim=null_sim,
                rec_mean={m: float(np.mean(v)) for m, v in rec.items()},
                rec_at={m: {f'{t}': float(np.mean(np.array(v) >= t)) for t in THETAS} for m, v in rec.items()}))
        # aggregate over clusters
        pm = np.mean([r['retrained_pred_mean'] for r in rows]); mm = np.mean([r['measured_mean'] for r in rows])
        sg = np.mean([r['retrained_sigma_mean'] for r in rows]); nl = np.mean([r['null_sim'] for r in rows])
        st = np.mean([r['rec_mean']['stga_ecfp'] for r in rows]); ga = np.mean([r['rec_mean']['graphga'] for r in rows])
        results[target] = dict(n_clusters=len(rows), retrained_pred=float(pm), measured=float(mm),
                               retrained_sigma=float(sg), null_sim=float(nl),
                               rec_stga=float(st), rec_graphga=float(ga), rows=rows)
        print(f"== {target}: retrained-oracle pred(held)={pm:.2f} vs measured={mm:.2f} "
              f"sigma(held)={sg:.2f} | recovery stga={st:.2f} GA={ga:.2f} null={nl:.2f}", flush=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'config': vars(args), 'results': results}, f, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
