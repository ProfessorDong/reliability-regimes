"""Per-optimizer, lambda-controlled novelty frontier with training-distance (reviewer
sections 3.2, 3.3, and the corrected mechanism).

The original sweep used one optimizer (the BRICS fragment-edit search) at fixed
lambda=0.1. Here we sweep the novelty weight w_nu in {0,0.5,1,2} for TWO further
optimizers (vanilla Graph GA and ST-GA/mean-surrogate triage) and at TWO penalties
lambda in {0,0.1}, over five targets and 15 seeds. For the top molecules of each run we
measure four ACHIEVED quantities (novelty term removed): support-novelty (vs the k
seeds), d_train (distance to the FULL oracle-training set), oracle predicted potency,
and oracle ensemble disagreement sigma_T.

This tests (a) that the novelty->sigma_T relation is not an artifact of the -lambda*sigma
penalty (lambda=0), (b) that it holds across optimizers, and (c) the corrected
mechanism: pushing novelty drives d_train up, and sigma_T tracks d_train, not just
support-novelty. Writes frontier_v2_results.json + frontier_v2_analysis.json.

    python -m reliability.run_frontier_v2 --seeds 15
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability.oracle import load_target_data, load_oracle, predict_pic50, featurize  # noqa: E402
from reliability.reward import TargetReward  # noqa: E402
from reliability.dynamics import LatentEmbedder, LatentSurrogate  # noqa: E402
from reliability.surrogate_ga import SurrogateTriagedGA  # noqa: E402
from reliability.graph_ga import GraphGA, _csmi  # noqa: E402
from reliability.run_applicability_v1 import max_tanimoto_to_set  # noqa: E402
from reliability.run_fewshot_v1 import REWARD_CFG, DEFAULT_CFG, OUT_DIR  # noqa: E402

BUDGET = 300
K = 10
WNOV = [0.0, 0.5, 1.0, 2.0]
LAMBDAS = [0.0, 0.1]
OPTS = ['graphga', 'stga']


def run_one(opt, target, reward, wm, seeds):
    if opt == 'graphga':
        ga = GraphGA(target, reward, seed=0)
    else:
        ga = SurrogateTriagedGA(target, reward, surrogate=wm, mode='surrogate', beta=0.0, seed=0)
    ga.run(seeds, budget=BUDGET)
    return ga


def measure(top, seeds_canon, oracle, X_train, X_sup):
    top = [s for s in top if s not in seeds_canon]
    if not top:
        return None
    Xt = featurize(top)
    pot, sig = predict_pic50(oracle, top, return_std=True)
    nov = 1.0 - max_tanimoto_to_set(Xt, X_sup)
    dtr = 1.0 - max_tanimoto_to_set(Xt, X_train)
    return dict(novelty=float(nov.mean()), d_train=float(dtr.mean()),
                potency=float(pot.mean()), sigma=float(sig.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['scd1', 'fads', 'nk1r', 'drd2', 'drd3'])
    ap.add_argument('--seeds', type=int, default=15)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'frontier_v2_results.json'))
    args = ap.parse_args()
    embedder = LatentEmbedder(device='cuda')
    results = {}
    for target in args.targets:
        smiles, acts, thr = load_target_data(target)
        actives = [s for s, a in zip(smiles, acts) if a >= thr]
        oracle = load_oracle(target)
        X_train = featurize(smiles)
        cfg = REWARD_CFG.get(target, DEFAULT_CFG)
        rows = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 * seed + K)
            seeds = [actives[i] for i in rng.choice(len(actives), min(K, len(actives)), replace=False)]
            seeds_canon = set(c for c in (_csmi(s) for s in seeds) if c)
            X_sup = featurize(seeds)
            for lam in LAMBDAS:
                for wnov in WNOV:
                    w = dict(cfg['weights']);
                    rw = TargetReward(target, anti_targets=cfg['anti'], lambda_unc=lam,
                                   weights=w, novelty_anchors=seeds, w_novelty=wnov)
                    for opt in OPTS:
                        wm = LatentSurrogate(embedder, n_ensemble=5, device='cuda', seed=seed) \
                            if opt == 'stga' else None
                        # fresh reward per run (oracle-call counter); same config
                        rwr = TargetReward(target, anti_targets=cfg['anti'], lambda_unc=lam,
                                        weights=w, novelty_anchors=seeds, w_novelty=wnov)
                        ga = run_one(opt, target, rwr, wm, seeds)
                        m = measure(ga.results()['top_smiles'], seeds_canon, oracle, X_train, X_sup)
                        if m:
                            m.update(dict(seed=seed, lam=lam, wnov=wnov, opt=opt))
                            rows.append(m)
        results[target] = rows
        # quick per-target console summary (opt=stga, lam=0.1)
        sub = [r for r in rows if r['opt'] == 'stga' and r['lam'] == 0.1]
        if len(sub) > 3:
            nv = np.array([r['novelty'] for r in sub]); sg = np.array([r['sigma'] for r in sub])
            dt = np.array([r['d_train'] for r in sub]); pt = np.array([r['potency'] for r in sub])
            print(f"{target} [stga,lam.1]: nov->sig {stats.spearmanr(nv,sg)[0]:+.2f}  "
                  f"nov->dtrain {stats.spearmanr(nv,dt)[0]:+.2f}  nov->pot {stats.spearmanr(nv,pt)[0]:+.2f}",
                  flush=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'config': vars(args), 'results': results}, f, indent=2)

    # analysis: per (opt, lam) pooled + per-target Spearman
    analysis = {}
    allrows = [r for rows in results.values() for r in rows]
    for opt in OPTS:
        for lam in LAMBDAS:
            key = f'{opt}_lam{lam}'
            sub = [r for r in allrows if r['opt'] == opt and r['lam'] == lam]
            nv = np.array([r['novelty'] for r in sub]); sg = np.array([r['sigma'] for r in sub])
            dt = np.array([r['d_train'] for r in sub]); pt = np.array([r['potency'] for r in sub])
            per_t = {}
            for target in args.targets:
                st = [r for r in results[target] if r['opt'] == opt and r['lam'] == lam]
                if len(st) > 3:
                    a = np.array([r['novelty'] for r in st]); b = np.array([r['sigma'] for r in st])
                    c = np.array([r['potency'] for r in st])
                    per_t[target] = dict(nov_sig=float(stats.spearmanr(a, b)[0]),
                                         nov_pot=float(stats.spearmanr(a, c)[0]))
            analysis[key] = dict(
                n=len(sub),
                nov_sig=float(stats.spearmanr(nv, sg)[0]), nov_sig_p=float(stats.spearmanr(nv, sg)[1]),
                nov_dtrain=float(stats.spearmanr(nv, dt)[0]),
                nov_pot=float(stats.spearmanr(nv, pt)[0]),
                dtrain_sig=float(stats.spearmanr(dt, sg)[0]),
                per_target=per_t)
    with open(os.path.join(OUT_DIR, 'frontier_v2_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    for key, a in analysis.items():
        print(f"{key}: nov->sig {a['nov_sig']:+.3f}  nov->dtrain {a['nov_dtrain']:+.3f}  "
              f"nov->pot {a['nov_pot']:+.3f}  dtrain->sig {a['dtrain_sig']:+.3f}  (n={a['n']})")
    print('wrote', args.out, 'and frontier_v2_analysis.json')


if __name__ == '__main__':
    main()
