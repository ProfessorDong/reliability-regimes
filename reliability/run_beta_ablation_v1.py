"""beta ablation + leakage-clean primary metric (reviewer points 5, 6).

Does the uncertainty term in the acquisition a = mu + beta*sigma help END TO END?
The mid-search triage diagnostic suggested mean-only (beta=0) ranks better, but no
end-to-end ablation existed. Here we run ST-GA at beta in {0, 0.5, 1, 2} plus vanilla
Graph GA, at k=10 over all five targets and 25 seeds, and report final Top-10 reward
under three metrics:
  * top10_all   : all evaluated molecules (the paper's current metric; includes seeds)
  * top10_gen   : excluding the k support seeds
  * top10_novel : excluding seeds AND exact oracle-training compounds (leakage-clean)
plus a normalized best-reward AUC over the oracle budget.

If beta=0 matches or beats beta>0, the method is mean-surrogate triage, not
uncertainty-aware. Writes beta_ablation.json.

    python -m reliability.run_beta_ablation_v1 --seeds 25
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability.oracle import load_target_data  # noqa: E402
from reliability.reward import TargetReward  # noqa: E402
from reliability.dynamics import LatentEmbedder, LatentSurrogate  # noqa: E402
from reliability.surrogate_ga import SurrogateTriagedGA  # noqa: E402
from reliability.graph_ga import GraphGA, _csmi  # noqa: E402
from reliability.run_fewshot_v1 import REWARD_CFG, DEFAULT_CFG, OUT_DIR  # noqa: E402

BUDGET = 300
LAMBDA_UNC = 0.1
K = 10
BETAS = [0.0, 0.5, 1.0, 2.0]


def top_m(scores_dict, exclude, m=10):
    vals = sorted((v for s, v in scores_dict.items() if s not in exclude), reverse=True)
    return float(np.mean(vals[:m])) if vals else float('nan')


def auc_norm(trajectory, budget):
    if not trajectory:
        return float('nan')
    c = np.array([x for x, _ in trajectory], float)
    b = np.array([y for _, y in trajectory], float)
    c = np.concatenate([c, [budget]]); b = np.concatenate([b, [b[-1]]])
    return float(np.trapz(b, c) / (budget * b.max())) if b.max() > 0 else float('nan')


def metrics(ga, seeds_canon, train_canon):
    return dict(
        top10_all=top_m(ga.scores, set()),
        top10_gen=top_m(ga.scores, seeds_canon),
        top10_novel=top_m(ga.scores, seeds_canon | train_canon),
        auc=auc_norm(ga.trajectory, BUDGET),
        n_unique=len(ga.scores),
    )


def run_cell(target, seed, embedder, actives, train_canon, cfg):
    rng = np.random.default_rng(1000 * seed + K)
    seeds = [actives[i] for i in rng.choice(len(actives), min(K, len(actives)), replace=False)]
    seeds_canon = set(c for c in (_csmi(s) for s in seeds) if c)
    out = {}
    for beta in BETAS:
        rw = TargetReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'])
        wm = LatentSurrogate(embedder, n_ensemble=5, device='cuda', seed=seed)
        ga = SurrogateTriagedGA(target, rw, surrogate=wm, mode='surrogate', beta=beta, seed=seed)
        ga.run(seeds, budget=BUDGET)
        out[f'beta{beta}'] = metrics(ga, seeds_canon, train_canon)
    rw = TargetReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'])
    gg = GraphGA(target, rw, seed=seed)
    gg.run(seeds, budget=BUDGET)
    out['graphga'] = metrics(gg, seeds_canon, train_canon)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['scd1', 'fads', 'nk1r', 'drd2', 'drd3'])
    ap.add_argument('--seeds', type=int, default=25)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'beta_ablation.json'))
    args = ap.parse_args()
    embedder = LatentEmbedder(device='cuda')
    from scipy import stats
    results, summary = {}, {}
    for target in args.targets:
        smiles, acts, thr = load_target_data(target)
        actives = [s for s, a in zip(smiles, acts) if a >= thr]
        train_canon = set(c for c in (_csmi(s) for s in smiles) if c)
        cfg = REWARD_CFG.get(target, DEFAULT_CFG)
        cells = [run_cell(target, s, embedder, actives, train_canon, cfg) for s in range(args.seeds)]
        results[target] = cells
        keys = [f'beta{b}' for b in BETAS] + ['graphga']
        summary[target] = {k: {m: float(np.mean([c[k][m] for c in cells]))
                               for m in ['top10_all', 'top10_gen', 'top10_novel', 'auc']}
                           for k in keys}
        # paired beta=1 vs beta=0 on the leakage-clean metric
        b1 = np.array([c['beta1.0']['top10_novel'] for c in cells])
        b0 = np.array([c['beta0.0']['top10_novel'] for c in cells])
        t, p = stats.ttest_rel(b1, b0)
        summary[target]['beta1_vs_beta0_novel'] = dict(delta=float((b1 - b0).mean()), p=float(p))
        print(f"== {target}: "
              + ' '.join(f"b{b}={summary[target][f'beta{b}']['top10_novel']:.3f}" for b in BETAS)
              + f" GA={summary[target]['graphga']['top10_novel']:.3f}"
              + f" | b1-b0={summary[target]['beta1_vs_beta0_novel']['delta']:+.3f}"
              + f" (p={summary[target]['beta1_vs_beta0_novel']['p']:.2f})", flush=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'config': vars(args), 'betas': BETAS, 'summary': summary, 'results': results}, f, indent=2)
    # pooled beta1 vs beta0 (equal target weight) on leakage-clean metric
    d = [summary[t]['beta1_vs_beta0_novel']['delta'] for t in args.targets]
    print(f"POOLED beta1-beta0 (top10_novel), per-target deltas: "
          f"{[round(x,3) for x in d]}  mean={np.mean(d):+.3f}")
    print('wrote', args.out)


if __name__ == '__main__':
    main()
