"""Does the dual-encoder latent surrogate earn its keep? (reviewer section 11).

The paper argues that building the triage surrogate on the oracle's own ECFP would be
"circular". That is a weak defence: a surrogate trained from a limited number of oracle
queries is a legitimate surrogate even when its representation matches the oracle's, and
it is the key baseline for whether the frozen dual encoder adds value. Since the method
is mean-surrogate triage (beta=0), we compare, at k=10 over five targets and 25 seeds:
  * ST-GA with the dual-encoder latent surrogate (beta=0)
  * ST-GA with an online ECFP random-forest surrogate (beta=0)
  * vanilla Graph GA
on the leakage-clean primary metric (top-10 excluding seeds and exact training compounds).
Writes ecfp_baseline.json.

    python -m reliability.run_ecfp_baseline_v1 --seeds 25
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability.oracle import load_target_data, featurize  # noqa: E402
from reliability.reward import TargetReward  # noqa: E402
from reliability.dynamics import LatentEmbedder, LatentSurrogate  # noqa: E402
from reliability.surrogate_ga import SurrogateTriagedGA  # noqa: E402
from reliability.graph_ga import GraphGA, _csmi  # noqa: E402
from reliability.run_fewshot_v1 import REWARD_CFG, DEFAULT_CFG, OUT_DIR  # noqa: E402

BUDGET = 300
LAMBDA_UNC = 0.1
K = 10


class ECFPSurrogate:
    """Online ECFP random-forest surrogate; same interface as LatentSurrogate."""
    def __init__(self, n_estimators=100, seed=0):
        self.ne = n_estimators; self.seed = seed
        self.Xbuf = None; self.ybuf = None
        self.rf = None; self.mu = 0.0; self.sd = 1.0; self.fitted = False

    def update(self, smiles, rewards, epochs=None):
        X = featurize(list(smiles)); r = np.asarray(rewards, float)
        if self.Xbuf is None:
            self.Xbuf, self.ybuf = X, r
        else:
            self.Xbuf = np.vstack([self.Xbuf, X]); self.ybuf = np.concatenate([self.ybuf, r])
        self.mu = float(self.ybuf.mean()); self.sd = float(self.ybuf.std() + 1e-6)
        yz = (self.ybuf - self.mu) / self.sd
        self.rf = RandomForestRegressor(n_estimators=self.ne, min_samples_leaf=2,
                                        n_jobs=-1, random_state=self.seed).fit(self.Xbuf, yz)
        self.fitted = True

    def predict(self, smiles):
        X = featurize(list(smiles))
        P = np.stack([t.predict(X) for t in self.rf.estimators_], 0)
        return P.mean(0) * self.sd + self.mu, P.std(0) * self.sd


def top10_novel(ga, seeds_canon, train_canon):
    vals = sorted((v for s, v in ga.scores.items()
                   if s not in seeds_canon and s not in train_canon), reverse=True)
    return float(np.mean(vals[:10])) if vals else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['scd1', 'fads', 'nk1r', 'drd2', 'drd3'])
    ap.add_argument('--seeds', type=int, default=25)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'ecfp_baseline.json'))
    args = ap.parse_args()
    embedder = LatentEmbedder(device='cuda')
    from scipy import stats
    results, summary = {}, {}
    for target in args.targets:
        smiles, acts, thr = load_target_data(target)
        actives = [s for s, a in zip(smiles, acts) if a >= thr]
        train_canon = set(c for c in (_csmi(s) for s in smiles) if c)
        cfg = REWARD_CFG.get(target, DEFAULT_CFG)
        rows = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 * seed + K)
            seeds = [actives[i] for i in rng.choice(len(actives), min(K, len(actives)), replace=False)]
            sc = set(c for c in (_csmi(s) for s in seeds) if c)
            row = {}
            # latent surrogate
            rw = TargetReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'])
            wm = LatentSurrogate(embedder, n_ensemble=5, device='cuda', seed=seed)
            g = SurrogateTriagedGA(target, rw, surrogate=wm, mode='surrogate', beta=0.0, seed=seed); g.run(seeds, BUDGET)
            row['latent'] = top10_novel(g, sc, train_canon)
            # ecfp surrogate
            rw = TargetReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'])
            g = SurrogateTriagedGA(target, rw, surrogate=ECFPSurrogate(seed=seed), mode='surrogate', beta=0.0, seed=seed)
            g.run(seeds, BUDGET); row['ecfp'] = top10_novel(g, sc, train_canon)
            # vanilla graph ga
            rw = TargetReward(target, anti_targets=cfg['anti'], lambda_unc=LAMBDA_UNC, weights=cfg['weights'])
            g = GraphGA(target, rw, seed=seed); g.run(seeds, BUDGET)
            row['graphga'] = top10_novel(g, sc, train_canon)
            rows.append(row)
        results[target] = rows
        lat = np.array([r['latent'] for r in rows]); ec = np.array([r['ecfp'] for r in rows])
        ga = np.array([r['graphga'] for r in rows])
        summary[target] = dict(
            latent=float(lat.mean()), ecfp=float(ec.mean()), graphga=float(ga.mean()),
            latent_minus_ecfp=float((lat - ec).mean()), p_lat_ecfp=float(stats.ttest_rel(lat, ec)[1]),
            latent_minus_ga=float((lat - ga).mean()), ecfp_minus_ga=float((ec - ga).mean()))
        s = summary[target]
        print(f"== {target}: latent={s['latent']:.3f} ecfp={s['ecfp']:.3f} GA={s['graphga']:.3f} "
              f"| lat-ecfp={s['latent_minus_ecfp']:+.3f} (p={s['p_lat_ecfp']:.2f}) "
              f"lat-GA={s['latent_minus_ga']:+.3f} ecfp-GA={s['ecfp_minus_ga']:+.3f}", flush=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'config': vars(args), 'summary': summary, 'results': results}, f, indent=2)
    dle = [summary[t]['latent_minus_ecfp'] for t in args.targets]
    deg = [summary[t]['ecfp_minus_ga'] for t in args.targets]
    print(f"POOLED latent-ecfp per-target: {[round(x,3) for x in dle]} mean={np.mean(dle):+.3f}")
    print(f"POOLED ecfp-GA   per-target: {[round(x,3) for x in deg]} mean={np.mean(deg):+.3f}")
    print('wrote', args.out)


if __name__ == '__main__':
    main()
