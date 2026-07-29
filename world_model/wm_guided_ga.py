"""World-model-guided Graph GA (the engineered improvement; pre-registered test).

Hypothesis: the world model adds value not as a stand-alone optimizer but as a
cheap TRIAGE on top of the strongest search operator (Graph GA crossover/mutation).
Each generation we generate a LARGE offspring pool with GA operators (free, no
oracle) and use the latent world-model surrogate (UCB = mean + beta*std) to pick
which few to actually score with the real oracle, under a fixed oracle budget.

Modes:
  'wm'         : surrogate UCB triage of a pool of size eval_per_gen * pool_mult
  'randtriage' : random triage of the SAME-size pool (attribution control: isolates
                 the world model's contribution from merely having a bigger pool)
  (vanilla Graph GA = graph_ga.GraphGA, already run, evaluates eval_per_gen offspring)

run(seed_smiles, budget) -> results (same shape as Planner/GraphGA).

    python -m world_model.wm_guided_ga --target drd3 --mode wm --budget 300
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.graph_ga import crossover, mutate, _csmi  # noqa: E402
from world_model.dynamics import LatentWorldModel  # noqa: E402


class WMGuidedGA:
    def __init__(self, target, reward, world_model=None, pop_size=60, eval_per_gen=40,
                 pool_mult=5, mut_rate=0.5, beta=1.0, mode='wm', seed=0):
        assert mode in ('wm', 'randtriage')
        self.target = target
        self.reward = reward
        self.wm = world_model
        self.pop_size = pop_size
        self.eval_per_gen = eval_per_gen
        self.pool_mult = pool_mult
        self.mut_rate = mut_rate
        self.beta = beta
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.scores = {}
        self.trajectory = []

    def _evaluate(self, smiles):
        new = [s for s in dict.fromkeys(smiles) if s and s not in self.scores]
        if not new:
            return
        r = np.asarray(self.reward(new))
        for s, v in zip(new, r):
            self.scores[s] = float(v)
        if self.wm is not None:
            self.wm.update(new, r, epochs=200)
        self.trajectory.append((self.reward.n_oracle_calls, max(self.scores.values())))

    def _parents(self, pop, fit):
        f = np.clip(np.asarray(fit) - min(fit) + 1e-3, 1e-3, None)
        p = f / f.sum()
        i, j = self.rng.choice(len(pop), size=2, p=p)
        return pop[i], pop[j]

    def _make_pool(self, pop, fit, target_size):
        pool, guard = [], 0
        seen = set(self.scores)
        while len(pool) < target_size and guard < target_size * 25:
            guard += 1
            a, b = self._parents(pop, fit)
            child = crossover(a, b, self.rng)
            if child is None:
                continue
            if self.rng.random() < self.mut_rate:
                child = mutate(child, self.rng)
            if child and child not in seen:
                seen.add(child); pool.append(child)
        return pool

    def run(self, seed_smiles, budget):
        seeds = [s for s in (_csmi(x) for x in seed_smiles) if s]
        self._evaluate(seeds)
        target_pool = self.eval_per_gen * self.pool_mult
        while self.reward.n_oracle_calls < budget:
            pop = sorted(self.scores, key=self.scores.get, reverse=True)[:self.pop_size]
            fit = [self.scores[s] for s in pop]
            pool = self._make_pool(pop, fit, target_pool)
            if not pool:
                break
            # exact-budget accounting: never exceed `budget` total oracle calls
            n_take = min(self.eval_per_gen, budget - self.reward.n_oracle_calls)
            if n_take <= 0:
                break
            if self.mode == 'wm' and self.wm is not None and self.wm.fitted:
                mean, std = self.wm.predict(pool)
                acq = mean + self.beta * std
                chosen = [pool[i] for i in np.argsort(-acq)[:n_take]]
            else:
                idx = self.rng.permutation(len(pool))[:n_take]
                chosen = [pool[i] for i in idx]
            self._evaluate(chosen)
        return self.results()

    def results(self):
        items = sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)
        rewards = np.array([v for _, v in items])
        return dict(target=self.target, method=f'wmga_{self.mode}',
                    oracle_calls=self.reward.n_oracle_calls, n_unique=len(items),
                    best_reward=float(rewards[0]) if len(rewards) else float('nan'),
                    top10_mean=float(rewards[:10].mean()) if len(rewards) else float('nan'),
                    trajectory=self.trajectory, top_smiles=[s for s, _ in items[:20]])


if __name__ == '__main__':
    import argparse
    from world_model.oracle import load_target_data
    from world_model.reward import CWMReward
    from world_model.dynamics import LatentEmbedder
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='drd3')
    ap.add_argument('--mode', default='wm', choices=['wm', 'randtriage'])
    ap.add_argument('--budget', type=int, default=300)
    ap.add_argument('--k', type=int, default=10)
    args = ap.parse_args()
    smiles, acts, thr = load_target_data(args.target)
    actives = [s for s, a in zip(smiles, acts) if a >= thr]
    rng = np.random.default_rng(0)
    seeds = [actives[i] for i in rng.choice(len(actives), args.k, replace=False)]
    rw = CWMReward(args.target, lambda_unc=0.05, weights={'potency': 1.0, 'qed': 0.3, 'sa': 0.3})
    wm = LatentWorldModel(LatentEmbedder(device='cuda'), n_ensemble=5, device='cuda', seed=0) \
        if args.mode == 'wm' else None
    ga = WMGuidedGA(args.target, rw, world_model=wm, mode=args.mode, seed=0)
    res = ga.run(seeds, budget=args.budget)
    print(f"{res['method']} {args.target}: calls={res['oracle_calls']} "
          f"best={res['best_reward']:.3f} top10={res['top10_mean']:.3f} unique={res['n_unique']}")
