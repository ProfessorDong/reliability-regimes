"""Budgeted imagined-rollout planner for the Chemistry World Model.

Bayesian-optimization-style molecular search with a learned world model and
actionable moves:

  seed from known actives -> repeat until the oracle-call budget is spent:
    1. expand the current best molecules into actionable neighbors (actions.py),
    2. score every neighbor with the world model M_phi (imagination, no oracle):
       acquisition = mean + beta * std  (UCB),
    3. evaluate only the top-Q by acquisition with the REAL oracle (reward.py),
    4. add to the pool, update M_phi.

The sample-efficiency claim is the best-reward-vs-oracle-calls trajectory. Setting
`mode='random'` replaces the acquisition with random selection (identical budget,
no world model): the gap is the controlled imagination ablation (Section V-B/D).

Run:
    python -m world_model.planner --target drd3 --budget 400 --compare
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_model.oracle import load_target_data  # noqa: E402
from world_model.reward import CWMReward  # noqa: E402
from world_model.actions import FragmentLibrary, ActionSpace, AdmissibilityFilter  # noqa: E402
from world_model.dynamics import LatentEmbedder, LatentWorldModel  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
RDLogger.DisableLog('rdApp.*')


def _canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m is not None else None


class Planner:
    def __init__(self, target, reward: CWMReward, action_space: ActionSpace,
                 world_model: LatentWorldModel | None, admissibility: AdmissibilityFilter,
                 beta=1.0, expand_top=20, neighbors_per=24, eval_per_round=20,
                 mode='wm', grow_anchors=True, grow_quantile=0.6, seed=0):
        self.target = target
        self.reward = reward
        self.actions = action_space
        self.wm = world_model
        self.adm = admissibility
        self.beta = beta
        self.expand_top = expand_top
        self.neighbors_per = neighbors_per
        self.eval_per_round = eval_per_round
        self.mode = mode  # 'wm' (UCB acquisition) or 'random' (ablation)
        self.grow_anchors = grow_anchors
        self.grow_quantile = grow_quantile
        self.rng = np.random.default_rng(seed)
        self.scores = {}        # canonical smiles -> real reward
        self.trajectory = []    # (oracle_calls, best_reward)

    def _evaluate(self, smiles):
        """Spend real oracle calls on new molecules; record rewards."""
        new = [s for s in dict.fromkeys(smiles) if s and s not in self.scores]
        if not new:
            return []
        r = np.asarray(self.reward(new))
        for s, v in zip(new, r):
            self.scores[s] = float(v)
        if self.wm is not None:
            self.wm.update(new, r, epochs=200)
        if self.grow_anchors and len(self.scores) > 1:
            cutoff = np.quantile(list(self.scores.values()), self.grow_quantile)
            self.adm.add_anchors([s for s, v in zip(new, r) if v >= cutoff])
        best = max(self.scores.values())
        self.trajectory.append((self.reward.n_oracle_calls, best))
        return new

    def _acquire(self, candidates):
        """Rank candidates; return them best-first under the active mode."""
        candidates = [c for c in candidates if c not in self.scores]
        if not candidates:
            return []
        if self.mode == 'random' or self.wm is None or not self.wm.fitted:
            order = self.rng.permutation(len(candidates))
        else:
            mean, std = self.wm.predict(candidates)
            acq = mean + self.beta * std
            order = np.argsort(-acq)
        return [candidates[i] for i in order]

    def run(self, seed_smiles, budget, warmup=40):
        # warm start: evaluate a random sample of known actives to seed M_phi
        seeds = [s for s in (_canon(x) for x in seed_smiles) if s]
        self.rng.shuffle(seeds)
        self._evaluate(seeds[:warmup])

        while self.reward.n_oracle_calls < budget:
            # expand current best molecules into actionable neighbors
            ranked = sorted(self.scores, key=self.scores.get, reverse=True)
            frontier = ranked[:self.expand_top]
            cand = set()
            for s in frontier:
                for n in self.actions.neighbors(s, k=self.neighbors_per):
                    if self.adm.is_admissible(n):
                        cand.add(n)
            if not cand:
                break
            chosen = self._acquire(list(cand))[:self.eval_per_round]
            if not chosen:
                break
            self._evaluate(chosen)
        return self.results()

    def results(self):
        items = sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)
        rewards = np.array([v for _, v in items])
        return dict(
            target=self.target, mode=self.mode,
            oracle_calls=self.reward.n_oracle_calls,
            n_unique=len(items),
            best_reward=float(rewards[0]) if len(rewards) else float('nan'),
            top10_mean=float(rewards[:10].mean()) if len(rewards) else float('nan'),
            top100_mean=float(rewards[:100].mean()) if len(rewards) else float('nan'),
            trajectory=self.trajectory,
            top_smiles=[s for s, _ in items[:20]],
        )


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='drd3')
    ap.add_argument('--budget', type=int, default=400)
    ap.add_argument('--lib_n', type=int, default=1500)
    ap.add_argument('--compare', action='store_true', help='run wm vs random ablation')
    args = ap.parse_args()
    device = 'cuda'

    smiles, acts, thr = load_target_data(args.target)
    order = np.argsort(acts)
    actives = [smiles[i] for i in order[-min(args.lib_n, len(smiles)):]]
    lib = FragmentLibrary().add_smiles(actives)
    adm = AdmissibilityFilter(actives, threshold=0.4)
    embedder = LatentEmbedder(device=device)

    def make(mode, seed=0):
        rw = CWMReward(args.target, lambda_unc=0.05,
                       weights={'potency': 1.0, 'selectivity': 0.0, 'qed': 0.3, 'sa': 0.3})
        wm = LatentWorldModel(embedder, n_ensemble=5, device=device, seed=seed) if mode == 'wm' else None
        sp = ActionSpace(lib, max_neighbors=24, seed=seed)
        return Planner(args.target, rw, sp, wm, adm, mode=mode, seed=seed)

    out = {}
    modes = ['wm', 'random'] if args.compare else ['wm']
    for mode in modes:
        res = make(mode).run(actives, budget=args.budget)
        out[mode] = res
        print(f"[{mode:6}] calls={res['oracle_calls']} best={res['best_reward']:.3f} "
              f"top10={res['top10_mean']:.3f} top100={res['top100_mean']:.3f} "
              f"unique={res['n_unique']}")
    if args.compare:
        d = out['wm']['top10_mean'] - out['random']['top10_mean']
        print(f"world-model top10 advantage at fixed budget: {d:+.3f}")
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'outputs', 'cwm_v1'), exist_ok=True)
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'outputs', 'cwm_v1', f'planner_smoke_{args.target}.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote', p)
