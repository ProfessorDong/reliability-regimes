"""Graph Genetic Algorithm baseline (Jensen, Chem. Sci. 2019; GB-GA).

The standard, strong oracle-budget baseline for molecular optimization (tops the
PMO benchmark). Distinct from the reliability analysis: crossover recombines
fragments cut at ARBITRARY acyclic bonds (not synthesis-aware BRICS bonds), and
mutation is atom/bond-level. It uses the SAME TargetReward oracle, the SAME k-seed
start, and the SAME oracle-call budget, so the comparison is apples-to-apples.

Exposes run(seed_smiles, budget) -> results with the same shape as Planner.results().

    python -m reliability.graph_ga --target drd3 --budget 300
"""
from __future__ import annotations
import os, sys
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import RWMol

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability.actions import _attach, _canon, _dummy_atoms  # reuse fragment-join helpers
RDLogger.DisableLog('rdApp.*')

ORGANIC = [6, 7, 8, 9, 16, 17, 35]  # C N O F S Cl Br
BONDS = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]


def _csmi(s):
    """Canonical SMILES from a SMILES string (None if invalid)."""
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m is not None else None


def _acyclic_single_bonds(mol):
    return [b.GetIdx() for b in mol.GetBonds()
            if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()]


def _fragments_on_random_bond(mol, rng):
    bonds = _acyclic_single_bonds(mol)
    if not bonds:
        return []
    bidx = int(rng.choice(bonds))
    frag = Chem.FragmentOnBonds(mol, [bidx], addDummies=True)
    pieces = Chem.GetMolFrags(frag, asMols=True, sanitizeFrags=False)
    return [p for p in pieces if len(_dummy_atoms(p)) == 1]


def crossover(a, b, rng, tries=10):
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if ma is None or mb is None:
        return None
    for _ in range(tries):
        fa = _fragments_on_random_bond(ma, rng)
        fb = _fragments_on_random_bond(mb, rng)
        if not fa or not fb:
            continue
        child = _attach(fa[int(rng.integers(len(fa)))], fb[int(rng.integers(len(fb)))])
        if child:
            return child
    return None


def mutate(smiles, rng):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return smiles
    op = rng.integers(4)
    rw = RWMol(mol)
    try:
        if op == 0:  # change a heavy-atom element (valence permitting)
            heavy = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() > 1]
            i = int(rng.choice(heavy))
            rw.GetAtomWithIdx(i).SetAtomicNum(int(rng.choice(ORGANIC)))
        elif op == 1:  # change a bond order
            if rw.GetNumBonds() == 0:
                return smiles
            b = rw.GetBondWithIdx(int(rng.integers(rw.GetNumBonds())))
            if not b.IsInRing():
                b.SetBondType(BONDS[int(rng.integers(len(BONDS)))])
        elif op == 2:  # append an atom to a random heavy atom
            i = int(rng.integers(rw.GetNumAtoms()))
            j = rw.AddAtom(Chem.Atom(int(rng.choice(ORGANIC))))
            rw.AddBond(i, j, Chem.BondType.SINGLE)
        else:  # delete a terminal atom
            term = [a.GetIdx() for a in rw.GetAtoms() if a.GetDegree() == 1]
            if term:
                rw.RemoveAtom(int(rng.choice(term)))
        m = rw.GetMol(); Chem.SanitizeMol(m)
        cs = _canon(m)
        return cs or smiles
    except Exception:
        return smiles


class GraphGA:
    def __init__(self, target, reward, population_size=60, mutation_rate=0.5,
                 offspring_per_gen=40, seed=0):
        self.target = target
        self.reward = reward
        self.pop_size = population_size
        self.mut_rate = mutation_rate
        self.offspring = offspring_per_gen
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
        self.trajectory.append((self.reward.n_oracle_calls, max(self.scores.values())))

    def _select_parents(self, pop, fitness):
        f = np.clip(np.asarray(fitness) - min(fitness) + 1e-3, 1e-3, None)
        p = f / f.sum()
        i, j = self.rng.choice(len(pop), size=2, p=p)
        return pop[i], pop[j]

    def run(self, seed_smiles, budget):
        seeds = [s for s in (_csmi(x) for x in seed_smiles) if s]
        self._evaluate(seeds)
        while self.reward.n_oracle_calls < budget:
            pop = sorted(self.scores, key=self.scores.get, reverse=True)[:self.pop_size]
            fit = [self.scores[s] for s in pop]
            # exact-budget accounting: never exceed `budget` total oracle calls
            n_take = min(self.offspring, budget - self.reward.n_oracle_calls)
            if n_take <= 0:
                break
            offspring = []
            guard = 0
            while len(offspring) < n_take and guard < self.offspring * 20:
                guard += 1
                a, b = self._select_parents(pop, fit)
                child = crossover(a, b, self.rng)
                if child is None:
                    continue
                if self.rng.random() < self.mut_rate:
                    child = mutate(child, self.rng)
                if child and child not in self.scores:
                    offspring.append(child)
            if not offspring:
                break
            self._evaluate(offspring)
        return self.results()

    def results(self):
        items = sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)
        rewards = np.array([v for _, v in items])
        return dict(target=self.target, method='graphga',
                    oracle_calls=self.reward.n_oracle_calls, n_unique=len(items),
                    best_reward=float(rewards[0]) if len(rewards) else float('nan'),
                    top10_mean=float(rewards[:10].mean()) if len(rewards) else float('nan'),
                    top100_mean=float(rewards[:100].mean()) if len(rewards) else float('nan'),
                    trajectory=self.trajectory, top_smiles=[s for s, _ in items[:20]])


if __name__ == '__main__':
    import argparse
    from reliability.oracle import load_target_data
    from reliability.reward import TargetReward
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='drd3')
    ap.add_argument('--budget', type=int, default=300)
    ap.add_argument('--k', type=int, default=10)
    args = ap.parse_args()

    smiles, acts, thr = load_target_data(args.target)
    actives = [s for s, a in zip(smiles, acts) if a >= thr]
    rng = np.random.default_rng(0)
    seeds = [actives[i] for i in rng.choice(len(actives), args.k, replace=False)]
    rw = TargetReward(args.target, lambda_unc=0.05,
                   weights={'potency': 1.0, 'selectivity': 0.0, 'qed': 0.3, 'sa': 0.3})
    ga = GraphGA(args.target, rw, seed=0)
    res = ga.run(seeds, budget=args.budget)
    valid = sum(Chem.MolFromSmiles(s) is not None for s in res['top_smiles'])
    print(f"GraphGA {args.target}: calls={res['oracle_calls']} best={res['best_reward']:.3f} "
          f"top10={res['top10_mean']:.3f} unique={res['n_unique']} top20_valid={valid}/20")
    for s in res['top_smiles'][:3]:
        print('   ', s)
