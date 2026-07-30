"""Actionable local-edit action space for the reliability analysis.

Actions are MMP-style R-group replacements realized through BRICS chemistry: a
molecule is cut at one BRICS bond, the smaller terminal fragment is swapped for a
compatible fragment drawn from a library built from known actives (+ optional
ChEMBL). Because every edit is a single synthesis-inspired substitution from real
chemistry, generated molecules stay reachable from known actives ("actionable"),
which keeps them inside the reward oracle's applicability domain.

`AdmissibilityFilter` scores how well a candidate connects back to the known
actives (nearest-anchor Tanimoto), yielding the transfer-degree and
isolated-node-ratio metrics used in the paper.

Smoke test:
    python -m reliability.actions --target drd3
"""
from __future__ import annotations
import os, sys
from collections import defaultdict
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import BRICS, AllChem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RDLogger.DisableLog('rdApp.*')


def _canon(mol):
    try:
        s = Chem.MolToSmiles(mol)
        return s if Chem.MolFromSmiles(s) is not None else None
    except Exception:
        return None


def _dummy_atoms(mol):
    return [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]


def _attach(frag_a, frag_b):
    """Join two single-dummy fragments at their dummy atoms; return canonical SMILES."""
    da = _dummy_atoms(frag_a); db = _dummy_atoms(frag_b)
    if len(da) != 1 or len(db) != 1:
        return None
    combo = Chem.CombineMols(frag_a, frag_b)
    rw = Chem.RWMol(combo)
    dums = _dummy_atoms(rw)
    if len(dums) != 2:
        return None
    a_dum, b_dum = dums
    a_nbr = a_dum.GetNeighbors(); b_nbr = b_dum.GetNeighbors()
    if not a_nbr or not b_nbr:
        return None
    ai, bi = a_nbr[0].GetIdx(), b_nbr[0].GetIdx()
    rw.AddBond(ai, bi, Chem.BondType.SINGLE)
    for idx in sorted([a_dum.GetIdx(), b_dum.GetIdx()], reverse=True):
        rw.RemoveAtom(idx)
    try:
        m = rw.GetMol(); Chem.SanitizeMol(m)
        return _canon(m)
    except Exception:
        return None


def _label_of(frag):
    """The single dummy's BRICS isotope label (int) of a one-dummy fragment, or None."""
    dums = _dummy_atoms(frag)
    if len(dums) != 1:
        return None
    return dums[0].GetIsotope()


class FragmentLibrary:
    """Terminal (single-attachment) BRICS fragments indexed by dummy label."""

    def __init__(self):
        self.by_label = defaultdict(set)

    def add_smiles(self, smiles_iter):
        for s in smiles_iter:
            mol = Chem.MolFromSmiles(str(s))
            if mol is None:
                continue
            bonds = list(BRICS.FindBRICSBonds(mol))
            for (a1, a2), (l1, l2) in bonds:
                frags = Chem.FragmentOnBonds(mol, [mol.GetBondBetweenAtoms(a1, a2).GetIdx()],
                                             dummyLabels=[(int(l1), int(l2))])
                pieces = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=False)
                for p in pieces:
                    if len(_dummy_atoms(p)) == 1 and p.GetNumHeavyAtoms() <= 18:
                        lbl = _label_of(p)
                        cs = _canon(p)
                        if lbl is not None and cs:
                            self.by_label[lbl].add(cs)
        return self

    def fragments(self, label):
        return list(self.by_label.get(label, ()))

    def size(self):
        return sum(len(v) for v in self.by_label.values())


class ActionSpace:
    """Enumerate local R-group-swap neighbors of a molecule."""

    def __init__(self, library: FragmentLibrary, max_neighbors=64, seed=0):
        self.lib = library
        self.max_neighbors = max_neighbors
        self.rng = np.random.default_rng(seed)

    def neighbors(self, smiles, k=None):
        k = k or self.max_neighbors
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return []
        seed_canon = Chem.MolToSmiles(mol)
        bonds = list(BRICS.FindBRICSBonds(mol))
        out = set()
        for (a1, a2), (l1, l2) in bonds:
            bidx = mol.GetBondBetweenAtoms(a1, a2).GetIdx()
            frags = Chem.FragmentOnBonds(mol, [bidx], dummyLabels=[(int(l1), int(l2))])
            pieces = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=False)
            if len(pieces) != 2:
                continue
            # keep the larger fragment, replace the smaller terminal one
            big, small = sorted(pieces, key=lambda p: p.GetNumHeavyAtoms(), reverse=True)
            if len(_dummy_atoms(big)) != 1 or len(_dummy_atoms(small)) != 1:
                continue
            removed_label = _label_of(small)
            replacements = self.lib.fragments(removed_label)
            self.rng.shuffle(replacements)
            for rsmi in replacements[:max(4, k // max(1, len(bonds)))]:
                rfrag = Chem.MolFromSmiles(rsmi)
                if rfrag is None:
                    continue
                cand = _attach(big, rfrag)
                if cand and cand != seed_canon:
                    out.add(cand)
                if len(out) >= k:
                    return list(out)
        return list(out)


class AdmissibilityFilter:
    """Connect candidates back to known actives via nearest-anchor Tanimoto."""

    def __init__(self, anchor_smiles, threshold=0.4, nbits=1024, radius=2):
        self.threshold = threshold
        self.nbits, self.radius = nbits, radius
        self.anchors = [self._fp(s) for s in anchor_smiles]
        self.anchors = [fp for fp in self.anchors if fp is not None]
        self._anchor_smiles = set(str(s) for s in anchor_smiles)

    def add_anchors(self, smiles_iter):
        """Grow the reachable region (MolWorld-style graph expansion): newly
        accepted high-reward molecules become anchors, so subsequent edits can
        walk away from the original seeds through chains of admissible steps."""
        for s in smiles_iter:
            s = str(s)
            if s in self._anchor_smiles:
                continue
            fp = self._fp(s)
            if fp is not None:
                self.anchors.append(fp)
                self._anchor_smiles.add(s)

    def _fp(self, smiles):
        m = Chem.MolFromSmiles(str(smiles))
        if m is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(m, self.radius, nBits=self.nbits)

    def max_similarity(self, smiles):
        fp = self._fp(smiles)
        if fp is None or not self.anchors:
            return 0.0
        return float(max(DataStructs.BulkTanimotoSimilarity(fp, self.anchors)))

    def transfer_degree(self, smiles):
        """Number of anchors within the similarity threshold (the MMP-degree analogue)."""
        fp = self._fp(smiles)
        if fp is None or not self.anchors:
            return 0
        sims = DataStructs.BulkTanimotoSimilarity(fp, self.anchors)
        return int(np.sum(np.asarray(sims) >= self.threshold))

    def is_admissible(self, smiles):
        return self.max_similarity(smiles) >= self.threshold


if __name__ == '__main__':
    import argparse
    from reliability.oracle import load_target_data
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='drd3')
    ap.add_argument('--lib_n', type=int, default=1500, help='actives used to build the library')
    args = ap.parse_args()

    smiles, acts, thr = load_target_data(args.target)
    order = np.argsort(acts)
    actives = [smiles[i] for i in order[-min(args.lib_n, len(smiles)):]]
    lib = FragmentLibrary().add_smiles(actives)
    print(f"target={args.target}: library {lib.size()} terminal fragments over "
          f"{len(lib.by_label)} BRICS labels, from {len(actives)} actives")

    space = ActionSpace(lib, max_neighbors=32, seed=0)
    adm = AdmissibilityFilter(actives, threshold=0.4)

    seed = actives[-1]
    nbrs = space.neighbors(seed, k=32)
    sims = [adm.max_similarity(n) for n in nbrs]
    valid = sum(Chem.MolFromSmiles(n) is not None for n in nbrs)
    print(f"seed: {seed}")
    print(f"  {len(nbrs)} neighbors, {valid} valid; mean nearest-active Tanimoto "
          f"{np.mean(sims):.3f} (local-edit check: should be high, <1)")
    print(f"  admissible (>=0.4): {sum(s >= 0.4 for s in sims)}/{len(nbrs)}; "
          f"seed transfer-degree={adm.transfer_degree(seed)}")
    for n in nbrs[:5]:
        print(f"    {n}   sim={adm.max_similarity(n):.2f}")
