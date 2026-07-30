"""Structure standardization and grouping (single source of truth).

Canonical SMILES treat salts, charge states and some tautomer variants of the same
compound as distinct structures. Grouping folds on canonical SMILES therefore leaves
residual leakage: measured on the curated files, 104 NK1R, 47 DRD2 and 33 DRD3 records
are duplicate parents that canonical-SMILES grouping does not catch (SCD-1 has 136 that
it does catch). All splitting and deduplication in this study group on the standardized
parent InChIKey produced here.

The parent is the largest organic fragment after normalization, with charges neutralized.

    from reliability.standardize import parent_inchikey, load_standardized
"""
from __future__ import annotations
import functools
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')

_norm = rdMolStandardize.Normalizer()
_lfc = rdMolStandardize.LargestFragmentChooser()
_unc = rdMolStandardize.Uncharger()


@functools.lru_cache(maxsize=200000)
def parent_inchikey(smiles):
    """Standardized-parent InChIKey, or None if the structure cannot be processed."""
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    try:
        m = _unc.uncharge(_lfc.choose(_norm.normalize(m)))
        Chem.SanitizeMol(m)
        return Chem.InchiToInchiKey(Chem.MolToInchi(m))
    except Exception:
        return None


@functools.lru_cache(maxsize=200000)
def parent_smiles(smiles):
    """Canonical SMILES of the standardized parent, or None."""
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    try:
        m = _unc.uncharge(_lfc.choose(_norm.normalize(m)))
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def load_standardized(target):
    """Load a curated target file and collapse to one row per standardized parent.

    Returns (smiles, y, threshold, stats) where `smiles` are parent SMILES and `y` the
    median activity of every record sharing that parent InChIKey.
    """
    from reliability.oracle import TARGETS
    path, thr = TARGETS[target]
    df = pd.read_csv(path)
    keys, smis = [], []
    for s in df['SMILES']:
        keys.append(parent_inchikey(s))
        smis.append(parent_smiles(s))
    df = df.assign(_key=keys, _smi=smis).dropna(subset=['_key', '_smi'])
    n_rows = len(df)
    g = df.groupby('_key')
    agg = g['pIC50'].median()
    rep = g['pIC50'].std().dropna()
    first_smi = g['_smi'].first()
    stats = dict(
        n_rows_parsed=int(n_rows),
        n_rows_raw=int(len(keys)),
        n_unique_parents=int(len(agg)),
        n_duplicate_rows=int(n_rows - len(agg)),
        n_compounds_with_replicates=int(len(rep)),
        mean_within_compound_sd=float(rep.mean()) if len(rep) else 0.0,
        max_replicates=int(g.size().max()),
    )
    keys_sorted = list(agg.index)
    return ([first_smi[k] for k in keys_sorted],
            agg.loc[keys_sorted].values.astype(float), thr, stats)


if __name__ == '__main__':
    from reliability.oracle import TARGETS
    print(f"{'target':6s} {'rows':>6s} {'parents':>8s} {'dup':>5s} {'w/repl':>7s} {'SD':>5s}")
    tot = 0
    for t in TARGETS:
        s, y, thr, st = load_standardized(t)
        tot += st['n_unique_parents']
        print(f"{t:6s} {st['n_rows_raw']:6d} {st['n_unique_parents']:8d} "
              f"{st['n_duplicate_rows']:5d} {st['n_compounds_with_replicates']:7d} "
              f"{st['mean_within_compound_sd']:5.2f}")
    print(f"{'TOTAL':6s} {'':6s} {tot:8d}")
