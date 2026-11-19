"""Rebuild the NK1R examples from the archived nk1r_combined.csv (release v1.8).

Run with RDKit 2026.03.1 and scikit-learn 1.8.0. Selection precedes model fitting
and uses only chemical structures once the 10 active training leads have been drawn.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / 'source_data' / 'nk1r_combined.csv'
EXPECTED_SHA256 = '4a682b39118bdb41b894bd54daaf216f875733e24240aaacbf2cc9d7ee04a9c0'
COHORT_FILE = HERE.parent / 'source_data' / 'reliability_v2_analysis.json'
COHORT_SHA256 = 'd0f11ac18e66dce801940ad2dd58b8ea5b9c5c4be48782d4c6f4d0a3fd6ed8b4'


def read_parents(path):
    normalizer = rdMolStandardize.Normalizer()
    chooser = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()
    records = []
    for entry in pd.read_csv(path).itertuples():
        mol = Chem.MolFromSmiles(entry.SMILES)
        if mol is None:
            continue
        parent = uncharger.uncharge(chooser.choose(normalizer.normalize(mol)))
        Chem.SanitizeMol(parent)
        key = Chem.InchiToInchiKey(Chem.MolToInchi(parent))
        records.append((key, Chem.MolToSmiles(parent), entry.pIC50))
    rows = pd.DataFrame(records, columns=['key', 'smiles', 'pActivity'])
    return rows.groupby('key', sort=True).agg({'smiles': 'first', 'pActivity': 'median'})


def nearest_training(index, references, all_fps):
    """Break exact Tanimoto ties by lexicographic standardized parent InChIKey."""
    references = np.sort(references)
    ref_fps = [all_fps[i] for i in references]
    similarities = np.array([DataStructs.BulkTanimotoSimilarity(all_fps[i], ref_fps)
                             for i in index])
    positions = similarities.argmax(axis=1)
    return references[positions], similarities[np.arange(len(index)), positions]


def select_examples(frame):
    """Select two examples without using either their measurements or their predictions."""
    lead_threshold = frame.lead_sim.quantile(1 / 3)
    novel = frame[frame.lead_sim <= lead_threshold]
    low, midpoint, high = novel.train_sim.quantile([0.25, 0.5, 0.75])
    readable = novel[novel.drawable]
    near = readable[readable.train_sim > midpoint]
    far = readable[readable.train_sim < midpoint]
    eligible_pairs = []
    for near_id, a in near.iterrows():
        subset = far[(far.lead_id == a.lead_id) & ((far.lead_sim - a.lead_sim).abs() <= 0.005)]
        for far_id, b in subset.iterrows():
            # The two on-page training neighbors must be distinct from the common lead.
            if len({int(a.lead_id), int(a.train_id), int(b.train_id)}) != 3:
                continue
            quantile_distance = abs(a.train_sim - high) + abs(b.train_sim - low)
            eligible_pairs.append((quantile_distance, abs(b.lead_sim - a.lead_sim), near_id, far_id))
    if not eligible_pairs:
        raise RuntimeError('No pair meets the prespecified structural criteria.')
    _, _, near_id, far_id = min(eligible_pairs)
    return [near_id, far_id], {
        'max_lead_similarity_upper_novelty_third': float(lead_threshold),
        'train_similarity_quartiles_in_upper_novelty_third': [float(v) for v in (low, midpoint, high)],
        'n_heldout_upper_novelty_third': len(novel),
        'n_structurally_eligible_pairs': len(eligible_pairs),
        'max_difference_in_lead_similarity': 0.005,
        'max_heavy_atoms_in_any_depicted_structure': 45,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--cohort-results', type=Path, default=COHORT_FILE,
                        help='Frozen reliability_v2_analysis.json from the v1.8 release')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--out', type=Path, default=HERE / 'figure1_examples.json')
    args = parser.parse_args()
    digest = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError('Source file differs from the v1.8 submission-release CSV.')
    raw_summary = args.cohort_results.read_bytes()
    if hashlib.sha256(raw_summary).hexdigest() != COHORT_SHA256:
        raise ValueError('Cohort results differ from the v1.8 release.')
    nk1r = json.loads(raw_summary)['nk1r']
    RDLogger.DisableLog('rdApp.warning')
    parents = read_parents(args.csv)
    assert len(parents) == 2952
    mols = [Chem.MolFromSmiles(s) for s in parents.smiles]
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fps = [fpgen.GetFingerprint(m) for m in mols]
    y = parents.pActivity.to_numpy()
    train, test = next(KFold(5, shuffle=True, random_state=0).split(parents))
    active = train[y[train] >= 7.0]
    leads = active[np.random.default_rng(0).choice(len(active), size=10, replace=False)]
    train_id, train_sim = nearest_training(test, train, fps)
    lead_id, lead_sim = nearest_training(test, leads, fps)
    n_heavy = np.array([m.GetNumHeavyAtoms() for m in mols])
    frame = pd.DataFrame({
        'train_id': train_id, 'train_sim': train_sim, 'lead_id': lead_id, 'lead_sim': lead_sim,
        'drawable': np.maximum.reduce([n_heavy[test], n_heavy[train_id], n_heavy[lead_id]]) <= 45,
    }, index=test)
    selected, selection = select_examples(frame)
    assert set(train).isdisjoint(test) and set(leads).issubset(train)
    assert set(selected).issubset(test)
    x = np.asarray([np.asarray(fp, dtype=np.float32) for fp in fps])
    forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=args.jobs, random_state=0)
    forest.fit(x[train], y[train])
    tree_values = np.stack([tree.predict(x[selected]) for tree in forest.estimators_])

    def structure(i):
        return {'parent_key': parents.index[i], 'smiles': parents.smiles.iloc[i],
                'measured_pActivity': float(y[i])}

    examples = []
    for j, i in enumerate(selected):
        item = frame.loc[i]
        pred = tree_values[:, j].mean()
        examples.append({
            'group': ['nearer', 'farther'][j], 'query': structure(i),
            'predicted_pActivity': float(pred), 'absolute_error': float(abs(y[i] - pred)),
            'tree_sd': float(tree_values[:, j].std()),
            'nearest_lead': structure(int(item.lead_id)), 'lead_similarity': float(item.lead_sim),
            'nearest_training': structure(int(item.train_id)), 'training_similarity': float(item.train_sim),
        })
    out = {
        'target': 'NK1R', 'source': 'nk1r_combined.csv', 'source_sha256': digest,
        'source_release_doi': '10.5281/zenodo.21938599',
        'rdkit': rdkit.__version__, 'sklearn': sklearn.__version__, 'numpy': np.__version__,
        'n_parent_structures': len(parents), 'fold_one_based': 1, 'n_training': len(train),
        'n_held_out': len(test), 'n_starting_leads': len(leads),
        'starting_lead_parent_keys': parents.index[leads].tolist(), 'selection': selection,
        'examples': examples,
        'original_cohort_summary': {
            'source': 'reliability_v2_analysis.json', 'source_sha256': COHORT_SHA256,
            'nearer_n': nk1r['n_novel_in_domain'], 'farther_n': nk1r['n_novel_out_domain'],
            'nearer_rmse': nk1r['novel_in_domain_rmse'], 'farther_rmse': nk1r['novel_out_domain_rmse'],
        },
    }
    args.out.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
