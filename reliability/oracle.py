"""Shared oracle API for the Chemistry World Model reward.

Defines the 5 generation targets, ECFP4 featurization, and a clean
load / predict interface with epistemic uncertainty (ensemble std). The
generation reward uses `R = mean - lambda * std` so the generator is penalized
for drifting where the oracle is uncertain (outside its applicability domain).

Oracles are built and frozen by `oracle_sanity_gate.py` into
outputs/frozen/oracle_{target}.pkl.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_pipeline_v3 import load_smiles_activity, smi_to_morgan  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'frozen')

# target -> (data path, active/inactive pIC50 threshold). Thresholds follow the
# v4 pipeline conventions; DRD3 uses 7.0 (its distribution is shifted high).
_D = os.path.join(BASE_DIR, 'data')
TARGETS = {
    'scd1': (os.path.join(_D, 'scd1_binding.csv')
             if os.path.exists(os.path.join(_D, 'scd1_binding.csv'))
             else os.path.normpath(os.path.join(BASE_DIR, '..', '..',
                  'SCD1 dataset filtered 20230412', 'scd1_binding.csv')), 7.0),
    'fads': (os.path.join(_D, 'fatty_acid_desaturase_bioactivity.csv'), 6.5),
    'nk1r': (os.path.join(_D, 'nk1r_combined.csv'), 7.0),
    'drd2': (os.path.join(_D, 'drd2_bioactivity.csv'), 6.5),
    'drd3': (os.path.join(_D, 'drd3_chembl.csv'), 7.0),
}

# class A GPCRs with tractable structures -> eligible for the V-E docking check.
GPCR_DOCKING = ('nk1r', 'drd2', 'drd3')


def load_target_data(target):
    path, thr = TARGETS[target]
    smiles, acts = load_smiles_activity(path)
    return smiles, np.asarray(acts, dtype=float), thr


def featurize(smiles_list):
    """1024-bit ECFP4 (radius 2). Invalid SMILES -> zero vector."""
    X = []
    for s in smiles_list:
        fp = smi_to_morgan(s)
        X.append(fp if fp is not None else np.zeros(1024, dtype=np.float32))
    return np.asarray(X, dtype=np.float32)


def load_oracle(target):
    """Return the frozen oracle dict: {model, model_name, n_bits, radius, threshold}."""
    import joblib
    return joblib.load(os.path.join(OUT_DIR, f'oracle_{target}.pkl'))


def _ensemble_preds(model, X):
    """Per-estimator predictions [n_estimators, N] for RF / ExtraTrees."""
    est = getattr(model, 'estimators_', None)
    if est is None:
        return model.predict(X)[None, :]
    return np.stack([t.predict(X) for t in est], axis=0)


def predict_pic50(oracle, smiles_list, return_std=False):
    """Predict pIC50 (and optional epistemic std) for a list of SMILES.

    `oracle` may be a target name (str) or a loaded oracle dict.
    """
    if isinstance(oracle, str):
        oracle = load_oracle(oracle)
    model = oracle['model']
    X = featurize(smiles_list)
    if return_std:
        P = _ensemble_preds(model, X)
        return P.mean(axis=0), P.std(axis=0)
    return model.predict(X)
