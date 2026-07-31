#!/usr/bin/env python3
"""Why the activity model is a tree ensemble and not a small neural network.

The Results and the Methods both state that small neural predictors collapse toward the
training mean and cannot serve as a reward, quoting a spread across held-out compounds. That
number was carried over from an earlier round of the work and was not reproducible from
anything in this repository: the model bake-off in oracle_sanity_gate.py compares three tree
ensembles and never fits a network, so nothing here could confirm or refute it.

This script measures it. For each target it fits the same kind of small multilayer perceptron
the Methods describe for the latent surrogate, two hidden layers of width 128 with rectified
linear units, on the same 1024-bit ECFP4 features, under the same five-fold split the activity
model uses, and records the standard deviation of the out-of-fold predictions. A model that has
collapsed to the training mean predicts nearly the same value everywhere, so that spread is
small against the spread of the measured activities it is trying to reproduce.

Writes outputs/frozen/mlp_collapse.json.

    python -m reliability.mlp_collapse_check
"""
from __future__ import annotations
import json
import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability.oracle import TARGETS, featurize  # noqa: E402
from reliability.standardize import load_standardized  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'outputs', 'frozen', 'mlp_collapse.json')


def main():
    res = {}
    for t in sorted(TARGETS):
        smiles, y, _thr, _st = load_standardized(t)
        X = featurize(smiles)
        y = np.asarray(y, float)
        oof_mlp = np.zeros_like(y)
        oof_rf = np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            sc = StandardScaler().fit(X[tr])
            m = MLPRegressor(hidden_layer_sizes=(128, 128), activation='relu',
                             learning_rate_init=1e-3, alpha=1e-4, max_iter=200,
                             random_state=0)
            m.fit(sc.transform(X[tr]), y[tr])
            oof_mlp[te] = m.predict(sc.transform(X[te]))
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                       n_jobs=-1, random_state=0).fit(X[tr], y[tr])
            oof_rf[te] = rf.predict(X[te])
        res[t] = dict(n=int(len(y)),
                      sd_measured=float(y.std(ddof=1)),
                      sd_mlp_pred=float(oof_mlp.std(ddof=1)),
                      sd_rf_pred=float(oof_rf.std(ddof=1)),
                      mlp_spread_ratio=float(oof_mlp.std(ddof=1) / y.std(ddof=1)),
                      rf_spread_ratio=float(oof_rf.std(ddof=1) / y.std(ddof=1)))
        print('  %-6s measured SD %.2f | MLP prediction SD %.2f (%.0f%% of it) | '
              'RF %.2f (%.0f%%)'
              % (t, res[t]['sd_measured'], res[t]['sd_mlp_pred'],
                 100 * res[t]['mlp_spread_ratio'], res[t]['sd_rf_pred'],
                 100 * res[t]['rf_spread_ratio']))
    res['max_mlp_pred_sd'] = max(v['sd_mlp_pred'] for v in res.values()
                                 if isinstance(v, dict))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
