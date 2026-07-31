#!/usr/bin/env python3
"""Response statistics of the Bemis-Murcko scaffold-split held-out fold.

Supplementary Table S2 reports a coefficient of determination on the scaffold split, and for
FADS it is -222.96. That reads as a software fault. It is not: R^2 normalises the residual sum
of squares by the held-out fold's OWN variance, and the FADS scaffold fold is nearly constant,
so the denominator is tiny while the absolute error is ordinary. The implied RMSE recovers that:

    R^2 = 1 - MSE / Var(y_test)   =>   RMSE = SD(y_test) * sqrt(1 - R^2)

This script records the fold size, the held-out and overall response spread, and the RMSE the
reported R^2 implies, so the table can say why the value is what it is instead of leaving a
reader to assume a bug. It refits nothing: the split is deterministic and the R^2 is read from
the frozen oracle_metrics.json.

Writes outputs/frozen/scaffold_fold_stats.json.
"""
from __future__ import annotations
import csv
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.chemistry import scaffold_split  # noqa: E402
from reliability.oracle import TARGETS  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'outputs', 'frozen', 'scaffold_fold_stats.json')
MET = os.path.join(BASE, 'outputs', 'frozen', 'oracle_metrics.json')


def main():
    om = {e['target']: e for e in json.load(open(MET))['results']}
    res = {}
    for t in sorted(TARGETS):
        path, _ = TARGETS[t]
        rows = list(csv.DictReader(open(path)))
        smi = [r['SMILES'] for r in rows]
        y = np.array([float(r['pIC50']) for r in rows], float)
        _, _, te = scaffold_split(smi, frac_train=0.8, frac_val=0.1)
        te = np.asarray(te, int)
        sd_all, sd_te = float(y.std(ddof=1)), float(y[te].std(ddof=1))
        r2 = om[t].get('scaffold', {}).get('r2')
        rmse = float(sd_te * np.sqrt(1.0 - r2)) if isinstance(r2, (int, float)) else None
        res[t] = dict(n_all=int(len(y)), n_test=int(len(te)),
                      sd_all=sd_all, sd_test=sd_te,
                      var_ratio=float((sd_all / sd_te) ** 2) if sd_te > 0 else None,
                      r2=r2, implied_rmse=rmse)
        print('  %-6s n_test %4d  SD(test) %.3f  SD(all) %.3f  ratio %7.1f  R2 %9.2f  RMSE %.2f'
              % (t, len(te), sd_te, sd_all, res[t]['var_ratio'], r2, rmse))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
