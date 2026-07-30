"""Multi-objective, uncertainty-aware reward for the reliability analysis.

R(x) = w_pot * potency_norm
     + w_sel * selectivity        (margin vs. specified anti-targets; opt-in)
     + w_qed * QED
     + w_sa  * sa_norm            (Ertl synthetic accessibility, inverted)
     - lambda_unc * epistemic_std (oracle ensemble std; penalizes OOD drift)

Potency comes from the frozen RF-on-ECFP oracle (reliability.oracle). The
epistemic-uncertainty penalty operationalizes the applicability-domain finding:
the generator is discouraged from drifting where the oracle is unreliable.

Invalid SMILES receive `invalid_reward`. An oracle-call counter supports PMO-style
budgeting. SMILES are featurized once and reused across potency / anti-targets /
uncertainty.

Smoke test:
    python -m reliability.reward --target drd3 --anti drd2
"""
from __future__ import annotations
import os, sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, RDConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # noqa: E402  (RDKit contrib Ertl SA score)
from reliability.oracle import (  # noqa: E402
    load_oracle, featurize, _ensemble_preds, TARGETS, load_target_data,
)

DEFAULT_WEIGHTS = {'potency': 1.0, 'selectivity': 0.0, 'qed': 0.3, 'sa': 0.3}


def _qed(mol):
    try:
        return float(QED.qed(mol))
    except Exception:
        return 0.0


def _sa_norm(mol):
    # Ertl SA in [1 (easy), 10 (hard)] -> [1 (easy), 0 (hard)]
    try:
        sa = float(sascorer.calculateScore(mol))
    except Exception:
        return 0.0
    return float(np.clip((10.0 - sa) / 9.0, 0.0, 1.0))


class TargetReward:
    def __init__(self, target, anti_targets=(), weights=None,
                 lambda_unc=0.0, invalid_reward=-1.0,
                 pic50_lo=4.0, pic50_hi=10.0, sel_scale=3.0,
                 novelty_anchors=None, w_novelty=0.0):
        assert target in TARGETS, target
        self.target = target
        self.oracle = load_oracle(target)
        self.model = self.oracle['model']
        self.anti = {t: load_oracle(t)['model'] for t in anti_targets}
        self.weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
        self.lambda_unc = float(lambda_unc)
        self.invalid_reward = float(invalid_reward)
        self.pic50_lo, self.pic50_hi = float(pic50_lo), float(pic50_hi)
        self.sel_scale = float(sel_scale)
        self.w_novelty = float(w_novelty)
        # ECFP of novelty anchors (the k few-shot seeds) for a novelty incentive
        self._A = featurize(list(novelty_anchors)) if novelty_anchors else None
        self._A_norm = self._A.sum(1) if self._A is not None else None
        self.n_oracle_calls = 0

    def _novelty(self, X):
        """1 - max Tanimoto(candidate, anchors), vectorized over ECFP bit arrays."""
        if self._A is None or self.w_novelty == 0:
            return np.zeros(len(X))
        inter = X @ self._A.T                      # [m, n_anchor]
        denom = X.sum(1)[:, None] + self._A_norm[None, :] - inter
        tani = np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)
        return 1.0 - tani.max(axis=1)

    def _norm_pic50(self, p):
        return np.clip((p - self.pic50_lo) / (self.pic50_hi - self.pic50_lo), 0.0, 1.0)

    def components(self, smiles):
        """Return per-molecule component dict (arrays aligned to `smiles`)."""
        if isinstance(smiles, str):
            smiles = [smiles]
        n = len(smiles)
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        valid = np.array([m is not None for m in mols])

        pot = np.full(n, np.nan); std = np.zeros(n)
        sel = np.zeros(n); qed = np.zeros(n); sa = np.zeros(n); nov = np.zeros(n)

        vidx = np.where(valid)[0]
        if len(vidx):
            vsmi = [smiles[i] for i in vidx]
            X = featurize(vsmi)
            self.n_oracle_calls += len(vsmi)
            if self.lambda_unc > 0:
                P = _ensemble_preds(self.model, X)  # [n_est, m]
                p_t = P.mean(axis=0); s_t = P.std(axis=0)
            else:
                p_t = self.model.predict(X); s_t = np.zeros(len(vidx))
            pot[vidx] = p_t; std[vidx] = s_t
            if self.anti:
                anti_p = np.max([a.predict(X) for a in self.anti.values()], axis=0)
                sel[vidx] = np.clip((p_t - anti_p) / self.sel_scale, 0.0, 1.0)
            nov[vidx] = self._novelty(X)
            for j, i in enumerate(vidx):
                qed[i] = _qed(mols[i]); sa[i] = _sa_norm(mols[i])

        return dict(valid=valid, pic50=pot, pic50_norm=self._norm_pic50(pot),
                    uncertainty=std, selectivity=sel, qed=qed, sa_norm=sa, novelty=nov)

    def __call__(self, smiles, return_components=False):
        scalar = isinstance(smiles, str)
        c = self.components(smiles)
        w = self.weights
        r = (w['potency'] * np.nan_to_num(c['pic50_norm'])
             + w['selectivity'] * c['selectivity']
             + w['qed'] * c['qed']
             + w['sa'] * c['sa_norm']
             + self.w_novelty * c['novelty']
             - self.lambda_unc * c['uncertainty'])
        r = np.where(c['valid'], r, self.invalid_reward)
        if return_components:
            return (float(r[0]) if scalar else r), c
        return float(r[0]) if scalar else r


if __name__ == '__main__':
    import argparse, pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='drd3')
    ap.add_argument('--anti', nargs='*', default=[])
    args = ap.parse_args()

    rw = TargetReward(args.target, anti_targets=args.anti, lambda_unc=0.1,
                   weights={'potency': 1.0, 'selectivity': 0.5 if args.anti else 0.0,
                            'qed': 0.3, 'sa': 0.3})
    smiles, acts, thr = load_target_data(args.target)
    order = np.argsort(acts)
    actives = [smiles[i] for i in order[-50:]]
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chembl = pd.read_csv(os.path.join(base, 'data', 'chembl_pretrain_smiles.csv'))
    col = 'SMILES' if 'SMILES' in chembl.columns else chembl.columns[0]
    rng = np.random.default_rng(0)
    decoys = [str(chembl[col].iloc[i]) for i in rng.choice(len(chembl), 50, replace=False)]

    ra, ca = rw(actives, return_components=True)
    rd, cd = rw(decoys, return_components=True)
    print(f"target={args.target} anti={args.anti} weights={rw.weights} lambda_unc={rw.lambda_unc}")
    print(f"  actives: reward {np.mean(ra):.3f}+/-{np.std(ra):.3f}  "
          f"pIC50 {np.nanmean(ca['pic50']):.2f}  QED {ca['qed'].mean():.2f}  "
          f"SAnorm {ca['sa_norm'].mean():.2f}  unc {ca['uncertainty'].mean():.3f}")
    print(f"  decoys : reward {np.mean(rd):.3f}+/-{np.std(rd):.3f}  "
          f"pIC50 {np.nanmean(cd['pic50']):.2f}  QED {cd['qed'].mean():.2f}  "
          f"SAnorm {cd['sa_norm'].mean():.2f}  unc {cd['uncertainty'].mean():.3f}")
    sep = np.mean(ra) - np.mean(rd)
    print(f"  separation (actives - decoys) = {sep:.3f}  "
          f"[{'OK' if sep > 0 else 'FAIL'}]   oracle_calls={rw.n_oracle_calls}")
