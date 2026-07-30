"""Latent surrogate for the reliability analysis.

The surrogate M_phi predicts a candidate's reward from the Part-1 dual-encoder
latent (R^128) WITHOUT calling the real oracle, enabling cheap imagined rollouts.
It is an ensemble of small MLPs trained online on (latent, real_reward) pairs
gathered during search; ensemble disagreement gives epistemic uncertainty for an
acquisition function (UCB) in the planner.

Why the dual-encoder latent (not ECFP): the reward oracle is RF-on-ECFP, so a
surrogate on ECFP could trivially memorize it (circular). The latent is a genuinely
different representation, so M_phi is an honest approximation and the
sample-efficiency claim (imagination reduces oracle calls) is meaningful. The
latent also places the surrogate in the Dreamer / JEPA / folding-WM lineage.

Smoke test (does M_phi approximate the reward landscape from few evaluations?):
    python -m reliability.dynamics --target drd3
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dual_encoder import DualEncoder  # noqa: E402
from data.dataset import DualDataset, collate_dual  # noqa: E402
from run_moe_predictor import _GraphBatch  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODER_CKPT = os.path.join(BASE_DIR, 'outputs', 'encoder', 'dual_encoder_clean.pt')


class LatentEmbedder:
    """Frozen Part-1 dual encoder; maps SMILES -> R^128 (loaded once, reused)."""

    def __init__(self, device='cuda', weights_path=ENCODER_CKPT):
        self.device = device
        self.encoder = DualEncoder(
            node_features=75, edge_features=6, graph_hidden_dims=[128, 256, 128],
            vocab_size=128, smiles_hidden_dim=256, output_dim=128, num_heads=4,
            dropout=0.1, fusion_type='attention').to(device)
        self.encoder.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        self.encoder.eval()

    def embed(self, smiles_list, bs=128):
        smiles_list = list(smiles_list)
        ds = DualDataset(smiles_list, np.zeros(len(smiles_list), dtype=np.float32), max_length=120)
        loader = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_dual)
        out = []
        with torch.no_grad():
            for batch in loader:
                gb = _GraphBatch(batch['graph']).to(self.device)
                tok = batch['tokens'].to(self.device); ln = batch['lengths']
                out.append(self.encoder(gb, tok, ln).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


class _RewardMLP(nn.Module):
    def __init__(self, d=128, h=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LatentSurrogate:
    """Ensemble reward surrogate over the dual-encoder latent, with online updates."""

    def __init__(self, embedder: LatentEmbedder, n_ensemble=5, hidden=128,
                 device='cuda', seed=0):
        self.embedder = embedder
        self.device = device
        self.n_ensemble = n_ensemble
        torch.manual_seed(seed)
        self.members = [_RewardMLP(128, hidden).to(device) for _ in range(n_ensemble)]
        self._Z = np.zeros((0, 128), dtype=np.float32)
        self._y = np.zeros((0,), dtype=np.float32)
        self._ymean, self._ystd = 0.0, 1.0
        self.fitted = False
        self.rng = np.random.default_rng(seed)

    def add(self, smiles_list, rewards):
        Z = self.embedder.embed(smiles_list)
        self._Z = np.concatenate([self._Z, Z], axis=0)
        self._y = np.concatenate([self._y, np.asarray(rewards, dtype=np.float32)])

    def fit(self, epochs=300, lr=1e-3, weight_decay=1e-4, bootstrap=True):
        if len(self._y) < 4:
            return self
        self._ymean, self._ystd = float(self._y.mean()), float(self._y.std() + 1e-6)
        yt = (self._y - self._ymean) / self._ystd
        Zt = torch.tensor(self._Z, device=self.device)
        ytt = torch.tensor(yt, device=self.device)
        n = len(yt)
        for m in self.members:
            idx = (self.rng.integers(0, n, size=n) if bootstrap else np.arange(n))
            xb = Zt[idx]; yb = ytt[idx]
            opt = Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
            m.train()
            for _ in range(epochs):
                opt.zero_grad()
                loss = nn.functional.mse_loss(m(xb), yb)
                loss.backward(); opt.step()
            m.eval()
        self.fitted = True
        return self

    def update(self, smiles_list, rewards, **fit_kw):
        self.add(smiles_list, rewards)
        return self.fit(**fit_kw)

    def predict(self, smiles_list, return_std=True):
        Z = torch.tensor(self.embedder.embed(smiles_list), device=self.device)
        with torch.no_grad():
            P = np.stack([(m(Z).cpu().numpy() * self._ystd + self._ymean)
                          for m in self.members], axis=0)
        mean = P.mean(axis=0)
        if return_std:
            return mean, P.std(axis=0)
        return mean


if __name__ == '__main__':
    import argparse
    import pandas as pd
    from scipy.stats import spearmanr
    from reliability.reward import TargetReward
    from reliability.oracle import load_target_data

    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='drd3')
    ap.add_argument('--n_eval', type=int, default=200, help='molecules the WM is trained on')
    ap.add_argument('--n_test', type=int, default=200)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rw = TargetReward(args.target, lambda_unc=0.0)
    smiles, acts, thr = load_target_data(args.target)
    chembl = pd.read_csv(os.path.join(BASE_DIR, 'data', 'chembl_pretrain_smiles.csv'))
    col = 'SMILES' if 'SMILES' in chembl.columns else chembl.columns[0]
    rng = np.random.default_rng(0)
    # pool = mix of actives and random ChEMBL so reward has spread
    pool = list(rng.choice(smiles, min(len(smiles), 300), replace=False)) + \
        [str(chembl[col].iloc[i]) for i in rng.choice(len(chembl), 300, replace=False)]
    pool = [s for s in pool if __import__('rdkit').Chem.MolFromSmiles(s) is not None]
    rng.shuffle(pool)
    rewards = np.asarray(rw(pool))
    tr, teidx = pool[:args.n_eval], pool[args.n_eval:args.n_eval + args.n_test]
    ytr, yte = rewards[:args.n_eval], rewards[args.n_eval:args.n_eval + args.n_test]

    emb = LatentEmbedder(device=device)
    wm = LatentSurrogate(emb, n_ensemble=5, device=device, seed=0)
    wm.update(tr, ytr, epochs=300)
    pmean, pstd = wm.predict(teidx)

    err = np.abs(pmean - yte)
    print(f"target={args.target}  trained on {len(tr)} evals, tested on {len(teidx)}")
    print(f"  reward-surrogate Spearman(pred, true) = {spearmanr(pmean, yte).correlation:.3f}")
    print(f"  reward-surrogate R2 = {1 - np.sum((yte-pmean)**2)/np.sum((yte-yte.mean())**2):.3f}")
    print(f"  mean |err| = {err.mean():.3f}; uncertainty-error corr = "
          f"{spearmanr(pstd, err).correlation:.3f} (should be >0: high unc -> high err)")
