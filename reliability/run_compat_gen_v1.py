"""Compatibility-governed imagination for generation (Section V-D, secondary).

Does the SAME source-target compatibility metric C_nn (from the prediction paper)
that predicts when few-shot transfer helps PREDICTION also predict when warm-starting
the surrogate on a source target helps few-shot GENERATION on a target?

For each ordered pair (source S -> target T) among the 5 generation targets:
  warm : pre-load the WM buffer with (source molecules, source-oracle reward), fit,
         then run WM-guided GA on T (tight budget so the prior matters).
  cold : identical WM-guided GA on T with a fresh WM (no source).
  gain(S->T, seed) = top10(warm) - top10(cold).
Regress gain on C_nn(S,T) (reused from outputs/compat_v8_16tgt.json).

Reward is potency+QED+SA (no selectivity) for all targets, for clean cross-target
comparison. Tight budget=150.

    python -m reliability.run_compat_gen_v1 --targets scd1 fads nk1r drd2 drd3 --seeds 15
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability.oracle import load_target_data, load_oracle, TARGETS  # noqa: E402
from reliability.reward import TargetReward  # noqa: E402
from reliability.dynamics import LatentEmbedder, LatentSurrogate  # noqa: E402
from reliability.surrogate_ga import SurrogateTriagedGA  # noqa: E402
from reliability.run_fewshot_v1 import OUT_DIR  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET = 150
K = 10
N_SOURCE = 400
WEIGHTS = {'potency': 1.0, 'selectivity': 0.0, 'qed': 0.3, 'sa': 0.3}


def load_cnn():
    d = json.load(open(os.path.join(BASE, 'outputs', 'frozen', 'compat_v8_16tgt.json')))
    return {(r['source'], r['target']): r['C_nn'] for r in d['rows']}


def _seeds_for(target, k, seed):
    smiles, acts, thr = load_target_data(target)
    actives = [s for s, a in zip(smiles, acts) if a >= thr]
    rng = np.random.default_rng(1000 * seed + k)
    return [actives[i] for i in rng.choice(len(actives), min(k, len(actives)), replace=False)]


def _source_pretrained_wm(embedder, source, seed):
    """WM buffer pre-loaded with (source molecules, source-oracle reward), fitted."""
    smiles, acts, thr = load_target_data(source)
    rng = np.random.default_rng(7 * seed + 13)
    idx = rng.choice(len(smiles), min(N_SOURCE, len(smiles)), replace=False)
    sample = [smiles[i] for i in idx]
    rw_s = TargetReward(source, lambda_unc=0.0, weights=WEIGHTS)
    rewards = np.asarray(rw_s(sample))
    wm = LatentSurrogate(embedder, n_ensemble=5, device='cuda', seed=seed)
    wm.add(sample, rewards); wm.fit(epochs=250)
    return wm


def _run_T(target, seeds, wm):
    rw = TargetReward(target, lambda_unc=0.1, weights=WEIGHTS)
    ga = SurrogateTriagedGA(target, rw, surrogate=wm, mode='surrogate', seed=hash(tuple(seeds)) % (2**31))
    return ga.run(seeds, budget=BUDGET)['top10_mean']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['scd1', 'fads', 'nk1r', 'drd2', 'drd3'])
    ap.add_argument('--seeds', type=int, default=15)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'compat_gen_results.json'))
    args = ap.parse_args()

    cnn = load_cnn()
    embedder = LatentEmbedder(device='cuda')
    results = []
    for T in args.targets:
        # cold baseline per (T, seed), reused across all sources
        cold = []
        for s in range(args.seeds):
            seeds = _seeds_for(T, K, s)
            wm = LatentSurrogate(embedder, n_ensemble=5, device='cuda', seed=s)
            cold.append(_run_T(T, seeds, wm))
        cold = np.array(cold)
        for S in args.targets:
            if S == T:
                continue
            gains = []
            for s in range(args.seeds):
                seeds = _seeds_for(T, K, s)
                wm = _source_pretrained_wm(embedder, S, s)
                warm = _run_T(T, seeds, wm)
                gains.append(warm - cold[s])
            gains = np.array(gains)
            row = dict(source=S, target=T, C_nn=cnn.get((S, T)),
                       n_target=len(load_target_data(T)[0]),
                       gain_mean=float(gains.mean()), gain_std=float(gains.std()),
                       cold_top10=float(cold.mean()), per_seed_gain=gains.tolist())
            results.append(row)
            print(f"== {S}->{T}: C_nn={row['C_nn']:.3f} gain={row['gain_mean']:+.4f}"
                  f"±{row['gain_std']:.3f}", flush=True)
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump({'config': vars(args), 'budget': BUDGET, 'k': K, 'results': results}, f, indent=2)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
