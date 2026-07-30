"""Pool-based lead optimization against REAL measured activity (Tier-3 validation).

Every other experiment in this study optimizes a model-defined reward, so a gain means
"the search found molecules the model likes". This experiment removes that circularity.
The pool is the target's real measured compounds. All labels are hidden except k=10
known actives. Each method then spends a fixed budget of 300 queries, and every query
reveals the compound's TRUE measured pIC50. Success is measured against the withheld
ground truth, not against a prediction:

  * hits found        : number of true top-1% compounds acquired
  * best true pIC50   : the strongest compound actually acquired
  * mean true pIC50   : quality of the acquired set
  * enrichment        : hit rate relative to random acquisition

Acquisition strategies compared (all share the same model class, budget and seeds):
  random      : uniform sampling (the null)
  greedy      : highest predicted potency (mean only)
  ucb         : mean + beta * sigma   (optimistic / exploratory)
  lcb         : mean - kappa * sigma  (conservative, reliability-aware)
  conformal   : highest conformal LOWER bound, mean - q_alpha * sigma, with q_alpha
                calibrated on held-out revealed labels each round

This directly tests whether accounting for model disagreement improves REAL measured
outcomes, and whether being conservative or optimistic pays under a fixed budget.
Writes poolopt_analysis.json.

    python -m reliability.run_poolopt_v1 --seeds 20
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reliability.oracle import TARGETS, featurize  # noqa: E402
from reliability.run_reliability_v2 import load_deduplicated  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'outputs', 'frozen', 'poolopt_analysis.json')
RF_KW = dict(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0)
BUDGET, BATCH, K_SEED = 300, 30, 10
METHODS = ['random', 'greedy', 'ucb', 'lcb', 'conformal']
BETA, KAPPA = 1.0, 1.0


def run_one(method, X, y, seed_idx, top_mask, budget, rng):
    """Acquire `budget` compounds from the pool; return acquired indices."""
    n = len(y)
    known = list(seed_idx)                      # labels revealed (seeds are free)
    acquired = []                               # queries that count against budget
    while len(acquired) < budget:
        take = min(BATCH, budget - len(acquired))
        pool = np.setdiff1d(np.arange(n), np.array(known, dtype=int), assume_unique=False)
        if len(pool) == 0:
            break
        if method == 'random':
            pick = rng.choice(pool, size=min(take, len(pool)), replace=False)
        else:
            rf = RandomForestRegressor(**RF_KW).fit(X[known], y[known])
            P = np.stack([t.predict(X[pool]) for t in rf.estimators_], 0)
            mu, sd = P.mean(0), P.std(0)
            if method == 'greedy':
                score = mu
            elif method == 'ucb':
                score = mu + BETA * sd
            elif method == 'lcb':
                score = mu - KAPPA * sd
            else:                                # conformal lower bound
                # calibrate on a held-out quarter of the revealed labels
                kn = np.array(known)
                perm = rng.permutation(len(kn))
                ncal = max(10, len(kn) // 4)
                cal, ptr = kn[perm[:ncal]], kn[perm[ncal:]]
                if len(ptr) < 5:
                    score = mu
                else:
                    rf2 = RandomForestRegressor(**RF_KW).fit(X[ptr], y[ptr])
                    Pc = np.stack([t.predict(X[cal]) for t in rf2.estimators_], 0)
                    mc, sc = Pc.mean(0), Pc.std(0)
                    s = np.abs(y[cal] - mc) / (sc + 1e-3)
                    m = len(s); kq = int(np.ceil((m + 1) * 0.9))
                    q = float(np.sort(s)[kq - 1]) if kq <= m else float(np.max(s))
                    score = mu - q * sd          # 90% conformal lower bound
            pick = pool[np.argsort(-score)[:take]]
        acquired.extend(pick.tolist()); known.extend(pick.tolist())
    return np.array(acquired[:budget], dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', nargs='+', default=['scd1', 'fads', 'nk1r', 'drd2', 'drd3'])
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    from scipy import stats as sps
    results, summary = {}, {}
    for tgt in args.targets:
        smiles, y, thr, dup = load_deduplicated(tgt)
        X = featurize(smiles); y = np.asarray(y, float)
        cutoff = float(np.quantile(y, 0.99))
        top_mask = y >= cutoff                    # TRUE top-1% by measured activity
        actives = np.where(y >= thr)[0]
        rows = []
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 + s)
            seed_idx = rng.choice(actives, size=min(K_SEED, len(actives)), replace=False)
            # seeds must not already be top-1%, else the task is partly solved
            seed_idx = np.array([i for i in seed_idx if not top_mask[i]])
            if len(seed_idx) < 3:
                continue
            row = {'n_seed': int(len(seed_idx))}
            for m in METHODS:
                acq = run_one(m, X, y, seed_idx, top_mask, BUDGET, np.random.default_rng(1000 + s))
                row[m] = dict(hits=int(top_mask[acq].sum()),
                              best=float(y[acq].max()),
                              mean=float(y[acq].mean()))
            rows.append(row)
        results[tgt] = rows
        agg = {}
        for m in METHODS:
            h = np.array([r[m]['hits'] for r in rows], float)
            b = np.array([r[m]['best'] for r in rows], float)
            mn = np.array([r[m]['mean'] for r in rows], float)
            agg[m] = dict(hits=float(h.mean()), hits_sd=float(h.std(ddof=1)),
                          best=float(b.mean()), mean=float(mn.mean()))
        base = agg['random']['hits']
        for m in METHODS:
            agg[m]['enrichment_vs_random'] = float(agg[m]['hits'] / base) if base > 0 else float('nan')
        for m in ['greedy', 'ucb', 'lcb', 'conformal']:
            a = np.array([r[m]['hits'] for r in rows], float)
            g = np.array([r['greedy']['hits'] for r in rows], float)
            if m != 'greedy':
                agg[m]['delta_vs_greedy'] = float((a - g).mean())
                agg[m]['p_vs_greedy'] = float(sps.ttest_rel(a, g)[1])
        agg['n_top1pct_in_pool'] = int(top_mask.sum()); agg['pool_size'] = int(len(y))
        agg['cutoff_pIC50'] = cutoff; agg['n_runs'] = len(rows)
        summary[tgt] = agg
        print(f"== {tgt} (pool {len(y)}, {int(top_mask.sum())} true top-1%): "
              + '  '.join(f"{m}={agg[m]['hits']:.1f}" for m in METHODS)
              + f"  | best true pIC50 greedy={agg['greedy']['best']:.2f} lcb={agg['lcb']['best']:.2f}",
              flush=True)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'config': vars(args), 'summary': summary, 'results': results}, f, indent=2)
    print('\n=== hits found (mean over seeds), enrichment vs random ===')
    for m in METHODS:
        e = [summary[t][m]['enrichment_vs_random'] for t in summary]
        print(f"  {m:10s} enrichment {np.mean(e):.2f}x   per-target {[round(x,1) for x in e]}")
    print('wrote', args.out)


if __name__ == '__main__':
    main()
