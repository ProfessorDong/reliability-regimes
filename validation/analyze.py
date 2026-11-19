"""Compute the prespecified target-level validation from the archived parent tables."""

import argparse
import csv
import json
import platform
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rdkit
import scipy
import sklearn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from prepare import ROOT, TARGETS

DRAWS = 100


def ranked_residual(x, controls):
    design = np.column_stack([np.ones(len(x)), *controls])
    return x - design @ np.linalg.lstsq(design, x, rcond=None)[0]


def partial(x, y, controls):
    return float(np.corrcoef(ranked_residual(x, controls), ranked_residual(y, controls))[0, 1])


def paired_strata(test, novelty, distance, near_weight, far_weight):
    """Accumulate weights with the same fold/novelty composition in both groups."""
    eligible = test[novelty >= np.quantile(novelty, 2 / 3)]
    eligible_novelty = novelty[novelty >= np.quantile(novelty, 2 / 3)]
    edges = np.unique(np.quantile(eligible_novelty, [0.2, 0.4, 0.6, 0.8]))
    groups = np.searchsorted(edges, eligible_novelty, side="left")
    used = 0
    for category in np.unique(groups):
        members = eligible[groups == category]
        cut = np.median(distance[members])
        near = members[distance[members] <= cut]
        far = members[distance[members] > cut]
        if not len(far) or not len(near):
            continue
        near_weight[near] += len(members) / len(near)
        far_weight[far] += len(members) / len(far)
        used += len(members)
    return len(eligible), used


def clustered_interval(scaffolds, squared_error, near_weight, far_weight):
    _, cluster = np.unique(scaffolds, return_inverse=True)
    n = int(cluster.max()) + 1
    groups = np.vstack(
        [
            np.bincount(cluster, weights=w)
            for w in (near_weight, far_weight, near_weight * squared_error, far_weight * squared_error)
        ]
    )
    # Each resample includes whole scaffolds. The membership weights integrate all lead draws.
    rng = np.random.default_rng(60641)
    counts = rng.multinomial(n, np.ones(n) / n, size=2000)
    nw, fw, ne, fe = groups @ counts.T
    keep = (nw > 0) & (fw > 0)
    delta = np.sqrt(fe[keep] / fw[keep]) - np.sqrt(ne[keep] / nw[keep])
    return (n, np.count_nonzero(keep), *np.quantile(delta, [0.025, 0.975]))


def scaffold_label(mol):
    # Delete stereochemistry before extracting the achiral scaffold. Stereo bonds
    # that lose a defining substituent in the scaffold can otherwise be invalid.
    achiral = Chem.Mol(mol)
    Chem.RemoveStereochemistry(achiral)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=achiral, includeChirality=False)


def run_target(args):
    target, workers = args
    RDLogger.DisableLog("rdApp.*")
    table = pd.read_csv(ROOT / "parents" / f"{target}.csv")
    assert table.parent_key.is_unique and table.parent_key.is_monotonic_increasing
    y = table.pActivity.to_numpy()
    assert len(y) <= 1200 and np.isfinite(y).all()
    mols = [Chem.MolFromSmiles(s) for s in table.smiles]
    assert all(Chem.MolToInchiKey(m) == k for m, k in zip(mols, table.parent_key))
    scaf = [scaffold_label(m) for m in mols]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=False)
    fps = [gen.GetFingerprint(m) for m in mols]
    x = np.asarray(fps, dtype=np.float32)
    similarity = np.stack([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])
    pred, sig, dist = [np.full(len(y), np.nan) for _ in range(3)]
    novelty = np.full((len(y), DRAWS), np.nan)
    fold_id = np.full(len(y), -1, dtype=int)
    n_actives = []
    for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=0).split(x)):
        assert np.intersect1d(train, test).size == 0 and np.all(fold_id[test] == -1)
        fold_id[test] = fold
        forest = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=1.0,
            bootstrap=True,
            criterion="squared_error",
            random_state=0,
            n_jobs=workers,
        ).fit(x[train], y[train])
        member_pred = np.stack([tree.predict(x[test]) for tree in forest.estimators_])
        pred[test], sig[test] = member_pred.mean(axis=0), member_pred.std(axis=0)
        dist[test] = 1 - similarity[np.ix_(test, train)].max(axis=1)
        active = train[y[train] >= 7]
        n_actives.append(len(active))
        if len(active):
            for draw in range(DRAWS):
                rng = np.random.default_rng(np.random.SeedSequence([20261119, fold, draw]))
                leads = rng.choice(active, size=min(10, len(active)), replace=False)
                novelty[test, draw] = 1 - similarity[np.ix_(test, leads)].max(axis=1)
    assert (
        np.all(fold_id >= 0)
        and np.isfinite(pred).all()
        and np.isfinite(sig).all()
        and np.isfinite(dist).all()
    )
    with_leads = np.isfinite(novelty[:, 0])
    assert np.array_equal(with_leads, np.isfinite(novelty).all(axis=1))
    assert (novelty[with_leads] >= dist[with_leads, None] - 1e-12).all()
    e = np.abs(pred - y)
    e2 = e**2
    near, far = np.zeros(len(y)), np.zeros(len(y))
    near_novelty_sum = far_novelty_sum = eligible_count = used_count = 0
    e_rank = stats.rankdata(e[with_leads])
    dist_rank = stats.rankdata(dist[with_leads])
    sig_rank = stats.rankdata(sig[with_leads])
    rho_novelty, partial_novelty, partial_sigma, original_delta = [], [], [], []
    for draw in range(DRAWS):
        n = novelty[:, draw]
        wn, wf = np.zeros(len(y)), np.zeros(len(y))
        for fold in range(5):
            test = np.flatnonzero((fold_id == fold) & with_leads)
            if len(test):
                eligible, used = paired_strata(test, n[test], dist, wn, wf)
                eligible_count += eligible
                used_count += used
        assert np.isclose(wn.sum(), wf.sum())
        near += wn
        far += wf
        near_novelty_sum += np.dot(wn[with_leads], n[with_leads])
        far_novelty_sum += np.dot(wf[with_leads], n[with_leads])
        ranks = stats.rankdata(n[with_leads])
        rho_novelty.append(np.corrcoef(e_rank, ranks)[0, 1])
        partial_novelty.append(partial(e_rank, ranks, [dist_rank]))
        partial_sigma.append(partial(e_rank, sig_rank, [dist_rank, ranks]))
        high = with_leads & (n >= np.quantile(n[with_leads], 2 / 3))
        cut = np.median(dist[high])
        original_delta.append(
            np.sqrt(e2[high & (dist > cut)].mean()) - np.sqrt(e2[high & (dist <= cut)].mean())
        )
    assert np.isclose(near.sum(), far.sum()) and np.isclose(near.sum(), used_count)
    near_novelty = near_novelty_sum / near.sum()
    far_novelty = far_novelty_sum / far.sum()
    near /= near.sum()
    far /= far.sum()
    near_rmse = np.sqrt(near @ e2)
    far_rmse = np.sqrt(far @ e2)
    nscaf, nboot, lo, hi = clustered_interval(scaf, e2, near, far)
    lowest = np.argsort(sig, kind="stable")[: int(len(y) * 0.2)]
    row = dict(
        target=target,
        label=TARGETS[target][0],
        family=TARGETS[target][2],
        n=len(y),
        near_rmse=near_rmse,
        far_rmse=far_rmse,
        delta_rmse=far_rmse - near_rmse,
        ci_low=lo,
        ci_high=hi,
        near_novelty=near_novelty,
        far_novelty=far_novelty,
        near_distance=near @ dist,
        far_distance=far @ dist,
        near_activity=near @ y,
        far_activity=far @ y,
        rho_sigma_error=stats.spearmanr(e, sig).statistic,
        rho_distance_error=stats.spearmanr(e, dist).statistic,
        rho_novelty_error=np.median(rho_novelty),
        partial_sigma=np.median(partial_sigma),
        partial_novelty=np.median(partial_novelty),
        rmse_all=np.sqrt(e2.mean()),
        rmse_low20=np.sqrt(e2[lowest].mean()),
        sensitivity_near_win_fraction=np.mean(np.asarray(original_delta) > 0),
        sensitivity_median_delta=np.median(original_delta),
        scaffold_groups=int(nscaf),
        bootstrap_valid=int(nboot),
        leads_min=min(n_actives),
        leads_max=max(n_actives),
        eligible_fraction_in_valid_strata=used_count / eligible_count,
        n_unique_near=int(np.count_nonzero(near)),
        n_unique_far=int(np.count_nonzero(far)),
    )
    table = table.assign(
        fold=fold_id,
        scaffold=scaf,
        prediction=pred,
        disagreement=sig,
        distance=dist,
        novelty_draw0=novelty[:, 0],
        near_weight=near,
        far_weight=far,
    )
    table.to_csv(
        ROOT / "predictions" / f"{target}.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    print(f"{target} complete", flush=True)
    return row


def summarize(rows):
    df = pd.DataFrame(rows)
    effects = df.delta_rmse.to_numpy()
    mean = effects.mean()
    bounds = stats.t.interval(0.95, len(effects) - 1, loc=mean, scale=stats.sem(effects))
    return dict(
        n_targets=len(rows),
        n_target_parent_records=int(df.n.sum()),
        n_models=len(rows) * 5,
        n_crossfamily_targets_positive=int((effects > 0).sum()),
        delta_rmse_mean=mean,
        working_target_ci=list(bounds),
        target_median=np.median(effects),
        means_by_target_class=df.groupby("family").delta_rmse.mean().to_dict(),
        counts_by_target_class=df.groupby("family").size().to_dict(),
        means=df.select_dtypes(include="number").mean().to_dict(),
        median_partial_sigma=float(df.partial_sigma.median()),
        median_partial_novelty=float(df.partial_novelty.median()),
        max_absolute_novelty_imbalance=float((df.far_novelty - df.near_novelty).abs().max()),
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            sklearn=sklearn.__version__,
            pandas=pd.__version__,
            rdkit=rdkit.__version__,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-workers", type=int, default=3)
    parser.add_argument("--tree-workers", type=int, default=3)
    args = parser.parse_args()
    (ROOT / "predictions").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.target_workers) as pool:
        rows = list(pool.map(run_target, [(t, args.tree_workers) for t in TARGETS]))
    with (ROOT / "target_results.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "summary.json").write_text(json.dumps(summarize(rows), indent=2, allow_nan=False) + "\n")
