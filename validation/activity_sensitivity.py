"""Post hoc sensitivity stratifying the primary comparison by measured activity."""

from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats

from analyze import ROOT, TARGETS, DRAWS, clustered_interval


def run_target(target):
    RDLogger.DisableLog("rdApp.*")
    data = pd.read_csv(
        ROOT / "predictions" / f"{target}.csv.gz", float_precision="round_trip", keep_default_na=False
    )
    y = data.pActivity.to_numpy()
    bins = np.floor(2 * y).astype(int)
    dist = data.distance.to_numpy()
    squared_error = (data.prediction.to_numpy() - y) ** 2
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=False)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in data.smiles]
    similarities = np.stack([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])
    primary_near, primary_far, near, far = [np.zeros(len(y)) for _ in range(4)]
    primary_count = retained = 0
    near_novelty = far_novelty = 0.0
    for fold in range(5):
        test = np.flatnonzero(data.fold.to_numpy() == fold)
        train = np.flatnonzero(data.fold.to_numpy() != fold)
        active = train[y[train] >= 7]
        np.testing.assert_allclose(
            dist[test], 1 - similarities[np.ix_(test, train)].max(axis=1), atol=1e-14, rtol=0
        )
        if not len(active):
            continue
        for draw in range(DRAWS):
            rng = np.random.default_rng(np.random.SeedSequence([20261119, fold, draw]))
            starts = rng.choice(active, size=min(len(active), 10), replace=False)
            nov = 1 - similarities[np.ix_(test, starts)].max(axis=1)
            assert (nov >= dist[test] - 1e-12).all()
            if draw == 0:
                np.testing.assert_allclose(nov, data.novelty_draw0.to_numpy()[test], atol=1e-14, rtol=0)
            high = nov >= np.quantile(nov, 2 / 3)
            cuts = np.unique(np.quantile(nov[high], [0.2, 0.4, 0.6, 0.8]))
            strata = np.searchsorted(cuts, nov, side="left")
            for s in np.unique(strata[high]):
                local = np.flatnonzero(high & (strata == s))
                group = test[local]
                is_near = dist[group] <= np.median(dist[group])
                if is_near.all() or not is_near.any():
                    continue
                primary_count += len(local)
                primary_near[group[is_near]] += len(group) / is_near.sum()
                primary_far[group[~is_near]] += len(group) / (~is_near).sum()
                for b in np.unique(bins[group]):
                    inside = bins[group] == b
                    close = inside & is_near
                    remote = inside & ~is_near
                    if not close.any() or not remote.any():
                        continue
                    size = inside.sum()
                    retained += size
                    wn = size / close.sum()
                    wf = size / remote.sum()
                    near[group[close]] += wn
                    far[group[remote]] += wf
                    near_novelty += wn * nov[local[close]].sum()
                    far_novelty += wf * nov[local[remote]].sum()
    # The independent reconstruction must agree with the deposited primary weights.
    assert np.isclose(primary_count, primary_near.sum()) and np.isclose(primary_count, primary_far.sum())
    np.testing.assert_allclose(primary_near / primary_near.sum(), data.near_weight, atol=1e-14, rtol=0)
    np.testing.assert_allclose(primary_far / primary_far.sum(), data.far_weight, atol=1e-14, rtol=0)
    assert np.isclose(near.sum(), retained) and np.isclose(far.sum(), retained)
    near /= retained
    far /= retained
    for b in np.unique(bins):
        assert np.isclose(near[bins == b].sum(), far[bins == b].sum())
    close_rmse = np.sqrt(near @ squared_error)
    distant_rmse = np.sqrt(far @ squared_error)
    _, nboot, lo, hi = clustered_interval(data.scaffold, squared_error, near, far)
    return dict(
        target=target,
        primary_assignments=int(primary_count),
        retained_assignments=int(retained),
        retained_fraction=retained / primary_count,
        unique_near=int((near > 0).sum()),
        unique_far=int((far > 0).sum()),
        near_rmse=close_rmse,
        far_rmse=distant_rmse,
        delta_rmse=distant_rmse - close_rmse,
        ci_low=lo,
        ci_high=hi,
        near_activity=near @ y,
        far_activity=far @ y,
        near_novelty=near_novelty / retained,
        far_novelty=far_novelty / retained,
        near_distance=near @ dist,
        far_distance=far @ dist,
        bootstrap_valid=int(nboot),
    )


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as pool:
        frame = pd.DataFrame(list(pool.map(run_target, TARGETS)))
    frame.to_csv(ROOT / "activity_sensitivity.csv", index=False)
    d = frame.delta_rmse.to_numpy()
    summary = dict(
        positive_targets=int((d > 0).sum()),
        mean_delta=d.mean(),
        working_target_ci=list(stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))),
        pooled_retained_assignments=int(frame.retained_assignments.sum()),
        pooled_primary_assignments=int(frame.primary_assignments.sum()),
        pooled_retained_fraction=frame.retained_assignments.sum() / frame.primary_assignments.sum(),
        min_retained_fraction=frame.retained_fraction.min(),
        max_retained_fraction=frame.retained_fraction.max(),
        equal_target_means=frame.select_dtypes("number").mean().to_dict(),
    )
    (ROOT / "activity_sensitivity.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(
        frame[["target", "delta_rmse", "ci_low", "ci_high", "retained_fraction"]]
        .round(4)
        .to_string(index=False),
        flush=True,
    )
