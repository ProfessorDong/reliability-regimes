"""Audit released inputs, folds, predictions, target statistics, and panel aggregation."""

import gzip
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

from analyze import ROOT, TARGETS, clustered_interval


def main():
    target_results = pd.read_csv(ROOT / "target_results.csv").set_index("target")
    manifest = json.loads((ROOT / "input_manifest.json").read_text())
    assert set(target_results.index) == set(TARGETS) == {r["target"] for r in manifest}
    assert (
        hashlib.sha256((ROOT / "PROTOCOL.md").read_bytes()).hexdigest()
        == "0d08020701f3c06c055c18dbeb435bf0a80cd944c66a58589c28d2e8ec8aa6b9"
    )
    all_keys = []
    deltas = []
    for record in manifest:
        target = record["target"]
        raw = gzip.decompress((ROOT / "sources" / f"{target}_chembl.csv.gz").read_bytes())
        assert hashlib.sha256(raw).hexdigest() == record["source_sha256"]
        assert len(raw.splitlines()) - 1 == record["raw_rows"]
        assert (
            record["raw_rows"] == record["invalid_rows"] + record["pooled_rows"] + record["parent_count"]
        )
        parents_path = ROOT / "parents" / f"{target}.csv"
        assert hashlib.sha256(parents_path.read_bytes()).hexdigest() == record["parents_sha256"]
        parent = pd.read_csv(parents_path)
        df = pd.read_csv(
            ROOT / "predictions" / f"{target}.csv.gz",
            keep_default_na=False,
            float_precision="round_trip",
        )
        assert df.parent_key.is_unique and df.parent_key.is_monotonic_increasing
        assert df.parent_key.equals(parent.parent_key) and df.smiles.equals(parent.smiles)
        assert np.allclose(df.pActivity, parent.pActivity)
        assert len(df) == record["modeled"] == min(record["parent_count"], 1200)
        all_keys += df.parent_key.tolist()
        row = target_results.loc[target]
        assert row.n == len(df) and row.label == record["label"] and row.family == record["family"]
        assert (
            len(df.novelty_draw0) == len(df.distance) and (df.novelty_draw0 + 1e-12 >= df.distance).all()
        )
        assert (df.near_weight >= 0).all() and (df.far_weight >= 0).all()
        assert np.isclose(df.near_weight.sum(), 1) and np.isclose(df.far_weight.sum(), 1)
        folds = KFold(5, shuffle=True, random_state=0)
        counts = []
        for fold, (tr, te) in enumerate(folds.split(df)):
            assert (df.fold.iloc[te] == fold).all() and (df.fold.iloc[tr] != fold).all()
            counts.append((df.pActivity.iloc[tr] >= 7).sum())
        assert min(counts) == row.leads_min and max(counts) == row.leads_max
        error = (df.pActivity - df.prediction).to_numpy()
        e2 = error**2
        wn, wf = df.near_weight.to_numpy(), df.far_weight.to_numpy()
        near, far = np.sqrt(wn @ e2), np.sqrt(wf @ e2)
        low = np.argsort(df.disagreement.to_numpy(), kind="stable")[: len(df) // 5]
        checks = dict(
            near_rmse=near,
            far_rmse=far,
            delta_rmse=far - near,
            near_distance=wn @ df.distance,
            far_distance=wf @ df.distance,
            near_activity=wn @ df.pActivity,
            far_activity=wf @ df.pActivity,
            rmse_all=np.sqrt(e2.mean()),
            rmse_low20=np.sqrt(e2[low].mean()),
            rho_sigma_error=stats.spearmanr(df.disagreement, np.abs(error)).statistic,
            rho_distance_error=stats.spearmanr(df.distance, np.abs(error)).statistic,
        )
        for metric, value in checks.items():
            assert np.isclose(row[metric], value, atol=1e-9), (target, metric, row[metric], value)
        nscaf, nboot, lo, hi = clustered_interval(df.scaffold, e2, wn, wf)
        assert row.scaffold_groups == nscaf and row.bootstrap_valid == nboot == 2000
        assert np.allclose([row.ci_low, row.ci_high], [lo, hi])
        deltas.append(far - near)
    summary = json.loads((ROOT / "summary.json").read_text())
    assert summary["n_targets"] == len(deltas) == 11
    assert summary["n_target_parent_records"] == len(all_keys) == 13200
    assert summary["n_models"] == 55
    assert summary["n_crossfamily_targets_positive"] == sum(d > 0 for d in deltas)
    assert np.isclose(summary["delta_rmse_mean"], np.mean(deltas))
    assert np.allclose(
        summary["working_target_ci"],
        stats.t.interval(0.95, 10, loc=np.mean(deltas), scale=stats.sem(deltas)),
    )
    n_distinct = len(set(all_keys))
    assert n_distinct == 12309
    print(
        f"Verified 11 source archives, {len(all_keys):,} target–parents ({n_distinct:,} unique keys), five folds and 2,000 scaffold resamples per target, and the panel summary."
    )


if __name__ == "__main__":
    main()
