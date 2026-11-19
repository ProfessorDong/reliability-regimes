"""Check the frozen inputs, recorded cohorts and all reported dated point estimates."""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from snapshot import HERE, LABELS, TARGETS
from summarize import DISPLAY


def verify():
    frozen = pd.read_csv(HERE / "checksums.tsv", sep="\t")
    for rec in frozen.itertuples():
        assert hashlib.sha256((HERE / rec.path).read_bytes()).hexdigest() == rec.sha256, rec.path
    status = json.loads((HERE / "records/chembl_status.json").read_text())
    assert status["chembl_db_version"] == "ChEMBL_37"
    summary = pd.read_csv(HERE / "summary.csv").set_index("target")
    leads = pd.read_csv(HERE / "lead_summary.csv").set_index("target")
    for name, target_id in TARGETS.items():
        results = HERE / "results"
        row = summary.loc[LABELS[name]]
        flow = json.loads((results / f"{name}_flow.json").read_text())
        source = pd.read_csv(HERE / "records" / f"{name}.csv.gz")
        assert len(source) == flow["records_total"] == source.activity_id.nunique()
        assert source.target_chembl_id.eq(target_id).all()
        assert source.standard_units.eq("nM").all() and source.standard_relation.eq("=").all()
        assert source.activity_id.is_monotonic_increasing
        assert flow["records_total"] == sum(flow[k] for k in ["records_kept", "bad_value", "bad_structure", "nonhuman"])
        assert flow["pre_total"] == row.pre_total and flow["post_total"] == row.post_total
        cohort = pd.read_csv(results / f"{name}_cohort.csv.gz").set_index("inchikey")
        hist = cohort.partition.isin(["training", "calibration"])
        assert (cohort.loc[hist].first_year < 2015).all() and (cohort.loc[~hist].first_year >= 2015).all()
        assert np.array_equal(cohort.loc[~hist, "n_dated"], cohort.loc[~hist, "n_records"])
        assert all(cohort.hash == [hashlib.sha256(f"{name}|{k}".encode()).hexdigest() for k in cohort.index])
        assert len(cohort) == len(set(cohort.index)) == min(1250,row.pre_total) + min(1250,row.post_total)
        for group, count in [("training", row.n_train), ("calibration", row.n_cal), ("future", row.n_test)]:
            assert cohort.partition.eq(group).sum() == count
        future = pd.read_csv(results / f"{name}_future_predictions.csv.gz").set_index("inchikey")
        assert list(future.index) == list(cohort.index[~hist])
        np.testing.assert_allclose(future.activity, cohort.loc[~hist, "y"], rtol=1e-12)
        err = abs(future.activity - future.prediction)
        np.testing.assert_array_equal(future.covered, err <= future.adaptive_90_halfwidth)
        ctl = pd.read_csv(results / f"{name}_random_controls.csv")
        assert list(ctl.replicate) == list(range(32))
        for c, temporal in [("rmse", np.sqrt(np.mean(err**2))), ("rho", spearmanr(err, future.disagreement).statistic),
                            ("coverage", future.covered.mean())]:
            np.testing.assert_allclose(row["temporal_"+c], temporal, atol=1e-12)
            np.testing.assert_allclose([row["control_"+c], row["control_"+c+"_min"], row["control_"+c+"_max"]],
                                       [ctl[c].mean(), ctl[c].min(), ctl[c].max()], atol=1e-12)
        lead = pd.read_csv(results / f"{name}_lead_draws.csv")
        assert list(lead.draw) == list(range(100))
        np.testing.assert_allclose(lead.far_rmse - lead.near_rmse, lead.difference, atol=1e-12)
        calc = [lead.difference.median(), lead.difference.gt(0).mean(), lead.difference.quantile(.025),
                lead.difference.quantile(.975)]
        np.testing.assert_allclose(leads.loc[LABELS[name], ["delta", "positive", "pct025", "pct975"]], calc, atol=1e-12)
    for filename in ["table.tex", "lead_table.tex", "interval_table.tex"]:
        table = (HERE / filename).read_text()
        for label in DISPLAY.values():
            assert label + " & " in table
    print("Verified frozen checksums, four complete target snapshots, cohort partitions, temporal metrics, "
          "32 controls per target, 100 lead draws per target and all three table row sets.")


if __name__ == "__main__":
    verify()
