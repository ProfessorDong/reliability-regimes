"""Run the four target-specific dated replications specified in PROTOCOL.md."""
import argparse
import csv
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
import sklearn
from sklearn.ensemble import RandomForestRegressor

from snapshot import HERE, LABELS, TARGETS

RDLogger.DisableLog("rdApp.*")
NORMALIZER = rdMolStandardize.Normalizer()
LARGEST = rdMolStandardize.LargestFragmentChooser()
UNCHARGER = rdMolStandardize.Uncharger()
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
COLS = ["rmse", "rho", "coverage"]
OUT = HERE / "results"


@lru_cache(maxsize=32768)
def standardize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        parent = UNCHARGER.uncharge(LARGEST.choose(NORMALIZER.normalize(mol)))
        Chem.SanitizeMol(parent)
        key = Chem.MolToInchiKey(parent)
        return (key, Chem.MolToSmiles(parent)) if key else None
    except (RuntimeError, ValueError):
        return None


def parent_cohort(target):
    path = HERE / "records" / f"{target}.csv.gz"
    rows = []
    flow = {"records_total": 0, "nonhuman": 0, "bad_value": 0, "bad_structure": 0}
    with gzip.open(path, "rt") as stream:
        for row in csv.DictReader(stream):
            flow["records_total"] += 1
            if row["target_organism"] != "Homo sapiens":
                flow["nonhuman"] += 1
                continue
            try:
                value = float(row["standard_value"])
            except (TypeError, ValueError):
                value = np.nan
            if not np.isfinite(value) or value <= 0:
                flow["bad_value"] += 1
                continue
            parent = standardize(row["canonical_smiles"])
            if parent is None:
                flow["bad_structure"] += 1
                continue
            year = int(row["document_year"]) if row["document_year"] else np.nan
            rows.append((*parent, year, 9 - np.log10(value), row["standard_type"]))
    records = pd.DataFrame(rows, columns=["inchikey", "smiles", "year", "activity", "type"])
    flow["records_kept"] = len(rows)
    flow["endpoint_records"] = records["type"].value_counts().to_dict()
    groups = records.groupby("inchikey", sort=True)
    parents = groups.agg(smiles=("smiles", "first"), first_year=("year", "min"),
                         n_dated=("year", "count"), n_records=("year", "size"),
                         y=("activity", "median"))
    flow["parents_total"] = len(parents)
    early = records.loc[records.year < 2015].groupby("inchikey").activity.median()
    parents.loc[early.index, "y"] = early
    earlier = parents.first_year < 2015
    later = (parents.first_year >= 2015) & (parents.n_records == parents.n_dated)
    flow["undated_only"] = int(parents.first_year.isna().sum())
    flow["excluded_later_with_missing_year"] = int(((parents.first_year >= 2015) & ~later).sum())
    flow["pre_total"] = int(earlier.sum())
    flow["post_total"] = int(later.sum())
    assert flow["parents_total"] == sum(flow[k] for k in ["undated_only", "excluded_later_with_missing_year", "pre_total", "post_total"])
    print(f"{target} source {flow}", flush=True)
    parents["hash"] = [hashlib.sha256(f"{target}|{key}".encode()).hexdigest() for key in parents.index]
    sample = pd.concat([parents.loc[mask].sort_values("hash").head(1250) for mask in (earlier, later)])
    return sample, flow


def fit(features, y, train, cal, test):
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=1, random_state=0)
    model.fit(features[train], y[train])
    ix = np.concatenate([cal, test])
    x_eval = features[ix]
    trees = np.asarray([tree.predict(x_eval) for tree in model.estimators_])
    pred = trees.mean(axis=0)
    sd = trees.std(axis=0)
    err = np.abs(y[ix] - pred)
    n = len(cal)
    rank = int(np.ceil((n + 1) * 0.90))
    quantile = np.sort(err[:n] / (sd[:n] + 0.001))[rank - 1]
    covered = err[n:] <= quantile * (sd[n:] + 0.001)
    return pred[n:], sd[n:], covered, quantile


def metrics(err, sd, covered):
    return np.array([np.sqrt(np.mean(np.square(err))), spearmanr(sd, err).statistic, np.mean(covered)])


def scaffold_bootstrap(mols, keys, err, sd, covered):
    names = [MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or key for key, mol in zip(keys, mols)]
    unique, inverse = np.unique(names, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(7)
    draws = []
    for _ in range(2000):
        i = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        draws.append(metrics(err[i], sd[i], covered[i]))
    return len(groups), np.nanpercentile(draws, [2.5, 97.5], axis=0)


def lead_distances(fps, y, train, test, err, target):
    active = np.flatnonzero(y[train] >= 6.5)
    result = {"target": LABELS[target], "pre_actives": len(active)}
    if len(active) < 10:
        result["reason"] = "Fewer than ten historical training actives"
        return result
    tr_fp = [fps[i] for i in train]
    similarity = np.asarray([DataStructs.BulkTanimotoSimilarity(fps[i], tr_fp) for i in test])
    distance = 1 - similarity.max(axis=1)
    rng = np.random.default_rng(602)
    results = []
    for draw in range(100):
        lead = rng.choice(active, size=10, replace=False)
        novelty = 1 - similarity[:, lead].max(axis=1)
        # test is already ordered by the target/parent hash independently of activity.
        ordered = np.argsort(-novelty, kind="stable")
        top = ordered[:len(ordered) // 3]
        nearer, farther = [], []
        for b in np.array_split(top, 5):
            order = b[np.argsort(distance[b], kind="stable")]
            half = len(b) // 2
            left, right = order[:half], order[::-1][:half]
            keep = distance[left] < distance[right]
            nearer.extend(left[keep])
            farther.extend(right[keep])
        a, b = np.array(nearer), np.array(farther)
        if len(a) == 0:
            continue
        near = np.sqrt(np.mean(err[a]**2))
        far = np.sqrt(np.mean(err[b]**2))
        results.append(dict(draw=draw, near_rmse=near, far_rmse=far, difference=far-near,
                            pairs=len(a), mean_abs_seed_distance_diff=np.mean(np.abs(novelty[a]-novelty[b])),
                            mean_train_distance_diff=np.mean(distance[b]-distance[a])))
    frame = pd.DataFrame(results)
    frame.to_csv(OUT / f"{target}_lead_draws.csv", index=False)
    if frame.empty:
        result["reason"] = "No pairs with unequal training distance"
        return result
    diff = frame.difference
    result.update(valid_draws=len(frame), median_near=frame.near_rmse.median(), median_far=frame.far_rmse.median(),
                  delta=diff.median(), pct025=diff.quantile(0.025), pct975=diff.quantile(0.975),
                  positive=diff.gt(0).mean(), median_pairs=frame.pairs.median(),
                  median_abs_seed_distance_diff=frame.mean_abs_seed_distance_diff.median(),
                  median_train_distance_diff=frame.mean_train_distance_diff.median())
    return result


def analyze(target):
    OUT.mkdir(exist_ok=True)
    sample, flow = parent_cohort(target)
    (OUT / f"{target}_flow.json").write_text(json.dumps(flow, indent=2) + "\n")
    if flow["pre_total"] < 100 or flow["post_total"] < 50:
        print(f"{target}: ineligible", flush=True)
        return
    mols = [Chem.MolFromSmiles(s) for s in sample.smiles]
    fps = [MORGAN.GetFingerprint(m) for m in mols]
    features = np.array(fps, dtype=np.float32)
    y = sample.y.to_numpy()
    old = np.flatnonzero(sample.first_year < 2015)
    test = np.flatnonzero(sample.first_year >= 2015)
    old = np.random.default_rng(0).permutation(old)
    n_cal = max(50, len(old)//4)
    cal, train = old[:n_cal], old[n_cal:]
    pred, sd, covered, quantile = fit(features, y, train, cal, test)
    err = np.abs(pred - y[test])
    temporal = metrics(err, sd, covered)
    result = dict(target=LABELS[target], pre_total=flow["pre_total"], post_total=flow["post_total"],
                  n_train=len(train), n_cal=len(cal), n_test=len(test))
    sample["partition"] = "future"
    sample.loc[sample.index[train], "partition"] = "training"
    sample.loc[sample.index[cal], "partition"] = "calibration"
    sample.to_csv(OUT / f"{target}_cohort.csv.gz", compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(dict(inchikey=sample.index[test], activity=y[test], prediction=pred, disagreement=sd,
                      adaptive_90_halfwidth=quantile*(sd+.001), covered=covered)).to_csv(
                      OUT / f"{target}_future_predictions.csv.gz", index=False,
                      compression={"method": "gzip", "mtime": 0})
    lead = lead_distances(fps, y, train, test, err, target)
    pd.DataFrame([lead]).to_csv(OUT / f"{target}_lead_row.csv", index=False)
    n_scaffolds, bounds = scaffold_bootstrap([mols[i] for i in test], sample.index[test], err, sd, covered)
    result["n_scaffolds"] = n_scaffolds
    print(f"{target}: temporal {temporal}, lead {lead}; fitting random controls", flush=True)
    random_controls = []
    for rep in range(32):
        order = np.random.default_rng(100+rep).permutation(len(sample))
        te, ca, tr = np.split(order, [len(test), len(test)+len(cal)])
        ctl_pred, ctl_sd, ctl_cov, _ = fit(features, y, tr, ca, te)
        random_controls.append(metrics(np.abs(ctl_pred-y[te]), ctl_sd, ctl_cov))
        if (rep+1) % 4 == 0:
            print(f"{target}: {rep+1}/32", flush=True)
    controls = pd.DataFrame(random_controls, columns=COLS)
    controls.index.name = "replicate"
    controls.to_csv(OUT / f"{target}_random_controls.csv")
    for j, column in enumerate(COLS):
        result.update({f"temporal_{column}": temporal[j], f"control_{column}": controls[column].mean(),
                       f"temporal_{column}_lo": bounds[0,j], f"temporal_{column}_hi": bounds[1,j],
                       f"control_{column}_min": controls[column].min(), f"control_{column}_max": controls[column].max()})
    pd.DataFrame([result]).to_csv(OUT / f"{target}_row.csv", index=False)
    print(f"{target} finished; RDKit {rdBase.rdkitVersion}, sklearn {sklearn.__version__}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=TARGETS)
    analyze(parser.parse_args().target)
