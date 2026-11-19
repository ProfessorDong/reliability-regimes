# Reproducing the additional-target validation

`PROTOCOL.md` specifies the analysis before estimation of the additional-target endpoints. The hash in `check.py` identifies that file. This is a target-level extension of an existing study on five targets. It concerns random structure-disjoint cross-validation on 11 additional ChEMBL targets. The source archives have no dates for temporal validation.

Run these commands in this directory with Python 3.12.13 and the packages in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python prepare.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python analyze.py
OPENBLAS_NUM_THREADS=1 python check.py
python overlap.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python activity_sensitivity.py
```

`prepare.py` reads the eleven compressed CSVs in `sources/`, standardizes all source rows, and writes the retained parent tables and `input_manifest.json`. It defaults to five processes. `PREP_WORKERS` sets the number of processes. `analyze.py` defaults to three targets in parallel and three threads per forest. For serial execution use `python analyze.py --target-workers 1 --tree-workers 1`.

`target_results.csv` contains the prespecified endpoints and diagnostics for each target. `summary.json` summarizes effects with equal target weights and records the software versions. The primary effect is far-group RMSE minus near-group RMSE. The groups are compared within fold and seed-novelty strata and averaged over 100 lead draws. `near_novelty` and `far_novelty` are weighted mean seed-novelty values. The two `distance` columns contain the weighted mean distance to the full training fold. `leads_min` and `leads_max` are the minimum and maximum number of available training-fold actives from which the ten leads were drawn. The scaffold bounds condition on the fitted models and fixed test assignments. `predictions/` contains the held-out predictions, scaffold groups, measured activities and normalized compound weights. `fingerprint_overlap.csv` is a post hoc sensitivity calculation excluding held-out parents whose ECFP4 fingerprints match a training parent exactly, with the primary weights retained and renormalized for this diagnostic.

`diagnostic_plan.md` describes both post hoc diagnostics. `activity_sensitivity.csv` and `activity_sensitivity.json` give the outcome-informed half-log-bin analysis, including the denominator and number of retained high-novelty compound assignments. `activity_sensitivity.py` also independently reconstructs all original near/far weights, fold-specific nearest-neighbor distances, and lead distances in the fixed first draw and checks them against the deposited predictions.

The sources are the project's previously curated ChEMBL data. `sources/curate_chembl_v8.py` preserves the original extraction code for provenance. The original code accepted approximate and missing measurement relations as well as exact ones, and filtered out-of-range activities. The archived CSVs, originally named `*_chembl.csv`, are provided byte-for-byte after decompression. The input manifest records SHA-256 values over the uncompressed files and over each modeled parent table. Each row in a source CSV is a median for a canonical structure, and all target identifiers are recorded in the manifest. ChEMBL is provided by the European Molecular Biology Laboratory's European Bioinformatics Institute under CC BY-SA 3.0. See [ChEMBL](https://www.ebi.ac.uk/chembl/) and its [license](https://creativecommons.org/licenses/by-sa/3.0/).
