# Dated cross-family replication

The source records are exact human-target activities from ChEMBL release 37, downloaded on November 19, 2026. The four targets and all evaluation rules are specified in PROTOCOL.md. Analysis uses a separate chronological sample for each target. All dates refer to publication of an eligible activity measurement against that target.

## Contents and reproducibility

- `records/<target>.csv.gz` contains all rows returned by the target query, including rows later excluded in curation. Activity, assay and document identifiers trace every input to ChEMBL. The frozen service version is in `records/chembl_status.json`. ChEMBL provides its data under the [Creative Commons Attribution-ShareAlike 3.0 Unported license](https://www.ebi.ac.uk/chembl/). The ChEMBL-derived extracts and processed compound-level datasets here retain that license. See [ChEMBL 2024 database article](https://doi.org/10.1093/nar/gkad1004).
- `snapshot.py` documents the exact paginated extraction, verifies the returned target, filters, ordered activity IDs and pagination, and rebuilds the CSV snapshots. The live service can change. Use the frozen CSVs to reproduce the submitted analysis.
- `analyze.py` standardizes structures, rebuilds dated labels, selects parents, fits and evaluates the forests and produces the individual and target-level results. Earlier and later pools are sorted on the SHA-256 hash of the lower-case target name, vertical bar and standardized parent InChIKey, retaining the first 1,250 in each era. Empty Murcko scaffolds are assigned their own parent grouping.
- `results/<target>_flow.json` documents curation. The sampled parents and their era and split membership are in `results/<target>_cohort.csv.gz`. Individual temporal predictions are in `results/<target>_future_predictions.csv.gz`. The control replicates and seed draws are also retained as CSV. Summary CSVs and TeX tables are generated from the target-level files.
- Apply the 2015 date rule only to eligible, target-specific records. A compound can have been published elsewhere beforehand. Publication year is a surrogate for availability of the reported assay results. The sample caps, shared structures across target cohorts, endpoint mixing and number of random controls should be considered when interpreting the results.

Run from this directory in a Python 3 environment with NumPy, pandas, scipy, scikit-learn and RDKit installed:

```sh
python analyze.py faah
python analyze.py ptgs2
python analyze.py kdr
python analyze.py herg
python summarize.py
python verify.py
```

`verify.py` checks the integrity of frozen inputs, samples, fitted prediction files, derived summary metrics and the generated tables. The four analysis commands can run concurrently in separate processes. Each fit uses one worker. See `versions.json` for the exact package versions used.
