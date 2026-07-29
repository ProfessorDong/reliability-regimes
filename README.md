# Quantifying the reliability cost of novelty in molecular design

Code, curated data, and frozen experimental outputs for the study of how prediction
reliability degrades as a molecular search is pushed toward novelty, across five targets
relevant to therapy and molecular imaging: **SCD-1**, **FADS** (endoplasmic-reticulum
membrane lipid desaturases) and **NK1R**, **DRD2**, **DRD3** (class A G-protein-coupled
receptors).

Author: **Liang Dong**.

Every number reported in the manuscript and its Supplementary Information is produced by a
frozen analysis script run against a frozen output file in `outputs/cwm_v1/`, and is checked
by `verify_results.py`.

## Summary of what this code establishes

1. **A validated reliability score.** The standard deviation of the per-tree predictions of a
   random-forest activity model, `sigma_T`, predicts the model's *measured* error
   (pooled Spearman 0.40 over 21,173 out-of-fold predictions). Retaining the
   lowest-disagreement fifth of predictions reduces pooled RMSE from 0.71 to 0.41 pIC50.
   The score remains predictive after controlling for both support-novelty and distance to
   the training set (partial Spearman 0.38).
2. **Novelty drives molecules off the training distribution.** Sweeping a novelty incentive
   for two optimizers at two uncertainty penalties (1,200 runs) drives generated molecules
   away from the training distribution (Spearman 0.97-0.98), raising `sigma_T` and lowering
   predicted potency. This holds with the uncertainty penalty set to zero. Support-novelty
   by itself is a weak predictor of measured error (pooled Spearman 0.04).
3. **A fingerprint surrogate improves budgeted search.** An online ECFP random-forest
   surrogate that triages a Graph GA offspring pool under a fixed evaluation budget improves
   the top-10 reward over vanilla Graph GA in all 15 target-by-k cells (target-level mean
   +0.057, 95% CI [+0.030, +0.083], P = 0.004), beats a same-pool random-triage control, and
   outperforms a learned dual-encoder latent surrogate on every target.
4. **Reported negative results.** The uncertainty term in the acquisition does not help
   end to end; a learned latent representation underperforms fingerprints; source-target
   compatibility does not predict warm-start benefit; and targeted scaffold recovery is not
   improved by the triage.

## Reproducing the results

```bash
pip install -r requirements.txt

# 1. Build and gate the activity models (writes outputs/cwm_v1/oracle_{target}.pkl).
#    Required before any search experiment; the .pkl files are NOT in this repository
#    (see "Large files" below).
python -m world_model.oracle_sanity_gate

# 2. Reliability analyses (CPU only; these refit the models internally)
python -m world_model.analyze_calibration      # sigma_T vs measured error
python -m world_model.run_applicability_v1     # distance-to-training, partial correlations
python -m world_model.run_reliability_v1       # measured error vs novelty; risk-coverage

# 3. Search experiments (need the .pkl activity models from step 1)
python -m world_model.run_frontier_v2          # novelty sweep x 2 optimizers x 2 penalties
python -m world_model.run_methods_v2           # primary method comparison (fingerprint surrogate)
python -m world_model.run_ecfp_baseline_v1     # fingerprint vs dual-encoder latent surrogate
python -m world_model.run_beta_ablation_v1     # uncertainty-acquisition ablation
python -m world_model.run_recovery_v2          # leakage-free scaffold-cluster recovery
python -m world_model.run_hierstats_v1         # hierarchical / target-level statistics

# 4. Check every reported number against the frozen outputs
python verify_results.py
```

`verify_results.py` asserts 60 numeric claims and exits non-zero on any mismatch.

## Repository layout

```
data/                     curated per-target activity data (SMILES, pIC50) + featurizers
  scd1_binding.csv                         762 compounds  (active threshold 7.0)
  fatty_acid_desaturase_bioactivity.csv  1,187            (active threshold 6.5)
  nk1r_combined.csv                      3,056            (active threshold 7.0)
  drd2_bioactivity.csv                   9,966            (active threshold 6.5)
  drd3_chembl.csv                        6,202            (active threshold 7.0)
world_model/              activity model, reward, search operators, experiments
  oracle.py                     activity model API, target config, ECFP featurization
  oracle_sanity_gate.py         model selection + gating; writes the frozen .pkl models
  reward.py                     multi-objective reward with the uncertainty penalty
  actions.py                    BRICS fragment-edit action space + admissibility filter
  graph_ga.py                   Graph GA baseline (crossover / mutation)
  wm_guided_ga.py               surrogate-triaged search
  dynamics.py                   dual-encoder latent surrogate (ablation only)
  run_reliability_v1.py         measured error vs novelty and distance; risk-coverage
  run_applicability_v1.py       distance-to-training analysis; partial correlations
  run_frontier_v2.py            novelty sweep across optimizers and penalties
  run_methods_v2.py             primary method comparison (fingerprint surrogate)
  run_ecfp_baseline_v1.py       fingerprint vs latent surrogate
  run_beta_ablation_v1.py       uncertainty-acquisition ablation
  run_recovery_v2.py            leakage-free scaffold-cluster recovery
  run_hierstats_v1.py           hierarchical / target-level statistics
  analyze_*.py                  frozen analysis scripts for the individual experiments
outputs/cwm_v1/           frozen result files; the source of every reported number
verify_results.py         asserts every reported number against those files
```

## Data

The per-target datasets are curated from ChEMBL and the primary literature under a uniform
pipeline. Activities are pooled to pIC50; replicate ChEMBL measurements are aggregated to one
value per canonical structure before thresholding, while the SCD-1 literature-curated binding
set retains a small number of replicate records. FADS aggregates FADS1 and FADS2 desaturase
activity drawn mostly from a curated literature panel, so its activity function is the most
heterogeneous of the five.

Throughout, "few-seed" refers to the k in {5, 10, 20} starting active compounds supplied to
the search, not to the activity model, which is trained on all measured compounds for a target.

## Large files

The frozen activity models `outputs/cwm_v1/oracle_{target}.pkl` are not included: the DRD2
model exceeds the 100 MB per-file limit on GitHub. Rebuild them with
`python -m world_model.oracle_sanity_gate`, which reproduces them deterministically from the
data in this repository. The frozen JSON outputs that back the reported numbers are included,
so `verify_results.py` runs without rebuilding the models.

## A note on internal naming

The package directory (`world_model/`), some module and class names (`wm_guided_ga`,
`WMGuidedGA`, `LatentWorldModel`, `CWMReward`) and the keys of the frozen JSON outputs
retain identifiers from an earlier version of this work. They are kept unchanged so that the
frozen outputs and the reproduction path stay valid, and they carry no meaning beyond naming.
The primary method reported in the manuscript is the fingerprint-surrogate triage in
`run_methods_v2.py`.

## License

MIT. See `LICENSE`.
