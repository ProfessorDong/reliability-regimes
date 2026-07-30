"""Analysis package for the reliability-regimes study.

Modules
-------
oracle.py             per-target random-forest activity models and their data paths
standardize.py        structure standardization and parent-InChIKey grouping
run_reliability_v2.py in-distribution error stratification and risk-coverage
run_conformal_v1.py   split conformal intervals, standard and disagreement-normalized
run_temporal_v1.py    temporal split with its size-matched random control
run_poolopt_v1.py     acquisition against measured activity under a fixed query budget
run_methods_v2.py     fingerprint-surrogate triage, the method reported in the manuscript
graph_ga.py           the genetic-algorithm baseline
surrogate_ga.py       the surrogate-triaged search built on it
dynamics.py           the dual encoder used for the latent-surrogate ablation
reward.py             the per-target composite reward the searches optimize

Frozen outputs backing every reported number are in outputs/frozen; verify_results.py
checks them.
"""
