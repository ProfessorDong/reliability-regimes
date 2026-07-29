"""Reliability and budgeted-search experiments for molecular design.

This package contains the activity models, the reward, the search operators, and the
experiment and analysis scripts behind "Quantifying the reliability cost of novelty in
molecular design".

    oracle.py               activity model API, target configuration, ECFP featurization
    oracle_sanity_gate.py   model selection and gating; writes the frozen .pkl models
    reward.py               multi-objective reward with the uncertainty penalty
    actions.py              BRICS fragment-edit action space and admissibility filter
    graph_ga.py             Graph GA baseline (crossover / mutation)
    wm_guided_ga.py         surrogate-triaged search over the Graph GA offspring pool
    dynamics.py             dual-encoder latent surrogate (reported as an ablation)
    run_*.py                experiments; analyze_*.py   frozen analyses

Naming note: the package directory, some module names (`wm_guided_ga`), class names
(`LatentWorldModel`, `WMGuidedGA`, `CWMReward`) and the keys of the frozen JSON outputs
retain identifiers from an earlier version of this work. They are kept unchanged so that
the frozen outputs and the reproduction path remain valid; they carry no meaning beyond
naming. The primary method in the manuscript is the fingerprint-surrogate triage
implemented by `run_methods_v2.py` on top of `wm_guided_ga.WMGuidedGA`.
"""

__all__: list[str] = []
