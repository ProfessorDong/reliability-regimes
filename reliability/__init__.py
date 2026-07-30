"""Chemistry World Model (CWM) package for few-shot theranostic molecule generation.

Design phase only. See PLAN_code.md for the module-by-module build spec and
../../../JournalPapers_generation/DESIGN.md for the scientific blueprint.

Planned modules (not yet implemented):
    actions.py        MMP / fragment (BRICS-RECAP) action space + actionability filter
    dynamics.py       learned latent-dynamics transition model M_phi over dual-encoder R^128
    reward.py         multi-objective reward wrapping the Part-1 MoEPredictor (+ QED, SA, selectivity)
    planner.py        beam / model-predictive rollout in latent space (PMO oracle budget)
"""

__all__: list[str] = []
