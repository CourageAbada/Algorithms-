"""Phase 2B research layer: frozen policies, eligibility/segments, label specifications, cost model, chronological folds.

Nothing here trains a model or evaluates profitability. Everything an experiment may depend on is frozen in
``research/phase2b/*.json`` and verified by ``fxscalp.research.spec.load_frozen_spec`` so a runner cannot silently deviate.
"""
