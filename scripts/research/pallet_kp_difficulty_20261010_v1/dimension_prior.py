"""Explicit initial-dimension prior ablation; no new coordinate or truth input.

This is a new causal diagnostic, not a change to the published finite solver.
The original numerical pool, residual rule and free six-DOF refit are retained.
The initial pose chooses a registered width/depth branch only. Its coordinates
may themselves be wrong: this prior is deliberately disclosed, not advertised
as a new observation or independent solution to global pose ambiguity.
"""
import numpy as np
from ..pallet_observation_refiner_20261009_v1.solver import HypothesisBank


class DimensionPriorBank:
    def __init__(self, bank, initial_pose):
        if not isinstance(bank, HypothesisBank):
            raise TypeError('Expected the unchanged published HypothesisBank')
        self.bank = bank
        extent = initial_pose.get('cf_extents') if initial_pose.get('available') else None
        self.initial_label = initial_pose.get('selected_hypothesis')
        self.extent = None if extent is None else np.asarray(extent, dtype=float)
        matches = [] if self.extent is None else [i for i, dim in enumerate(bank.dims)
                     if np.allclose(dim, self.extent, rtol=0, atol=1e-9)]
        self.index = matches[0] if len(matches) == 1 else None

    def __getattr__(self, name):
        return getattr(self.bank, name)

    def solve(self, excluded=(), robust=True, hidden=()):
        if not robust:
            raise ValueError('The new ablation is a robust-solver dimension prior')
        removed = set(excluded) | set(hidden)
        usable = [i for i in self.bank.eligible if i not in removed]
        if self.index is None:
            answer = self.bank.solve(excluded=range(8), robust=True, hidden=hidden)
            answer.update(reason='PRIOR_UNAVAILABLE', used=usable, excluded=sorted(removed),
                          prior_used=False, initial_dimension_prior_available=False)
            return answer
        before = self.bank.ledger.copy()
        # No generation is needed when fewer than four actual inputs remain.
        if len(usable) >= 4:
            self.bank._build()
        original = self.bank.hypotheses
        allowed = None if original is None else [c for c in original if c.dim == self.index]
        try:
            self.bank.hypotheses = allowed
            answer = self.bank.solve(excluded=excluded, robust=True, hidden=hidden)
        finally:
            self.bank.hypotheses = original
        answer.update(prior_used=True, initial_dimension_prior_available=True,
                      initial_dimension_prior=dict(origin='frozen same-coordinate initial pose',
                          label=self.initial_label, cf_extents=self.extent.tolist(),
                          dimension_index=self.index, rotation_translation_residual_prior=False,
                          inherited_initial_errors_possible=True),
                      hypothesis_count_before_prior=len(original or []),
                      hypothesis_count_after_prior=len(allowed or []),
                      operation_counts={k: self.bank.ledger[k]-before[k] for k in before},
                      hypothesis_bank_counts=self.bank.ledger.copy())
        if answer['available']:
            assert np.allclose(answer['cf_extents'], self.extent, rtol=0, atol=1e-9)
            assert not set(answer['fit_input_ids']) & removed
        return answer
