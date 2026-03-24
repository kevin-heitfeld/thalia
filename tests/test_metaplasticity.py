"""Tests for MetaplasticityStrategy wrapper.

Validates per-synapse plasticity rate modulation:
1. Depression after weight modification
2. Recovery toward rate_rest
3. Consolidation accumulation lowers rate_rest
4. rate_min floor is respected
5. Sparse and dense paths produce identical results
6. No-modification passthrough (recovery without depression)
7. Wrapping preserves base strategy internal state
"""

from __future__ import annotations

import torch
import pytest

from thalia.learning.strategies import (
    LearningConfig,
    LearningStrategy,
    MetaplasticityConfig,
    MetaplasticityStrategy,
    STDPConfig,
    STDPStrategy,
    TagAndCaptureConfig,
    TagAndCaptureStrategy,
    ThreeFactorConfig,
    ThreeFactorStrategy,
)


# =============================================================================
# Helpers
# =============================================================================

DEVICE = torch.device("cpu")


class _ConstantDeltaStrategy(LearningStrategy):
    """Test helper: always adds a fixed Δw to every synapse."""

    def __init__(self, delta: float = 0.01):
        super().__init__(config=LearningConfig())
        self._delta = delta

    def compute_update(self, weights, pre_spikes, post_spikes, **kwargs):
        return weights + self._delta

    def compute_update_sparse(self, values, row_indices, col_indices,
                               n_post, n_pre, pre_spikes, post_spikes, **kwargs):
        return values + self._delta


class _ZeroDeltaStrategy(LearningStrategy):
    """Test helper: never modifies weights."""

    def __init__(self):
        super().__init__(config=LearningConfig())

    def compute_update(self, weights, pre_spikes, post_spikes, **kwargs):
        return weights.clone()

    def compute_update_sparse(self, values, row_indices, col_indices,
                               n_post, n_pre, pre_spikes, post_spikes, **kwargs):
        return values.clone()


def _make_meta(
    base: LearningStrategy | None = None,
    *,
    delta: float = 0.01,
    tau_recovery_ms: float = 5000.0,
    depression_strength: float = 5.0,
    tau_consolidation_ms: float = 300000.0,
    consolidation_sensitivity: float = 0.1,
    rate_min: float = 0.1,
) -> MetaplasticityStrategy:
    if base is None:
        base = _ConstantDeltaStrategy(delta=delta)
    cfg = MetaplasticityConfig(
        tau_recovery_ms=tau_recovery_ms,
        depression_strength=depression_strength,
        tau_consolidation_ms=tau_consolidation_ms,
        consolidation_sensitivity=consolidation_sensitivity,
        rate_min=rate_min,
    )
    return MetaplasticityStrategy(base, cfg)


def _step(meta: MetaplasticityStrategy, weights: torch.Tensor,
          n_pre: int = 4, n_post: int = 3) -> torch.Tensor:
    """Run one compute_update step with dummy spikes."""
    pre = torch.ones(n_pre, dtype=torch.bool, device=DEVICE)
    post = torch.ones(n_post, dtype=torch.bool, device=DEVICE)
    return meta.compute_update(weights, pre, post)


# =============================================================================
# Config validation
# =============================================================================

class TestMetaplasticityConfig:

    def test_valid_defaults(self):
        cfg = MetaplasticityConfig()
        assert cfg.tau_recovery_ms == 5000.0
        assert cfg.rate_min == 0.1

    def test_invalid_tau_recovery(self):
        with pytest.raises(Exception):
            MetaplasticityConfig(tau_recovery_ms=0.0)

    def test_invalid_rate_min(self):
        with pytest.raises(Exception):
            MetaplasticityConfig(rate_min=0.0)

    def test_invalid_rate_min_above_one(self):
        with pytest.raises(Exception):
            MetaplasticityConfig(rate_min=1.5)

    def test_invalid_consolidation_sensitivity(self):
        with pytest.raises(Exception):
            MetaplasticityConfig(consolidation_sensitivity=-1.0)


# =============================================================================
# Core dynamics
# =============================================================================

class TestPlasticityRateDepression:
    """After a large Δw, plasticity_rate decreases proportionally to β."""

    def test_depression_occurs(self):
        meta = _make_meta(delta=0.01, depression_strength=5.0)
        n_pre, n_post = 4, 3
        weights = torch.full((n_post, n_pre), 0.5, device=DEVICE)

        _step(meta, weights)

        # plasticity_rate should be below 1.0 after modification
        assert meta.plasticity_rate.max().item() < 1.0

    def test_larger_delta_causes_more_depression(self):
        meta_small = _make_meta(delta=0.005, depression_strength=5.0)
        meta_large = _make_meta(delta=0.020, depression_strength=5.0)
        n_pre, n_post = 4, 3
        w = torch.full((n_post, n_pre), 0.5, device=DEVICE)

        _step(meta_small, w.clone())
        _step(meta_large, w.clone())

        # Larger Δw → lower plasticity_rate
        assert meta_large.plasticity_rate.mean().item() < meta_small.plasticity_rate.mean().item()

    def test_higher_beta_causes_more_depression(self):
        meta_low = _make_meta(delta=0.01, depression_strength=2.0)
        meta_high = _make_meta(delta=0.01, depression_strength=10.0)
        w = torch.full((3, 4), 0.5, device=DEVICE)

        _step(meta_low, w.clone())
        _step(meta_high, w.clone())

        assert meta_high.plasticity_rate.mean().item() < meta_low.plasticity_rate.mean().item()


class TestPlasticityRateRecovery:
    """After depression, rate recovers toward rate_rest with τ_recovery."""

    def test_recovery_after_no_activity(self):
        meta = _make_meta(delta=0.01, depression_strength=10.0, tau_recovery_ms=100.0)
        n_pre, n_post = 4, 3
        w = torch.full((n_post, n_pre), 0.5, device=DEVICE)

        # Depress
        _step(meta, w.clone())
        rate_after_depression = meta.plasticity_rate.mean().item()

        # Now run steps with zero delta (no modification → pure recovery)
        meta_zero = _make_meta(
            base=_ZeroDeltaStrategy(),
            tau_recovery_ms=100.0,
            depression_strength=10.0,
        )
        # Transfer metastate
        meta_zero.setup(n_pre, n_post, DEVICE)
        meta_zero.plasticity_rate.copy_(meta.plasticity_rate)
        meta_zero.consolidation.copy_(meta.consolidation)

        # Run many recovery steps
        w2 = torch.full((n_post, n_pre), 0.5, device=DEVICE)
        for _ in range(500):
            _step(meta_zero, w2)

        rate_after_recovery = meta_zero.plasticity_rate.mean().item()
        assert rate_after_recovery > rate_after_depression

    def test_faster_tau_means_faster_recovery(self):
        w = torch.full((3, 4), 0.5, device=DEVICE)

        # Both start depressed
        meta_fast = _make_meta(
            base=_ZeroDeltaStrategy(), tau_recovery_ms=50.0, depression_strength=10.0
        )
        meta_slow = _make_meta(
            base=_ZeroDeltaStrategy(), tau_recovery_ms=5000.0, depression_strength=10.0
        )
        for m in [meta_fast, meta_slow]:
            m.setup(4, 3, DEVICE)
            m.plasticity_rate.fill_(0.3)

        # Run 100 recovery steps
        for _ in range(100):
            _step(meta_fast, w.clone())
            _step(meta_slow, w.clone())

        assert meta_fast.plasticity_rate.mean().item() > meta_slow.plasticity_rate.mean().item()


class TestConsolidation:
    """Repeated modifications increase consolidation, lowering rate_rest."""

    def test_consolidation_accumulates(self):
        meta = _make_meta(delta=0.01)
        w = torch.full((3, 4), 0.5, device=DEVICE)

        _step(meta, w.clone())
        consol_1 = meta.consolidation.mean().item()

        _step(meta, w.clone())
        consol_2 = meta.consolidation.mean().item()

        assert consol_2 > consol_1

    def test_consolidated_synapses_have_lower_rate_rest(self):
        # High consolidation → low rate_rest → low plasticity_rate ceiling
        meta = _make_meta(
            base=_ZeroDeltaStrategy(),
            tau_recovery_ms=10.0,  # Very fast recovery
            consolidation_sensitivity=0.01,  # Very sensitive
        )
        meta.setup(4, 3, DEVICE)
        meta.consolidation.fill_(1.0)  # Heavy consolidation
        meta.plasticity_rate.fill_(0.5)  # Start partially recovered

        w = torch.full((3, 4), 0.5, device=DEVICE)
        # Many recovery steps converge toward rate_rest
        for _ in range(1000):
            _step(meta, w)

        # rate_rest ≈ rate_min + (1 - rate_min) * exp(-1.0/0.01) ≈ rate_min
        assert meta.plasticity_rate.mean().item() < 0.15


class TestRateMinFloor:
    """plasticity_rate never drops below rate_min."""

    def test_floor_respected(self):
        meta = _make_meta(
            delta=0.1, depression_strength=100.0, rate_min=0.05
        )
        w = torch.full((3, 4), 0.5, device=DEVICE)

        # Massive depression
        for _ in range(50):
            _step(meta, w.clone())

        assert meta.plasticity_rate.min().item() >= 0.05 - 1e-7


class TestSparseDenseEquivalence:
    """Dense and sparse paths produce identical results for same connectivity."""

    def test_equivalence(self):
        n_pre, n_post = 8, 6
        delta = 0.01
        cfg = MetaplasticityConfig(
            tau_recovery_ms=5000.0,
            depression_strength=5.0,
            tau_consolidation_ms=300000.0,
            consolidation_sensitivity=0.1,
            rate_min=0.1,
        )

        # Dense path
        meta_dense = MetaplasticityStrategy(_ConstantDeltaStrategy(delta), cfg)
        meta_dense.setup(n_pre, n_post, DEVICE)

        # Sparse path — full connectivity
        meta_sparse = MetaplasticityStrategy(_ConstantDeltaStrategy(delta), cfg)
        meta_sparse.setup(n_pre, n_post, DEVICE)

        w_dense = torch.full((n_post, n_pre), 0.5, device=DEVICE)
        pre = torch.ones(n_pre, dtype=torch.bool, device=DEVICE)
        post = torch.ones(n_post, dtype=torch.bool, device=DEVICE)

        # Build full COO indices
        rows = torch.arange(n_post, device=DEVICE).repeat_interleave(n_pre)
        cols = torch.arange(n_pre, device=DEVICE).repeat(n_post)
        values = torch.full((n_post * n_pre,), 0.5, device=DEVICE)

        # Run 5 steps on both paths
        for _ in range(5):
            w_dense = meta_dense.compute_update(w_dense, pre, post)
            values = meta_sparse.compute_update_sparse(
                values, rows, cols, n_post, n_pre, pre, post,
            )

        # Compare
        dense_flat = w_dense.flatten()
        torch.testing.assert_close(dense_flat, values, atol=1e-6, rtol=1e-5)

        # Also compare metastate
        sparse_rates = meta_sparse.plasticity_rate[rows, cols]
        dense_rates = meta_dense.plasticity_rate.flatten()
        torch.testing.assert_close(dense_rates, sparse_rates, atol=1e-6, rtol=1e-5)


class TestNoModificationPassthrough:
    """When base strategy produces Δw = 0, metastate still recovers."""

    def test_recovery_without_modification(self):
        meta = _make_meta(base=_ZeroDeltaStrategy(), tau_recovery_ms=50.0)
        meta.setup(4, 3, DEVICE)
        meta.plasticity_rate.fill_(0.3)

        w = torch.full((3, 4), 0.5, device=DEVICE)
        for _ in range(200):
            _step(meta, w)

        # Should recover toward ~1.0 (consolidation is 0, so rate_rest ≈ 1.0)
        assert meta.plasticity_rate.mean().item() > 0.9


class TestWrappingPreservesBase:
    """Base strategy's internal state updates correctly through wrapper."""

    def test_stdp_traces_update_through_meta(self):
        stdp = STDPStrategy(STDPConfig(a_plus=0.01, a_minus=0.012))
        meta = MetaplasticityStrategy(stdp, MetaplasticityConfig())

        n_pre, n_post = 4, 3
        meta.setup(n_pre, n_post, DEVICE)

        pre = torch.tensor([1.0, 0.0, 1.0, 0.0], device=DEVICE)
        post = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
        w = torch.full((n_post, n_pre), 0.5, device=DEVICE)

        meta.compute_update(w, pre, post)

        # STDP traces should have been updated (not all zeros)
        assert stdp._is_setup
        traces = stdp.trace_manager
        assert traces.input_trace.sum().item() > 0 or traces.output_trace.sum().item() > 0


class TestEffectiveScaling:
    """Verify that plasticity_rate actually scales the effective weight change."""

    def test_half_rate_half_change(self):
        delta = 0.02
        meta = _make_meta(delta=delta, depression_strength=0.0)  # No depression
        n_pre, n_post = 4, 3
        meta.setup(n_pre, n_post, DEVICE)
        meta.plasticity_rate.fill_(0.5)  # 50% plasticity

        w = torch.full((n_post, n_pre), 0.5, device=DEVICE)
        result = meta.compute_update(
            w,
            torch.ones(n_pre, dtype=torch.bool),
            torch.ones(n_post, dtype=torch.bool),
        )
        actual_dw = (result - w).mean().item()
        expected_dw = delta * 0.5
        assert abs(actual_dw - expected_dw) < 1e-6

    def test_full_rate_full_change(self):
        delta = 0.02
        meta = _make_meta(delta=delta, depression_strength=0.0)
        n_pre, n_post = 4, 3
        meta.setup(n_pre, n_post, DEVICE)
        meta.plasticity_rate.fill_(1.0)

        w = torch.full((n_post, n_pre), 0.5, device=DEVICE)
        result = meta.compute_update(
            w,
            torch.ones(n_pre, dtype=torch.bool),
            torch.ones(n_post, dtype=torch.bool),
        )
        actual_dw = (result - w).mean().item()
        assert abs(actual_dw - delta) < 1e-6


class TestTagAndCaptureInteraction:
    """Verify correct nesting with TagAndCaptureStrategy."""

    def test_meta_wrapping_tag_and_capture(self):
        base = ThreeFactorStrategy(ThreeFactorConfig(learning_rate=0.01))
        tac = TagAndCaptureStrategy(base, TagAndCaptureConfig(tag_decay=0.95))
        meta = MetaplasticityStrategy(tac, MetaplasticityConfig())

        n_pre, n_post = 4, 3
        meta.setup(n_pre, n_post, DEVICE)

        pre = torch.tensor([1.0, 0.0, 1.0, 0.0], device=DEVICE)
        post = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
        w = torch.full((n_post, n_pre), 0.5, device=DEVICE)

        # Should not raise
        result = meta.compute_update(w, pre, post, modulator=0.5)
        assert result.shape == w.shape

        # Tags should have been updated inside T&C
        assert hasattr(tac, "tags")
        assert tac.tags.shape == (n_post, n_pre)
