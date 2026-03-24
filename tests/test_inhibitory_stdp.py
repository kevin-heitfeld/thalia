"""Tests for InhibitorySTDPStrategy (Vogels et al. 2011).

Validates:
1. Config validation (tau, alpha constraints)
2. Trace decay dynamics
3. Symmetric potentiation (both pre-before-post and post-before-pre)
4. Depression from alpha offset (pre alone → LTD)
5. Homeostatic convergence toward target rate
6. Dense and sparse paths produce identical results
7. update_temporal_parameters correctly rescales decay
"""

from __future__ import annotations

import torch
import pytest

from thalia.learning.strategies import (
    InhibitorySTDPConfig,
    InhibitorySTDPStrategy,
)
from thalia.utils import decay_float


DEVICE = torch.device("cpu")
N_PRE = 10
N_POST = 8


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> InhibitorySTDPConfig:
    return InhibitorySTDPConfig(
        learning_rate=0.01,
        tau_istdp=20.0,
        alpha=0.1,
        w_min=0.0,
        w_max=1.0,
    )


@pytest.fixture
def strategy(default_config: InhibitorySTDPConfig) -> InhibitorySTDPStrategy:
    s = InhibitorySTDPStrategy(default_config)
    s.setup(N_PRE, N_POST, DEVICE)
    return s


# =============================================================================
# Config Validation
# =============================================================================


class TestInhibitorySTDPConfig:
    def test_valid_config(self) -> None:
        cfg = InhibitorySTDPConfig(learning_rate=0.001, tau_istdp=20.0, alpha=0.12)
        assert cfg.tau_istdp == 20.0
        assert cfg.alpha == 0.12

    def test_negative_tau_raises(self) -> None:
        with pytest.raises(Exception):
            InhibitorySTDPConfig(tau_istdp=-1.0)

    def test_zero_tau_raises(self) -> None:
        with pytest.raises(Exception):
            InhibitorySTDPConfig(tau_istdp=0.0)

    def test_negative_alpha_raises(self) -> None:
        with pytest.raises(Exception):
            InhibitorySTDPConfig(alpha=-0.1)

    def test_zero_alpha_valid(self) -> None:
        cfg = InhibitorySTDPConfig(alpha=0.0)
        assert cfg.alpha == 0.0


# =============================================================================
# Trace Dynamics
# =============================================================================


class TestTraceDynamics:
    def test_traces_initialized_to_zero(self, strategy: InhibitorySTDPStrategy) -> None:
        assert torch.all(strategy.pre_trace == 0)
        assert torch.all(strategy.post_trace == 0)

    def test_pre_trace_incremented_on_spike(self, strategy: InhibitorySTDPStrategy) -> None:
        weights = torch.zeros(N_POST, N_PRE)
        pre = torch.zeros(N_PRE, dtype=torch.bool)
        post = torch.zeros(N_POST, dtype=torch.bool)
        pre[0] = True

        strategy.compute_update(weights, pre, post)
        # After one step with spike at neuron 0, pre_trace[0] should be ~1 * decay + 1
        # Actually: trace = 0 * decay + 1.0 (spike added after decay)
        assert strategy.pre_trace[0] > 0.9
        assert strategy.pre_trace[1] == 0.0

    def test_trace_decays_without_spikes(self, strategy: InhibitorySTDPStrategy) -> None:
        weights = torch.zeros(N_POST, N_PRE)
        pre = torch.zeros(N_PRE, dtype=torch.bool)
        post = torch.zeros(N_POST, dtype=torch.bool)
        pre[0] = True

        # First step: spike
        strategy.compute_update(weights, pre, post)
        trace_after_spike = strategy.pre_trace[0].item()

        # Second step: no spike → should decay
        pre[0] = False
        strategy.compute_update(weights, pre, post)
        trace_after_decay = strategy.pre_trace[0].item()

        assert trace_after_decay < trace_after_spike
        # Should decay by factor ~exp(-dt/tau)
        expected = trace_after_spike * decay_float(1.0, 20.0)
        assert abs(trace_after_decay - expected) < 1e-5


# =============================================================================
# Weight Updates
# =============================================================================


class TestWeightUpdates:
    def test_pre_post_coincidence_potentiates(
        self, strategy: InhibitorySTDPStrategy, default_config: InhibitorySTDPConfig
    ) -> None:
        """When both pre and post fire, net effect should be potentiation."""
        weights = torch.full((N_POST, N_PRE), 0.5)
        pre = torch.zeros(N_PRE, dtype=torch.bool)
        post = torch.zeros(N_POST, dtype=torch.bool)
        pre[0] = True
        post[0] = True

        # Run several steps of coincident firing
        for _ in range(50):
            weights = strategy.compute_update(weights, pre, post)

        # Weight at (0,0) should have increased from 0.5
        assert weights[0, 0].item() > 0.5

    def test_pre_alone_depresses(
        self, strategy: InhibitorySTDPStrategy, default_config: InhibitorySTDPConfig
    ) -> None:
        """Pre-spike alone (no post) should cause net depression via -alpha term."""
        weights = torch.full((N_POST, N_PRE), 0.5)
        pre = torch.zeros(N_PRE, dtype=torch.bool)
        post = torch.zeros(N_POST, dtype=torch.bool)
        pre[0] = True  # Only pre fires

        for _ in range(50):
            weights = strategy.compute_update(weights, pre, post)

        # Weight should decrease due to -alpha * pre_spike term
        assert weights[0, 0].item() < 0.5

    def test_no_spikes_no_change(self, strategy: InhibitorySTDPStrategy) -> None:
        """With no spikes, weights should not change."""
        weights = torch.full((N_POST, N_PRE), 0.5)
        pre = torch.zeros(N_PRE, dtype=torch.bool)
        post = torch.zeros(N_POST, dtype=torch.bool)

        new_weights = strategy.compute_update(weights, pre, post)
        assert torch.allclose(new_weights, weights)

    def test_symmetric_potentiation(self, strategy: InhibitorySTDPStrategy) -> None:
        """Both pre-before-post and post-before-pre should potentiate.

        Unlike classical STDP where post-before-pre causes LTD, iSTDP
        has symmetric potentiation for both temporal orderings.
        """
        weights_causal = torch.full((N_POST, N_PRE), 0.5)
        weights_anticausal = torch.full((N_POST, N_PRE), 0.5)

        pre = torch.zeros(N_PRE, dtype=torch.bool)
        post = torch.zeros(N_POST, dtype=torch.bool)
        silent_pre = torch.zeros(N_PRE, dtype=torch.bool)
        silent_post = torch.zeros(N_POST, dtype=torch.bool)

        # Causal: pre fires first, then post fires
        s_causal = InhibitorySTDPStrategy(strategy.config)
        s_causal.setup(N_PRE, N_POST, DEVICE)
        pre[0] = True
        post[0] = True
        # Step 1: pre fires
        weights_causal = s_causal.compute_update(weights_causal, pre, silent_post)
        # Step 2: post fires (pre trace still active)
        weights_causal = s_causal.compute_update(weights_causal, silent_pre, post)

        # Anticausal: post fires first, then pre fires
        s_anti = InhibitorySTDPStrategy(strategy.config)
        s_anti.setup(N_PRE, N_POST, DEVICE)
        # Step 1: post fires
        weights_anticausal = s_anti.compute_update(weights_anticausal, silent_pre, post)
        # Step 2: pre fires (post trace still active)
        weights_anticausal = s_anti.compute_update(weights_anticausal, pre, silent_post)

        # Both should show potentiation at [0, 0]
        assert weights_causal[0, 0].item() > 0.5, "Causal pairing should potentiate"
        assert weights_anticausal[0, 0].item() > 0.5, "Anti-causal pairing should potentiate"


# =============================================================================
# Homeostatic Convergence
# =============================================================================


class TestHomeostaticConvergence:
    def test_high_post_rate_increases_weights(self) -> None:
        """When post fires too much, inhibitory weights should increase to suppress it."""
        cfg = InhibitorySTDPConfig(learning_rate=0.01, tau_istdp=20.0, alpha=0.05)
        s = InhibitorySTDPStrategy(cfg)
        s.setup(N_PRE, N_POST, DEVICE)

        weights = torch.full((N_POST, N_PRE), 0.3)
        pre = torch.zeros(N_PRE, dtype=torch.bool)
        post = torch.zeros(N_POST, dtype=torch.bool)

        # Simulate high post firing rate (post fires every step) + some pre activity
        pre[0] = True
        post[0] = True

        for _ in range(100):
            weights = s.compute_update(weights, pre, post)

        # With high co-firing rate, weights should increase substantially
        assert weights[0, 0].item() > 0.3

    def test_low_post_rate_decreases_weights(self) -> None:
        """When post is silent but pre fires, inhibitory weights should decrease."""
        cfg = InhibitorySTDPConfig(learning_rate=0.01, tau_istdp=20.0, alpha=0.1)
        s = InhibitorySTDPStrategy(cfg)
        s.setup(N_PRE, N_POST, DEVICE)

        weights = torch.full((N_POST, N_PRE), 0.5)
        pre = torch.ones(N_PRE, dtype=torch.bool)  # All pre fire
        post = torch.zeros(N_POST, dtype=torch.bool)  # No post fires

        for _ in range(100):
            weights = s.compute_update(weights, pre, post)

        # Pre-only causes -alpha depression → weights should decrease
        assert weights[0, 0].item() < 0.5


# =============================================================================
# Sparse vs Dense Consistency
# =============================================================================


class TestSparseConsistency:
    def test_sparse_matches_dense(self) -> None:
        """Sparse-native path should produce identical results to dense path."""
        cfg = InhibitorySTDPConfig(learning_rate=0.01, tau_istdp=20.0, alpha=0.1)

        # Dense strategy
        s_dense = InhibitorySTDPStrategy(cfg)
        s_dense.setup(N_PRE, N_POST, DEVICE)

        # Sparse strategy
        s_sparse = InhibitorySTDPStrategy(cfg)
        s_sparse.setup(N_PRE, N_POST, DEVICE)

        # Create sparse representation of a fully-connected weight matrix
        weights = torch.rand(N_POST, N_PRE) * 0.5 + 0.1
        row_idx = []
        col_idx = []
        vals = []
        for i in range(N_POST):
            for j in range(N_PRE):
                if weights[i, j] > 0:
                    row_idx.append(i)
                    col_idx.append(j)
                    vals.append(weights[i, j].item())
        row_indices = torch.tensor(row_idx, dtype=torch.long)
        col_indices = torch.tensor(col_idx, dtype=torch.long)
        values = torch.tensor(vals)

        # Generate mixed spikes
        torch.manual_seed(42)
        pre = torch.zeros(N_PRE, dtype=torch.bool)
        post = torch.zeros(N_POST, dtype=torch.bool)
        pre[:3] = True
        post[:2] = True

        # Run both paths
        dense_result = s_dense.compute_update(weights.clone(), pre, post)
        sparse_result = s_sparse.compute_update_sparse(
            values.clone(), row_indices, col_indices,
            N_POST, N_PRE, pre, post,
        )

        # Reconstruct dense from sparse
        dense_from_sparse = torch.zeros(N_POST, N_PRE)
        dense_from_sparse[row_indices, col_indices] = sparse_result

        assert torch.allclose(dense_result, dense_from_sparse, atol=1e-6), (
            f"Max diff: {(dense_result - dense_from_sparse).abs().max().item()}"
        )


# =============================================================================
# Temporal Parameter Update
# =============================================================================


class TestTemporalParameters:
    def test_update_temporal_parameters(self, default_config: InhibitorySTDPConfig) -> None:
        """Changing dt should update decay factors."""
        s = InhibitorySTDPStrategy(default_config)
        s.setup(N_PRE, N_POST, DEVICE)

        original_decay = s._decay_val
        s.update_temporal_parameters(2.0)  # Double the timestep
        new_decay = s._decay_val

        # Larger dt → more decay per step (smaller decay factor)
        expected = decay_float(2.0, 20.0)
        assert abs(new_decay - expected) < 1e-10
        assert new_decay != original_decay


# =============================================================================
# Ensure Setup
# =============================================================================


class TestEnsureSetup:
    def test_ensure_setup_initializes(self, default_config: InhibitorySTDPConfig) -> None:
        """ensure_setup should lazily initialize buffers."""
        s = InhibitorySTDPStrategy(default_config)
        assert not hasattr(s, "pre_trace")

        pre = torch.zeros(5, dtype=torch.bool)
        post = torch.zeros(3, dtype=torch.bool)
        weights = torch.zeros(3, 5)

        s.compute_update(weights, pre, post)
        assert hasattr(s, "pre_trace")
        assert s.pre_trace.shape == (5,)
        assert s.post_trace.shape == (3,)

    def test_ensure_setup_reinitializes_on_size_change(
        self, default_config: InhibitorySTDPConfig
    ) -> None:
        """If population sizes change, ensure_setup should reinitialize."""
        s = InhibitorySTDPStrategy(default_config)
        s.setup(5, 3, DEVICE)
        assert s.pre_trace.shape == (5,)

        # Change sizes
        s.ensure_setup(10, 6, DEVICE)
        assert s.pre_trace.shape == (10,)
        assert s.post_trace.shape == (6,)
