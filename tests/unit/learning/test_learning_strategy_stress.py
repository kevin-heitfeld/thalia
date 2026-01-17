"""
Stress tests for learning strategies - robustness and edge cases.

Tests numerical stability, extreme inputs, and convergence properties
for all learning rule implementations. Part of Phase 1.3 test improvements.

Tests:
- NaN/Inf handling
- Extreme weight values
- Zero activity patterns
- Convergence properties
- Multi-timestep stability
- Numerical precision edge cases

Author: Thalia Test Quality Audit
Date: December 21, 2025
"""

import numpy as np
import pytest
import torch

from thalia.learning import (
    BCMStrategy,
    BCMStrategyConfig,
    HebbianConfig,
    HebbianStrategy,
    STDPConfig,
    STDPStrategy,
    ThreeFactorConfig,
    ThreeFactorStrategy,
)


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


class TestHebbianStress:
    """Stress tests for Hebbian learning rule."""

    def test_hebbian_handles_zero_activity(self, device):
        """Hebbian should handle zero pre/post activity gracefully."""
        config = HebbianConfig(learning_rate=0.1, w_min=0.0, w_max=1.0)
        strategy = HebbianStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5
        pre = torch.zeros(10, device=device)
        post = torch.zeros(5, device=device)

        new_weights, metrics = strategy.compute_update(weights, pre, post)

        # No change with zero activity
        assert torch.allclose(new_weights, weights)
        assert not torch.isnan(new_weights).any()
        assert not torch.isinf(new_weights).any()
        assert metrics["mean_change"] == 0.0

    def test_hebbian_clamps_extreme_weights(self, device):
        """Hebbian should respect weight bounds with extreme inputs."""
        config = HebbianConfig(learning_rate=1.0, w_min=0.0, w_max=1.0)  # High LR
        strategy = HebbianStrategy(config)

        # Start at max
        weights = torch.ones(5, 10, device=device)
        pre = torch.ones(10, device=device)
        post = torch.ones(5, device=device)

        new_weights, _ = strategy.compute_update(weights, pre, post)

        # Should stay clamped at max
        assert torch.allclose(new_weights, torch.ones_like(new_weights))
        assert new_weights.max() <= 1.0
        assert not torch.isnan(new_weights).any()

    def test_hebbian_stable_over_many_timesteps(self, device):
        """Hebbian should remain stable over extended simulation."""
        config = HebbianConfig(learning_rate=0.001, w_min=0.0, w_max=1.0)
        strategy = HebbianStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5

        # Run 10,000 timesteps with consistent activity
        for _ in range(10000):
            pre = torch.rand(10, device=device)
            post = torch.rand(5, device=device)
            weights, _ = strategy.compute_update(weights, pre, post)

            # Check stability invariants
            assert not torch.isnan(weights).any(), "NaN detected in weights"
            assert not torch.isinf(weights).any(), "Inf detected in weights"
            assert weights.min() >= 0.0, "Weights below minimum"
            assert weights.max() <= 1.0, "Weights above maximum"

    def test_hebbian_handles_sparse_activity(self, device):
        """Hebbian should handle sparse (1% active) patterns."""
        config = HebbianConfig(learning_rate=0.1, w_min=0.0, w_max=1.0)
        strategy = HebbianStrategy(config)

        weights = torch.ones(100, 200, device=device) * 0.5

        # Only 1% of neurons active
        pre = torch.rand(200, device=device) > 0.99
        post = torch.rand(100, device=device) > 0.99

        new_weights, metrics = strategy.compute_update(weights, pre.float(), post.float())

        assert not torch.isnan(new_weights).any()
        assert not torch.isinf(new_weights).any()
        assert 0.0 <= new_weights.min() <= new_weights.max() <= 1.0


class TestSTDPStress:
    """Stress tests for STDP learning rule."""

    def test_stdp_trace_decay_stability(self, device):
        """STDP traces should decay properly without activity."""
        config = STDPConfig(
            learning_rate=0.001,
            a_plus=0.01,
            a_minus=0.012,
            tau_plus=20.0,
            tau_minus=20.0,
            w_min=0.0,
            w_max=1.0,
        )
        strategy = STDPStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5

        # Initial spikes
        pre = torch.ones(10, device=device)
        post = torch.ones(5, device=device)
        weights, metrics = strategy.compute_update(weights, pre, post)
        initial_trace = metrics["pre_trace_mean"]

        # Let traces decay (100 timesteps, no spikes)
        for _ in range(100):
            pre = torch.zeros(10, device=device)
            post = torch.zeros(5, device=device)
            weights, metrics = strategy.compute_update(weights, pre, post)

        final_trace = metrics["pre_trace_mean"]

        # Traces should decay toward zero
        assert final_trace < initial_trace
        assert final_trace >= 0.0  # Never negative
        assert not torch.isnan(weights).any()

    def test_stdp_handles_continuous_spiking(self, device):
        """STDP should handle sustained high-frequency spiking."""
        config = STDPConfig(
            learning_rate=0.001,
            a_plus=0.01,
            a_minus=0.012,
            tau_plus=20.0,
            w_min=0.0,
            w_max=1.0,
        )
        strategy = STDPStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5

        # All neurons spike every timestep (extreme case)
        for _ in range(1000):
            pre = torch.ones(10, device=device)
            post = torch.ones(5, device=device)
            weights, metrics = strategy.compute_update(weights, pre, post)

            assert not torch.isnan(weights).any()
            assert not torch.isinf(weights).any()
            assert 0.0 <= weights.min() <= weights.max() <= 1.0

    def test_stdp_timing_precision(self, device):
        """STDP should differentiate pre-before-post vs post-before-pre."""
        config = STDPConfig(
            learning_rate=0.01,
            a_plus=0.01,
            a_minus=0.012,
            tau_plus=20.0,
            tau_minus=20.0,
            w_min=0.0,
            w_max=1.0,
        )
        strategy = STDPStrategy(config)

        weights_potentiate = torch.ones(5, 10, device=device) * 0.5
        weights_depress = torch.ones(5, 10, device=device) * 0.5

        # Scenario 1: Pre → Post (potentiation)
        pre = torch.ones(10, device=device)
        post = torch.zeros(5, device=device)
        weights_potentiate, _ = strategy.compute_update(weights_potentiate, pre, post)

        pre = torch.zeros(10, device=device)
        post = torch.ones(5, device=device)
        weights_potentiate, _ = strategy.compute_update(weights_potentiate, pre, post)

        # Scenario 2: Post → Pre (depression)
        pre = torch.zeros(10, device=device)
        post = torch.ones(5, device=device)
        weights_depress, _ = strategy.compute_update(weights_depress, pre, post)

        pre = torch.ones(10, device=device)
        post = torch.zeros(5, device=device)
        weights_depress, _ = strategy.compute_update(weights_depress, pre, post)

        # Potentiation should increase weights more than depression
        assert weights_potentiate.mean() > 0.5  # Increased
        assert weights_depress.mean() < 0.5  # Decreased


class TestBCMStress:
    """Stress tests for BCM learning rule."""

    def test_bcm_threshold_convergence(self, device):
        """BCM threshold should converge to average activity squared."""
        config = BCMStrategyConfig(
            learning_rate=0.01,
            tau_theta=10.0,  # Fast adaptation for testing
            theta_init=0.1,
            w_min=0.0,
            w_max=1.0,
        )
        strategy = BCMStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5

        # Constant activity level
        activity_level = 0.6
        thresholds = []

        for _ in range(500):
            pre = torch.rand(10, device=device)
            post = torch.ones(5, device=device) * activity_level
            weights, metrics = strategy.compute_update(weights, pre, post)
            thresholds.append(metrics["theta_mean"])

        # Threshold should converge to activity^2
        expected_theta = activity_level**2
        final_theta = thresholds[-1]

        assert abs(final_theta - expected_theta) < 0.05  # Within 5%
        assert not np.isnan(final_theta)

    def test_bcm_handles_zero_threshold(self, device):
        """BCM should handle case when threshold is zero."""
        config = BCMStrategyConfig(
            learning_rate=0.01,
            tau_theta=100.0,
            theta_init=0.0,  # Start at zero
            w_min=0.0,
            w_max=1.0,
        )
        strategy = BCMStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5
        pre = torch.rand(10, device=device)
        post = torch.rand(5, device=device)

        new_weights, metrics = strategy.compute_update(weights, pre, post)

        assert not torch.isnan(new_weights).any()
        assert not torch.isinf(new_weights).any()
        assert metrics["theta_mean"] >= 0.0  # Non-negative

    def test_bcm_stability_with_bimodal_activity(self, device):
        """BCM should stabilize with alternating high/low activity."""
        config = BCMStrategyConfig(
            learning_rate=0.001,
            tau_theta=50.0,
            theta_init=0.1,
            w_min=0.0,
            w_max=1.0,
        )
        strategy = BCMStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5

        for i in range(2000):
            pre = torch.rand(10, device=device)
            # Alternate between high and low activity
            post = torch.ones(5, device=device) * (0.9 if i % 2 == 0 else 0.1)
            weights, _ = strategy.compute_update(weights, pre, post)

            assert not torch.isnan(weights).any()
            assert not torch.isinf(weights).any()
            assert 0.0 <= weights.min() <= weights.max() <= 1.0


class TestThreeFactorStress:
    """Stress tests for three-factor learning rule."""

    def test_three_factor_eligibility_accumulation(self, device):
        """Eligibility traces should accumulate properly."""
        config = ThreeFactorConfig(
            learning_rate=0.01,
            eligibility_tau=50.0,
            w_min=0.0,
            w_max=1.0,
        )
        strategy = ThreeFactorStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5

        # Build eligibility over time
        eligibilities = []
        for _ in range(100):
            pre = torch.ones(10, device=device)
            post = torch.ones(5, device=device)
            weights, metrics = strategy.compute_update(
                weights, pre, post, modulator=0.0  # No learning yet
            )
            eligibilities.append(metrics["eligibility_mean"])

        # Eligibility should increase initially
        assert eligibilities[50] > eligibilities[0]
        # Should plateau (not grow unbounded)
        assert eligibilities[-1] < eligibilities[50] * 2.0

    def test_three_factor_delayed_reward(self, device):
        """Three-factor should handle delayed modulator signal."""
        config = ThreeFactorConfig(
            learning_rate=0.01,
            eligibility_tau=100.0,
            w_min=0.0,
            w_max=1.0,
        )
        strategy = ThreeFactorStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5
        initial_weights = weights.clone()

        # Activity without modulator (build eligibility)
        for _ in range(50):
            pre = torch.ones(10, device=device)
            post = torch.ones(5, device=device)
            weights, _ = strategy.compute_update(weights, pre, post, modulator=0.0)

        # Delayed reward arrives
        for _ in range(10):
            pre = torch.zeros(10, device=device)
            post = torch.zeros(5, device=device)
            weights, _ = strategy.compute_update(weights, pre, post, modulator=1.0)

        # Weights should have changed due to eligibility + delayed reward
        assert not torch.allclose(weights, initial_weights)
        assert not torch.isnan(weights).any()

    def test_three_factor_handles_negative_modulator(self, device):
        """Three-factor should handle negative modulation (punishment)."""
        config = ThreeFactorConfig(
            learning_rate=0.01,
            eligibility_tau=50.0,
            w_min=0.0,
            w_max=1.0,
        )
        strategy = ThreeFactorStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5

        # Build eligibility
        for _ in range(20):
            pre = torch.ones(10, device=device)
            post = torch.ones(5, device=device)
            weights, _ = strategy.compute_update(weights, pre, post, modulator=0.0)

        initial_weights = weights.clone()

        # Negative modulator (punishment)
        for _ in range(20):
            pre = torch.zeros(10, device=device)
            post = torch.zeros(5, device=device)
            weights, _ = strategy.compute_update(weights, pre, post, modulator=-0.5)

        # Weights should decrease with negative modulation
        assert weights.mean() < initial_weights.mean()
        assert not torch.isnan(weights).any()
        assert weights.min() >= 0.0  # Still respects bounds

    def test_three_factor_extreme_modulator(self, device):
        """Three-factor should clamp updates with extreme modulator."""
        config = ThreeFactorConfig(
            learning_rate=0.1,
            eligibility_tau=50.0,
            w_min=0.0,
            w_max=1.0,
        )
        strategy = ThreeFactorStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5

        # Build eligibility
        for _ in range(20):
            pre = torch.ones(10, device=device)
            post = torch.ones(5, device=device)
            weights, _ = strategy.compute_update(weights, pre, post, modulator=0.0)

        # Extreme positive modulator
        for _ in range(10):
            pre = torch.zeros(10, device=device)
            post = torch.zeros(5, device=device)
            weights, _ = strategy.compute_update(weights, pre, post, modulator=100.0)

        # Should be clamped at max
        assert torch.allclose(weights, torch.ones_like(weights))
        assert not torch.isnan(weights).any()


class TestWeightInitializationEdgeCases:
    """Test strategies with extreme initial weight conditions."""

    def test_hebbian_with_zero_weights(self, device):
        """Hebbian should handle zero-initialized weights."""
        config = HebbianConfig(learning_rate=0.01, w_min=0.0, w_max=1.0)
        strategy = HebbianStrategy(config)

        weights = torch.zeros(5, 10, device=device)
        pre = torch.rand(10, device=device)
        post = torch.rand(5, device=device)

        new_weights, metrics = strategy.compute_update(weights, pre, post)

        assert not torch.isnan(new_weights).any()
        assert new_weights.min() >= 0.0
        assert metrics["mean_change"] >= 0.0

    def test_stdp_with_uniform_weights(self, device):
        """STDP should handle uniform weight initialization."""
        config = STDPConfig(
            learning_rate=0.001,
            a_plus=0.01,
            a_minus=0.012,
            tau_plus=20.0,
            w_min=0.0,
            w_max=1.0,
        )
        strategy = STDPStrategy(config)

        # All weights identical
        weights = torch.ones(5, 10, device=device) * 0.5

        for _ in range(100):
            pre = torch.rand(10, device=device) > 0.9
            post = torch.rand(5, device=device) > 0.9
            weights, _ = strategy.compute_update(weights, pre.float(), post.float())

        # Should break symmetry
        assert weights.std() > 0.001
        assert not torch.isnan(weights).any()

    def test_bcm_recovers_from_extreme_threshold(self, device):
        """BCM should recover if threshold is initialized extremely high."""
        config = BCMStrategyConfig(
            learning_rate=0.01,
            tau_theta=50.0,
            theta_init=100.0,  # Extremely high
            w_min=0.0,
            w_max=1.0,
        )
        strategy = BCMStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5

        # Run with moderate activity
        for _ in range(500):
            pre = torch.rand(10, device=device)
            post = torch.ones(5, device=device) * 0.3
            weights, metrics = strategy.compute_update(weights, pre, post)

        # Threshold should have adapted down
        final_theta = metrics["theta_mean"]
        assert final_theta < 1.0  # Much lower than initial 100.0
        assert not torch.isnan(weights).any()


class TestMetricValidation:
    """Validate that learning metrics are computed correctly."""

    def test_hebbian_metrics_consistency(self, device):
        """Hebbian metrics should be self-consistent."""
        config = HebbianConfig(learning_rate=0.1, w_min=0.0, w_max=1.0)
        strategy = HebbianStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5
        pre = torch.rand(10, device=device)
        post = torch.rand(5, device=device)

        new_weights, metrics = strategy.compute_update(weights, pre, post)

        # mean_change should match actual change
        actual_change = (new_weights - weights).abs().mean().item()
        reported_change = metrics["mean_change"]
        assert abs(actual_change - reported_change) < 1e-6

        # Metrics should be non-negative
        assert metrics["mean_change"] >= 0.0

    def test_stdp_trace_metrics_bounds(self, device):
        """STDP trace metrics should stay within reasonable bounds."""
        config = STDPConfig(
            learning_rate=0.001,
            a_plus=0.01,
            a_minus=0.012,
            tau_plus=20.0,
            w_min=0.0,
            w_max=1.0,
        )
        strategy = STDPStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5

        max_pre_trace = 0.0
        max_post_trace = 0.0

        for _ in range(1000):
            pre = torch.rand(10, device=device) > 0.9
            post = torch.rand(5, device=device) > 0.9
            _, metrics = strategy.compute_update(weights, pre.float(), post.float())

            max_pre_trace = max(max_pre_trace, metrics["pre_trace_mean"])
            max_post_trace = max(max_post_trace, metrics["post_trace_mean"])

        # Traces should plateau, not grow unbounded
        assert max_pre_trace < 5.0  # Reasonable upper bound
        assert max_post_trace < 5.0

    def test_three_factor_eligibility_bounds(self, device):
        """Eligibility should not grow unbounded without modulator."""
        config = ThreeFactorConfig(
            learning_rate=0.01,
            eligibility_tau=50.0,
            w_min=0.0,
            w_max=1.0,
        )
        strategy = ThreeFactorStrategy(config)

        weights = torch.ones(5, 10, device=device) * 0.5

        eligibilities = []
        for _ in range(500):
            pre = torch.ones(10, device=device)
            post = torch.ones(5, device=device)
            weights, metrics = strategy.compute_update(weights, pre, post, modulator=0.0)
            eligibilities.append(metrics["eligibility_mean"])

        # Should plateau (eligibility decays and accumulates to equilibrium)
        assert eligibilities[-1] < eligibilities[0] * 100  # Not growing unbounded
        assert max(eligibilities) < 100.0  # Reasonable upper bound
        # Check that it plateaus (last 10% should be stable)
        recent = eligibilities[-50:]
        assert max(recent) - min(recent) < 10.0  # Stable plateau


class TestCrossStrategyStability:
    """Cross-cutting stability tests for all strategies."""

    @pytest.mark.parametrize(
        "strategy_type,config",
        [
            ("hebbian", HebbianConfig(learning_rate=0.01, w_min=0.0, w_max=1.0)),
            (
                "stdp",
                STDPConfig(
                    learning_rate=0.001,
                    a_plus=0.01,
                    a_minus=0.012,
                    tau_plus=20.0,
                    w_min=0.0,
                    w_max=1.0,
                ),
            ),
            (
                "bcm",
                BCMStrategyConfig(
                    learning_rate=0.01, tau_theta=100.0, theta_init=0.1, w_min=0.0, w_max=1.0
                ),
            ),
            (
                "three_factor",
                ThreeFactorConfig(learning_rate=0.01, eligibility_tau=100.0, w_min=0.0, w_max=1.0),
            ),
        ],
    )
    def test_strategy_handles_nan_inputs(self, strategy_type, config, device):
        """All strategies should detect and handle NaN inputs gracefully.

        Current behavior: NaN propagates through computation (PyTorch default).
        This test documents current behavior. Future improvement could add
        explicit NaN detection and error handling.
        """
        strategy_map = {
            "hebbian": HebbianStrategy,
            "stdp": STDPStrategy,
            "bcm": BCMStrategy,
            "three_factor": ThreeFactorStrategy,
        }

        strategy = strategy_map[strategy_type](config)

        weights = torch.ones(5, 10, device=device) * 0.5
        pre = torch.tensor([float("nan")] * 10, device=device)
        post = torch.ones(5, device=device)

        kwargs = {}
        if strategy_type == "three_factor":
            kwargs["modulator"] = 1.0

        # Current behavior: NaN propagates (PyTorch default)
        new_weights, _ = strategy.compute_update(weights, pre, post, **kwargs)

        # Document that NaN propagates (expected current behavior)
        assert torch.isnan(
            new_weights
        ).any(), "NaN inputs currently propagate through learning strategies"

    @pytest.mark.parametrize(
        "strategy_type,config",
        [
            ("hebbian", HebbianConfig(learning_rate=0.01, w_min=0.0, w_max=1.0)),
            (
                "stdp",
                STDPConfig(
                    learning_rate=0.001,
                    a_plus=0.01,
                    a_minus=0.012,
                    tau_plus=20.0,
                    w_min=0.0,
                    w_max=1.0,
                ),
            ),
            (
                "bcm",
                BCMStrategyConfig(
                    learning_rate=0.01, tau_theta=100.0, theta_init=0.1, w_min=0.0, w_max=1.0
                ),
            ),
            (
                "three_factor",
                ThreeFactorConfig(learning_rate=0.01, eligibility_tau=100.0, w_min=0.0, w_max=1.0),
            ),
        ],
    )
    def test_strategy_weight_conservation(self, strategy_type, config, device):
        """Weight changes should be bounded by learning rate."""
        strategy_map = {
            "hebbian": HebbianStrategy,
            "stdp": STDPStrategy,
            "bcm": BCMStrategy,
            "three_factor": ThreeFactorStrategy,
        }

        strategy = strategy_map[strategy_type](config)

        weights = torch.ones(5, 10, device=device) * 0.5
        pre = torch.rand(10, device=device)
        post = torch.rand(5, device=device)

        kwargs = {}
        if strategy_type == "three_factor":
            kwargs["modulator"] = 1.0

        new_weights, metrics = strategy.compute_update(weights, pre, post, **kwargs)

        # Weight changes should be reasonable (not jumping by huge amounts)
        max_change = (new_weights - weights).abs().max().item()

        # With learning rates around 0.01-0.001, max single-step change should be small
        assert max_change < 0.5, f"Excessive weight change: {max_change}"
        assert metrics["mean_change"] >= 0.0  # Mean absolute change is non-negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
