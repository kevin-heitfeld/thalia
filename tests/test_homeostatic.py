"""Tests for homeostatic plasticity mechanisms."""

import pytest
import torch

from thalia.learning.homeostatic import (
    IntrinsicPlasticity,
    IntrinsicPlasticityConfig,
    SynapticScaling,
    SynapticScalingConfig,
)


class TestIntrinsicPlasticity:
    """Tests for intrinsic plasticity (threshold adaptation)."""

    def test_initialization(self):
        """Test IP initializes correctly."""
        ip = IntrinsicPlasticity(n_neurons=100)
        ip.reset(batch_size=16)

        assert ip.rate_avg.shape == (16, 100)
        assert ip.rate_avg.sum() == 0

    def test_threshold_increase_on_high_activity(self):
        """Test threshold increases when firing rate is too high."""
        config = IntrinsicPlasticityConfig(target_rate=10.0, learning_rate=0.1)
        ip = IntrinsicPlasticity(n_neurons=10, config=config)
        ip.reset(batch_size=1)

        # Simulate high activity (100% firing)
        for _ in range(100):
            delta = ip(torch.ones(1, 10))

        # Threshold should increase (positive delta)
        assert delta.mean() > 0

    def test_threshold_decrease_on_low_activity(self):
        """Test threshold decreases when firing rate is too low."""
        config = IntrinsicPlasticityConfig(target_rate=10.0, learning_rate=0.1)
        ip = IntrinsicPlasticity(n_neurons=10, config=config)
        ip.reset(batch_size=1)

        # Simulate no activity
        for _ in range(100):
            delta = ip(torch.zeros(1, 10))

        # Threshold should decrease (negative delta)
        assert delta.mean() < 0

    def test_rate_estimate(self):
        """Test firing rate estimation."""
        # Use faster averaging for test
        config = IntrinsicPlasticityConfig(tau_avg=100.0)  # Fast averaging
        ip = IntrinsicPlasticity(n_neurons=5, config=config)
        ip.reset(batch_size=1)

        # 50% firing over many steps should give ~500 Hz estimate
        for _ in range(1000):
            spikes = (torch.rand(1, 5) > 0.5).float()
            ip(spikes)

        rate = ip.get_rate_estimate()
        # Should be approximately 500 Hz (50% * 1000 Hz max)
        # With exponential averaging, allow wider tolerance
        assert 300 < rate.mean().item() < 700


class TestSynapticScaling:
    """Tests for synaptic scaling."""

    def test_initialization(self):
        """Test SS initializes correctly."""
        ss = SynapticScaling(n_neurons=50)
        ss.reset(batch_size=8)

        assert ss.rate_avg.shape == (8, 50)
        assert ss.scale.shape == (8, 50)
        assert (ss.scale == 1.0).all()

    def test_scale_up_on_low_activity(self):
        """Test scaling increases when activity is low."""
        config = SynapticScalingConfig(target_rate=50.0, learning_rate=0.01)
        ss = SynapticScaling(n_neurons=10, config=config)
        ss.reset(batch_size=1)

        # Low activity
        for _ in range(1000):
            ss(torch.zeros(1, 10))

        # Scale should increase above 1
        assert ss.scale.mean() > 1.0

    def test_scale_down_on_high_activity(self):
        """Test scaling decreases when activity is high."""
        config = SynapticScalingConfig(target_rate=10.0, learning_rate=0.01)
        ss = SynapticScaling(n_neurons=10, config=config)
        ss.reset(batch_size=1)

        # High activity (every step)
        for _ in range(1000):
            ss(torch.ones(1, 10))

        # Scale should decrease below 1
        assert ss.scale.mean() < 1.0

    def test_scale_bounds(self):
        """Test scaling stays within configured bounds."""
        config = SynapticScalingConfig(
            target_rate=10.0,
            learning_rate=0.1,
            scale_min=0.5,
            scale_max=2.0
        )
        ss = SynapticScaling(n_neurons=10, config=config)
        ss.reset(batch_size=1)

        # Extreme activity
        for _ in range(1000):
            ss(torch.ones(1, 10))

        assert ss.scale.min() >= 0.5
        assert ss.scale.max() <= 2.0
