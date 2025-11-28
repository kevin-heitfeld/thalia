"""Tests for reward-modulated STDP."""

import pytest
import torch

from thalia.learning.reward import RewardModulatedSTDP, RSTDPConfig


class TestRewardModulatedSTDP:
    """Tests for R-STDP with eligibility traces."""

    def test_initialization(self):
        """Test R-STDP initializes correctly."""
        rstdp = RewardModulatedSTDP(n_pre=100, n_post=50)
        rstdp.reset(batch_size=8)

        assert rstdp.trace_pre.shape == (8, 100)
        assert rstdp.trace_post.shape == (8, 50)
        assert rstdp.eligibility.shape == (100, 50)

    def test_eligibility_accumulates(self):
        """Test eligibility trace accumulates from STDP."""
        rstdp = RewardModulatedSTDP(n_pre=10, n_post=10)
        rstdp.reset(batch_size=1)

        # Initial eligibility is zero
        assert rstdp.eligibility.abs().sum() == 0

        # Correlated activity should create eligibility
        for _ in range(10):
            pre = torch.zeros(1, 10)
            pre[0, 0] = 1
            rstdp.update_traces(pre, torch.zeros(1, 10))

            post = torch.zeros(1, 10)
            post[0, 0] = 1
            rstdp.update_traces(torch.zeros(1, 10), post)

        # Should have positive eligibility for synapse (0,0)
        assert rstdp.eligibility[0, 0] > 0

    def test_eligibility_decays(self):
        """Test eligibility trace decays over time."""
        config = RSTDPConfig(tau_eligibility=100.0)  # Fast decay for testing
        rstdp = RewardModulatedSTDP(n_pre=5, n_post=5, config=config)
        rstdp.reset(batch_size=1)

        # Create some eligibility
        rstdp.update_traces(torch.ones(1, 5), torch.zeros(1, 5))
        rstdp.update_traces(torch.zeros(1, 5), torch.ones(1, 5))
        initial = rstdp.eligibility.clone()

        # Let it decay
        for _ in range(100):
            rstdp.update_traces(torch.zeros(1, 5), torch.zeros(1, 5))

        # Should have decayed
        assert rstdp.eligibility.abs().sum() < initial.abs().sum()

    def test_positive_reward_strengthens(self):
        """Test positive reward creates positive weight change for positive eligibility."""
        rstdp = RewardModulatedSTDP(n_pre=5, n_post=5)
        rstdp.reset(batch_size=1)

        # Create positive eligibility (pre before post = LTP)
        rstdp.update_traces(torch.ones(1, 5), torch.zeros(1, 5))
        rstdp.update_traces(torch.zeros(1, 5), torch.ones(1, 5))

        # Apply positive reward
        dw = rstdp.apply_reward(reward=1.0)

        # Weight change should be positive (strengthening)
        assert dw.sum() > 0

    def test_negative_reward_weakens(self):
        """Test negative reward (punishment) weakens eligible synapses."""
        rstdp = RewardModulatedSTDP(n_pre=5, n_post=5)
        rstdp.reset(batch_size=1)

        # Create positive eligibility
        rstdp.update_traces(torch.ones(1, 5), torch.zeros(1, 5))
        rstdp.update_traces(torch.zeros(1, 5), torch.ones(1, 5))

        # Apply negative reward (punishment)
        dw = rstdp.apply_reward(reward=-1.0)

        # Weight change should be negative (weakening)
        assert dw.sum() < 0

    def test_no_reward_no_change(self):
        """Test zero reward produces no weight change."""
        rstdp = RewardModulatedSTDP(n_pre=5, n_post=5)
        rstdp.reset(batch_size=1)

        # Create eligibility
        rstdp.update_traces(torch.ones(1, 5), torch.ones(1, 5))

        # Zero reward
        dw = rstdp.apply_reward(reward=0.0)

        assert dw.sum() == 0

    def test_apply_and_update_clamps(self):
        """Test apply_reward_and_update respects weight bounds."""
        config = RSTDPConfig(learning_rate=10.0, w_min=0.0, w_max=1.0)
        rstdp = RewardModulatedSTDP(n_pre=5, n_post=5, config=config)
        rstdp.reset(batch_size=1)

        # Create large eligibility
        for _ in range(10):
            rstdp.update_traces(torch.ones(1, 5), torch.ones(1, 5))

        weights = torch.ones(5, 5) * 0.5

        # Large reward should push toward max
        new_weights = rstdp.apply_reward_and_update(weights, reward=100.0)
        assert new_weights.max() <= 1.0

        # Large punishment should push toward min
        new_weights = rstdp.apply_reward_and_update(weights, reward=-100.0)
        assert new_weights.min() >= 0.0
