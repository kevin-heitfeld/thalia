"""Tests for learning rules."""

import pytest
import torch

from thalia.learning.stdp import STDP, STDPConfig


class TestSTDP:
    """Tests for STDP learning rule."""

    def test_stdp_initialization(self):
        """Test STDP initializes correctly."""
        stdp = STDP(n_pre=100, n_post=50)
        stdp.reset_traces(batch_size=16)

        assert stdp.trace_pre.shape == (16, 100)
        assert stdp.trace_post.shape == (16, 50)

    def test_trace_decay(self):
        """Test eligibility traces decay."""
        stdp = STDP(n_pre=10, n_post=10)
        stdp.reset_traces(batch_size=1)

        # Simulate pre spike
        pre_spikes = torch.zeros(1, 10)
        pre_spikes[0, 0] = 1

        stdp(pre_spikes, torch.zeros(1, 10))
        trace_after_spike = stdp.trace_pre[0, 0].item()

        # Step without spikes
        stdp(torch.zeros(1, 10), torch.zeros(1, 10))
        trace_after_decay = stdp.trace_pre[0, 0].item()

        assert trace_after_decay < trace_after_spike

    def test_weight_update_shape(self):
        """Test weight update has correct shape."""
        stdp = STDP(n_pre=20, n_post=10)

        pre_spikes = (torch.rand(8, 20) > 0.8).float()
        post_spikes = (torch.rand(8, 10) > 0.8).float()

        dw = stdp(pre_spikes, post_spikes)

        # Weight update should be (n_pre, n_post) or similar
        assert dw.shape == (20, 10)

    def test_potentiation(self):
        """Test LTP: post after pre should give positive update."""
        config = STDPConfig(a_plus=0.1, a_minus=0.1)
        stdp = STDP(n_pre=1, n_post=1, config=config)
        stdp.reset_traces(batch_size=1)

        # Pre spike first
        dw1 = stdp(torch.ones(1, 1), torch.zeros(1, 1))

        # Post spike after (should cause LTP)
        dw2 = stdp(torch.zeros(1, 1), torch.ones(1, 1))

        # Second update should be positive (potentiation)
        assert dw2.item() > 0

    def test_depression(self):
        """Test LTD: pre after post should give negative update."""
        config = STDPConfig(a_plus=0.1, a_minus=0.1)
        stdp = STDP(n_pre=1, n_post=1, config=config)
        stdp.reset_traces(batch_size=1)

        # Post spike first
        dw1 = stdp(torch.zeros(1, 1), torch.ones(1, 1))

        # Pre spike after (should cause LTD)
        dw2 = stdp(torch.ones(1, 1), torch.zeros(1, 1))

        # Second update should be negative (depression)
        assert dw2.item() < 0

    def test_batch_processing(self):
        """Test STDP works with batched inputs."""
        stdp = STDP(n_pre=50, n_post=30)

        for t in range(10):
            pre = (torch.rand(32, 50) > 0.9).float()
            post = (torch.rand(32, 30) > 0.9).float()
            dw = stdp(pre, post)

        # Should complete without error
        assert dw.shape == (50, 30)
