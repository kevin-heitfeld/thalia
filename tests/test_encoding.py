"""Tests for encoding strategies."""

import pytest
import torch

from thalia.encoding.poisson import poisson_encode, generate_spike_train
from thalia.encoding.rate import rate_encode


class TestPoissonEncoding:
    """Tests for Poisson spike generation."""

    def test_output_shape(self):
        """Test output has correct shape."""
        rates = torch.ones(8, 100) * 50  # 50 Hz
        spikes = poisson_encode(rates, duration=100, dt=1.0)

        assert spikes.shape == (100, 8, 100)

    def test_rate_correspondence(self):
        """Test higher rates produce more spikes."""
        low_rate = torch.ones(1, 100) * 10
        high_rate = torch.ones(1, 100) * 100

        low_spikes = poisson_encode(low_rate, duration=1000, dt=1.0)
        high_spikes = poisson_encode(high_rate, duration=1000, dt=1.0)

        assert high_spikes.sum() > low_spikes.sum()

    def test_binary_output(self):
        """Test spikes are binary."""
        rates = torch.ones(1, 10) * 50
        spikes = poisson_encode(rates, duration=100)

        assert ((spikes == 0) | (spikes == 1)).all()

    def test_generate_spike_train(self):
        """Test convenience function."""
        spikes = generate_spike_train(rate=100, duration=100, n_neurons=10)

        assert spikes.shape == (100, 1, 10)


class TestRateEncoding:
    """Tests for rate encoding."""

    def test_output_shape(self):
        """Test output shape."""
        values = torch.rand(16, 50)
        spikes = rate_encode(values, duration=100)

        assert spikes.shape == (100, 16, 50)

    def test_value_mapping(self):
        """Test higher values produce more spikes."""
        low_vals = torch.zeros(1, 100)
        high_vals = torch.ones(1, 100)

        low_spikes = rate_encode(low_vals, duration=1000)
        high_spikes = rate_encode(high_vals, duration=1000)

        assert low_spikes.sum() < high_spikes.sum()

    def test_clamping(self):
        """Test values are clamped properly."""
        # Values outside [0,1]
        values = torch.tensor([[1.5, -0.5]])
        spikes = rate_encode(values, duration=100)

        # Should not crash, spikes should be valid
        assert ((spikes == 0) | (spikes == 1)).all()
