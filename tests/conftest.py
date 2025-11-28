"""Shared test fixtures and configuration."""

import pytest
import torch


@pytest.fixture
def device():
    """Get available device (prefer GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 32


@pytest.fixture
def n_neurons():
    """Standard neuron count for tests."""
    return 100


@pytest.fixture
def n_timesteps():
    """Standard simulation duration."""
    return 100


@pytest.fixture
def random_spikes(batch_size, n_neurons, n_timesteps):
    """Generate random spike train for testing."""
    return (torch.rand(n_timesteps, batch_size, n_neurons) > 0.9).float()
