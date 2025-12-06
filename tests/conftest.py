"""Shared test fixtures and configuration."""

import pytest
import torch
import numpy as np


@pytest.fixture(autouse=True)
def set_random_seed():
    """Ensure reproducible tests by setting all random seeds.
    
    This fixture runs automatically for every test to ensure deterministic behavior.
    Tests that explicitly need different seeds can override this.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # For even more reproducibility (may slow down tests slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
