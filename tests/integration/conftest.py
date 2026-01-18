"""Shared fixtures for integration tests."""

import pytest
import torch

from thalia.config import BrainConfig


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def brain_config(device):
    """Create BrainConfig for testing."""
    return BrainConfig(device=device, dt_ms=1.0, theta_frequency_hz=8.0)
