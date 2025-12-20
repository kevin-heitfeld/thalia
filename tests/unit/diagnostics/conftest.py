"""Pytest fixtures for diagnostics unit tests."""

import pytest
import torch

from thalia.config import GlobalConfig


@pytest.fixture
def device():
    """Device for testing (CPU by default)."""
    return torch.device("cpu")


@pytest.fixture
def global_config(device):
    """Global configuration for brain."""
    return GlobalConfig(
        device=str(device),
        dt_ms=1.0,
    )
