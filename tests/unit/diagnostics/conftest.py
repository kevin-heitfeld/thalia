"""Pytest fixtures for diagnostics unit tests."""

import pytest
import torch

from thalia.config import BrainConfig


@pytest.fixture
def device():
    """Device for testing (CPU by default)."""
    return torch.device("cpu")


@pytest.fixture
def brain_config(device):
    """Brain configuration for testing."""
    return BrainConfig(
        device=str(device),
        dt_ms=1.0,
    )
