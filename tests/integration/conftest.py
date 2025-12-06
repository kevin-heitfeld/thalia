"""
Pytest configuration and fixtures for integration tests.

This file provides fixtures for testing component interactions at different
complexity levels, following the layer hierarchy:

Level 0: PRIMITIVES (neurons, synapses, traces)
Level 1: LEARNING RULES (STDP, BCM, three-factor)
Level 2: STABILITY (homeostasis, E/I balance, normalization)
Level 3: REGIONS (cortex, hippocampus, striatum, PFC - isolated)
Level 4: INTEGRATION (full brain, pathways, communication)
"""

import pytest
import torch

# Only import what we know exists and works
from thalia.learning.bcm import BCMRule
from thalia.learning.unified_homeostasis import UnifiedHomeostasis
from thalia.learning.ei_balance import EIBalanceRegulator
from thalia.learning.intrinsic_plasticity import IntrinsicPlasticity
from thalia.core.normalization import DivisiveNormalization
from thalia.regions import LayeredCortex
from thalia.regions.cortex import LayeredCortexConfig


# ==============================================================================
# Level 0: PRIMITIVES
# ==============================================================================

@pytest.fixture
def small_weight_matrix():
    """Small weight matrix for testing learning rules."""
    return torch.randn(32, 64) * 0.1


# ==============================================================================
# Level 1: LEARNING RULES
# ==============================================================================

@pytest.fixture
def bcm_rule():
    """BCM learning rule."""
    return BCMRule(n_neurons=64)


# ==============================================================================
# Level 2: STABILITY MECHANISMS
# ==============================================================================

@pytest.fixture
def homeostasis():
    """Unified homeostasis mechanism."""
    return UnifiedHomeostasis(n_neurons=64)


@pytest.fixture
def ei_balance():
    """E/I balance regulator."""
    return EIBalanceRegulator()


@pytest.fixture
def intrinsic_plasticity():
    """Intrinsic plasticity for threshold adaptation."""
    return IntrinsicPlasticity(n_neurons=64)


@pytest.fixture
def divisive_norm():
    """Divisive normalization for gain control."""
    return DivisiveNormalization()


# ==============================================================================
# Level 3: REGIONS (Isolated)
# ==============================================================================

@pytest.fixture
def layered_cortex():
    """LayeredCortex region for integration testing."""
    config = LayeredCortexConfig(
        n_input=128,
        n_output=64,
    )
    return LayeredCortex(config)


@pytest.fixture
def layered_cortex_with_robustness():
    """LayeredCortex with robustness mechanisms enabled."""
    from thalia.config import RobustnessConfig

    config = LayeredCortexConfig(
        n_input=128,
        n_output=64,
        robustness=RobustnessConfig.stable(),
    )
    return LayeredCortex(config)


# TODO: Restore these fixtures when the respective classes are available
# These were temporarily commented out during Phase 1 integration test development

# @pytest.fixture
# def hippocampus():
#     """Trisynaptic hippocampus for integration testing."""
#     from thalia.regions.hippocampus import TrisynapticHippocampus, TrisynapticConfig
#     config = TrisynapticConfig(
#         dg_size=64,
#         ca3_size=64,
#         ca1_size=64,
#     )
#     return TrisynapticHippocampus(config)


# @pytest.fixture
# def striatum():
#     """Striatum region for integration testing."""
#     from thalia.regions.striatum import Striatum
#     return Striatum(
#         n_units=64,
#         action_dim=4,
#     )


# @pytest.fixture
# def prefrontal():
#     """Prefrontal cortex for integration testing."""
#     from thalia.regions.prefrontal import Prefrontal
#     return Prefrontal(
#         n_units=64,
#         working_memory_capacity=8,
#     )


# @pytest.fixture
# def cerebellum():
#     """Cerebellum for integration testing."""
#     from thalia.regions.cerebellum import Cerebellum
#     return Cerebellum(
#         n_mossy=64,
#         n_granule=256,
#         n_purkinje=32,
#     )


# ==============================================================================
# Level 4: INTEGRATION (Helper fixtures)
# ==============================================================================

@pytest.fixture
def dummy_sensory_input():
    """Dummy sensory input for full brain testing."""
    return torch.randn(1, 128, 512)  # [batch, time, features]


@pytest.fixture
def dummy_reward_signal():
    """Dummy reward signal for reinforcement learning testing."""
    return torch.tensor([1.0])  # Positive reward


# ==============================================================================
# Test Configuration
# ==============================================================================

@pytest.fixture
def health_monitor():
    """Health monitor for detecting issues during integration tests."""
    from thalia.diagnostics import HealthMonitor, HealthConfig
    return HealthMonitor(HealthConfig())


@pytest.fixture
def strict_health_config():
    """Strict health configuration for catching subtle issues."""
    from thalia.diagnostics import HealthConfig
    return HealthConfig(
        spike_rate_min=0.005,  # Stricter
        spike_rate_max=0.3,    # Stricter
        severity_threshold=5.0,  # Report even minor issues
    )
