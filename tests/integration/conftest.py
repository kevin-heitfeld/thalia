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

from thalia.core import LIFNeuron, ConductanceLIF, DendriticNeuron
from thalia.learning import STDPRule, BCMRule, ThreeFactorSTDP, UnifiedHomeostasis
from thalia.learning import EIBalanceRegulator, IntrinsicPlasticity
from thalia.core import DivisiveNormalization
from thalia.regions import LayeredCortex, Striatum, Prefrontal, Cerebellum
from thalia.regions.cortex import LayeredCortexConfig
from thalia.regions.hippocampus import TrisynapticHippocampus, TrisynapticConfig


# ==============================================================================
# Level 0: PRIMITIVES
# ==============================================================================

@pytest.fixture
def lif_neuron():
    """Basic LIF neuron for testing."""
    return LIFNeuron(
        v_thresh=1.0,
        v_rest=0.0,
        tau_mem=20.0,
        tau_refrac=2.0,
    )


@pytest.fixture
def conductance_neuron():
    """Conductance-based LIF neuron for testing."""
    return ConductanceLIF(
        v_thresh=1.0,
        v_rest=-70.0,
        tau_mem=20.0,
        e_exc=0.0,
        e_inh=-80.0,
    )


@pytest.fixture
def dendritic_neuron():
    """Dendritic neuron with nonlinear integration."""
    return DendriticNeuron(
        n_branches=4,
        inputs_per_branch=64,
        branch_threshold=5.0,
        branch_gain=2.0,
    )


@pytest.fixture
def small_weight_matrix():
    """Small weight matrix for testing learning rules."""
    return torch.randn(32, 64) * 0.1


# ==============================================================================
# Level 1: LEARNING RULES
# ==============================================================================

@pytest.fixture
def stdp_rule():
    """STDP learning rule."""
    return STDPRule(
        a_plus=0.01,
        a_minus=0.01,
        tau_plus=20.0,
        tau_minus=20.0,
    )


@pytest.fixture
def bcm_rule():
    """BCM learning rule."""
    return BCMRule(
        learning_rate=0.001,
        tau_theta=10000.0,
        p=2.0,
    )


@pytest.fixture
def three_factor_stdp():
    """Three-factor STDP with dopamine modulation."""
    return ThreeFactorSTDP(
        a_plus=0.01,
        a_minus=0.01,
        tau_plus=20.0,
        tau_minus=20.0,
        tau_eligibility=1000.0,
    )


# ==============================================================================
# Level 2: STABILITY MECHANISMS
# ==============================================================================

@pytest.fixture
def homeostasis():
    """Unified homeostasis mechanism."""
    return UnifiedHomeostasis(
        target_rate=0.05,
        tau_homeostasis=10000.0,
        adaptation_lr=0.0001,
        tau_avg_rate=1000.0,
    )


@pytest.fixture
def ei_balance():
    """E/I balance regulator."""
    return EIBalanceRegulator(
        target_ratio=4.0,
        adaptation_rate=0.01,
    )


@pytest.fixture
def intrinsic_plasticity():
    """Intrinsic plasticity for threshold adaptation."""
    return IntrinsicPlasticity(
        target_rate=0.05,
        learning_rate=0.0001,
        tau_avg=1000.0,
    )


@pytest.fixture
def divisive_norm():
    """Divisive normalization for gain control."""
    return DivisiveNormalization(
        semi_saturation=1.0,
        epsilon=1e-6,
    )


# ==============================================================================
# Level 3: REGIONS (Isolated)
# ==============================================================================

@pytest.fixture
def layered_cortex():
    """LayeredCortex region for integration testing."""
    config = LayeredCortexConfig(
        n_minicolumns=8,
        neurons_per_minicolumn=16,
        use_dendritic_nonlinearity=False,  # Simpler for testing
    )
    return LayeredCortex(config)


@pytest.fixture
def layered_cortex_with_robustness():
    """LayeredCortex with robustness mechanisms enabled."""
    from thalia.config import RobustnessConfig
    
    config = LayeredCortexConfig(
        n_minicolumns=8,
        neurons_per_minicolumn=16,
        robustness=RobustnessConfig.stable(),
    )
    return LayeredCortex(config)


@pytest.fixture
def hippocampus():
    """Trisynaptic hippocampus for integration testing."""
    config = TrisynapticConfig(
        dg_size=64,
        ca3_size=64,
        ca1_size=64,
    )
    return TrisynapticHippocampus(config)


@pytest.fixture
def striatum():
    """Striatum region for integration testing."""
    return Striatum(
        n_units=64,
        action_dim=4,
    )


@pytest.fixture
def prefrontal():
    """Prefrontal cortex for integration testing."""
    return Prefrontal(
        n_units=64,
        working_memory_capacity=8,
    )


@pytest.fixture
def cerebellum():
    """Cerebellum for integration testing."""
    return Cerebellum(
        n_mossy=64,
        n_granule=256,
        n_purkinje=32,
    )


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
