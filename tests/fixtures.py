"""Pytest fixtures for standard test configurations and objects.

This module provides reusable fixtures for creating brain region configs,
neuron models, and other commonly used test objects with sensible defaults.
"""

import pytest
import torch

from thalia.regions.cortex import LayeredCortex, LayeredCortexConfig
from thalia.regions.cerebellum import Cerebellum, CerebellumConfig
from thalia.regions.striatum import Striatum, StriatumConfig
from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
from thalia.regions.hippocampus import TrisynapticHippocampus, TrisynapticConfig
from thalia.core.neuron import LIFNeuron, LIFConfig, ConductanceLIF
from thalia.core.dendritic import DendriticNeuron


# =============================================================================
# STANDARD DIMENSIONS
# =============================================================================

@pytest.fixture
def small_n_input():
    """Small input size for fast tests."""
    return 32


@pytest.fixture
def small_n_output():
    """Small output size for fast tests."""
    return 16


@pytest.fixture
def small_vocab_size():
    """Small vocabulary for language tests."""
    return 100


@pytest.fixture
def small_n_timesteps():
    """Small number of timesteps for tests."""
    return 10


# =============================================================================
# BRAIN REGION CONFIGS
# =============================================================================

@pytest.fixture
def layered_cortex_config(small_n_input, small_n_output):
    """Minimal LayeredCortexConfig for fast tests."""
    return LayeredCortexConfig(
        n_input=small_n_input,
        n_output=small_n_output,
        l4_ratio=1.0,
        l23_ratio=1.5,
        l5_ratio=1.0,
        dual_output=True,
    )


@pytest.fixture
def cerebellum_config(small_n_input, small_n_output):
    """Minimal CerebellumConfig for fast tests."""
    return CerebellumConfig(
        n_input=small_n_input,
        n_output=small_n_output,
    )


@pytest.fixture
def striatum_config(small_n_input):
    """Standard Striatum config for testing."""
    return StriatumConfig(
        n_input=small_n_input,
        n_output=8,  # 8 actions
        population_coding=True,
        neurons_per_action=10,
    )


@pytest.fixture
def prefrontal_config(small_n_input):
    """Minimal PrefrontalConfig for fast tests."""
    return PrefrontalConfig(
        n_input=small_n_input,
        n_output=small_n_input,
    )


@pytest.fixture
def hippocampus_config(small_n_input, small_n_output):
    """Minimal TrisynapticConfig for fast tests."""
    return TrisynapticConfig(
        n_input=small_n_input,
        n_output=small_n_output,
    )


# =============================================================================
# BRAIN REGIONS (Instantiated)
# =============================================================================

@pytest.fixture
def layered_cortex(layered_cortex_config):
    """Instantiated LayeredCortex for testing."""
    return LayeredCortex(layered_cortex_config)


@pytest.fixture
def cerebellum(cerebellum_config):
    """Instantiated Cerebellum for testing."""
    return Cerebellum(cerebellum_config)


@pytest.fixture
def striatum(striatum_config):
    """Instantiated Striatum for testing."""
    return Striatum(striatum_config)


@pytest.fixture
def prefrontal(prefrontal_config):
    """Instantiated Prefrontal for testing."""
    return Prefrontal(prefrontal_config)


@pytest.fixture
def hippocampus(hippocampus_config):
    """Instantiated TrisynapticHippocampus for testing."""
    return TrisynapticHippocampus(hippocampus_config)


# =============================================================================
# NEURON CONFIGS
# =============================================================================

@pytest.fixture
def lif_config():
    """Standard LIF neuron configuration for testing."""
    return LIFConfig(
        n_neurons=64,
        v_threshold=1.0,
        v_rest=0.0,
        tau_mem=20.0,
    )


@pytest.fixture
def conductance_lif_config():
    """ConductanceLIF neuron configuration for testing."""
    from thalia.core.neuron import ConductanceLIFConfig
    return ConductanceLIFConfig(
        C_m=1.0,
        g_L=0.05,
        E_L=0.0,
        E_E=3.0,
        E_I=-0.5,
        tau_E=5.0,
        tau_I=10.0,
        v_threshold=1.0,
        v_reset=0.0,
        tau_ref=2.0,
        dt=1.0,
        tau_adapt=100.0,
        adapt_increment=0.0,
        E_adapt=-0.5,
        noise_std=0.0,
    )


@pytest.fixture
def dendritic_config():
    """DendriticNeuron configuration for testing."""
    from thalia.core.dendritic import DendriticNeuronConfig, DendriticBranchConfig
    from thalia.core.neuron import ConductanceLIFConfig
    from dataclasses import dataclass, field
    
    @dataclass
    class Config:
        n_branches: int = 5
        inputs_per_branch: int = 10
        branch_config: object = field(default_factory=lambda: DendriticBranchConfig())
        soma_config: object = field(default_factory=lambda: ConductanceLIFConfig())
        input_routing: str = "fixed"
    return Config()


# =============================================================================
# NEURON MODELS
# =============================================================================

@pytest.fixture
def lif_neuron(n_neurons, lif_config):
    """Instantiated LIF neuron for testing."""
    return LIFNeuron(n_neurons=n_neurons, config=lif_config)


@pytest.fixture
def conductance_lif_neuron(n_neurons, conductance_lif_config):
    """Instantiated conductance LIF neuron for testing."""
    return ConductanceLIF(n_neurons=n_neurons, config=conductance_lif_config)


@pytest.fixture
def dendritic_neuron(dendritic_config):
    """Instantiated DendriticNeuron for testing."""
    return DendriticNeuron(n_neurons=20, config=dendritic_config)


# =============================================================================
# TEST INPUTS
# =============================================================================

@pytest.fixture
def standard_input(batch_size, n_neurons):
    """Standard random input tensor."""
    return torch.randn(batch_size, n_neurons)


@pytest.fixture
def poisson_spikes(batch_size, n_neurons, n_timesteps):
    """Poisson spike train with 10% firing rate."""
    return (torch.rand(n_timesteps, batch_size, n_neurons) < 0.1).float()


@pytest.fixture
def dense_spikes(batch_size, n_neurons, n_timesteps):
    """Dense spike train with 30% firing rate."""
    return (torch.rand(n_timesteps, batch_size, n_neurons) < 0.3).float()


@pytest.fixture
def sparse_spikes(batch_size, n_neurons, n_timesteps):
    """Very sparse spike train with 1% firing rate."""
    return (torch.rand(n_timesteps, batch_size, n_neurons) < 0.01).float()


@pytest.fixture
def binary_pattern(n_neurons):
    """Random binary pattern for testing."""
    return torch.randint(0, 2, (n_neurons,)).float()


@pytest.fixture
def sequence_patterns(n_neurons):
    """Sequence of 5 distinct binary patterns."""
    return torch.randint(0, 2, (5, n_neurons)).float()


# =============================================================================
# SMALL VARIANTS (for language tests)
# =============================================================================

@pytest.fixture
def small_n_neurons():
    """Small number of neurons for fast language tests."""
    return 64


@pytest.fixture
def test_tokens(small_vocab_size):
    """Batch of test token IDs."""
    return torch.randint(0, small_vocab_size, (2, 10))


@pytest.fixture
def test_spikes(small_n_neurons, small_n_timesteps):
    """Test spike tensor for language decoding."""
    return (torch.rand(2, 10, small_n_timesteps, small_n_neurons) > 0.9).float()


# =============================================================================
# REGION SPECS (for factory testing)
# =============================================================================

@pytest.fixture
def region_specs(layered_cortex_config, cerebellum_config, striatum_config, prefrontal_config, hippocampus_config):
    """Standard region specifications for factory testing."""
    return {
        "cortex": layered_cortex_config,
        "cerebellum": cerebellum_config,
        "striatum": striatum_config,
        "prefrontal": prefrontal_config,
        "hippocampus": hippocampus_config,
    }


# =============================================================================
# LEARNING/TRAINING HELPERS
# =============================================================================

@pytest.fixture
def learning_input_pattern(n_neurons):
    """Consistent input pattern for learning tests."""
    torch.manual_seed(42)  # Ensure reproducibility
    return torch.randn(1, n_neurons) * 2.0


@pytest.fixture
def target_spike_pattern(n_neurons):
    """Target spike pattern for supervised learning tests."""
    torch.manual_seed(43)
    return (torch.rand(1, n_neurons) > 0.7).float()


@pytest.fixture
def reward_signal():
    """Reward signal for reinforcement learning tests."""
    return torch.tensor(1.0)


@pytest.fixture
def dopamine_signal():
    """Dopamine signal for neuromodulation tests."""
    return 0.5


# =============================================================================
# DIAGNOSTIC HELPERS
# =============================================================================

@pytest.fixture
def test_weights(n_neurons):
    """Test weight matrix."""
    return torch.randn(n_neurons, n_neurons).abs()


@pytest.fixture
def test_membrane(batch_size, n_neurons):
    """Test membrane potential tensor."""
    return torch.randn(batch_size, n_neurons) * 0.5


# =============================================================================
# SCOPE VARIANTS
# =============================================================================

@pytest.fixture(scope="module")
def expensive_neuron_model(n_neurons, lif_config):
    """Module-scoped neuron for tests that need the same model.

    Use this when multiple tests need the same neuron model and
    creating it is expensive. Resets state between tests.
    """
    model = LIFNeuron(n_neurons=n_neurons, config=lif_config)

    # Reset state before each test that uses this fixture
    @pytest.fixture(autouse=True)
    def reset_model():
        model.reset_state()
        yield

    return model


@pytest.fixture(scope="session")
def test_device():
    """Session-scoped device (prefer GPU if available).

    Use this for consistency across all tests in a session.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
