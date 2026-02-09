"""Integration test for Striatum DA receptor system.

Tests spiking DA integration:
- DA receptor converts VTA spikes to concentration
- Per-neuron DA modulation of learning
- D1/D2 opponent dynamics with spiking DA
"""

import pytest
import torch

from thalia.brain.configs import StriatumConfig
from thalia.brain.regions.striatum import Striatum


@pytest.fixture
def striatum_config():
    """Create minimal Striatum configuration for testing."""
    return StriatumConfig(
        n_actions=3,
        neurons_per_action=10,
        d1_fraction=0.5,
        learning_rate=0.001,
        eligibility_tau_ms=1000.0,
        w_min=0.0,
        w_max=1.0,
        fsi_ratio=0.0,  # Disable FSI for simpler test
        device="cpu",
    )


@pytest.fixture
def striatum_region(striatum_config):
    """Create Striatum region instance."""
    n_msn = striatum_config.n_actions * striatum_config.neurons_per_action
    d1_size = int(n_msn * striatum_config.d1_fraction)
    d2_size = n_msn - d1_size

    region_layer_sizes = {
        "d1_size": d1_size,
        "d2_size": d2_size,
        "n_actions": striatum_config.n_actions,
        "neurons_per_action": striatum_config.neurons_per_action,
    }

    striatum = Striatum(config=striatum_config, region_layer_sizes=region_layer_sizes)

    # Add a test input source
    striatum.add_input_source(
        source_name="test_input",
        target_layer="d1",  # Not used (both D1/D2 get weights)
        n_input=50,
        sparsity=0.0,
        weight_scale=1.0,
    )

    return striatum


def test_striatum_da_receptor_initialization(striatum_region):
    """Test DA receptors are properly initialized."""
    assert hasattr(striatum_region, "da_receptor_d1")
    assert hasattr(striatum_region, "da_receptor_d2")
    assert striatum_region.da_receptor_d1 is not None
    assert striatum_region.da_receptor_d2 is not None


def test_striatum_receives_vta_spikes(striatum_region):
    """Test Striatum can receive and process VTA DA spikes."""
    # Create mock VTA output (100 DA neurons)
    n_da_neurons = 100
    vta_spikes = torch.zeros(n_da_neurons, dtype=torch.bool)
    vta_spikes[:20] = True  # 20% firing (burst-like)

    # Create test input
    test_spikes = torch.rand(50) > 0.9  # Sparse input

    # Forward pass with VTA DA
    striatum_region._forward_internal(
        inputs={"vta:da_output": vta_spikes, "test_input": test_spikes}
    )

    # Check DA concentration was updated
    assert striatum_region._da_concentration_d1.sum() > 0, "D1 DA concentration not updated"
    assert striatum_region._da_concentration_d2.sum() > 0, "D2 DA concentration not updated"


def test_striatum_da_concentration_dynamics(striatum_region):
    """Test DA concentration rises and decays properly."""
    n_da_neurons = 100

    # Initial burst
    burst_spikes = torch.ones(n_da_neurons, dtype=torch.bool)
    test_input = torch.rand(50) > 0.9

    striatum_region._forward_internal(
        inputs={"vta:da_output": burst_spikes, "test_input": test_input}
    )
    peak_concentration = striatum_region._da_concentration_d1.mean().item()

    # Allow decay (no more DA spikes)
    for _ in range(100):  # 100ms
        striatum_region._forward_internal(inputs={"test_input": test_input})

    decayed_concentration = striatum_region._da_concentration_d1.mean().item()

    # Assert rise and decay
    assert peak_concentration > 0.01, "DA concentration failed to rise"
    assert decayed_concentration < peak_concentration, "DA concentration failed to decay"


def test_striatum_da_modulates_learning(striatum_region):
    """Test DA concentration modulates weight updates."""
    n_da_neurons = 100
    test_input = torch.rand(50) > 0.8

    # Get initial weights
    initial_weights_d1 = striatum_region.synaptic_weights["test_input_d1"].data.clone()

    # Run with high DA (burst)
    burst_spikes = torch.ones(n_da_neurons, dtype=torch.bool)
    for _ in range(10):
        striatum_region._forward_internal(
            inputs={"vta:da_output": burst_spikes, "test_input": test_input}
        )

    high_da_weights = striatum_region.synaptic_weights["test_input_d1"].data.clone()

    # Reset and run with low DA (pause)
    striatum_region.synaptic_weights["test_input_d1"].data = initial_weights_d1.clone()
    striatum_region._da_concentration_d1.zero_()
    striatum_region._da_concentration_d2.zero_()

    no_da_spikes = torch.zeros(n_da_neurons, dtype=torch.bool)
    for _ in range(10):
        striatum_region._forward_internal(
            inputs={"vta:da_output": no_da_spikes, "test_input": test_input}
        )

    low_da_weights = striatum_region.synaptic_weights["test_input_d1"].data.clone()

    # Weights should change more with DA present
    high_da_change = (high_da_weights - initial_weights_d1).abs().sum().item()
    low_da_change = (low_da_weights - initial_weights_d1).abs().sum().item()

    assert high_da_change > low_da_change, "DA failed to modulate learning"


def test_striatum_d1_d2_opponent_with_da(striatum_region):
    """Test D1/D2 opponent dynamics with spiking DA."""
    n_da_neurons = 100
    test_input = torch.rand(50) > 0.8

    # D1 should be enhanced by DA (Gs-coupled)
    # D2 should be suppressed by DA (Gi-coupled)

    # Deliver DA burst
    burst_spikes = torch.ones(n_da_neurons, dtype=torch.bool)

    for _ in range(20):
        striatum_region._forward_internal(
            inputs={"vta:da_output": burst_spikes, "test_input": test_input}
        )

    diagnostics = striatum_region.get_diagnostics()

    # Check DA receptor diagnostics exist
    assert "da_receptors" in diagnostics
    assert "d1_da_concentration_mean" in diagnostics["da_receptors"]
    assert "d2_da_concentration_mean" in diagnostics["da_receptors"]

    # Check DA concentrations are positive
    d1_da = diagnostics["da_receptors"]["d1_da_concentration_mean"]
    d2_da = diagnostics["da_receptors"]["d2_da_concentration_mean"]

    assert d1_da > 0, "D1 DA concentration should be positive after burst"
    assert d2_da > 0, "D2 DA concentration should be positive after burst"


def test_striatum_fallback_to_scalar_da(striatum_region):
    """Test Striatum falls back to scalar DA when VTA not connected."""
    test_input = torch.rand(50) > 0.8

    # Run without VTA input (should use neuromodulator_state.dopamine)
    striatum_region.neuromodulator_state.dopamine = 0.8  # High baseline

    striatum_region._forward_internal(inputs={"test_input": test_input})

    # DA concentration should match scalar value
    d1_da = striatum_region._da_concentration_d1.mean().item()
    assert abs(d1_da - 0.8) < 0.01, "Failed to fallback to scalar DA"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
