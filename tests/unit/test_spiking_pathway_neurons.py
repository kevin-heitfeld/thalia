"""Test SpikingPathway with ConductanceLIF neurons."""

import pytest
import torch

from thalia.integration.spiking_pathway import (
    SpikingPathway,
    SpikingPathwayConfig,
    TemporalCoding,
)


def test_spiking_pathway_uses_conductance_lif():
    """Verify SpikingPathway uses ConductanceLIF neurons."""
    config = SpikingPathwayConfig(
        source_size=64,
        target_size=32,
        device=torch.device("cpu"),
    )
    pathway = SpikingPathway(config)

    # Should have neurons attribute
    assert hasattr(pathway, "neurons")
    assert pathway.neurons is not None

    # Check neuron type
    from thalia.core.neuron import ConductanceLIF

    assert isinstance(pathway.neurons, ConductanceLIF)
    assert pathway.neurons.n_neurons == 32


def test_spiking_pathway_forward_with_neurons():
    """Test forward pass uses ConductanceLIF for spike generation."""
    config = SpikingPathwayConfig(
        source_size=64,
        target_size=32,
        device=torch.device("cpu"),
    )
    pathway = SpikingPathway(config)

    # Create input spikes
    input_spikes = torch.zeros(64)
    input_spikes[0:10] = 1.0  # 10 spikes

    # Forward pass
    output_spikes = pathway(input_spikes)

    # Should return boolean spikes
    assert output_spikes.dtype == torch.bool
    assert output_spikes.shape == (32,)

    # Neuron membrane should be updated
    assert pathway.neurons.membrane is not None
    assert pathway.neurons.membrane.shape == (32,)


def test_spiking_pathway_phase_coding():
    """Test phase coding modulation works with ConductanceLIF."""
    config = SpikingPathwayConfig(
        source_size=64,
        target_size=32,
        temporal_coding=TemporalCoding.PHASE,
        oscillation_freq_hz=8.0,  # Theta
        phase_precision=0.5,
        device=torch.device("cpu"),
    )
    pathway = SpikingPathway(config)

    # Create strong input
    input_spikes = torch.ones(64)

    # Forward at different phases
    output_at_phase_0 = pathway(input_spikes, time_ms=0.0)
    pathway.reset_state()

    output_at_phase_pi = pathway(input_spikes, time_ms=62.5)  # 1/8 Hz = 125ms, half = 62.5ms
    pathway.reset_state()

    # Should get different spike counts at different phases
    # (though both should be boolean tensors)
    assert output_at_phase_0.dtype == torch.bool
    assert output_at_phase_pi.dtype == torch.bool


def test_spiking_pathway_reset_resets_neurons():
    """Verify reset_state() resets neuron state."""
    config = SpikingPathwayConfig(
        source_size=64,
        target_size=32,
        device=torch.device("cpu"),
    )
    pathway = SpikingPathway(config)

    # Run forward to change state
    input_spikes = torch.ones(64)
    pathway(input_spikes)

    # Reset
    pathway.reset_state()

    # Membrane should be back to rest
    assert torch.allclose(
        pathway.neurons.membrane,
        torch.full((32,), pathway.config.v_rest),
    )


def test_spiking_pathway_add_neurons_expands_neuron_object():
    """Test add_neurons() expands ConductanceLIF."""
    config = SpikingPathwayConfig(
        source_size=64,
        target_size=32,
        device=torch.device("cpu"),
    )
    pathway = SpikingPathway(config)

    # Initial size
    assert pathway.neurons.n_neurons == 32
    assert pathway.weights.shape == (32, 64)

    # Add neurons
    pathway.add_neurons(n_new=8)

    # Should have expanded
    assert pathway.config.target_size == 40
    assert pathway.neurons.n_neurons == 40
    assert pathway.weights.shape == (40, 64)
    assert pathway.neurons.membrane.shape == (40,)


def test_spiking_pathway_state_includes_neuron_state():
    """Test get_state/load_state handle neuron state."""
    config = SpikingPathwayConfig(
        source_size=64,
        target_size=32,
        device=torch.device("cpu"),
    )
    pathway = SpikingPathway(config)

    # Run forward to change state
    input_spikes = torch.ones(64)
    pathway(input_spikes)

    # Get state
    state = pathway.get_state()

    # Should include neuron state
    assert "neuron_state" in state
    assert "neurons" in state["neuron_state"]

    # Create new pathway
    pathway2 = SpikingPathway(config)

    # Load state
    pathway2.load_state(state)

    # Neuron state should match
    assert torch.allclose(pathway2.neurons.membrane, pathway.neurons.membrane)


def test_spiking_pathway_learning_with_neurons():
    """Test STDP learning works with ConductanceLIF neurons."""
    config = SpikingPathwayConfig(
        source_size=64,
        target_size=32,
        stdp_lr=0.01,
        device=torch.device("cpu"),
    )
    pathway = SpikingPathway(config)

    initial_weights = pathway.weights.data.clone()

    # Run multiple timesteps with correlated activity
    for _ in range(10):
        input_spikes = torch.zeros(64)
        input_spikes[0:10] = 1.0
        pathway(input_spikes)

    # Weights should have changed (STDP during forward)
    assert not torch.allclose(pathway.weights.data, initial_weights)

    # Get learning metrics
    metrics = pathway.get_learning_metrics()
    assert "total_ltp" in metrics
    assert "total_ltd" in metrics


def test_spiking_pathway_diagnostics_use_neuron_state():
    """Test get_diagnostics() reports neuron membrane stats."""
    config = SpikingPathwayConfig(
        source_size=64,
        target_size=32,
        device=torch.device("cpu"),
    )
    pathway = SpikingPathway(config)

    # Run forward
    input_spikes = torch.ones(64)
    pathway(input_spikes)

    # Get diagnostics
    diag = pathway.get_diagnostics()

    # Should include membrane stats from neurons
    assert "membrane_mean" in diag
    assert "membrane_std" in diag

    # Should match neuron state
    assert diag["membrane_mean"] == pytest.approx(
        pathway.neurons.membrane.mean().item(),
        rel=1e-5,
    )
