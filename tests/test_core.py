"""Tests for core SNN components."""

import pytest
import torch

from thalia.core.neuron import LIFNeuron, LIFConfig
from thalia.core.layer import SNNLayer
from thalia.core.network import SNNNetwork


class TestLIFNeuron:
    """Tests for LIF neuron model."""

    def test_initialization(self):
        """Test neuron initializes with correct dimensions."""
        neuron = LIFNeuron(n_neurons=100)
        neuron.reset_state(batch_size=32)

        assert neuron.membrane.shape == (32, 100)
        assert neuron.membrane.min().item() == neuron.config.v_rest

    def test_reset(self):
        """Test membrane reset after spike."""
        config = LIFConfig(v_threshold=1.0, v_reset=0.0)
        neuron = LIFNeuron(n_neurons=10, config=config)
        neuron.reset_state(batch_size=1)

        # Force membrane above threshold
        neuron.membrane = torch.full((1, 10), 1.5)
        spikes, _ = neuron(torch.zeros(1, 10))

        assert spikes.sum() > 0
        # Membrane should reset where spikes occurred
        assert (neuron.membrane[spikes.bool()] == config.v_reset).all()

    def test_membrane_decay(self):
        """Test membrane potential decays toward rest."""
        config = LIFConfig(v_rest=0.0, tau_mem=20.0)
        neuron = LIFNeuron(n_neurons=10, config=config)
        neuron.reset_state(batch_size=1)

        # Set initial membrane above rest
        neuron.membrane = torch.full((1, 10), 0.5)
        initial = neuron.membrane.clone()

        # Step with no input
        neuron(torch.zeros(1, 10))

        # Should decay toward rest (0.0)
        assert (neuron.membrane < initial).all()
        assert (neuron.membrane > config.v_rest).all()

    def test_input_integration(self):
        """Test that input increases membrane potential."""
        neuron = LIFNeuron(n_neurons=10)
        neuron.reset_state(batch_size=1)
        initial = neuron.membrane.clone()

        # Apply positive input (below threshold to avoid spike)
        neuron(torch.ones(1, 10) * 0.3)

        # Membrane should increase
        assert (neuron.membrane > initial).all()

    def test_spike_generation(self):
        """Test that spikes are generated when threshold is crossed."""
        config = LIFConfig(v_threshold=1.0)
        neuron = LIFNeuron(n_neurons=5, config=config)
        neuron.reset_state(batch_size=1)

        # Strong input should cause spikes
        spikes, _ = neuron(torch.ones(1, 5) * 2.0)

        assert spikes.sum() > 0
        assert ((spikes == 0) | (spikes == 1)).all()


class TestSNNLayer:
    """Tests for SNN layer."""

    def test_layer_forward(self):
        """Test layer forward pass."""
        layer = SNNLayer(n_neurons=100, input_size=50)
        layer.reset_state(batch_size=16)

        input_spikes = (torch.rand(16, 50) > 0.8).float()
        spikes, voltages = layer(input_spikes=input_spikes)

        assert spikes.shape == (16, 100)
        assert voltages.shape == (16, 100)
        assert spikes.dtype == torch.float32
        assert ((spikes == 0) | (spikes == 1)).all()

    def test_recurrent_connections(self):
        """Test layer with recurrent weights."""
        layer = SNNLayer(n_neurons=50, input_size=30, recurrent=True)
        layer.reset_state(batch_size=8)

        input_spikes = (torch.rand(8, 30) > 0.8).float()
        spikes, _ = layer(input_spikes=input_spikes)

        assert spikes.shape == (8, 50)

    def test_layer_no_input_size(self):
        """Test layer without external input."""
        layer = SNNLayer(n_neurons=50, recurrent=True)
        layer.reset_state(batch_size=4)

        # Can run with just external current
        spikes, _ = layer(external_current=torch.randn(4, 50) * 0.5)

        assert spikes.shape == (4, 50)


class TestSNNNetwork:
    """Tests for multi-layer network."""

    def test_network_construction(self):
        """Test network builds correctly."""
        network = SNNNetwork(layer_sizes=[784, 400, 100])

        assert len(network.layers) == 2  # 2 layers (not counting input)
        assert network.layers[0].n_neurons == 400
        assert network.layers[1].n_neurons == 100

    def test_network_forward(self):
        """Test full network forward pass."""
        network = SNNNetwork(layer_sizes=[100, 50, 10])
        network.reset_state(batch_size=8)

        input_spikes = (torch.rand(8, 100) > 0.8).float()
        output, all_spikes = network(input_spikes)

        assert output.shape == (8, 10)
        assert len(all_spikes) == 2  # One output per layer

    def test_network_with_recurrence(self):
        """Test network with recurrent layers."""
        network = SNNNetwork(
            layer_sizes=[50, 30, 10],
            recurrent=True,
            recurrent_connectivity=0.1
        )
        network.reset_state(batch_size=4)

        # Run multiple timesteps
        for t in range(10):
            input_spikes = (torch.rand(4, 50) > 0.9).float()
            output, _ = network(input_spikes)

        assert output.shape == (4, 10)
