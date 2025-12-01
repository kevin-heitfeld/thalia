"""Tests for core SNN components."""

import pytest
import torch

from thalia.core.neuron import LIFNeuron, LIFConfig, ConductanceLIF, ConductanceLIFConfig
from thalia.core.layer import SNNLayer
from thalia.core.network import SNNNetwork
from thalia.core.dendritic import (
    DendriticBranch,
    DendriticBranchConfig,
    DendriticNeuron,
    DendriticNeuronConfig,
    create_clustered_input,
    create_scattered_input,
)


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


class TestConductanceLIF:
    """Tests for conductance-based LIF neuron model."""

    def test_initialization(self):
        """Test neuron initializes with correct dimensions."""
        neuron = ConductanceLIF(n_neurons=100)
        neuron.reset_state(batch_size=32)

        assert neuron.membrane.shape == (32, 100)
        assert neuron.g_E.shape == (32, 100)
        assert neuron.g_I.shape == (32, 100)
        # Membrane should start at leak reversal (resting potential)
        assert torch.allclose(neuron.membrane, torch.full_like(neuron.membrane, neuron.config.E_L))

    def test_reversal_potential_limits(self):
        """Test that membrane cannot exceed reversal potentials."""
        config = ConductanceLIFConfig(
            E_E=3.0,   # Excitatory reversal
            E_I=-0.5,  # Inhibitory reversal
            E_L=0.0,   # Leak reversal
            v_threshold=10.0,  # High threshold to prevent spikes
        )
        neuron = ConductanceLIF(n_neurons=10, config=config)
        neuron.reset_state(batch_size=1)

        # Apply massive excitatory input
        for _ in range(1000):
            neuron(torch.ones(1, 10) * 100.0, None)

        # Membrane should approach but not exceed E_E
        assert (neuron.membrane < config.E_E).all()
        assert (neuron.membrane > config.E_L).all()  # Should be above rest

        # Reset and apply massive inhibitory input
        neuron.reset_state(batch_size=1)
        for _ in range(1000):
            neuron(torch.zeros(1, 10), torch.ones(1, 10) * 100.0)

        # Membrane should approach but not exceed E_I (below rest)
        assert (neuron.membrane > config.E_I).all()
        assert (neuron.membrane < config.E_L).all()  # Should be below rest

    def test_spike_generation(self):
        """Test that spikes are generated when threshold is crossed."""
        config = ConductanceLIFConfig(v_threshold=1.0, E_E=3.0)
        neuron = ConductanceLIF(n_neurons=5, config=config)
        neuron.reset_state(batch_size=1)

        # Strong excitatory input should cause spikes
        spikes = None
        for _ in range(50):
            spikes, _ = neuron(torch.ones(1, 5) * 0.5)
            if spikes.sum() > 0:
                break

        assert spikes is not None and spikes.sum() > 0
        assert ((spikes == 0) | (spikes == 1)).all()

    def test_spike_reset(self):
        """Test membrane reset after spike."""
        config = ConductanceLIFConfig(v_threshold=1.0, v_reset=0.0)
        neuron = ConductanceLIF(n_neurons=10, config=config)
        neuron.reset_state(batch_size=1)

        # Force membrane above threshold
        neuron.membrane = torch.full((1, 10), 1.5)
        spikes, _ = neuron(torch.zeros(1, 10))

        assert spikes.sum() > 0
        # Membrane should reset where spikes occurred
        assert (neuron.membrane[spikes.bool()] == config.v_reset).all()

    def test_conductance_decay(self):
        """Test that conductances decay over time."""
        config = ConductanceLIFConfig(tau_E=5.0, tau_I=10.0)
        neuron = ConductanceLIF(n_neurons=10, config=config)
        neuron.reset_state(batch_size=1)

        # Apply input once
        neuron(torch.ones(1, 10) * 0.5, torch.ones(1, 10) * 0.3)
        g_E_after_input = neuron.g_E.clone()
        g_I_after_input = neuron.g_I.clone()

        # Step without input
        for _ in range(10):
            neuron(torch.zeros(1, 10), torch.zeros(1, 10))

        # Conductances should have decayed
        assert (neuron.g_E < g_E_after_input).all()
        assert (neuron.g_I < g_I_after_input).all()
        # But not to zero yet
        assert neuron.g_E.sum() > 0
        assert neuron.g_I.sum() > 0

    def test_shunting_inhibition(self):
        """Test that inhibition is divisive (shunting), not subtractive."""
        config = ConductanceLIFConfig(E_E=3.0, E_I=-0.5, v_threshold=10.0)
        neuron = ConductanceLIF(n_neurons=10, config=config)

        # Case 1: Excitation alone
        neuron.reset_state(batch_size=1)
        for _ in range(20):
            neuron(torch.ones(1, 10) * 0.5, None)
        v_exc_only = neuron.membrane.clone()

        # Case 2: Same excitation + inhibition
        neuron.reset_state(batch_size=1)
        for _ in range(20):
            neuron(torch.ones(1, 10) * 0.5, torch.ones(1, 10) * 0.5)
        v_exc_plus_inh = neuron.membrane.clone()

        # With shunting: inhibition reduces the EFFECT of excitation
        # (divides current), not just subtracts from it
        assert (v_exc_plus_inh < v_exc_only).all()
        # And the result should still be above rest (not negative)
        assert (v_exc_plus_inh > config.E_I).all()

    def test_forward_current_compatibility(self):
        """Test the convenience method for current-based input."""
        neuron = ConductanceLIF(n_neurons=10)
        neuron.reset_state(batch_size=1)

        # Positive current → excitation
        spikes, v = neuron.forward_current(torch.ones(1, 10) * 0.5)
        assert v.shape == (1, 10)

        # Negative current → inhibition
        neuron.reset_state(batch_size=1)
        neuron.membrane = torch.ones(1, 10) * 0.5  # Start above rest
        _, v_after = neuron.forward_current(torch.ones(1, 10) * -0.5)
        # Inhibition should pull toward E_I
        assert (v_after < 0.5).all()

    def test_adaptation(self):
        """Test spike-frequency adaptation.

        Adaptation should reduce firing rate as the adaptation conductance
        builds up. We test this by comparing early ISIs (before adaptation
        builds) vs later ISIs (after adaptation accumulates).
        """
        config = ConductanceLIFConfig(
            adapt_increment=1.0,   # Strong adaptation increment per spike
            tau_adapt=500.0,       # Slow decay so it accumulates over many spikes
            v_threshold=1.0,
            E_adapt=-2.0,          # Strong hyperpolarization
            tau_E=2.0,             # Fast synaptic integration
        )
        neuron = ConductanceLIF(n_neurons=1, config=config)
        neuron.reset_state(batch_size=1)

        # Strong constant input that would cause regular spiking without adaptation
        input_conductance = torch.ones(1, 1) * 0.5

        # Collect spike times
        spike_times: list[int] = []
        for t in range(500):
            spikes, _ = neuron(input_conductance)
            if spikes.item() > 0:
                spike_times.append(t)

        # With adaptation, inter-spike intervals should INCREASE over time
        # (firing rate decreases as adaptation builds up)
        assert len(spike_times) >= 5, "Need multiple spikes to test adaptation"

        # Compare early vs late ISIs
        isis = [spike_times[i+1] - spike_times[i] for i in range(len(spike_times)-1)]
        early_isi = isis[0]
        late_isi = isis[-1]

        assert late_isi > early_isi, \
            f"ISI should increase with adaptation: early={early_isi}, late={late_isi}"

    def test_refractory_period(self):
        """Test absolute refractory period."""
        config = ConductanceLIFConfig(tau_ref=2.0, dt=0.1)  # 20 timesteps refractory
        neuron = ConductanceLIF(n_neurons=5, config=config)
        neuron.reset_state(batch_size=1)

        # Force a spike
        neuron.membrane = torch.full((1, 5), 1.5)
        spikes1, _ = neuron(torch.zeros(1, 5))
        assert spikes1.sum() == 5

        # Immediately try again with strong input - should be refractory
        for _ in range(5):  # Still in refractory (need 20 steps)
            spikes, _ = neuron(torch.ones(1, 5) * 0.5)
            assert spikes.sum() == 0  # Can't spike during refractory


class TestDendriticBranch:
    """Tests for DendriticBranch with NMDA nonlinearity."""

    def test_initialization(self):
        """Test branch initializes correctly."""
        branch = DendriticBranch(n_inputs=50)
        branch.reset_state(batch_size=4)

        assert branch.weights.shape == (50,)
        assert branch.plateau.shape == (4,)
        assert (branch.plateau == 0).all()

    def test_subthreshold_linear(self):
        """Test that subthreshold inputs are integrated (near-)linearly."""
        config = DendriticBranchConfig(
            nmda_threshold=0.5,
            subthreshold_attenuation=1.0,  # No attenuation for this test
        )
        branch = DendriticBranch(n_inputs=10, config=config)
        branch.reset_state(batch_size=1)

        # Set uniform weights for predictable sum
        branch.weights.data = torch.ones(10) * 0.1

        # Weak input - should be below threshold
        weak_input = torch.ones(1, 10) * 0.1
        output = branch(weak_input)

        # Expected: 10 * 0.1 * 0.1 = 0.1 (below threshold of 0.5)
        expected_linear = 0.1
        assert output.item() < config.nmda_threshold  # Below threshold
        # Should be close to linear (within attenuation)
        assert abs(output.item() - expected_linear) < 0.05

    def test_suprathreshold_amplification(self):
        """Test that suprathreshold inputs get NMDA amplification."""
        config = DendriticBranchConfig(
            nmda_threshold=0.3,
            nmda_gain=3.0,
            subthreshold_attenuation=0.8,
        )
        branch = DendriticBranch(n_inputs=10, config=config)
        branch.reset_state(batch_size=1)

        branch.weights.data = torch.ones(10) * 0.2

        # Strong input - should be above threshold
        strong_input = torch.ones(1, 10) * 0.5
        output = branch(strong_input)

        # Linear sum would be 10 * 0.2 * 0.5 = 1.0
        linear_sum = 1.0

        # With NMDA amplification, should be > linear
        assert output.item() > linear_sum
        # But less than gain * linear (due to saturation)
        assert output.item() < config.nmda_gain * linear_sum

    def test_plateau_persistence(self):
        """Test that NMDA plateaus persist across timesteps."""
        config = DendriticBranchConfig(
            nmda_threshold=0.3,
            plateau_tau_ms=50.0,  # 50ms plateau
            dt=1.0,
        )
        branch = DendriticBranch(n_inputs=10, config=config)
        branch.reset_state(batch_size=1)

        branch.weights.data = torch.ones(10) * 0.2

        # Strong input to trigger plateau
        strong_input = torch.ones(1, 10) * 0.5
        output1 = branch(strong_input)

        # Now give zero input - plateau should sustain output
        zero_input = torch.zeros(1, 10)
        output2 = branch(zero_input)

        # Output should still be significant due to plateau
        assert output2.item() > 0.1
        # But less than the triggered output (decaying)
        assert output2.item() < output1.item()

    def test_saturation(self):
        """Test that output saturates at saturation_level."""
        config = DendriticBranchConfig(
            saturation_level=2.0,
            nmda_gain=10.0,  # Very high gain
        )
        branch = DendriticBranch(n_inputs=10, config=config)
        branch.reset_state(batch_size=1)

        branch.weights.data = torch.ones(10) * 1.0  # Large weights

        # Massive input
        huge_input = torch.ones(1, 10) * 10.0
        output = branch(huge_input)

        # Should be capped near saturation level
        assert output.item() <= config.saturation_level * 1.1  # Allow small margin


class TestDendriticNeuron:
    """Tests for DendriticNeuron with multiple branches."""

    def test_initialization(self):
        """Test neuron initializes with correct dimensions."""
        config = DendriticNeuronConfig(
            n_branches=4,
            inputs_per_branch=25,
        )
        neuron = DendriticNeuron(n_neurons=10, config=config)
        neuron.reset_state(batch_size=2)

        assert neuron.branch_weights.shape == (10, 4, 25)
        assert neuron.branch_plateaus.shape == (2, 10, 4)
        assert neuron.soma.membrane.shape == (2, 10)

    def test_clustered_vs_scattered(self):
        """Test that clustered inputs produce stronger responses than scattered.

        This is THE key test for dendritic nonlinearity:
        Same total input, but clustered on one branch should trigger
        NMDA spike and produce stronger per-branch response.

        We compare the MAXIMUM branch output (not sum) because:
        - Clustered: One branch gets strong input → NMDA spike → high output
        - Scattered: All branches get weak input → no NMDA spike → low output each
        """
        config = DendriticNeuronConfig(
            n_branches=4,
            inputs_per_branch=25,
            branch_config=DendriticBranchConfig(
                nmda_threshold=0.5,   # Higher threshold so scattered stays below
                nmda_gain=3.0,
                subthreshold_attenuation=0.5,  # Strong attenuation below threshold
            ),
            soma_config=ConductanceLIFConfig(v_threshold=10.0),
        )
        neuron = DendriticNeuron(n_neurons=1, config=config)

        # Set uniform weights for controlled test
        neuron.branch_weights.data = torch.ones_like(neuron.branch_weights) * 0.15

        # Create clustered input: 20 active inputs all on branch 0
        # Branch 0 gets: 20 * 0.15 * 1.0 = 3.0 (well above threshold 0.5)
        clustered = create_clustered_input(
            n_inputs=100,
            n_active=20,
            cluster_branch=0,
            n_branches=4,
        ).unsqueeze(0)

        # Create scattered input: 5 active inputs on each of 4 branches
        # Each branch gets: 5 * 0.15 * 1.0 = 0.75 (slightly above threshold)
        # But we'll use fewer to stay below threshold
        scattered = torch.zeros(1, 100)
        for b in range(4):
            start = b * 25
            scattered[0, start:start + 3] = 1.0  # Only 3 per branch
        # Each branch gets: 3 * 0.15 = 0.45 (below threshold 0.5)

        # Process clustered input
        neuron.reset_state(batch_size=1)
        _, _, branch_out_c = neuron.forward_with_branch_info(clustered)
        max_branch_clustered = branch_out_c.max().item()

        # Process scattered input
        neuron.reset_state(batch_size=1)
        _, _, branch_out_s = neuron.forward_with_branch_info(scattered)
        max_branch_scattered = branch_out_s.max().item()

        # Clustered should produce higher MAX branch output due to NMDA spike
        assert max_branch_clustered > max_branch_scattered, \
            f"Clustered max ({max_branch_clustered:.3f}) should exceed scattered max ({max_branch_scattered:.3f})"

    def test_branch_independence(self):
        """Test that branches operate independently."""
        config = DendriticNeuronConfig(
            n_branches=4,
            inputs_per_branch=25,
            input_routing="fixed",
        )
        neuron = DendriticNeuron(n_neurons=1, config=config)
        neuron.reset_state(batch_size=1)

        # Activate only branch 0
        inputs = torch.zeros(1, 100)
        inputs[0, 0:10] = 1.0  # Only first 10 inputs (branch 0)

        _, _, branch_outputs = neuron.forward_with_branch_info(inputs)

        # Branch 0 should be active, others should be near zero
        assert branch_outputs[0, 0, 0] > branch_outputs[0, 0, 1]
        assert branch_outputs[0, 0, 0] > branch_outputs[0, 0, 2]
        assert branch_outputs[0, 0, 0] > branch_outputs[0, 0, 3]

    def test_forward_produces_spikes(self):
        """Test that the neuron can produce spikes."""
        config = DendriticNeuronConfig(
            n_branches=4,
            inputs_per_branch=25,
            soma_config=ConductanceLIFConfig(v_threshold=1.0),
        )
        neuron = DendriticNeuron(n_neurons=5, config=config)
        neuron.reset_state(batch_size=1)

        # Strong input to all branches
        strong_input = torch.ones(1, 100) * 0.5

        total_spikes = 0
        for _ in range(100):
            spikes, _ = neuron(strong_input)
            total_spikes += spikes.sum().item()

        assert total_spikes > 0, "Neuron should produce spikes with strong input"

    def test_inhibition_reduces_firing(self):
        """Test that inhibitory input reduces firing."""
        config = DendriticNeuronConfig(
            n_branches=4,
            inputs_per_branch=25,
            branch_config=DendriticBranchConfig(
                nmda_threshold=0.15,
                nmda_gain=2.0,
            ),
            soma_config=ConductanceLIFConfig(
                v_threshold=1.0,
                E_I=-0.5,  # Inhibitory reversal below rest
            ),
        )
        neuron = DendriticNeuron(n_neurons=5, config=config)

        # Set weights for predictable behavior
        neuron.branch_weights.data = torch.ones_like(neuron.branch_weights) * 0.05

        strong_input = torch.ones(1, 100) * 0.5
        inhibition = torch.ones(1, 5) * 2.0  # Strong inhibition

        # Without inhibition
        neuron.reset_state(batch_size=1)
        spikes_no_inh = 0
        for _ in range(200):
            spikes, _ = neuron(strong_input, g_inh=None)
            spikes_no_inh += spikes.sum().item()

        # With inhibition
        neuron.reset_state(batch_size=1)
        spikes_with_inh = 0
        for _ in range(200):
            spikes, _ = neuron(strong_input, g_inh=inhibition)
            spikes_with_inh += spikes.sum().item()

        # Inhibition should reduce spike count (or at least not increase)
        assert spikes_with_inh <= spikes_no_inh, \
            f"Inhibition should not increase spikes: with={spikes_with_inh}, without={spikes_no_inh}"

    def test_repr(self):
        """Test string representation."""
        config = DendriticNeuronConfig(n_branches=4, inputs_per_branch=25)
        neuron = DendriticNeuron(n_neurons=10, config=config)

        repr_str = repr(neuron)
        assert "DendriticNeuron" in repr_str
        assert "n=10" in repr_str
        assert "branches=4" in repr_str
