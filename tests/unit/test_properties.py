"""
Property-based tests using Hypothesis.

These tests verify that components maintain invariants across
a wide range of randomly generated inputs.
"""

import pytest
import torch
from hypothesis import given, strategies as st, settings, assume

from thalia.core.neuron import LIFNeuron, LIFConfig
from thalia.core.dendritic import DendriticNeuron, DendriticNeuronConfig
from tests.test_utils import assert_spike_train_valid, assert_membrane_potential_valid


# Custom strategies for generating test data
@st.composite
def valid_neuron_counts(draw):
    """Generate valid neuron counts (1-1000)."""
    return draw(st.integers(min_value=1, max_value=1000))


@st.composite
def valid_batch_sizes(draw):
    """Generate valid batch sizes (1-128)."""
    return draw(st.integers(min_value=1, max_value=128))


@st.composite
def valid_time_constants(draw):
    """Generate valid time constants (0.1-1000)."""
    return draw(st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False))


@st.composite
def valid_input_tensor(draw, batch_size, n_features):
    """Generate valid input tensor with reasonable values."""
    # Generate values in a reasonable range [-5, 5]
    values = draw(st.lists(
        st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=batch_size * n_features,
        max_size=batch_size * n_features
    ))
    tensor = torch.tensor(values).reshape(batch_size, n_features)
    return tensor


@pytest.mark.unit
class TestLIFNeuronProperties:
    """Property-based tests for LIF neuron."""

    @given(
        n_neurons=valid_neuron_counts(),
        batch_size=valid_batch_sizes(),
    )
    @settings(max_examples=50, deadline=1000)  # Run 50 random examples
    def test_shape_consistency(self, n_neurons, batch_size):
        """Test that LIF neuron maintains shape consistency for any valid inputs."""
        neuron = LIFNeuron(n_neurons=n_neurons)
        neuron.reset_state()

        # Membrane shape should match
        assert neuron.membrane.shape == (batch_size, n_neurons)

        # Forward pass should maintain batch dimension
        input_current = torch.randn(batch_size, n_neurons)
        spikes, traces = neuron(input_current)

        assert spikes.shape == (batch_size, n_neurons)
        assert_spike_train_valid(spikes)

    @given(
        n_neurons=valid_neuron_counts(),
        tau_mem=valid_time_constants(),
    )
    @settings(max_examples=30, deadline=1000)
    def test_membrane_stays_bounded(self, n_neurons, tau_mem):
        """Test that membrane potential stays in reasonable range."""
        config = LIFConfig(tau_mem=tau_mem, v_threshold=1.0, v_rest=0.0)
        neuron = LIFNeuron(n_neurons=n_neurons, config=config)
        neuron.reset_state()

        # Run for multiple timesteps with random input
        for _ in range(20):
            input_current = torch.randn(4, n_neurons) * 0.5
            spikes, _ = neuron(input_current)

        # Membrane should stay bounded (not explode to infinity)
        assert_membrane_potential_valid(
            neuron.membrane,
            v_rest=config.v_rest,
            v_threshold=config.v_threshold,
            tolerance=3.0
        )

    @given(
        n_neurons=valid_neuron_counts(),
    )
    @settings(max_examples=30, deadline=1000)
    def test_spikes_are_always_binary(self, n_neurons):
        """Test that spikes are always 0 or 1, never intermediate values."""
        neuron = LIFNeuron(n_neurons=n_neurons)
        neuron.reset_state()

        # Try various input magnitudes
        for magnitude in [0.1, 0.5, 1.0, 2.0, 5.0]:
            input_current = torch.randn(8, n_neurons) * magnitude
            spikes, _ = neuron(input_current)

            # Every spike value must be exactly 0 or 1
            unique_values = torch.unique(spikes)
            assert all(v in [0.0, 1.0] for v in unique_values.tolist()), \
                f"Found non-binary spike values: {unique_values.tolist()}"

    @given(
        n_neurons=st.integers(min_value=1, max_value=500),
    )
    @settings(max_examples=30, deadline=1000)
    def test_reset_clears_state(self, n_neurons):
        """Test that reset_state properly clears neuron state."""
        batch_size = 1  # THALIA enforces single-instance architecture
        neuron = LIFNeuron(n_neurons=n_neurons)

        # Run with some input to build up state
        neuron.reset_state()
        for _ in range(10):
            neuron(torch.randn(batch_size, n_neurons))

        # Reset and check membrane is at rest
        neuron.reset_state()
        assert torch.allclose(
            neuron.membrane,
            torch.full_like(neuron.membrane, neuron.config.v_rest),
            atol=1e-6
        )

    @given(
        n_neurons=valid_neuron_counts(),
    )
    @settings(max_examples=20, deadline=1000)
    def test_zero_input_produces_valid_output(self, n_neurons):
        """Test that zero input always produces valid (non-NaN) output."""
        neuron = LIFNeuron(n_neurons=n_neurons)
        neuron.reset_state()

        # Zero input should still produce valid spikes (just zeros)
        spikes, _ = neuron(torch.zeros(4, n_neurons))

        assert_spike_train_valid(spikes)
        assert not torch.isnan(neuron.membrane).any()


@pytest.mark.unit
class TestDendriticNeuronProperties:
    """Property-based tests for dendritic neurons."""

    @given(
        n_neurons=st.integers(min_value=1, max_value=100),
        n_branches=st.integers(min_value=1, max_value=10),
        inputs_per_branch=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=30, deadline=2000)
    def test_dendritic_shape_consistency(self, n_neurons, n_branches, inputs_per_branch):
        """Test dendritic neuron maintains correct shapes."""
        batch_size = 1  # THALIA enforces single-instance architecture
        config = DendriticNeuronConfig(
            n_branches=n_branches,
            inputs_per_branch=inputs_per_branch,
        )
        neuron = DendriticNeuron(n_neurons=n_neurons, config=config)
        neuron.reset_state()

        total_inputs = n_branches * inputs_per_branch
        input_spikes = torch.randn(batch_size, total_inputs)

        output = neuron(input_spikes)

        # DendriticNeuron returns (spikes, membrane) tuple
        if isinstance(output, tuple):
            spikes, membrane = output
            assert spikes.shape == (batch_size, n_neurons), \
                f"Expected spike shape ({batch_size}, {n_neurons}), got {spikes.shape}"
            assert membrane.shape == (batch_size, n_neurons), \
                f"Expected membrane shape ({batch_size}, {n_neurons}), got {membrane.shape}"
        else:
            assert output.shape == (batch_size, n_neurons), \
                f"Expected shape ({batch_size}, {n_neurons}), got {output.shape}"


@pytest.mark.unit
class TestNumericalStabilityProperties:
    """Property-based tests for numerical stability."""

    @given(
        n_neurons=st.integers(min_value=10, max_value=100),
        n_timesteps=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=20, deadline=2000)
    def test_long_simulations_stay_stable(self, n_neurons, n_timesteps):
        """Test that long simulations don't produce NaN/Inf."""
        neuron = LIFNeuron(n_neurons=n_neurons)
        neuron.reset_state()

        # Run for many timesteps
        for t in range(n_timesteps):
            input_current = torch.randn(4, n_neurons) * 0.5
            spikes, _ = neuron(input_current)

            # Check that state remains valid
            assert not torch.isnan(neuron.membrane).any(), \
                f"NaN detected in membrane at timestep {t}"
            assert not torch.isinf(neuron.membrane).any(), \
                f"Inf detected in membrane at timestep {t}"
            assert_spike_train_valid(spikes)

    @given(
        n_neurons=st.integers(min_value=10, max_value=100),
        input_scale=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=1000)
    def test_scales_gracefully_with_input_magnitude(self, n_neurons, input_scale):
        """Test that neurons handle different input magnitudes gracefully."""
        neuron = LIFNeuron(n_neurons=n_neurons)
        neuron.reset_state()

        # Test with scaled input
        input_current = torch.randn(4, n_neurons) * input_scale
        spikes, _ = neuron(input_current)

        # Should produce valid output regardless of scale
        assert_spike_train_valid(spikes)
        assert not torch.isnan(neuron.membrane).any()
        assert not torch.isinf(neuron.membrane).any()


@pytest.mark.unit
class TestInvariantProperties:
    """Test invariants that should always hold."""

    @given(
        n_neurons=valid_neuron_counts(),
    )
    @settings(max_examples=30, deadline=1000)
    def test_spike_count_never_exceeds_neuron_count(self, n_neurons):
        """Test that number of spikes never exceeds number of neurons."""
        neuron = LIFNeuron(n_neurons=n_neurons)
        neuron.reset_state()

        # Even with very strong input
        strong_input = torch.ones(16, n_neurons) * 100.0
        spikes, _ = neuron(strong_input)

        # Each batch item can have at most n_neurons spikes
        spikes_per_batch = spikes.sum(dim=1)
        assert (spikes_per_batch <= n_neurons).all(), \
            f"Found {spikes_per_batch.max().item()} spikes but only {n_neurons} neurons"

    @given(
        n_neurons=valid_neuron_counts(),
    )
    @settings(max_examples=20, deadline=1000)
    def test_membrane_decays_toward_rest(self, n_neurons):
        """Test that membrane decays toward rest without input."""
        neuron = LIFNeuron(n_neurons=n_neurons, config=LIFConfig(v_rest=0.0))
        neuron.reset_state()

        # Set membrane above rest
        neuron.membrane = torch.full((4, n_neurons), 0.5)

        # Step with zero input multiple times
        for _ in range(10):
            neuron(torch.zeros(4, n_neurons))

        # Should have decayed toward rest (0.0)
        assert (neuron.membrane < 0.5).any(), "Membrane should decay toward rest"
        assert (neuron.membrane >= 0.0).all(), "Membrane should not go below rest (much)"

    @given(
        n_neurons=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=20, deadline=1000)
    def test_identical_inputs_produce_identical_outputs(self, n_neurons):
        """Test determinism: same input should give same output."""
        # Create two identical neurons
        config = LIFConfig(v_threshold=1.0, v_rest=0.0, tau_mem=20.0)
        neuron1 = LIFNeuron(n_neurons=n_neurons, config=config)
        neuron2 = LIFNeuron(n_neurons=n_neurons, config=config)

        # Reset to same state
        neuron1.reset_state()
        neuron2.reset_state()

        # Apply same input
        torch.manual_seed(42)  # Ensure same random input
        input_current = torch.randn(4, n_neurons)

        spikes1, _ = neuron1(input_current.clone())
        spikes2, _ = neuron2(input_current.clone())

        # Should produce identical results
        assert torch.allclose(spikes1, spikes2), \
            "Identical inputs should produce identical outputs"
        assert torch.allclose(neuron1.membrane, neuron2.membrane), \
            "Membrane states should be identical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
