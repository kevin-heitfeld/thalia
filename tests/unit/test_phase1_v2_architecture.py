"""
Unit tests for Phase 1 v2.0 architecture components.

Tests AxonalProjection (pure routing) and AfferentSynapses (weights + learning).
"""

import pytest
import torch

from thalia.pathways.axonal_projection import AxonalProjection
from thalia.synapses import AfferentSynapses, AfferentSynapsesConfig


class TestAxonalProjection:
    """Test AxonalProjection spike routing."""

    def test_single_source_routing(self):
        """Test basic single-source routing."""
        source_size = 128
        projection = AxonalProjection(
            sources=[("cortex", "l5", source_size, 2.0)],
            device="cpu",
            dt_ms=1.0,
        )

        assert projection.n_output == source_size
        assert len(projection.sources) == 1
        assert projection.sources[0].region_name == "cortex"
        assert projection.sources[0].port == "l5"

    def test_multi_source_concatenation(self):
        """Test multi-source concatenation."""
        sizes = [128, 64, 32]
        projection = AxonalProjection(
            sources=[
                ("cortex", "l5", sizes[0], 2.0),
                ("hippocampus", None, sizes[1], 3.0),
                ("pfc", None, sizes[2], 2.0),
            ],
            device="cpu",
            dt_ms=1.0,
        )

        # Total output: sum of all source sizes
        expected_total = sum(sizes)
        assert projection.n_output == expected_total
        assert len(projection.sources) == len(sizes)

    def test_forward_concatenation(self):
        """Test that forward() properly concatenates spikes."""
        projection = AxonalProjection(
            sources=[
                ("cortex", None, 10, 2.0),
                ("hippocampus", None, 5, 3.0),
            ],
            device="cpu",
            dt_ms=1.0,
        )

        # Create test spikes
        cortex_spikes = torch.ones(10, dtype=torch.bool)
        hipp_spikes = torch.zeros(5, dtype=torch.bool)

        source_outputs = {
            "cortex": cortex_spikes,
            "hippocampus": hipp_spikes,
        }

        # Forward with delays (always active in new architecture)
        # AxonalProjection now returns dict preserving source identity
        # With delay_ms=2.0 and dt_ms=1.0, need 3 forward calls to see spikes:
        # t=0: write spikes, read t=-2 (empty)
        # t=1: write zeros, read t=-1 (empty)
        # t=2: write zeros, read t=0 (spikes appear!)
        projection.forward(source_outputs)  # t=0
        projection.forward({
            "cortex": torch.zeros_like(cortex_spikes),
            "hippocampus": torch.zeros_like(hipp_spikes),
        })  # t=1
        delayed_outputs = projection.forward({
            "cortex": torch.zeros_like(cortex_spikes),
            "hippocampus": torch.zeros_like(hipp_spikes),
        })  # t=2 - now cortex spikes appear (2ms delay)

        # Check dict structure
        assert isinstance(delayed_outputs, dict)
        assert "cortex" in delayed_outputs
        assert "hippocampus" in delayed_outputs

        # Check individual outputs
        assert delayed_outputs["cortex"].shape == (10,), \
            f"Cortex output shape should be (10,), got {delayed_outputs['cortex'].shape}"
        assert delayed_outputs["cortex"].dtype == torch.bool, \
            f"Cortex output should be bool spikes, got {delayed_outputs['cortex'].dtype}"
        assert not torch.isnan(delayed_outputs["cortex"].float()).any(), \
            "Cortex output contains NaN values"

        assert delayed_outputs["hippocampus"].shape == (5,), \
            f"Hippocampus output shape should be (5,), got {delayed_outputs['hippocampus'].shape}"
        assert delayed_outputs["hippocampus"].dtype == torch.bool, \
            f"Hippocampus output should be bool spikes, got {delayed_outputs['hippocampus'].dtype}"
        assert not torch.isnan(delayed_outputs["hippocampus"].float()).any(), \
            "Hippocampus output contains NaN values"

        assert delayed_outputs["cortex"].all()  # Cortex ones appear after 2ms delay
        assert not delayed_outputs["hippocampus"].any()  # Hippocampus zeros (3ms delay, not reached yet)

        # Test concatenation (if target region needs it)
        concatenated = torch.cat([delayed_outputs["cortex"], delayed_outputs["hippocampus"]])
        assert concatenated.shape == (15,), \
            f"Concatenated output shape should be (15,), got {concatenated.shape}"
        assert concatenated.dtype == torch.bool, \
            f"Concatenated output should be bool spikes, got {concatenated.dtype}"
        assert not torch.isnan(concatenated.float()).any(), \
            "Concatenated output contains NaN values"
        assert concatenated[:10].all()  # First 10 are ones from cortex
        assert not concatenated[10:].any()  # Last 5 are zeros from hippocampus

    @pytest.mark.parametrize("delay_ms,expected_steps", [
        (1.0, 2),  # 1ms delay = 2 steps to see spikes
        (2.0, 3),  # 2ms delay = 3 steps
        (5.0, 6),  # 5ms delay = 6 steps
    ])
    def test_axonal_delays_various_durations(self, delay_ms, expected_steps):
        """Test axonal delays with various durations.

        Why this test exists: Validates that CircularDelayBuffer correctly
        implements axonal transmission delays across different timescales.
        Biologically, different fiber types have different conduction velocities.
        """
        projection = AxonalProjection(
            sources=[("cortex", None, 5, delay_ms)],
            device="cpu",
            dt_ms=1.0,
        )

        # First timestep: input spikes
        spikes_t0 = torch.ones(5, dtype=torch.bool)
        output_t0 = projection.forward({"cortex": spikes_t0})

        # Delays cause zeros initially
        assert not output_t0["cortex"].any()

        # Advance until expected_steps - 1
        for step in range(1, expected_steps - 1):
            output = projection.forward({"cortex": torch.zeros(5, dtype=torch.bool)})
            assert not output["cortex"].any(), f"Spikes appeared too early at step {step}"

        # At expected_steps, delayed spikes should appear
        final_output = projection.forward({"cortex": torch.zeros(5, dtype=torch.bool)})
        assert final_output["cortex"].all(), f"Spikes didn't appear at expected step {expected_steps}"

    def test_axonal_delays(self):
        """Test that axonal delays work."""
        projection = AxonalProjection(
            sources=[("cortex", None, 5, 2.0)],
            device="cpu",
            dt_ms=1.0,
        )

        # First timestep: input spikes
        spikes_t0 = torch.ones(5, dtype=torch.bool)
        output_t0 = projection.forward({"cortex": spikes_t0})

        # Delays cause zeros initially (dict output)
        assert isinstance(output_t0, dict)
        assert not output_t0["cortex"].any()

        # Second timestep: no input
        output_t1 = projection.forward({"cortex": torch.zeros(5, dtype=torch.bool)})

        # Still zeros (delay = 2 steps)
        assert not output_t1["cortex"].any()

        # Third timestep: delayed spikes appear
        output_t2 = projection.forward({"cortex": torch.zeros(5, dtype=torch.bool)})
        assert output_t2["cortex"].all()  # Original spikes now appear

    @pytest.mark.parametrize("initial_size,growth_amount", [
        (64, 16),
        (128, 20),
        (256, 50),
    ])
    def test_grow_source_various_sizes(self, initial_size, growth_amount):
        """Test growing sources with various initial sizes and growth amounts.

        Why this test exists: Validates that axonal projections correctly handle
        growth at different scales, which is critical for curriculum learning
        where brain regions expand during training.
        """
        projection = AxonalProjection(
            sources=[("cortex", None, initial_size, 2.0)],
            device="cpu",
            dt_ms=1.0,
        )

        assert projection.n_output == initial_size

        # Grow cortex
        new_size = initial_size + growth_amount
        projection.grow_source("cortex", new_size=new_size)

        assert projection.n_output == new_size
        assert projection.sources[0].size == new_size

        # Validate that forward pass works after growth
        test_spikes = torch.zeros(new_size, dtype=torch.bool)
        test_spikes[:10] = True
        output = projection.forward({"cortex": test_spikes})
        assert output["cortex"].shape == (new_size,), \
            f"Output shape should be ({new_size},) after growth, got {output['cortex'].shape}"
        assert output["cortex"].dtype == torch.bool, \
            f"Output should be bool spikes, got {output['cortex'].dtype}"
        assert not torch.isnan(output["cortex"].float()).any(), \
            "Output contains NaN values after growth"

    def test_grow_output(self):
        """Test that reset clears delay buffers."""
        projection = AxonalProjection(
            sources=[("cortex", None, 5, 2.0)],
            device="cpu",
            dt_ms=1.0,
        )

        # Add some spikes
        projection.forward({"cortex": torch.ones(5, dtype=torch.bool)})

        # Reset
        projection.reset_state()

        # Check buffers are zeroed
        for buffer in projection._delay_buffers.values():
            assert not buffer.buffer.any()  # Check the internal tensor


class TestAfferentSynapses:
    """Test AfferentSynapses synaptic integration."""

    def test_initialization(self):
        """Test synaptic layer initialization."""
        config = AfferentSynapsesConfig(
            n_neurons=70,
            n_inputs=224,
            learning_rule="three_factor",
            learning_rate=0.001,
            device="cpu",
        )

        synapses = AfferentSynapses(config)

        assert synapses.weights.shape == (70, 224), \
            f"Weight shape should be (70, 224), got {synapses.weights.shape}"
        assert not torch.isnan(synapses.weights).any(), \
            "Weights contain NaN values"
        assert not torch.isinf(synapses.weights).any(), \
            "Weights contain Inf values"
        assert synapses.learning_strategy is not None

    def test_forward_integration(self):
        """Test synaptic integration."""
        config = AfferentSynapsesConfig(
            n_neurons=10,
            n_inputs=5,
            learning_rule="hebbian",
            learning_rate=0.001,
            device="cpu",
        )

        synapses = AfferentSynapses(config)

        # Input spikes
        input_spikes = torch.ones(5, dtype=torch.bool)

        # Forward pass
        synaptic_current = synapses(input_spikes)

        assert synaptic_current.shape == (10,), \
            f"Synaptic current shape should be (10,), got {synaptic_current.shape}"
        assert synaptic_current.dtype == torch.float32, \
            f"Synaptic current should be float32, got {synaptic_current.dtype}"
        assert not torch.isnan(synaptic_current).any(), \
            "Synaptic current contains NaN values"
        assert not torch.isinf(synaptic_current).any(), \
            "Synaptic current contains Inf values"

    def test_learning(self):
        """Test that learning updates weights."""
        config = AfferentSynapsesConfig(
            n_neurons=10,
            n_inputs=5,
            learning_rule="hebbian",
            learning_rate=0.1,
            device="cpu",
        )

        synapses = AfferentSynapses(config)

        # Store initial weights
        initial_weights = synapses.weights.data.clone()

        # Apply learning
        pre_spikes = torch.ones(5, dtype=torch.bool)
        post_spikes = torch.ones(10, dtype=torch.bool)

        metrics = synapses.apply_learning(pre_spikes, post_spikes)

        # Weights should have changed
        assert not torch.allclose(synapses.weights.data, initial_weights)

        # Check metrics
        assert "mean_change" in metrics
        assert metrics["mean_change"] > 0  # Should have LTP

    def test_grow_input(self):
        """Test growing input dimension."""
        n_neurons = 70
        initial_inputs = 224
        growth_amount = 20
        config = AfferentSynapsesConfig(
            n_neurons=n_neurons,
            n_inputs=initial_inputs,
            learning_rule="hebbian",
            device="cpu",
        )

        synapses = AfferentSynapses(config)

        assert synapses.weights.shape == (n_neurons, initial_inputs), \
            f"Initial weight shape should be ({n_neurons}, {initial_inputs}), got {synapses.weights.shape}"

        # Grow inputs
        synapses.grow_input(n_new=growth_amount)

        new_inputs = initial_inputs + growth_amount
        assert synapses.weights.shape == (n_neurons, new_inputs), \
            f"Weight shape after growth should be ({n_neurons}, {new_inputs}), got {synapses.weights.shape}"
        assert not torch.isnan(synapses.weights).any(), \
            "Weights contain NaN after input growth"
        assert not torch.isinf(synapses.weights).any(), \
            "Weights contain Inf after input growth"
        assert synapses.config.n_inputs == new_inputs

    def test_grow_output(self):
        """Test growing output dimension."""
        config = AfferentSynapsesConfig(
            n_neurons=70,
            n_inputs=224,
            learning_rule="hebbian",
            device="cpu",
        )

        synapses = AfferentSynapses(config)

        assert synapses.weights.shape == (70, 224), \
            f"Initial weight shape should be (70, 224), got {synapses.weights.shape}"

        # Grow by 20 neurons
        synapses.grow_output(20)

        assert synapses.weights.shape == (90, 224), \
            f"Weight shape after growth should be (90, 224), got {synapses.weights.shape}"
        assert not torch.isnan(synapses.weights).any(), \
            "Weights contain NaN after output growth"
        assert not torch.isinf(synapses.weights).any(), \
            "Weights contain Inf after output growth"
        assert synapses.config.n_neurons == 90

    def test_checkpoint_state(self):
        """Test state save/load."""
        config = AfferentSynapsesConfig(
            n_neurons=10,
            n_inputs=5,
            learning_rule="hebbian",
            device="cpu",
        )

        synapses = AfferentSynapses(config)

        # Get state
        state = synapses.get_state()

        assert "weights" in state
        assert "config" in state
        assert state["weights"].shape == (10, 5), \
            f"Saved weights shape should be (10, 5), got {state['weights'].shape}"
        assert not torch.isnan(state["weights"]).any(), \
            "Saved weights contain NaN values"
        assert not torch.isinf(state["weights"]).any(), \
            "Saved weights contain Inf values"

        # Save original weights
        original_weights = synapses.weights.data.clone()

        # Modify weights
        synapses.weights.data.fill_(0.5)
        assert torch.allclose(synapses.weights.data, torch.tensor(0.5))

        # Restore state
        synapses.load_state(state)

        # Weights should be restored to original
        assert torch.allclose(synapses.weights.data, original_weights)


class TestIntegration:
    """Test AxonalProjection + AfferentSynapses integration."""

    def test_routing_to_synapses(self):
        """Test complete pathway: routing → synaptic integration."""
        # Create projection (3 sources)
        projection = AxonalProjection(
            sources=[
                ("cortex", None, 128, 2.0),
                ("hippocampus", None, 64, 3.0),
                ("pfc", None, 32, 2.0),
            ],
            device="cpu",
            dt_ms=1.0,
        )

        # Create afferent synapses for striatum
        synapses_config = AfferentSynapsesConfig(
            n_neurons=70,
            n_inputs=224,  # Must match projection output
            learning_rule="three_factor",
            device="cpu",
        )
        synapses = AfferentSynapses(synapses_config)

        # Create source spikes
        source_outputs = {
            "cortex": torch.ones(128, dtype=torch.bool),
            "hippocampus": torch.zeros(64, dtype=torch.bool),
            "pfc": torch.ones(32, dtype=torch.bool),
        }

        # Route spikes (now returns dict, delays always active)
        routed_spikes = projection.forward(source_outputs)
        assert isinstance(routed_spikes, dict)

        # Concatenate for synaptic processing
        concatenated = torch.cat([
            routed_spikes["cortex"],
            routed_spikes["hippocampus"],
            routed_spikes["pfc"]
        ])
        assert concatenated.shape == (224,), \
            f"Concatenated shape should be (224,), got {concatenated.shape}"
        assert concatenated.dtype == torch.bool, \
            f"Concatenated output should be bool spikes, got {concatenated.dtype}"
        assert not torch.isnan(concatenated.float()).any(), \
            "Concatenated output contains NaN values"

        # Integrate through synapses
        synaptic_current = synapses(concatenated)
        assert synaptic_current.shape == (70,), \
            f"Synaptic current shape should be (70,), got {synaptic_current.shape}"
        assert synaptic_current.dtype == torch.float32, \
            f"Synaptic current should be float32, got {synaptic_current.dtype}"
        assert not torch.isnan(synaptic_current).any(), \
            "Synaptic current contains NaN values"
        assert not torch.isinf(synaptic_current).any(), \
            "Synaptic current contains Inf values"

    def test_growth_coordination(self):
        """Test coordinated growth of projection and synapses."""
        # Initial setup
        initial_size = 128
        growth_amount = 20
        projection = AxonalProjection(
            sources=[("cortex", None, initial_size, 2.0)],
            device="cpu",
            dt_ms=1.0,
        )

        synapses_config = AfferentSynapsesConfig(
            n_neurons=70,
            n_inputs=initial_size,
            learning_rule="hebbian",
            device="cpu",
        )
        synapses = AfferentSynapses(synapses_config)

        # Cortex grows
        new_size = initial_size + growth_amount
        projection.grow_source("cortex", new_size=new_size)
        synapses.grow_input(n_new=growth_amount)

        # Verify sizes match
        assert projection.n_output == new_size
        assert synapses.config.n_inputs == new_size
        assert synapses.weights.shape == (70, new_size), \
            f"Weight shape after coordinated growth should be (70, {new_size}), got {synapses.weights.shape}"
        assert not torch.isnan(synapses.weights).any(), \
            "Weights contain NaN after coordinated growth"
        assert not torch.isinf(synapses.weights).any(), \
            "Weights contain Inf after coordinated growth"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
