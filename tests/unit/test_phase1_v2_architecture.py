"""
Unit tests for Phase 1 v2.0 architecture components.

Tests AxonalProjection (pure routing) and AfferentSynapses (weights + learning).
"""

import pytest
import torch

from thalia.pathways.axonal_projection import AxonalProjection, SourceSpec
from thalia.synapses import AfferentSynapses, AfferentSynapsesConfig


class TestAxonalProjection:
    """Test AxonalProjection spike routing."""

    def test_single_source_routing(self):
        """Test basic single-source routing."""
        projection = AxonalProjection(
            sources=[("cortex", "l5", 128, 2.0)],
            device="cpu",
            dt_ms=1.0,
        )

        assert projection.n_output == 128
        assert len(projection.sources) == 1
        assert projection.sources[0].region_name == "cortex"
        assert projection.sources[0].port == "l5"

    def test_multi_source_concatenation(self):
        """Test multi-source concatenation."""
        projection = AxonalProjection(
            sources=[
                ("cortex", "l5", 128, 2.0),
                ("hippocampus", None, 64, 3.0),
                ("pfc", None, 32, 2.0),
            ],
            device="cpu",
            dt_ms=1.0,
        )

        # Total output: 128 + 64 + 32 = 224
        assert projection.n_output == 224
        assert len(projection.sources) == 3

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

        # Forward without delays for testing
        # AxonalProjection now returns dict preserving source identity
        delayed_outputs = projection.forward(source_outputs, apply_delays=False)

        # Check dict structure
        assert isinstance(delayed_outputs, dict)
        assert "cortex" in delayed_outputs
        assert "hippocampus" in delayed_outputs

        # Check individual outputs
        assert delayed_outputs["cortex"].shape == (10,)
        assert delayed_outputs["hippocampus"].shape == (5,)
        assert delayed_outputs["cortex"].all()  # All ones
        assert not delayed_outputs["hippocampus"].any()  # All zeros

        # Test concatenation (if target region needs it)
        concatenated = torch.cat([delayed_outputs["cortex"], delayed_outputs["hippocampus"]])
        assert concatenated.shape == (15,)
        assert concatenated[:10].all()  # First 10 are ones
        assert not concatenated[10:].any()  # Last 5 are zeros

    def test_axonal_delays(self):
        """Test that axonal delays work."""
        projection = AxonalProjection(
            sources=[("cortex", None, 5, 2.0)],
            device="cpu",
            dt_ms=1.0,
        )

        # First timestep: input spikes
        spikes_t0 = torch.ones(5, dtype=torch.bool)
        output_t0 = projection.forward({"cortex": spikes_t0}, apply_delays=True)

        # Delays cause zeros initially (dict output)
        assert isinstance(output_t0, dict)
        assert not output_t0["cortex"].any()

        # Second timestep: no input
        output_t1 = projection.forward({"cortex": torch.zeros(5, dtype=torch.bool)}, apply_delays=True)

        # Still zeros (delay = 2 steps)
        assert not output_t1["cortex"].any()

        # Third timestep: delayed spikes appear
        output_t2 = projection.forward({"cortex": torch.zeros(5, dtype=torch.bool)}, apply_delays=True)
        assert output_t2["cortex"].all()  # Original spikes now appear

    def test_grow_source(self):
        """Test growing a source."""
        projection = AxonalProjection(
            sources=[("cortex", None, 128, 2.0)],
            device="cpu",
            dt_ms=1.0,
        )

        assert projection.n_output == 128

        # Grow cortex by 20 neurons
        projection.grow_source("cortex", new_size=148)

        assert projection.n_output == 148
        assert projection.sources[0].size == 148

    def test_reset(self):
        """Test that reset clears delay buffers."""
        projection = AxonalProjection(
            sources=[("cortex", None, 5, 2.0)],
            device="cpu",
            dt_ms=1.0,
        )

        # Add some spikes
        projection.forward({"cortex": torch.ones(5, dtype=torch.bool)}, apply_delays=True)

        # Reset
        projection.reset_state()

        # Check buffers are zeroed
        for buffer in projection._delay_buffers.values():
            assert not buffer.any()


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

        assert synapses.weights.shape == (70, 224)
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

        assert synaptic_current.shape == (10,)
        assert synaptic_current.dtype == torch.float32

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
        config = AfferentSynapsesConfig(
            n_neurons=70,
            n_inputs=224,
            learning_rule="hebbian",
            device="cpu",
        )

        synapses = AfferentSynapses(config)

        assert synapses.weights.shape == (70, 224)

        # Grow by 20 inputs
        synapses.grow_input(n_new=20)

        assert synapses.weights.shape == (70, 244)
        assert synapses.config.n_inputs == 244

    def test_grow_output(self):
        """Test growing output dimension."""
        config = AfferentSynapsesConfig(
            n_neurons=70,
            n_inputs=224,
            learning_rule="hebbian",
            device="cpu",
        )

        synapses = AfferentSynapses(config)

        assert synapses.weights.shape == (70, 224)

        # Grow by 20 neurons
        synapses.grow_output(n_new=20)

        assert synapses.weights.shape == (90, 224)
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
        assert state["weights"].shape == (10, 5)

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

        # Route spikes (now returns dict)
        routed_spikes = projection.forward(source_outputs, apply_delays=False)
        assert isinstance(routed_spikes, dict)

        # Concatenate for synaptic processing
        concatenated = torch.cat([
            routed_spikes["cortex"],
            routed_spikes["hippocampus"],
            routed_spikes["pfc"]
        ])
        assert concatenated.shape == (224,)

        # Integrate through synapses
        synaptic_current = synapses(concatenated)
        assert synaptic_current.shape == (70,)

    def test_growth_coordination(self):
        """Test coordinated growth of projection and synapses."""
        # Initial setup
        projection = AxonalProjection(
            sources=[("cortex", None, 128, 2.0)],
            device="cpu",
            dt_ms=1.0,
        )

        synapses_config = AfferentSynapsesConfig(
            n_neurons=70,
            n_inputs=128,
            learning_rule="hebbian",
            device="cpu",
        )
        synapses = AfferentSynapses(synapses_config)

        # Cortex grows by 20 neurons
        projection.grow_source("cortex", new_size=148)
        synapses.grow_input(n_new=20)

        # Verify sizes match
        assert projection.n_output == 148
        assert synapses.config.n_inputs == 148
        assert synapses.weights.shape == (70, 148)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
