"""
Tests for neurogenesis history tracking in Prefrontal and Hippocampus regions.

Tests that neuron birth timestamps are properly tracked and recorded in checkpoints.
"""

import pytest

from thalia.regions.hippocampus.config import HippocampusConfig
from thalia.regions.hippocampus.trisynaptic import TrisynapticHippocampus
from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig


def create_test_prefrontal(
    input_size: int = 64, n_neurons: int = 50, device: str = "cpu"
) -> Prefrontal:
    """Create Prefrontal for testing with new (config, sizes, device) pattern."""
    sizes = {"input_size": input_size, "n_neurons": n_neurons}
    config = PrefrontalConfig(device=device)
    return Prefrontal(config, sizes, device)


def create_test_hippocampus(
    input_size: int = 100, total_neurons: int = 200, device: str = "cpu"
) -> TrisynapticHippocampus:
    """Create Hippocampus for testing with new (config, sizes, device) pattern."""
    sizes = {
        "input_size": input_size,
        "dg_size": int(total_neurons * 0.4),
        "ca3_size": int(total_neurons * 0.3),
        "ca2_size": int(total_neurons * 0.1),
        "ca1_size": int(total_neurons * 0.2),
    }
    config = HippocampusConfig(device=device)
    return TrisynapticHippocampus(config, sizes, device)


class TestPrefrontalNeurogenesisTracking:
    """Test neurogenesis history tracking for Prefrontal cortex."""

    def test_initial_birth_steps_all_zero(self):
        """All initial neurons should have birth_step=0."""
        pfc = create_test_prefrontal(input_size=64, n_neurons=50, device="cpu")

        # Use public checkpoint API to validate birth tracking
        checkpoint = pfc.checkpoint_manager.get_neuromorphic_state()
        assert "neurons" in checkpoint
        neurons = checkpoint["neurons"]

        # Validate shape and birth steps via checkpoint
        assert len(neurons) == 50
        for neuron in neurons:
            assert "created_step" in neuron
            assert neuron["created_step"] == 0

    def test_training_step_update(self):
        """Training step should be updateable."""
        pfc = create_test_prefrontal(input_size=64, n_neurons=50, device="cpu")

        pfc.set_training_step(1000)
        # Verify training step persisted by growing and checking birth step
        pfc.grow_neurons(n_new=5)

        checkpoint = pfc.checkpoint_manager.get_neuromorphic_state()
        neurons = checkpoint["neurons"]
        # New neurons (last 5) should have been born at step 1000
        for neuron in neurons[-5:]:
            assert neuron["created_step"] == 1000

    def test_grow_output_tracks_birth_steps(self):
        """New neurons should record their birth timestep."""
        pfc = create_test_prefrontal(input_size=64, n_neurons=50, device="cpu")

        # Set training step
        pfc.set_training_step(5000)

        # Grow by 20 neurons
        pfc.grow_neurons(n_new=20)

        # Validate via checkpoint API
        checkpoint = pfc.checkpoint_manager.get_neuromorphic_state()
        neurons = checkpoint["neurons"]

        assert len(neurons) == 70
        # Original neurons have step 0
        for neuron in neurons[:50]:
            assert neuron["created_step"] == 0, "Original neurons should have birth_step=0"
        # New neurons have step 5000
        for neuron in neurons[50:]:
            assert neuron["created_step"] == 5000, "New neurons should have birth_step=5000"

    def test_multiple_growth_events(self):
        """Multiple growth events should track different timesteps."""
        pfc = create_test_prefrontal(input_size=64, n_neurons=50, device="cpu")

        # First growth at step 1000
        pfc.set_training_step(1000)
        pfc.grow_neurons(n_new=10)

        # Validate first growth via checkpoint
        checkpoint = pfc.checkpoint_manager.get_neuromorphic_state()
        neurons = checkpoint["neurons"]
        assert len(neurons) == 60
        for neuron in neurons[:50]:
            assert neuron["created_step"] == 0
        for neuron in neurons[50:60]:
            assert neuron["created_step"] == 1000

        # Second growth at step 5000
        pfc.set_training_step(5000)
        pfc.grow_neurons(n_new=15)

        # Validate second growth via checkpoint
        checkpoint = pfc.checkpoint_manager.get_neuromorphic_state()
        neurons = checkpoint["neurons"]
        assert len(neurons) == 75
        for neuron in neurons[:50]:
            assert neuron["created_step"] == 0, "Original neurons"
        for neuron in neurons[50:60]:
            assert neuron["created_step"] == 1000, "First growth neurons"
        for neuron in neurons[60:75]:
            assert neuron["created_step"] == 5000, "Second growth neurons"

    def test_checkpoint_uses_birth_steps(self):
        """Checkpoint neuromorphic format should include birth steps."""
        pfc = create_test_prefrontal(input_size=64, n_neurons=50, device="cpu")

        # Grow at different timesteps
        pfc.set_training_step(1000)
        pfc.grow_neurons(n_new=10)

        # Get neuromorphic checkpoint
        state = pfc.checkpoint_manager.get_neuromorphic_state()

        assert "neurons" in state
        neurons = state["neurons"]
        assert len(neurons) == 60

        # Check neuron IDs contain birth steps
        for i in range(50):
            assert neurons[i]["id"] == f"pfc_neuron_{i}_step0"
            assert neurons[i]["created_step"] == 0

        for i in range(50, 60):
            assert neurons[i]["id"] == f"pfc_neuron_{i}_step1000"
            assert neurons[i]["created_step"] == 1000


class TestHippocampusNeurogenesisTracking:
    """Test neurogenesis history tracking for Hippocampus."""

    def test_initial_birth_steps_all_zero(self):
        """All initial neurons should have birth_step=0."""
        hippo = create_test_hippocampus(input_size=64, total_neurons=512, device="cpu")

        # Validate via checkpoint API
        checkpoint = hippo.checkpoint_manager.get_neuromorphic_state()
        assert "neurons" in checkpoint
        neurons = checkpoint["neurons"]

        # All layers should have neurons with birth_step=0
        dg_neurons = [n for n in neurons if n["layer"] == "DG"]
        ca3_neurons = [n for n in neurons if n["layer"] == "CA3"]
        ca1_neurons = [n for n in neurons if n["layer"] == "CA1"]

        assert len(dg_neurons) > 0, "Should have DG neurons"
        assert len(ca3_neurons) > 0, "Should have CA3 neurons"
        assert len(ca1_neurons) > 0, "Should have CA1 neurons"

        for neuron in neurons:
            assert (
                neuron["created_step"] == 0
            ), f"Initial {neuron['layer']} neurons should have birth_step=0"

    def test_training_step_update(self):
        """Training step should be updateable."""
        hippo = create_test_hippocampus(input_size=64, total_neurons=200, device="cpu")

        hippo.set_training_step(2000)
        # Verify training step persisted by growing and checking birth step
        hippo.grow_layer("CA1", n_new=2)

        checkpoint = hippo.checkpoint_manager.get_neuromorphic_state()
        neurons = checkpoint["neurons"]
        # New CA1 neurons should have been born at step 2000
        ca1_neurons = [n for n in neurons if n["layer"] == "CA1"]
        new_neurons = [n for n in ca1_neurons if n["created_step"] == 2000]
        assert len(new_neurons) > 0, "Should have new neurons born at step 2000"

    def test_grow_output_tracks_all_layers(self):
        """New neurons in all layers should record their birth timestep."""
        hippo = create_test_hippocampus(input_size=64, total_neurons=512, device="cpu")

        initial_dg = hippo.dg_size
        initial_ca3 = hippo.ca3_size
        initial_ca1 = hippo.ca1_size

        # Set training step and grow
        hippo.set_training_step(3000)
        hippo.grow_layer("CA1", n_new=8)  # Add 8 CA1 neurons

        # Check all layers grew
        assert hippo.dg_size > initial_dg
        assert hippo.ca3_size > initial_ca3
        assert hippo.ca1_size == initial_ca1 + 8

        # Validate birth steps via checkpoint API
        checkpoint = hippo.checkpoint_manager.get_neuromorphic_state()
        neurons = checkpoint["neurons"]

        # Group neurons by layer (sorting by ID not reliable - just count by birth step)
        dg_neurons = [n for n in neurons if n["layer"] == "DG"]
        ca3_neurons = [n for n in neurons if n["layer"] == "CA3"]
        ca1_neurons = [n for n in neurons if n["layer"] == "CA1"]

        # Check that some neurons are original (birth_step=0) and some are new (birth_step=3000)
        dg_original = [n for n in dg_neurons if n["created_step"] == 0]
        dg_new = [n for n in dg_neurons if n["created_step"] == 3000]

        ca3_original = [n for n in ca3_neurons if n["created_step"] == 0]
        ca3_new = [n for n in ca3_neurons if n["created_step"] == 3000]

        ca1_original = [n for n in ca1_neurons if n["created_step"] == 0]
        ca1_new = [n for n in ca1_neurons if n["created_step"] == 3000]

        # All layers should have both original and new neurons
        assert len(dg_original) == initial_dg, f"Expected {initial_dg} original DG neurons"
        assert len(dg_new) > 0, "Should have new DG neurons"

        assert len(ca3_original) == initial_ca3, f"Expected {initial_ca3} original CA3 neurons"
        assert len(ca3_new) > 0, "Should have new CA3 neurons"

        assert len(ca1_original) == initial_ca1, f"Expected {initial_ca1} original CA1 neurons"
        assert len(ca1_new) == 8, "Should have exactly 8 new CA1 neurons (as requested)"

        # Grow at specific timestep
        hippo.set_training_step(4500)
        hippo.grow_layer("CA1", n_new=4)

        # Get neuromorphic checkpoint
        state = hippo.checkpoint_manager.get_neuromorphic_state()

        assert "neurons" in state
        neurons = state["neurons"]

        # Check neuron IDs contain correct birth steps
        # Initial neurons should have step0, new ones should have step4500
        dg_neurons = [n for n in neurons if n["layer"] == "DG"]
        ca3_neurons = [n for n in neurons if n["layer"] == "CA3"]
        ca1_neurons = [n for n in neurons if n["layer"] == "CA1"]

        # At least some neurons should have birth_step=0
        initial_dg = sum(1 for n in dg_neurons if n["created_step"] == 0)
        initial_ca3 = sum(1 for n in ca3_neurons if n["created_step"] == 0)
        initial_ca1 = sum(1 for n in ca1_neurons if n["created_step"] == 0)

        assert initial_dg > 0
        assert initial_ca3 > 0
        assert initial_ca1 > 0

        # And some should have birth_step=4500
        new_dg = sum(1 for n in dg_neurons if n["created_step"] == 4500)
        new_ca3 = sum(1 for n in ca3_neurons if n["created_step"] == 4500)
        new_ca1 = sum(1 for n in ca1_neurons if n["created_step"] == 4500)

        assert new_dg > 0
        assert new_ca3 > 0
        assert new_ca1 > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
