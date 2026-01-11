"""
Tests for neurogenesis history tracking in Prefrontal and Hippocampus regions.

Tests that neuron birth timestamps are properly tracked and recorded in checkpoints.
"""

import torch
import pytest

from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
from thalia.regions.hippocampus.trisynaptic import TrisynapticHippocampus
from thalia.regions.hippocampus.config import HippocampusConfig


def create_test_prefrontal(input_size: int = 64, n_neurons: int = 50, device: str = "cpu") -> Prefrontal:
    """Create Prefrontal for testing with new (config, sizes, device) pattern."""
    sizes = {"input_size": input_size, "n_neurons": n_neurons}
    config = PrefrontalConfig(device=device)
    return Prefrontal(config, sizes, device)


def create_test_hippocampus(input_size: int = 100, total_neurons: int = 200, device: str = "cpu") -> TrisynapticHippocampus:
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

        assert hasattr(pfc, '_neuron_birth_steps')
        assert pfc._neuron_birth_steps.shape == (50,)
        assert torch.all(pfc._neuron_birth_steps == 0)

    def test_training_step_update(self):
        """Training step should be updateable."""
        pfc = create_test_prefrontal(input_size=64, n_neurons=50, device="cpu")

        pfc.set_training_step(1000)
        assert pfc._current_training_step == 1000

    def test_grow_output_tracks_birth_steps(self):
        """New neurons should record their birth timestep."""
        pfc = create_test_prefrontal(input_size=64, n_neurons=50, device="cpu")

        # Set training step
        pfc.set_training_step(5000)

        # Grow by 20 neurons
        pfc.grow_neurons(n_new=20)

        # Check birth steps
        assert pfc._neuron_birth_steps.shape == (70,)
        assert torch.all(pfc._neuron_birth_steps[:50] == 0)  # Original neurons
        assert torch.all(pfc._neuron_birth_steps[50:] == 5000)  # New neurons

    def test_multiple_growth_events(self):
        """Multiple growth events should track different timesteps."""
        pfc = create_test_prefrontal(input_size=64, n_neurons=50, device="cpu")

        # First growth at step 1000
        pfc.set_training_step(1000)
        pfc.grow_neurons(n_new=10)
        assert pfc._neuron_birth_steps.shape == (60,)
        assert torch.all(pfc._neuron_birth_steps[:50] == 0)
        assert torch.all(pfc._neuron_birth_steps[50:60] == 1000)

        # Second growth at step 5000
        pfc.set_training_step(5000)
        pfc.grow_neurons(n_new=15)
        assert pfc._neuron_birth_steps.shape == (75,)
        assert torch.all(pfc._neuron_birth_steps[:50] == 0)
        assert torch.all(pfc._neuron_birth_steps[50:60] == 1000)
        assert torch.all(pfc._neuron_birth_steps[60:75] == 5000)

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

        assert hasattr(hippo, '_neuron_birth_steps_dg')
        assert hasattr(hippo, '_neuron_birth_steps_ca3')
        assert hasattr(hippo, '_neuron_birth_steps_ca1')

        assert torch.all(hippo._neuron_birth_steps_dg == 0)
        assert torch.all(hippo._neuron_birth_steps_ca3 == 0)
        assert torch.all(hippo._neuron_birth_steps_ca1 == 0)

    def test_training_step_update(self):
        """Training step should be updateable."""
        hippo = create_test_hippocampus(input_size=64, total_neurons=200, device="cpu")

        hippo.set_training_step(2000)
        assert hippo._current_training_step == 2000

    def test_grow_output_tracks_all_layers(self):
        """New neurons in all layers should record their birth timestep."""
        hippo = create_test_hippocampus(input_size=64, total_neurons=512, device="cpu")

        initial_dg = hippo.dg_size
        initial_ca3 = hippo.ca3_size
        initial_ca1 = hippo.ca1_size

        # Set training step and grow
        hippo.set_training_step(3000)
        hippo.grow_layer('CA1', n_new=8)  # Add 8 CA1 neurons

        # Check all layers grew
        assert hippo.dg_size > initial_dg
        assert hippo.ca3_size > initial_ca3
        assert hippo.ca1_size == initial_ca1 + 8

        # Check birth steps for new neurons
        new_dg_start = initial_dg
        new_ca3_start = initial_ca3
        new_ca1_start = initial_ca1

        assert torch.all(hippo._neuron_birth_steps_dg[:new_dg_start] == 0)
        assert torch.all(hippo._neuron_birth_steps_dg[new_dg_start:] == 3000)

        assert torch.all(hippo._neuron_birth_steps_ca3[:new_ca3_start] == 0)
        assert torch.all(hippo._neuron_birth_steps_ca3[new_ca3_start:] == 3000)

        assert torch.all(hippo._neuron_birth_steps_ca1[:new_ca1_start] == 0)
        assert torch.all(hippo._neuron_birth_steps_ca1[new_ca1_start:] == 3000)

    def test_checkpoint_uses_birth_steps(self):
        """Checkpoint neuromorphic format should include birth steps for all layers."""
        hippo = create_test_hippocampus(input_size=64, total_neurons=200, device="cpu")

        # Grow at specific timestep
        hippo.set_training_step(4500)
        hippo.grow_layer('CA1', n_new=4)

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
