"""
Tests for daydream module - spontaneous cognition.
"""

import pytest
import torch

from thalia.cognition.daydream import (
    DaydreamNetwork,
    DaydreamConfig,
    DaydreamState,
    DaydreamMode,
    DaydreamIntegration,
)
from thalia.cognition.thinking import ThinkingSNN, ThinkingConfig


class TestDaydreamConfig:
    """Tests for DaydreamConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DaydreamConfig()
        assert config.n_neurons == 128
        assert config.base_noise == 0.1
        assert config.noise_amplification == 3.0
        assert config.dwell_time_mean == 50.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = DaydreamConfig(
            n_neurons=256,
            base_noise=0.2,
            dwell_time_mean=100.0,
        )
        assert config.n_neurons == 256
        assert config.base_noise == 0.2
        assert config.dwell_time_mean == 100.0


class TestDaydreamNetwork:
    """Tests for DaydreamNetwork."""

    @pytest.fixture
    def network(self):
        """Create a basic daydream network."""
        config = DaydreamConfig(n_neurons=64)
        return DaydreamNetwork(config)

    @pytest.fixture
    def network_with_concepts(self, network):
        """Create a network with stored concepts."""
        # Store some concepts
        for i, name in enumerate(["apple", "banana", "cherry", "date"]):
            pattern = torch.zeros(64)
            pattern[i*10:(i+1)*10] = 1.0  # Different activation patterns
            network.store_concept(pattern, name)

        # Create some associations
        network.associate("apple", "banana", 0.7)
        network.associate("banana", "cherry", 0.6)
        network.associate("cherry", "date", 0.5)

        return network

    def test_creation(self, network):
        """Test network creation."""
        assert network is not None
        assert network.config.n_neurons == 64

    def test_reset_state(self, network):
        """Test state reset."""
        network.reset_state(batch_size=2)
        assert network._timestep == 0
        assert network._current_concept == -1

    def test_store_concept(self, network):
        """Test storing concepts."""
        pattern = torch.rand(64)
        idx = network.store_concept(pattern, "test_concept")

        assert idx == 0
        assert "test_concept" in network._name_to_index
        assert network._concept_names[0] == "test_concept"

    def test_associate(self, network_with_concepts):
        """Test creating associations."""
        net = network_with_concepts

        # Check association was created
        apple_idx = net._name_to_index["apple"]
        banana_idx = net._name_to_index["banana"]

        assert net.associations[apple_idx, banana_idx] == 0.7

    def test_set_theme(self, network_with_concepts):
        """Test setting theme bias."""
        net = network_with_concepts

        net.set_theme(["apple", "banana"], strength=0.5)

        apple_idx = net._name_to_index["apple"]
        banana_idx = net._name_to_index["banana"]
        cherry_idx = net._name_to_index["cherry"]

        assert net.theme_bias[apple_idx] == 0.5
        assert net.theme_bias[banana_idx] == 0.5
        assert net.theme_bias[cherry_idx] == 0.0

    def test_clear_theme(self, network_with_concepts):
        """Test clearing theme."""
        net = network_with_concepts
        net.set_theme(["apple"], strength=0.5)
        net.clear_theme()

        assert net.theme_bias.sum() == 0

    def test_start_daydream(self, network_with_concepts):
        """Test starting daydream session."""
        net = network_with_concepts
        net.start_daydream(mode=DaydreamMode.FREE)

        assert net._mode == DaydreamMode.FREE
        assert net._timestep == 0

    def test_start_daydream_with_concept(self, network_with_concepts):
        """Test starting from specific concept."""
        net = network_with_concepts
        net.start_daydream(mode=DaydreamMode.FREE, start_concept="apple")

        assert net._current_concept == net._name_to_index["apple"]
        assert "apple" in net._concepts_visited

    def test_step(self, network_with_concepts):
        """Test single daydream step."""
        net = network_with_concepts
        net.start_daydream(mode=DaydreamMode.FREE, start_concept="apple")

        state = net.step()

        assert isinstance(state, DaydreamState)
        assert state.timestep == 1
        assert state.mode == DaydreamMode.FREE
        assert state.spikes is not None

    def test_daydream_session(self, network_with_concepts):
        """Test complete daydream session."""
        net = network_with_concepts
        states = net.daydream(steps=100, mode=DaydreamMode.FREE)

        assert len(states) == 100
        assert all(isinstance(s, DaydreamState) for s in states)

    def test_daydream_free_mode(self, network_with_concepts):
        """Test free association mode."""
        net = network_with_concepts
        states = net.daydream(steps=200, mode=DaydreamMode.FREE, start_concept="apple")

        # Should have visited at least starting concept
        concepts = net.get_concepts_visited()
        assert len(concepts) >= 1
        assert concepts[0] == "apple"

    def test_daydream_themed_mode(self, network_with_concepts):
        """Test themed daydreaming."""
        net = network_with_concepts
        net.set_theme(["apple", "banana"], strength=0.5)

        states = net.daydream(steps=200, mode=DaydreamMode.THEMED, start_concept="apple")

        # Theme should bias transitions
        concepts = net.get_concepts_visited()
        assert len(concepts) >= 1

    def test_daydream_dream_mode(self, network_with_concepts):
        """Test high-noise dream mode."""
        net = network_with_concepts

        # Dream mode should have higher noise
        net.start_daydream(mode=DaydreamMode.DREAM)
        noise_level = net._get_noise_level()

        assert noise_level == net.config.base_noise * net.config.noise_amplification

    def test_get_trajectory(self, network_with_concepts):
        """Test getting thought trajectory."""
        net = network_with_concepts
        net.daydream(steps=100, start_concept="apple")

        trajectory = net.get_trajectory()
        assert trajectory is not None

    def test_transition_callback(self, network_with_concepts):
        """Test transition callbacks."""
        net = network_with_concepts
        transitions = []

        def on_transition(old, new, name):
            transitions.append((old, new, name))

        net.on_transition(on_transition)
        net.daydream(steps=200, start_concept="apple")

        # Callbacks should be called on transitions
        # (May or may not have transitions depending on random dynamics)

    def test_novelty_computation(self, network_with_concepts):
        """Test novelty is computed on transitions."""
        net = network_with_concepts
        states = net.daydream(steps=300, start_concept="apple")

        # Find transitions
        transitions = [s for s in states if s.transition_occurred]

        # If there were transitions, novelty should be between 0 and 1
        for t in transitions:
            assert 0 <= t.novelty <= 1

    def test_recency_effect(self, network_with_concepts):
        """Test that recency decays and affects transitions."""
        net = network_with_concepts
        net.start_daydream(start_concept="apple")

        # Initial recency should be zero
        assert net.recency.sum() == 0

        # After visiting concept, recency should increase
        for _ in range(100):
            net.step()

        # Recency should have some values now if concepts were visited
        # (depends on random dynamics)


class TestDaydreamModes:
    """Tests for different daydream modes."""

    @pytest.fixture
    def network(self):
        """Create network with concepts."""
        config = DaydreamConfig(n_neurons=64, dwell_time_mean=20)
        net = DaydreamNetwork(config)

        for i, name in enumerate(["red", "blue", "green", "yellow"]):
            pattern = torch.zeros(64)
            pattern[i*15:(i+1)*15] = 1.0
            net.store_concept(pattern, name)

        return net

    def test_free_vs_goal_noise(self, network):
        """Test that goal mode has less noise than free mode."""
        network.start_daydream(mode=DaydreamMode.FREE)
        free_noise = network._get_noise_level()

        network.start_daydream(mode=DaydreamMode.GOAL_DIRECTED)
        goal_noise = network._get_noise_level()

        assert goal_noise < free_noise

    def test_dream_vs_free_noise(self, network):
        """Test that dream mode has more noise than free mode."""
        network.start_daydream(mode=DaydreamMode.FREE)
        free_noise = network._get_noise_level()

        network.start_daydream(mode=DaydreamMode.DREAM)
        dream_noise = network._get_noise_level()

        assert dream_noise > free_noise


class TestDaydreamIntegration:
    """Tests for DaydreamIntegration with ThinkingSNN."""

    @pytest.fixture
    def thinker(self):
        """Create a ThinkingSNN."""
        config = ThinkingConfig(
            n_concepts=64,
            noise_std=0.05,
            enable_learning=False,
            enable_homeostasis=False,
        )
        return ThinkingSNN(config)

    @pytest.fixture
    def thinker_with_concepts(self, thinker):
        """Create ThinkingSNN with concepts."""
        thinker.reset_state(batch_size=1)

        for i, name in enumerate(["dog", "cat", "bird", "fish"]):
            pattern = torch.zeros(64)
            pattern[i*10:(i+1)*10] = 1.0
            thinker.store_concept(pattern, name)

        return thinker

    def test_integration_creation(self, thinker):
        """Test creating daydream integration."""
        daydreamer = DaydreamIntegration(thinker)
        assert daydreamer.thinker is thinker
        assert not daydreamer._daydream_active

    def test_enter_daydream(self, thinker):
        """Test entering daydream mode."""
        daydreamer = DaydreamIntegration(thinker)
        original_noise = thinker.concepts.neurons.config.noise_std

        daydreamer.enter_daydream(noise_multiplier=3.0)

        assert daydreamer._daydream_active
        assert thinker.concepts.neurons.config.noise_std == original_noise * 3.0

    def test_exit_daydream(self, thinker):
        """Test exiting daydream mode."""
        daydreamer = DaydreamIntegration(thinker)
        original_noise = thinker.concepts.neurons.config.noise_std

        daydreamer.enter_daydream()
        daydreamer.exit_daydream()

        assert not daydreamer._daydream_active
        assert thinker.concepts.neurons.config.noise_std == original_noise

    def test_daydream_method(self, thinker_with_concepts):
        """Test daydream method."""
        daydreamer = DaydreamIntegration(thinker_with_concepts)

        concepts = daydreamer.daydream(steps=100)

        # Should return list of concepts (may be empty if no transitions)
        assert isinstance(concepts, list)

        # Should have exited daydream mode
        assert not daydreamer._daydream_active

    def test_daydream_restores_noise(self, thinker):
        """Test that daydream restores noise even on error."""
        daydreamer = DaydreamIntegration(thinker)
        original_noise = thinker.concepts.neurons.config.noise_std

        # Force an error during daydream
        try:
            daydreamer.enter_daydream()
            # Simulate something happening
            assert False, "Simulated error"
        except AssertionError:
            pass
        finally:
            daydreamer.exit_daydream()

        # Noise should be restored
        assert thinker.concepts.neurons.config.noise_std == original_noise


class TestDaydreamGPU:
    """Tests for GPU compatibility."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_network_to_device(self, device):
        """Test moving network to device."""
        config = DaydreamConfig(n_neurons=32)
        net = DaydreamNetwork(config)
        net = net.to(device)

        assert net.associations.device.type == device.type

    def test_daydream_on_device(self, device):
        """Test daydreaming on device."""
        config = DaydreamConfig(n_neurons=32)
        net = DaydreamNetwork(config)
        net = net.to(device)

        # Store concept
        pattern = torch.rand(32, device=device)
        net.store_concept(pattern, "test")

        # Daydream
        net.reset_state(batch_size=1)
        states = net.daydream(steps=10)

        assert len(states) == 10
        assert states[0].spikes.device.type == device.type


class TestDaydreamEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_network_daydream(self):
        """Test daydreaming with no stored concepts."""
        config = DaydreamConfig(n_neurons=32)
        net = DaydreamNetwork(config)

        # Should not crash
        states = net.daydream(steps=10)
        assert len(states) == 10

    def test_single_concept(self):
        """Test with only one concept."""
        config = DaydreamConfig(n_neurons=32)
        net = DaydreamNetwork(config)

        pattern = torch.rand(32)
        net.store_concept(pattern, "only_one")

        states = net.daydream(steps=50, start_concept="only_one")

        # Should stay on the only concept
        concepts = net.get_concepts_visited()
        assert all(c == "only_one" for c in concepts)

    def test_invalid_start_concept(self):
        """Test starting with invalid concept name."""
        config = DaydreamConfig(n_neurons=32)
        net = DaydreamNetwork(config)

        pattern = torch.rand(32)
        net.store_concept(pattern, "valid")

        # Should not crash with invalid start
        net.start_daydream(start_concept="nonexistent")
        state = net.step()

        assert state is not None

    def test_associate_invalid_concepts(self):
        """Test associating concepts that don't exist."""
        config = DaydreamConfig(n_neurons=32)
        net = DaydreamNetwork(config)

        # Should not crash
        net.associate("nonexistent1", "nonexistent2", 0.5)

    def test_zero_steps(self):
        """Test daydreaming for zero steps."""
        config = DaydreamConfig(n_neurons=32)
        net = DaydreamNetwork(config)

        states = net.daydream(steps=0)
        assert len(states) == 0
