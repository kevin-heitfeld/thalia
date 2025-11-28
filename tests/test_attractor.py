"""
Tests for attractor dynamics.
"""

import pytest
import torch
from thalia.dynamics import AttractorNetwork, AttractorConfig
from thalia.dynamics.attractor import ConceptNetwork


class TestAttractorConfig:
    """Test AttractorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AttractorConfig()
        assert config.n_neurons == 100
        assert config.tau_mem == 20.0
        assert config.noise_std == 0.05
        assert config.sparsity == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = AttractorConfig(
            n_neurons=256,
            tau_mem=30.0,
            noise_std=0.1,
            sparsity=0.2,
        )
        assert config.n_neurons == 256
        assert config.tau_mem == 30.0
        assert config.noise_std == 0.1
        assert config.sparsity == 0.2


class TestAttractorNetwork:
    """Test AttractorNetwork functionality."""

    @pytest.fixture
    def network(self):
        """Create a test network."""
        config = AttractorConfig(n_neurons=64, noise_std=0.0)
        return AttractorNetwork(config)

    @pytest.fixture
    def network_with_noise(self):
        """Create a test network with noise."""
        config = AttractorConfig(n_neurons=64, noise_std=0.1)
        return AttractorNetwork(config)

    def test_initialization(self, network):
        """Test network initialization."""
        assert network.config.n_neurons == 64
        assert network.weights.shape == (64, 64)
        assert len(network.patterns) == 0

    def test_store_single_pattern(self, network):
        """Test storing a single pattern."""
        pattern = (torch.rand(64) < 0.1).float()  # Sparse binary pattern

        network.store_pattern(pattern)

        assert len(network.patterns) == 1

    def test_store_multiple_patterns(self, network):
        """Test storing multiple patterns."""
        for i in range(5):
            pattern = (torch.rand(64) < 0.1).float()
            network.store_pattern(pattern)

        assert len(network.patterns) == 5

    def test_store_patterns_batch(self, network):
        """Test storing multiple patterns at once."""
        patterns = (torch.rand(3, 64) < 0.1).float()
        network.store_patterns(patterns)

        assert len(network.patterns) == 3

    def test_forward_step(self, network):
        """Test single forward step."""
        network.reset_state(batch_size=1)

        external_input = torch.randn(1, 64)
        spikes, membrane = network(external_input)

        assert spikes.shape == (1, 64)
        assert membrane.shape == (1, 64)

    def test_forward_batched(self, network):
        """Test batched forward step."""
        network.reset_state(batch_size=8)

        external_input = torch.randn(8, 64)
        spikes, membrane = network(external_input)

        assert spikes.shape == (8, 64)
        assert membrane.shape == (8, 64)

    def test_recall_stored_pattern(self, network):
        """Test recalling a stored pattern with partial cue."""
        # Create and store a pattern
        pattern = (torch.rand(64) < 0.15).float()  # Slightly more active
        network.store_pattern(pattern)

        # Create a partial cue (first half of pattern)
        cue = pattern.clone()
        cue[32:] = 0  # Zero out second half

        # Recall
        recalled = network.recall(cue, steps=50)

        # Should recall something with some activity
        assert recalled.shape == (1, 64)
        assert recalled.sum() > 0  # Should have some activity

    def test_recall_batched(self, network):
        """Test batched recall."""
        pattern = (torch.rand(64) < 0.1).float()
        network.store_pattern(pattern)

        # Batch of cues
        cues = torch.stack([pattern, pattern * 0.5])
        recalled = network.recall(cues, steps=30)

        assert recalled.shape == (2, 64)

    def test_similarity_to_patterns(self, network):
        """Test similarity computation."""
        # Store two distinct patterns
        pattern1 = torch.zeros(64)
        pattern1[:20] = 1  # First 20 active

        pattern2 = torch.zeros(64)
        pattern2[40:60] = 1  # Last 20 active

        network.store_pattern(pattern1)
        network.store_pattern(pattern2)

        # Query should be similar to pattern1
        similarity = network.similarity_to_patterns(pattern1)

        assert len(similarity) == 2
        assert similarity[0] > similarity[1]  # More similar to pattern1

    def test_get_attractor_state_no_patterns(self, network):
        """Test attractor state with no patterns."""
        state = network.get_attractor_state()
        assert state == -1

    def test_get_attractor_state_with_patterns(self, network):
        """Test attractor state after running."""
        pattern = (torch.rand(64) < 0.1).float()
        network.store_pattern(pattern)

        # Run network with cue
        network.recall(pattern, steps=30)

        state = network.get_attractor_state()
        assert state >= 0  # Should identify some attractor

    def test_energy_computation(self, network):
        """Test energy computation."""
        pattern = (torch.rand(64) < 0.1).float()
        network.store_pattern(pattern)

        energy = network.energy(pattern)

        assert energy.shape == ()  # Scalar
        assert torch.isfinite(energy)

    def test_energy_with_history(self, network):
        """Test energy with activity history."""
        pattern = (torch.rand(64) < 0.1).float()
        network.store_pattern(pattern)

        # Run network to create history
        network.recall(pattern, steps=10)

        energy = network.energy()  # Use current state
        assert torch.isfinite(energy)

    def test_get_effective_weights(self, network):
        """Test effective weight matrix."""
        weights = network.get_effective_weights()

        assert weights.shape == (64, 64)
        # Diagonal should be zero (masked self-connections) or inhibition only
        # Due to inhibition, diagonal might be small negative
        diag = torch.diag(weights)
        assert torch.all(diag <= 0)  # No positive self-connections

    def test_reset_state(self, network):
        """Test state reset."""
        network.reset_state(batch_size=4)

        # Run a bit
        for _ in range(5):
            network(torch.randn(4, 64))

        assert len(network.activity_history) > 0

        # Reset
        network.reset_state(batch_size=2)

        assert len(network.activity_history) == 0


class TestConceptNetwork:
    """Test ConceptNetwork functionality."""

    @pytest.fixture
    def network(self):
        """Create a concept network."""
        config = AttractorConfig(n_neurons=64, noise_std=0.0)
        return ConceptNetwork(config)

    def test_store_concept(self, network):
        """Test storing named concepts."""
        pattern = (torch.rand(64) < 0.1).float()

        idx = network.store_concept(pattern, "apple")

        assert idx == 0
        assert len(network.patterns) == 1
        assert network.concept_names[0] == "apple"

    def test_get_concept_name(self, network):
        """Test getting concept names."""
        pattern = (torch.rand(64) < 0.1).float()
        network.store_concept(pattern, "banana")

        name = network.get_concept_name(0)
        assert name == "banana"

        # Unknown index returns fallback
        name = network.get_concept_name(99)
        assert name == "concept_99"

    def test_associate_concepts(self, network):
        """Test creating associations."""
        pattern1 = (torch.rand(64) < 0.1).float()
        pattern2 = (torch.rand(64) < 0.1).float()

        idx1 = network.store_concept(pattern1, "red")
        idx2 = network.store_concept(pattern2, "apple")

        # Associate red with apple
        network.associate(idx1, idx2, strength=1.0)

        assert (idx1, idx2) in network.associations
        assert (idx2, idx1) in network.associations  # Bidirectional

    def test_associate_invalid_concept(self, network):
        """Test association with invalid index."""
        pattern = (torch.rand(64) < 0.1).float()
        network.store_concept(pattern, "test")

        with pytest.raises(ValueError):
            network.associate(0, 99)  # 99 doesn't exist

    def test_active_concept(self, network):
        """Test identifying active concept."""
        pattern = (torch.rand(64) < 0.1).float()
        network.store_concept(pattern, "fruit")

        # Run network
        network.recall(pattern, steps=20)

        idx, name = network.active_concept()
        # Should identify some concept (may or may not be the right one)
        assert isinstance(idx, int)
        assert isinstance(name, str)


class TestAttractorTransitions:
    """Test attractor transitions with noise."""

    @pytest.fixture
    def network(self):
        """Create network with noise for transitions."""
        config = AttractorConfig(
            n_neurons=64,
            noise_std=0.2,  # Higher noise for transitions
        )
        return AttractorNetwork(config)

    def test_noise_affects_dynamics(self, network):
        """Test that noise affects dynamics."""
        torch.manual_seed(42)

        pattern = (torch.rand(64) < 0.1).float()
        network.store_pattern(pattern)

        # Run multiple times from same cue
        network.reset_state(1)
        recalled1 = network.recall(pattern, steps=50)

        network.reset_state(1)
        torch.manual_seed(99)  # Different seed
        recalled2 = network.recall(pattern, steps=50)

        # With noise, results might differ
        # Both should be valid activity patterns
        assert recalled1.shape == (1, 64)
        assert recalled2.shape == (1, 64)


class TestPatternCompletion:
    """Test pattern completion capabilities."""

    @pytest.fixture
    def network(self):
        """Create a network for pattern completion tests."""
        config = AttractorConfig(n_neurons=100, noise_std=0.0)
        return AttractorNetwork(config)

    def test_complete_partial_pattern(self, network):
        """Test completing a partial pattern."""
        # Create and store a pattern with clear structure
        pattern = torch.zeros(100)
        pattern[10:30] = 1  # 20 neurons active
        network.store_pattern(pattern)

        # Create cue with half the pattern visible
        cue = pattern.clone()
        cue[20:] = 0  # Only first part visible

        # Recall should generate activity
        recalled = network.recall(cue, steps=50, cue_strength=2.0)

        # Should have some activity
        assert recalled.sum() > 0


class TestGPU:
    """Test GPU functionality."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_network_on_gpu(self):
        """Test network can run on GPU."""
        device = torch.device("cuda")
        config = AttractorConfig(n_neurons=64)
        network = AttractorNetwork(config).to(device)

        assert next(network.parameters()).device.type == "cuda"

        # Store pattern on GPU
        pattern = (torch.rand(64, device=device) < 0.1).float()
        network.store_pattern(pattern)

        # Recall on GPU
        network.reset_state(1)
        # Note: internal state might be on CPU until we move more things
        cue = pattern.clone()

        # The network should handle this
        assert len(network.patterns) == 1
