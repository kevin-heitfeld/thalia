"""Tests for test fixtures and utilities.

Validates that all standard fixtures work correctly and provide
consistent, expected values.
"""

import pytest
import torch
from tests.test_utils import TestFixtures


class TestStandardFixtures:
    """Test that standard pytest fixtures work correctly."""
    
    def test_device_fixture(self, device):
        """Test device fixture returns valid device."""
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]
    
    def test_dimension_fixtures(self, batch_size, n_neurons, n_timesteps, 
                               small_n_input, small_n_output):
        """Test dimension fixtures provide expected values."""
        assert batch_size == 1  # Single instance
        assert n_neurons == 100
        assert n_timesteps == 100
        assert small_n_input == 32
        assert small_n_output == 16


class TestRegionConfigFixtures:
    """Test that region config fixtures work correctly."""
    
    def test_layered_cortex_config(self, layered_cortex_config, small_n_input, small_n_output):
        """Test LayeredCortex config fixture."""
        assert layered_cortex_config.n_input == small_n_input
        assert layered_cortex_config.n_output == small_n_output
        assert layered_cortex_config.dual_output is True
    
    def test_cerebellum_config(self, cerebellum_config, small_n_input, small_n_output):
        """Test Cerebellum config fixture."""
        assert cerebellum_config.n_input == small_n_input
        assert cerebellum_config.n_output == small_n_output
    
    def test_striatum_config(self, striatum_config, small_n_input):
        """Test Striatum config fixture."""
        assert striatum_config.n_input == small_n_input
        assert striatum_config.n_output == 8  # 8 actions
        assert striatum_config.population_coding is True
    
    def test_prefrontal_config(self, prefrontal_config, small_n_input):
        """Test Prefrontal config fixture."""
        assert prefrontal_config.n_input == small_n_input
        assert prefrontal_config.n_output == small_n_input
    
    def test_hippocampus_config(self, hippocampus_config, small_n_input, small_n_output):
        """Test Hippocampus config fixture."""
        assert hippocampus_config.n_input == small_n_input
        assert hippocampus_config.n_output == small_n_output


class TestRegionFixtures:
    """Test that instantiated region fixtures work correctly."""
    
    def test_layered_cortex_fixture(self, layered_cortex, small_n_input):
        """Test LayeredCortex fixture creates valid region."""
        assert layered_cortex is not None
        # Test basic forward pass
        input_spikes = torch.randn(small_n_input)
        output = layered_cortex.forward(input_spikes)
        assert output is not None
    
    def test_cerebellum_fixture(self, cerebellum):
        """Test Cerebellum fixture creates valid region."""
        assert cerebellum is not None
    
    def test_striatum_fixture(self, striatum):
        """Test Striatum fixture creates valid region."""
        assert striatum is not None
    
    def test_prefrontal_fixture(self, prefrontal):
        """Test Prefrontal fixture creates valid region."""
        assert prefrontal is not None
    
    def test_hippocampus_fixture(self, hippocampus):
        """Test Hippocampus fixture creates valid region."""
        assert hippocampus is not None


class TestNeuronFixtures:
    """Test that neuron fixtures work correctly."""
    
    def test_lif_config_fixture(self, lif_config):
        """Test LIF config fixture."""
        assert lif_config.v_threshold == 1.0
        assert lif_config.v_rest == 0.0
        assert lif_config.tau_mem == 20.0
    
    def test_lif_neuron_fixture(self, lif_neuron, n_neurons):
        """Test LIF neuron fixture."""
        assert lif_neuron is not None
        assert lif_neuron.n_neurons == n_neurons
    
    def test_conductance_lif_fixture(self, conductance_lif_neuron, n_neurons):
        """Test conductance LIF fixture."""
        assert conductance_lif_neuron is not None
        assert conductance_lif_neuron.n_neurons == n_neurons
    
    def test_dendritic_neuron_fixture(self, dendritic_neuron):
        """Test dendritic neuron fixture."""
        assert dendritic_neuron is not None
        assert dendritic_neuron.n_neurons == 20


class TestInputFixtures:
    """Test that input fixtures work correctly."""
    
    def test_standard_input(self, standard_input, batch_size, n_neurons):
        """Test standard input fixture."""
        assert standard_input.shape == (batch_size, n_neurons)
        assert torch.isfinite(standard_input).all()
    
    def test_poisson_spikes(self, poisson_spikes, batch_size, n_neurons, n_timesteps):
        """Test Poisson spikes fixture."""
        assert poisson_spikes.shape == (n_timesteps, batch_size, n_neurons)
        assert poisson_spikes.dtype == torch.float32
        # Should be binary
        unique = torch.unique(poisson_spikes)
        assert len(unique) <= 2
        assert all(v in [0.0, 1.0] for v in unique.tolist())
    
    def test_sparse_spikes(self, sparse_spikes):
        """Test sparse spikes have low firing rate."""
        firing_rate = sparse_spikes.mean().item()
        assert 0.0 <= firing_rate < 0.05  # Should be very sparse
    
    def test_dense_spikes(self, dense_spikes):
        """Test dense spikes have higher firing rate."""
        firing_rate = dense_spikes.mean().item()
        assert 0.2 < firing_rate < 0.4  # Should be dense
    
    def test_binary_pattern(self, binary_pattern, n_neurons):
        """Test binary pattern fixture."""
        assert binary_pattern.shape == (n_neurons,)
        assert all(v in [0.0, 1.0] for v in torch.unique(binary_pattern).tolist())
    
    def test_sequence_patterns(self, sequence_patterns, n_neurons):
        """Test sequence patterns fixture."""
        assert sequence_patterns.shape == (5, n_neurons)


class TestTestFixturesClass:
    """Test the TestFixtures utility class."""
    
    @pytest.fixture
    def fixtures(self):
        """Create TestFixtures instance."""
        return TestFixtures(device="cpu")
    
    def test_create_lif_neuron(self, fixtures):
        """Test LIF neuron creation."""
        neuron = fixtures.create_lif_neuron(n_neurons=50)
        assert neuron.n_neurons == 50
    
    def test_create_cortex(self, fixtures):
        """Test cortex creation."""
        cortex = fixtures.create_cortex(n_input=32, n_output=16)
        assert cortex.config.n_input == 32
        # With dual_output=True, actual output = l23 + l5 = 24 + 16 = 40
        assert cortex.config.n_output == 40
    
    def test_create_poisson_spikes(self, fixtures):
        """Test Poisson spike generation."""
        spikes = fixtures.create_poisson_spikes(
            rate=0.2, 
            n_neurons=100, 
            n_timesteps=50
        )
        assert spikes.shape == (50, 1, 100)
        # Check firing rate is approximately correct
        firing_rate = spikes.mean().item()
        assert 0.1 < firing_rate < 0.3  # Roughly 20%
    
    def test_create_learning_scenario(self, fixtures):
        """Test learning scenario creation."""
        scenario = fixtures.create_learning_scenario(n_neurons=100)
        
        assert 'neuron' in scenario
        assert 'input_pattern' in scenario
        assert 'target_pattern' in scenario
        assert 'reward' in scenario
        
        assert scenario['neuron'].n_neurons == 100
        assert scenario['input_pattern'].shape == (1, 100)
        assert scenario['target_pattern'].shape == (100,)
        assert scenario['reward'].item() == 1.0
    
    def test_create_region_test_scenario(self, fixtures):
        """Test region test scenario creation."""
        scenario = fixtures.create_region_test_scenario(n_input=64, n_output=32)
        
        assert 'region' in scenario
        assert 'input_spikes' in scenario
        assert 'expected_output_shape' in scenario
        
        assert scenario['region'].config.n_input == 64
        assert scenario['input_spikes'].shape[2] == 64  # n_neurons dimension
        assert scenario['expected_output_shape'] == (1, 32)


class TestRegionSpecsFixture:
    """Test region specs fixture for factory testing."""
    
    def test_region_specs(self, region_specs):
        """Test region specs fixture provides all standard configs."""
        assert "cortex" in region_specs
        assert "cerebellum" in region_specs
        assert "striatum" in region_specs
        
        # Check configs are valid
        from thalia.regions.cortex import LayeredCortexConfig
        from thalia.regions.cerebellum import CerebellumConfig
        from thalia.regions.striatum import StriatumConfig
        
        assert isinstance(region_specs["cortex"], LayeredCortexConfig)
        assert isinstance(region_specs["cerebellum"], CerebellumConfig)
        assert isinstance(region_specs["striatum"], StriatumConfig)


class TestLearningHelpers:
    """Test learning/training helper fixtures."""
    
    def test_learning_input_pattern(self, learning_input_pattern, n_neurons):
        """Test learning input pattern is consistent."""
        assert learning_input_pattern.shape == (n_neurons,)
        
        # Should be reproducible
        torch.manual_seed(42)
        expected = torch.randn(n_neurons) * 2.0
        assert torch.allclose(learning_input_pattern, expected)
    
    def test_target_spike_pattern(self, target_spike_pattern, n_neurons):
        """Test target spike pattern."""
        assert target_spike_pattern.shape == (n_neurons,)
        assert all(v in [0.0, 1.0] for v in torch.unique(target_spike_pattern).tolist())
    
    def test_reward_signal(self, reward_signal):
        """Test reward signal fixture."""
        assert reward_signal.item() == 1.0
    
    def test_dopamine_signal(self, dopamine_signal):
        """Test dopamine signal fixture."""
        assert dopamine_signal == 0.5


class TestDiagnosticHelpers:
    """Test diagnostic helper fixtures."""
    
    def test_test_weights(self, test_weights, n_neurons):
        """Test weights fixture."""
        assert test_weights.shape == (n_neurons, n_neurons)
        assert (test_weights >= 0).all()  # Should be positive
    
    def test_test_membrane(self, test_membrane, batch_size, n_neurons):
        """Test membrane potential fixture."""
        assert test_membrane.shape == (batch_size, n_neurons)
        assert torch.isfinite(test_membrane).all()


class TestLanguageFixtures:
    """Test language-specific fixtures."""
    
    def test_small_n_neurons(self, small_n_neurons):
        """Test small neurons for language."""
        assert small_n_neurons == 64
    
    def test_small_vocab_size(self, small_vocab_size):
        """Test small vocabulary."""
        assert small_vocab_size == 100
    
    def test_small_n_timesteps(self, small_n_timesteps):
        """Test small timesteps."""
        assert small_n_timesteps == 10
    
    def test_test_tokens(self, test_tokens, small_vocab_size):
        """Test token fixture."""
        assert test_tokens.shape == (2, 10)
        assert (test_tokens >= 0).all()
        assert (test_tokens < small_vocab_size).all()
    
    def test_test_spikes(self, test_spikes, small_n_neurons, small_n_timesteps):
        """Test spike fixture for language."""
        assert test_spikes.shape == (2, 10, small_n_timesteps, small_n_neurons)
        # Should be sparse binary
        unique = torch.unique(test_spikes)
        assert len(unique) <= 2
