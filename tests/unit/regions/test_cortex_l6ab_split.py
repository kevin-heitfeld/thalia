"""
Unit tests for L6a/L6b dual corticothalamic pathway split.

Tests:
- L6a and L6b layer initialization (separate neuron populations)
- Dual pathway architecture (L6a→TRN inhibitory, L6b→relay excitatory)
- Port-based routing ("l6a", "l6b")
- Axonal delay configuration (L6a=10ms slow, L6b=5ms fast)
- Weight initialization for dual pathways
- Forward pass with separate L6a and L6b spike outputs
"""

import pytest
import torch

from thalia.regions.cortex.config import LayeredCortexConfig
from thalia.regions.cortex.layered_cortex import LayeredCortex


@pytest.fixture
def device():
    """Device for testing (CPU by default)."""
    return torch.device("cpu")


@pytest.fixture
def cortex_config_l6ab(device):
    """Cortex configuration with L6a/L6b split."""
    return LayeredCortexConfig(
        n_input=128,
        n_output=320,  # Must equal l23_size + l5_size
        # Specify all layer sizes
        l4_size=128,
        l23_size=192,
        l5_size=128,
        l6a_size=150,  # Type I corticothalamic (→TRN)
        l6b_size=100,  # Type II corticothalamic (→relay)
        l6a_sparsity=0.12,
        l6b_sparsity=0.15,
        l23_to_l6a_strength=1.0,
        l23_to_l6b_strength=1.2,
        l6a_to_trn_strength=0.8,
        l6b_to_relay_strength=1.0,
        l23_to_l6a_delay_ms=0.0,  # Internal delay
        l23_to_l6b_delay_ms=0.0,  # Internal delay
        l6a_to_trn_delay_ms=10.0,  # Slow pathway
        l6b_to_relay_delay_ms=5.0,  # Fast pathway
        dt_ms=1.0,
        device=str(device),
    )


class TestL6abSplit:
    """Test L6a/L6b dual pathway split."""

    def test_l6ab_initialization(self, cortex_config_l6ab, device):
        """Test that L6a and L6b are initialized as separate neuron populations."""
        cortex = LayeredCortex(cortex_config_l6ab)

        # Check L6a neurons
        assert hasattr(cortex, "l6a_neurons")
        assert cortex.l6a_neurons is not None
        assert cortex.l6a_size == 150

        # Check L6b neurons
        assert hasattr(cortex, "l6b_neurons")
        assert cortex.l6b_neurons is not None
        assert cortex.l6b_size == 100

        # Verify separate populations (not shared)
        assert cortex.l6a_neurons is not cortex.l6b_neurons

    def test_l6ab_weights(self, cortex_config_l6ab, device):
        """Test that L6a and L6b have separate weight matrices."""
        cortex = LayeredCortex(cortex_config_l6ab)

        # L2/3 → L6a weights
        assert cortex.w_l23_l6a.shape == (150, 192)

        # L2/3 → L6b weights
        assert cortex.w_l23_l6b.shape == (100, 192)

        # Weights should be different (not shared)
        assert not torch.equal(
            cortex.w_l23_l6a[:100, :],  # First 100 rows of L6a
            cortex.w_l23_l6b  # L6b weights
        )

    def test_l6ab_forward_pass(self, cortex_config_l6ab, device):
        """Test that forward pass generates separate L6a and L6b spikes."""
        cortex = LayeredCortex(cortex_config_l6ab)

        sensory_input = torch.zeros(128, dtype=torch.bool, device=device)
        sensory_input[0:20] = True  # Activate some inputs

        cortex.reset_state()

        # Run several timesteps
        for _ in range(10):
            _ = cortex(sensory_input)

        # Check that L6a spikes are generated
        assert cortex.state.l6a_spikes is not None
        assert cortex.state.l6a_spikes.shape == (150,)
        assert cortex.state.l6a_spikes.dtype == torch.bool

        # Check that L6b spikes are generated
        assert cortex.state.l6b_spikes is not None
        assert cortex.state.l6b_spikes.shape == (100,)
        assert cortex.state.l6b_spikes.dtype == torch.bool

        # Spikes should be independent
        # (not guaranteed to differ every timestep, but check shape/type)
        assert cortex.state.l6a_spikes.shape != cortex.state.l6b_spikes.shape

    def test_l6ab_port_routing(self, cortex_config_l6ab, device):
        """Test that get_output() correctly routes L6a and L6b ports."""
        cortex = LayeredCortex(cortex_config_l6ab)

        sensory_input = torch.zeros(128, dtype=torch.bool, device=device)
        sensory_input[0:20] = True

        cortex.reset_state()
        _ = cortex(sensory_input)

        # Get L6a output via port
        l6a_output = cortex.get_output("l6a")
        assert l6a_output is not None
        assert l6a_output.shape == (150,)
        assert torch.equal(l6a_output, cortex.state.l6a_spikes)

        # Get L6b output via port
        l6b_output = cortex.get_output("l6b")
        assert l6b_output is not None
        assert l6b_output.shape == (100,)
        assert torch.equal(l6b_output, cortex.state.l6b_spikes)

        # Verify outputs are different tensors
        assert l6a_output.shape != l6b_output.shape

    def test_l6ab_delay_configuration(self, cortex_config_l6ab, device):
        """Test that L6a and L6b have correct axonal delay configuration."""
        cortex = LayeredCortex(cortex_config_l6ab)

        # Check internal delays (L2/3 → L6a/L6b)
        assert cortex.config.l23_to_l6a_delay_ms == 0.0
        assert cortex.config.l23_to_l6b_delay_ms == 0.0

        # Check external delays (L6a→TRN, L6b→relay)
        assert cortex.config.l6a_to_trn_delay_ms == 10.0  # Slow pathway
        assert cortex.config.l6b_to_relay_delay_ms == 5.0  # Fast pathway

        # Verify L6a is slower (type I inhibitory modulation)
        assert cortex.config.l6a_to_trn_delay_ms > cortex.config.l6b_to_relay_delay_ms

    def test_l6ab_no_legacy_ports(self, cortex_config_l6ab, device):
        """Test that legacy 'l6' and 'l6_feedback' ports raise ValueError."""
        cortex = LayeredCortex(cortex_config_l6ab)

        sensory_input = torch.zeros(128, dtype=torch.bool, device=device)
        cortex.reset_state()
        _ = cortex(sensory_input)

        # Legacy ports should raise ValueError (not supported)
        with pytest.raises(ValueError, match="Invalid port 'l6'"):
            _ = cortex.get_output("l6")

        with pytest.raises(ValueError, match="Invalid port 'l6_feedback'"):
            _ = cortex.get_output("l6_feedback")

    def test_l6ab_state_reset(self, cortex_config_l6ab, device):
        """Test that reset_state() properly clears L6a and L6b state."""
        cortex = LayeredCortex(cortex_config_l6ab)

        sensory_input = torch.zeros(128, dtype=torch.bool, device=device)
        sensory_input[0:20] = True

        cortex.reset_state()

        # Generate some activity
        for _ in range(5):
            _ = cortex(sensory_input)

        # Reset state
        cortex.reset_state()

        # State should be cleared
        assert cortex.state.l6a_spikes is None or not cortex.state.l6a_spikes.any()
        assert cortex.state.l6b_spikes is None or not cortex.state.l6b_spikes.any()

        # Delay buffers should be reset (single tensors, not lists)
        if hasattr(cortex, "_l23_l6a_delay_buffer") and cortex._l23_l6a_delay_buffer is not None:
            assert not cortex._l23_l6a_delay_buffer.any()

        if hasattr(cortex, "_l23_l6b_delay_buffer") and cortex._l23_l6b_delay_buffer is not None:
            assert not cortex._l23_l6b_delay_buffer.any()
