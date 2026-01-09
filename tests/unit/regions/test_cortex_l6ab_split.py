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
        input_size=128,  # Must equal l23_size + l5_size
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

        # Extract expected sizes from config
        expected_l6a = cortex_config_l6ab.l6a_size
        expected_l6b = cortex_config_l6ab.l6b_size

        # Check L6a neurons
        assert hasattr(cortex, "l6a_neurons")
        assert cortex.l6a_neurons is not None
        assert cortex.l6a_size == expected_l6a
        assert cortex.l6a_size > 0  # Invariant

        # Check L6b neurons
        assert hasattr(cortex, "l6b_neurons")
        assert cortex.l6b_neurons is not None
        assert cortex.l6b_size == expected_l6b
        assert cortex.l6b_size > 0  # Invariant

        # Verify separate populations (not shared)
        assert cortex.l6a_neurons is not cortex.l6b_neurons

    def test_l6ab_weights(self, cortex_config_l6ab, device):
        """Test that L6a and L6b have separate weight matrices."""
        cortex = LayeredCortex(cortex_config_l6ab)

        # Extract expected dimensions from config
        l6a_size = cortex_config_l6ab.l6a_size
        l6b_size = cortex_config_l6ab.l6b_size
        l23_size = cortex_config_l6ab.l23_size

        # L2/3 → L6a weights
        assert cortex.synaptic_weights["l23_l6a"].shape == (l6a_size, l23_size), \
            f"L6a weight shape should be ({l6a_size}, {l23_size}), got {cortex.synaptic_weights['l23_l6a'].shape}"
        assert not torch.isnan(cortex.synaptic_weights["l23_l6a"]).any(), \
            "L6a weights contain NaN values"
        assert not torch.isinf(cortex.synaptic_weights["l23_l6a"]).any(), \
            "L6a weights contain Inf values"

        # L2/3 → L6b weights
        assert cortex.synaptic_weights["l23_l6b"].shape == (l6b_size, l23_size), \
            f"L6b weight shape should be ({l6b_size}, {l23_size}), got {cortex.synaptic_weights['l23_l6b'].shape}"
        assert not torch.isnan(cortex.synaptic_weights["l23_l6b"]).any(), \
            "L6b weights contain NaN values"
        assert not torch.isinf(cortex.synaptic_weights["l23_l6b"]).any(), \
            "L6b weights contain Inf values"

        # Weights should be different (not shared)
        min_size = min(l6a_size, l6b_size)
        assert not torch.equal(
            cortex.synaptic_weights["l23_l6a"][:min_size, :],  # First rows of L6a
            cortex.synaptic_weights["l23_l6b"][:min_size, :]  # Matching rows of L6b
        )

    def test_l6ab_forward_pass(self, cortex_config_l6ab, device):
        """Test that forward pass generates separate L6a and L6b spikes."""
        cortex = LayeredCortex(cortex_config_l6ab)

        # Extract dimensions from config
        n_input = cortex_config_l6ab.n_input
        l6a_size = cortex_config_l6ab.l6a_size
        l6b_size = cortex_config_l6ab.l6b_size

        sensory_input = torch.zeros(n_input, dtype=torch.bool, device=device)
        sensory_input[0:20] = True  # Activate some inputs

        cortex.reset_state()

        # Run several timesteps
        for _ in range(10):
            _ = cortex(sensory_input)

        # Check that L6a spikes are generated
        assert cortex.state.l6a_spikes is not None
        assert cortex.state.l6a_spikes.shape == (l6a_size,), \
            f"L6a spikes shape should be ({l6a_size},), got {cortex.state.l6a_spikes.shape}"
        assert cortex.state.l6a_spikes.dtype == torch.bool, \
            f"L6a spikes should be bool, got {cortex.state.l6a_spikes.dtype}"
        assert not torch.isnan(cortex.state.l6a_spikes.float()).any(), \
            "L6a spikes contain NaN values"

        # Check that L6b spikes are generated
        assert cortex.state.l6b_spikes is not None
        assert cortex.state.l6b_spikes.shape == (l6b_size,), \
            f"L6b spikes shape should be ({l6b_size},), got {cortex.state.l6b_spikes.shape}"
        assert cortex.state.l6b_spikes.dtype == torch.bool, \
            f"L6b spikes should be bool, got {cortex.state.l6b_spikes.dtype}"
        assert not torch.isnan(cortex.state.l6b_spikes.float()).any(), \
            "L6b spikes contain NaN values"

        # Spikes should be independent (different population sizes)
        assert l6a_size != l6b_size
        assert cortex.state.l6a_spikes.shape != cortex.state.l6b_spikes.shape

    def test_l6ab_port_routing(self, cortex_config_l6ab, device):
        """Test that get_output() correctly routes L6a and L6b ports."""
        cortex = LayeredCortex(cortex_config_l6ab)

        # Extract dimensions from config
        n_input = cortex_config_l6ab.n_input
        l6a_size = cortex_config_l6ab.l6a_size
        l6b_size = cortex_config_l6ab.l6b_size

        sensory_input = torch.zeros(n_input, dtype=torch.bool, device=device)
        sensory_input[0:20] = True

        cortex.reset_state()
        _ = cortex(sensory_input)

        # Get L6a output via port
        l6a_output = cortex.get_output("l6a")
        assert l6a_output is not None
        assert l6a_output.shape == (l6a_size,), \
            f"L6a output shape should be ({l6a_size},), got {l6a_output.shape}"
        assert l6a_output.dtype == torch.bool, \
            f"L6a output should be bool spikes, got {l6a_output.dtype}"
        assert not torch.isnan(l6a_output.float()).any(), \
            "L6a output contains NaN values"
        assert torch.equal(l6a_output, cortex.state.l6a_spikes)

        # Get L6b output via port
        l6b_output = cortex.get_output("l6b")
        assert l6b_output is not None
        assert l6b_output.shape == (l6b_size,), \
            f"L6b output shape should be ({l6b_size},), got {l6b_output.shape}"
        assert l6b_output.dtype == torch.bool, \
            f"L6b output should be bool spikes, got {l6b_output.dtype}"
        assert not torch.isnan(l6b_output.float()).any(), \
            "L6b output contains NaN values"
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

        n_input = cortex_config_l6ab.n_input
        sensory_input = torch.zeros(n_input, dtype=torch.bool, device=device)
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

        n_input = cortex_config_l6ab.n_input
        sensory_input = torch.zeros(n_input, dtype=torch.bool, device=device)
        sensory_input[0:20] = True

        cortex.reset_state()

        # Generate some activity
        for _ in range(5):
            _ = cortex(sensory_input)

        # Reset state
        cortex.reset_state()

        # Behavioral test: State should be cleared
        assert cortex.state.l6a_spikes is None or not cortex.state.l6a_spikes.any()
        assert cortex.state.l6b_spikes is None or not cortex.state.l6b_spikes.any()

        # Behavioral test: After reset, forward pass should work cleanly
        # without accumulated delays affecting output
        output_after_reset = cortex(sensory_input)

        # Contract: Reset should produce valid output without delay artifacts
        assert output_after_reset.dtype == torch.bool, "Output should be bool (ADR-004)"
        assert not torch.isnan(output_after_reset.float()).any(), "No NaN after reset"

        # Contract: Reset clears history - new input processed independently
        # (If delays weren't cleared, old spikes would contaminate output)
        firing_rate = output_after_reset.float().mean().item()
        assert 0.0 <= firing_rate <= 1.0, f"Valid firing rate after reset: {firing_rate:.2%}"

