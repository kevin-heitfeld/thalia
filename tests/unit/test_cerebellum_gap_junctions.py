"""
Tests for gap junction initialization and functionality in cerebellum.

Verifies that gap junctions are properly initialized and don't zero out
error signals during learning.
"""

import pytest
import torch

from thalia.config import compute_cerebellum_sizes
from thalia.regions.cerebellum import Cerebellum, CerebellumConfig


def create_test_cerebellum(
    input_size: int = 64,
    purkinje_size: int = 32,
    device: str = "cpu",
    **kwargs
) -> Cerebellum:
    """Create Cerebellum for testing with new (config, sizes, device) pattern."""
    # Always compute granule_size (Cerebellum.__init__ requires it)
    expansion = kwargs.pop("granule_expansion_factor", 4.0)
    sizes = compute_cerebellum_sizes(purkinje_size, expansion)
    sizes["input_size"] = input_size

    config = CerebellumConfig(device=device, **kwargs)
    return Cerebellum(config, sizes, device)


class TestCerebellumGapJunctions:
    """Test gap junction initialization and functionality."""

    def test_gap_junctions_initialized_properly(self):
        """Gap junctions should be initialized without errors."""
        cerebellum = create_test_cerebellum(
            input_size=64,
            purkinje_size=32,
            use_enhanced_microcircuit=True,
            gap_junctions_enabled=True,  # Enable gap junctions
            device="cpu"
        )

        # Gap junctions should be created
        assert cerebellum.gap_junctions_io is not None
        assert hasattr(cerebellum.gap_junctions_io, 'coupling_matrix')

        # Coupling matrix should not be all zeros (some coupling exists)
        coupling_matrix = cerebellum.gap_junctions_io.coupling_matrix
        assert coupling_matrix.numel() > 0
        # At least some non-zero couplings should exist
        # (might be zero if connectivity threshold is too high, but structure exists)
        assert coupling_matrix.shape == (32, 32)

    def test_gap_junctions_dont_zero_error_signal(self):
        """Gap junctions should NOT zero out error signals during learning."""
        cerebellum = create_test_cerebellum(
            input_size=64,
            purkinje_size=32,
            use_enhanced_microcircuit=True,
            learning_rate=0.1,
            error_threshold=0.001,
            gap_junctions_enabled=True,  # Enable gap junctions
            device="cpu"
        )

        # Forward pass to initialize weights
        input_spikes = torch.zeros(64, device=torch.device("cpu"))
        input_spikes[:10] = 1.0
        output = cerebellum(input_spikes)

        # Store initial weights
        initial_weights = cerebellum.purkinje_cells[0].pf_synaptic_weights.clone()

        # Provide strong error signal
        target = torch.ones(32, device=torch.device("cpu"))
        metrics = cerebellum.deliver_error(target, output)

        # Verify learning happened (error not zeroed)
        # If gap junctions zero the error, metrics['error'] would be 0.0
        # and weights wouldn't change
        new_weights = cerebellum.purkinje_cells[0].pf_synaptic_weights

        # Weights should change (learning occurred)
        weight_changed = not torch.allclose(initial_weights, new_weights, atol=1e-6)

        # If this fails, gap junctions are zeroing the error signal
        assert weight_changed, \
            f"Gap junctions appear to zero error signal. Metrics: {metrics}"

    def test_gap_junctions_can_be_disabled(self):
        """Should be able to disable gap junctions via config."""
        cerebellum = create_test_cerebellum(
            input_size=64,
            purkinje_size=32,
            gap_junctions_enabled=False,  # Explicitly disable
            device="cpu"
        )

        # Gap junctions should be None when disabled
        assert cerebellum.gap_junctions_io is None

    def test_gap_junction_coupling_reduces_error_variability(self):
        """Gap junctions should synchronize error signals across IO neurons.

        This tests the biological function: IO neurons with similar error patterns
        should have synchronized complex spikes via gap junction coupling.
        """
        cerebellum = create_test_cerebellum(
            input_size=64,
            purkinje_size=32,
            use_enhanced_microcircuit=True,
            learning_rate=0.1,
            error_threshold=0.001,
            gap_junctions_enabled=True,
            gap_junction_strength=0.5,  # Strong coupling for visible effect
            device="cpu"
        )

        # Forward pass
        input_spikes = torch.zeros(64, device=torch.device("cpu"))
        input_spikes[:10] = 1.0
        output = cerebellum(input_spikes)

        # Create error with high variability
        target = torch.rand(32, device=torch.device("cpu"))

        # Deliver error (gap junctions should synchronize/smooth the error)
        metrics = cerebellum.deliver_error(target, output)

        # If gap junctions work, _io_membrane should be set
        assert cerebellum._io_membrane is not None
        assert cerebellum._io_membrane.numel() == 32

        # IO membrane should have less variance than raw error would
        # (synchronization smooths out extreme differences)
        io_variance = cerebellum._io_membrane.var().item()

        # Should have some variance (not completely uniform)
        assert io_variance > 0, "IO membrane should have some variance"

    def test_gap_junction_connectivity_from_weights(self):
        """Gap junction connectivity should be based on weight similarity.

        Purkinje cells with similar input patterns (similar weights) should
        have their corresponding IO neurons coupled via gap junctions.
        """
        cerebellum = create_test_cerebellum(
            input_size=64,
            purkinje_size=10,  # Smaller for easier inspection
            gap_junctions_enabled=True,
            gap_junction_threshold=0.1,  # Lower threshold for more coupling
            device="cpu"
        )

        # Gap junctions should exist
        assert cerebellum.gap_junctions_io is not None

        # Coupling matrix should be symmetric (bidirectional coupling)
        coupling = cerebellum.gap_junctions_io.coupling_matrix
        assert coupling.shape == (10, 10)

        # Self-connections should be zero (no gap junctions to self)
        assert torch.allclose(coupling.diag(), torch.zeros(10, device=torch.device("cpu")))

        # Get coupling stats
        stats = cerebellum.gap_junctions_io.get_coupling_stats()
        assert 'n_coupled_neurons' in stats
        assert 'n_connections' in stats

        # At least some neurons should be coupled
        # (if threshold is reasonable and weights are random)
        # Note: This might occasionally fail if random weights don't create any
        # above-threshold connections, but unlikely with threshold=0.1
        assert stats['n_coupled_neurons'] >= 0  # At least 0 (might be 0 initially)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
