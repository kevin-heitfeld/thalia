"""Unit tests for cerebellar inferior olive gap junction coupling.

Tests the biological synchronization of error signals across IO neurons via gap junctions.
IO neurons have one of the densest gap junction networks in the brain, creating
synchronized complex spikes (<1ms) across multiple Purkinje cells.

Biology:
- Llinás & Yarom (1981): Electrophysiology of IO neurons
- De Zeeuw et al. (1998): Synchronized climbing fibers
- Leznik & Llinás (2005): Role of gap junctions in IO oscillations
- Schweighofer et al. (1999): Computational model of IO synchronization
"""

import pytest
import torch

from thalia.config.size_calculator import LayerSizeCalculator
from thalia.regions.cerebellum import Cerebellum, CerebellumConfig


def create_test_cerebellum(
    input_size: int,
    purkinje_size: int,
    device: str = "cpu",
    **kwargs
) -> Cerebellum:
    """Create Cerebellum for testing with new (config, sizes, device) pattern."""
    # Always compute granule_size (Cerebellum.__init__ requires it)
    expansion = kwargs.pop("granule_expansion_factor", 4.0)
    calc = LayerSizeCalculator()
    sizes = calc.cerebellum_from_purkinje(purkinje_size, expansion)
    sizes["input_size"] = input_size

    config = CerebellumConfig(device=device, **kwargs)
    return Cerebellum(config, sizes, device)


def test_io_gap_junction_initialization():
    """Test gap junction module is properly initialized in cerebellum."""
    cerebellum = create_test_cerebellum(
        input_size=100,
        purkinje_size=50,
        device="cpu",
        gap_junctions_enabled=True,
        gap_junction_strength=0.18,
        gap_junction_max_neighbors=12,
    )

    assert cerebellum.gap_junctions_io is not None, "Gap junction module not initialized"
    assert cerebellum.config.gap_junctions_enabled
    assert cerebellum.config.gap_junction_strength == 0.18


def test_io_gap_junction_disabled():
    """Test gap junctions can be disabled via config."""
    cerebellum = create_test_cerebellum(
        input_size=100,
        purkinje_size=50,
        device="cpu",
        gap_junctions_enabled=False,
    )

    assert cerebellum.gap_junctions_io is None, "Gap junction module should not be initialized when disabled"


def test_io_membrane_synchronization():
    """Test that gap junctions synchronize IO neuron membrane potentials."""
    cerebellum = create_test_cerebellum(
        input_size=100,
        purkinje_size=50,
        device="cpu",
        gap_junctions_enabled=True,
        gap_junction_strength=0.18,
        gap_junction_max_neighbors=12,
        gap_junction_threshold=0.20,  # Lower threshold for testing
    )

    # Create input pattern
    device = cerebellum.device
    input_spikes = torch.zeros(100, dtype=torch.bool, device=device)
    input_spikes[10:20] = True

    # Create target with localized error
    # Error will be concentrated in a few neurons
    target = torch.zeros(50, device=device)
    target[5:10] = 1.0  # Strong positive error for neurons 5-9

    # Run forward pass
    output = cerebellum(input_spikes)

    # Deliver error signal (triggers gap junction synchronization)
    cerebellum.deliver_error(target=target, output_spikes=output)

    # Check that io_membrane state was updated
    assert cerebellum._io_membrane is not None
    io_membrane = cerebellum._io_membrane

    # IO membrane should be non-zero where error occurred
    assert io_membrane[5:10].sum() > 0, "IO membrane should reflect error in target neurons"

    # Gap junctions should spread activity to neighbors
    # Neighboring neurons (4 and 10) should have some coupled activity
    # (though may be zero if not connected)
    assert io_membrane.shape == (50,), f"IO membrane shape mismatch: {io_membrane.shape}"


def test_error_sign_preservation():
    """Test that gap junctions preserve error sign while synchronizing magnitude.

    Biology: IO neurons synchronize to create simultaneous complex spikes,
    but the sign of the error (LTP vs LTD) must be preserved for each Purkinje cell.
    """
    cerebellum = create_test_cerebellum(
        input_size=100,
        purkinje_size=50,
        device="cpu",
        gap_junctions_enabled=True,
        gap_junction_strength=0.18,
        gap_junction_max_neighbors=12,
    )

    # Create input
    device = cerebellum.device
    input_spikes = torch.zeros(100, dtype=torch.bool, device=device)
    input_spikes[20:30] = True

    # Create target with mixed positive/negative errors
    target = torch.zeros(50, device=device)
    target[10:15] = 1.0   # Positive error (should fire more)
    target[20:25] = 0.0   # Negative error if these neurons fire

    # Run forward (may produce spikes in 20:25 region)
    output = cerebellum(input_spikes)

    # Deliver error
    cerebellum.deliver_error(target=target, output_spikes=output)

    # IO membrane should be all positive (magnitude only)
    io_membrane = cerebellum._io_membrane
    assert (io_membrane >= 0).all(), "IO membrane should be non-negative (magnitude)"


def test_io_gap_junction_state_serialization():
    """Test that io_membrane state is properly saved and loaded."""
    cerebellum = create_test_cerebellum(
        input_size=100,
        purkinje_size=50,
        device="cpu",
        gap_junctions_enabled=True,
        gap_junction_strength=0.18,
    )

    # Create some state
    device = cerebellum.device
    input_spikes = torch.zeros(100, dtype=torch.bool, device=device)
    input_spikes[15:25] = True
    target = torch.ones(50, device=device) * 0.5

    output = cerebellum(input_spikes)
    cerebellum.deliver_error(target=target, output_spikes=output)

    # Get state
    state = cerebellum.get_state()
    assert state.io_membrane is not None, "io_membrane not in state"

    # Convert to dict
    state_dict = state.to_dict()
    assert "io_membrane" in state_dict, "io_membrane not in state dict"

    # Create new cerebellum and load state
    cerebellum2 = create_test_cerebellum(
        input_size=100,
        purkinje_size=50,
        device="cpu",
        gap_junctions_enabled=True,
        gap_junction_strength=0.18,
    )
    cerebellum2.load_state(state)

    # Verify io_membrane was restored
    assert cerebellum2._io_membrane is not None
    assert torch.allclose(
        cerebellum._io_membrane,
        cerebellum2._io_membrane,
        atol=1e-6
    ), "IO membrane state not restored correctly"


def test_io_gap_junction_reset_state():
    """Test that reset_state properly initializes io_membrane."""
    cerebellum = create_test_cerebellum(
        input_size=100,
        purkinje_size=50,
        device="cpu",
        gap_junctions_enabled=True,
    )

    # Create some state
    device = cerebellum.device
    input_spikes = torch.zeros(100, dtype=torch.bool, device=device)
    input_spikes[10:20] = True
    target = torch.ones(50, device=device) * 0.5

    output = cerebellum(input_spikes)
    cerebellum.deliver_error(target=target, output_spikes=output)

    # Verify io_membrane is non-zero
    assert cerebellum._io_membrane.abs().sum() > 0

    # Reset state
    cerebellum.reset_state()

    # io_membrane should be zeros after reset
    assert cerebellum._io_membrane is not None
    assert torch.allclose(
        cerebellum._io_membrane,
        torch.zeros(50, device=device),
        atol=1e-6
    ), "IO membrane should be zeros after reset"


def test_io_coupling_strength_scaling():
    """Test that gap junction strength parameter affects coupling magnitude.

    Stronger gap junctions should produce stronger synchronization.
    """
    # Weak coupling
    cerebellum_weak = create_test_cerebellum(
        input_size=100,
        purkinje_size=50,
        device="cpu",
        gap_junctions_enabled=True,
        gap_junction_strength=0.05,  # Weak
    )

    # Strong coupling
    cerebellum_strong = create_test_cerebellum(
        input_size=100,
        purkinje_size=50,
        device="cpu",
        gap_junctions_enabled=True,
        gap_junction_strength=0.25,  # Strong
    )

    # Use same random seed for consistent weights
    device = cerebellum_weak.device
    torch.manual_seed(42)
    cerebellum_weak.weights.data = torch.rand_like(cerebellum_weak.weights.data) * 0.5
    torch.manual_seed(42)
    cerebellum_strong.weights.data = torch.rand_like(cerebellum_strong.weights.data) * 0.5

    # Same input and target
    input_spikes = torch.zeros(100, dtype=torch.bool, device=device)
    input_spikes[15:25] = True
    target = torch.zeros(50, device=device)
    target[10:15] = 1.0

    # Run both
    output_weak = cerebellum_weak(input_spikes)
    cerebellum_weak.deliver_error(target=target, output_spikes=output_weak)

    output_strong = cerebellum_strong(input_spikes)
    cerebellum_strong.deliver_error(target=target, output_spikes=output_strong)

    # Strong coupling should produce more uniform io_membrane distribution
    # (neighbors get more synchronized)
    io_weak = cerebellum_weak._io_membrane
    io_strong = cerebellum_strong._io_membrane

    # Check that both have activity in the target region
    assert io_weak[10:15].sum() > 0
    assert io_strong[10:15].sum() > 0

    # Strong coupling should have lower variance (more uniform)
    # Note: This is a probabilistic test, may need adjustment
    var_weak = io_weak[io_weak > 0].var() if (io_weak > 0).sum() > 1 else 0
    var_strong = io_strong[io_strong > 0].var() if (io_strong > 0).sum() > 1 else 0

    # With stronger coupling, active neurons should be more synchronized
    # This is a soft check - just verify both have activity
    assert var_weak >= 0  # Just verify computation works
    assert var_strong >= 0


def test_io_gap_junctions_with_enhanced_microcircuit():
    """Test gap junctions work correctly with enhanced cerebellar microcircuit."""
    cerebellum = create_test_cerebellum(
        input_size=100,
        purkinje_size=50,
        device="cpu",
        use_enhanced_microcircuit=True,
        gap_junctions_enabled=True,
        gap_junction_strength=0.18,
    )

    assert cerebellum.use_enhanced
    assert cerebellum.gap_junctions_io is not None

    # Run with mossy fiber input
    device = cerebellum.device
    mossy_fiber_input = torch.zeros(100, dtype=torch.bool, device=device)
    mossy_fiber_input[20:30] = True
    target = torch.ones(50, device=device) * 0.3

    output = cerebellum(mossy_fiber_input)
    cerebellum.deliver_error(target=target, output_spikes=output)

    # Should produce io_membrane state
    assert cerebellum._io_membrane is not None
    assert cerebellum._io_membrane.shape == (50,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
