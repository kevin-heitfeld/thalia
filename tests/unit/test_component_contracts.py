"""
Universal contract tests for neural components.

Tests common contracts that all components must satisfy:
- Reset clears state
- Forward pass produces valid output
- State management works correctly

This consolidates redundant tests across the test suite into
parameterized contract tests that validate universal properties.

Note: Growth and detailed component tests remain in component-specific test files
since growth APIs and behavior vary significantly between components.
"""

import pytest
import torch

from thalia.regions.thalamus import ThalamicRelay, ThalamicRelayConfig
from thalia.config.size_calculator import LayerSizeCalculator


# Component factories - only using simple components that share contracts
def create_thalamus():
    """Create thalamic relay with minimal config."""
    config = ThalamicRelayConfig()
    calc = LayerSizeCalculator()
    sizes = calc.thalamus_from_relay(relay_size=80)
    sizes['input_size'] = 100  # Override with test-specific input size
    return ThalamicRelay(config=config, sizes=sizes, device="cpu")


# Test data creators
def create_spike_input(size: int) -> torch.Tensor:
    """Create sparse binary spike input."""
    return (torch.rand(size) > 0.8).float()


@pytest.mark.parametrize(
    "component_factory,input_size",
    [
        (create_thalamus, 100),
    ],
    ids=["thalamus"],
)
def test_component_reset_contract(component_factory, input_size):
    """Test that components properly reset state.

    Universal contract:
    - After reset, component should have clean state
    - Membrane potentials should be near rest
    - Can run forward without issues after reset
    """
    component = component_factory()

    # Run forward to dirty state
    test_input = create_spike_input(input_size)
    _ = component(test_input)

    # Reset
    component.reset_state()

    # Validate clean state (check neurons if present)
    if hasattr(component, "neurons") and component.neurons is not None:
        if hasattr(component.neurons, "membrane") and component.neurons.membrane is not None:
            membrane = component.neurons.membrane
            assert not torch.isnan(membrane).any(), "Membrane contains NaN after reset"
            assert not torch.isinf(membrane).any(), "Membrane contains Inf after reset"
            # Membrane should be close to rest (typically 0.0)
            assert torch.abs(membrane).max() < 0.5, "Membrane not near rest after reset"

    # Can run forward again without issues
    output = component(test_input)
    assert output is not None, "Forward failed after reset"
    assert not torch.isnan(output.float()).any(), "Output contains NaN"


@pytest.mark.parametrize(
    "component_factory,input_size,expected_output_size",
    [
        (create_thalamus, 100, 80),
    ],
    ids=["thalamus"],
)
def test_component_forward_contract(component_factory, input_size, expected_output_size):
    """Test forward pass produces valid output.

    Universal contract:
    - Output shape matches config
    - Output is valid tensor (no NaN/Inf)
    - Output dtype is spike-compatible (bool, uint8, or float)
    """
    component = component_factory()
    input_spikes = create_spike_input(input_size)

    output = component(input_spikes)

    # Shape contract
    assert output.shape[0] == expected_output_size, (
        f"Output shape should be ({expected_output_size},), got {output.shape}"
    )

    # Dtype contract
    assert output.dtype in [torch.bool, torch.uint8, torch.float32], (
        f"Output dtype should be spike-compatible, got {output.dtype}"
    )

    # Validity contract
    assert not torch.isnan(output.float()).any(), "Output contains NaN"
    assert not torch.isinf(output.float()).any(), "Output contains Inf"


@pytest.mark.parametrize(
    "component_factory",
    [
        create_thalamus,
    ],
    ids=["thalamus"],
)
def test_component_state_dict_contract(component_factory):
    """Test state_dict/load_state_dict contract.

    Universal contract:
    - state_dict() returns dict with tensors
    - load_state_dict() restores exact state
    - State persists through save/load cycle
    """
    component = component_factory()

    # Get state
    state = component.state_dict()

    assert isinstance(state, dict), "state_dict() should return dict"
    assert len(state) > 0, "state_dict() should not be empty"

    # All values should be tensors
    for key, value in state.items():
        assert isinstance(value, torch.Tensor), f"State[{key}] should be tensor, got {type(value)}"

    # Modify component state (if neurons available)
    if hasattr(component, "neurons") and component.neurons is not None:
        if hasattr(component.neurons, "membrane") and component.neurons.membrane is not None:
            original_membrane = component.neurons.membrane.clone()
            component.neurons.membrane.fill_(0.5)  # Dirty state

            # Load original state
            component.load_state_dict(state)

            # State should be restored
            assert torch.allclose(component.neurons.membrane, original_membrane, atol=1e-6), (
                "Membrane state not restored correctly"
            )


@pytest.mark.parametrize(
    "component_factory,input_size",
    [
        (create_thalamus, 100),
    ],
    ids=["thalamus"],
)
def test_component_handles_empty_dict_input(component_factory, input_size):
    """Test components handle empty dict gracefully (Phase 2 improvement).

    Some components accept Dict[str, Tensor] for multi-source inputs.
    They should handle empty dict without crashing.

    Note: Thalamus uses single tensor input, so this test validates
    that components maintain stability even with edge case inputs.
    """
    component = component_factory()

    # For single-input components, test with zero input instead
    zero_input = torch.zeros(input_size, dtype=torch.bool)

    # Should not crash with minimal input
    output = component(zero_input)

    # Contract: valid output
    assert output is not None, "Component should handle zero input"
    assert not torch.isnan(output.float()).any(), "Output contains NaN"
    assert not torch.isinf(output.float()).any(), "Output contains Inf"


# NOTE: Growth tests are component-specific and remain in individual test files
# because:
# 1. Growth APIs differ significantly (grow_output vs grow_layer vs grow_source)
# 2. Growth behavior varies (some grow neurons, others grow weights differently)
# 3. Validation requirements are component-specific
#
# See:
# - tests/unit/test_thalamus.py for thalamus growth
# - tests/unit/test_striatum_d1d2_delays.py for striatum growth
# - tests/unit/regions/test_cortex_l6ab_split.py for cortex layer growth
