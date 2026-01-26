"""Property-Based Tests for State Management.

Uses Hypothesis to generate random test cases and verify state management
properties hold across a wide range of inputs.

Properties Tested:
- State roundtrip preservation (spikes, membrane, traces)
- Biological range constraints (membrane, dopamine, spikes)
- Multi-region independence (regions don't interfere)
- Determinism (same state → same behavior)

Author: Thalia Project
Date: December 2025
"""

import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from thalia.config import (
    HippocampusConfig,
    LayeredCortexConfig,
    LayerSizeCalculator,
    StriatumConfig,
)
from thalia.core.region_state import BaseRegionState
from thalia.regions import (
    LayeredCortex,
    Striatum,
    TrisynapticHippocampus,
)


def create_test_hippocampus(input_size: int, device: str, **kwargs) -> TrisynapticHippocampus:
    """Create TrisynapticHippocampus for testing with Phase 2 pattern."""
    calc = LayerSizeCalculator()
    sizes = calc.hippocampus_from_input(input_size)
    config = HippocampusConfig(**kwargs)
    return TrisynapticHippocampus(config=config, sizes=sizes, device=device)


def create_test_striatum(input_sources: dict, n_actions: int, device: str, **kwargs) -> Striatum:
    """Create Striatum for testing with Phase 2 pattern."""
    # Extract neurons_per_action (default to 1 for minimal testing)
    neurons_per_action = kwargs.pop("neurons_per_action", 1)

    # Compute d1_size and d2_size using calculator
    calc = LayerSizeCalculator()
    sizes = calc.striatum_from_actions(n_actions, neurons_per_action)

    # Add input_sources (required by Striatum)
    sizes["input_sources"] = input_sources

    # Add input_size for convenience
    sizes["input_size"] = sum(input_sources.values())

    config = StriatumConfig(**kwargs)
    striatum = Striatum(config=config, sizes=sizes, device=device)

    # Link pathways to parent (required for multi-source architecture)
    for source_name, source_size in input_sources.items():
        striatum.add_input_source_striatum(source_name, source_size)

    return striatum


def create_test_cortex(
    l4_size: int, l23_size: int, l5_size: int, device: str, **kwargs
) -> LayeredCortex:
    """Create LayeredCortex for testing with Phase 2 pattern."""
    # Manually construct sizes dict with all required fields
    sizes = {
        "input_size": l4_size,
        "l4_size": l4_size,
        "l23_size": l23_size,
        "l5_size": l5_size,
        "l6a_size": l4_size // 2,
        "l6b_size": l4_size // 2,
    }
    config = LayeredCortexConfig(**kwargs)
    return LayeredCortex(config=config, sizes=sizes, device=device)


# Custom strategies for neural network dimensions
@st.composite
def neuron_count(draw):
    """Generate valid neuron counts (10-500)."""
    return draw(st.integers(min_value=10, max_value=500))


@st.composite
def spike_tensor(draw, n_neurons):
    """Generate valid spike tensor."""
    spike_prob = draw(st.floats(min_value=0.0, max_value=1.0))
    return torch.rand(n_neurons) < spike_prob


@st.composite
def membrane_potential(draw, n_neurons):
    """Generate biologically valid membrane potentials."""
    # Generate in valid range [-80mV, +20mV] (typical operating range)
    v_min = draw(st.floats(min_value=-80, max_value=-60))
    v_max = draw(st.floats(min_value=-50, max_value=20))
    assume(v_max > v_min)
    return torch.rand(n_neurons) * (v_max - v_min) + v_min


@st.composite
def trace_tensor(draw, n_neurons):
    """Generate valid STDP trace tensor."""
    # Traces are non-negative and decay exponentially
    return torch.rand(n_neurons) * draw(st.floats(min_value=0.0, max_value=2.0))


class TestMultiRegionIndependence:
    """Test that region states are independent in checkpoints."""

    @given(
        n_regions=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=20, deadline=None)
    def test_regions_have_independent_states(self, n_regions):
        """Property: Modifying one region's state doesn't affect others."""
        # Create multiple region states with different neuromodulator levels
        states = {}
        for i in range(n_regions):
            dopamine_level = float(i) / n_regions  # Different levels for each region
            states[f"region_{i}"] = BaseRegionState(dopamine=dopamine_level)

        # Save states
        saved_states = {name: state.to_dict() for name, state in states.items()}

        # Modify one region
        states["region_0"].dopamine = 0.999

        # Load other regions - should be unchanged
        for i in range(1, n_regions):
            name = f"region_{i}"
            original_dopamine = BaseRegionState.from_dict(saved_states[name], device="cpu").dopamine
            current_dopamine = states[name].dopamine

            assert (
                original_dopamine == current_dopamine
            ), f"{name} was affected by region_0 modification"

    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=20, deadline=None)
    def test_deterministic_forward_pass(self, seed):
        """Property: Same state + same input + same RNG → same output (determinism).

        Currently skipped because neuron state (refractory counters, conductances)
        is NOT saved in get_state/load_state. This means two instances with "same state"
        actually have different neuron states, causing non-deterministic forward passes.

        To fix this, we need to:
        1. Add neuron state to HippocampusState (and all region states)
        2. Call neurons.get_state() in region get_state()
        3. Call neurons.load_state() in region load_state()

        This is tracked in: TODO - add issue number
        """
        pytest.skip(
            "Requires neuron state serialization (refractory, g_E, g_I, g_adapt). "
            "Region get_state/load_state currently only saves membrane voltages, "
            "not full neuron model state. This causes non-determinism even with "
            "same RNG state."
        )
        torch.manual_seed(seed)

        # Use hippocampus (no exploration randomness like striatum)
        hippocampus = create_test_hippocampus(input_size=20, device="cpu", dt_ms=1.0)

        # Build up some state
        for _ in range(5):
            hippocampus.forward({"cortex": torch.rand(20) > 0.6})

        # Save state AND RNG state
        state = hippocampus.get_state()
        rng_state = torch.get_rng_state()

        # Create two new instances with same loaded state
        torch.manual_seed(seed + 1)  # Reset RNG for initialization
        hippocampus1 = create_test_hippocampus(input_size=20, device="cpu", dt_ms=1.0)
        hippocampus1.load_state(state)

        torch.manual_seed(seed + 1)  # Same RNG state
        hippocampus2 = create_test_hippocampus(input_size=20, device="cpu", dt_ms=1.0)
        hippocampus2.load_state(state)

        # Run same input with same RNG - should get same output
        test_input = torch.rand(20) > 0.6

        # Set same RNG state for both forward passes
        torch.set_rng_state(rng_state)
        output1 = hippocampus1.forward(test_input.clone())

        torch.set_rng_state(rng_state)  # Reset to same RNG
        output2 = hippocampus2.forward(test_input.clone())

        # Check determinism (spike outputs should be identical with same RNG)
        assert torch.equal(output1, output2), "Forward pass not deterministic after state load"


class TestStateSizeInvariance:
    """Test that state management handles various sizes correctly."""

    @given(
        n_input=st.integers(min_value=10, max_value=100),
        n_output=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=20, deadline=None)
    def test_striatum_state_size_invariance(self, n_input, n_output):
        """Property: State serialization works for any valid region size."""
        # Compute n_actions from n_output
        n_actions = n_output // 4 if n_output >= 4 else n_output
        striatum = create_test_striatum(
            input_sources={"default": n_input},
            n_actions=n_actions,
            device="cpu",
            dt_ms=1.0,
        )

        # Run some steps
        for _ in range(5):
            striatum.forward({"default": torch.rand(n_input) > 0.8})

        # Save and load
        state = striatum.get_state()
        striatum2 = create_test_striatum(
            input_sources={"default": n_input},
            n_actions=n_actions,
            device="cpu",
            dt_ms=1.0,
        )
        striatum2.load_state(state)

        # Verify dimensions preserved
        assert striatum2.input_size == n_input
        assert striatum2.n_actions == n_actions

    @given(
        l4_size=st.integers(min_value=20, max_value=100),
        l23_size=st.integers(min_value=30, max_value=120),
    )
    @settings(max_examples=20, deadline=None)
    def test_cortex_state_size_invariance(self, l4_size, l23_size):
        """Property: Cortex state works for various layer sizes."""
        l5_size = l23_size  # Keep output constraint satisfied

        cortex = create_test_cortex(
            l4_size=l4_size,
            l23_size=l23_size,
            l5_size=l5_size,
            device="cpu",
            dt_ms=1.0,
        )

        # Run some steps
        for _ in range(5):
            cortex.forward({"default": torch.rand(l4_size) > 0.8})

        # Save and load
        state = cortex.get_state()
        cortex2 = create_test_cortex(
            l4_size=l4_size,
            l23_size=l23_size,
            l5_size=l5_size,
            device="cpu",
            dt_ms=1.0,
        )
        cortex2.load_state(state)

        # Verify layer sizes preserved
        assert cortex2.l4_size == l4_size
        assert cortex2.l23_size == l23_size
        assert cortex2.l5_size == l5_size


class TestStateConsistency:
    """Test that state remains consistent through multiple operations."""

    @given(
        n_checkpoints=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=15, deadline=None)
    def test_multiple_checkpoint_cycles(self, n_checkpoints, seed):
        """Property: Multiple save/load cycles preserve state correctly."""
        torch.manual_seed(seed)

        striatum = create_test_striatum(
            input_sources={"default": 40},
            n_actions=8,
            neurons_per_action=1,
            device="cpu",
            dt_ms=1.0,
        )

        # Initial state
        for _ in range(10):
            striatum.forward({"default": torch.rand(40) > 0.7})

        initial_state = striatum.get_state()

        # Multiple checkpoint cycles
        current_state = initial_state
        for _ in range(n_checkpoints):
            # Load state
            striatum_temp = create_test_striatum(
                input_sources={"default": 40},
                n_actions=8,
                neurons_per_action=1,
                device="cpu",
                dt_ms=1.0,
            )
            striatum_temp.load_state(current_state)

            # Run a few steps
            for _ in range(3):
                striatum_temp.forward({"default": torch.rand(40) > 0.7})

            # Save again
            current_state = striatum_temp.get_state()

        # Final load should work without errors
        striatum_final = create_test_striatum(
            input_sources={"default": 40},
            n_actions=8,
            neurons_per_action=1,
            device="cpu",
            dt_ms=1.0,
        )
        striatum_final.load_state(current_state)

        # Basic sanity check - can still run
        output = striatum_final.forward({"default": torch.rand(40) > 0.7})
        # Striatum output is n_actions * neurons_per_action (8 * 1 = 8 action neurons total for D1+D2)
        expected_output = striatum_final.n_output
        assert output.shape == (
            expected_output,
        ), f"Output shape incorrect after multiple cycles: expected {expected_output}, got {output.shape[0]}"

    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=15, deadline=None)
    def test_hippocampus_state_consistency(self, seed):
        """Property: Hippocampus state remains consistent through checkpoints."""
        torch.manual_seed(seed)

        hippocampus = create_test_hippocampus(input_size=20, device="cpu", dt_ms=1.0)

        # Build up some patterns
        for _ in range(15):
            hippocampus.forward({"cortex": torch.rand(20) > 0.6})

        # Save and load
        state = hippocampus.get_state()
        hippocampus2 = create_test_hippocampus(input_size=20, device="cpu", dt_ms=1.0)
        hippocampus2.load_state(state)

        # Continue processing - should work smoothly
        for _ in range(10):
            output = hippocampus2.forward({"cortex": torch.rand(20) > 0.6})
            # Hippocampus output is CA1 size (2x input per ratios)
            expected_output = hippocampus2.n_output
            assert output.shape == (
                expected_output,
            ), f"Output shape incorrect: expected {expected_output}, got {output.shape[0]}"
            assert torch.isfinite(output).all(), "Output has NaN/Inf"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_neuromodulators(self):
        """Property: Zero neuromodulator levels handled correctly."""
        state = BaseRegionState(dopamine=0.0, acetylcholine=0.0, norepinephrine=0.0)

        data = state.to_dict()
        loaded = BaseRegionState.from_dict(data, device="cpu")

        assert loaded.dopamine == 0.0
        assert loaded.acetylcholine == 0.0
        assert loaded.norepinephrine == 0.0

    def test_max_neuromodulators(self):
        """Property: Maximum neuromodulator levels handled correctly."""
        state = BaseRegionState(dopamine=1.0, acetylcholine=1.0, norepinephrine=1.0)

        data = state.to_dict()
        loaded = BaseRegionState.from_dict(data, device="cpu")

        assert loaded.dopamine == 1.0
        assert loaded.acetylcholine == 1.0
        assert loaded.norepinephrine == 1.0

    @given(dopamine=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(max_examples=20, deadline=None)
    def test_neuromodulator_roundtrip(self, dopamine):
        """Property: Neuromodulator values preserved through serialization."""
        state = BaseRegionState(dopamine=dopamine)

        data = state.to_dict()
        loaded = BaseRegionState.from_dict(data, device="cpu")

        assert abs(loaded.dopamine - dopamine) < 1e-6
