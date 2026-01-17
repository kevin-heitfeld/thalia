"""Tests for Striatum using unified RegionTestBase framework.

Demonstrates unified testing pattern with region-specific tests for
the striatum (D1/D2 opponent pathways for action selection).

Author: Thalia Project
Date: December 22, 2025 (Tier 3.4 implementation)
Updated: December 2025 (Phase 1 migration to (config, sizes, device) pattern)
"""

from typing import Any, Dict

import torch

from tests.utils.region_test_base import RegionTestBase
from thalia.config.size_calculator import LayerSizeCalculator
from thalia.regions.striatum import Striatum
from thalia.regions.striatum.config import StriatumConfig


class TestStriatum(RegionTestBase):
    """Test Striatum implementation using unified test framework."""

    def create_region(self, **kwargs):
        """Create Striatum instance for testing.

        Uses new (config, sizes, device) pattern:
        - Config contains only behavioral parameters
        - Sizes computed via LayerSizeCalculator
        - Device passed explicitly
        """
        # Extract size-related parameters
        n_actions = kwargs.pop("n_actions", 4)
        neurons_per_action = kwargs.pop("neurons_per_action", 10)
        device = kwargs.pop("device", "cpu")

        # Remove input_sources if present (no longer needed for config)
        kwargs.pop("input_sources", None)

        # Compute sizes using calculator
        calc = LayerSizeCalculator()
        sizes = calc.striatum_from_actions(
            n_actions=n_actions, neurons_per_action=neurons_per_action
        )

        # Add input_size to sizes dict (striatum needs this)
        # Use default of 100 for testing (can be overridden)
        sizes["input_size"] = kwargs.pop("input_size", 100)

        # Create config with behavioral parameters only
        config = StriatumConfig(**kwargs)

        # Create striatum with new pattern
        striatum = Striatum(config=config, sizes=sizes, device=device)

        # Add default input source (required for multi-source architecture)
        striatum.add_input_source_striatum("default", sizes["input_size"])

        return striatum

    def get_default_params(self):
        """Return default striatum parameters."""
        return {
            "input_size": 100,  # Total input size
            "n_actions": 5,  # Number of actions
            "neurons_per_action": 4,  # 20 total neurons (5 actions × 4)
            "rpe_enabled": True,
            "use_goal_conditioning": False,  # Disable for simpler testing
            "device": "cpu",
            "dt_ms": 1.0,
        }

    def get_min_params(self):
        """Return minimal valid parameters for quick tests."""
        return {
            "input_size": 20,  # Total input size
            "n_actions": 3,  # 3 actions
            "neurons_per_action": 2,  # 6 total neurons
            "device": "cpu",
            "dt_ms": 1.0,
        }

    def get_input_dict(self, n_input, device="cpu"):
        """Return dict input for striatum (supports multi-source)."""
        return {
            "cortex": torch.zeros(n_input // 2, device=device),
            "thalamus": torch.zeros(n_input // 2, device=device),
        }

    # =========================================================================
    # OVERRIDE BASE TESTS FOR POPULATION CODING
    # =========================================================================

    def test_grow_output(self):
        """Test striatum can grow output dimension (accounts for population coding)."""
        params = self.get_default_params()
        region = self.create_region(**params)

        original_n_actions = region.n_actions
        neurons_per_action = params["neurons_per_action"]
        n_new_actions = 10

        # Grow output (adds actions, which adds neurons_per_action neurons per action)
        region.grow_output(n_new_actions)

        # Verify instance variables updated (n_output is total neurons, not actions)
        expected_total_neurons = (original_n_actions + n_new_actions) * neurons_per_action
        assert region.n_output == expected_total_neurons
        assert region.n_actions == original_n_actions + n_new_actions

        # Verify forward pass still works with new size
        input_size = params["input_size"]
        input_spikes = torch.zeros(input_size, device=region.device)
        output = region.forward({"default": input_spikes})

        # Output is per-neuron (d1_spikes), same as n_output
        assert (
            output.shape[0] == expected_total_neurons
        ), f"Expected {expected_total_neurons} neurons, got {output.shape[0]}"

    def test_grow_input(self):
        """Test striatum can grow input dimension via grow_source (accounts for population coding)."""
        params = self.get_default_params()
        region = self.create_region(**params)

        original_n_input = params["input_size"]
        n_new_input = 20

        # Grow input using grow_source (multi-source architecture)
        region.grow_source("default", original_n_input + n_new_input)

        # Verify forward pass still works with larger input
        input_spikes = torch.zeros(original_n_input + n_new_input, device=region.device)
        output = region.forward({"default": input_spikes})

        # Output size should NOT change (still same number of actions/neurons)
        expected_neurons = region.n_output
        assert (
            output.shape[0] == expected_neurons
        ), f"Expected {expected_neurons} neurons, got {output.shape[0]}"

    def test_growth_preserves_state(self):
        """Test growth preserves existing neuron state (striatum-specific).

        Striatum organizes membrane state as [all_D1, all_D2], not [old, new].
        After growth, old neurons are not contiguous in concatenated state,
        so we need to check D1 and D2 pathways separately.
        """
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward pass to initialize state
        input_size = self._get_input_size(params)
        input_dict = self.get_input_dict(input_size, device=region.device.type)
        region.forward(input_dict)

        # Get D1 and D2 states before growth
        d1_membrane_before = region.d1_pathway.neurons.membrane.clone()
        d2_membrane_before = region.d2_pathway.neurons.membrane.clone()
        n_d1_before = d1_membrane_before.shape[0]
        n_d2_before = d2_membrane_before.shape[0]

        # Grow output
        region.grow_output(10)

        # Get D1 and D2 states after growth
        d1_membrane_after = region.d1_pathway.neurons.membrane
        d2_membrane_after = region.d2_pathway.neurons.membrane

        # Verify original D1 neurons preserved
        assert torch.allclose(
            d1_membrane_before, d1_membrane_after[:n_d1_before], atol=1e-5
        ), "D1 pathway state not preserved after growth"

        # Verify original D2 neurons preserved
        assert torch.allclose(
            d2_membrane_before, d2_membrane_after[:n_d2_before], atol=1e-5
        ), "D2 pathway state not preserved after growth"

    # =========================================================================
    # STRIATUM-SPECIFIC TESTS
    # =========================================================================

    def test_d1_d2_pathways(self):
        """Test striatum has both D1 (Go) and D2 (NoGo) pathways."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Verify D1 and D2 pathways exist
        assert hasattr(region, "d1_pathway")
        assert hasattr(region, "d2_pathway")

        # Forward pass
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        output = region.forward({"default": input_spikes})

        # Verify output shape matches action neurons
        expected_neurons = region.n_output
        assert output.shape[0] == expected_neurons

    def test_initialization(self):
        """Test striatum initializes with n_output expansion for population coding."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # With population coding, n_output gets expanded
        # Original: n_actions = 5 actions
        # Expanded: n_output = (d1_size + d2_size) = 5 * 4 = 20 neurons
        expected_n_output = params["n_actions"] * params["neurons_per_action"]
        assert region.n_output == expected_n_output
        assert region.input_size == self._get_input_size(params)

    def test_initialization_minimal(self):
        """Test striatum initializes with minimal params and expansion."""
        params = self.get_min_params()
        region = self.create_region(**params)

        # Verify dimensions are correct
        expected_n_output = params["n_actions"] * params["neurons_per_action"]
        assert region.n_output == expected_n_output

    def test_forward_pass_tensor_input(self):
        """Test forward returns expanded neuron output (uses dict wrapper)."""
        params = self.get_default_params()
        region = self.create_region(**params)
        input_spikes = torch.zeros(self._get_input_size(params), device=region.device)
        output = region.forward({"default": input_spikes})

        # Output is total neuron count (D1 + D2)
        expected_n_output = region.n_output
        assert output.shape[0] == expected_n_output

    def test_forward_pass_dict_input(self):
        """Test forward with dict input returns expanded output."""
        params = self.get_default_params()
        region = self.create_region(**params)
        # Note: Multi-source striatum needs sources added first
        region.add_input_source_striatum("cortex", self._get_input_size(params) // 2)
        region.add_input_source_striatum("thalamus", self._get_input_size(params) // 2)

        input_dict = self.get_input_dict(self._get_input_size(params), device=region.device.type)
        output = region.forward(input_dict)

        expected_n_output = region.n_output
        assert output.shape[0] == expected_n_output

    def test_forward_pass_zero_input(self):
        """Test zero input handling with expansion."""
        params = self.get_default_params()
        region = self.create_region(**params)
        input_spikes = torch.zeros(self._get_input_size(params), device=region.device)
        output = region.forward({"default": input_spikes})

        expected_n_output = region.n_output
        assert output.shape[0] == expected_n_output

    def test_forward_pass_multiple_calls(self):
        """Test multiple forwards with expansion."""
        params = self.get_default_params()
        region = self.create_region(**params)
        input_spikes = torch.zeros(self._get_input_size(params), device=region.device)
        expected_n_output = region.n_output

        for _ in range(10):
            output = region.forward({"default": input_spikes})
            assert output.shape[0] == expected_n_output

    def test_population_coding(self):
        """Test population coding for action representation."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward pass
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        region.forward({"default": input_spikes})

        # Get state
        state = region.get_state()

        # Verify vote accumulation for population coding
        # Votes are per-action (not per-neuron)
        if hasattr(state, "d1_votes_accumulated"):
            assert state.d1_votes_accumulated is not None
            assert state.d1_votes_accumulated.shape[0] == region.n_actions  # n_actions

        if hasattr(state, "d2_votes_accumulated"):
            assert state.d2_votes_accumulated is not None
            assert state.d2_votes_accumulated.shape[0] == region.n_actions  # n_actions

    def test_action_selection(self):
        """Test striatum selects actions based on vote competition."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Multiple forward passes to accumulate votes
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        for _ in range(10):
            region.forward({"default": input_spikes})

        # Check if action was selected
        state = region.get_state()
        if hasattr(state, "last_action"):
            # Last action should be valid (0 to n_actions-1)
            if state.last_action is not None:
                assert 0 <= state.last_action < region.n_actions

    def test_dopamine_modulation(self):
        """Test dopamine modulates learning in D1/D2 pathways."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Set dopamine via neuromodulators
        if hasattr(region, "set_neuromodulators"):
            region.set_neuromodulators(dopamine=0.8)

        # Forward pass
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        region.forward({"default": input_spikes})

        # Verify dopamine stored in state
        state = region.get_state()
        if hasattr(state, "dopamine"):
            assert state.dopamine == 0.8

    def test_rpe_computation(self):
        """Test reward prediction error (RPE) computation."""
        params = self.get_default_params()
        params["rpe_enabled"] = True
        region = self.create_region(**params)

        # Forward passes
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        for _ in range(5):
            region.forward({"default": input_spikes})

        # Check RPE state
        state = region.get_state()
        if hasattr(state, "last_rpe"):
            # RPE should be a scalar value
            if state.last_rpe is not None:
                assert isinstance(state.last_rpe, (int, float))

        if hasattr(state, "value_estimates"):
            # Should have value estimates per action
            if state.value_estimates is not None:
                assert state.value_estimates.shape[0] == region.n_actions

    def test_exploration_uncertainty(self):
        """Test exploration based on uncertainty."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward passes
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        for _ in range(10):
            region.forward({"default": input_spikes})

        # Check exploration state
        state = region.get_state()
        if hasattr(state, "exploring"):
            assert isinstance(state.exploring, bool)

        if hasattr(state, "last_uncertainty"):
            if state.last_uncertainty is not None:
                assert 0.0 <= state.last_uncertainty <= 1.0

    def test_eligibility_traces(self):
        """Test three-factor learning with eligibility traces."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward passes to build eligibility traces
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        for _ in range(20):
            region.forward({"default": input_spikes})

        # Verify pathways have eligibility traces
        if hasattr(region.d1_pathway, "eligibility_traces"):
            traces = region.d1_pathway.eligibility_traces
            if traces is not None:
                # Should have same shape as weights
                assert traces.shape == region.d1_pathway.synaptic_weights["default"].shape

    def test_pathway_delays(self):
        """Test D1/D2 pathways have different conduction delays."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Check delay buffers
        state = region.get_state()
        if hasattr(state, "d1_delay_buffer") and hasattr(state, "d2_delay_buffer"):
            # D1 and D2 should have different delay buffer sizes
            # (D2 pathway typically has longer delays)
            if state.d1_delay_buffer is not None and state.d2_delay_buffer is not None:
                d1_delay = state.d1_delay_buffer.shape[0]
                d2_delay = state.d2_delay_buffer.shape[0]
                # D2 delay should be longer than D1
                assert d2_delay >= d1_delay

    def test_homeostatic_regulation(self):
        """Test homeostatic regulation maintains firing rates."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Many forward passes to trigger homeostasis
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        for _ in range(100):
            region.forward({"default": input_spikes})

        # Check homeostatic state
        state = region.get_state()
        if hasattr(state, "activity_ema"):
            assert isinstance(state.activity_ema, float)
            # EMA should be between 0 and 1
            assert 0.0 <= state.activity_ema <= 1.0

        if hasattr(state, "homeostatic_scaling_applied"):
            assert isinstance(state.homeostatic_scaling_applied, bool)

    def test_goal_conditioning(self):
        """Test goal conditioning from PFC (when enabled)."""
        params = self.get_default_params()
        params["use_goal_conditioning"] = True
        params["pfc_size"] = 16
        region = self.create_region(**params)

        # Forward pass
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        region.forward({"default": input_spikes})

        # Check goal modulation weights
        state = region.get_state()
        if hasattr(state, "pfc_modulation_d1") and hasattr(state, "pfc_modulation_d2"):
            if state.pfc_modulation_d1 is not None:
                # Should have modulation weights for D1 neurons × PFC size
                # D1 has d1_size neurons
                expected_shape = (region.d1_size, params["pfc_size"])
                assert state.pfc_modulation_d1.shape == expected_shape


# Standard tests (initialization, forward, growth, state, device, neuromodulators, diagnostics)
# inherited from RegionTestBase - eliminates ~100 lines of boilerplate
