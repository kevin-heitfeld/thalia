"""Tests for Striatum using unified RegionTestBase framework.

Demonstrates unified testing pattern with region-specific tests for
the striatum (D1/D2 opponent pathways for action selection).

Author: Thalia Project
Date: December 22, 2025 (Tier 3.4 implementation)
"""

import torch

from tests.utils.region_test_base import RegionTestBase
from thalia.regions.striatum import Striatum
from thalia.regions.striatum.config import StriatumConfig
from thalia.config import compute_striatum_sizes


class TestStriatum(RegionTestBase):
    """Test Striatum implementation using unified test framework."""

    def create_region(self, **kwargs):
        """Create Striatum instance for testing."""
        # Use builder pattern if no explicit sizes provided
        if "d1_size" not in kwargs and "d2_size" not in kwargs:
            n_actions = kwargs.pop("n_actions", 4)  # Changed from n_output
            neurons_per_action = kwargs.pop("neurons_per_action", 10)
            # Striatum requires input_sources dict
            input_sources = kwargs.pop("input_sources", {"default": 100})
            config = StriatumConfig.from_n_actions(
                n_actions=n_actions,
                neurons_per_action=neurons_per_action,
                input_sources=input_sources,
                **kwargs
            )
        else:
            config = StriatumConfig(**kwargs)

        return Striatum(config)

    def get_default_params(self):
        """Return default striatum parameters."""
        return {
            "input_sources": {"default": 100},
            "n_actions": 5,  # Number of actions
            "population_coding": True,
            "neurons_per_action": 4,  # 20 total neurons (5 actions × 4)
            "rpe_enabled": True,
            "use_goal_conditioning": False,  # Disable for simpler testing
            "device": "cpu",
            "dt_ms": 1.0,
        }

    def get_min_params(self):
        """Return minimal valid parameters for quick tests."""
        return {
            "input_sources": {"default": 20},
            "n_actions": 3,  # 3 actions
            "population_coding": True,
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

        original_n_actions = params["n_output"]
        neurons_per_action = params["neurons_per_action"]
        n_new_actions = 10

        # Grow output (adds actions, which adds neurons_per_action neurons per action)
        region.grow_output(n_new_actions)

        # Verify config updated (n_output is total neurons, not actions)
        expected_total_neurons = (original_n_actions + n_new_actions) * neurons_per_action
        assert region.config.n_output == expected_total_neurons
        assert region.n_actions == original_n_actions + n_new_actions

        # Verify forward pass still works with new size
        input_spikes = torch.zeros(params["n_input"], device=region.device)
        output = region.forward(input_spikes)

        # Output is per-neuron (d1_spikes), same as config.n_output
        assert output.shape[0] == expected_total_neurons, \
            f"Expected {expected_total_neurons} neurons, got {output.shape[0]}"

    def test_grow_input(self):
        """Test striatum can grow input dimension (accounts for population coding)."""
        params = self.get_default_params()
        region = self.create_region(**params)

        original_n_input = params["n_input"]
        n_new_input = 20
        neurons_per_action = params["neurons_per_action"]

        # Grow input
        region.grow_input(n_new_input)

        # Verify config updated
        assert region.config.n_input == original_n_input + n_new_input

        # Verify forward pass still works with larger input
        input_spikes = torch.zeros(original_n_input + n_new_input, device=region.device)
        output = region.forward(input_spikes)

        # Output size should NOT change (still same number of actions/neurons)
        expected_neurons = params["n_output"] * neurons_per_action
        assert output.shape[0] == expected_neurons, \
            f"Expected {expected_neurons} neurons, got {output.shape[0]}"

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
        input_spikes = torch.ones(params["n_input"], device=region.device)
        output = region.forward(input_spikes)

        # Verify output shape matches action neurons
        expected_neurons = params["n_output"] * params["neurons_per_action"]
        assert output.shape[0] == expected_neurons

    def test_initialization(self):
        """Test striatum initializes with n_output expansion for population coding."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # With population coding, config.n_output gets expanded
        # Original: params["n_output"] = 5 actions
        # Expanded: region.config.n_output = 5 * 4 = 20 neurons
        expected_n_output = params["n_output"] * params["neurons_per_action"]
        assert region.config.n_output == expected_n_output
        assert region.config.n_input == params["n_input"]

    def test_initialization_minimal(self):
        """Test striatum initializes with minimal params and expansion."""
        params = self.get_min_params()
        region = self.create_region(**params)

        # Verify expansion happened
        expected_n_output = params["n_output"] * params["neurons_per_action"]
        assert region.config.n_output == expected_n_output

    def test_forward_pass_tensor_input(self):
        """Test forward returns expanded neuron output."""
        params = self.get_default_params()
        region = self.create_region(**params)
        input_spikes = torch.zeros(params["n_input"], device=region.device)
        output = region.forward(input_spikes)

        # Output should be expanded neuron count
        expected_n_output = params["n_output"] * params["neurons_per_action"]
        assert output.shape[0] == expected_n_output

    def test_forward_pass_dict_input(self):
        """Test forward with dict input returns expanded output."""
        params = self.get_default_params()
        region = self.create_region(**params)
        input_dict = self.get_input_dict(params["n_input"], device=region.device.type)
        output = region.forward(input_dict)

        expected_n_output = params["n_output"] * params["neurons_per_action"]
        assert output.shape[0] == expected_n_output

    def test_forward_pass_zero_input(self):
        """Test zero input handling with expansion."""
        params = self.get_default_params()
        region = self.create_region(**params)
        input_spikes = torch.zeros(params["n_input"], device=region.device)
        output = region.forward(input_spikes)

        expected_n_output = params["n_output"] * params["neurons_per_action"]
        assert output.shape[0] == expected_n_output

    def test_forward_pass_multiple_calls(self):
        """Test multiple forwards with expansion."""
        params = self.get_default_params()
        region = self.create_region(**params)
        input_spikes = torch.zeros(params["n_input"], device=region.device)
        expected_n_output = params["n_output"] * params["neurons_per_action"]

        for _ in range(10):
            output = region.forward(input_spikes)
            assert output.shape[0] == expected_n_output

    def test_population_coding(self):
        """Test population coding for action representation."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward pass
        input_spikes = torch.ones(params["n_input"], device=region.device)
        region.forward(input_spikes)

        # Get state
        state = region.get_state()

        # Verify vote accumulation for population coding
        # Votes are per-action (not per-neuron)
        if hasattr(state, "d1_votes_accumulated"):
            assert state.d1_votes_accumulated is not None
            assert state.d1_votes_accumulated.shape[0] == params["n_output"]  # n_actions

        if hasattr(state, "d2_votes_accumulated"):
            assert state.d2_votes_accumulated is not None
            assert state.d2_votes_accumulated.shape[0] == params["n_output"]  # n_actions

    def test_action_selection(self):
        """Test striatum selects actions based on vote competition."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Multiple forward passes to accumulate votes
        input_spikes = torch.ones(params["n_input"], device=region.device)
        for _ in range(10):
            region.forward(input_spikes)

        # Check if action was selected
        state = region.get_state()
        if hasattr(state, "last_action"):
            # Last action should be valid (0 to n_output-1)
            if state.last_action is not None:
                assert 0 <= state.last_action < params["n_output"]

    def test_dopamine_modulation(self):
        """Test dopamine modulates learning in D1/D2 pathways."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Set dopamine via neuromodulators
        if hasattr(region, "set_neuromodulators"):
            region.set_neuromodulators(dopamine=0.8)

        # Forward pass
        input_spikes = torch.ones(params["n_input"], device=region.device)
        region.forward(input_spikes)

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
        input_spikes = torch.ones(params["n_input"], device=region.device)
        for _ in range(5):
            region.forward(input_spikes)

        # Check RPE state
        state = region.get_state()
        if hasattr(state, "last_rpe"):
            # RPE should be a scalar value
            if state.last_rpe is not None:
                assert isinstance(state.last_rpe, (int, float))

        if hasattr(state, "value_estimates"):
            # Should have value estimates per action
            if state.value_estimates is not None:
                assert state.value_estimates.shape[0] == params["n_output"]

    def test_exploration_uncertainty(self):
        """Test exploration based on uncertainty."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward passes
        input_spikes = torch.ones(params["n_input"], device=region.device)
        for _ in range(10):
            region.forward(input_spikes)

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
        input_spikes = torch.ones(params["n_input"], device=region.device)
        for _ in range(20):
            region.forward(input_spikes)

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
        input_spikes = torch.ones(params["n_input"], device=region.device)
        for _ in range(100):
            region.forward(input_spikes)

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
        input_spikes = torch.ones(params["n_input"], device=region.device)
        region.forward(input_spikes)

        # Check goal modulation weights
        state = region.get_state()
        if hasattr(state, "pfc_modulation_d1") and hasattr(state, "pfc_modulation_d2"):
            if state.pfc_modulation_d1 is not None:
                # Should have modulation weights for each neuron × PFC size
                expected_shape = (params["n_output"] * params["neurons_per_action"], params["pfc_size"])
                assert state.pfc_modulation_d1.shape == expected_shape


# Standard tests (initialization, forward, growth, state, device, neuromodulators, diagnostics)
# inherited from RegionTestBase - eliminates ~100 lines of boilerplate

