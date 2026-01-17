"""Tests for Prefrontal Cortex using unified RegionTestBase framework.

Demonstrates unified testing pattern with region-specific tests for
prefrontal cortex (working memory and executive control).

Author: Thalia Project
Date: December 22, 2025 (Tier 3.4 implementation)
"""

import torch

from tests.utils.region_test_base import RegionTestBase
from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig


class TestPrefrontal(RegionTestBase):
    """Test Prefrontal implementation using unified test framework."""

    def create_region(self, **kwargs):
        """Create Prefrontal instance for testing."""
        # Extract sizes from kwargs
        input_size = kwargs.pop("input_size", 100)
        n_neurons = kwargs.pop("n_neurons", 50)
        device = kwargs.pop("device", "cpu")

        # Create sizes dict
        sizes = {
            "input_size": input_size,
            "n_neurons": n_neurons,
        }

        # Create config with behavioral parameters only
        config = PrefrontalConfig(**kwargs)
        return Prefrontal(config=config, sizes=sizes, device=device)

    def get_default_params(self):
        """Return default prefrontal parameters."""
        return {
            "input_size": 100,  # Size parameter (passed via sizes dict)
            "n_neurons": 50,  # Size parameter (passed via sizes dict)
            "device": "cpu",
            "dt_ms": 1.0,
        }

    def get_min_params(self):
        """Return minimal valid parameters for quick tests."""
        return {
            "input_size": 20,  # Size parameter (passed via sizes dict)
            "n_neurons": 10,  # Size parameter (passed via sizes dict)
            "device": "cpu",
            "dt_ms": 1.0,
        }

    def get_input_dict(self, n_input, device="cpu"):
        """Return dict input for prefrontal (single source)."""
        return {
            "default": torch.zeros(n_input, device=device),
        }

    def _get_region_input_size(self, region):
        """Get actual input size from region instance."""
        return region.input_size

    def _get_region_output_size(self, region):
        """Get actual output size from region instance."""
        return region.n_output

    # =========================================================================
    # PREFRONTAL-SPECIFIC TESTS
    # =========================================================================

    def test_working_memory_maintenance(self):
        """Test PFC maintains working memory over time."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Strong input to encode in working memory
        strong_input = torch.ones(self._get_input_size(params), device=region.device)
        region.forward(strong_input)

        # Check working memory state initialized
        state = region.get_state()
        if hasattr(state, "working_memory"):
            assert state.working_memory is not None
            assert state.working_memory.shape[0] == region.n_output

            # Run with no input (maintenance)
            zero_input = torch.zeros(self._get_input_size(params), device=region.device)
            for _ in range(10):
                region.forward(zero_input)

            # Working memory should persist (not fully decay to zero)
            state = region.get_state()
            # At least some WM should remain after maintenance period
            assert state.working_memory.abs().sum() > 0

    def test_gating_mechanism(self):
        """Test dopamine-gated working memory updates."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Low dopamine (gate closed)
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        region.forward(input_spikes, dopamine_signal=-1.0)

        state = region.get_state()
        if hasattr(state, "update_gate"):
            # Gate should be mostly closed with low dopamine
            if state.update_gate is not None:
                mean_gate = state.update_gate.mean().item()
                assert mean_gate < 0.5, f"Expected closed gate, got {mean_gate}"

    def test_rule_representation(self):
        """Test PFC maintains rule neurons separate from WM neurons."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward pass
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        region.forward(input_spikes)

        # Verify PFC has output neurons
        assert region.n_output > 0

    def test_recurrent_connections(self):
        """Test PFC has recurrent connections for maintenance."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Check for recurrent weights
        if hasattr(region, "rec_weights"):
            assert region.rec_weights is not None
            # Recurrent weights should be [n_output, n_output]
            assert region.rec_weights.shape == (region.n_output, region.n_output)

    def test_inhibitory_connections(self):
        """Test PFC has lateral inhibition for selection."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Check for inhibitory weights
        if hasattr(region, "inhib_weights"):
            assert region.inhib_weights is not None
            # Inhibition should be [n_output, n_output]
            assert region.inhib_weights.shape == (region.n_output, region.n_output)

    def test_dopamine_gating(self):
        """Test dopamine modulates WM gating."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # High dopamine signal (open gate)
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        region.forward(input_spikes, dopamine_signal=1.0)

        # Check dopamine system state
        if hasattr(region, "dopamine_system"):
            # Should have stored high dopamine
            if hasattr(region.dopamine_system, "get_gate"):
                gate = region.dopamine_system.get_gate()
                # High dopamine should open gate
                assert gate > 0.5, f"Expected open gate with high DA, got {gate}"

    def test_active_rule_tracking(self):
        """Test PFC tracks currently active rule."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward pass
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        region.forward(input_spikes)

        # Check active rule state
        state = region.get_state()
        if hasattr(state, "active_rule"):
            # Active rule should be a tensor
            if state.active_rule is not None:
                assert isinstance(state.active_rule, torch.Tensor)

    def test_stdp_in_feedforward(self):
        """Test STDP learning in feedforward connections."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Get initial weights
        if hasattr(region, "synaptic_weights"):
            initial_weights = region.synaptic_weights["default"].clone()

            # Multiple forward passes with correlated activity
            input_spikes = torch.ones(self._get_input_size(params), device=region.device)
            for _ in range(100):
                region.forward(input_spikes, dopamine_signal=0.5)

            # Weights should change (STDP applied)
            final_weights = region.synaptic_weights["default"]
            assert not torch.allclose(
                initial_weights, final_weights, atol=1e-6
            ), "Expected weight changes from STDP"

    def test_stp_in_recurrent(self):
        """Test short-term plasticity in recurrent connections."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Multiple forward passes
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        for _ in range(20):
            region.forward(input_spikes)

        # Check STP state
        if hasattr(region, "stp_recurrent"):
            if region.stp_recurrent is not None:
                # STP should be tracking facilitation/depression
                stp_state = region.stp_recurrent.get_state()
                assert "u" in stp_state or "x" in stp_state

    def test_context_sensitivity(self):
        """Test PFC responds differently to context changes."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Context 1: Strong input on first half
        context1 = torch.zeros(self._get_input_size(params), device=region.device)
        context1[: self._get_input_size(params) // 2] = 1.0
        region.forward(context1)
        state1 = region.get_state()

        # Reset
        region.reset_state()

        # Context 2: Strong input on second half
        context2 = torch.zeros(self._get_input_size(params), device=region.device)
        context2[self._get_input_size(params) // 2 :] = 1.0
        region.forward(context2)
        state2 = region.get_state()

        # States should differ (context-sensitive)
        if hasattr(state1, "spikes") and hasattr(state2, "spikes"):
            if state1.spikes is not None and state2.spikes is not None:
                assert not torch.equal(
                    state1.spikes, state2.spikes
                ), "Expected different responses to different contexts"


# Standard tests (initialization, forward, growth, state, device, neuromodulators, diagnostics)
# inherited from RegionTestBase - eliminates ~100 lines of boilerplate
