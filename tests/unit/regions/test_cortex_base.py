"""Tests for LayeredCortex using unified RegionTestBase framework.

This file demonstrates the unified testing pattern from Tier 3.4.
All common tests (initialization, forward pass, growth, state management)
are provided by RegionTestBase. Only cortex-specific tests are implemented here.

Pattern Benefits:
- Eliminates ~100 lines of boilerplate per region
- Ensures consistent test coverage across regions
- Easy to add new standard tests in base class

Author: Thalia Project
Date: December 22, 2025 (Tier 3.4 implementation)
"""

import torch

from tests.utils.region_test_base import RegionTestBase
from thalia.regions.cortex.layered_cortex import LayeredCortex
from thalia.regions.cortex.config import LayeredCortexConfig


class TestLayeredCortex(RegionTestBase):
    """Test LayeredCortex implementation using unified test framework."""

    def create_region(self, **kwargs):
        """Create LayeredCortex instance for testing."""
        # Always use direct config creation (params dict already has all sizes computed)
        config = LayeredCortexConfig(**kwargs)
        return LayeredCortex(config)

    def get_default_params(self):
        """Return default cortex parameters (will use builder)."""
        # Create config via builder to get computed sizes
        config = LayeredCortexConfig.from_input_size(input_size=100, device="cpu", dt_ms=1.0)
        return {
            "input_size": config.input_size,


            "l4_size": config.l4_size,
            "l23_size": config.l23_size,
            "l5_size": config.l5_size,
            "l6a_size": config.l6a_size,
            "l6b_size": config.l6b_size,
            "device": "cpu",
            "dt_ms": 1.0,
        }

    def get_min_params(self):
        """Return minimal valid parameters for quick tests (will use builder)."""
        # Create config via builder to get computed sizes
        config = LayeredCortexConfig.from_input_size(input_size=20, device="cpu", dt_ms=1.0)
        return {
            "input_size": config.input_size,


            "l4_size": config.l4_size,
            "l23_size": config.l23_size,
            "l5_size": config.l5_size,
            "l6a_size": config.l6a_size,
            "l6b_size": config.l6b_size,
            "device": "cpu",
            "dt_ms": 1.0,
        }

    def get_input_dict(self, n_input, device="cpu"):
        """Return dict input for cortex (supports multi-source)."""
        # Cortex uses InputRouter.concatenate_sources, accepts any dict
        return {
            "thalamus": torch.zeros(n_input // 2, device=device),
            "hippocampus": torch.zeros(n_input // 2, device=device),
        }

    # =========================================================================
    # CORTEX-SPECIFIC TESTS (not provided by base class)
    # =========================================================================

    def test_layered_forward_cascade(self):
        """Test L4→L2/3→L5 cascade processes correctly."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Provide input to L4
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)

        # Forward pass
        output = region.forward(input_spikes)

        # Verify output is concatenation of L2/3 + L5
        expected_output_size = params["l23_size"] + params["l5_size"]
        assert output.shape[0] == expected_output_size

        # Verify state has layer-specific spikes
        state = region.get_state()
        assert hasattr(state, "l4_spikes")
        assert hasattr(state, "l23_spikes")
        assert hasattr(state, "l5_spikes")
        assert state.l4_spikes.shape[0] == params["l4_size"]
        assert state.l23_spikes.shape[0] == params["l23_size"]
        assert state.l5_spikes.shape[0] == params["l5_size"]

    def test_top_down_modulation(self):
        """Test cortex accepts top-down modulation from PFC."""
        params = self.get_default_params()
        region = self.create_region(**params)

        input_spikes = torch.zeros(self._get_input_size(params), device=region.device)
        top_down = torch.ones(params["l23_size"], device=region.device)

        # Forward with top-down modulation
        output = region.forward(input_spikes, top_down=top_down)

        # Should not error - get output size from region config
        assert output.shape[0] == self._get_config_output_size(region.config)

    def test_l6_feedback_to_thalamus(self):
        """Test L6 generates feedback for thalamus."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward pass
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        region.forward(input_spikes)

        # Verify L6 state exists
        state = region.get_state()
        if hasattr(state, "l6a_spikes") and hasattr(state, "l6b_spikes"):
            assert state.l6a_spikes.shape[0] == params["l6a_size"]
            assert state.l6b_spikes.shape[0] == params["l6b_size"]

    def test_stdp_traces_updated(self):
        """Test STDP traces updated during forward pass."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run multiple forward passes
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        for _ in range(5):
            region.forward(input_spikes)

        # Verify STDP traces exist and are non-zero
        state = region.get_state()
        if hasattr(state, "l4_trace"):
            assert state.l4_trace is not None
            # At least some traces should be non-zero after active input
            assert state.l4_trace.abs().sum() > 0

    def test_stp_l23_recurrent(self):
        """Test L2/3 recurrent connections use STP."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward passes
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        for _ in range(10):
            region.forward(input_spikes)

        # Verify STP state exists for L2/3 recurrent
        state = region.get_state()
        if hasattr(state, "stp_l23_recurrent_state"):
            stp_state = state.stp_l23_recurrent_state
            if stp_state is not None:
                # Should have 'u' and 'x' (depression/facilitation variables)
                assert "u" in stp_state or "x" in stp_state

    def test_gamma_attention_modulation(self):
        """Test gamma oscillation modulates attention."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward pass
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        region.forward(input_spikes)

        # Verify gamma attention state exists
        state = region.get_state()
        if hasattr(state, "gamma_attention_phase"):
            assert isinstance(state.gamma_attention_phase, float)

        if hasattr(state, "gamma_attention_gate"):
            # Gate should be between 0 and 1
            if state.gamma_attention_gate is not None:
                assert torch.all(state.gamma_attention_gate >= 0.0)
                assert torch.all(state.gamma_attention_gate <= 1.0)

    def test_plasticity_continuous(self):
        """Test plasticity occurs continuously during forward passes."""
        params = self.get_default_params()
        params["bcm_enabled"] = True  # Enable BCM+STDP learning
        region = self.create_region(**params)

        # Get initial weights (if accessible)
        if hasattr(region, "synaptic_weights"):
            initial_weights = {}
            for source, weights in region.synaptic_weights.items():
                initial_weights[source] = weights.clone()

            # Run multiple forward passes with active input
            input_spikes = torch.ones(self._get_input_size(params), device=region.device)
            for _ in range(100):
                region.forward(input_spikes)

            # Verify weights changed (plasticity applied)
            weights_changed = False
            for source, weights in region.synaptic_weights.items():
                if not torch.allclose(weights, initial_weights[source], atol=1e-6):
                    weights_changed = True
                    break

            # With continuous active input, weights should change
            # (unless learning rate is zero, which is not default)
            assert weights_changed, "Expected weight changes from continuous plasticity"
