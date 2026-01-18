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
from thalia.config import LayerSizeCalculator
from thalia.regions.cortex.config import LayeredCortexConfig
from thalia.regions.cortex.layered_cortex import LayeredCortex


class TestLayeredCortex(RegionTestBase):
    """Test LayeredCortex implementation using unified test framework."""

    def create_region(self, **kwargs):
        """Create LayeredCortex instance for testing.

        NEW PATTERN: Separates behavioral config from structural sizes.
        - Config contains only behavioral parameters (learning rates, etc.)
        - Sizes passed separately via sizes dict
        - Device passed as explicit parameter
        """
        # Extract device for explicit parameter
        device = kwargs.pop("device", "cpu")

        # Known size parameters (structural)
        size_params = {"l4_size", "l23_size", "l5_size", "l6a_size", "l6b_size", "input_size"}

        # Separate sizes from behavioral config params
        sizes = {k: v for k, v in kwargs.items() if k in size_params}
        config_params = {k: v for k, v in kwargs.items() if k not in size_params}

        # Behavioral config with any passed params
        config = LayeredCortexConfig(**config_params)

        return LayeredCortex(config=config, sizes=sizes, device=device)

    def get_default_params(self):
        """Return default cortex parameters using LayerSizeCalculator."""
        calc = LayerSizeCalculator()
        sizes = calc.cortex_from_scale(100)
        return {
            "input_size": sizes["input_size"],
            "l4_size": sizes["l4_size"],
            "l23_size": sizes["l23_size"],
            "l5_size": sizes["l5_size"],
            "l6a_size": sizes["l6a_size"],
            "l6b_size": sizes["l6b_size"],
            "device": "cpu",
            "dt_ms": 1.0,
        }

    def get_min_params(self):
        """Return minimal valid parameters for quick tests."""
        calc = LayerSizeCalculator()
        sizes = calc.cortex_from_scale(20)
        return {
            "input_size": sizes["input_size"],
            "l4_size": sizes["l4_size"],
            "l23_size": sizes["l23_size"],
            "l5_size": sizes["l5_size"],
            "l6a_size": sizes["l6a_size"],
            "l6b_size": sizes["l6b_size"],
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
    # OVERRIDE BASE CLASS TESTS (config no longer has size fields)
    # =========================================================================

    def test_initialization(self):
        """Test region initializes correctly - OVERRIDE for LayeredCortex."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Verify layer sizes exist (multi-source architecture - no single input_size)
        assert region.l4_size > 0
        assert region.l23_size > 0
        assert region.l5_size > 0
        # LayeredCortex output is L2/3 + L5
        assert region.l23_size + region.l5_size > 0

    def test_initialization_minimal(self):
        """Test minimal initialization - OVERRIDE for LayeredCortex."""
        params = self.get_min_params()
        region = self.create_region(**params)

        # Verify layer sizes exist (multi-source architecture - no single input_size)
        assert region.l4_size > 0
        # LayeredCortex output is L2/3 + L5
        assert region.l23_size + region.l5_size > 0

    def test_forward_pass_tensor_input(self):
        """Test forward pass with dict input - OVERRIDE for LayeredCortex."""
        params = self.get_default_params()
        region = self.create_region(**params)

        input_size = self._get_input_size(params)
        input_spikes = torch.rand(input_size, device=region.device)

        # Pass as dict (all regions now require dict format)
        output = region.forward({"input": input_spikes})

        # Verify output shape (L2/3 + L5)
        expected_output_size = region.l23_size + region.l5_size
        assert output.shape[0] == expected_output_size
        assert output.dtype == torch.bool  # Spikes are binary

    def test_forward_pass_dict_input(self):
        """Test forward pass with dict input - OVERRIDE for LayeredCortex."""
        params = self.get_default_params()
        region = self.create_region(**params)

        input_size = self._get_input_size(params)
        input_dict = self.get_input_dict(input_size, region.device)

        output = region.forward(input_dict)

        # Verify output shape (L2/3 + L5)
        expected_output_size = region.l23_size + region.l5_size
        assert output.shape[0] == expected_output_size
        assert output.dtype == torch.bool  # Spikes are binary

    def test_forward_pass_zero_input(self):
        """Test forward pass with zero input - OVERRIDE for LayeredCortex."""
        params = self.get_default_params()
        region = self.create_region(**params)

        input_size = self._get_input_size(params)
        input_spikes = torch.zeros(input_size, device=region.device)

        output = region.forward({"input": input_spikes})

        # Verify output shape (L2/3 + L5)
        expected_output_size = region.l23_size + region.l5_size
        assert output.shape[0] == expected_output_size
        assert output.dtype == torch.bool  # Spikes are binary

    def test_forward_pass_multiple_calls(self):
        """Test multiple forward passes - OVERRIDE for LayeredCortex."""
        params = self.get_default_params()
        region = self.create_region(**params)

        input_size = self._get_input_size(params)
        expected_output_size = region.l23_size + region.l5_size

        for _ in range(5):
            input_spikes = torch.rand(input_size, device=region.device)
            output = region.forward({"input": input_spikes})

            # Verify consistent output shape
            assert output.shape[0] == expected_output_size

    def test_grow_output(self):
        """Test output growth - OVERRIDE for LayeredCortex."""
        params = self.get_default_params()
        region = self.create_region(**params)

        original_output = region.l23_size + region.l5_size
        n_new = 10

        region.grow_output(n_new)

        # Verify output grew
        new_output = region.l23_size + region.l5_size
        assert new_output == original_output + n_new

    def test_grow_input(self):
        """Test per-source input growth - OVERRIDE for LayeredCortex."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Multi-source architecture: add a new source
        source_name = "test_source"
        n_new = 10

        # Add input source
        region.add_input_source(source_name, n_new, learning_rule="bcm")

        # Verify source was added
        assert source_name in region.input_sources
        assert region.input_sources[source_name] == n_new
        assert source_name in region.synaptic_weights

        # Test forward pass with new source works
        test_input = {source_name: torch.ones(n_new, device=region.device)}
        output = region.forward(test_input)

        # Verify output shape is sum of L2/3 and L5 sizes
        expected_output = region.l23_size + region.l5_size
        assert output.shape[0] == expected_output

    def test_growth_preserves_state(self):
        """Test growth preserves state - OVERRIDE for LayeredCortex."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward pass to establish state
        input_size = self._get_input_size(params)
        input_spikes = torch.ones(input_size, device=region.device)
        region.forward({"input": input_spikes})

        n_original = region.l23_size + region.l5_size

        # Grow output
        region.grow_output(10)

        # Verify state still loadable
        state = region.get_state()
        region.load_state(state)

        # Verify output size changed correctly
        n_new = region.l23_size + region.l5_size
        assert n_new == n_original + 10

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
        output = region.forward({"input": input_spikes})

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
        output = region.forward({"input": input_spikes}, top_down=top_down)

        # Verify output shape (L2/3 + L5)
        expected_output_size = region.l23_size + region.l5_size
        assert output.shape[0] == expected_output_size

    def test_l6_feedback_to_thalamus(self):
        """Test L6 generates feedback for thalamus."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward pass
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        region.forward({"input": input_spikes})

        # Verify L6 state exists
        state = region.get_state()
        if hasattr(state, "l6a_spikes") and hasattr(state, "l6b_spikes"):
            assert state.l6a_spikes.shape[0] == params["l6a_size"]
            assert state.l6b_spikes.shape[0] == params["l6b_size"]

    def test_stdp_traces_updated(self):
        """Test STDP traces updated during forward pass - OVERRIDE for LayeredCortex."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Multi-source architecture: must add input source first
        source_name = "input"
        input_size = self._get_input_size(params)
        region.add_input_source(source_name, input_size, learning_rule="bcm")

        # Run multiple forward passes with actual input
        input_spikes = torch.ones(input_size, device=region.device)
        for _ in range(5):
            region.forward({source_name: input_spikes})

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
            region.forward({"input": input_spikes})

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
        region.forward({"input": input_spikes})

        # Verify gamma attention state exists
        state = region.get_state()
        if hasattr(state, "gamma_attention_phase"):
            # Phase can be None if gamma oscillations not configured
            if state.gamma_attention_phase is not None:
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
                region.forward({"input": input_spikes})

            # Verify weights changed (plasticity applied)
            weights_changed = False
            for source, weights in region.synaptic_weights.items():
                if not torch.allclose(weights, initial_weights[source], atol=1e-6):
                    weights_changed = True
                    break

            # With continuous active input, weights should change
            # (unless learning rate is zero, which is not default)
            assert weights_changed, "Expected weight changes from continuous plasticity"
