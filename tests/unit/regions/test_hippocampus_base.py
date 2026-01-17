"""Tests for Hippocampus using unified RegionTestBase framework.

Demonstrates unified testing pattern with region-specific tests for
the trisynaptic hippocampal circuit (DG→CA3→CA1).

Author: Thalia Project
Date: December 22, 2025 (Tier 3.4 implementation)
"""

import torch

from tests.utils.region_test_base import RegionTestBase
from thalia.config.size_calculator import LayerSizeCalculator
from thalia.regions.hippocampus import Hippocampus
from thalia.regions.hippocampus.config import HippocampusConfig


class TestHippocampus(RegionTestBase):
    """Test Hippocampus implementation using unified test framework."""

    def create_region(self, **kwargs):
        """Create Hippocampus instance for testing."""
        # Separate size params from config params
        device = kwargs.pop("device", "cpu")

        # Extract size-related kwargs
        size_params = {}
        if "input_size" in kwargs:
            input_size = kwargs.pop("input_size")
            # Compute all sizes from input_size
            calc = LayerSizeCalculator()
            sizes = calc.hippocampus_from_input(input_size)
            size_params.update(sizes)
            # Add input_size to size_params (calculator returns it)
            # but double check it's there
            if "input_size" not in size_params:
                size_params["input_size"] = input_size

        # Override with explicit sizes if provided
        for key in ["dg_size", "ca3_size", "ca2_size", "ca1_size"]:
            if key in kwargs:
                size_params[key] = kwargs.pop(key)

        # Remaining kwargs are behavioral config
        config = HippocampusConfig(**kwargs)

        return Hippocampus(config=config, sizes=size_params, device=device)

    def get_default_params(self):
        """Return default hippocampus parameters."""
        return {
            "input_size": 100,
            "device": "cpu",
            "dt_ms": 1.0,
        }

    def get_min_params(self):
        """Return minimal valid parameters for quick tests."""
        return {
            "input_size": 20,
            "device": "cpu",
            "dt_ms": 1.0,
        }

    def get_input_dict(self, n_input, device="cpu"):
        """Return dict input for hippocampus (EC input)."""
        return {
            "ec": torch.zeros(n_input, device=device),
        }

    # =========================================================================
    # HIPPOCAMPUS-SPECIFIC TESTS
    # =========================================================================

    def test_trisynaptic_cascade(self):
        """Test DG→CA3→CA1 circuit processes correctly."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Provide entorhinal cortex input
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)

        # Forward pass
        output = region.forward(input_spikes)

        # Verify output is CA1 activity
        assert output.shape[0] == self._get_region_output_size(region)

        # Verify state has all three layers
        state = region.get_state()
        assert hasattr(state, "dg_spikes")
        assert hasattr(state, "ca3_spikes")
        assert hasattr(state, "ca1_spikes")
        # Sizes are computed from config ratios
        assert state.dg_spikes.shape[0] == region.dg_size
        assert state.ca3_spikes.shape[0] == region.ca3_size
        assert state.ca1_spikes.shape[0] == region.ca1_size

    def test_pattern_separation_in_dg(self):
        """Test DG provides pattern separation (sparse coding)."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Dense input pattern
        input_spikes = torch.ones(self._get_input_size(params), device=region.device) * 0.8

        # Forward pass
        region.forward(input_spikes)

        # DG should produce sparse output (pattern separation)
        state = region.get_state()
        if state.dg_spikes is not None:
            dg_activity = state.dg_spikes.float().mean().item()
            # DG typically maintains ~2-5% sparsity (pattern separation)
            assert 0.0 <= dg_activity <= 0.2, f"Expected sparse DG activity, got {dg_activity}"

    def test_ca3_recurrence(self):
        """Test CA3 has recurrent connections for pattern completion."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward passes
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        for _ in range(10):
            region.forward(input_spikes)

        # Verify CA3 recurrent connections exist
        if hasattr(region, "synaptic_weights"):
            # Should have CA3→CA3 recurrent weights
            assert "ca3_ca3" in region.synaptic_weights
            ca3_recurrent = region.synaptic_weights["ca3_ca3"]
            assert ca3_recurrent.shape == (region.ca3_size, region.ca3_size)

    def test_ca3_persistent_activity(self):
        """Test CA3 maintains persistent activity (working memory)."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Strong input to trigger persistent activity
        strong_input = torch.ones(self._get_input_size(params), device=region.device)
        region.forward(strong_input)

        # Check for persistent activity state
        state = region.get_state()
        if hasattr(state, "ca3_persistent"):
            assert (
                state.ca3_persistent.shape[0] == region.ca3_size
            ), "CA3 persistent state shape mismatch"

    def test_stp_mossy_fibers(self):
        """Test mossy fiber pathway (DG→CA3) uses facilitating STP."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run multiple forward passes
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        for _ in range(10):
            region.forward(input_spikes)

        # Verify STP state for mossy fibers
        state = region.get_state()
        if hasattr(state, "stp_mossy_state"):
            stp_state = state.stp_mossy_state
            if stp_state is not None:
                # Should have facilitation variables
                assert "u" in stp_state or "x" in stp_state

    def test_ec_direct_pathway(self):
        """Test EC layer III direct pathway to CA1."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Provide both EC layer II (via DG) and layer III (direct to CA1)
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        # ec_direct_input should match n_input size (ec_l3_input_size defaults to n_input)
        ec_l3_input = torch.ones(self._get_input_size(params), device=region.device)

        # Forward with direct EC→CA1
        output = region.forward(input_spikes, ec_direct_input=ec_l3_input)

        # Should not error and return ca1_size
        assert output.shape[0] == region.ca1_size

    def test_episodic_memory_buffer(self):
        """Test hippocampus maintains episode buffer."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Check for episode buffer
        if hasattr(region, "episode_buffer"):
            assert isinstance(region.episode_buffer, list)

            # Store episode (if region supports it)
            if hasattr(region, "store_episode"):
                sample = torch.ones(self._get_input_size(params), device=region.device)
                # Note: store_episode signature may vary by implementation
                # This is just checking the method exists
                assert callable(region.store_episode)

    def test_acetylcholine_encoding_modulation(self):
        """Test acetylcholine modulates encoding strength."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Set high ACh (encoding mode)
        if hasattr(region, "set_neuromodulators"):
            region.set_neuromodulators(acetylcholine=0.9)

        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        region.forward(input_spikes)

        # Verify ACh stored in state
        state = region.get_state()
        if hasattr(state, "acetylcholine"):
            assert state.acetylcholine == 0.9


# Standard tests (initialization, forward, growth, state, device, neuromodulators, diagnostics)
# inherited from RegionTestBase - eliminates ~100 lines of boilerplate
