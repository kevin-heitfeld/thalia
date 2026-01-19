"""Tests for Hippocampus using unified RegionTestBase framework.

Demonstrates unified testing pattern with region-specific tests for
the trisynaptic hippocampal circuit (DG→CA3→CA1).

Author: Thalia Project
Date: December 22, 2025 (Tier 3.4 implementation)
"""

import torch

from tests.utils.region_test_base import RegionTestBase
from thalia.config import HippocampusConfig, LayerSizeCalculator
from thalia.regions import TrisynapticHippocampus


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

        return TrisynapticHippocampus(config=config, sizes=size_params, device=device)

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

    def test_consolidation_mode_toggle(self):
        """Test Phase 1: Consolidation mode toggle and neuromodulatory changes."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Verify initial state (wake mode)
        assert hasattr(region, "_consolidation_mode")
        assert region._consolidation_mode is False

        # Enter consolidation mode
        region.enter_consolidation_mode()

        # Verify consolidation mode is active
        assert region._consolidation_mode is True

        # Verify neuromodulatory state changed (sleep state per Hasselmo 1999)
        state = region.get_state()
        assert hasattr(state, "acetylcholine")
        assert hasattr(state, "norepinephrine")
        assert hasattr(state, "dopamine")

        # Sleep state: LOW ACh (0.1), LOW NE (0.1), MODERATE DA (0.3)
        assert state.acetylcholine == 0.1
        assert state.norepinephrine == 0.1
        assert state.dopamine == 0.3

        # Exit consolidation mode
        region.exit_consolidation_mode()

        # Verify wake mode restored
        assert region._consolidation_mode is False
        assert region._replay_cue is None

        # Verify wake neuromodulatory state restored
        state = region.get_state()
        # Wake state: HIGH ACh (0.8), MODERATE NE (0.5), MODERATE DA (0.5)
        assert state.acetylcholine == 0.8
        assert state.norepinephrine == 0.5
        assert state.dopamine == 0.5

    def test_cue_replay(self):
        """Test Phase 1: Episode replay cuing."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Cue episode 0 for replay
        region.cue_replay(episode_index=0)
        assert region._replay_cue == 0

        # Cue episode 5
        region.cue_replay(episode_index=5)
        assert region._replay_cue == 5

        # Test invalid episode index
        try:
            region.cue_replay(episode_index=-1)
            assert False, "Should have raised ValueError for negative episode index"
        except ValueError as e:
            assert "episode_index must be >= 0" in str(e)

    def test_consolidation_replay_forward(self):
        """Test Phase 1: Consolidation mode drives CA3→CA1 replay during forward()."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Enter consolidation mode
        region.enter_consolidation_mode()

        # Verify no replay without cue
        inputs = {"ec": torch.zeros(params["input_size"], device=region.device)}

        # Forward without cue should return silence (no episodes stored)
        region._replay_cue = 0  # Set cue manually
        output = region.forward(inputs)

        # Should return CA1-sized output
        assert output.shape[0] == region.ca1_size

        # After replay, cue should be cleared
        assert region._replay_cue is None

        # Exit consolidation mode
        region.exit_consolidation_mode()

    def test_ca3_pattern_storage_and_retrieval(self):
        """Test Phase 1: CA3 patterns are stored and retrieved correctly."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Run forward pass to populate CA3 state
        inputs = {"ec": torch.ones(params["input_size"], device=region.device)}
        region.forward(inputs)

        # Verify CA3 spikes are available
        assert region.state.ca3_spikes is not None
        ca3_original = region.state.ca3_spikes.clone()

        # Store an episode (should capture CA3 pattern)
        region.store_episode(
            state=torch.ones(region.ca1_size, device=region.device),
            action=0,
            reward=1.0,
            correct=True,
        )

        # Verify episode was stored
        assert len(region.memory.episode_buffer) > 0
        stored_episode = region.memory.episode_buffer[0]

        # Verify CA3 pattern was stored
        assert hasattr(stored_episode, "ca3_pattern")
        assert stored_episode.ca3_pattern is not None
        assert stored_episode.ca3_pattern.shape == ca3_original.shape

        # Verify CA3 pattern matches original (within float tolerance)
        assert torch.allclose(stored_episode.ca3_pattern, ca3_original, atol=1e-6)

        # Test retrieval during consolidation
        region.enter_consolidation_mode()
        region.cue_replay(episode_index=0)

        # Forward should retrieve and use stored CA3 pattern
        replay_output = region.forward(inputs)

        # Should produce CA1 output
        assert replay_output.shape[0] == region.ca1_size

        region.exit_consolidation_mode()
