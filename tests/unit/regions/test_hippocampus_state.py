"""Tests for HippocampusState edge cases and hippocampus-specific validation.

Basic state serialization tests (to_dict, from_dict, reset, roundtrip, file I/O)
are now covered by RegionTestBase (tests/utils/region_test_base.py).

This file focuses on:
- Nested STP state for 4 pathways (mossy, schaffer, ec_ca1, ca3_recurrent)
- CA3 persistent activity preservation
- Memory trace handling
- Edge cases not covered by base class

Created: October 10, 2024
Updated: December 22, 2025 (migrated basic tests to RegionTestBase)
"""

import pytest
import torch

from thalia.regions.hippocampus import Hippocampus, HippocampusState
from thalia.regions.hippocampus.config import HippocampusConfig
from thalia.core.region_state import save_region_state, load_region_state


class TestHippocampusStateEdgeCases:
    """Edge cases and hippocampus-specific validation."""

    def test_to_dict_with_stp_state(self):
        """Test to_dict() captures nested STP state for all 4 pathways."""
        # All 4 hippocampal pathways with STP
        stp_mossy = {"u": torch.rand(3, 4), "x": torch.rand(3, 4)}
        stp_schaffer = {"u": torch.rand(3, 3), "x": torch.rand(3, 3)}
        stp_ec_ca1 = {"u": torch.rand(3, 5), "x": torch.rand(3, 5)}
        stp_ca3_recurrent = {"u": torch.rand(3, 3), "x": torch.rand(3, 3)}

        state = HippocampusState(
            dg_spikes=torch.zeros(4),
            ca3_spikes=torch.zeros(3),
            ca1_spikes=torch.zeros(3),
            stp_mossy_state=stp_mossy,
            stp_schaffer_state=stp_schaffer,
            stp_ec_ca1_state=stp_ec_ca1,
            stp_ca3_recurrent_state=stp_ca3_recurrent,
        )

        data = state.to_dict()

        # Verify all 4 STP dicts preserved
        assert data["stp_mossy_state"] is not None
        assert torch.equal(data["stp_mossy_state"]["u"], stp_mossy["u"])
        assert data["stp_schaffer_state"] is not None
        assert torch.equal(data["stp_schaffer_state"]["u"], stp_schaffer["u"])
        assert data["stp_ec_ca1_state"] is not None
        assert torch.equal(data["stp_ec_ca1_state"]["u"], stp_ec_ca1["u"])
        assert data["stp_ca3_recurrent_state"] is not None
        assert torch.equal(data["stp_ca3_recurrent_state"]["u"], stp_ca3_recurrent["u"])

    def test_from_dict_with_nested_stp_state(self):
        """Test from_dict() restores all 4 nested STP pathway states."""
        stp_mossy = {"u": torch.rand(3, 4), "x": torch.rand(3, 4)}
        stp_schaffer = {"u": torch.rand(3, 3), "x": torch.rand(3, 3)}

        data = {
            "ca3_spikes": torch.zeros(3),
            "stp_mossy_state": stp_mossy,
            "stp_schaffer_state": stp_schaffer,
        }

        state = HippocampusState.from_dict(data)

        # Verify nested STP states restored
        assert torch.equal(state.stp_mossy_state["u"], stp_mossy["u"])
        assert torch.equal(state.stp_schaffer_state["u"], stp_schaffer["u"])

    def test_load_state_with_missing_stp_modules(self):
        """Test load_state() with missing STP modules (hippocampus-specific)."""
        config = HippocampusConfig(
            n_input=10, n_output=8,
            dg_size=10, ca3_size=6, ca1_size=8,
            stp_enabled=False,  # STP disabled
            device="cpu", dt_ms=1.0,
        )
        hippocampus = Hippocampus(config)

        # State with STP, but region has no STP modules
        state = HippocampusState(
            ca3_spikes=torch.zeros(6),
            ca1_spikes=torch.zeros(8),
            stp_mossy_state={"u": torch.rand(6, 10), "x": torch.rand(6, 10)},
        )

        # Should load without error, ignoring STP state
        hippocampus.load_state(state)

        # Verify basic state loaded
        assert torch.all(hippocampus.state.ca3_spikes == 0)
        assert torch.all(hippocampus.state.ca1_spikes == 0)

    def test_serialization_with_all_four_stp_pathways(self):
        """Test full serialization roundtrip with all 4 STP pathways."""
        stp_mossy = {"u": torch.rand(3, 4), "x": torch.rand(3, 4)}
        stp_schaffer = {"u": torch.rand(3, 3), "x": torch.rand(3, 3)}
        stp_ec_ca1 = {"u": torch.rand(3, 5), "x": torch.rand(3, 5)}
        stp_ca3_recurrent = {"u": torch.rand(3, 3), "x": torch.rand(3, 3)}

        state1 = HippocampusState(
            dg_spikes=torch.rand(4) > 0.5,
            ca3_spikes=torch.rand(3) > 0.5,
            ca1_spikes=torch.rand(3) > 0.5,
            ca3_persistent=torch.rand(3),
            stp_mossy_state=stp_mossy,
            stp_schaffer_state=stp_schaffer,
            stp_ec_ca1_state=stp_ec_ca1,
            stp_ca3_recurrent_state=stp_ca3_recurrent,
        )

        # Roundtrip
        data = state1.to_dict()
        state2 = HippocampusState.from_dict(data)

        # Verify all 4 STP pathways preserved
        assert torch.equal(state2.stp_mossy_state["u"], stp_mossy["u"])
        assert torch.equal(state2.stp_schaffer_state["u"], stp_schaffer["u"])
        assert torch.equal(state2.stp_ec_ca1_state["u"], stp_ec_ca1["u"])
        assert torch.equal(state2.stp_ca3_recurrent_state["u"], stp_ca3_recurrent["u"])
        # Verify CA3 persistent activity preserved
        assert torch.allclose(state2.ca3_persistent, state1.ca3_persistent)

    def test_save_cuda_load_cpu(self, tmp_path):
        """Test save on CUDA, load on CPU (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create state on CUDA
        state1 = HippocampusState(
            ca3_spikes=torch.zeros(3, device="cuda:0"),
            ca1_spikes=torch.zeros(3, device="cuda:0"),
            ca3_persistent=torch.rand(3, device="cuda:0"),
        )

        # Save to file
        filepath = tmp_path / "hippo_state.pt"
        save_region_state(state1, str(filepath))

        # Load to CPU
        state2 = load_region_state(HippocampusState, str(filepath), device="cpu")

        # Verify device transfer
        assert state2.ca3_spikes.device.type == "cpu"
        assert state2.ca3_persistent.device.type == "cpu"

        # Verify values match
        assert torch.equal(state2.ca3_spikes, state1.ca3_spikes.cpu())
        assert torch.allclose(state2.ca3_persistent, state1.ca3_persistent.cpu())
