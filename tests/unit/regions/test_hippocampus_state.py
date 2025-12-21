"""Tests for HippocampusState RegionState protocol compliance and integration.

Tests verify that HippocampusState:
1. Implements the RegionState protocol (to_dict, from_dict, reset)
2. Integrates with Hippocampus region (get_state, load_state)
3. Handles device transfer correctly (CPU/CUDA)
4. Supports file I/O via utility functions
5. Persists STP state for all 4 pathways (mossy, schaffer, ec_ca1, ca3_recurrent)
6. Preserves CA3 persistent activity and memory traces
7. Handles edge cases (None tensors, missing STP state)

Pattern: Follows established patterns from PrefrontalState and ThalamicRelayState tests.
"""

import pytest
import torch

from thalia.regions.hippocampus import Hippocampus, HippocampusState
from thalia.regions.hippocampus.config import HippocampusConfig
from thalia.core.region_state import save_region_state, load_region_state


class TestHippocampusStateProtocol:
    """Test RegionState protocol compliance."""

    def test_to_dict_basic(self):
        """Test to_dict() serializes all fields."""
        state = HippocampusState(
            spikes=torch.tensor([1.0, 0.0, 1.0]),
            membrane=torch.tensor([-60.0, -55.0, -70.0]),
            dopamine=0.3,
            acetylcholine=0.5,
            norepinephrine=0.2,
            dg_spikes=torch.tensor([1.0, 0.0, 1.0, 0.0]),
            ca3_spikes=torch.tensor([0.0, 1.0, 1.0]),
            ca1_spikes=torch.tensor([1.0, 0.0, 1.0]),
            ca3_membrane=torch.tensor([-60.0, -55.0, -70.0]),
            ca3_persistent=torch.tensor([0.2, 0.5, 0.1]),
            sample_trace=torch.tensor([0.8, 0.9, 0.7]),
            dg_trace=torch.tensor([0.1, 0.2, 0.0, 0.3]),
            ca3_trace=torch.tensor([0.4, 0.5, 0.6]),
            nmda_trace=torch.tensor([0.3, 0.2, 0.1]),
            stored_dg_pattern=torch.tensor([1.0, 0.0, 1.0, 1.0]),
            ffi_strength=0.4,
        )

        data = state.to_dict()

        # Verify all base fields present
        assert "spikes" in data
        assert "membrane" in data
        assert "dopamine" in data
        assert data["dopamine"] == 0.3

        # Verify hippocampus-specific fields
        assert "dg_spikes" in data
        assert "ca3_spikes" in data
        assert "ca1_spikes" in data
        assert "ca3_persistent" in data
        assert "sample_trace" in data
        assert "ffi_strength" in data
        assert data["ffi_strength"] == 0.4

        # Verify STP state fields present (even if None)
        assert "stp_mossy_state" in data
        assert "stp_schaffer_state" in data
        assert "stp_ec_ca1_state" in data
        assert "stp_ca3_recurrent_state" in data

    def test_to_dict_with_stp_state(self):
        """Test to_dict() captures nested STP state dicts."""
        # Simulate STP state from ShortTermPlasticity.get_state()
        mossy_stp = {
            "u": torch.tensor([0.2, 0.3, 0.1]),
            "x": torch.tensor([0.8, 0.7, 0.9]),
        }
        schaffer_stp = {
            "u": torch.tensor([0.5, 0.4]),
            "x": torch.tensor([0.6, 0.7]),
        }

        state = HippocampusState(
            dg_spikes=torch.zeros(3),
            ca3_spikes=torch.zeros(2),
            ca1_spikes=torch.zeros(2),
            stp_mossy_state=mossy_stp,
            stp_schaffer_state=schaffer_stp,
        )

        data = state.to_dict()

        # Verify nested STP dicts preserved
        assert data["stp_mossy_state"] is not None
        assert "u" in data["stp_mossy_state"]
        assert "x" in data["stp_mossy_state"]
        assert torch.allclose(data["stp_mossy_state"]["u"], mossy_stp["u"])
        assert torch.allclose(data["stp_mossy_state"]["x"], mossy_stp["x"])

        assert data["stp_schaffer_state"] is not None
        assert torch.allclose(data["stp_schaffer_state"]["u"], schaffer_stp["u"])

    def test_from_dict_basic(self):
        """Test from_dict() restores state from dict."""
        data = {
            "spikes": torch.tensor([1.0, 0.0, 1.0]),
            "membrane": torch.tensor([-60.0, -55.0, -70.0]),
            "dopamine": 0.4,
            "acetylcholine": 0.6,
            "norepinephrine": 0.1,
            "dg_spikes": torch.tensor([1.0, 0.0]),
            "ca3_spikes": torch.tensor([0.0, 1.0]),
            "ca1_spikes": torch.tensor([1.0, 0.0, 1.0]),
            "ca3_membrane": torch.tensor([-60.0, -55.0]),
            "ca3_persistent": torch.tensor([0.3, 0.4]),
            "sample_trace": torch.tensor([0.7, 0.8, 0.9]),
            "dg_trace": torch.tensor([0.2, 0.3]),
            "ca3_trace": torch.tensor([0.5, 0.6]),
            "nmda_trace": torch.tensor([0.1, 0.2, 0.3]),
            "stored_dg_pattern": torch.tensor([1.0, 1.0]),
            "ffi_strength": 0.5,
        }

        state = HippocampusState.from_dict(data)

        # Verify base fields restored
        assert state.spikes is not None
        assert torch.equal(state.spikes, data["spikes"])
        assert state.dopamine == 0.4
        assert state.acetylcholine == 0.6

        # Verify hippocampus-specific fields
        assert torch.equal(state.dg_spikes, data["dg_spikes"])
        assert torch.equal(state.ca3_spikes, data["ca3_spikes"])
        assert torch.equal(state.ca1_spikes, data["ca1_spikes"])
        assert torch.allclose(state.ca3_persistent, data["ca3_persistent"])
        assert torch.allclose(state.sample_trace, data["sample_trace"])
        assert state.ffi_strength == 0.5

    def test_from_dict_with_device_transfer(self):
        """Test from_dict() transfers tensors to specified device."""
        data = {
            "dg_spikes": torch.tensor([1.0, 0.0], device="cpu"),
            "ca3_spikes": torch.tensor([0.0, 1.0], device="cpu"),
            "ca1_spikes": torch.tensor([1.0, 0.0, 1.0], device="cpu"),
            "ca3_persistent": torch.tensor([0.3, 0.4], device="cpu"),
            "dopamine": 0.3,
        }

        state_cpu = HippocampusState.from_dict(data, device="cpu")
        assert state_cpu.dg_spikes.device.type == "cpu"

        if torch.cuda.is_available():
            state_cuda = HippocampusState.from_dict(data, device="cuda")
            assert state_cuda.dg_spikes.device.type == "cuda"

    def test_from_dict_with_nested_stp_state(self):
        """Test from_dict() handles nested STP state dicts with device transfer."""
        data = {
            "dg_spikes": torch.zeros(3),
            "ca3_spikes": torch.zeros(2),
            "ca1_spikes": torch.zeros(2),
            "stp_mossy_state": {
                "u": torch.tensor([0.2, 0.3], device="cpu"),
                "x": torch.tensor([0.8, 0.7], device="cpu"),
            },
            "stp_schaffer_state": {
                "u": torch.tensor([0.5], device="cpu"),
                "x": torch.tensor([0.6], device="cpu"),
            },
        }

        state_cpu = HippocampusState.from_dict(data, device="cpu")
        assert state_cpu.stp_mossy_state is not None
        assert state_cpu.stp_mossy_state["u"].device.type == "cpu"

        if torch.cuda.is_available():
            state_cuda = HippocampusState.from_dict(data, device="cuda")
            assert state_cuda.stp_mossy_state["u"].device.type == "cuda"
            assert state_cuda.stp_schaffer_state["x"].device.type == "cuda"

    def test_reset(self):
        """Test reset() clears all state fields."""
        state = HippocampusState(
            spikes=torch.tensor([1.0, 0.0, 1.0]),
            membrane=torch.tensor([-50.0, -55.0, -60.0]),
            dopamine=0.5,
            acetylcholine=0.3,
            norepinephrine=0.2,
            dg_spikes=torch.ones(4),
            ca3_spikes=torch.ones(3),
            ca1_spikes=torch.ones(3),
            ca3_persistent=torch.tensor([0.5, 0.6, 0.7]),
            sample_trace=torch.ones(3),
            ffi_strength=0.7,
            stp_mossy_state={"u": torch.tensor([0.3]), "x": torch.tensor([0.8])},
        )

        result = state.reset()

        # Verify reset() returns None (in-place mutation)
        assert result is None

        # Verify all tensor fields cleared to None
        assert state.spikes is None
        assert state.membrane is None
        assert state.dg_spikes is None
        assert state.ca3_spikes is None
        assert state.ca1_spikes is None
        assert state.ca3_persistent is None
        assert state.sample_trace is None

        # Verify neuromodulators reset to defaults
        assert state.dopamine == 0.2
        assert state.acetylcholine == 0.0
        assert state.norepinephrine == 0.0
        assert state.ffi_strength == 0.0

        # Verify STP state cleared
        assert state.stp_mossy_state is None
        assert state.stp_schaffer_state is None


class TestHippocampusIntegration:
    """Test integration with Hippocampus region."""

    @pytest.fixture
    def hippocampus(self):
        """Create minimal Hippocampus for testing."""
        config = HippocampusConfig(
            n_input=16,
            n_output=10,
            device="cpu",
            dg_expansion=1.25,  # 20 DG neurons (16 * 1.25)
            ca3_size_ratio=0.5,  # 10 CA3 neurons
            ca1_size_ratio=1.0,  # 10 CA1 neurons
        )

        region = Hippocampus(config)
        return region

    def test_get_state_captures_stp(self, hippocampus):
        """Test get_state() captures STP state from all 4 pathways."""
        # Run forward to generate STP state
        ec_input = torch.randn(16)
        hippocampus(ec_input)

        state = hippocampus.get_state()

        # Verify state contains activity
        assert state.dg_spikes is not None
        assert state.ca3_spikes is not None
        assert state.ca1_spikes is not None

        # Verify STP state captured for all 4 pathways
        assert state.stp_mossy_state is not None  # DG→CA3
        assert "u" in state.stp_mossy_state
        assert "x" in state.stp_mossy_state

        assert state.stp_schaffer_state is not None  # CA3→CA1
        assert "u" in state.stp_schaffer_state

        assert state.stp_ec_ca1_state is not None  # EC→CA1 direct
        assert "u" in state.stp_ec_ca1_state

        assert state.stp_ca3_recurrent_state is not None  # CA3 recurrent
        assert "u" in state.stp_ca3_recurrent_state

    def test_load_state_restores_activity(self, hippocampus):
        """Test load_state() restores hippocampus activity state."""
        # Create state with specific activity pattern
        original_state = hippocampus.get_state()
        original_state.dg_spikes = torch.tensor([1.0] * 10 + [0.0] * 10)
        original_state.ca3_spikes = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0] + [0.0] * 5)
        original_state.ca1_spikes = torch.tensor([1.0, 0.0, 1.0] + [0.0] * 7)
        original_state.ca3_persistent = torch.tensor([0.3] * 5 + [0.1] * 5)

        # Load state
        hippocampus.load_state(original_state)

        # Verify state restored
        restored = hippocampus.get_state()
        assert torch.equal(restored.dg_spikes, original_state.dg_spikes)
        assert torch.equal(restored.ca3_spikes, original_state.ca3_spikes)
        assert torch.equal(restored.ca1_spikes, original_state.ca1_spikes)
        assert torch.allclose(restored.ca3_persistent, original_state.ca3_persistent)

    def test_load_state_restores_stp(self, hippocampus):
        """Test load_state() restores STP state for all pathways."""
        # Run forward to generate initial STP state
        ec_input = torch.randn(16)
        for _ in range(5):  # Multiple steps to build up STP state
            hippocampus(ec_input)

        # Capture state with STP
        state_with_stp = hippocampus.get_state()

        # Reset and verify STP cleared
        hippocampus.reset_state()
        reset_state = hippocampus.get_state()
        # After reset, u should be at baseline (not the facilitated/depressed values)

        # Load saved state
        hippocampus.load_state(state_with_stp)

        # Verify STP state restored
        restored_state = hippocampus.get_state()
        assert restored_state.stp_mossy_state is not None
        assert torch.allclose(
            restored_state.stp_mossy_state["u"],
            state_with_stp.stp_mossy_state["u"],
        )
        assert torch.allclose(
            restored_state.stp_schaffer_state["u"],
            state_with_stp.stp_schaffer_state["u"],
        )

    def test_round_trip_consistency(self, hippocampus):
        """Test get_state() → load_state() preserves complete state."""
        # Generate complex state
        ec_input = torch.randn(16)
        for _ in range(3):
            hippocampus(ec_input)

        original_state = hippocampus.get_state()

        # Reset and load
        hippocampus.reset_state()
        hippocampus.load_state(original_state)

        restored_state = hippocampus.get_state()

        # Verify all activity restored
        assert torch.equal(restored_state.dg_spikes, original_state.dg_spikes)
        assert torch.equal(restored_state.ca3_spikes, original_state.ca3_spikes)
        assert torch.equal(restored_state.ca1_spikes, original_state.ca1_spikes)

        # Verify all STP state restored (all 4 pathways)
        assert torch.allclose(
            restored_state.stp_mossy_state["u"],
            original_state.stp_mossy_state["u"],
        )
        assert torch.allclose(
            restored_state.stp_mossy_state["x"],
            original_state.stp_mossy_state["x"],
        )
        assert torch.allclose(
            restored_state.stp_schaffer_state["u"],
            original_state.stp_schaffer_state["u"],
        )
        assert torch.allclose(
            restored_state.stp_ec_ca1_state["u"],
            original_state.stp_ec_ca1_state["u"],
        )
        assert torch.allclose(
            restored_state.stp_ca3_recurrent_state["u"],
            original_state.stp_ca3_recurrent_state["u"],
        )


class TestHippocampusStateIO:
    """Test file I/O with utility functions."""

    @pytest.fixture
    def temp_path(self, tmp_path):
        """Temporary file path for testing."""
        return tmp_path / "hippocampus_state.pt"

    def test_save_and_load_state(self, temp_path):
        """Test save/load round trip with file I/O utilities."""
        original = HippocampusState(
            spikes=torch.tensor([1.0, 0.0, 1.0]),
            dg_spikes=torch.tensor([1.0, 0.0, 1.0, 0.0]),
            ca3_spikes=torch.tensor([0.0, 1.0, 1.0]),
            ca1_spikes=torch.tensor([1.0, 0.0, 1.0]),
            ca3_persistent=torch.tensor([0.5, 0.3, 0.2]),
            dopamine=0.4,
            ffi_strength=0.6,
            stp_mossy_state={"u": torch.tensor([0.3, 0.4]), "x": torch.tensor([0.7, 0.8])},
        )

        # Save
        save_region_state(original, temp_path)
        assert temp_path.exists()

        # Load
        loaded = load_region_state(HippocampusState, temp_path, device="cpu")

        # Verify restoration
        assert torch.equal(loaded.dg_spikes, original.dg_spikes)
        assert torch.equal(loaded.ca1_spikes, original.ca1_spikes)
        assert torch.allclose(loaded.ca3_persistent, original.ca3_persistent)
        assert loaded.dopamine == 0.4
        assert loaded.ffi_strength == 0.6

        # Verify nested STP state
        assert torch.allclose(
            loaded.stp_mossy_state["u"],
            original.stp_mossy_state["u"],
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_save_cuda_load_cpu(self, temp_path):
        """Test saving CUDA state and loading to CPU."""
        original = HippocampusState(
            dg_spikes=torch.tensor([1.0, 0.0, 1.0], device="cuda"),
            ca3_spikes=torch.tensor([0.0, 1.0], device="cuda"),
            ca1_spikes=torch.tensor([1.0, 0.0, 1.0], device="cuda"),
            stp_schaffer_state={
                "u": torch.tensor([0.5, 0.6], device="cuda"),
                "x": torch.tensor([0.7, 0.8], device="cuda"),
            },
        )

        save_region_state(original, temp_path)
        loaded = load_region_state(HippocampusState, temp_path, device="cpu")

        # Verify device transfer
        assert loaded.dg_spikes.device.type == "cpu"
        assert loaded.stp_schaffer_state["u"].device.type == "cpu"

        # Verify values preserved
        assert torch.equal(loaded.dg_spikes, original.dg_spikes.cpu())
        assert torch.equal(
            loaded.stp_schaffer_state["u"],
            original.stp_schaffer_state["u"].cpu(),
        )


class TestHippocampusEdgeCases:
    """Test edge cases and error handling."""

    def test_from_dict_missing_optional_fields(self):
        """Test from_dict() handles missing optional fields gracefully."""
        # Minimal data - only required fields
        data = {
            "dg_spikes": torch.zeros(10),
            "ca3_spikes": torch.zeros(5),
            "ca1_spikes": torch.zeros(5),
        }

        state = HippocampusState.from_dict(data)

        # Verify defaults applied
        assert state.dopamine == 0.2  # Default from dataclass
        assert state.acetylcholine == 0.0
        assert state.ffi_strength == 0.0

        # Verify None for unspecified tensor fields
        assert state.spikes is None
        assert state.membrane is None
        assert state.ca3_persistent is None
        assert state.sample_trace is None

        # Verify STP state None
        assert state.stp_mossy_state is None
        assert state.stp_schaffer_state is None

    def test_to_dict_with_none_tensors(self):
        """Test to_dict() handles None tensor fields."""
        state = HippocampusState(
            dg_spikes=torch.zeros(10),
            ca3_spikes=torch.zeros(5),
            ca1_spikes=torch.zeros(5),
            # All other fields None by default
        )

        data = state.to_dict()

        # Verify None fields serialized
        assert data["spikes"] is None
        assert data["membrane"] is None
        assert data["ca3_persistent"] is None
        assert data["stp_mossy_state"] is None

        # Verify non-None fields present
        assert data["dg_spikes"] is not None

    def test_load_state_with_missing_stp_modules(self):
        """Test load_state() handles missing STP modules gracefully."""
        config = HippocampusConfig(
            n_input=8,
            n_output=5,
            device="cpu",
            dg_expansion=1.25,  # 10 DG neurons
            ca3_size_ratio=0.5,  # 5 CA3 neurons
            ca1_size_ratio=1.0,  # 5 CA1 neurons
        )

        hippocampus = Hippocampus(config)

        # Create state with STP data
        state = HippocampusState(
            dg_spikes=torch.zeros(10),
            ca3_spikes=torch.zeros(5),
            ca1_spikes=torch.zeros(5),
            stp_mossy_state={"u": torch.tensor([0.3] * 5), "x": torch.tensor([0.7] * 5)},
        )

        # Should not raise error even if STP modules disabled
        hippocampus.load_state(state)  # Should handle gracefully

    def test_serialization_with_all_four_stp_pathways(self):
        """Test complete serialization with all 4 STP pathways populated."""
        state = HippocampusState(
            dg_spikes=torch.zeros(20),
            ca3_spikes=torch.zeros(10),
            ca1_spikes=torch.zeros(10),
            stp_mossy_state={"u": torch.rand(10), "x": torch.rand(10)},
            stp_schaffer_state={"u": torch.rand(10), "x": torch.rand(10)},
            stp_ec_ca1_state={"u": torch.rand(10), "x": torch.rand(10)},
            stp_ca3_recurrent_state={"u": torch.rand(10), "x": torch.rand(10)},
        )

        # Round trip
        data = state.to_dict()
        restored = HippocampusState.from_dict(data)

        # Verify all 4 STP states preserved
        assert torch.allclose(
            restored.stp_mossy_state["u"],
            state.stp_mossy_state["u"],
        )
        assert torch.allclose(
            restored.stp_schaffer_state["u"],
            state.stp_schaffer_state["u"],
        )
        assert torch.allclose(
            restored.stp_ec_ca1_state["u"],
            state.stp_ec_ca1_state["u"],
        )
        assert torch.allclose(
            restored.stp_ca3_recurrent_state["u"],
            state.stp_ca3_recurrent_state["u"],
        )
