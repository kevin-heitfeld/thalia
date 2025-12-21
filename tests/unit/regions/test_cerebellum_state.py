"""Tests for Cerebellum RegionState protocol implementation.

Tests the CerebellumState dataclass implementation of the RegionState protocol,
including serialization, device transfer, and integration with the Cerebellum region.

Phase 3.1 of state management refactoring.
"""

import pytest
import torch

from thalia.regions.cerebellum_region import Cerebellum, CerebellumConfig, CerebellumState
from thalia.core.region_state import save_region_state, load_region_state


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def device() -> str:
    """Test device (CPU)."""
    return "cpu"


@pytest.fixture
def cerebellum_config(device: str) -> CerebellumConfig:
    """Create minimal Cerebellum configuration."""
    return CerebellumConfig(
        n_input=20,
        n_output=10,
        use_enhanced_microcircuit=False,  # Classic mode for simpler testing
        stp_enabled=True,
        device=device,
        dt_ms=1.0,
    )


@pytest.fixture
def cerebellum_region(cerebellum_config: CerebellumConfig) -> Cerebellum:
    """Create Cerebellum region instance."""
    region = Cerebellum(cerebellum_config)
    return region


@pytest.fixture
def sample_state(device: str) -> CerebellumState:
    """Create a sample CerebellumState with non-zero values."""
    return CerebellumState(
        # Trace manager state
        input_trace=torch.rand(20, device=device),
        output_trace=torch.rand(10, device=device),
        stdp_eligibility=torch.rand(10, 20, device=device),

        # Climbing fiber error
        climbing_fiber_error=torch.rand(10, device=device),

        # Neuron state (classic mode)
        v_mem=torch.rand(10, device=device) * (-70.0) + (-70.0),
        g_exc=torch.rand(10, device=device) * 0.1,
        g_inh=torch.rand(10, device=device) * 0.05,

        # STP state
        stp_pf_purkinje_state={
            "u": torch.rand(10, 20, device=device) * 0.5,
            "x": torch.ones(10, 20, device=device),
        },

        # Neuromodulators
        dopamine=0.5,
        acetylcholine=0.3,
        norepinephrine=0.2,

        # Enhanced microcircuit (None for classic mode)
        granule_layer_state=None,
        purkinje_cells_state=None,
        deep_nuclei_state=None,
        stp_mf_granule_state=None,
    )


# ============================================================================
# TEST PROTOCOL COMPLIANCE
# ============================================================================


class TestCerebellumStateProtocol:
    """Test CerebellumState implements RegionState protocol."""

    def test_to_dict_basic(self, sample_state: CerebellumState):
        """Test to_dict() serializes all fields."""
        state_dict = sample_state.to_dict()

        # Check required keys
        assert "input_trace" in state_dict
        assert "output_trace" in state_dict
        assert "stdp_eligibility" in state_dict
        assert "climbing_fiber_error" in state_dict
        assert "v_mem" in state_dict
        assert "g_exc" in state_dict
        assert "g_inh" in state_dict
        assert "dopamine" in state_dict
        assert "acetylcholine" in state_dict
        assert "norepinephrine" in state_dict
        assert "stp_pf_purkinje_state" in state_dict

        # Check tensor serialization
        assert isinstance(state_dict["input_trace"], torch.Tensor)
        assert isinstance(state_dict["output_trace"], torch.Tensor)
        assert isinstance(state_dict["stdp_eligibility"], torch.Tensor)

        # Check nested dict serialization
        assert isinstance(state_dict["stp_pf_purkinje_state"], dict)
        assert "u" in state_dict["stp_pf_purkinje_state"]
        assert "x" in state_dict["stp_pf_purkinje_state"]

    def test_from_dict_basic(self, sample_state: CerebellumState, device: str):
        """Test from_dict() deserializes correctly."""
        state_dict = sample_state.to_dict()
        restored_state = CerebellumState.from_dict(state_dict, device=device)

        # Check tensors restored
        assert torch.allclose(restored_state.input_trace, sample_state.input_trace)
        assert torch.allclose(restored_state.output_trace, sample_state.output_trace)
        assert torch.allclose(restored_state.stdp_eligibility, sample_state.stdp_eligibility)
        assert torch.allclose(restored_state.climbing_fiber_error, sample_state.climbing_fiber_error)

        # Check scalars restored
        assert restored_state.dopamine == sample_state.dopamine
        assert restored_state.acetylcholine == sample_state.acetylcholine
        assert restored_state.norepinephrine == sample_state.norepinephrine

    def test_reset_clears_state(self, sample_state: CerebellumState):
        """Test reset() clears all state tensors in-place."""
        sample_state.reset()

        # Check traces cleared
        assert torch.allclose(sample_state.input_trace, torch.zeros_like(sample_state.input_trace))
        assert torch.allclose(sample_state.output_trace, torch.zeros_like(sample_state.output_trace))
        assert torch.allclose(sample_state.stdp_eligibility, torch.zeros_like(sample_state.stdp_eligibility))

        # Check error cleared
        assert torch.allclose(sample_state.climbing_fiber_error, torch.zeros_like(sample_state.climbing_fiber_error))

        # Check neuron state cleared
        assert torch.all(sample_state.v_mem == -70.0)  # Reset to resting potential
        assert torch.allclose(sample_state.g_exc, torch.zeros_like(sample_state.g_exc))
        assert torch.allclose(sample_state.g_inh, torch.zeros_like(sample_state.g_inh))

        # Check STP state cleared
        if sample_state.stp_pf_purkinje_state is not None:
            assert torch.allclose(
                sample_state.stp_pf_purkinje_state["x"],
                torch.ones_like(sample_state.stp_pf_purkinje_state["x"])
            )

    def test_device_transfer_cpu_to_cpu(self, sample_state: CerebellumState):
        """Test device transfer from CPU to CPU (no-op)."""
        state_dict = sample_state.to_dict()
        restored_state = CerebellumState.from_dict(state_dict, device="cpu")

        assert restored_state.input_trace.device.type == "cpu"
        assert restored_state.output_trace.device.type == "cpu"
        assert torch.allclose(restored_state.input_trace, sample_state.input_trace)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_transfer_cpu_to_cuda(self, sample_state: CerebellumState):
        """Test device transfer from CPU to CUDA."""
        state_dict = sample_state.to_dict()
        restored_state = CerebellumState.from_dict(state_dict, device="cuda")

        assert restored_state.input_trace.device.type == "cuda"
        assert restored_state.output_trace.device.type == "cuda"
        assert torch.allclose(restored_state.input_trace.cpu(), sample_state.input_trace)

    def test_roundtrip_serialization(self, sample_state: CerebellumState, device: str):
        """Test serialize → deserialize → serialize produces identical results."""
        dict1 = sample_state.to_dict()
        restored = CerebellumState.from_dict(dict1, device=device)
        dict2 = restored.to_dict()

        # Check all tensor keys match
        for key in ["input_trace", "output_trace", "stdp_eligibility", "climbing_fiber_error"]:
            assert torch.allclose(dict1[key], dict2[key])

        # Check scalar keys match
        assert dict1["dopamine"] == dict2["dopamine"]
        assert dict1["acetylcholine"] == dict2["acetylcholine"]
        assert dict1["norepinephrine"] == dict2["norepinephrine"]


# ============================================================================
# TEST INTEGRATION WITH CEREBELLUM REGION
# ============================================================================


class TestCerebellumStateIntegration:
    """Test CerebellumState integration with Cerebellum region."""

    def test_get_state_returns_cerebellum_state(self, cerebellum_region: Cerebellum):
        """Test get_state() returns CerebellumState instance."""
        state = cerebellum_region.get_state()

        assert isinstance(state, CerebellumState)
        assert state.input_trace is not None
        assert state.output_trace is not None
        assert state.stdp_eligibility is not None

    def test_load_state_restores_traces(self, cerebellum_region: Cerebellum, sample_state: CerebellumState):
        """Test load_state() restores all state correctly."""
        # Store original state
        original_input_trace = cerebellum_region.input_trace.clone()

        # Load new state
        cerebellum_region.load_state(sample_state)

        # Verify state changed
        assert not torch.allclose(cerebellum_region.input_trace, original_input_trace)
        assert torch.allclose(cerebellum_region.input_trace, sample_state.input_trace)
        assert torch.allclose(cerebellum_region.output_trace, sample_state.output_trace)

    def test_state_roundtrip_through_region(self, cerebellum_region: Cerebellum):
        """Test get_state() → load_state() roundtrip preserves state."""
        # Run forward pass to generate non-zero state
        input_spikes = torch.rand(20) > 0.5
        cerebellum_region.forward({"default": input_spikes.float()})

        # Get state
        state = cerebellum_region.get_state()

        # Reset region
        cerebellum_region.reset_state()

        # Load state back
        cerebellum_region.load_state(state)

        # Verify state restored
        restored_state = cerebellum_region.get_state()
        assert torch.allclose(restored_state.input_trace, state.input_trace)
        assert torch.allclose(restored_state.output_trace, state.output_trace)

    def test_state_device_consistency(self, cerebellum_region: Cerebellum):
        """Test state tensors are on correct device."""
        state = cerebellum_region.get_state()

        assert state.input_trace.device.type == cerebellum_region.device.type
        assert state.output_trace.device.type == cerebellum_region.device.type
        assert state.stdp_eligibility.device.type == cerebellum_region.device.type


# ============================================================================
# TEST FILE I/O WITH UTILITIES
# ============================================================================


class TestCerebellumStateFileIO:
    """Test CerebellumState file I/O with save/load utilities."""

    def test_save_and_load_state(self, sample_state: CerebellumState, tmp_path, device: str):
        """Test save_region_state() and load_region_state() utilities."""
        save_path = tmp_path / "cerebellum_state.pt"

        # Save state
        save_region_state(sample_state, save_path)
        assert save_path.exists()

        # Load state
        loaded_state = load_region_state(CerebellumState, save_path, device=device)

        # Verify state matches
        assert isinstance(loaded_state, CerebellumState)
        assert torch.allclose(loaded_state.input_trace, sample_state.input_trace)
        assert torch.allclose(loaded_state.output_trace, sample_state.output_trace)
        assert loaded_state.dopamine == sample_state.dopamine

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_save_cpu_load_cuda(self, sample_state: CerebellumState, tmp_path):
        """Test saving on CPU and loading on CUDA."""
        save_path = tmp_path / "cerebellum_state_cpu.pt"

        # Save CPU state
        save_region_state(sample_state, save_path)

        # Load on CUDA
        loaded_state = load_region_state(CerebellumState, save_path, device="cuda")

        assert loaded_state.input_trace.device.type == "cuda"
        assert torch.allclose(loaded_state.input_trace.cpu(), sample_state.input_trace)


# ============================================================================
# TEST EDGE CASES
# ============================================================================


class TestCerebellumStateEdgeCases:
    """Test edge cases and error handling."""

    def test_none_optional_fields(self, device: str):
        """Test state with None optional fields (enhanced microcircuit disabled)."""
        state = CerebellumState(
            input_trace=torch.zeros(20, device=device),
            output_trace=torch.zeros(10, device=device),
            stdp_eligibility=torch.zeros(10, 20, device=device),
            climbing_fiber_error=torch.zeros(10, device=device),
            v_mem=torch.full((10,), -70.0, device=device),
            g_exc=torch.zeros(10, device=device),
            g_inh=torch.zeros(10, device=device),
            stp_pf_purkinje_state=None,  # STP disabled
            dopamine=0.0,
            acetylcholine=0.0,
            norepinephrine=0.0,
            granule_layer_state=None,
            purkinje_cells_state=None,
            deep_nuclei_state=None,
            stp_mf_granule_state=None,
        )

        # Should serialize/deserialize correctly
        state_dict = state.to_dict()
        assert state_dict["stp_pf_purkinje_state"] is None

        restored = CerebellumState.from_dict(state_dict, device=device)
        assert restored.stp_pf_purkinje_state is None

    def test_enhanced_microcircuit_state(self, device: str):
        """Test state with enhanced microcircuit fields populated."""
        state = CerebellumState(
            input_trace=torch.zeros(20, device=device),
            output_trace=torch.zeros(10, device=device),
            stdp_eligibility=torch.zeros(10, 80, device=device),  # Expanded for granule layer
            climbing_fiber_error=torch.zeros(10, device=device),
            v_mem=None,  # No classic neurons
            g_exc=None,
            g_inh=None,
            stp_pf_purkinje_state=None,
            dopamine=0.5,
            acetylcholine=0.3,
            norepinephrine=0.2,
            granule_layer_state={"granule_spikes": torch.zeros(80, device=device)},
            purkinje_cells_state=[{"v_mem": torch.tensor(-70.0, device=device)} for _ in range(10)],
            deep_nuclei_state={"dcn_activity": torch.zeros(10, device=device)},
            stp_mf_granule_state={"u": torch.rand(80, 20, device=device)},
        )

        # Should serialize/deserialize correctly
        state_dict = state.to_dict()
        assert state_dict["granule_layer_state"] is not None
        assert state_dict["purkinje_cells_state"] is not None

        restored = CerebellumState.from_dict(state_dict, device=device)
        assert restored.granule_layer_state is not None
        assert len(restored.purkinje_cells_state) == 10

    def test_empty_traces(self, device: str):
        """Test state with zero-initialized traces."""
        state = CerebellumState(
            input_trace=torch.zeros(20, device=device),
            output_trace=torch.zeros(10, device=device),
            stdp_eligibility=torch.zeros(10, 20, device=device),
            climbing_fiber_error=torch.zeros(10, device=device),
            v_mem=torch.full((10,), -70.0, device=device),
            g_exc=torch.zeros(10, device=device),
            g_inh=torch.zeros(10, device=device),
            stp_pf_purkinje_state=None,
            dopamine=0.0,
            acetylcholine=0.0,
            norepinephrine=0.0,
        )

        # Should handle all-zero tensors correctly
        state_dict = state.to_dict()
        restored = CerebellumState.from_dict(state_dict, device=device)

        assert torch.allclose(restored.input_trace, torch.zeros(20, device=device))
        assert torch.allclose(restored.output_trace, torch.zeros(10, device=device))

    def test_2d_traces_preserved(self, device: str):
        """Test 2D eligibility trace shape preservation."""
        eligibility = torch.rand(10, 20, device=device)

        state = CerebellumState(
            input_trace=torch.zeros(20, device=device),
            output_trace=torch.zeros(10, device=device),
            stdp_eligibility=eligibility,
            climbing_fiber_error=torch.zeros(10, device=device),
            v_mem=torch.full((10,), -70.0, device=device),
            g_exc=torch.zeros(10, device=device),
            g_inh=torch.zeros(10, device=device),
            stp_pf_purkinje_state=None,
            dopamine=0.0,
            acetylcholine=0.0,
            norepinephrine=0.0,
        )

        # Check shape preserved through roundtrip
        state_dict = state.to_dict()
        restored = CerebellumState.from_dict(state_dict, device=device)

        assert restored.stdp_eligibility.shape == (10, 20)
        assert torch.allclose(restored.stdp_eligibility, eligibility)
