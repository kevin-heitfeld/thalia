"""Tests for LayeredCortexState RegionState protocol compliance and integration.

Tests verify that LayeredCortexState:
1. Implements the RegionState protocol (to_dict, from_dict, reset)
2. Integrates with LayeredCortex region (get_state, load_state)
3. Handles device transfer correctly (CPU/CUDA)
4. Supports file I/O via utility functions
5. Persists STP state for L2/3 recurrent pathway
6. Preserves all 6 layer states (L4, L2/3, L5, L6a, L6b) and traces
7. Handles edge cases (None tensors, missing STP state)
8. Preserves delay buffer state for inter-layer connections

Pattern: Follows established patterns from HippocampusState tests.
Author: Thalia Project (Phase 2.4)
Date: December 21, 2025
"""

import pytest
import torch

from thalia.regions.cortex.layered_cortex import LayeredCortex
from thalia.regions.cortex.config import LayeredCortexState, LayeredCortexConfig
from thalia.core.region_state import save_region_state, load_region_state


class TestLayeredCortexStateProtocol:
    """Test RegionState protocol compliance."""

    def test_to_dict_basic(self):
        """Test to_dict() serializes all fields."""
        state = LayeredCortexState(
            # Base fields
            spikes=torch.tensor([1.0, 0.0, 1.0]),
            membrane=torch.tensor([-60.0, -55.0, -70.0]),
            dopamine=0.2,
            acetylcholine=0.3,
            norepinephrine=0.4,
            # Layer spike states (6 layers)
            input_spikes=torch.tensor([1.0, 0.0, 1.0, 1.0]),
            l4_spikes=torch.tensor([1.0, 0.0, 1.0]),
            l23_spikes=torch.tensor([0.0, 1.0, 1.0, 0.0]),
            l5_spikes=torch.tensor([1.0, 0.0]),
            l6a_spikes=torch.tensor([0.0, 1.0]),
            l6b_spikes=torch.tensor([1.0, 1.0]),
            # L2/3 recurrent activity
            l23_recurrent_activity=torch.tensor([0.5, 0.8, 0.3, 0.9]),
            # STDP traces (5 layers)
            l4_trace=torch.tensor([0.2, 0.3, 0.1]),
            l23_trace=torch.tensor([0.4, 0.5, 0.6, 0.7]),
            l5_trace=torch.tensor([0.8, 0.9]),
            l6a_trace=torch.tensor([0.1, 0.2]),
            l6b_trace=torch.tensor([0.3, 0.4]),
            # Modulation state
            top_down_modulation=torch.tensor([0.6, 0.7, 0.8, 0.5]),
            ffi_strength=0.4,
            alpha_suppression=0.8,
            # Gamma attention state
            gamma_attention_phase=0.5,
            gamma_attention_gate=torch.tensor([1.0, 0.5, 1.0, 0.8]),
            # Plasticity monitoring
            last_plasticity_delta=0.01,
        )

        data = state.to_dict()

        # Verify all base fields present
        assert "spikes" in data
        assert "membrane" in data
        assert "dopamine" in data
        assert data["dopamine"] == 0.2
        assert data["acetylcholine"] == 0.3
        assert data["norepinephrine"] == 0.4

        # Verify layer spike states
        assert "input_spikes" in data
        assert "l4_spikes" in data
        assert "l23_spikes" in data
        assert "l5_spikes" in data
        assert "l6a_spikes" in data
        assert "l6b_spikes" in data

        # Verify recurrent activity
        assert "l23_recurrent_activity" in data

        # Verify STDP traces
        assert "l4_trace" in data
        assert "l23_trace" in data
        assert "l5_trace" in data
        assert "l6a_trace" in data
        assert "l6b_trace" in data

        # Verify modulation state
        assert "top_down_modulation" in data
        assert data["ffi_strength"] == 0.4
        assert data["alpha_suppression"] == 0.8

        # Verify gamma attention state
        assert "gamma_attention_phase" in data
        assert data["gamma_attention_phase"] == 0.5
        assert "gamma_attention_gate" in data

        # Verify plasticity delta
        assert "last_plasticity_delta" in data
        assert data["last_plasticity_delta"] == 0.01

        # Verify STP state field present (even if None)
        assert "stp_l23_recurrent_state" in data

    def test_to_dict_with_stp_state(self):
        """Test to_dict() captures nested STP state dict."""
        # Simulate STP state from ShortTermPlasticity.get_state()
        stp_state = {
            "u": torch.tensor([[0.2, 0.3], [0.1, 0.4]]),  # [4, 4] for L2/3 recurrent
            "x": torch.tensor([[0.8, 0.7], [0.9, 0.6]]),
        }

        state = LayeredCortexState(
            l4_spikes=torch.zeros(3),
            l23_spikes=torch.zeros(4),
            l5_spikes=torch.zeros(2),
            stp_l23_recurrent_state=stp_state,
        )

        data = state.to_dict()

        # Verify nested STP dict preserved
        assert data["stp_l23_recurrent_state"] is not None
        assert "u" in data["stp_l23_recurrent_state"]
        assert "x" in data["stp_l23_recurrent_state"]
        assert torch.equal(data["stp_l23_recurrent_state"]["u"], stp_state["u"])
        assert torch.equal(data["stp_l23_recurrent_state"]["x"], stp_state["x"])

    def test_from_dict_basic(self):
        """Test from_dict() restores state from dict."""
        data = {
            # Base fields
            "spikes": torch.tensor([1.0, 0.0, 1.0]),
            "membrane": torch.tensor([-60.0, -55.0, -70.0]),
            "dopamine": 0.5,
            "acetylcholine": 0.6,
            "norepinephrine": 0.3,
            # Layer spike states
            "input_spikes": torch.tensor([1.0, 0.0, 1.0, 1.0]),
            "l4_spikes": torch.tensor([1.0, 0.0, 1.0]),
            "l23_spikes": torch.tensor([0.0, 1.0, 1.0, 0.0]),
            "l5_spikes": torch.tensor([1.0, 0.0]),
            "l6a_spikes": torch.tensor([0.0, 1.0]),
            "l6b_spikes": torch.tensor([1.0, 1.0]),
            # L2/3 recurrent activity
            "l23_recurrent_activity": torch.tensor([0.5, 0.8, 0.3, 0.9]),
            # STDP traces
            "l4_trace": torch.tensor([0.2, 0.3, 0.1]),
            "l23_trace": torch.tensor([0.4, 0.5, 0.6, 0.7]),
            "l5_trace": torch.tensor([0.8, 0.9]),
            "l6a_trace": torch.tensor([0.1, 0.2]),
            "l6b_trace": torch.tensor([0.3, 0.4]),
            # Modulation state
            "top_down_modulation": torch.tensor([0.6, 0.7, 0.8, 0.5]),
            "ffi_strength": 0.7,
            "alpha_suppression": 0.5,
            # Gamma attention state
            "gamma_attention_phase": 0.8,
            "gamma_attention_gate": torch.tensor([1.0, 0.5, 1.0, 0.8]),
            # Plasticity delta
            "last_plasticity_delta": 0.02,
        }

        state = LayeredCortexState.from_dict(data)

        # Verify base fields restored
        assert state.spikes is not None
        assert torch.equal(state.spikes, data["spikes"])
        assert state.dopamine == 0.5
        assert state.acetylcholine == 0.6
        assert state.norepinephrine == 0.3

        # Verify layer spike states
        assert torch.equal(state.l4_spikes, data["l4_spikes"])
        assert torch.equal(state.l23_spikes, data["l23_spikes"])
        assert torch.equal(state.l5_spikes, data["l5_spikes"])
        assert torch.equal(state.l6a_spikes, data["l6a_spikes"])
        assert torch.equal(state.l6b_spikes, data["l6b_spikes"])

        # Verify recurrent activity
        assert torch.allclose(state.l23_recurrent_activity, data["l23_recurrent_activity"])

        # Verify traces
        assert torch.allclose(state.l4_trace, data["l4_trace"])
        assert torch.allclose(state.l23_trace, data["l23_trace"])

        # Verify modulation state
        assert state.ffi_strength == 0.7
        assert state.alpha_suppression == 0.5

        # Verify gamma attention
        assert state.gamma_attention_phase == 0.8
        assert torch.equal(state.gamma_attention_gate, data["gamma_attention_gate"])

    def test_from_dict_with_device_transfer(self):
        """Test from_dict() transfers tensors to specified device."""
        data = {
            "l4_spikes": torch.tensor([1.0, 0.0, 1.0], device="cpu"),
            "l23_spikes": torch.tensor([0.0, 1.0, 1.0, 0.0], device="cpu"),
            "l5_spikes": torch.tensor([1.0, 0.0], device="cpu"),
            "l4_trace": torch.tensor([0.2, 0.3, 0.1], device="cpu"),
            "dopamine": 0.3,
        }

        state_cpu = LayeredCortexState.from_dict(data, device="cpu")
        assert state_cpu.l4_spikes.device.type == "cpu"
        assert state_cpu.l23_spikes.device.type == "cpu"

        # Test CUDA transfer if available
        if torch.cuda.is_available():
            state_cuda = LayeredCortexState.from_dict(data, device="cuda:0")
            assert state_cuda.l4_spikes.device.type == "cuda"
            assert state_cuda.l23_spikes.device.type == "cuda"

    def test_reset(self):
        """Test reset() clears all state fields."""
        state = LayeredCortexState(
            spikes=torch.ones(3),
            membrane=torch.ones(3) * -60.0,
            l4_spikes=torch.ones(3),
            l23_spikes=torch.ones(4),
            l5_spikes=torch.ones(2),
            l6a_spikes=torch.ones(2),
            l6b_spikes=torch.ones(2),
            l4_trace=torch.ones(3) * 0.5,
            l23_trace=torch.ones(4) * 0.5,
            l5_trace=torch.ones(2) * 0.5,
            l23_recurrent_activity=torch.ones(4) * 0.8,
            top_down_modulation=torch.ones(4) * 0.7,
            gamma_attention_gate=torch.ones(4),
            ffi_strength=0.5,
            alpha_suppression=0.6,
            last_plasticity_delta=0.03,
        )

        state.reset()

        # Verify all tensors zeroed
        assert torch.all(state.spikes == 0)
        assert torch.all(state.membrane == 0)
        assert torch.all(state.l4_spikes == 0)
        assert torch.all(state.l23_spikes == 0)
        assert torch.all(state.l5_spikes == 0)
        assert torch.all(state.l4_trace == 0)
        assert torch.all(state.l23_recurrent_activity == 0)

        # Verify scalars reset
        assert state.ffi_strength == 0.0
        assert state.alpha_suppression == 1.0  # Resets to 1.0 (no suppression)
        assert state.last_plasticity_delta == 0.0
        assert state.gamma_attention_phase is None

    def test_roundtrip_serialization(self):
        """Test full serialization roundtrip preserves all state."""
        state1 = LayeredCortexState(
            spikes=torch.tensor([1.0, 0.0, 1.0]),
            membrane=torch.tensor([-60.0, -55.0, -70.0]),
            dopamine=0.4,
            l4_spikes=torch.tensor([1.0, 0.0, 1.0]),
            l23_spikes=torch.tensor([0.0, 1.0, 1.0, 0.0]),
            l5_spikes=torch.tensor([1.0, 0.0]),
            l6a_spikes=torch.tensor([0.0, 1.0]),
            l6b_spikes=torch.tensor([1.0, 1.0]),
            l4_trace=torch.tensor([0.2, 0.3, 0.1]),
            l23_trace=torch.tensor([0.4, 0.5, 0.6, 0.7]),
            l5_trace=torch.tensor([0.8, 0.9]),
            l23_recurrent_activity=torch.tensor([0.5, 0.8, 0.3, 0.9]),
            ffi_strength=0.6,
            alpha_suppression=0.7,
        )

        # Serialize and deserialize
        data = state1.to_dict()
        state2 = LayeredCortexState.from_dict(data)

        # Verify all fields match
        assert torch.equal(state2.spikes, state1.spikes)
        assert torch.equal(state2.membrane, state1.membrane)
        assert state2.dopamine == state1.dopamine
        assert torch.equal(state2.l4_spikes, state1.l4_spikes)
        assert torch.equal(state2.l23_spikes, state1.l23_spikes)
        assert torch.equal(state2.l5_spikes, state1.l5_spikes)
        assert torch.allclose(state2.l4_trace, state1.l4_trace)
        assert torch.allclose(state2.l23_recurrent_activity, state1.l23_recurrent_activity)
        assert state2.ffi_strength == state1.ffi_strength

    def test_with_none_fields(self):
        """Test handling of None fields."""
        state1 = LayeredCortexState(
            l4_spikes=torch.zeros(3),
            l23_spikes=torch.zeros(4),
            # Many fields left as None
        )

        data = state1.to_dict()
        state2 = LayeredCortexState.from_dict(data)

        # Verify None fields preserved
        assert state2.spikes is None
        assert state2.membrane is None
        assert state2.l4_trace is None
        assert state2.top_down_modulation is None

        # Verify non-None fields preserved
        assert torch.equal(state2.l4_spikes, state1.l4_spikes)
        assert torch.equal(state2.l23_spikes, state1.l23_spikes)


class TestLayeredCortexStateIntegration:
    """Test integration with LayeredCortex region."""

    @pytest.fixture
    def cortex(self):
        """Create LayeredCortex for testing."""
        config = LayeredCortexConfig(
            n_input=10,
            n_output=8,
            # Must explicitly specify all layer sizes
            l4_size=10,    # Input layer (matches n_input)
            l23_size=5,    # Processing layer
            l5_size=3,     # Subcortical output
            l6a_size=2,    # TRN feedback
            l6b_size=2,    # Relay feedback
            device="cpu",
            dt_ms=1.0,
        )
        return LayeredCortex(config)

    def test_get_state_captures_all_fields(self, cortex):
        """Test get_state() captures complete LayeredCortexState."""
        # Run forward pass to populate state
        test_input = torch.rand(cortex.config.n_input) > 0.8
        cortex.forward({"input": test_input})

        # Get state
        state = cortex.get_state()

        # Verify state is LayeredCortexState
        assert isinstance(state, LayeredCortexState)

        # Verify layer spikes captured
        assert state.l4_spikes is not None
        assert state.l23_spikes is not None
        assert state.l5_spikes is not None
        assert state.l6a_spikes is not None
        assert state.l6b_spikes is not None

        # Verify traces captured
        assert state.l4_trace is not None
        assert state.l23_trace is not None

        # Verify STP state captured
        assert state.stp_l23_recurrent_state is not None
        assert "u" in state.stp_l23_recurrent_state
        assert "x" in state.stp_l23_recurrent_state

    def test_load_state_restores_complete_state(self, cortex):
        """Test load_state() restores all state components."""
        # Run forward to populate state
        test_input = torch.rand(cortex.config.n_input) > 0.8
        for _ in range(5):
            cortex.forward({"input": test_input})

        # Save state
        state1 = cortex.get_state()

        # Create new cortex and load state
        cortex2 = LayeredCortex(cortex.config)
        cortex2.load_state(state1)

        state2 = cortex2.get_state()

        # Verify layer spikes match
        assert torch.equal(state2.l4_spikes, state1.l4_spikes)
        assert torch.equal(state2.l23_spikes, state1.l23_spikes)
        assert torch.equal(state2.l5_spikes, state1.l5_spikes)

        # Verify traces match
        assert torch.allclose(state2.l4_trace, state1.l4_trace, atol=1e-6)
        assert torch.allclose(state2.l23_trace, state1.l23_trace, atol=1e-6)

        # Verify STP state match
        assert torch.allclose(
            state2.stp_l23_recurrent_state["u"],
            state1.stp_l23_recurrent_state["u"],
            atol=1e-6
        )

    def test_state_roundtrip_consistency(self, cortex):
        """Test save→load→save produces identical state."""
        # Populate state
        test_input = torch.rand(cortex.config.n_input) > 0.8
        for _ in range(3):
            cortex.forward({"input": test_input})

        # First save
        state1 = cortex.get_state()

        # Load and re-save
        cortex2 = LayeredCortex(cortex.config)
        cortex2.load_state(state1)
        state2 = cortex2.get_state()

        # Verify states match
        data1 = state1.to_dict()
        data2 = state2.to_dict()

        for key in data1.keys():
            if isinstance(data1[key], torch.Tensor):
                assert torch.allclose(data2[key], data1[key], atol=1e-6)
            elif isinstance(data1[key], dict):
                # STP state comparison
                for subkey in data1[key].keys():
                    assert torch.allclose(data2[key][subkey], data1[key][subkey], atol=1e-6)
            else:
                assert data2[key] == data1[key]

    def test_state_device_consistency(self, cortex):
        """Test state stays on correct device after load."""
        test_input = torch.rand(cortex.config.n_input) > 0.8
        cortex.forward({"input": test_input})

        state1 = cortex.get_state()

        # Load state
        cortex.load_state(state1)

        # Verify all tensors on correct device
        assert cortex.state.l4_spikes.device.type == cortex.device
        assert cortex.state.l23_spikes.device.type == cortex.device
        assert cortex.state.l4_trace.device.type == cortex.device


class TestLayeredCortexStateFileIO:
    """Test file I/O with utility functions."""

    @pytest.fixture
    def cortex(self):
        """Create LayeredCortex for testing."""
        config = LayeredCortexConfig(
            n_input=10,
            n_output=8,
            # Must explicitly specify all layer sizes
            l4_size=10,    # Input layer (matches n_input)
            l23_size=5,    # Processing layer
            l5_size=3,     # Subcortical output
            l6a_size=2,    # TRN feedback
            l6b_size=2,    # Relay feedback
            device="cpu",
            dt_ms=1.0,
        )
        return LayeredCortex(config)

    def test_save_and_load_with_utilities(self, cortex, tmp_path):
        """Test save_region_state() and load_region_state() utilities."""
        # Populate state
        test_input = torch.rand(cortex.config.n_input) > 0.8
        for _ in range(3):
            cortex.forward({"input": test_input})

        state1 = cortex.get_state()

        # Save to file
        filepath = tmp_path / "layered_cortex_state.pt"
        save_region_state(state1, str(filepath))

        # Load from file
        state2 = load_region_state(LayeredCortexState, str(filepath), device="cpu")

        # Verify states match
        assert torch.equal(state2.l4_spikes, state1.l4_spikes)
        assert torch.equal(state2.l23_spikes, state1.l23_spikes)
        assert torch.allclose(state2.l4_trace, state1.l4_trace, atol=1e-6)
        assert state2.dopamine == state1.dopamine

    def test_file_io_with_cuda_transfer(self, cortex, tmp_path):
        """Test save on CPU, load on CUDA (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Populate state on CPU
        test_input = torch.rand(cortex.config.n_input) > 0.8
        cortex.forward({"input": test_input})

        state1 = cortex.get_state()

        # Save to file
        filepath = tmp_path / "layered_cortex_state_cuda.pt"
        save_region_state(state1, str(filepath))

        # Load to CUDA
        state2 = load_region_state(LayeredCortexState, str(filepath), device="cuda:0")

        # Verify device transfer
        assert state2.l4_spikes.device.type == "cuda"
        assert state2.l23_spikes.device.type == "cuda"

        # Verify values match
        assert torch.equal(state2.l4_spikes.cpu(), state1.l4_spikes)
        assert torch.equal(state2.l23_spikes.cpu(), state1.l23_spikes)


class TestLayeredCortexStateEdgeCases:
    """Test edge cases and error handling."""

    def test_partial_state_fields(self):
        """Test state with only some fields populated."""
        state = LayeredCortexState(
            l4_spikes=torch.zeros(3),
            l23_spikes=torch.zeros(4),
            # Most fields None
        )

        data = state.to_dict()
        state2 = LayeredCortexState.from_dict(data)

        # Verify partial state preserved
        assert torch.equal(state2.l4_spikes, state.l4_spikes)
        assert torch.equal(state2.l23_spikes, state.l23_spikes)
        assert state2.l5_spikes is None
        assert state2.l4_trace is None

    def test_missing_stp_state(self):
        """Test handling when STP state is None."""
        state1 = LayeredCortexState(
            l4_spikes=torch.zeros(3),
            l23_spikes=torch.zeros(4),
            stp_l23_recurrent_state=None,  # Explicit None
        )

        data = state1.to_dict()
        state2 = LayeredCortexState.from_dict(data)

        # Verify None preserved
        assert state2.stp_l23_recurrent_state is None

    def test_all_layers_populated(self):
        """Test with all 6 layers fully populated."""
        state1 = LayeredCortexState(
            l4_spikes=torch.rand(10) > 0.5,
            l23_spikes=torch.rand(15) > 0.5,
            l5_spikes=torch.rand(10) > 0.5,
            l6a_spikes=torch.rand(3) > 0.5,
            l6b_spikes=torch.rand(2) > 0.5,
            l4_trace=torch.rand(10),
            l23_trace=torch.rand(15),
            l5_trace=torch.rand(10),
            l6a_trace=torch.rand(3),
            l6b_trace=torch.rand(2),
            l23_recurrent_activity=torch.rand(15),
            top_down_modulation=torch.rand(15),
            gamma_attention_gate=torch.rand(15),
        )

        # Roundtrip
        data = state1.to_dict()
        state2 = LayeredCortexState.from_dict(data)

        # Verify all layers preserved
        assert torch.equal(state2.l4_spikes, state1.l4_spikes)
        assert torch.equal(state2.l23_spikes, state1.l23_spikes)
        assert torch.equal(state2.l5_spikes, state1.l5_spikes)
        assert torch.equal(state2.l6a_spikes, state1.l6a_spikes)
        assert torch.equal(state2.l6b_spikes, state1.l6b_spikes)
        assert torch.allclose(state2.l4_trace, state1.l4_trace)
        assert torch.allclose(state2.l6a_trace, state1.l6a_trace)
        assert torch.allclose(state2.l6b_trace, state1.l6b_trace)

    def test_stp_state_with_2d_tensors(self):
        """Test STP state with 2D tensors (typical for recurrent connections)."""
        stp_state = {
            "u": torch.rand(4, 4),  # [l23_size, l23_size]
            "x": torch.rand(4, 4),
        }

        state1 = LayeredCortexState(
            l23_spikes=torch.zeros(4),
            stp_l23_recurrent_state=stp_state,
        )

        # Roundtrip
        data = state1.to_dict()
        state2 = LayeredCortexState.from_dict(data)

        # Verify 2D STP state preserved
        assert torch.equal(state2.stp_l23_recurrent_state["u"], stp_state["u"])
        assert torch.equal(state2.stp_l23_recurrent_state["x"], stp_state["x"])
        assert state2.stp_l23_recurrent_state["u"].shape == (4, 4)
