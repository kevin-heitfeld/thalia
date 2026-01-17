"""Tests for LayeredCortexState edge cases and cortex-specific validation.

Basic state serialization tests (to_dict, from_dict, reset, roundtrip, file I/O)
are now covered by RegionTestBase (tests/utils/region_test_base.py).

This file focuses on:
- Nested STP state dict handling
- Complex CUDA transfer scenarios
- Cortex-specific 6-layer validation
- Edge cases not covered by base class

Created: October 10, 2024
Updated: December 22, 2025 (migrated basic tests to RegionTestBase)
"""

import pytest
import torch

from thalia.core.region_state import load_region_state, save_region_state
from thalia.regions.cortex.config import LayeredCortexConfig, LayeredCortexState
from thalia.regions.cortex.layered_cortex import LayeredCortex


class TestLayeredCortexStateEdgeCases:
    """Edge cases and cortex-specific validation."""

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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_file_io_with_cuda_transfer(self, tmp_path):
        """Test save on CPU, load on CUDA (if available)."""

        # Create state on CPU
        sizes = {
            "input_size": 10,
            "l4_size": 10,
            "l23_size": 5,
            "l5_size": 3,
            "l6a_size": 2,
            "l6b_size": 2,
        }
        config = LayeredCortexConfig(dt_ms=1.0)
        cortex = LayeredCortex(config=config, sizes=sizes, device="cpu")

        test_input = torch.rand(10) > 0.8
        cortex.forward({"input": test_input})
        state1 = cortex.get_state()

        # Save to file
        filepath = tmp_path / "cortex_state_cuda.pt"
        save_region_state(state1, str(filepath))

        # Load to CUDA
        state2 = load_region_state(LayeredCortexState, str(filepath), device="cuda:0")

        # Verify device transfer
        assert state2.l4_spikes.device.type == "cuda"
        assert state2.l23_spikes.device.type == "cuda"

        # Verify values match
        assert torch.equal(state2.l4_spikes.cpu(), state1.l4_spikes)
        assert torch.equal(state2.l23_spikes.cpu(), state1.l23_spikes)

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
        """Test with all 6 layers fully populated (cortex-specific)."""
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

        # Verify all 6 cortex layers preserved
        assert torch.equal(state2.l4_spikes, state1.l4_spikes)
        assert torch.equal(state2.l23_spikes, state1.l23_spikes)
        assert torch.equal(state2.l5_spikes, state1.l5_spikes)
        assert torch.equal(state2.l6a_spikes, state1.l6a_spikes)
        assert torch.equal(state2.l6b_spikes, state1.l6b_spikes)
        assert torch.allclose(state2.l4_trace, state1.l4_trace)
        assert torch.allclose(state2.l6a_trace, state1.l6a_trace)
        assert torch.allclose(state2.l6b_trace, state1.l6b_trace)

    def test_stp_state_with_2d_tensors(self):
        """Test STP state with 2D tensors (recurrent connections)."""
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
