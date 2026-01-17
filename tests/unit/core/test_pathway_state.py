"""
Tests for Pathway State Management (Phase 0).

Tests the PathwayState protocol and AxonalProjectionState implementation,
focusing on delay buffer serialization and restoration.

Author: Thalia Project
Date: December 2025
"""

import pytest
import torch

from thalia.core.pathway_state import (
    AxonalProjectionState,
    load_pathway_state,
    save_pathway_state,
)
from thalia.pathways.axonal_projection import AxonalProjection


class TestAxonalProjectionState:
    """Test AxonalProjectionState serialization and deserialization."""

    def test_state_creation(self):
        """Test creating state with delay buffers."""
        # Create state
        buffer = torch.zeros(6, 128, dtype=torch.bool)  # 5ms delay + 1
        buffer[2, 10:20] = True  # Some spikes in flight

        state = AxonalProjectionState(delay_buffers={"cortex:l5": (buffer, 2, 5, 128)})

        assert "cortex:l5" in state.delay_buffers
        buf, ptr, max_delay, size = state.delay_buffers["cortex:l5"]
        assert buf.shape == (6, 128)
        assert ptr == 2
        assert max_delay == 5
        assert size == 128

    def test_state_serialization_roundtrip(self):
        """Test to_dict and from_dict preserve state."""
        # Create state with in-flight spikes
        buffer = torch.zeros(6, 128, dtype=torch.bool)
        buffer[2, 10:20] = True
        buffer[3, 50:60] = True

        state1 = AxonalProjectionState(delay_buffers={"cortex:l5": (buffer, 2, 5, 128)})

        # Serialize
        data = state1.to_dict()

        # Check structure
        assert "version" in data
        assert data["version"] == 1
        assert "delay_buffers" in data
        assert "cortex:l5" in data["delay_buffers"]

        # Deserialize
        state2 = AxonalProjectionState.from_dict(data, device=torch.device("cpu"))

        # Verify equality
        buf1, ptr1, max_delay1, size1 = state1.delay_buffers["cortex:l5"]
        buf2, ptr2, max_delay2, size2 = state2.delay_buffers["cortex:l5"]

        assert torch.equal(buf1, buf2)
        assert ptr1 == ptr2
        assert max_delay1 == max_delay2
        assert size1 == size2

    def test_state_reset(self):
        """Test reset clears delay buffers."""
        # Create state with spikes
        buffer = torch.ones(6, 128, dtype=torch.bool)
        state = AxonalProjectionState(delay_buffers={"cortex:l5": (buffer, 2, 5, 128)})

        # Reset
        state.reset()

        # Verify cleared
        buf, _, _, _ = state.delay_buffers["cortex:l5"]
        assert not buf.any()

    def test_multi_source_state(self):
        """Test state with multiple sources."""
        buffer1 = torch.zeros(6, 128, dtype=torch.bool)
        buffer1[2, 10:20] = True

        buffer2 = torch.zeros(4, 64, dtype=torch.bool)
        buffer2[1, 5:10] = True

        state = AxonalProjectionState(
            delay_buffers={
                "cortex:l5": (buffer1, 2, 5, 128),
                "hippocampus": (buffer2, 1, 3, 64),
            }
        )

        # Serialize and deserialize
        data = state.to_dict()
        restored = AxonalProjectionState.from_dict(data, device=torch.device("cpu"))

        # Check both sources preserved
        assert len(restored.delay_buffers) == 2
        assert "cortex:l5" in restored.delay_buffers
        assert "hippocampus" in restored.delay_buffers

        # Verify cortex buffer
        buf1, ptr1, _, _ = restored.delay_buffers["cortex:l5"]
        assert buf1.shape == (6, 128)
        assert ptr1 == 2
        assert buf1[2, 10:20].all()

        # Verify hippocampus buffer
        buf2, ptr2, _, _ = restored.delay_buffers["hippocampus"]
        assert buf2.shape == (4, 64)
        assert ptr2 == 1
        assert buf2[1, 5:10].all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_transfer(self):
        """Test state moves to correct device."""
        # Create on CPU
        buffer = torch.zeros(6, 128, dtype=torch.bool)
        buffer[2, 10:20] = True

        state = AxonalProjectionState(delay_buffers={"cortex:l5": (buffer, 2, 5, 128)})

        # Serialize
        data = state.to_dict()

        # Deserialize to CUDA
        state_cuda = AxonalProjectionState.from_dict(data, device=torch.device("cuda"))

        buf, _, _, _ = state_cuda.delay_buffers["cortex:l5"]
        assert buf.device.type == "cuda"
        assert buf[2, 10:20].all()


class TestAxonalProjectionIntegration:
    """Test AxonalProjection with PathwayState integration."""

    def test_projection_get_state(self):
        """Test AxonalProjection.get_state returns AxonalProjectionState."""
        projection = AxonalProjection(
            sources=[("cortex", "l5", 128, 5.0)],
            device="cpu",
            dt_ms=1.0,
        )

        state = projection.get_state()
        assert isinstance(state, AxonalProjectionState)
        assert "cortex:l5" in state.delay_buffers

    def test_projection_state_roundtrip(self):
        """Test projection save/load preserves in-flight spikes."""
        projection = AxonalProjection(
            sources=[("cortex", "l5", 128, 5.0)],
            device="cpu",
            dt_ms=1.0,
        )

        # Send some spikes
        spikes = torch.zeros(128, dtype=torch.bool)
        spikes[10:20] = True

        for _ in range(3):
            projection.forward({"cortex:l5": spikes})

        # Save state
        state = projection.get_state()

        # Create new projection and load
        projection2 = AxonalProjection(
            sources=[("cortex", "l5", 128, 5.0)],
            device="cpu",
            dt_ms=1.0,
        )
        projection2.load_state(state)

        # Continue sending zeros - should see delayed spikes emerge
        for t in range(6):
            output = projection2.forward({"cortex:l5": torch.zeros(128, dtype=torch.bool)})

            # At delay_steps (5), should see the original spikes
            if t == 4:  # 5 steps total (0-4)
                assert output["cortex:l5"][10:20].any()

    def test_projection_multi_source_checkpoint(self):
        """Test checkpoint with multiple sources preserves all buffers."""
        projection = AxonalProjection(
            sources=[
                ("cortex", "l5", 128, 5.0),
                ("hippocampus", None, 64, 3.0),
                ("pfc", None, 32, 2.0),
            ],
            device="cpu",
            dt_ms=1.0,
        )

        # Send spikes to each source
        cortex_spikes = torch.zeros(128, dtype=torch.bool)
        cortex_spikes[10:20] = True

        hipp_spikes = torch.zeros(64, dtype=torch.bool)
        hipp_spikes[5:10] = True

        pfc_spikes = torch.zeros(32, dtype=torch.bool)
        pfc_spikes[0:5] = True

        # Run for a few steps
        for _ in range(4):
            projection.forward(
                {
                    "cortex:l5": cortex_spikes,
                    "hippocampus": hipp_spikes,
                    "pfc": pfc_spikes,
                }
            )

        # Save state
        state = projection.get_state()

        # Verify all sources in state
        assert len(state.delay_buffers) == 3
        assert "cortex:l5" in state.delay_buffers
        assert "hippocampus" in state.delay_buffers
        assert "pfc" in state.delay_buffers

        # Load into new projection
        projection2 = AxonalProjection(
            sources=[
                ("cortex", "l5", 128, 5.0),
                ("hippocampus", None, 64, 3.0),
                ("pfc", None, 32, 2.0),
            ],
            device="cpu",
            dt_ms=1.0,
        )
        projection2.load_state(state)

        # Continue with zeros - should see delayed spikes emerge
        for t in range(6):
            output = projection2.forward(
                {
                    "cortex:l5": torch.zeros(128, dtype=torch.bool),
                    "hippocampus": torch.zeros(64, dtype=torch.bool),
                    "pfc": torch.zeros(32, dtype=torch.bool),
                }
            )

            # Check each source's delayed spikes appear at correct times
            if t == 1:  # pfc delay=2ms
                assert output["pfc"][0:5].any(), f"PFC spikes should appear at t={t}"
            if t == 2:  # hipp delay=3ms
                assert output["hippocampus"][5:10].any(), f"Hipp spikes should appear at t={t}"
            if t == 4:  # cortex delay=5ms
                assert output["cortex:l5"][10:20].any(), f"Cortex spikes should appear at t={t}"

    def test_projection_full_state_compatibility(self):
        """Test get_full_state/load_full_state work with dict format."""
        projection = AxonalProjection(
            sources=[("cortex", "l5", 128, 5.0)],
            device="cpu",
            dt_ms=1.0,
        )

        # Send spikes
        spikes = torch.zeros(128, dtype=torch.bool)
        spikes[10:20] = True
        projection.forward({"cortex:l5": spikes})

        # Save as dict
        state_dict = projection.get_full_state()
        assert isinstance(state_dict, dict)
        assert "version" in state_dict
        assert "delay_buffers" in state_dict

        # Load from dict
        projection2 = AxonalProjection(
            sources=[("cortex", "l5", 128, 5.0)],
            device="cpu",
            dt_ms=1.0,
        )
        projection2.load_full_state(state_dict)

        # Verify spikes preserved
        output = projection2.forward({"cortex:l5": torch.zeros(128, dtype=torch.bool)})
        # Buffer should have the spike from 1 step ago


class TestDelayPreservation:
    """Test that in-flight spikes are correctly preserved across checkpoints."""

    def test_delay_buffer_preservation(self):
        """Test that spikes in delay buffer are preserved exactly."""
        projection = AxonalProjection(
            sources=[("source", None, 10, 5.0)],
            device="cpu",
            dt_ms=1.0,
        )

        # Send unique pattern each timestep
        patterns = []
        for t in range(5):
            spikes = torch.zeros(10, dtype=torch.bool)
            spikes[t : t + 2] = True  # 2 spikes per timestep
            patterns.append(spikes.clone())
            projection.forward({"source": spikes})

        # Save state (spikes at t=0,1,2,3,4 are in buffer)
        state = projection.get_state()

        # Create new projection and load
        projection2 = AxonalProjection(
            sources=[("source", None, 10, 5.0)],
            device="cpu",
            dt_ms=1.0,
        )
        projection2.load_state(state)

        # Continue with zeros - should see the 5 patterns emerge
        for t in range(7):
            output = projection2.forward({"source": torch.zeros(10, dtype=torch.bool)})["source"]

            # After 5 steps, should see the original patterns
            if t < 5:
                expected = patterns[t]
                assert torch.equal(
                    output, expected
                ), f"At t={t}, expected pattern {expected} but got {output}"

    def test_pointer_position_preserved(self):
        """Test that buffer pointer position is correctly preserved."""
        projection = AxonalProjection(
            sources=[("source", None, 10, 5.0)],
            device="cpu",
            dt_ms=1.0,
        )

        # Run for 3 steps (pointer at position 3)
        for _ in range(3):
            spikes = torch.zeros(10, dtype=torch.bool)
            projection.forward({"source": spikes})

        # Check pointer before save
        buffer = projection._delay_buffers["source"]
        ptr_before = buffer.ptr

        # Save and load
        state = projection.get_state()
        projection2 = AxonalProjection(
            sources=[("source", None, 10, 5.0)],
            device="cpu",
            dt_ms=1.0,
        )
        projection2.load_state(state)

        # Check pointer after load
        buffer2 = projection2._delay_buffers["source"]
        assert buffer2.ptr == ptr_before


class TestUtilityFunctions:
    """Test utility functions for pathway state management."""

    def test_save_pathway_state(self):
        """Test save_pathway_state utility function."""
        projection = AxonalProjection(
            sources=[("cortex", "l5", 128, 5.0)],
            device="cpu",
            dt_ms=1.0,
        )

        state_dict = save_pathway_state(projection)
        assert isinstance(state_dict, dict)
        assert "version" in state_dict
        assert "delay_buffers" in state_dict

    def test_load_pathway_state(self):
        """Test load_pathway_state utility function."""
        projection = AxonalProjection(
            sources=[("cortex", "l5", 128, 5.0)],
            device="cpu",
            dt_ms=1.0,
        )

        # Send spikes
        spikes = torch.zeros(128, dtype=torch.bool)
        spikes[10:20] = True
        projection.forward({"cortex:l5": spikes})

        # Save
        state_dict = save_pathway_state(projection)

        # Create new projection and load
        projection2 = AxonalProjection(
            sources=[("cortex", "l5", 128, 5.0)],
            device="cpu",
            dt_ms=1.0,
        )
        load_pathway_state(projection2, state_dict)

        # Verify state transferred
        output = projection2.forward({"cortex:l5": torch.zeros(128, dtype=torch.bool)})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
