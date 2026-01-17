"""
Tests for per-target axonal delay variation.

Tests realistic axonal branching where the same source sends spikes
to multiple targets with different conduction velocities.

Author: Thalia Project
Date: December 23, 2025
"""

import pytest
import torch

from thalia.pathways.axonal_projection import AxonalProjection, SourceSpec


class TestPerTargetDelays:
    """Test per-target delay variation in AxonalProjection."""

    def test_source_spec_default_delay(self):
        """Test SourceSpec with single default delay."""
        spec = SourceSpec(
            region_name="cortex",
            port="l5",
            size=128,
            delay_ms=5.0
        )

        # Should use default delay for any target
        assert spec.get_delay_for_target("striatum") == 5.0
        assert spec.get_delay_for_target("thalamus") == 5.0
        assert spec.get_delay_for_target("hippocampus") == 5.0

    def test_source_spec_per_target_delays(self):
        """Test SourceSpec with per-target delay overrides."""
        spec = SourceSpec(
            region_name="cortex",
            port="l5",
            size=128,
            delay_ms=5.0,  # Default
            target_delays={
                "striatum": 3.0,    # Fast
                "thalamus": 10.0,   # Slow
            }
        )

        # Should use target-specific delays when available
        assert spec.get_delay_for_target("striatum") == 3.0
        assert spec.get_delay_for_target("thalamus") == 10.0
        # Should fall back to default for unlisted targets
        assert spec.get_delay_for_target("hippocampus") == 5.0

    def test_axonal_projection_single_delay(self):
        """Test AxonalProjection with traditional single delay."""
        projection = AxonalProjection(
            sources=[("cortex", "l5", 128, 5.0)],
            device="cpu",
            dt_ms=1.0,
            target_name="striatum"
        )

        # Send spikes and check delay
        spikes = torch.ones(128, dtype=torch.bool)

        # Forward for 6 timesteps (5ms delay + 1)
        for _ in range(6):
            output = projection.forward({"cortex:l5": spikes})

        # After 6 steps, delayed spikes should appear
        assert output["cortex:l5"].sum() == 128

    def test_axonal_projection_per_target_delays_striatum(self):
        """Test AxonalProjection selects correct delay for striatum target."""
        # Create projection targeting striatum (should use 3ms delay)
        projection_striatum = AxonalProjection(
            sources=[
                ("cortex", "l5", 128, 5.0, {"striatum": 3.0, "thalamus": 10.0})
            ],
            device="cpu",
            dt_ms=1.0,
            target_name="striatum"  # Key: target name determines delay
        )

        spikes = torch.ones(128, dtype=torch.bool)

        # Should use 3ms delay for striatum
        # Need 4 forward calls: t=0 (write), t=1, t=2, t=3 (read with delay=3)
        outputs = []
        for _ in range(4):
            output = projection_striatum.forward({"cortex:l5": spikes})
            outputs.append(output["cortex:l5"].sum().item())

        # At t=3, delayed spikes (from t=0) should appear
        assert outputs[3] == 128, f"Expected 128 spikes at t=3, got {outputs[3]}"

    def test_axonal_projection_per_target_delays_thalamus(self):
        """Test AxonalProjection selects correct delay for thalamus target."""
        # Create projection targeting thalamus (should use 10ms delay)
        projection_thalamus = AxonalProjection(
            sources=[
                ("cortex", "l5", 128, 5.0, {"striatum": 3.0, "thalamus": 10.0})
            ],
            device="cpu",
            dt_ms=1.0,
            target_name="thalamus"  # Key: different target = different delay
        )

        spikes = torch.ones(128, dtype=torch.bool)

        # Should use 10ms delay for thalamus
        # Need 11 forward calls: t=0 (write), ..., t=10 (read with delay=10)
        outputs = []
        for _ in range(11):
            output = projection_thalamus.forward({"cortex:l5": spikes})
            outputs.append(output["cortex:l5"].sum().item())

        # At t=10, delayed spikes (from t=0) should appear
        assert outputs[10] == 128, f"Expected 128 spikes at t=10, got {outputs[10]}"

    def test_multi_source_per_target_delays(self):
        """Test multiple sources with different per-target delays."""
        projection = AxonalProjection(
            sources=[
                ("cortex", "l5", 128, 5.0, {"striatum": 2.0}),  # Fast cortex→striatum
                ("hippocampus", None, 64, 5.0, {"striatum": 6.0}),  # Slower hipp→striatum
            ],
            device="cpu",
            dt_ms=1.0,
            target_name="striatum"
        )

        cortex_spikes = torch.ones(128, dtype=torch.bool)
        hipp_spikes = torch.ones(64, dtype=torch.bool)

        # Cortex should arrive at t=2, hippocampus at t=6
        outputs_cortex = []
        outputs_hipp = []

        for _ in range(7):
            output = projection.forward({
                "cortex:l5": cortex_spikes,
                "hippocampus": hipp_spikes
            })
            outputs_cortex.append(output["cortex:l5"].sum().item())
            outputs_hipp.append(output["hippocampus"].sum().item())

        # Cortex arrives earlier (2ms delay)
        assert outputs_cortex[2] == 128
        # Hippocampus arrives later (6ms delay)
        assert outputs_hipp[6] == 64

    def test_repr_with_per_target_delays(self):
        """Test __repr__ shows target-specific delay information."""
        projection = AxonalProjection(
            sources=[
                ("cortex", "l5", 128, 5.0, {"striatum": 3.0, "thalamus": 10.0})
            ],
            device="cpu",
            dt_ms=1.0,
            target_name="striatum"
        )

        repr_str = repr(projection)
        # Should show the target-specific delay (3ms) not default (5ms)
        assert "3.0ms→striatum" in repr_str or "3ms→striatum" in repr_str

    def test_no_target_name_uses_default(self):
        """Test that without target_name, default delay is used."""
        projection = AxonalProjection(
            sources=[
                ("cortex", "l5", 128, 5.0, {"striatum": 3.0, "thalamus": 10.0})
            ],
            device="cpu",
            dt_ms=1.0,
            # target_name=None (implicit)
        )

        spikes = torch.ones(128, dtype=torch.bool)

        # Should use default 5ms delay
        outputs = []
        for _ in range(6):
            output = projection.forward({"cortex:l5": spikes})
            outputs.append(output["cortex:l5"].sum().item())

        # At t=5, delayed spikes should appear (default delay)
        assert outputs[5] == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
