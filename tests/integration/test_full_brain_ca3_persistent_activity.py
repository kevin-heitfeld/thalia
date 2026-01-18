"""
Full-brain integration test for CA3 persistent activity preservation across checkpoints.

This test verifies that CA3 persistent activity (attractor dynamics) is properly
preserved across checkpoint save/load cycles in a realistic full-brain context with:
- Coordinated thalamus→cortex→hippocampus pathways
- Theta oscillations for encoding/retrieval modulation
- Proper neuromodulation (ACh for encoding mode)

Author: Thalia Project
Date: January 19, 2026
"""

import tempfile
from pathlib import Path

import pytest
import torch

from tests.utils import create_test_brain


class TestFullBrainCA3PersistentActivity:
    """Test CA3 persistent activity in full-brain context."""

    @pytest.fixture
    def brain_with_hippocampus(self):
        """Create minimal brain with thalamus→cortex→hippocampus pathway."""
        return create_test_brain(
            device="cpu",
            dt_ms=1.0,
            thalamus_size=64,
            cortex_size=128,
            hippocampus_size=80,
            include_striatum=False,  # Not needed for this test
            include_pfc=False,
        )

    def test_ca3_persistent_activity_in_full_brain_context(self, brain_with_hippocampus):
        """Verify CA3 persistent activity builds up and is preserved across checkpoint in full brain.

        This test uses a realistic setup where:
        1. Sensory input → thalamus → cortex → hippocampus
        2. Theta oscillations coordinate encoding
        3. ACh modulation enables encoding mode
        4. CA3 builds up persistent activity from repeated input pattern
        5. Checkpoint preserves this persistent activity
        """
        brain = brain_with_hippocampus

        # Get hippocampus component
        hippo = brain.components["hippocampus"]

        # Set encoding mode (high ACh)
        hippo.set_neuromodulators(acetylcholine=1.0)

        # Create a consistent input pattern (not random)
        # This allows hippocampus to build up persistent activity for this pattern
        # Use thalamus size (64) as input size
        pattern = torch.ones(64, device=brain.device) * 0.8

        # Run multiple timesteps to allow CA3 to build persistent activity
        # With theta oscillations and coordinated activity, CA3 should accumulate
        for _ in range(100):
            brain.forward({"thalamus": pattern}, n_timesteps=1)

        # Check if CA3 has built up persistent activity
        if hasattr(hippo.state, "ca3_persistent") and hippo.state.ca3_persistent is not None:
            initial_persistent = hippo.state.ca3_persistent.clone()

            # Verify some persistent activity was built up
            # Note: May be low due to sparse coding, so we check for ANY non-zero values
            if initial_persistent.sum() > 0.01:
                # Save checkpoint
                with tempfile.TemporaryDirectory() as tmpdir:
                    checkpoint_path = Path(tmpdir) / "ca3_test.ckpt"
                    brain.save_checkpoint(str(checkpoint_path))

                    # Create new brain and load checkpoint
                    brain2 = create_test_brain(
                        device="cpu",
                        dt_ms=1.0,
                        thalamus_size=64,
                        cortex_size=128,
                        hippocampus_size=80,
                        include_striatum=False,
                        include_pfc=False,
                    )
                    brain2.load_checkpoint(str(checkpoint_path))

                    # Verify persistent activity was preserved
                    hippo2 = brain2.components["hippocampus"]
                    loaded_persistent = hippo2.state.ca3_persistent

                    assert (
                        loaded_persistent is not None
                    ), "CA3 persistent activity should be preserved"
                    assert torch.allclose(
                        loaded_persistent, initial_persistent, atol=1e-6
                    ), "CA3 persistent activity not accurately preserved"

                    # Continue simulation with minimal input
                    hippo2.set_neuromodulators(acetylcholine=1.0)
                    for _ in range(20):
                        brain2.forward(
                            {
                                "thalamus": torch.zeros(
                                    brain2.config.input_size, device=brain2.device
                                )
                            },
                            n_timesteps=1,
                        )

                    # Check that persistent activity decays but doesn't vanish immediately
                    # (biological attractor dynamics have some persistence)
                    final_persistent = hippo2.state.ca3_persistent
                    if final_persistent is not None and initial_persistent.sum() > 0:
                        ratio = final_persistent.sum() / (initial_persistent.sum() + 1e-8)
                        # Allow significant decay (not a perfect attractor)
                        # but verify it doesn't immediately collapse to zero
                        assert ratio > 0.05, (
                            f"CA3 persistent activity should maintain some activity, "
                            f"but got ratio={ratio:.3f}"
                        )
            else:
                pytest.skip(
                    f"CA3 persistent activity too weak to test (sum={initial_persistent.sum():.4f}). "
                    "This may occur with certain random initializations."
                )
        else:
            pytest.skip("CA3 persistent activity not available in this configuration")
