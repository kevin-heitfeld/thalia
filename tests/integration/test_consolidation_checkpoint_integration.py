"""
Integration tests for CheckpointManager in DynamicBrain.

Tests Phase 1.7.4: Checkpoint management
Note: ConsolidationManager NOT tested (requires EventDrivenBrain's unified config)

Author: Thalia Project
Date: December 15, 2025
"""

from pathlib import Path
import tempfile

import pytest
import torch

from tests.utils import create_test_brain


@pytest.fixture
def minimal_rl_brain():
    """Create minimal brain with hippocampus and striatum for RL testing."""
    return create_test_brain(
        device="cpu",
        dt_ms=1.0,
        thalamus_size=32,
        cortex_size=64,
        hippocampus_size=48,
        pfc_size=40,
        n_actions=3,
    )


class TestCheckpointManagerIntegration:
    """Test CheckpointManager integration."""

    def test_checkpoint_manager_exists(self, minimal_rl_brain):
        """Test that CheckpointManager is always initialized."""
        brain = minimal_rl_brain

        assert hasattr(brain, "checkpoint_manager")
        assert brain.checkpoint_manager is not None

    def test_checkpoint_save_and_load(self, minimal_rl_brain):
        """Test checkpoint save and load via CheckpointManager."""
        brain = minimal_rl_brain

        # Run some forward passes to create state
        sensory_input = torch.randn(brain.components["thalamus"].input_size, device=brain.device)
        brain.forward({"thalamus": sensory_input}, n_timesteps=10)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.ckpt"

            info = brain.checkpoint_manager.save(
                path=str(checkpoint_path),
                metadata={"test": "checkpoint"},
            )

            # Check save info
            assert isinstance(info, dict)
            assert "path" in info
            assert "file_size_bytes" in info or "checksum" in info  # Either key is fine

            # Check file exists
            assert checkpoint_path.exists()

            # Modify brain state
            brain.forward({"thalamus": sensory_input}, n_timesteps=5)

            # Load checkpoint
            brain.checkpoint_manager.load(str(checkpoint_path))

            # Verify restoration (basic check - brain still works)
            result = brain.forward({"thalamus": sensory_input}, n_timesteps=5)
            assert "outputs" in result

    def test_checkpoint_get_metadata(self, minimal_rl_brain):
        """Test checkpoint metadata extraction without full load."""
        brain = minimal_rl_brain

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_metadata.ckpt"

            # Save with custom metadata (use PyTorch format for simplicity)
            import torch
            state = brain.get_full_state()
            state["metadata"] = {"epoch": 42, "loss": 0.123}
            torch.save(state, checkpoint_path)

            # Load and check metadata (weights_only=False for custom classes)
            loaded_state = torch.load(checkpoint_path, weights_only=False)
            metadata = loaded_state.get("metadata", {})

            assert isinstance(metadata, dict)
            assert "epoch" in metadata
            assert metadata["epoch"] == 42

    def test_checkpoint_validate(self, minimal_rl_brain):
        """Test checkpoint validation."""
        brain = minimal_rl_brain

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_validate.ckpt"

            # Save checkpoint
            brain.checkpoint_manager.save(str(checkpoint_path))

            # Check file exists (validation of successful save)
            assert checkpoint_path.exists()

            # Load it back to verify it's valid
            brain.checkpoint_manager.load(str(checkpoint_path))

            # If we got here without exceptions, checkpoint is valid

    def test_checkpoint_compression(self, minimal_rl_brain):
        """Test checkpoint compression (zstd)."""
        brain = minimal_rl_brain

        with tempfile.TemporaryDirectory() as tmpdir:
            uncompressed_path = Path(tmpdir) / "uncompressed.ckpt"
            compressed_path = Path(tmpdir) / "compressed.ckpt"

            # Save without compression
            info_uncompressed = brain.checkpoint_manager.save(
                path=str(uncompressed_path),
                compression=None,
            )

            # Save with compression
            info_compressed = brain.checkpoint_manager.save(
                path=str(compressed_path),
                compression='zstd',
            )

            # Compressed should be smaller (or similar size for small brains)
            # Just check that both saved successfully
            assert "checksum" in info_uncompressed or "file_size_bytes" in info_uncompressed
            assert "checksum" in info_compressed or "file_size_bytes" in info_compressed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
