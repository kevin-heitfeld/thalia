"""
Unit tests for checkpoint versioning.

Tests that checkpoints track version information correctly and that
version compatibility checking works during load.
"""

import tempfile
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from thalia import __version__ as THALIA_VERSION
from thalia.io.binary_format import MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION, CheckpointHeader
from thalia.io.checkpoint import BrainCheckpoint


class MockBrain:
    """Minimal mock brain for testing checkpoint save/load."""

    def __init__(self):
        self.device = "cpu"
        self.config = type("obj", (object,), {"device": "cpu"})()
        self._growth_history = []

    def get_full_state(self):
        """Return minimal state dict."""
        return {
            "regions": {
                "test_region": {
                    "neuron_state": {
                        "membrane": torch.ones(100),
                    },
                    "weights": {
                        "test_weight": torch.rand(100, 50),
                    },
                }
            },
            "training_steps": 1000,
            "config": {"test": "config"},
        }


def test_checkpoint_includes_versions():
    """Checkpoint metadata should include version information."""
    brain = MockBrain()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".thalia") as f:
        checkpoint_path = f.name

    try:
        # Save checkpoint
        save_metadata = BrainCheckpoint.save(brain, checkpoint_path)

        # Check that save metadata includes basic info
        assert "file_size" in save_metadata
        assert "total_neurons" in save_metadata

        # Load info to check stored metadata
        info = BrainCheckpoint.info(checkpoint_path)
        metadata = info["metadata"]

        # Check that metadata includes version info
        assert "thalia_version" in metadata
        assert metadata["thalia_version"] == THALIA_VERSION

        assert "checkpoint_format_version" in metadata
        expected_format = f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}"
        assert metadata["checkpoint_format_version"] == expected_format

        assert "pytorch_version" in metadata
        assert metadata["pytorch_version"] == torch.__version__

        assert "timestamp" in metadata

    finally:
        Path(checkpoint_path).unlink(missing_ok=True)


def test_checkpoint_info_shows_versions():
    """Checkpoint info should display version information."""
    brain = MockBrain()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".thalia") as f:
        checkpoint_path = f.name

    try:
        # Save checkpoint
        BrainCheckpoint.save(brain, checkpoint_path)

        # Get info
        info = BrainCheckpoint.info(checkpoint_path)

        # Check version fields in info
        assert "version" in info
        expected_version = f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}"
        assert info["version"] == expected_version

        # Metadata should include Thalia version
        assert "metadata" in info
        metadata = info["metadata"]
        assert "thalia_version" in metadata
        assert metadata["thalia_version"] == THALIA_VERSION

    finally:
        Path(checkpoint_path).unlink(missing_ok=True)


@pytest.mark.parametrize(
    "version_patch,expect_error,error_match",
    [
        ("major", True, "Incompatible checkpoint format version"),
        ("minor", False, "Checkpoint format version mismatch"),
    ],
)
def test_load_different_version(version_patch, expect_error, error_match):
    """Test loading checkpoints with different version components.

    Why this test exists: Validates version compatibility checking to ensure
    that incompatible checkpoints (major version mismatch) fail fast, while
    compatible checkpoints (minor version mismatch) warn but succeed.

    Cases:
    - major: Incompatible format - should raise ValueError
    - minor: Compatible but different - should warn but load
    """
    brain = MockBrain()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".thalia") as f:
        checkpoint_path = f.name

    try:
        # Save checkpoint
        BrainCheckpoint.save(brain, checkpoint_path)

        # Monkey-patch version to simulate different version
        if version_patch == "major":
            patch_target = "thalia.io.checkpoint.MAJOR_VERSION"
            patch_value = MAJOR_VERSION + 1
        else:  # minor
            patch_target = "thalia.io.checkpoint.MINOR_VERSION"
            patch_value = MINOR_VERSION + 1

        with patch(patch_target, patch_value):
            if expect_error:
                with pytest.raises(ValueError, match=error_match):
                    BrainCheckpoint.load(checkpoint_path)
            else:
                with pytest.warns(UserWarning, match=error_match):
                    state = BrainCheckpoint.load(checkpoint_path)
                    # Test loaded state structure
                    assert "regions" in state, "Loaded state should contain regions"

    finally:
        Path(checkpoint_path).unlink(missing_ok=True)


def test_load_different_thalia_version_warns():
    """Loading checkpoint from different Thalia version should warn."""
    brain = MockBrain()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".thalia") as f:
        checkpoint_path = f.name

    try:
        # Save checkpoint with mocked version
        with patch("thalia.io.checkpoint.THALIA_VERSION", "0.1.0"):
            BrainCheckpoint.save(brain, checkpoint_path)

        # Load with current version (should warn about version mismatch)
        if THALIA_VERSION != "0.1.0":  # Only test if versions actually differ
            with pytest.warns(UserWarning, match="Checkpoint was saved with Thalia"):
                state = BrainCheckpoint.load(checkpoint_path)
                assert state is not None

    finally:
        Path(checkpoint_path).unlink(missing_ok=True)


def test_load_compatible_version_no_warning():
    """Loading checkpoint with same version should not warn."""
    brain = MockBrain()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".thalia") as f:
        checkpoint_path = f.name

    try:
        # Save checkpoint
        BrainCheckpoint.save(brain, checkpoint_path)

        # Load should succeed without warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            state = BrainCheckpoint.load(checkpoint_path)
            # Test loaded state structure (non-None guaranteed by load success)
            assert "regions" in state, "Loaded state should contain regions"

    finally:
        Path(checkpoint_path).unlink(missing_ok=True)


def test_version_in_binary_header():
    """Binary header should contain version information."""
    header = CheckpointHeader(
        magic=b"THAL",
        major_version=MAJOR_VERSION,
        minor_version=MINOR_VERSION,
        patch_version=PATCH_VERSION,
        flags=0,
        timestamp=int(time.time()),
        metadata_offset=256,
        metadata_length=1000,
        region_index_offset=1256,
        region_index_length=500,
        connectivity_offset=0,
        connectivity_length=0,
        total_neurons=1000,
        total_synapses=50000,
        training_steps=5000,
        num_regions=5,
        checksum_type=1,
    )

    # Serialize and deserialize
    header_bytes = header.to_bytes()
    assert len(header_bytes) == 256

    restored_header = CheckpointHeader.from_bytes(header_bytes)

    # Version should be preserved
    assert restored_header.major_version == MAJOR_VERSION
    assert restored_header.minor_version == MINOR_VERSION
    assert restored_header.patch_version == PATCH_VERSION
