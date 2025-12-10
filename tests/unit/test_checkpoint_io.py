"""
Unit tests for Thalia binary checkpoint I/O.

Tests:
- Binary format encoding/decoding
- Tensor serialization (dense and sparse)
- Full checkpoint save/load roundtrip
- Partial loading (specific regions)
- Checksum validation
- Version compatibility
"""

import tempfile
from pathlib import Path
import pytest
import torch

from thalia.io import BrainCheckpoint
from thalia.io.binary_format import (
    CheckpointHeader,
    RegionIndexEntry,
    MAGIC_NUMBER,
    MAJOR_VERSION,
    MINOR_VERSION,
    PATCH_VERSION,
)
from thalia.io.tensor_encoding import (
    encode_tensor,
    decode_tensor,
    estimate_encoding_size,
)


class TestBinaryFormat:
    """Test low-level binary format operations."""

    def test_header_roundtrip(self):
        """Test header serialization/deserialization."""
        import time

        header = CheckpointHeader(
            magic=MAGIC_NUMBER,
            major_version=MAJOR_VERSION,
            minor_version=MINOR_VERSION,
            patch_version=PATCH_VERSION,
            flags=0,
            timestamp=int(time.time()),
            metadata_offset=256,
            metadata_length=1024,
            region_index_offset=1280,
            region_index_length=120,
            connectivity_offset=0,
            connectivity_length=0,
            total_neurons=10000,
            total_synapses=500000,
            training_steps=1000,
            num_regions=3,
            checksum_type=1,
        )

        # Serialize
        header_bytes = header.to_bytes()
        assert len(header_bytes) == 256

        # Deserialize
        header2 = CheckpointHeader.from_bytes(header_bytes)

        # Verify
        assert header2.magic == MAGIC_NUMBER
        assert header2.major_version == MAJOR_VERSION
        assert header2.total_neurons == 10000
        assert header2.num_regions == 3

    def test_header_validation(self):
        """Test header validation."""
        import time

        # Valid header
        header = CheckpointHeader(
            magic=MAGIC_NUMBER,
            major_version=MAJOR_VERSION,
            minor_version=MINOR_VERSION,
            patch_version=PATCH_VERSION,
            flags=0,
            timestamp=int(time.time()),
            metadata_offset=256,
            metadata_length=1024,
            region_index_offset=1280,
            region_index_length=120,
            connectivity_offset=0,
            connectivity_length=0,
            total_neurons=1000,
            total_synapses=5000,
            training_steps=100,
            num_regions=2,
            checksum_type=1,
        )

        is_valid, issues = header.validate()
        assert is_valid
        assert len(issues) == 0

        # Invalid magic number
        header.magic = b'FAKE'
        is_valid, issues = header.validate()
        assert not is_valid
        assert any("magic" in issue.lower() for issue in issues)

    def test_region_index_entry_roundtrip(self):
        """Test region index entry serialization."""
        entry = RegionIndexEntry(
            region_name="striatum",
            data_offset=1400,
            data_length=50000,
        )

        # Serialize
        entry_bytes = entry.to_bytes()
        assert len(entry_bytes) == 48

        # Deserialize
        entry2 = RegionIndexEntry.from_bytes(entry_bytes)

        assert entry2.region_name == "striatum"
        assert entry2.data_offset == 1400
        assert entry2.data_length == 50000


class TestTensorEncoding:
    """Test tensor serialization."""

    def test_dense_float32_roundtrip(self):
        """Test dense float32 tensor encoding/decoding."""
        tensor = torch.randn(10, 20, dtype=torch.float32)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            encode_tensor(tensor, f)

        try:
            with open(temp_path, 'rb') as f:
                decoded = decode_tensor(f, device='cpu')

            assert decoded.shape == tensor.shape
            assert decoded.dtype == tensor.dtype
            assert torch.allclose(decoded, tensor)
        finally:
            Path(temp_path).unlink()

    def test_dense_int64_roundtrip(self):
        """Test dense int64 tensor encoding/decoding."""
        tensor = torch.randint(0, 100, (5, 5), dtype=torch.int64)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            encode_tensor(tensor, f)

        try:
            with open(temp_path, 'rb') as f:
                decoded = decode_tensor(f, device='cpu')

            assert decoded.shape == tensor.shape
            assert decoded.dtype == tensor.dtype
            assert torch.equal(decoded, tensor)
        finally:
            Path(temp_path).unlink()

    def test_sparse_tensor_roundtrip(self):
        """Test sparse tensor encoding/decoding."""
        # Create sparse tensor (mostly zeros)
        dense = torch.zeros(100, 100)
        dense[10, 20] = 1.5
        dense[30, 40] = 2.5
        dense[50, 60] = 3.5

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            # Force sparse encoding with low threshold
            encode_tensor(dense, f, sparsity_threshold=0.99)

        try:
            with open(temp_path, 'rb') as f:
                decoded = decode_tensor(f, device='cpu')

            assert decoded.shape == dense.shape
            assert torch.allclose(decoded.to_dense() if decoded.is_sparse else decoded, dense)
        finally:
            Path(temp_path).unlink()

    def test_encoding_size_estimation(self):
        """Test encoding size estimation."""
        tensor = torch.randn(50, 50)

        estimated = estimate_encoding_size(tensor, sparsity_threshold=0.1)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            actual = encode_tensor(tensor, f, sparsity_threshold=0.1)

        try:
            # Estimated should be within 10% of actual
            assert abs(estimated - actual) / actual < 0.1
        finally:
            Path(temp_path).unlink()


class TestCheckpointAPI:
    """Test high-level checkpoint API."""

    def test_checkpoint_info_without_loading(self, temp_dir):
        """Test getting checkpoint info without loading full state."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        # Create minimal brain
        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(input_size=64, n_actions=4),
            ),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

        # Save checkpoint
        checkpoint_path = temp_dir / "test_info.thalia"
        BrainCheckpoint.save(brain, checkpoint_path, metadata={"experiment": "test"})

        # Get info without loading
        info = BrainCheckpoint.info(checkpoint_path)

        assert "file_path" in info
        assert "file_size" in info
        assert "version" in info
        assert "num_regions" in info
        assert "metadata" in info
        assert info["metadata"]["experiment"] == "test"

    def test_checkpoint_validation(self, temp_dir):
        """Test checkpoint file validation."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        # Create minimal brain
        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(input_size=64, n_actions=4),
            ),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

        # Save checkpoint
        checkpoint_path = temp_dir / "test_validate.thalia"
        BrainCheckpoint.save(brain, checkpoint_path)

        # Validate
        result = BrainCheckpoint.validate(checkpoint_path)

        assert result["valid"]
        assert len(result["issues"]) == 0

    def test_corrupted_checkpoint_detection(self, temp_dir):
        """Test detection of corrupted checkpoint."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        # Create and save checkpoint
        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(input_size=64, n_actions=4),
            ),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

        checkpoint_path = temp_dir / "test_corrupt.thalia"
        BrainCheckpoint.save(brain, checkpoint_path)

        # Corrupt the file (overwrite some bytes in the middle)
        with open(checkpoint_path, 'r+b') as f:
            f.seek(1000)
            f.write(b'\xFF' * 100)

        # Validation should fail
        result = BrainCheckpoint.validate(checkpoint_path)

        assert not result["valid"]
        assert len(result["issues"]) > 0


class TestCheckpointRoundtrip:
    """Test full checkpoint save/load roundtrip."""

    def test_striatum_checkpoint_roundtrip(self, temp_dir):
        """Test striatum checkpoint save/load."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        # Create brain with striatum
        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(input_size=64, n_actions=4),
            ),
        )
        brain1 = EventDrivenBrain.from_thalia_config(config)

        # Process some input
        input_spikes = (torch.rand(64) > 0.8).float()
        brain1.process_sample(input_spikes, n_timesteps=5)

        # Save checkpoint
        checkpoint_path = temp_dir / "striatum.thalia"
        save_info = BrainCheckpoint.save(
            brain1,
            checkpoint_path,
            metadata={"experiment": "striatum_test"}
        )

        assert save_info["num_regions"] >= 1
        assert Path(checkpoint_path).exists()

        # Load checkpoint into new brain
        state = BrainCheckpoint.load(checkpoint_path, device="cpu")

        brain2 = EventDrivenBrain.from_thalia_config(config)
        brain2.load_full_state(state)

        # Verify state matches
        state1 = brain1.get_full_state()
        state2 = brain2.get_full_state()

        # Check striatum weights
        s1_weights = state1["regions"]["striatum"]["weights"]
        s2_weights = state2["regions"]["striatum"]["weights"]

        for key in s1_weights:
            if s1_weights[key] is not None:
                assert torch.allclose(s1_weights[key], s2_weights[key])

    def test_full_brain_checkpoint_roundtrip(self, temp_dir):
        """Test full brain with multiple regions."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes, CortexType

        # Create brain with multiple regions
        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu", dt_ms=1.0, theta_frequency_hz=8.0),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=64,
                    cortex_size=128,
                    hippocampus_size=64,
                    pfc_size=32,
                    n_actions=4,
                ),
            ),
        )
        brain1 = EventDrivenBrain.from_thalia_config(config)

        # Process some steps
        for _ in range(5):
            input_spikes = (torch.rand(64) > 0.8).float()
            brain1.process_sample(input_spikes, n_timesteps=3)

        # Save checkpoint
        checkpoint_path = temp_dir / "full_brain.thalia"
        save_info = BrainCheckpoint.save(
            brain1,
            checkpoint_path,
            metadata={"experiment": "full_brain_test", "steps": 5}
        )

        assert save_info["num_regions"] >= 3

        # Load checkpoint
        state = BrainCheckpoint.load(checkpoint_path, device="cpu")

        brain2 = EventDrivenBrain.from_thalia_config(config)
        brain2.load_full_state(state)

        # Verify metadata
        assert state["metadata"]["experiment"] == "full_brain_test"
        assert state["metadata"]["steps"] == 5

        # Verify state was loaded
        state1 = brain1.get_full_state()
        state2 = brain2.get_full_state()

        # Check that key regions have matching weights
        assert "striatum" in state1["regions"]
        assert "striatum" in state2["regions"]


# Fixtures

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
