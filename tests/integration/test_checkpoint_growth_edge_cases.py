"""
Integration tests for checkpoint-growth edge cases and failure modes.

Tests error handling, validation, and resilience of the checkpoint system.

Test Coverage:
- Corrupted checkpoints
- Version mismatches
- Memory limits
- Concurrent growth during load
- Partial failures
- Recovery strategies
"""

import threading

import pytest
import torch

from thalia.regions.striatum import Striatum
from thalia.regions.striatum.config import StriatumConfig


@pytest.fixture
def device():
    """Return device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def base_config(device):
    """Create base striatum config."""
    return StriatumConfig(
        n_input=100,
        n_output=5,  # Number of actions
        growth_enabled=True,
        device=device,
    )


@pytest.fixture
def striatum(base_config):
    """Create striatum instance."""
    region = Striatum(base_config)
    region.reset_state()
    return region


class TestCorruptedCheckpoints:
    """Test handling of corrupted checkpoint data."""

    def test_missing_required_fields(self, striatum, tmp_path):
        """Missing required fields should raise clear error."""
        checkpoint_path = tmp_path / "missing_fields.ckpt"

        # Create checkpoint missing required field
        corrupted = {
            "format_version": "2.0.0",
            # Missing: "neuron_state"
            "pathway_state": {},
            "learning_state": {},
        }

        torch.save(corrupted, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        with pytest.raises(KeyError, match="neuron_state"):
            striatum.load_full_state(loaded)

    def test_invalid_tensor_dtype(self, striatum, tmp_path):
        """Invalid tensor dtype should raise error."""
        checkpoint_path = tmp_path / "bad_dtype.ckpt"

        # Create checkpoint with wrong dtype
        bad_state = {
            "format_version": "2.0.0",
            "neuron_state": {
                "n_neurons_active": 5,
                "n_neurons_capacity": 8,
                "membrane": torch.zeros(5, dtype=torch.int32),  # Wrong dtype!
            },
            "pathway_state": {},
            "learning_state": {},
        }

        torch.save(bad_state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        with pytest.raises(TypeError, match="dtype"):
            striatum.load_full_state(loaded)

    def test_negative_dimensions(self, striatum, tmp_path):
        """Negative dimensions should raise error."""
        checkpoint_path = tmp_path / "negative_dim.ckpt"

        bad_state = {
            "format_version": "2.0.0",
            "neuron_state": {
                "n_neurons_active": -5,  # Invalid!
                "n_neurons_capacity": 8,
                "membrane": torch.zeros(5),
            },
            "pathway_state": {},
            "learning_state": {},
        }

        torch.save(bad_state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        with pytest.raises(ValueError, match="negative"):
            striatum.load_full_state(loaded)

    def test_nan_values_in_state(self, striatum, tmp_path):
        """NaN values should be detected and handled."""
        checkpoint_path = tmp_path / "nan_values.ckpt"

        # Create checkpoint with NaN
        nan_state = striatum.get_full_state()
        nan_state["neuron_state"]["membrane"][0] = float("nan")

        torch.save(nan_state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        # Should either reject or warn
        with pytest.warns(UserWarning, match="NaN"):
            striatum.load_full_state(loaded, check_nan=True)

    def test_inf_values_in_weights(self, striatum, tmp_path):
        """Inf values should be detected."""
        checkpoint_path = tmp_path / "inf_weights.ckpt"

        # Create checkpoint with inf
        inf_state = striatum.get_full_state()
        inf_state["pathway_state"]["d1_pathway"]["weights"][0, 0] = float("inf")

        torch.save(inf_state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        with pytest.warns(UserWarning, match="inf"):
            striatum.load_full_state(loaded, check_inf=True)

    def test_corrupted_file_io_error(self, striatum, tmp_path):
        """Corrupted file should raise clear I/O error."""
        checkpoint_path = tmp_path / "corrupted.ckpt"

        # Write garbage data
        with open(checkpoint_path, "wb") as f:
            f.write(b"this is not a valid checkpoint file")

        with pytest.raises(Exception, match="corrupted|invalid|cannot load"):
            torch.load(checkpoint_path, weights_only=False)


class TestVersionMismatches:
    """Test handling of version incompatibilities."""

    def test_future_major_version_rejected(self, striatum, tmp_path):
        """Checkpoint from future major version should be rejected."""
        checkpoint_path = tmp_path / "future_major.ckpt"

        state = striatum.get_full_state()
        state["format_version"] = "99.0.0"

        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        with pytest.raises(ValueError, match="incompatible.*version"):
            striatum.load_full_state(loaded)

    def test_old_minor_version_warns(self, striatum, tmp_path):
        """Checkpoint from old minor version should warn but work."""
        checkpoint_path = tmp_path / "old_minor.ckpt"

        state = striatum.get_full_state()
        state["format_version"] = "2.0.0"  # Current is 2.1.0

        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        with pytest.warns(UserWarning, match="old version"):
            striatum.load_full_state(loaded)

    def test_thalia_version_mismatch_warns(self, striatum, tmp_path):
        """Checkpoint from different Thalia version should warn."""
        checkpoint_path = tmp_path / "different_thalia.ckpt"

        state = striatum.get_full_state()
        state["thalia_version"] = "0.1.0"  # Very old

        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        with pytest.warns(UserWarning, match="Thalia version"):
            striatum.load_full_state(loaded)

    def test_device_mismatch_converts(self, striatum, tmp_path):
        """Checkpoint from different device should auto-convert."""
        checkpoint_path = tmp_path / "device_mismatch.ckpt"

        # Save on CPU
        cpu_striatum = Striatum(striatum.config)
        cpu_striatum.device = torch.device("cpu")
        cpu_striatum.reset_state()

        state = cpu_striatum.get_full_state()
        torch.save(state, checkpoint_path)

        # Load on current device (might be CUDA)
        loaded = torch.load(checkpoint_path, weights_only=False)

        # Should auto-convert device
        striatum.load_full_state(loaded)

        # Tensors should be on correct device
        assert striatum.membrane.device == striatum.device


class TestMemoryLimits:
    """Test behavior under memory constraints."""

    def test_large_checkpoint_streaming_load(self, base_config, tmp_path):
        """Very large checkpoint should use streaming load."""
        checkpoint_path = tmp_path / "huge.ckpt"

        # Create large region
        config = base_config
        config.n_actions = 10000

        large_region = Striatum(config)
        large_region.reset_state()

        # Save
        state = large_region.get_full_state()
        torch.save(state, checkpoint_path)

        # Load with memory constraints
        # Should stream rather than load all at once
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        large_region.load_full_state(loaded, streaming=True)

    def test_out_of_memory_during_growth(self, striatum, tmp_path):
        """OOM during growth should be handled gracefully."""
        checkpoint_path = tmp_path / "oom.ckpt"

        # Create checkpoint requiring huge growth
        huge_state = {
            "format_version": "2.0.0",
            "neuron_state": {
                "n_neurons_active": 1000000,  # 1M neurons
                "n_neurons_capacity": 1000000,
                "membrane": torch.zeros(1000000),
            },
            "pathway_state": {},
            "learning_state": {},
        }

        torch.save(huge_state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        # Should raise clear OOM error, not crash
        with pytest.raises(RuntimeError, match="out of memory"):
            striatum.load_full_state(loaded)

    def test_memory_efficient_format_used(self, base_config, tmp_path):
        """Should use memory-efficient format when memory constrained."""
        checkpoint_path = tmp_path / "memory_efficient.ckpt"

        config = base_config
        config.memory_limit_mb = 100  # Constrain memory

        region = Striatum(config)
        region.reset_state()

        # Should automatically choose memory-efficient format
        state = region.get_full_state()

        # Neuromorphic is more memory-efficient for sparse networks
        if region.sparsity < 0.3:
            assert state["format"] == "neuromorphic"


class TestConcurrentOperations:
    """Test thread safety and concurrent operations."""

    def test_concurrent_growth_during_load(self, striatum, tmp_path):
        """Growth during load should be thread-safe."""
        checkpoint_path = tmp_path / "concurrent.ckpt"

        state = striatum.get_full_state()
        torch.save(state, checkpoint_path)

        # Load in one thread
        def load_thread():
            loaded = torch.load(checkpoint_path, weights_only=False)
            striatum.load_full_state(loaded)

        # Grow in another thread
        def grow_thread():
            striatum.add_neurons(n_new=2)

        t1 = threading.Thread(target=load_thread)
        t2 = threading.Thread(target=grow_thread)

        # Should not crash (one operation should wait for the other)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    def test_concurrent_saves_are_safe(self, striatum, tmp_path):
        """Multiple concurrent saves should not corrupt checkpoints."""
        def save_thread(i):
            path = tmp_path / f"concurrent_{i}.ckpt"
            state = striatum.get_full_state()
            torch.save(state, path)

        # Launch multiple save threads
        threads = [threading.Thread(target=save_thread, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All checkpoints should be valid
        for i in range(5):
            path = tmp_path / f"concurrent_{i}.ckpt"
            loaded = torch.load(path, weights_only=False)
            assert loaded["format_version"] == "2.0.0"


class TestPartialFailures:
    """Test recovery from partial failures."""

    def test_pathway_load_failure_preserves_neurons(self, striatum, tmp_path):
        """Failure loading pathways should not corrupt neuron state."""
        checkpoint_path = tmp_path / "partial_fail.ckpt"

        # Create checkpoint with corrupted pathway
        state = striatum.get_full_state()
        state["pathway_state"]["d1_pathway"]["weights"] = None  # Corrupt!

        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        # Should load neurons successfully, warn about pathway
        with pytest.warns(UserWarning, match="pathway"):
            striatum.load_full_state(loaded, partial_ok=True)

        # Neuron state should be loaded
        assert striatum.membrane is not None

    def test_learning_state_failure_preserves_network(self, striatum, tmp_path):
        """Failure loading learning state should preserve network structure."""
        checkpoint_path = tmp_path / "learning_fail.ckpt"

        # Create checkpoint with corrupted learning state
        state = striatum.get_full_state()
        state["learning_state"]["eligibility_d1"] = None  # Corrupt!

        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        # Should load network, reinitialize learning state
        with pytest.warns(UserWarning, match="learning state"):
            striatum.load_full_state(loaded, partial_ok=True)

        # Network structure should be intact
        assert striatum.n_neurons_active == 5

    def test_rollback_on_validation_failure(self, striatum, tmp_path):
        """Failed validation should rollback to previous state."""
        checkpoint_path = tmp_path / "invalid.ckpt"

        # Save current state
        original_membrane = striatum.membrane.clone()

        # Create invalid checkpoint
        invalid_state = {
            "format_version": "2.0.0",
            "neuron_state": {
                "n_neurons_active": 5,
                "n_neurons_capacity": 8,
                "membrane": torch.full((5,), float("nan")),  # Invalid!
            },
            "pathway_state": {},
            "learning_state": {},
        }

        torch.save(invalid_state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        # Load should fail and rollback
        with pytest.raises(ValueError):
            striatum.load_full_state(loaded, validate=True, rollback_on_error=True)

        # State should be unchanged
        assert torch.allclose(striatum.membrane, original_membrane)


class TestRecoveryStrategies:
    """Test recovery from checkpoint failures."""

    def test_automatic_fallback_to_backup(self, striatum, tmp_path):
        """Corrupted checkpoint should auto-fallback to backup."""
        primary_path = tmp_path / "primary.ckpt"
        backup_path = tmp_path / "backup.ckpt"

        # Create valid backup
        state = striatum.get_full_state()
        torch.save(state, backup_path)

        # Create corrupted primary
        with open(primary_path, "wb") as f:
            f.write(b"corrupted")

        # Load should auto-fallback to backup
        from thalia.io.checkpoint_manager import CheckpointManager

        manager = CheckpointManager(striatum)

        with pytest.warns(UserWarning, match="fallback.*backup"):
            manager.load(primary_path, backup_path=backup_path)

    def test_checkpoint_validation_before_save(self, striatum, tmp_path):
        """Should validate state before saving checkpoint."""
        checkpoint_path = tmp_path / "validated.ckpt"

        # Corrupt state
        striatum.membrane[:] = float("nan")

        # Save should detect and reject
        with pytest.raises(ValueError, match="invalid state"):
            state = striatum.get_full_state(validate=True)

    def test_checkpoint_repair_utility(self, striatum, tmp_path):
        """Should provide utility to repair corrupted checkpoints."""
        checkpoint_path = tmp_path / "corrupted.ckpt"
        repaired_path = tmp_path / "repaired.ckpt"

        # Create checkpoint with minor corruption
        state = striatum.get_full_state()
        state["neuron_state"]["membrane"][0] = float("nan")
        torch.save(state, checkpoint_path)

        # Repair utility
        from thalia.io.checkpoint_repair import repair_checkpoint

        success = repair_checkpoint(
            checkpoint_path,
            repaired_path,
            strategies=["replace_nan", "clamp_inf"]
        )

        assert success

        # Repaired checkpoint should load
        repaired = torch.load(repaired_path, weights_only=False)
        striatum.load_full_state(repaired)


class TestValidationRules:
    """Test checkpoint validation rules."""

    def test_dimension_consistency_check(self, striatum, tmp_path):
        """Dimensions should be consistent across all tensors."""
        checkpoint_path = tmp_path / "inconsistent_dims.ckpt"

        # Create inconsistent state
        state = striatum.get_full_state()
        state["neuron_state"]["membrane"] = torch.zeros(5)
        state["pathway_state"]["d1_pathway"]["weights"] = torch.zeros(10, 100)  # Wrong!

        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        with pytest.raises(ValueError, match="dimension.*inconsistent"):
            striatum.load_full_state(loaded, validate=True)

    def test_capacity_greater_than_used(self, striatum, tmp_path):
        """Capacity must be >= used neurons."""
        checkpoint_path = tmp_path / "capacity_too_small.ckpt"

        state = striatum.get_full_state()
        state["neuron_state"]["n_neurons_active"] = 10
        state["neuron_state"]["n_neurons_capacity"] = 5  # Invalid!

        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        with pytest.raises(ValueError, match="capacity.*less than.*active"):
            striatum.load_full_state(loaded, validate=True)

    def test_weight_bounds_check(self, striatum, tmp_path):
        """Weights should be within reasonable bounds."""
        checkpoint_path = tmp_path / "extreme_weights.ckpt"

        state = striatum.get_full_state()
        state["pathway_state"]["d1_pathway"]["weights"][0, 0] = 1e10  # Too large!

        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        with pytest.warns(UserWarning, match="extreme weight"):
            striatum.load_full_state(loaded, validate=True, strict=False)

    def test_neuromorphic_id_uniqueness(self, base_config, tmp_path):
        """Neuromorphic format must have unique neuron IDs."""
        checkpoint_path = tmp_path / "duplicate_ids.ckpt"

        config = base_config
        config.checkpoint_format = "neuromorphic"

        region = Striatum(config)
        region.reset_state()

        state = region.get_full_state()

        # Create duplicate ID
        state["neurons"][1]["id"] = state["neurons"][0]["id"]

        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        with pytest.raises(ValueError, match="duplicate.*id"):
            region.load_full_state(loaded, validate=True)
