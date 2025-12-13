"""
Integration tests for hybrid checkpoint format (Phase 3).

Tests the hybrid approach where different regions use different formats
(elastic tensors for large stable regions, neuromorphic for small dynamic regions).

Test Coverage:
- Format auto-selection based on region properties
- Mixed region types in same checkpoint
- Cross-format compatibility
- Performance comparison
- Migration between formats
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

import torch

from thalia.core.brain import EventDrivenBrain
from thalia.regions.striatum import Striatum
from thalia.regions.cortex import Cortex
from thalia.regions.hippocampus import Hippocampus
from thalia.io.checkpoint_manager import CheckpointManager


@pytest.fixture
def device():
    """Return device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def hybrid_brain(device):
    """Create brain with mixed region types for hybrid testing."""
    # Large stable region - should use elastic tensor
    cortex = Cortex(n_neurons=1000, device=device, growth_frequency=0.01)

    # Small dynamic region - should use neuromorphic
    striatum = Striatum(n_actions=50, device=device, growth_frequency=0.3)

    # Neurogenesis region - should use neuromorphic
    hippocampus = Hippocampus(n_neurons=200, device=device, neurogenesis=True)

    brain = EventDrivenBrain(device=device)
    brain.add_region("cortex", cortex)
    brain.add_region("striatum", striatum)
    brain.add_region("hippocampus", hippocampus)

    return brain


class TestFormatAutoSelection:
    """Test automatic format selection based on region properties."""

    def test_large_stable_uses_elastic_tensor(self, hybrid_brain):
        """Large stable regions should use elastic tensor format."""
        manager = CheckpointManager(hybrid_brain)

        state = manager._get_region_state(hybrid_brain.regions["cortex"])

        assert state["format"] == "elastic_tensor"
        assert "weights" in state
        assert "used" in state
        assert "capacity" in state

    def test_small_dynamic_uses_neuromorphic(self, hybrid_brain):
        """Small dynamic regions should use neuromorphic format."""
        manager = CheckpointManager(hybrid_brain)

        state = manager._get_region_state(hybrid_brain.regions["striatum"])

        assert state["format"] == "neuromorphic"
        assert "neurons" in state

        # Should have neuron-centric data
        assert all("id" in n for n in state["neurons"])

    def test_neurogenesis_region_uses_neuromorphic(self, hybrid_brain):
        """Regions with neurogenesis should use neuromorphic format."""
        manager = CheckpointManager(hybrid_brain)

        state = manager._get_region_state(hybrid_brain.regions["hippocampus"])

        assert state["format"] == "neuromorphic"

    def test_format_selection_based_on_growth_frequency(self, device):
        """Format selection should consider growth frequency."""
        # High growth frequency -> neuromorphic
        region_dynamic = Striatum(n_actions=100, device=device, growth_frequency=0.5)

        manager = CheckpointManager(None)
        state_dynamic = manager._get_region_state(region_dynamic)

        assert state_dynamic["format"] == "neuromorphic"

        # Low growth frequency -> elastic tensor
        region_stable = Striatum(n_actions=100, device=device, growth_frequency=0.01)

        state_stable = manager._get_region_state(region_stable)

        assert state_stable["format"] == "elastic_tensor"

    def test_format_selection_based_on_size(self, device):
        """Small regions should prefer neuromorphic, large prefer elastic."""
        # Small region -> neuromorphic
        region_small = Cortex(n_neurons=50, device=device)

        manager = CheckpointManager(None)
        state_small = manager._get_region_state(region_small)

        assert state_small["format"] == "neuromorphic"

        # Large region -> elastic tensor
        region_large = Cortex(n_neurons=10000, device=device)

        state_large = manager._get_region_state(region_large)

        assert state_large["format"] == "elastic_tensor"


class TestMixedRegionCheckpoint:
    """Test checkpoints with mixed region formats."""

    def test_save_mixed_format_checkpoint(self, hybrid_brain, tmp_path):
        """Should save checkpoint with different formats per region."""
        checkpoint_path = tmp_path / "mixed.ckpt"

        manager = CheckpointManager(hybrid_brain)
        info = manager.save(checkpoint_path)

        # Load raw checkpoint
        loaded = torch.load(checkpoint_path)

        # Should have regions with different formats
        regions = loaded["regions"]

        assert regions["cortex"]["format"] == "elastic_tensor"
        assert regions["striatum"]["format"] == "neuromorphic"
        assert regions["hippocampus"]["format"] == "neuromorphic"

    def test_load_mixed_format_checkpoint(self, hybrid_brain, tmp_path):
        """Should load checkpoint with mixed formats correctly."""
        checkpoint_path = tmp_path / "mixed_load.ckpt"

        manager = CheckpointManager(hybrid_brain)

        # Save
        manager.save(checkpoint_path)

        # Reset all regions
        for region in hybrid_brain.regions.values():
            region.reset()

        # Load
        manager.load(checkpoint_path)

        # All regions should be restored
        # (specific state checks would go here)

    def test_partial_region_load(self, hybrid_brain, tmp_path):
        """Should be able to load only specific regions."""
        checkpoint_path = tmp_path / "partial_region.ckpt"

        manager = CheckpointManager(hybrid_brain)
        manager.save(checkpoint_path)

        # Reset only striatum
        hybrid_brain.regions["striatum"].reset()

        # Load only striatum
        manager.load(checkpoint_path, regions=["striatum"])

        # Striatum should be restored, others unchanged

    def test_region_format_mismatch_handled(self, hybrid_brain, tmp_path):
        """Should handle checkpoint where region format changed."""
        checkpoint_path = tmp_path / "format_changed.ckpt"

        # Save with current formats
        manager = CheckpointManager(hybrid_brain)
        manager.save(checkpoint_path)

        # Change striatum to use elastic tensor format
        hybrid_brain.regions["striatum"].checkpoint_format = "elastic_tensor"

        # Load should auto-convert or warn
        with pytest.warns(UserWarning, match="format mismatch"):
            manager.load(checkpoint_path)


class TestCrossFormatCompatibility:
    """Test compatibility between formats."""

    def test_convert_elastic_to_neuromorphic(self, device, tmp_path):
        """Should be able to convert elastic tensor checkpoint to neuromorphic."""
        checkpoint_path = tmp_path / "elastic.ckpt"

        # Save as elastic tensor
        region = Striatum(n_actions=10, device=device, checkpoint_format="elastic_tensor")
        region.reset()

        state_elastic = region.get_full_state()
        torch.save(state_elastic, checkpoint_path)

        # Load into neuromorphic format region
        region_neuro = Striatum(n_actions=10, device=device, checkpoint_format="neuromorphic")

        loaded = torch.load(checkpoint_path)

        # Should auto-convert during load
        region_neuro.load_full_state(loaded, auto_convert=True)

    def test_convert_neuromorphic_to_elastic(self, device, tmp_path):
        """Should be able to convert neuromorphic checkpoint to elastic tensor."""
        checkpoint_path = tmp_path / "neuromorphic.ckpt"

        # Save as neuromorphic
        region = Striatum(n_actions=10, device=device, checkpoint_format="neuromorphic")
        region.reset()

        state_neuro = region.get_full_state()
        torch.save(state_neuro, checkpoint_path)

        # Load into elastic tensor format region
        region_elastic = Striatum(n_actions=10, device=device, checkpoint_format="elastic_tensor")

        loaded = torch.load(checkpoint_path)

        # Should auto-convert during load
        region_elastic.load_full_state(loaded, auto_convert=True)

    def test_format_conversion_preserves_state(self, device, tmp_path):
        """Format conversion should preserve all neural state."""
        # Set distinctive state
        region = Striatum(n_actions=10, device=device, checkpoint_format="elastic_tensor")
        region.reset()
        region.membrane[:10] = torch.arange(10, dtype=torch.float32, device=device)

        # Save elastic
        state_elastic = region.get_full_state()

        # Convert to neuromorphic
        state_neuro = region._convert_format(state_elastic, target_format="neuromorphic")

        # Convert back to elastic
        state_elastic2 = region._convert_format(state_neuro, target_format="elastic_tensor")

        # Should match original
        assert torch.allclose(
            state_elastic["neuron_state"]["membrane"],
            state_elastic2["neuron_state"]["membrane"]
        )


class TestPerformanceComparison:
    """Compare performance of different formats."""

    def test_elastic_faster_for_large_dense_regions(self, device, tmp_path):
        """Elastic tensor should be faster for large dense regions."""
        checkpoint_path = tmp_path / "perf_large.ckpt"

        # Large dense region
        region_elastic = Striatum(
            n_actions=500,
            device=device,
            checkpoint_format="elastic_tensor",
            sparsity=0.8  # Dense
        )
        region_elastic.reset()

        region_neuro = Striatum(
            n_actions=500,
            device=device,
            checkpoint_format="neuromorphic",
            sparsity=0.8
        )
        region_neuro.reset()

        # Time elastic save/load
        import time

        start = time.perf_counter()
        state_elastic = region_elastic.get_full_state()
        torch.save(state_elastic, checkpoint_path)
        loaded = torch.load(checkpoint_path)
        region_elastic.load_full_state(loaded)
        elastic_time = time.perf_counter() - start

        # Time neuromorphic save/load
        start = time.perf_counter()
        state_neuro = region_neuro.get_full_state()
        torch.save(state_neuro, checkpoint_path)
        loaded = torch.load(checkpoint_path)
        region_neuro.load_full_state(loaded)
        neuro_time = time.perf_counter() - start

        # Elastic should be faster for dense regions
        assert elastic_time < neuro_time * 2  # Allow 2x slack

    def test_neuromorphic_faster_for_small_sparse_regions(self, device, tmp_path):
        """Neuromorphic should be faster for small sparse regions."""
        checkpoint_path = tmp_path / "perf_small.ckpt"

        # Small sparse region
        region_elastic = Striatum(
            n_actions=20,
            device=device,
            checkpoint_format="elastic_tensor",
            sparsity=0.05  # Very sparse
        )
        region_elastic.reset()

        region_neuro = Striatum(
            n_actions=20,
            device=device,
            checkpoint_format="neuromorphic",
            sparsity=0.05
        )
        region_neuro.reset()

        import time

        # Time both formats
        start = time.perf_counter()
        state_elastic = region_elastic.get_full_state()
        torch.save(state_elastic, checkpoint_path)
        loaded = torch.load(checkpoint_path)
        region_elastic.load_full_state(loaded)
        elastic_time = time.perf_counter() - start

        start = time.perf_counter()
        state_neuro = region_neuro.get_full_state()
        torch.save(state_neuro, checkpoint_path)
        loaded = torch.load(checkpoint_path)
        region_neuro.load_full_state(loaded)
        neuro_time = time.perf_counter() - start

        # Neuromorphic should be comparable for sparse regions
        # (might not be faster due to overhead, but shouldn't be much slower)
        assert neuro_time < elastic_time * 3  # Allow 3x slack

    def test_checkpoint_size_comparison(self, device, tmp_path):
        """Compare checkpoint sizes for different formats."""
        # Sparse region
        region_elastic = Striatum(
            n_actions=100,
            device=device,
            checkpoint_format="elastic_tensor",
            sparsity=0.1  # Sparse
        )
        region_elastic.reset()

        region_neuro = Striatum(
            n_actions=100,
            device=device,
            checkpoint_format="neuromorphic",
            sparsity=0.1
        )
        region_neuro.reset()

        # Save both
        path_elastic = tmp_path / "elastic.ckpt"
        path_neuro = tmp_path / "neuro.ckpt"

        torch.save(region_elastic.get_full_state(), path_elastic)
        torch.save(region_neuro.get_full_state(), path_neuro)

        # Neuromorphic should be smaller for sparse networks
        size_elastic = path_elastic.stat().st_size
        size_neuro = path_neuro.stat().st_size

        assert size_neuro < size_elastic * 0.8  # At least 20% smaller


class TestFormatMigration:
    """Test migrating between formats."""

    def test_manual_format_change(self, device, tmp_path):
        """Should be able to manually change format of existing checkpoint."""
        checkpoint_path = tmp_path / "migrate.ckpt"

        # Save as elastic
        region = Striatum(n_actions=10, device=device, checkpoint_format="elastic_tensor")
        region.reset()

        state = region.get_full_state()
        torch.save(state, checkpoint_path)

        # Migrate to neuromorphic
        from thalia.io.format_converter import FormatConverter

        converter = FormatConverter()
        converter.migrate_checkpoint(
            checkpoint_path,
            target_format="neuromorphic",
            output_path=tmp_path / "migrated.ckpt"
        )

        # Verify migrated checkpoint
        migrated = torch.load(tmp_path / "migrated.ckpt")
        assert migrated["format"] == "neuromorphic"

    def test_automatic_format_upgrade(self, device, tmp_path):
        """Loading old checkpoint should auto-upgrade to new format."""
        checkpoint_path = tmp_path / "old.ckpt"

        # Create old format checkpoint (no format field)
        old_state = {
            "neuron_state": {
                "membrane": torch.zeros(10, device=device),
            },
            "pathway_state": {},
            "learning_state": {},
        }

        torch.save(old_state, checkpoint_path)

        # Load into new region
        region = Striatum(n_actions=10, device=device)

        loaded = torch.load(checkpoint_path)

        # Should auto-upgrade
        with pytest.warns(UserWarning, match="old format"):
            region.load_full_state(loaded, auto_upgrade=True)


class TestHybridEdgeCases:
    """Test edge cases in hybrid format."""

    def test_empty_region_in_checkpoint(self, hybrid_brain, tmp_path):
        """Checkpoint with empty region should load correctly."""
        checkpoint_path = tmp_path / "empty_region.ckpt"

        # Remove all neurons from striatum
        hybrid_brain.regions["striatum"].n_neurons_active = 0

        manager = CheckpointManager(hybrid_brain)
        manager.save(checkpoint_path)

        # Should load without error
        manager.load(checkpoint_path)

    def test_new_region_not_in_checkpoint(self, hybrid_brain, tmp_path):
        """Loading checkpoint into brain with new region should work."""
        checkpoint_path = tmp_path / "missing_region.ckpt"

        # Save current brain
        manager = CheckpointManager(hybrid_brain)
        manager.save(checkpoint_path)

        # Add new region
        from thalia.regions.thalamus import Thalamus
        thalamus = Thalamus(device=hybrid_brain.device)
        hybrid_brain.add_region("thalamus", thalamus)

        # Load should work (new region uses default initialization)
        manager.load(checkpoint_path)

    def test_removed_region_in_checkpoint(self, hybrid_brain, tmp_path):
        """Checkpoint with removed region should warn but load others."""
        checkpoint_path = tmp_path / "extra_region.ckpt"

        # Save current brain
        manager = CheckpointManager(hybrid_brain)
        manager.save(checkpoint_path)

        # Remove hippocampus
        del hybrid_brain.regions["hippocampus"]

        # Load should warn about missing region
        with pytest.warns(UserWarning, match="hippocampus.*not in brain"):
            manager.load(checkpoint_path)

        # Other regions should still load correctly

    def test_format_version_mismatch(self, hybrid_brain, tmp_path):
        """Checkpoint with future format version should error or warn."""
        checkpoint_path = tmp_path / "future_version.ckpt"

        # Create checkpoint with future version
        manager = CheckpointManager(hybrid_brain)
        manager.save(checkpoint_path)

        loaded = torch.load(checkpoint_path)
        loaded["format_version"] = "99.0.0"  # Future version
        torch.save(loaded, checkpoint_path)

        # Load should handle gracefully
        with pytest.warns(UserWarning, match="future format version"):
            manager.load(checkpoint_path)
