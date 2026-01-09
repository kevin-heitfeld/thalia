"""
Unit tests for hybrid checkpoint format (Phase 3) - Striatum only.

Tests automatic format selection and loading for Striatum region.
"""

import pytest

import torch

from thalia.regions.striatum import Striatum
from thalia.regions.striatum.config import StriatumConfig


@pytest.fixture
def device():
    """Return device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_striatum(device):
    """Small striatum (should use neuromorphic format, no population coding)."""
    config = StriatumConfig(
        n_actions=5,  # Small: 5 actions
        neurons_per_action=1,
        input_sources={'default': 100},
        growth_enabled=True,
        device=device,
    )
    striatum = Striatum(config)
    striatum.reset_state()
    return striatum


@pytest.fixture
def small_striatum_population(device):
    """Small striatum WITH population coding (should use neuromorphic format)."""
    config = StriatumConfig(
        n_actions=5,  # 5 actions × 10 neurons/action = 50 neurons total
        neurons_per_action=10,
        input_sources={'default': 100},
        growth_enabled=True,
        device=device,
    )
    striatum = Striatum(config)
    striatum.reset_state()
    return striatum


@pytest.fixture
def large_striatum(device):
    """Large striatum (should use elastic tensor format, no population coding)."""
    config = StriatumConfig(
        n_actions=150,  # Large: 150 actions
        neurons_per_action=1,
        input_sources={'default': 100},
        growth_enabled=False,
        device=device,
    )
    striatum = Striatum(config)
    striatum.reset_state()
    return striatum


@pytest.fixture
def large_striatum_population(device):
    """Large striatum WITH population coding (should use elastic tensor format)."""
    config = StriatumConfig(
        n_actions=150,  # 150 actions × 10 neurons/action = 1500 neurons total
        neurons_per_action=10,
        input_sources={'default': 100},
        growth_enabled=False,
        device=device,
    )
    striatum = Striatum(config)
    striatum.reset_state()
    return striatum


class TestFormatAutoSelection:
    """Test automatic format selection."""

    def test_small_region_uses_neuromorphic(self, small_striatum, tmp_path):
        """Small regions should use neuromorphic format.

        BEHAVIORAL CONTRACT: Test by saving checkpoint and inspecting
        the selected_format in hybrid_metadata (public contract).
        """
        import torch

        # Save checkpoint (format auto-selected)
        checkpoint_path = tmp_path / "small_striatum.pt"
        small_striatum.checkpoint_manager.save(checkpoint_path)

        # Load checkpoint and inspect PUBLIC metadata
        state = torch.load(checkpoint_path, weights_only=False)
        assert "hybrid_metadata" in state, "Checkpoint should have hybrid_metadata"
        assert state["hybrid_metadata"]["selected_format"] == "neuromorphic", \
            "Small region should use neuromorphic format"

    def test_small_region_uses_neuromorphic_population_coding(self, small_striatum_population, tmp_path):
        """Small regions with population coding should use neuromorphic format.

        BEHAVIORAL CONTRACT: Test actual saved format, not internal decision logic.
        """
        import torch

        # 5 actions × 10 neurons = 50 neurons (still small, < 100)
        checkpoint_path = tmp_path / "small_striatum_pop.pt"
        small_striatum_population.checkpoint_manager.save(checkpoint_path)

        # Inspect PUBLIC metadata
        state = torch.load(checkpoint_path, weights_only=False)
        assert state["hybrid_metadata"]["selected_format"] == "neuromorphic", \
            "Small region with population coding should use neuromorphic format"

    def test_large_region_uses_elastic_tensor(self, large_striatum, tmp_path):
        """Large regions should use elastic tensor format.

        BEHAVIORAL CONTRACT: Validate format from saved checkpoint metadata.
        """
        import torch

        checkpoint_path = tmp_path / "large_striatum.pt"
        large_striatum.checkpoint_manager.save(checkpoint_path)

        # Inspect PUBLIC metadata
        state = torch.load(checkpoint_path, weights_only=False)
        assert state["hybrid_metadata"]["selected_format"] == "elastic_tensor", \
            "Large region should use elastic tensor format"

    def test_large_region_uses_elastic_tensor_population_coding(self, large_striatum_population, tmp_path):
        """Large regions with population coding should use elastic tensor format.

        BEHAVIORAL CONTRACT: Test the actual format used, not decision method.
        """
        import torch

        # 150 actions × 10 neurons = 1500 neurons (large, >> 100)
        checkpoint_path = tmp_path / "large_striatum_pop.pt"
        large_striatum_population.checkpoint_manager.save(checkpoint_path)

        # Inspect PUBLIC metadata
        state = torch.load(checkpoint_path, weights_only=False)
        assert state["hybrid_metadata"]["selected_format"] == "elastic_tensor", \
            "Large region with population coding should use elastic tensor format"

    def test_growth_enabled_small_uses_neuromorphic(self, device, tmp_path):
        """Small regions with growth enabled should use neuromorphic.

        BEHAVIORAL CONTRACT: Test actual saved format.
        """
        import torch

        config = StriatumConfig(
            n_actions=80,
            neurons_per_action=1,
            input_sources={'default': 100},
            growth_enabled=True,
            device=device,
        )
        striatum = Striatum(config)

        checkpoint_path = tmp_path / "growth_enabled.pt"
        striatum.checkpoint_manager.save(checkpoint_path)

        state = torch.load(checkpoint_path, weights_only=False)
        assert state["hybrid_metadata"]["selected_format"] == "neuromorphic", \
            "Small region with growth enabled should use neuromorphic format"

    def test_threshold_boundary(self, device, tmp_path):
        """Test format selection at size threshold.

        BEHAVIORAL CONTRACT: Verify format from actual saved checkpoints.
        """
        import torch

        # Just under threshold (100) - should use neuromorphic
        config_99 = StriatumConfig(n_actions=99, neurons_per_action=1, input_sources={'default': 100}, device=device)
        striatum_99 = Striatum(config_99)

        checkpoint_99 = tmp_path / "threshold_99.pt"
        striatum_99.checkpoint_manager.save(checkpoint_99)
        state_99 = torch.load(checkpoint_99, weights_only=False)
        assert state_99["hybrid_metadata"]["selected_format"] == "neuromorphic", \
            "99 neurons should use neuromorphic"

        # Just over threshold - should use elastic
        config_101 = StriatumConfig(n_actions=101, neurons_per_action=1, input_sources={'default': 100}, device=device, growth_enabled=False)
        striatum_101 = Striatum(config_101)

        checkpoint_101 = tmp_path / "threshold_101.pt"
        striatum_101.checkpoint_manager.save(checkpoint_101)
        state_101 = torch.load(checkpoint_101, weights_only=False)
        assert state_101["hybrid_metadata"]["selected_format"] == "elastic_tensor", \
            "101 neurons should use elastic tensor"


class TestHybridSaveLoad:
    """Test save/load with automatic format selection."""

    def test_save_small_creates_neuromorphic(self, small_striatum, tmp_path):
        """Saving small region should create neuromorphic checkpoint."""
        checkpoint_path = tmp_path / "small.ckpt"

        info = small_striatum.checkpoint_manager.save(checkpoint_path)

        assert info["format"] == "neuromorphic"
        assert checkpoint_path.exists()

        # Load and verify format
        loaded = torch.load(checkpoint_path, weights_only=False)
        assert loaded["format"] == "neuromorphic"
        assert "neurons" in loaded

    def test_save_small_creates_neuromorphic_population_coding(self, small_striatum_population, tmp_path):
        """Saving small region with population coding should create neuromorphic checkpoint."""
        checkpoint_path = tmp_path / "small_pop.ckpt"

        info = small_striatum_population.checkpoint_manager.save(checkpoint_path)

        assert info["format"] == "neuromorphic"
        assert checkpoint_path.exists()

        # Load and verify format
        loaded = torch.load(checkpoint_path, weights_only=False)
        assert loaded["format"] == "neuromorphic"
        assert "neurons" in loaded
        # Should have 50 neurons (5 actions × 10 neurons/action)
        assert len(loaded["neurons"]) == 50

    def test_save_large_creates_elastic(self, large_striatum, tmp_path):
        """Saving large region should create elastic tensor checkpoint."""
        checkpoint_path = tmp_path / "large.ckpt"

        info = large_striatum.checkpoint_manager.save(checkpoint_path)

        assert info["format"] == "elastic_tensor"
        assert checkpoint_path.exists()

        # Load and verify format
        loaded = torch.load(checkpoint_path, weights_only=False)
        assert loaded["format_version"] == "1.0.0"  # Elastic format version
        assert "hybrid_metadata" in loaded  # Has hybrid metadata

    def test_load_auto_detects_neuromorphic(self, small_striatum, tmp_path):
        """load() should auto-detect neuromorphic format."""
        checkpoint_path = tmp_path / "neuro.ckpt"

        # Save neuromorphic
        small_striatum.checkpoint_manager.save(checkpoint_path)

        # Reset and load
        small_striatum.reset_state()
        small_striatum.checkpoint_manager.load(checkpoint_path)

        # Should successfully load (no exceptions)

    def test_load_auto_detects_neuromorphic_population_coding(self, small_striatum_population, tmp_path):
        """load() should auto-detect neuromorphic format with population coding."""
        checkpoint_path = tmp_path / "neuro_pop.ckpt"

        # Save neuromorphic
        small_striatum_population.checkpoint_manager.save(checkpoint_path)

        # Reset and load
        small_striatum_population.reset_state()
        small_striatum_population.checkpoint_manager.load(checkpoint_path)

        # Should successfully load (no exceptions)

    def test_load_auto_detects_elastic(self, large_striatum, tmp_path):
        """load() should auto-detect elastic tensor format."""
        checkpoint_path = tmp_path / "elastic.ckpt"

        # Save elastic
        large_striatum.checkpoint_manager.save(checkpoint_path)

        # Reset and load
        large_striatum.reset_state()
        large_striatum.checkpoint_manager.load(checkpoint_path)

        # Should successfully load (no exceptions)

    def test_hybrid_metadata_included(self, small_striatum, tmp_path):
        """Hybrid checkpoints should include metadata about format selection."""
        checkpoint_path = tmp_path / "metadata.ckpt"

        small_striatum.checkpoint_manager.save(checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        assert "hybrid_metadata" in loaded
        assert loaded["hybrid_metadata"]["auto_selected"] is True
        assert "selected_format" in loaded["hybrid_metadata"]
        assert "selection_criteria" in loaded["hybrid_metadata"]

    def test_state_preserved_across_format_change(self, device, tmp_path):
        """State should be preserved when loading into different size (format)."""
        checkpoint_path = tmp_path / "state_preserve.ckpt"

        # Create small striatum (neuromorphic)
        small_config = StriatumConfig(n_actions=5, neurons_per_action=1, input_sources={'default': 100}, device=device)
        small = Striatum(small_config)
        small.reset_state()

        # Set distinctive weights
        small.d1_pathway.weights.data[0, 10] = 0.777
        small.d1_pathway.weights.data[1, 20] = 0.888

        # Save
        small.checkpoint_manager.save(checkpoint_path)

        # Load into same small striatum
        small2 = Striatum(small_config)
        small2.reset_state()
        small2.checkpoint_manager.load(checkpoint_path)

        # Verify weights restored
        assert abs(small2.d1_pathway.weights[0, 10].item() - 0.777) < 1e-6
        assert abs(small2.d1_pathway.weights[1, 20].item() - 0.888) < 1e-6

    def test_state_preserved_with_population_coding(self, device, tmp_path):
        """State should be preserved with population coding enabled."""
        checkpoint_path = tmp_path / "state_preserve_pop.ckpt"

        # Create small striatum with population coding (neuromorphic)
        config = StriatumConfig(
            n_actions=5,
            neurons_per_action=10,
            input_sources={'default': 100},
            device=device,
        )
        striatum = Striatum(config)
        striatum.reset_state()

        # Set distinctive weights in first few neurons
        striatum.d1_pathway.weights.data[0, 10] = 0.333
        striatum.d1_pathway.weights.data[5, 20] = 0.444  # Different action's neurons

        # Save
        striatum.checkpoint_manager.save(checkpoint_path)

        # Load into new instance
        striatum2 = Striatum(config)
        striatum2.reset_state()
        striatum2.checkpoint_manager.load(checkpoint_path)

        # Verify weights restored
        assert abs(striatum2.d1_pathway.weights[0, 10].item() - 0.333) < 1e-6
        assert abs(striatum2.d1_pathway.weights[5, 20].item() - 0.444) < 1e-6

    def test_load_rejects_checkpoint_without_metadata(self, small_striatum, tmp_path):
        """Loading checkpoint without hybrid_metadata should raise error."""
        checkpoint_path = tmp_path / "no_metadata.ckpt"

        # Create checkpoint without hybrid_metadata
        state = small_striatum.checkpoint_manager.get_full_state()
        torch.save(state, checkpoint_path)

        # Should raise error
        with pytest.raises(ValueError, match="hybrid_metadata"):
            small_striatum.checkpoint_manager.load(checkpoint_path)
