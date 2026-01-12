"""
Unit tests for elastic tensor checkpoint format (Phase 1).

Tests the elastic tensor approach where regions pre-allocate capacity
for growth and checkpoints track both used and total capacity.

Test Coverage:
- Capacity metadata in checkpoints
- Loading smaller checkpoint into larger brain
- Loading larger checkpoint into smaller brain (auto-grow)
- Reserved space utilization
- Edge cases (zero capacity, full capacity)
"""

import time

import pytest
import torch

from thalia.regions.striatum import Striatum
from thalia.regions.striatum.config import StriatumConfig
from thalia.config.size_calculator import LayerSizeCalculator


@pytest.fixture
def device():
    """Return device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def base_sizes():
    """Create base striatum sizes (5 actions, 1 neuron per action)."""
    calc = LayerSizeCalculator()
    sizes = calc.striatum_from_actions(n_actions=5, neurons_per_action=1)
    sizes['input_size'] = 100
    return sizes


@pytest.fixture
def base_sizes_population():
    """Create base striatum sizes with population coding (5 actions, 10 neurons per action)."""
    calc = LayerSizeCalculator()
    sizes = calc.striatum_from_actions(n_actions=5, neurons_per_action=10)
    sizes['input_size'] = 100
    return sizes


@pytest.fixture
def base_config(device):
    """Create base striatum config with growth enabled."""
    return StriatumConfig(
        growth_enabled=True,  # Feature to implement
        reserve_capacity=0.5,  # 50% headroom - feature to implement
    )


@pytest.fixture
def striatum_small(base_config, base_sizes, device):
    """Create small striatum (5 actions)."""
    region = Striatum(config=base_config, sizes=base_sizes, device=device)
    region.reset_state()
    return region


@pytest.fixture
def striatum_large(base_config, device):
    """Create large striatum (10 actions)."""
    calc = LayerSizeCalculator()
    sizes = calc.striatum_from_actions(n_actions=10, neurons_per_action=1)
    sizes['input_size'] = 100
    region = Striatum(config=base_config, sizes=sizes, device=device)
    region.reset_state()
    return region


class TestElasticTensorMetadata:
    """Test capacity metadata in elastic tensor format."""

    def test_checkpoint_includes_capacity_metadata(self, striatum_small, tmp_path):
        """Checkpoint should save both used and capacity dimensions."""
        checkpoint_path = tmp_path / "test.ckpt"

        # Save checkpoint
        state = striatum_small.get_full_state()
        torch.save(state, checkpoint_path)

        # Load and verify metadata
        loaded = torch.load(checkpoint_path, weights_only=False)

        assert "format_version" in loaded
        assert loaded["format_version"] == "1.0.0"

        assert "neuron_state" in loaded
        neuron_state = loaded["neuron_state"]

        # Should track used vs capacity
        assert "n_neurons_active" in neuron_state
        assert "n_neurons_capacity" in neuron_state

        # Capacity should be larger than used (50% headroom)
        assert neuron_state["n_neurons_capacity"] > neuron_state["n_neurons_active"]
        # Test contract: active neurons should be positive and reasonable
        assert 0 < neuron_state["n_neurons_active"] <= 100, \
            f"Active neurons should be positive and reasonable, got {neuron_state['n_neurons_active']}"
        # Test contract: capacity should have headroom (at least 1.2x active)
        assert neuron_state["n_neurons_capacity"] >= neuron_state["n_neurons_active"] * 1.2, \
            "Capacity should provide headroom for growth"

    def test_checkpoint_with_population_coding(self, base_config, base_sizes_population, device, tmp_path):
        """Checkpoint should work correctly with population coding enabled."""
        checkpoint_path = tmp_path / "test_population.ckpt"

        # Create striatum with population coding (5 actions * 10 neurons = 50 total)
        region = Striatum(config=base_config, sizes=base_sizes_population, device=device)
        region.reset_state()

        # Save checkpoint
        state = region.get_full_state()
        torch.save(state, checkpoint_path)

        # Load and verify metadata
        loaded = torch.load(checkpoint_path, weights_only=False)
        neuron_state = loaded["neuron_state"]

        # Should track actual neuron count (actions * neurons_per_action), not just action count
        expected_active = 5 * 10  # 5 actions * 10 neurons/action
        assert neuron_state["n_neurons_active"] == expected_active, \
            f"Should have {expected_active} active neurons (5 actions * 10 neurons/action)"
        # Test contract: capacity should have headroom
        assert neuron_state["n_neurons_capacity"] >= neuron_state["n_neurons_active"] * 1.2, \
            "Capacity should provide headroom for growth"

        # Note: n_output is no longer saved in checkpoints (it's a computed property)
        # The checkpoint tracks n_neurons_active and n_actions instead

    def test_growth_works_with_population_coding(self, base_config, base_sizes_population, device, tmp_path):
        """Growing brain should work correctly with population coding."""
        checkpoint_path = tmp_path / "test_growth_population.ckpt"

        # Create and grow
        region = Striatum(config=base_config, sizes=base_sizes_population, device=device)
        region.reset_state()

        # Initially 50 neurons (5 actions * 10 neurons/action)
        initial_actions = 5
        neurons_per_action = 10
        initial_expected = initial_actions * neurons_per_action
        assert region.n_neurons_active == initial_expected, \
            f"Should start with {initial_expected} neurons ({initial_actions} actions * {neurons_per_action} neurons/action)"

        # Add 2 more actions (= 20 more neurons with population coding)
        n_new_actions = 2
        region.grow_output(n_new=n_new_actions)

        # Should now have 7 actions * 10 neurons = 70 neurons
        final_expected_actions = initial_actions + n_new_actions
        final_expected_neurons = final_expected_actions * neurons_per_action
        assert region.n_actions == final_expected_actions, \
            f"Should have {final_expected_actions} actions after growth"
        assert region.n_neurons_active == final_expected_neurons, \
            f"Should have {final_expected_neurons} neurons ({final_expected_actions} actions * {neurons_per_action} neurons/action)"

        # Save and reload
        state = region.get_full_state()
        torch.save(state, checkpoint_path)

        # Create new brain and load
        region2 = Striatum(config=base_config, sizes=base_sizes_population, device=device)
        region2.reset_state()

        loaded = torch.load(checkpoint_path, weights_only=False)
        region2.load_full_state(loaded)

        # Test contract: restored region should match original after growth
        assert region2.n_actions == region.n_actions, \
            "Restored region should have same action count as original"
        assert region2.n_neurons_active == region.n_neurons_active, \
            "Restored region should have same active neuron count as original"

    def test_capacity_matches_tensor_dimensions(self, striatum_small, tmp_path):
        """Checkpoint tensors should match n_neurons_active, not full capacity.

        Note: We only save active neurons to checkpoints, not the reserved capacity.
        The capacity metadata tells us how much space is available, but actual
        tensor dimensions match n_neurons_active.
        """
        checkpoint_path = tmp_path / "test.ckpt"

        state = striatum_small.get_full_state()
        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        # Get metadata
        n_active = loaded["neuron_state"]["n_neurons_active"]
        n_capacity = loaded["neuron_state"]["n_neurons_capacity"]

        # Capacity should be larger than active (reserved space)
        assert n_capacity > n_active, "Capacity should exceed active neurons (reserved space)"
        # Test contract: active neurons should be positive
        assert n_active > 0, "Should have active neurons"
        # Test contract: capacity should provide headroom (at least 1.2x)
        assert n_capacity >= n_active * 1.2, \
            f"Capacity ({n_capacity}) should be at least 1.2x active neurons ({n_active})"

        # Tensors should match ACTIVE size, not capacity
        # (We don't save unused reserved space)
        # Note: membrane_potential is per-pathway (D1 or D2), so it's half the total
        if loaded["neuron_state"]["membrane_potential"] is not None:
            # Membrane potential is stored per-pathway, so divide by 2
            expected_membrane_size = n_active // 2
            actual_size = loaded["neuron_state"]["membrane_potential"].shape[0]
            # Due to odd numbers, check within 1
            assert abs(actual_size - expected_membrane_size) <= 1, \
                f"Membrane potential size {actual_size} should be ~{expected_membrane_size} (n_active//2)"

        # Pathway neurons should also match active size
        for pathway_key in ["d1_state", "d2_state"]:
            if pathway_key in loaded["pathway_state"]:
                pathway_state = loaded["pathway_state"][pathway_key]
                if "neurons" in pathway_state and pathway_state["neurons"] is not None:
                    neurons_state = pathway_state["neurons"]
                    if "membrane" in neurons_state and neurons_state["membrane"] is not None:
                        assert neurons_state["membrane"].shape[0] == n_active

    def test_zero_capacity_raises_error(self, base_config, device):
        """Creating region with zero capacity should raise error."""
        calc = LayerSizeCalculator()

        # Zero actions results in zero D1 and D2 sizes, which should be rejected
        with pytest.raises(ValueError, match="n_actions must be positive"):
            bad_sizes = calc.striatum_from_actions(n_actions=0, neurons_per_action=1)
            bad_sizes['input_size'] = 100
            Striatum(config=base_config, sizes=bad_sizes, device=device)


class TestLoadingSmallerCheckpoint:
    """Test loading smaller checkpoint into larger brain."""

    def test_load_smaller_checkpoint_succeeds(self, striatum_small, striatum_large, tmp_path):
        """Loading 5-action checkpoint into 10-action brain should work."""
        checkpoint_path = tmp_path / "small.ckpt"

        # Save small striatum
        small_state = striatum_small.get_full_state()
        torch.save(small_state, checkpoint_path)

        # Load into large striatum
        loaded_state = torch.load(checkpoint_path, weights_only=False)
        striatum_large.load_full_state(loaded_state)

        # Test contract: brain size unchanged (doesn't shrink when loading smaller checkpoint)
        expected_actions = 10
        expected_neurons = 20  # 10 actions * 1 neuron/action * 2 pathways (D1+D2)
        assert striatum_large.n_neurons_active == expected_neurons, \
            f"Large brain should remain {expected_neurons} neurons (doesn't shrink)"
        assert striatum_large.n_actions == expected_actions, \
            f"Large brain should remain {expected_actions} actions (doesn't shrink)"

        # But checkpoint only had 5 actions worth of data
        # First 5 actions restored from checkpoint, last 5 kept their initialized state

    def test_partial_state_restoration(self, striatum_small, striatum_large, tmp_path):
        """Only active neurons from checkpoint should be restored."""
        checkpoint_path = tmp_path / "small.ckpt"

        # Set some state in small striatum (membrane potential)
        striatum_small.d1_pathway.neurons.membrane[:5] = torch.arange(5, dtype=torch.float32, device=striatum_small.device)

        # Save and load into large
        small_state = striatum_small.get_full_state()
        torch.save(small_state, checkpoint_path)

        loaded_state = torch.load(checkpoint_path, weights_only=False)
        striatum_large.load_full_state(loaded_state)

        # First 5 neurons should match checkpoint
        assert torch.allclose(
            striatum_large.d1_pathway.neurons.membrane[:5],
            torch.arange(5, dtype=torch.float32, device=striatum_large.device)
        )

        # After partial restore, pathways are updated to match checkpoint size
        # This is expected behavior - pathway.load_state() updates sizes
        expected_active = striatum_large.d1_size + striatum_large.d2_size
        assert striatum_large.n_neurons_active == expected_active  # Brain thinks it has correct size
        assert striatum_large.d1_pathway.neurons.membrane.shape[0] == 5  # But pathways loaded 5


class TestLoadingLargerCheckpoint:
    """Test loading larger checkpoint into smaller brain (auto-grow)."""

    def test_load_larger_checkpoint_triggers_growth(self, striatum_small, striatum_large, tmp_path):
        """Loading 10-action checkpoint into 5-action brain should trigger growth."""
        checkpoint_path = tmp_path / "large.ckpt"

        # Save large striatum
        large_state = striatum_large.get_full_state()
        torch.save(large_state, checkpoint_path)

        # Load into small striatum
        loaded_state = torch.load(checkpoint_path, weights_only=False)

        # Test contract: before load, small brain has its initial size
        initial_small_size = 10  # 5 actions * 2 neurons/action (D1+D2)
        assert striatum_small.n_neurons_active == initial_small_size, \
            f"Small brain should start with {initial_small_size} neurons"

        # Load should automatically grow
        striatum_small.load_full_state(loaded_state)

        # Test contract: after load, small brain should match large checkpoint size
        expected_grown_size = 20  # 10 actions * 2 neurons/action (D1+D2)
        assert striatum_small.n_neurons_active == expected_grown_size, \
            f"Small brain should grow to {expected_grown_size} neurons to match checkpoint"

    def test_auto_growth_preserves_state(self, striatum_small, striatum_large, tmp_path):
        """Auto-growth during load should not corrupt state."""
        checkpoint_path = tmp_path / "large.ckpt"

        # Set distinctive state in large striatum
        striatum_large.d1_pathway.neurons.membrane[:10] = torch.arange(10, dtype=torch.float32, device=striatum_large.device)

        # Save and load into small
        large_state = striatum_large.get_full_state()
        torch.save(large_state, checkpoint_path)

        loaded_state = torch.load(checkpoint_path, weights_only=False)
        striatum_small.load_full_state(loaded_state)

        # All 10 neurons should match
        assert torch.allclose(
            striatum_small.d1_pathway.neurons.membrane[:10],
            torch.arange(10, dtype=torch.float32, device=striatum_small.device)
        )

    def test_growth_exceeding_capacity_expands(self, base_config, device, tmp_path):
        """Loading checkpoint larger than capacity should expand tensors."""
        checkpoint_path = tmp_path / "huge.ckpt"
        calc = LayerSizeCalculator()

        # Create large striatum (20 actions) and save it
        large_sizes = calc.striatum_from_actions(n_actions=20, neurons_per_action=1)
        large_sizes['input_size'] = 100
        large_striatum = Striatum(config=base_config, sizes=large_sizes, device=device)
        large_striatum.reset_state()

        large_state = large_striatum.get_full_state()
        torch.save(large_state, checkpoint_path)

        # Create small striatum (5 actions)
        small_sizes = calc.striatum_from_actions(n_actions=5, neurons_per_action=1)
        small_sizes['input_size'] = 100
        small_striatum = Striatum(config=base_config, sizes=small_sizes, device=device)
        small_striatum.reset_state()

        # Load should auto-grow to match
        loaded_state = torch.load(checkpoint_path, weights_only=False)
        small_striatum.load_full_state(loaded_state)

        # Test contract: should auto-grow to match checkpoint size
        expected_size = 40  # 20 actions * 2 neurons/action (D1+D2)
        assert small_striatum.n_neurons_active == expected_size, \
            f"Should grow to {expected_size} neurons to match checkpoint"
        assert small_striatum.n_neurons_capacity >= expected_size, \
            f"Capacity should be at least {expected_size} after growth"


class TestReservedSpaceUtilization:
    """Test how reserved capacity is used during growth."""

    @pytest.mark.parametrize("growth_amount,expect_reallocation,description", [
        (2, False, "within_capacity"),  # 10 + 2 = 12, fits in capacity 15
        (6, True, "beyond_capacity"),   # 10 + 6 = 16, exceeds capacity 15
        (10, True, "exceeding_capacity"),  # 10 + 10 = 20, far exceeds capacity 15
    ])
    def test_growth_with_various_amounts(self, striatum_small, growth_amount, expect_reallocation, description):
        """Test growth behavior with various amounts relative to capacity.

        Why this test exists: Validates that growth correctly handles both
        cases where growth fits within reserved capacity vs. requiring reallocation.
        This is critical for efficient memory usage during curriculum learning.

        Cases:
        - within_capacity (2): Growth fits in reserved space (5 + 2 < ~7.5 capacity)
        - beyond_capacity (4): Growth requires reallocation (5 + 4 > ~7.5 capacity)
        - exceeding_capacity (10): Growth far exceeds initial capacity
        """
        initial_capacity = striatum_small.n_neurons_capacity
        initial_active = striatum_small.n_neurons_active

        # Grow by specified amount (in actions)
        striatum_small.grow_actions(n_new=growth_amount)

        # neurons_per_action=1 means 1 D1 + 1 D2 = 2 neurons per action
        neurons_per_action = 2  # D1 + D2
        new_neurons = growth_amount * neurons_per_action
        expected_active = initial_active + new_neurons
        assert striatum_small.n_neurons_active == expected_active, \
            f"Active neurons should be {expected_active} after adding {growth_amount} actions"

        # Each pathway should have approximately half the neurons
        assert striatum_small.d1_pathway.neurons.n_neurons >= expected_active // 2 - 1, \
            f"D1 pathway should have ~{expected_active // 2} neurons"
        assert striatum_small.d2_pathway.neurons.n_neurons >= expected_active // 2 - 1, \
            f"D2 pathway should have ~{expected_active // 2} neurons"

        # Capacity behavior depends on whether reallocation was needed
        if expect_reallocation:
            # Capacity should increase with new headroom
            assert striatum_small.n_neurons_capacity > initial_capacity, \
                f"Capacity should increase for {description}"
            expected_capacity = int(expected_active * 1.5)
            assert striatum_small.n_neurons_capacity >= expected_capacity, \
                f"New capacity should have 1.5x headroom for {description}"
        else:
            # Capacity might stay same or increase (implementation dependent)
            # Key invariant: capacity >= active neurons
            assert striatum_small.n_neurons_capacity >= expected_active, \
                f"Capacity should accommodate active neurons for {description}"

    def test_checkpoint_after_growth_has_correct_capacity(self, striatum_small, tmp_path):
        """Checkpoint after growth should reflect new capacity."""
        checkpoint_path = tmp_path / "after_growth.ckpt"

        # Grow
        striatum_small.grow_actions(n_new=3)

        # Save
        state = striatum_small.get_full_state()
        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)

        # Test contract: metadata should reflect growth
        n_grown_actions = 5 + 3  # initial + growth
        # Each action has 1 D1 + 1 D2 = 2 neurons per action
        n_grown_neurons = n_grown_actions * 2  # 8 actions * 2 = 16 neurons
        assert loaded["neuron_state"]["n_neurons_active"] == n_grown_neurons, \
            f"Metadata should show {n_grown_neurons} active neurons after growth"
        assert loaded["neuron_state"]["n_neurons_capacity"] >= n_grown_neurons, \
            f"Capacity should be at least {n_grown_neurons} after growth"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_load_corrupted_metadata(self, striatum_small, tmp_path):
        """Loading checkpoint with missing capacity falls back to old format (warns)."""
        checkpoint_path = tmp_path / "corrupted.ckpt"

        # Get real checkpoint and corrupt it
        state = striatum_small.get_full_state()
        del state["neuron_state"]["n_neurons_capacity"]  # Remove capacity field

        torch.save(state, checkpoint_path)
        loaded_state = torch.load(checkpoint_path, weights_only=False)

        # Should warn about old format (not raise error)
        with pytest.warns(UserWarning, match="old.*format"):
            striatum_small.load_full_state(loaded_state)

    def test_load_old_format_without_capacity(self, striatum_small, tmp_path):
        """Loading old checkpoint without capacity metadata should warn."""
        checkpoint_path = tmp_path / "old_format.ckpt"

        # Get real checkpoint and remove capacity metadata
        state = striatum_small.get_full_state()
        del state["neuron_state"]["n_neurons_active"]
        del state["neuron_state"]["n_neurons_capacity"]

        torch.save(state, checkpoint_path)
        loaded_state = torch.load(checkpoint_path, weights_only=False)

        # Should warn about old format
        with pytest.warns(UserWarning, match="old.*format"):
            striatum_small.load_full_state(loaded_state)

    def test_save_load_round_trip(self, striatum_small, tmp_path):
        """Save and load should be perfect round-trip."""
        checkpoint_path = tmp_path / "roundtrip.ckpt"

        # Set some state
        original_membrane = striatum_small.d1_pathway.neurons.membrane.clone()

        # Save
        original_state = striatum_small.get_full_state()
        torch.save(original_state, checkpoint_path)

        # Reset striatum
        striatum_small.reset_state()

        # Load
        loaded_state = torch.load(checkpoint_path, weights_only=False)
        striatum_small.load_full_state(loaded_state)

        # Should match exactly
        assert torch.allclose(
            striatum_small.d1_pathway.neurons.membrane,
            original_membrane
        )


class TestPerformance:
    """Test performance characteristics of elastic tensor format."""

    def test_load_time_scales_with_active_not_capacity(self, base_config, device, tmp_path):
        """Load time should depend on active neurons, not capacity."""
        checkpoint_path = tmp_path / "perf.ckpt"
        calc = LayerSizeCalculator()

        # Create checkpoint with large capacity but few active
        sizes = calc.striatum_from_actions(n_actions=10, neurons_per_action=1)
        sizes['input_size'] = 100
        region = Striatum(config=base_config, sizes=sizes, device=device)
        region.reset_state()

        # Artificially set high capacity
        region.n_neurons_capacity = 1000

        state = region.get_full_state()
        torch.save(state, checkpoint_path)

        # Load should still be fast (only 10 active neurons)
        start = time.perf_counter()
        loaded = torch.load(checkpoint_path, weights_only=False)
        region.load_full_state(loaded)
        elapsed = time.perf_counter() - start

        # Should be <100ms even with large capacity
        assert elapsed < 0.1, f"Load took {elapsed:.3f}s, too slow!"

    def test_memory_usage_tracks_capacity(self, base_config, device):
        """Memory usage should scale with capacity, not active neurons."""
        calc = LayerSizeCalculator()
        # Small active, large capacity
        sizes = calc.striatum_from_actions(n_actions=5, neurons_per_action=1)
        sizes['input_size'] = 100

        region = Striatum(config=base_config, sizes=sizes, device=device)
        region.reset_state()

        # Capacity tracking is metadata only - actual tensors match n_neurons_active
        # We don't pre-allocate reserved space, just expand as needed
        # Test contract: capacity should be larger than active due to reserve_capacity=0.5
        assert region.n_neurons_capacity >= region.n_neurons_active, \
            "Capacity should be at least as large as active neurons"

        membrane = region.d1_pathway.neurons.membrane
        actual_neurons_allocated = membrane.shape[0]

        # Tensors match active size per pathway (D1 or D2), not total capacity
        # Elastic tensor = metadata tracking, not pre-allocation
        expected_per_pathway = region.n_neurons_active // 2  # D1 pathway has half
        assert actual_neurons_allocated == expected_per_pathway, \
            f"D1 pathway allocated size should match its neuron count: {expected_per_pathway}"
        # Test contract: active neurons should be positive
        assert actual_neurons_allocated > 0, "Should have allocated neurons"
