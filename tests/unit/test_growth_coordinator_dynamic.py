"""Unit tests for GrowthCoordinator - coordinated region and pathway growth."""

from unittest.mock import Mock
import pytest
from thalia.coordination.growth import GrowthCoordinator


def test_growth_coordinator_initialization():
    """GrowthCoordinator should initialize and track growth history."""
    brain = Mock()
    brain.pathway_manager = Mock()
    brain.regions = {}

    coordinator = GrowthCoordinator(brain)

    # Test contract: coordinator should start with empty history
    history = coordinator.get_growth_history()
    assert len(history) == 0, "New coordinator should have empty history"


def test_coordinate_growth_missing_region():
    """Should raise KeyError for missing region."""
    brain = Mock()
    brain.regions = {}
    brain.pathway_manager = Mock()

    coordinator = GrowthCoordinator(brain)

    with pytest.raises(KeyError, match="Region 'unknown' not found"):
        coordinator.coordinate_growth('unknown', n_new_neurons=10)


def test_state_persistence():
    """Growth coordinator should support checkpointing."""
    brain = Mock()
    brain.regions = {}
    brain.pathway_manager = Mock()

    coordinator = GrowthCoordinator(brain)

    # Add mock history
    coordinator.history = [
        {'region': 'cortex', 'n_neurons_added': 100},
        {'region': 'hippocampus', 'n_neurons_added': 50},
    ]

    # Save state
    n_history_entries = 2
    state = coordinator.get_state()
    assert 'history' in state, "State should contain history"
    assert len(state['history']) == n_history_entries, \
        f"State should preserve {n_history_entries} history entries"

    # Load state into new coordinator
    new_coordinator = GrowthCoordinator(brain)
    new_coordinator.load_state(state)

    # Test contract: loaded coordinator should match original
    assert len(new_coordinator.history) == len(coordinator.history), \
        "Loaded coordinator should have same history length"
    assert new_coordinator.history[0]['region'] == 'cortex', "First entry should be cortex"
    assert new_coordinator.history[1]['region'] == 'hippocampus', "Second entry should be hippocampus"


def test_growth_history_retrieval():
    """Should retrieve growth history."""
    brain = Mock()
    brain.regions = {}
    brain.pathway_manager = Mock()

    coordinator = GrowthCoordinator(brain)
    test_events = [{'test': 'event1'}, {'test': 'event2'}]
    coordinator.history = test_events

    history = coordinator.get_growth_history()
    # Test contract: should return all history entries
    assert len(history) == len(test_events), \
        f"Should return all {len(test_events)} history entries"
    assert history[0]['test'] == 'event1', "First entry should match"


def test_coordinate_growth_with_real_brain():
    """Integration test: coordinate growth on real brain components.

    This is an exact copy of the test but using DynamicBrain instead of EventDrivenBrain.

    This test complements the unit tests above by verifying that growth
    coordination works with actual DynamicBrain regions and pathways,
    not just mocks.

    Cortex uses grow_output() API for growth (not a separate growth_manager).
    """
    import torch
    from thalia.core.dynamic_brain import DynamicBrain
    from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

    # Create small real brain
    device = torch.device("cpu")
    config = ThaliaConfig(
        global_=GlobalConfig(device=str(device)),
        brain=BrainConfig(
            device=str(device),  # Must match global_.device
            sizes=RegionSizes(
                input_size=10,
                thalamus_size=20,
                cortex_size=30,
                hippocampus_size=40,
                pfc_size=20,
                n_actions=5,
            ),
        ),
    )
    brain = DynamicBrain.from_thalia_config(config)

    # Get initial sizes
    initial_cortex_size = brain.components['cortex'].config.n_output

    # Coordinate growth on cortex
    coordinator = GrowthCoordinator(brain)
    events = coordinator.coordinate_growth(
        region_name='cortex',
        n_new_neurons=10,
        reason='integration test'
    )

    # Contract: Should return events for region + connected pathways
    assert len(events) > 0, "Should return growth events"
    assert events[0].component_name == 'cortex', "First event should be cortex region"
    assert events[0].n_neurons_added == 10, "Should add 10 neurons to cortex"

    # Contract: Region should have grown
    new_cortex_size = brain.components['cortex'].config.n_output
    # Cortex distributes n_new across layers (L4/L2/3/L5), so actual growth may differ
    # from requested n_new due to layer ratios. Just verify it grew.
    assert new_cortex_size > initial_cortex_size, \
        f"Cortex should grow from {initial_cortex_size}, got {new_cortex_size}"

    # Contract: History should track operation
    assert len(coordinator.history) > 0, "Should have history entry"
    assert coordinator.history[-1]['region'] == 'cortex', \
        "History should record cortex growth"
    assert coordinator.history[-1]['n_neurons_added'] == 10, \
        "History should record correct neuron count"
