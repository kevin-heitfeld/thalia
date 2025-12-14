"""
Unit tests for GrowthCoordinator - coordinated region and pathway growth.
"""

from unittest.mock import Mock
import pytest
from thalia.coordination.growth import GrowthCoordinator, GrowthEvent


def test_growth_coordinator_initialization():
    """GrowthCoordinator should initialize and track growth history."""
    brain = Mock()
    brain.pathway_manager = Mock()
    brain.pathway_manager.pathways = {}
    brain.regions = {}

    coordinator = GrowthCoordinator(brain)

    # Test contract: coordinator should start with empty history
    history = coordinator.get_growth_history()
    assert len(history) == 0, "New coordinator should have empty history"


def test_find_input_pathways():
    """Coordinator should identify input pathways when region grows.

    BEHAVIORAL CONTRACT: When coordinate_growth is called for a region,
    all pathways targeting that region should grow their output dimensions.
    """
    brain = Mock()
    brain.regions = {'cortex': Mock(), 'hippocampus': Mock(), 'visual': Mock(), 'auditory': Mock()}

    # Create mock region with growth capability
    region = Mock()
    region.growth_manager = Mock()
    growth_event = GrowthEvent(
        timestamp='2025-12-12T10:00:00',
        component_name='cortex',
        component_type='region',
        event_type='add_neurons',
        n_neurons_added=100,
        reason='test growth',
    )
    region.growth_manager.add_neurons.return_value = growth_event
    brain.regions['cortex'] = region

    # Create mock pathways - input pathways TO cortex
    pathway1 = Mock()
    pathway1.target_name = 'cortex'
    pathway1.source_name = 'visual'
    pathway1.add_neurons = Mock(return_value=GrowthEvent(
        timestamp='2025-12-12T10:00:00',
        component_name='visual_to_cortex',
        component_type='pathway',
        event_type='add_neurons',
        n_neurons_added=100,
        reason='test growth',
    ))

    pathway2 = Mock()
    pathway2.target_name = 'hippocampus'
    pathway2.source_name = 'cortex'
    pathway2.add_neurons = Mock(return_value=GrowthEvent(
        timestamp='2025-12-12T10:00:00',
        component_name='cortex_to_hippocampus',
        component_type='pathway',
        event_type='add_neurons',
        n_neurons_added=100,
        reason='test growth',
    ))

    pathway3 = Mock()
    pathway3.target_name = 'cortex'
    pathway3.source_name = 'auditory'
    pathway3.add_neurons = Mock(return_value=GrowthEvent(
        timestamp='2025-12-12T10:00:00',
        component_name='auditory_to_cortex',
        component_type='pathway',
        event_type='add_neurons',
        n_neurons_added=100,
        reason='test growth',
    ))

    brain.pathway_manager = Mock()
    brain.pathway_manager.pathways = {
        'visual_to_cortex': pathway1,
        'cortex_to_hippocampus': pathway2,
        'auditory_to_cortex': pathway3,
    }

    coordinator = GrowthCoordinator(brain)
    events = coordinator.coordinate_growth(
        region_name='cortex',
        n_new_neurons=100,
        reason='test growth'
    )

    # BEHAVIORAL VALIDATION:
    # Both input pathways should have grown (n_new_post=100)
    pathway1.add_neurons.assert_called_once_with(
        n_new_pre=0, n_new_post=100, initialization='sparse_random', sparsity=0.1
    )
    pathway3.add_neurons.assert_called_once_with(
        n_new_pre=0, n_new_post=100, initialization='sparse_random', sparsity=0.1
    )
    # Output pathway should also grow (n_new_pre=100) for dimensional consistency
    pathway2.add_neurons.assert_called_once_with(
        n_new_pre=100, n_new_post=0, initialization='sparse_random', sparsity=0.1
    )
    # Should return events for region + all pathways (2 input + 1 output)
    assert len(events) >= 4


def test_coordinate_growth_finds_output_pathways():
    """Coordinator should identify and grow output pathways when region grows.

    BEHAVIORAL CONTRACT: When a region grows, all pathways sourced from that
    region should also grow to maintain dimensional consistency.
    """
    brain = Mock()

    # Create mock region
    region = Mock()
    region.growth_manager = Mock()
    growth_event = GrowthEvent(
        timestamp='2025-12-12T10:00:00',
        component_name='cortex',
        component_type='region',
        event_type='add_neurons',
        n_neurons_added=100,
        reason='test growth',
    )
    region.growth_manager.add_neurons.return_value = growth_event
    brain.regions = {'cortex': region, 'hippocampus': Mock(), 'striatum': Mock()}

    # Create mock pathways - output pathways FROM cortex
    pathway1 = Mock()
    pathway1.source_name = 'cortex'
    pathway1.target_name = 'hippocampus'
    pathway1.add_neurons = Mock(return_value=GrowthEvent(
        timestamp='2025-12-12T10:00:00',
        component_name='cortex_to_hippocampus',
        component_type='pathway',
        event_type='add_neurons',
        n_neurons_added=100,
        reason='coordinated growth',
    ))

    pathway2 = Mock()
    pathway2.source_name = 'cortex'
    pathway2.target_name = 'striatum'
    pathway2.add_neurons = Mock(return_value=GrowthEvent(
        timestamp='2025-12-12T10:00:00',
        component_name='cortex_to_striatum',
        component_type='pathway',
        event_type='add_neurons',
        n_neurons_added=100,
        reason='coordinated growth',
    ))

    pathway3 = Mock()
    pathway3.source_name = 'hippocampus'
    pathway3.target_name = 'cortex'
    pathway3.add_neurons = Mock(return_value=GrowthEvent(
        timestamp='2025-12-12T10:00:00',
        component_name='hippocampus_to_cortex',
        component_type='pathway',
        event_type='add_neurons',
        n_neurons_added=100,
        reason='coordinated growth',
    ))

    brain.pathway_manager = Mock()
    brain.pathway_manager.pathways = {
        'cortex_to_hippocampus': pathway1,
        'cortex_to_striatum': pathway2,
        'hippocampus_to_cortex': pathway3,
    }

    coordinator = GrowthCoordinator(brain)
    events = coordinator.coordinate_growth(
        region_name='cortex',
        n_new_neurons=100,
        reason='test growth'
    )

    # BEHAVIORAL VALIDATION: Both output pathways should have grown (n_new_pre=100)
    pathway1.add_neurons.assert_called_once_with(
        n_new_pre=100, n_new_post=0, initialization='sparse_random', sparsity=0.1
    )
    pathway2.add_neurons.assert_called_once_with(
        n_new_pre=100, n_new_post=0, initialization='sparse_random', sparsity=0.1
    )
    # Input pathway should also grow (n_new_post=100) for dimensional consistency
    pathway3.add_neurons.assert_called_once_with(
        n_new_pre=0, n_new_post=100, initialization='sparse_random', sparsity=0.1
    )
    # Should return events for region + all pathways (2 output + 1 input)
    assert len(events) >= 4


def test_coordinate_growth_basic():
    """Coordinator should grow region and connected pathways."""
    brain = Mock()

    # Create mock region with growth manager
    region = Mock()
    growth_manager = Mock()
    region.growth_manager = growth_manager

    # Mock growth event
    growth_event = GrowthEvent(
        timestamp='2025-12-12T10:00:00',
        component_name='cortex',
        component_type='region',
        event_type='add_neurons',
        n_neurons_added=100,
        reason='test growth',
    )
    growth_manager.add_neurons.return_value = growth_event

    brain.regions = {'cortex': region}

    # Create mock pathways
    input_pathway = Mock()
    input_pathway.target_name = 'cortex'
    input_pathway.source_name = 'visual'
    input_pathway.add_neurons = Mock(return_value=GrowthEvent(
        timestamp='2025-12-12T10:00:01',
        component_name='visual_to_cortex',
        component_type='pathway',
        event_type='add_neurons',
        n_neurons_added=100,
    ))

    output_pathway = Mock()
    output_pathway.source_name = 'cortex'
    output_pathway.target_name = 'hippocampus'
    output_pathway.add_neurons = Mock(return_value=GrowthEvent(
        timestamp='2025-12-12T10:00:02',
        component_name='cortex_to_hippocampus',
        component_type='pathway',
        event_type='add_neurons',
        n_neurons_added=100,
    ))

    brain.pathway_manager = Mock()
    brain.pathway_manager.pathways = {
        'visual_to_cortex': input_pathway,
        'cortex_to_hippocampus': output_pathway,
    }

    # Coordinate growth
    coordinator = GrowthCoordinator(brain)
    events = coordinator.coordinate_growth(
        region_name='cortex',
        n_new_neurons=100,
        reason='test growth',
    )

    # Test contract: should return events for region + all connected pathways
    expected_events = 3  # 1 region + 2 pathways
    assert len(events) == expected_events, \
        f"Should return {expected_events} events (1 region + 2 pathways)"
    assert events[0].component_name == 'cortex', "First event should be region"
    assert events[1].component_name == 'visual_to_cortex', "Second event should be input pathway"
    assert events[2].component_name == 'cortex_to_hippocampus', "Third event should be output pathway"

    # Verify pathway growth calls
    input_pathway.add_neurons.assert_called_once_with(
        n_new_pre=0,  # Input pathway: pre unchanged
        n_new_post=100,  # Post grows with region
        initialization='sparse_random',
        sparsity=0.1,
    )

    output_pathway.add_neurons.assert_called_once_with(
        n_new_pre=100,  # Output pathway: pre grows with region
        n_new_post=0,  # Post unchanged
        initialization='sparse_random',
        sparsity=0.1,
    )

    # Test contract: history should track growth operations
    assert len(coordinator.history) > 0, "Should have history entries"
    latest_entry = coordinator.history[-1]
    assert latest_entry['region'] == 'cortex', "Should record region name"
    assert latest_entry['n_neurons_added'] == 100, "Should record neuron count"


def test_coordinate_growth_missing_region():
    """Should raise KeyError for missing region."""
    brain = Mock()
    brain.regions = {}
    brain.pathway_manager = Mock()
    brain.pathway_manager.pathways = {}

    coordinator = GrowthCoordinator(brain)

    with pytest.raises(KeyError, match="Region 'unknown' not found"):
        coordinator.coordinate_growth('unknown', n_new_neurons=10)


def test_coordinate_growth_no_growth_manager():
    """Should raise AttributeError if region lacks growth_manager."""
    brain = Mock()
    region = Mock()
    del region.growth_manager  # No growth manager
    brain.regions = {'cortex': region}
    brain.pathway_manager = Mock()
    brain.pathway_manager.pathways = {}

    coordinator = GrowthCoordinator(brain)

    with pytest.raises(AttributeError, match="does not have growth_manager"):
        coordinator.coordinate_growth('cortex', n_new_neurons=10)


def test_state_persistence():
    """Growth coordinator should support checkpointing."""
    brain = Mock()
    brain.regions = {}
    brain.pathway_manager = Mock()
    brain.pathway_manager.pathways = {}

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
    brain.pathway_manager.pathways = {}

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

    This test complements the unit tests above by verifying that growth
    coordination works with actual EventDrivenBrain regions and pathways,
    not just mocks.
    """
    import torch
    from thalia.core.brain import EventDrivenBrain
    from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

    # Create small real brain
    device = torch.device("cpu")
    config = ThaliaConfig(
        global_=GlobalConfig(device=str(device)),
        brain=BrainConfig(sizes=RegionSizes(
            input_size=10,
            thalamus_size=20,
            cortex_size=30,
            hippocampus_size=40,
            pfc_size=20,
            n_actions=5,
        )),
    )
    brain = EventDrivenBrain.from_thalia_config(config)

    # Get initial sizes
    initial_cortex_size = brain.cortex.impl.config.n_output

    # Get pathway that should grow when cortex grows
    cortex_to_hippo = brain.pathway_manager.cortex_to_hippo
    initial_pathway_weights_shape = cortex_to_hippo.weights.shape

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
    new_cortex_size = brain.cortex.impl.config.n_output
    assert new_cortex_size == initial_cortex_size + 10, \
        f"Cortex should grow from {initial_cortex_size} to {initial_cortex_size + 10}"

    # Contract: Connected pathways should maintain dimensional consistency
    new_pathway_weights_shape = cortex_to_hippo.weights.shape
    assert new_pathway_weights_shape[1] == initial_pathway_weights_shape[1] + 10, \
        "Pathway input dimension should grow by 10 when source region (cortex) grows by 10"

    # Contract: History should track operation
    assert len(coordinator.history) > 0, "Should have history entry"
    assert coordinator.history[-1]['region'] == 'cortex', \
        "History should record cortex growth"
    assert coordinator.history[-1]['n_neurons_added'] == 10, \
        "History should record correct neuron count"
