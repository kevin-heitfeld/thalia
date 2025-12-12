"""
Unit tests for GrowthCoordinator - coordinated region and pathway growth.
"""

import pytest
from unittest.mock import Mock
from thalia.coordination.growth import GrowthCoordinator, GrowthEvent


def test_growth_coordinator_initialization():
    """GrowthCoordinator should initialize with brain reference."""
    brain = Mock()
    brain.pathway_manager = Mock()
    brain.pathway_manager.pathways = {}
    brain.regions = {}
    
    coordinator = GrowthCoordinator(brain)
    
    assert coordinator.brain is brain
    assert coordinator.pathway_manager is brain.pathway_manager
    assert isinstance(coordinator.history, list)
    assert len(coordinator.history) == 0


def test_find_input_pathways():
    """Coordinator should find pathways targeting a region."""
    brain = Mock()
    brain.regions = {'cortex': Mock(), 'hippocampus': Mock()}
    
    # Create mock pathways
    pathway1 = Mock()
    pathway1.target_name = 'cortex'
    pathway1.source_name = 'visual'
    
    pathway2 = Mock()
    pathway2.target_name = 'hippocampus'
    pathway2.source_name = 'cortex'
    
    pathway3 = Mock()
    pathway3.target_name = 'cortex'
    pathway3.source_name = 'auditory'
    
    brain.pathway_manager = Mock()
    brain.pathway_manager.pathways = {
        'visual_to_cortex': pathway1,
        'cortex_to_hippocampus': pathway2,
        'auditory_to_cortex': pathway3,
    }
    
    coordinator = GrowthCoordinator(brain)
    input_pathways = coordinator._find_input_pathways('cortex')
    
    # Should find 2 input pathways to cortex
    assert len(input_pathways) == 2
    pathway_names = [name for name, _ in input_pathways]
    assert 'visual_to_cortex' in pathway_names
    assert 'auditory_to_cortex' in pathway_names


def test_find_output_pathways():
    """Coordinator should find pathways sourced from a region."""
    brain = Mock()
    brain.regions = {'cortex': Mock(), 'hippocampus': Mock(), 'striatum': Mock()}
    
    # Create mock pathways
    pathway1 = Mock()
    pathway1.source_name = 'cortex'
    pathway1.target_name = 'hippocampus'
    
    pathway2 = Mock()
    pathway2.source_name = 'cortex'
    pathway2.target_name = 'striatum'
    
    pathway3 = Mock()
    pathway3.source_name = 'hippocampus'
    pathway3.target_name = 'cortex'
    
    brain.pathway_manager = Mock()
    brain.pathway_manager.pathways = {
        'cortex_to_hippocampus': pathway1,
        'cortex_to_striatum': pathway2,
        'hippocampus_to_cortex': pathway3,
    }
    
    coordinator = GrowthCoordinator(brain)
    output_pathways = coordinator._find_output_pathways('cortex')
    
    # Should find 2 output pathways from cortex
    assert len(output_pathways) == 2
    pathway_names = [name for name, _ in output_pathways]
    assert 'cortex_to_hippocampus' in pathway_names
    assert 'cortex_to_striatum' in pathway_names


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
    
    # Should have 3 events: region + 2 pathways
    assert len(events) == 3
    assert events[0].component_name == 'cortex'
    assert events[1].component_name == 'visual_to_cortex'
    assert events[2].component_name == 'cortex_to_hippocampus'
    
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
    
    # Verify history recorded
    assert len(coordinator.history) == 1
    assert coordinator.history[0]['region'] == 'cortex'
    assert coordinator.history[0]['n_neurons_added'] == 100


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
    state = coordinator.get_state()
    assert 'history' in state
    assert len(state['history']) == 2
    
    # Load state into new coordinator
    new_coordinator = GrowthCoordinator(brain)
    new_coordinator.load_state(state)
    
    assert len(new_coordinator.history) == 2
    assert new_coordinator.history[0]['region'] == 'cortex'
    assert new_coordinator.history[1]['region'] == 'hippocampus'


def test_growth_history_retrieval():
    """Should retrieve growth history."""
    brain = Mock()
    brain.regions = {}
    brain.pathway_manager = Mock()
    brain.pathway_manager.pathways = {}
    
    coordinator = GrowthCoordinator(brain)
    coordinator.history = [{'test': 'event1'}, {'test': 'event2'}]
    
    history = coordinator.get_growth_history()
    assert len(history) == 2
    assert history[0]['test'] == 'event1'
