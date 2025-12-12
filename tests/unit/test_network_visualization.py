"""
Unit tests for network topology visualization.

Tests verify that visualization functions correctly handle brain structures
without requiring NetworkX (graceful degradation).
"""

import pytest
from unittest.mock import Mock


def test_visualize_requires_networkx():
    """Visualization should raise ImportError if networkx not installed."""
    from thalia.visualization import network_graph
    
    # Temporarily disable networkx
    original_flag = network_graph.NETWORKX_AVAILABLE
    network_graph.NETWORKX_AVAILABLE = False
    
    try:
        # Create mock brain
        brain = Mock()
        brain.regions = {'cortex': Mock()}
        brain.pathway_manager = Mock()
        brain.pathway_manager.pathways = {}
        
        with pytest.raises(ImportError, match="NetworkX is required"):
            network_graph.visualize_brain_topology(brain)
    finally:
        # Restore flag
        network_graph.NETWORKX_AVAILABLE = original_flag


def test_export_graphviz_basic(tmp_path):
    """Export to DOT format should work without networkx."""
    from thalia.visualization.network_graph import export_topology_to_graphviz
    
    # Create mock brain
    brain = Mock()
    
    # Mock regions
    region1 = Mock()
    region1.n_neurons = 100
    region2 = Mock()
    region2.n_neurons = 50
    
    brain.regions = {
        'cortex': region1,
        'striatum': region2,
    }
    
    # Mock pathways
    pathway = Mock()
    pathway.source_name = 'cortex'
    pathway.target_name = 'striatum'
    pathway.weights = Mock()
    pathway.weights.mean.return_value.item.return_value = 0.5
    
    brain.pathway_manager = Mock()
    brain.pathway_manager.pathways = {'cortex_to_striatum': pathway}
    
    # Export
    output_file = tmp_path / "topology.dot"
    export_topology_to_graphviz(brain, str(output_file))
    
    # Verify file contents
    content = output_file.read_text()
    assert 'digraph BrainTopology' in content
    assert '"cortex"' in content
    assert '"striatum"' in content
    assert '100N' in content
    assert '50N' in content
    assert 'cortex" -> "striatum"' in content
    assert '0.500' in content


def test_get_neuron_count_various_attributes():
    """Helper should handle different region implementations."""
    from thalia.visualization.network_graph import _get_neuron_count
    
    # Region with n_neurons attribute
    region1 = Mock()
    region1.n_neurons = 100
    assert _get_neuron_count(region1) == 100
    
    # Region with config.n_neurons
    region2 = Mock()
    region2.config = Mock()
    region2.config.n_neurons = 50
    del region2.n_neurons  # Remove direct attribute
    assert _get_neuron_count(region2) == 50
    
    # Region with membrane shape
    region3 = Mock()
    region3.membrane = Mock()
    region3.membrane.shape = (75,)
    del region3.n_neurons
    del region3.config
    assert _get_neuron_count(region3) == 75
    
    # Region with no attributes
    region4 = Mock()
    del region4.n_neurons
    del region4.config
    del region4.membrane
    assert _get_neuron_count(region4) == 0


def test_get_region_type_classification():
    """Helper should correctly classify regions by name."""
    from thalia.visualization.network_graph import _get_region_type
    
    assert _get_region_type('visual_cortex') == 'sensory'
    assert _get_region_type('auditory_cortex') == 'sensory'
    assert _get_region_type('primary_cortex') == 'cortex'
    assert _get_region_type('striatum') == 'striatum'
    assert _get_region_type('hippocampus') == 'hippocampus'
    assert _get_region_type('cerebellum') == 'cerebellum'
    assert _get_region_type('thalamus') == 'thalamus'
    assert _get_region_type('vta') == 'vta'
    assert _get_region_type('prefrontal_cortex') == 'prefrontal'
    assert _get_region_type('motor_cortex') == 'motor'
    assert _get_region_type('unknown_region') == 'other'


def test_get_pathway_strength():
    """Helper should extract average weight from pathway."""
    from thalia.visualization.network_graph import _get_pathway_strength
    import torch
    
    # Pathway with weights
    pathway1 = Mock()
    pathway1.weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    assert abs(_get_pathway_strength(pathway1) - 0.25) < 1e-6
    
    # Pathway with None weights
    pathway2 = Mock()
    pathway2.weights = None
    assert _get_pathway_strength(pathway2) == 0.0
    
    # Pathway without weights attribute
    pathway3 = Mock()
    del pathway3.weights
    assert _get_pathway_strength(pathway3) == 0.0


def test_hierarchical_layout_structure():
    """Hierarchical layout should organize regions by type."""
    from thalia.visualization.network_graph import _hierarchical_layout
    import networkx as nx
    
    G = nx.DiGraph()
    region_info = {
        'visual': {'type': 'sensory', 'neurons': 100},
        'cortex': {'type': 'cortex', 'neurons': 200},
        'striatum': {'type': 'striatum', 'neurons': 150},
        'motor': {'type': 'motor', 'neurons': 80},
    }
    
    for name in region_info.keys():
        G.add_node(name)
    
    pos = _hierarchical_layout(G, region_info)
    
    # Check that all nodes have positions
    assert len(pos) == 4
    assert all(name in pos for name in region_info.keys())
    
    # Check that positions are tuples of 2 floats
    for name, (x, y) in pos.items():
        assert isinstance(x, float)
        assert isinstance(y, float)
    
    # Check layer ordering (sensory at top, motor at bottom)
    assert pos['visual'][1] > pos['cortex'][1]  # Sensory higher than cortex
    assert pos['cortex'][1] > pos['striatum'][1]  # Cortex higher than striatum
    assert pos['striatum'][1] > pos['motor'][1]  # Striatum higher than motor


@pytest.mark.skip(reason="Requires networkx and matplotlib - run manually")
def test_full_visualization():
    """Integration test with real brain (requires dependencies)."""
    from thalia.visualization import visualize_brain_topology
    import matplotlib.pyplot as plt
    
    # Create mock brain with realistic structure
    brain = Mock()
    
    # Create regions
    regions = {}
    for name, n_neurons in [('visual', 100), ('cortex', 200), ('striatum', 150)]:
        region = Mock()
        region.n_neurons = n_neurons
        regions[name] = region
    brain.regions = regions
    
    # Create pathways
    pathway1 = Mock()
    pathway1.source_name = 'visual'
    pathway1.target_name = 'cortex'
    pathway1.weights = Mock()
    pathway1.weights.mean.return_value.item.return_value = 0.5
    
    pathway2 = Mock()
    pathway2.source_name = 'cortex'
    pathway2.target_name = 'striatum'
    pathway2.weights = Mock()
    pathway2.weights.mean.return_value.item.return_value = 0.3
    
    brain.pathway_manager = Mock()
    brain.pathway_manager.pathways = {
        'visual_to_cortex': pathway1,
        'cortex_to_striatum': pathway2,
    }
    
    # Generate visualization
    G = visualize_brain_topology(brain, layout='hierarchical')
    
    # Verify graph structure
    assert len(G.nodes()) == 3
    assert len(G.edges()) == 2
    
    plt.close('all')
