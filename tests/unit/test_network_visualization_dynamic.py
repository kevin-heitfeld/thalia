"""
Unit tests for network topology visualization using DynamicBrain.

This is an exact copy of test_network_visualization.py but using DynamicBrain instead of EventDrivenBrain.

Tests verify that visualization functions correctly handle brain structures
without requiring NetworkX (graceful degradation).
"""

from unittest.mock import Mock
import pytest
import torch

from thalia.core.dynamic_brain import DynamicBrain
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes


@pytest.fixture
def device():
    """Return device for testing."""
    return torch.device("cpu")


@pytest.fixture
def small_test_brain(device):
    """Create minimal real brain for visualization tests."""
    config = ThaliaConfig(
        global_=GlobalConfig(device=str(device), dt_ms=1.0),
        brain=BrainConfig(
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
    brain.reset_state()
    return brain


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


def test_export_graphviz_basic(tmp_path, small_test_brain):
    """Export to DOT format should work without networkx."""
    from thalia.visualization.network_graph import export_topology_to_graphviz

    # Export real brain topology
    output_file = tmp_path / "topology.dot"
    export_topology_to_graphviz(small_test_brain, str(output_file))

    # Verify file contents contain actual brain structure
    content = output_file.read_text()
    assert 'digraph BrainTopology' in content, \
        "DOT file should have brain topology graph structure"

    # Check for actual brain regions (EventDrivenBrain has 6 regions)
    assert '"thalamus"' in content, "Should include thalamus region (sensory relay)"
    assert '"cortex"' in content, "Should include cortex region"
    assert '"hippocampus"' in content, "Should include hippocampus region"
    assert '"pfc"' in content, "Should include prefrontal cortex region"
    assert '"striatum"' in content, "Should include striatum region (action selection)"
    assert '"cerebellum"' in content, "Should include cerebellum region (motor control)"

    # Check that neuron counts are present (format: "XN")
    assert 'N' in content, "Should contain neuron count labels"

    # Verify pathways exist (format: "source" -> "target")
    assert '->' in content, "Should contain pathway connections"
    assert '"thalamus" -> "cortex"' in content, \
        "Should have thalamus to cortex pathway"

    # Check that weights are formatted correctly (0.000 format)
    import re
    weight_pattern = r'\d+\.\d{3}'
    assert re.search(weight_pattern, content), \
        "Should contain formatted pathway weights"


def test_get_neuron_count_various_attributes():
    """Helper should handle different region implementations."""
    from thalia.visualization.network_graph import _get_neuron_count

    # Region with n_neurons attribute (direct, not wrapped)
    region1 = Mock(spec=['n_neurons'])  # spec prevents .impl attribute
    expected_n_neurons = 100
    region1.n_neurons = expected_n_neurons
    assert _get_neuron_count(region1) == expected_n_neurons, \
        "Should extract neuron count from n_neurons attribute"

    # Region with config.n_neurons
    region2 = Mock(spec=['config'])
    region2.config = Mock(spec=['n_neurons'])
    expected_config_neurons = 50
    region2.config.n_neurons = expected_config_neurons
    assert _get_neuron_count(region2) == expected_config_neurons, \
        "Should extract neuron count from config.n_neurons"

    # Region with membrane shape
    region3 = Mock(spec=['membrane'])
    region3.membrane = Mock(spec=['shape'])
    expected_membrane_neurons = 75
    region3.membrane.shape = (expected_membrane_neurons,)
    assert _get_neuron_count(region3) == expected_membrane_neurons, \
        "Should extract neuron count from membrane.shape"

    # EventDrivenRegion adapter with impl
    region4 = Mock(spec=['impl'])
    region4.impl = Mock(spec=['n_neurons'])
    expected_impl_neurons = 120
    region4.impl.n_neurons = expected_impl_neurons
    assert _get_neuron_count(region4) == expected_impl_neurons, \
        "Should unwrap EventDrivenRegion adapter and extract from impl.n_neurons"

    # Region with no attributes
    region5 = Mock(spec=[])
    assert _get_neuron_count(region5) == 0, \
        "Should return 0 for region with no neuron count attributes"


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
    assert _get_pathway_strength(pathway2) == 0.0, \
        "Should return 0.0 when weights is None"

    # Pathway without weights attribute
    pathway3 = Mock()
    del pathway3.weights
    assert _get_pathway_strength(pathway3) == 0.0, \
        "Should return 0.0 when weights attribute is missing"


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

    # Test contract: all nodes should have positions
    expected_node_count = len(region_info)
    assert len(pos) == expected_node_count, \
        f"Should have positions for all {expected_node_count} nodes"
    assert all(name in pos for name in region_info.keys()), \
        "All region names should have positions"

    # Check that positions are tuples of 2 floats
    for name, (x, y) in pos.items():
        assert isinstance(x, float), f"x coordinate for {name} should be float"
        assert isinstance(y, float), f"y coordinate for {name} should be float"

    # Check layer ordering (sensory at top, motor at bottom)
    assert pos['visual'][1] > pos['cortex'][1]  # Sensory higher than cortex
    assert pos['cortex'][1] > pos['striatum'][1]  # Cortex higher than striatum
    assert pos['striatum'][1] > pos['motor'][1]  # Striatum higher than motor


@pytest.mark.skip(reason="Requires networkx and matplotlib - run manually")
def test_full_visualization(small_test_brain):
    """Integration test with real brain (requires dependencies)."""
    from thalia.visualization import visualize_brain_topology
    import matplotlib.pyplot as plt

    # Generate visualization from real brain
    G = visualize_brain_topology(small_test_brain, layout='hierarchical')

    # Test contract: graph should contain all brain regions (EventDrivenBrain has 6)
    expected_regions = {'thalamus', 'cortex', 'hippocampus', 'pfc', 'striatum', 'cerebellum'}
    assert len(G.nodes()) == len(expected_regions), \
        f"Graph should have {len(expected_regions)} nodes (brain regions)"

    # Verify core regions are present
    node_names = set(G.nodes())
    for region_name in expected_regions:
        assert region_name in node_names, \
            f"Core region '{region_name}' should be in graph"

    # Check that edges (pathways) exist
    assert len(G.edges()) > 0, \
        "Graph should have edges representing pathways"

    # Verify node attributes contain neuron counts
    for node in G.nodes():
        node_data = G.nodes[node]
        assert 'neurons' in node_data, \
            f"Node '{node}' should have neuron count attribute"
        assert node_data['neurons'] > 0, \
            f"Node '{node}' should have positive neuron count"

    plt.close('all')


def test_visualization_after_forward_pass(tmp_path, small_test_brain, device):
    """Integration test: visualize brain after processing input."""
    from thalia.visualization.network_graph import export_topology_to_graphviz

    # Run forward pass through real brain
    input_spikes = torch.zeros(10, device=device)  # 10 input neurons
    input_spikes[0] = 1.0  # Single spike on first neuron
    input_spikes[5] = 1.0  # Single spike on sixth neuron

    # Process input through brain
    small_test_brain.forward(input_spikes)

    # Export topology after processing
    output_file = tmp_path / "active_topology.dot"
    export_topology_to_graphviz(small_test_brain, str(output_file))

    # Verify export succeeded with active brain
    content = output_file.read_text()
    assert 'digraph BrainTopology' in content, \
        "Should export valid DOT graph after forward pass"

    # Check that all core regions are present (EventDrivenBrain has 6 regions)
    expected_regions = ['thalamus', 'cortex', 'hippocampus', 'pfc', 'striatum', 'cerebellum']
    for region in expected_regions:
        assert f'"{region}"' in content, \
            f"Region '{region}' should be in exported topology"

    # Verify pathway connections exist
    assert '->' in content, \
        "Pathways should be exported after forward pass"

    # Check that neuron counts are preserved after forward pass
    assert 'N' in content, \
        "Neuron count labels should be present after processing"
