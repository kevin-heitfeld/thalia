"""
Network Graph Visualization - Topology and Connectivity Analysis.

Visualizes brain region connectivity using NetworkX and Matplotlib.
Supports both static graphs and interactive exploration.

Key Features:
- Automatic layout of brain regions
- Pathway strength visualization
- Region size proportional to neuron count
- Export to Graphviz DOT format
- Connectivity matrix heatmaps

Usage:
    from thalia.visualization import visualize_brain_topology

    # Create NetworkX graph
    G = visualize_brain_topology(brain)

    # Show interactive plot
    plt.show()

    # Or export to DOT for external rendering
    export_topology_to_graphviz(brain, "brain_topology.dot")
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


def visualize_brain_topology(
    brain: Any,
    layout: str = 'hierarchical',
    figsize: Tuple[int, int] = (16, 12),
    node_size_scale: float = 1000.0,
    show_weights: bool = True,
    title: Optional[str] = None,
) -> Any:
    """Create NetworkX graph visualization of brain topology.

    Args:
        brain: DynamicBrain instance
        layout: Layout algorithm ('hierarchical', 'spring', 'circular', 'shell')
        figsize: Figure size in inches
        node_size_scale: Scale factor for node sizes (proportional to neuron count)
        show_weights: Whether to show connection weights on edges
        title: Optional plot title

    Returns:
        NetworkX DiGraph object

    Raises:
        ImportError: If networkx is not installed

    Example:
        >>> G = visualize_brain_topology(brain, layout='hierarchical')
        >>> plt.savefig('brain_topology.png', dpi=300)
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError(
            "NetworkX is required for network visualization. "
            "Install with: pip install networkx"
        )

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes (regions)
    region_info = {}
    for name, region in brain.regions.items():
        # Get neuron count
        n_neurons = _get_neuron_count(region)

        # Get region type/category for coloring
        region_type = _get_region_type(name)

        G.add_node(
            name,
            neurons=n_neurons,
            type=region_type,
            label=f"{name}\n({n_neurons}N)",
        )
        region_info[name] = {'neurons': n_neurons, 'type': region_type}

    # Add edges (pathways)
    for (source, target), pathway in brain.connections.items():

        # Get connection strength (average weight)
        strength = _get_pathway_strength(pathway)

        G.add_edge(
            source,
            target,
            weight=strength,
            label=f"{strength:.3f}" if show_weights else "",
            pathway=f"{source}->{target}",
        )

    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)

    # Choose layout
    if layout == 'hierarchical':
        pos = _hierarchical_layout(G, region_info)
    elif layout == 'spring':
        pos = nx.spring_layout(G, k=2.0, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Node colors by region type
    region_colors = {
        'sensory': '#FF6B6B',      # Red
        'cortex': '#4ECDC4',       # Teal
        'striatum': '#95E1D3',     # Mint
        'hippocampus': '#F38181',  # Pink
        'cerebellum': '#AA96DA',   # Purple
        'thalamus': '#FCBAD3',     # Light pink
        'vta': '#FFFFD2',          # Light yellow
        'prefrontal': '#A8D8EA',   # Light blue
        'motor': '#FFB6B9',        # Peach
        'other': '#E0E0E0',        # Gray
    }

    # Get node attributes
    node_colors = [region_colors.get(data['type'], '#E0E0E0') for _, data in G.nodes(data=True)]
    node_sizes = [data['neurons'] * node_size_scale / 100 for _, data in G.nodes(data=True)]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        ax=ax,
    )

    # Draw edges with varying thickness based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1.0
    edge_widths = [w / max_weight * 5.0 for w in edge_weights]

    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.6,
        edge_color='#888888',
        arrows=True,
        arrowsize=20,
        connectionstyle='arc3,rad=0.1',
        ax=ax,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        labels={node: data['label'] for node, data in G.nodes(data=True)},
        font_size=9,
        font_weight='bold',
        ax=ax,
    )

    # Draw edge labels if requested
    if show_weights:
        edge_labels = {(u, v): data['label'] for u, v, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=7,
            ax=ax,
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=color, label=rtype.capitalize())
        for rtype, color in region_colors.items()
        if rtype != 'other'
    ]
    ax.legend(
        handles=legend_patches,
        loc='upper right',
        framealpha=0.9,
        fontsize=10,
    )

    # Title and formatting
    if title is None:
        title = f"Brain Topology ({len(G.nodes())} regions, {len(G.edges())} pathways)"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()

    return G


def export_topology_to_graphviz(
    brain: Any,
    output_path: str,
    include_weights: bool = True,
) -> None:
    """Export brain topology to Graphviz DOT format.

    Useful for rendering with external tools or documentation.

    Args:
        brain: DynamicBrain instance
        output_path: Path to save .dot file
        include_weights: Whether to include edge weights

    Example:
        >>> export_topology_to_graphviz(brain, "topology.dot")
        >>> # Then render with: dot -Tpng -O topology.dot
    """
    output_path = Path(output_path)

    with open(output_path, 'w') as f:
        f.write("digraph BrainTopology {\n")
        f.write("  rankdir=LR;\n")
        f.write("  node [shape=box, style=rounded];\n\n")

        # Write nodes
        for name, region in brain.regions.items():
            n_neurons = _get_neuron_count(region)
            region_type = _get_region_type(name)
            color = _get_graphviz_color(region_type)

            f.write(f'  "{name}" [label="{name}\\n{n_neurons}N", fillcolor="{color}", style=filled];\n')

        f.write("\n")

        # Write edges - use brain.connections directly
        for (source, target), pathway in brain.connections.items():
            strength = _get_pathway_strength(pathway)

            if include_weights:
                f.write(f'  "{source}" -> "{target}" [label="{strength:.3f}"];\n')
            else:
                f.write(f'  "{source}" -> "{target}";\n')

        f.write("}\n")


def plot_connectivity_matrix(
    brain: Any,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'viridis',
) -> None:
    """Plot connectivity matrix heatmap.

    Shows connection strengths between all region pairs.

    Args:
        brain: DynamicBrain instance
        figsize: Figure size in inches
        cmap: Matplotlib colormap name
    """
    # Get region names
    region_names = list(brain.regions.keys())
    n_regions = len(region_names)

    # Build connectivity matrix
    matrix = np.zeros((n_regions, n_regions))

    for (source, target), pathway in brain.connections.items():
        try:
            source_idx = region_names.index(source)
            target_idx = region_names.index(target)
            strength = _get_pathway_strength(pathway)
            matrix[source_idx, target_idx] = strength
        except ValueError:
            # Source or target not in region list (shouldn't happen)
            continue

    # Plot
    _fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')

    # Axis labels
    ax.set_xticks(range(n_regions))
    ax.set_yticks(range(n_regions))
    ax.set_xticklabels(region_names, rotation=45, ha='right')
    ax.set_yticklabels(region_names)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Connection Strength', rotation=270, labelpad=20)

    # Title
    ax.set_title('Region Connectivity Matrix', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Target Region')
    ax.set_ylabel('Source Region')

    plt.tight_layout()


# =============================================================================
# Helper Functions
# =============================================================================

def _get_neuron_count(region: Any) -> int:
    """Get neuron count from region.

    Args:
        region: NeuralRegion instance to query

    Returns:
        Number of neurons in the region, or 0 if unavailable

    Note:
        Tries multiple attribute paths: n_neurons, config.n_neurons, membrane.shape[0]
    """
    # Try various attributes
    if hasattr(region, 'n_neurons'):
        return region.n_neurons
    elif hasattr(region, 'config') and hasattr(region.config, 'n_neurons'):
        return region.config.n_neurons
    elif hasattr(region, 'membrane') and hasattr(region.membrane, 'shape'):
        return region.membrane.shape[0]
    else:
        return 0


def _get_region_type(name: str) -> str:
    """Infer region type from name.

    Args:
        name: Region name (e.g., 'cortex', 'hippocampus')

    Returns:
        Region type category: 'prefrontal', 'sensory', 'striatum',
        'hippocampus', 'cerebellum', 'thalamus', 'vta', 'motor',
        'cortex', or 'other'

    Note:
        Uses substring matching on lowercased name
    """
    name_lower = name.lower()

    # Check specific types first (before more general ones)
    if 'prefrontal' in name_lower or 'pfc' in name_lower:
        return 'prefrontal'
    elif 'visual' in name_lower or 'auditory' in name_lower or 'sensory' in name_lower:
        return 'sensory'
    elif 'striatum' in name_lower:
        return 'striatum'
    elif 'hippocampus' in name_lower:
        return 'hippocampus'
    elif 'cerebellum' in name_lower:
        return 'cerebellum'
    elif 'thalamus' in name_lower:
        return 'thalamus'
    elif 'vta' in name_lower:
        return 'vta'
    elif 'motor' in name_lower:
        return 'motor'
    elif 'cortex' in name_lower:
        return 'cortex'
    else:
        return 'other'


def _get_pathway_strength(pathway: Any) -> float:
    """Get average connection strength from pathway.

    Args:
        pathway: AxonalProjection or pathway instance

    Returns:
        Mean absolute connection strength, or 0.0 if weights unavailable

    Note:
        Returns 0.0 for pathways without weights attribute
    """
    if hasattr(pathway, 'weights'):
        weights = pathway.weights
        if weights is not None:
            return float(weights.mean().item())
    return 0.0


def _hierarchical_layout(G: Any, region_info: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """Create hierarchical layout for brain regions.

    Arranges regions in layers:
    - Layer 0: Sensory regions
    - Layer 1: Cortex
    - Layer 2: Hippocampus, Striatum
    - Layer 3: Thalamus, Cerebellum
    - Layer 4: VTA, PFC
    - Layer 5: Motor
    """
    layers = {
        'sensory': 0,
        'cortex': 1,
        'hippocampus': 2,
        'striatum': 2,
        'thalamus': 3,
        'cerebellum': 3,
        'vta': 4,
        'prefrontal': 4,
        'motor': 5,
        'other': 3,
    }

    # Group nodes by layer
    layer_nodes = {}
    for node, info in region_info.items():
        layer = layers.get(info['type'], 3)
        if layer not in layer_nodes:
            layer_nodes[layer] = []
        layer_nodes[layer].append(node)

    # Assign positions
    pos = {}
    y_spacing = 2.0
    max_nodes_in_layer = max(len(nodes) for nodes in layer_nodes.values())

    for layer, nodes in layer_nodes.items():
        n_nodes = len(nodes)
        x_spacing = 3.0 if n_nodes > 1 else 0
        x_start = -(n_nodes - 1) * x_spacing / 2

        for i, node in enumerate(sorted(nodes)):
            x = x_start + i * x_spacing
            y = -layer * y_spacing
            pos[node] = (x, y)

    return pos


def _get_graphviz_color(region_type: str) -> str:
    """Get Graphviz color name for region type.

    Args:
        region_type: Region category (from _get_region_type)

    Returns:
        Graphviz color name for the region type

    Color mapping:
        - sensory: 'lightcoral'
        - cortex: 'lightblue'
        - striatum: 'lightgreen'
        - hippocampus: 'lightpink'
        - cerebellum: 'plum'
        - thalamus: 'wheat'
        - vta: 'lightyellow'
        - prefrontal: 'lightcyan'
        - motor: 'peachpuff'
        - other: 'lightgray' (default)
    """
    colors = {
        'sensory': 'lightcoral',
        'cortex': 'lightblue',
        'striatum': 'lightgreen',
        'hippocampus': 'lightpink',
        'cerebellum': 'plum',
        'thalamus': 'wheat',
        'vta': 'lightyellow',
        'prefrontal': 'lightcyan',
        'motor': 'peachpuff',
        'other': 'lightgray',
    }
    return colors.get(region_type, 'lightgray')
