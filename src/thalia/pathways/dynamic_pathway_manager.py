"""
Dynamic Pathway Manager for DynamicBrain.

Provides pathway management API for DynamicBrain's flexible component graph:
- Pathway diagnostics collection
- Coordinated pathway growth
- State save/load

Works with arbitrary connection graphs, supporting both single-source
and multi-source pathway architectures.

Author: Thalia Project
Date: December 15, 2025
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any

import torch

from thalia.core.protocols.component import LearnableComponent


class DynamicPathwayManager:
    """Manages pathways in DynamicBrain's flexible component graph.

    Provides pathway management API for:
    - Pathway diagnostics collection
    - Coordinated pathway growth
    - State save/load

    Works with arbitrary connection graphs.
    """

    def __init__(
        self,
        connections: Dict[Tuple[str, str], LearnableComponent],
        topology: Dict[str, List[str]],
        device: torch.device,
        dt_ms: float,
    ):
        """Initialize dynamic pathway manager.

        Args:
            connections: Dict mapping (source, target) to pathway instances
            topology: Adjacency list of component graph
            device: Torch device
            dt_ms: Simulation timestep in milliseconds
        """
        self.connections = connections
        self.topology = topology
        self.device = device
        self.dt_ms = dt_ms

        # Build reverse lookup: component -> connected pathways
        self._component_pathways: Dict[str, List[Tuple[str, str]]] = {}
        for source, targets in topology.items():
            if source not in self._component_pathways:
                self._component_pathways[source] = []
            for target in targets:
                self._component_pathways[source].append((source, target))
                if target not in self._component_pathways:
                    self._component_pathways[target] = []
                self._component_pathways[target].append((source, target))

    def get_all_pathways(self) -> Dict[str, LearnableComponent]:
        """Get all pathways as dict for compatibility.

        Returns:
            Dict mapping pathway names to pathway instances

        Example:
            pathways = manager.get_all_pathways()
            for name, pathway in pathways.items():
                print(f"{name}: {pathway}")
        """
        return {
            f"{src}_to_{tgt}": pathway
            for (src, tgt), pathway in self.connections.items()
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Collect diagnostics from all pathways.

        Returns:
            Dict mapping pathway names to diagnostic metrics

        Example:
            diag = manager.get_diagnostics()
            for pathway_name, metrics in diag.items():
                if 'weight_mean' in metrics:
                    print(f"{pathway_name} weights: {metrics['weight_mean']:.3f}")
        """
        diagnostics = {}

        for (src, tgt), pathway in self.connections.items():
            pathway_name = f"{src}_to_{tgt}"

            # Collect pathway diagnostics if available
            if hasattr(pathway, 'get_diagnostics'):
                try:
                    pathway_diag = pathway.get_diagnostics()
                    diagnostics[pathway_name] = pathway_diag
                except Exception as e:
                    diagnostics[pathway_name] = {'error': str(e)}
            else:
                # Fallback: collect basic weight statistics
                diag = {}
                if hasattr(pathway, 'weights'):
                    weights = pathway.weights.detach()
                    diag['weight_mean'] = float(weights.mean())
                    diag['weight_std'] = float(weights.std())
                    diag['weight_min'] = float(weights.min())
                    diag['weight_max'] = float(weights.max())

                diagnostics[pathway_name] = diag

        return diagnostics

    def grow_connected_pathways(
        self,
        component_name: str,
        growth_amount: int,
        grow_inputs: bool = True,
        grow_outputs: bool = True,
    ) -> None:
        """Grow pathways connected to a component.

        When a component grows, its connected pathways must grow too
        to maintain dimensional compatibility.

        With multi-source pathways, growing the source updates the pathway's
        input size tracking for that specific source, automatically handling
        dimension changes without adapter complexity.

        Args:
            component_name: Name of component that grew
            growth_amount: Number of neurons added
            grow_inputs: Grow pathways that target this component
            grow_outputs: Grow pathways that source from this component

        Example:
            # Component 'cortex' added 32 neurons
            manager.grow_connected_pathways('cortex', 32)
        """
        if component_name not in self._component_pathways:
            return  # No connected pathways

        for src, tgt in self._component_pathways[component_name]:
            pathway = self.connections.get((src, tgt))
            if pathway is None:
                continue

            # Grow pathway input dimension if this component is the source
            if grow_outputs and src == component_name:
                # Try to grow source (works for both AxonalProjection and legacy pathways)
                if hasattr(pathway, 'grow_source'):
                    try:
                        # For multi-source pathways, specify which source is growing
                        old_size = getattr(pathway, 'input_sizes', {}).get(src, pathway.config.n_input)
                        new_size = old_size + growth_amount
                        pathway.grow_source(src, new_size)
                    except Exception:
                        pass  # Pathway may not support growth
                elif hasattr(pathway, 'grow_input'):
                    # Fallback for single-source pathways without grow_source
                    try:
                        pathway.grow_input(growth_amount)
                    except Exception:
                        pass  # Pathway may not support growth

            # Grow pathway output dimension if this component is the target
            if grow_inputs and tgt == component_name:
                if hasattr(pathway, 'grow_output'):
                    try:
                        pathway.grow_output(growth_amount)
                    except Exception:
                        pass  # Pathway may not support growth

    def get_state(self) -> Dict[str, Any]:
        """Get state dict for checkpointing.

        Returns:
            State dict with pathway states
        """
        state = {}

        for (src, tgt), pathway in self.connections.items():
            pathway_name = f"{src}_to_{tgt}"

            # Get pathway state if available
            if hasattr(pathway, 'state_dict'):
                state[pathway_name] = pathway.state_dict()
            elif hasattr(pathway, 'get_state'):
                state[pathway_name] = pathway.get_state()
            else:
                # Fallback: save weights if available
                if hasattr(pathway, 'weights'):
                    state[pathway_name] = {
                        'weights': pathway.weights.detach().cpu()
                    }

        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state dict from checkpoint.

        Args:
            state: State dict with pathway states
        """
        for (src, tgt), pathway in self.connections.items():
            pathway_name = f"{src}_to_{tgt}"

            if pathway_name not in state:
                continue

            pathway_state = state[pathway_name]

            # Load pathway state
            if hasattr(pathway, 'load_state_dict'):
                pathway.load_state_dict(pathway_state)
            elif hasattr(pathway, 'load_state'):
                pathway.load_state(pathway_state)
            else:
                # Fallback: load weights if available
                if 'weights' in pathway_state and hasattr(pathway, 'weights'):
                    pathway.weights.data = pathway_state['weights'].to(self.device)

    @property
    def region_connections(self) -> Dict[str, List[str]]:
        """Get region connectivity graph.

        Returns:
            Adjacency list of region connections
        """
        return self.topology
