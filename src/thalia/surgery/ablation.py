"""
Ablation operations - Remove pathway connections.

Implements pathway ablation for testing necessity of connections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict
from dataclasses import dataclass

import torch

if TYPE_CHECKING:
    from thalia.core.dynamic_brain import DynamicBrain


@dataclass
class AblationState:
    """Saved state for pathway restoration."""
    pathway_name: str
    original_weights: Dict[str, torch.Tensor]
    original_plasticity: bool


_ablation_cache: Dict[str, AblationState] = {}


def ablate_pathway(
    brain: "DynamicBrain",
    pathway_name: str,
    save_for_restore: bool = True,
) -> None:
    """Remove a pathway connection between regions.

    **v3.0 Architecture Note**: The v3.0 architecture uses AxonalProjection
    (spike routing without learnable parameters). To ablate connections:

    **Recommended approaches**:
    1. Lesion the source region: `lesion_region(brain, "source_name")`
    2. Zero synaptic weights at target: `brain.components["target"].synaptic_weights["source"].zero_()`
    3. Remove from connections dict: `del brain.connections[("source", "target")]`

    This function remains for backward compatibility with weighted pathways but
    will raise NotImplementedError for AxonalProjection pathways.

    Args:
        brain: The brain to modify
        pathway_name: Name of pathway to ablate
        save_for_restore: Whether to save state for restoration

    Raises:
        NotImplementedError: For routing pathways (AxonalProjection)

    Example:
        >>> # For v3.0 architecture (AxonalProjection): lesion the source region
        >>> from thalia.surgery import lesion_region
        >>> lesion_region(brain, "cortex")  # Silence cortex → all its outputs stop
        >>>
        >>> # Or zero specific synaptic weights at target:
        >>> brain.components["prefrontal"].synaptic_weights["cortex:l5"].zero_()
    """
    # Get pathway
    pathway = _get_pathway(brain, pathway_name)

    # Check if pathway has learnable parameters
    has_learnable_params = any(p.requires_grad for p in pathway.parameters())

    if not has_learnable_params:
        raise NotImplementedError(
            f"Cannot ablate routing pathway '{pathway_name}' ({type(pathway).__name__}). "
            f"Routing pathways have no weights to ablate. "
            f"Use lesion_region() to silence the source region instead, "
            f"or manually remove the pathway from brain.connections."
        )

    # Save state if requested
    if save_for_restore:
        _save_ablation_state(brain, pathway_name, pathway)

    # Disable plasticity
    if hasattr(pathway, "plasticity_enabled"):
        pathway.plasticity_enabled = False

    # Zero weights
    for param in pathway.parameters():
        if param.requires_grad:
            param.data.zero_()

    print(f"✂️  Ablated pathway: {pathway_name}")


def restore_pathway(
    brain: "DynamicBrain",
    pathway_name: str,
) -> None:
    """Restore an ablated pathway to original state.

    Args:
        brain: The brain to modify
        pathway_name: Name of pathway to restore

    Raises:
        ValueError: If no ablation state was saved
    """
    if pathway_name not in _ablation_cache:
        raise ValueError(
            f"No saved state for pathway '{pathway_name}'. "
            f"Cannot restore."
        )

    state = _ablation_cache[pathway_name]
    pathway = _get_pathway(brain, pathway_name)

    # Restore weights
    for name, param in pathway.named_parameters():
        if name in state.original_weights:
            param.data.copy_(state.original_weights[name])

    # Restore plasticity
    if hasattr(pathway, "plasticity_enabled"):
        pathway.plasticity_enabled = state.original_plasticity

    # Remove from cache
    del _ablation_cache[pathway_name]

    print(f"✅ Restored pathway: {pathway_name}")


# Helper functions

def _get_pathway(brain: "DynamicBrain", pathway_name: str):
    """Get pathway by name.

    Converts string pathway names like 'cortex_to_hippocampus' to tuple keys like ('cortex', 'hippocampus').
    """
    # Access connections dict directly
    connections = brain.connections

    # Convert string name to tuple if needed
    if "_to_" in pathway_name:
        src, tgt = pathway_name.split("_to_")
        pathway_key = (src, tgt)
        if pathway_key in connections:
            return connections[pathway_key]

    # Try alternative names
    pathway_map = {
        "thalamus_to_cortex": ("thalamus", "cortex"),
        "cortex_to_hippocampus": ("cortex", "hippocampus"),
        "cortex_to_hippo": ("cortex", "hippocampus"),
        "cortex_to_pfc": ("cortex", "pfc"),
        "cortex_to_prefrontal": ("cortex", "pfc"),
        "cortex_to_striatum": ("cortex", "striatum"),
        "hippocampus_to_pfc": ("hippocampus", "pfc"),
        "hippo_to_pfc": ("hippocampus", "pfc"),
        "pfc_to_hippocampus": ("pfc", "hippocampus"),
        "pfc_to_hippo": ("pfc", "hippocampus"),
        "hippocampus_to_striatum": ("hippocampus", "striatum"),
        "hippo_to_striatum": ("hippocampus", "striatum"),
        "pfc_to_striatum": ("pfc", "striatum"),
        "prefrontal_to_striatum": ("pfc", "striatum"),
        "striatum_to_cerebellum": ("striatum", "cerebellum"),
        "pfc_to_cortex": ("pfc", "cortex"),  # Top-down attention
        "hippocampus_to_cortex": ("hippocampus", "cortex"),  # Replay
        "hippo_to_cortex": ("hippocampus", "cortex"),
    }

    if pathway_name in pathway_map:
        canonical_key = pathway_map[pathway_name]
        if canonical_key in connections:
            return connections[canonical_key]

    # Pathway not found
    available = list(connections.keys())
    raise ValueError(
        f"Unknown pathway: {pathway_name}. "
        f"Available pathways: {available}"
    )


def _save_ablation_state(
    brain: "DynamicBrain",
    pathway_name: str,
    pathway,
) -> AblationState:
    """Save pathway state before ablation."""
    # Save weights
    original_weights = {}
    for name, param in pathway.named_parameters():
        if param.requires_grad:
            original_weights[name] = param.data.clone()

    # Save plasticity state
    original_plasticity = getattr(pathway, "plasticity_enabled", True)

    # Create state object
    state = AblationState(
        pathway_name=pathway_name,
        original_weights=original_weights,
        original_plasticity=original_plasticity,
    )

    _ablation_cache[pathway_name] = state
    return state
