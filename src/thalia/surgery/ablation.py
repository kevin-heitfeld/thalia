"""
Ablation operations - Remove pathway connections.

Implements pathway ablation for testing necessity of connections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict
from dataclasses import dataclass

import torch

if TYPE_CHECKING:
    from thalia.core.brain import EventDrivenBrain


@dataclass
class AblationState:
    """Saved state for pathway restoration."""
    pathway_name: str
    original_weights: Dict[str, torch.Tensor]
    original_plasticity: bool


_ablation_cache: Dict[str, AblationState] = {}


def ablate_pathway(
    brain: "EventDrivenBrain",
    pathway_name: str,
    save_for_restore: bool = True,
) -> None:
    """Remove a pathway connection between regions.

    Effects:
    - Zero pathway weights
    - Disable pathway plasticity
    - Spikes no longer propagate through pathway

    Args:
        brain: The brain to modify
        pathway_name: Name of pathway to ablate
                     Common pathways:
                     - "thalamus_to_cortex"
                     - "cortex_to_hippocampus"
                     - "cortex_to_pfc"
                     - "cortex_to_striatum"
                     - "hippocampus_to_pfc"
                     - "pfc_to_striatum"
                     - "striatum_to_cerebellum"
        save_for_restore: Whether to save state for restoration

    Example:
        >>> ablate_pathway(brain, "cortex_to_pfc")
        >>> # Cortex can no longer influence working memory
    """
    # Get pathway
    pathway = _get_pathway(brain, pathway_name)

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
    brain: "EventDrivenBrain",
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

def _get_pathway(brain: "EventDrivenBrain", pathway_name: str):
    """Get pathway by name."""
    # Access pathways dict directly (brain.pathways is already the dict)
    pathways = brain.pathways

    # Try to get pathway
    if pathway_name in pathways:
        return pathways[pathway_name]

    # Try alternative names
    pathway_map = {
        "thalamus_to_cortex": "thalamus_to_cortex",
        "cortex_to_hippocampus": "cortex_to_hippo",
        "cortex_to_hippo": "cortex_to_hippo",
        "cortex_to_pfc": "cortex_to_pfc",
        "cortex_to_prefrontal": "cortex_to_pfc",
        "cortex_to_striatum": "cortex_to_striatum",
        "hippocampus_to_pfc": "hippo_to_pfc",
        "hippo_to_pfc": "hippo_to_pfc",
        "pfc_to_hippocampus": "pfc_to_hippo",
        "pfc_to_hippo": "pfc_to_hippo",
        "hippocampus_to_striatum": "hippo_to_striatum",
        "hippo_to_striatum": "hippo_to_striatum",
        "pfc_to_striatum": "pfc_to_striatum",
        "prefrontal_to_striatum": "pfc_to_striatum",
        "striatum_to_cerebellum": "striatum_to_cerebellum",
        "pfc_to_cortex": "attention",  # Top-down attention
        "hippocampus_to_cortex": "replay",  # Replay
        "hippo_to_cortex": "replay",
    }

    if pathway_name in pathway_map:
        canonical_name = pathway_map[pathway_name]
        if canonical_name in pathways:
            return pathways[canonical_name]

    # Pathway not found
    available = list(pathways.keys())
    raise ValueError(
        f"Unknown pathway: {pathway_name}. "
        f"Available pathways: {available}"
    )


def _save_ablation_state(
    brain: "EventDrivenBrain",
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
