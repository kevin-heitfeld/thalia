"""
Plasticity control - Freeze/unfreeze learning in regions and pathways.

Enables selective plasticity control for:
- Protecting learned features from catastrophic forgetting
- Transfer learning (freeze old, train new)
- Continual learning experiments
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thalia.core.brain import EventDrivenBrain


def freeze_region(
    brain: "EventDrivenBrain",
    region_name: str,
) -> None:
    """Disable plasticity in a brain region.

    Weights are preserved but no longer updated during learning.
    Useful for:
    - Protecting learned features
    - Transfer learning
    - Continual learning

    Args:
        brain: The brain to modify
        region_name: Name of region to freeze

    Example:
        >>> freeze_region(brain, "cortex")
        >>> # Cortical features are now fixed
        >>> train_on_new_task(brain)
        >>> # Cortex weights didn't change
    """
    from .lesion import _get_region

    region = _get_region(brain, region_name)

    # Disable plasticity
    if hasattr(region.impl, "plasticity_enabled"):
        region.impl.plasticity_enabled = False

    # Freeze parameters
    for param in region.impl.parameters():
        param.requires_grad = False

    print(f"❄️  Frozen region: {region_name} (plasticity disabled)")


def unfreeze_region(
    brain: "EventDrivenBrain",
    region_name: str,
) -> None:
    """Re-enable plasticity in a frozen region.

    Args:
        brain: The brain to modify
        region_name: Name of region to unfreeze

    Example:
        >>> unfreeze_region(brain, "cortex")
        >>> # Cortex can learn again
    """
    from .lesion import _get_region

    region = _get_region(brain, region_name)

    # Enable plasticity
    if hasattr(region.impl, "plasticity_enabled"):
        region.impl.plasticity_enabled = True

    # Unfreeze parameters
    for param in region.impl.parameters():
        param.requires_grad = True

    print(f"🔥 Unfrozen region: {region_name} (plasticity enabled)")


def freeze_pathway(
    brain: "EventDrivenBrain",
    pathway_name: str,
) -> None:
    """Disable plasticity in a pathway.

    Args:
        brain: The brain to modify
        pathway_name: Name of pathway to freeze

    Example:
        >>> freeze_pathway(brain, "cortex_to_hippocampus")
        >>> # Cortex→hippocampus connection is fixed
    """
    from .ablation import _get_pathway

    pathway = _get_pathway(brain, pathway_name)

    # Disable plasticity
    if hasattr(pathway, "plasticity_enabled"):
        pathway.plasticity_enabled = False

    # Freeze parameters
    for param in pathway.parameters():
        param.requires_grad = False

    print(f"❄️  Frozen pathway: {pathway_name}")


def unfreeze_pathway(
    brain: "EventDrivenBrain",
    pathway_name: str,
) -> None:
    """Re-enable plasticity in a frozen pathway.

    Args:
        brain: The brain to modify
        pathway_name: Name of pathway to unfreeze
    """
    from .ablation import _get_pathway

    pathway = _get_pathway(brain, pathway_name)

    # Enable plasticity
    if hasattr(pathway, "plasticity_enabled"):
        pathway.plasticity_enabled = True

    # Unfreeze parameters
    for param in pathway.parameters():
        param.requires_grad = True

    print(f"🔥 Unfrozen pathway: {pathway_name}")
