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
    from thalia.brain import DynamicBrain


def freeze_region(
    brain: "DynamicBrain",
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
    """


def unfreeze_region(
    brain: "DynamicBrain",
    region_name: str,
) -> None:
    """Re-enable plasticity in a frozen region.

    Args:
        brain: The brain to modify
        region_name: Name of region to unfreeze
    """


def freeze_pathway(
    brain: "DynamicBrain",
    pathway_name: str,
) -> None:
    """Disable plasticity in a pathway.

    Args:
        brain: The brain to modify
        pathway_name: Name of pathway to freeze
    """


def unfreeze_pathway(
    brain: "DynamicBrain",
    pathway_name: str,
) -> None:
    """Re-enable plasticity in a frozen pathway.

    Args:
        brain: The brain to modify
        pathway_name: Name of pathway to unfreeze
    """
