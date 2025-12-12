"""
Lesion operations - Simulate brain damage for neuroscience experiments.

Implements various types of lesions:
- Complete lesion: Silence entire region
- Partial lesion: Damage subset of neurons
- Temporary lesion: Reversible via context manager
- Restore: Undo lesion effects
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Dict
from dataclasses import dataclass

import torch

if TYPE_CHECKING:
    from thalia.core.brain import EventDrivenBrain


@dataclass
class LesionState:
    """Saved state for lesion reversal."""
    region_name: str
    original_weights: Dict[str, torch.Tensor]
    original_plasticity: bool
    lesion_type: str  # "complete" or "partial"
    lesioned_indices: Optional[torch.Tensor] = None


_lesion_cache: Dict[str, LesionState] = {}


def lesion_region(
    brain: "EventDrivenBrain",
    region_name: str,
    save_for_restore: bool = True,
) -> None:
    """Completely silence a brain region (simulate lesion).

    Effects:
    - Zero all weights (afferent and efferent)
    - Disable plasticity
    - Zero internal state (membrane potentials, etc.)

    Args:
        brain: The brain to modify
        region_name: Name of region to lesion
                    Options: "cortex", "hippocampus", "prefrontal",
                            "striatum", "cerebellum", "thalamus"
        save_for_restore: Whether to save original state for restoration

    Example:
        >>> lesion_region(brain, "hippocampus")
        >>> # Brain now has no episodic memory formation
    """
    # Get region adapter
    region = _get_region(brain, region_name)

    # Save original state if requested
    if save_for_restore:
        _save_lesion_state(brain, region_name, "complete")

    # Disable plasticity
    if hasattr(region.impl, "plasticity_enabled"):
        region.impl.plasticity_enabled = False

    # Zero all weights
    _zero_region_weights(region.impl)

    # Reset state to zero
    region.impl.reset_state()

    print(f"✂️  Lesioned region: {region_name}")


def partial_lesion(
    brain: "EventDrivenBrain",
    region_name: str,
    lesion_fraction: float = 0.5,
    save_for_restore: bool = True,
) -> None:
    """Lesion a subset of neurons in a region.

    Randomly selects neurons to lesion based on fraction.

    Args:
        brain: The brain to modify
        region_name: Name of region to partially lesion
        lesion_fraction: Fraction of neurons to lesion (0.0-1.0)
        save_for_restore: Whether to save state for restoration

    Example:
        >>> partial_lesion(brain, "cortex", lesion_fraction=0.3)
        >>> # 30% of cortical neurons are silenced
    """
    region = _get_region(brain, region_name)

    # Get region size
    n_neurons = _get_region_size(region.impl)
    n_lesioned = int(n_neurons * lesion_fraction)

    # Randomly select neurons to lesion
    lesioned_indices = torch.randperm(n_neurons)[:n_lesioned]

    # Save state if requested
    if save_for_restore:
        state = _save_lesion_state(brain, region_name, "partial")
        state.lesioned_indices = lesioned_indices

    # Disable plasticity
    if hasattr(region.impl, "plasticity_enabled"):
        region.impl.plasticity_enabled = False

    # Zero weights for lesioned neurons
    _zero_neurons_weights(region.impl, lesioned_indices)

    print(f"✂️  Partially lesioned region: {region_name} ({lesion_fraction:.0%})")


@contextmanager
def temporary_lesion(
    brain: "EventDrivenBrain",
    region_name: str,
    lesion_fraction: float = 1.0,
):
    """Context manager for temporary (reversible) lesion.

    Args:
        brain: The brain to modify
        region_name: Name of region to temporarily lesion
        lesion_fraction: Fraction of neurons to lesion (1.0 = complete)

    Example:
        >>> with temporary_lesion(brain, "hippocampus"):
        ...     performance = evaluate_memory_task(brain)
        >>> # Hippocampus is restored after block
    """
    # Save state
    if lesion_fraction >= 1.0:
        lesion_region(brain, region_name, save_for_restore=True)
    else:
        partial_lesion(brain, region_name, lesion_fraction, save_for_restore=True)

    try:
        yield
    finally:
        # Restore state
        restore_region(brain, region_name)


def restore_region(
    brain: "EventDrivenBrain",
    region_name: str,
) -> None:
    """Restore a lesioned region to original state.

    Args:
        brain: The brain to modify
        region_name: Name of region to restore

    Raises:
        ValueError: If no lesion state was saved for this region
    """
    if region_name not in _lesion_cache:
        raise ValueError(
            f"No saved state for region '{region_name}'. "
            f"Cannot restore."
        )

    state = _lesion_cache[region_name]
    region = _get_region(brain, region_name)

    # Restore weights
    _restore_weights(region.impl, state.original_weights)

    # Restore plasticity
    if hasattr(region.impl, "plasticity_enabled"):
        region.impl.plasticity_enabled = state.original_plasticity

    # Remove from cache
    del _lesion_cache[region_name]

    print(f"✅ Restored region: {region_name}")


# Helper functions

def _get_region(brain: "EventDrivenBrain", region_name: str):
    """Get region adapter by name."""
    name_map = {
        "cortex": "cortex",
        "hippocampus": "hippocampus",
        "prefrontal": "pfc",
        "pfc": "pfc",
        "striatum": "striatum",
        "cerebellum": "cerebellum",
        "thalamus": "thalamus",
    }

    if region_name not in name_map:
        raise ValueError(
            f"Unknown region: {region_name}. "
            f"Options: {list(name_map.keys())}"
        )

    attr_name = name_map[region_name]
    return getattr(brain, attr_name)


def _get_region_size(region_impl) -> int:
    """Get number of neurons in region."""
    if hasattr(region_impl, "n_output"):
        return region_impl.n_output
    elif hasattr(region_impl, "config") and hasattr(region_impl.config, "n_output"):
        return region_impl.config.n_output
    else:
        raise ValueError("Cannot determine region size")


def _save_lesion_state(
    brain: "EventDrivenBrain",
    region_name: str,
    lesion_type: str,
) -> LesionState:
    """Save region state before lesion."""
    region = _get_region(brain, region_name)

    # Save weights
    original_weights = {}
    for name, param in region.impl.named_parameters():
        if param.requires_grad:
            original_weights[name] = param.data.clone()

    # Save plasticity state
    original_plasticity = getattr(region.impl, "plasticity_enabled", True)

    # Create state object
    state = LesionState(
        region_name=region_name,
        original_weights=original_weights,
        original_plasticity=original_plasticity,
        lesion_type=lesion_type,
    )

    _lesion_cache[region_name] = state
    return state


def _zero_region_weights(region_impl) -> None:
    """Zero all weights in a region."""
    for param in region_impl.parameters():
        if param.requires_grad:
            param.data.zero_()


def _zero_neurons_weights(region_impl, neuron_indices: torch.Tensor) -> None:
    """Zero weights for specific neurons."""
    # Zero output weights (rows in weight matrices)
    for name, param in region_impl.named_parameters():
        if param.requires_grad and "weight" in name:
            # Zero rows corresponding to lesioned neurons
            param.data[neuron_indices] = 0.0


def _restore_weights(region_impl, original_weights: Dict[str, torch.Tensor]) -> None:
    """Restore weights from saved state."""
    for name, param in region_impl.named_parameters():
        if name in original_weights:
            param.data.copy_(original_weights[name])
