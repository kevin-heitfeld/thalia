"""
Factory functions for creating event-driven brain systems.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from typing import Dict

from .base import EventDrivenRegionBase, EventRegionConfig
from .cortex import EventDrivenCortex
from .hippocampus import EventDrivenHippocampus
from .pfc import EventDrivenPFC
from .striatum import EventDrivenStriatum


def create_event_driven_brain(
    n_input: int,
    n_output: int,
    hidden_size: int = 256,
    device: str = "cpu",
) -> Dict[str, EventDrivenRegionBase]:
    """Create a complete event-driven brain with all regions.

    This is a convenience function that creates and wires together
    all the brain regions for event-driven simulation.

    Args:
        n_input: Size of sensory input
        n_output: Number of output actions
        hidden_size: Size of hidden layers
        device: Torch device

    Returns:
        Dict mapping region names to EventDrivenRegion instances
    """
    from thalia.regions.cortex import LayeredCortex, LayeredCortexConfig
    from thalia.regions.hippocampus import Hippocampus, HippocampusConfig
    from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
    from thalia.regions.striatum import Striatum, StriatumConfig

    regions: Dict[str, EventDrivenRegionBase] = {}

    # Create base region instances
    cortex_config = LayeredCortexConfig(
        n_input=n_input,
        n_output=hidden_size,
        device=device,
    )
    cortex = LayeredCortex(cortex_config)

    hippo_config = HippocampusConfig(
        n_input=hidden_size,  # Receives from cortex
        n_output=hidden_size,
        device=device,
    )
    hippocampus = Hippocampus(hippo_config)

    pfc_config = PrefrontalConfig(
        n_input=hidden_size,
        n_output=hidden_size,
        device=device,
    )
    pfc = Prefrontal(pfc_config)

    striatum_config = StriatumConfig(
        n_input=hidden_size,
        n_output=n_output,
        device=device,
    )
    striatum = Striatum(striatum_config)

    # Wrap in event-driven adapters
    regions["cortex"] = EventDrivenCortex(
        EventRegionConfig(
            name="cortex",
            output_targets=["hippocampus", "pfc", "striatum"],
            device=device,
        ),
        cortex,
    )

    regions["hippocampus"] = EventDrivenHippocampus(
        EventRegionConfig(
            name="hippocampus",
            output_targets=["pfc", "cortex"],
            device=device,
        ),
        hippocampus,
    )

    regions["pfc"] = EventDrivenPFC(
        EventRegionConfig(
            name="pfc",
            output_targets=["cortex", "striatum", "hippocampus"],
            device=device,
        ),
        pfc,
    )

    regions["striatum"] = EventDrivenStriatum(
        EventRegionConfig(
            name="striatum",
            output_targets=["motor"],
            device=device,
        ),
        striatum,
    )

    return regions
