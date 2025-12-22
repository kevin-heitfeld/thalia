"""
Growth operations - Add new regions to trained brains.

Enables continual learning by growing new regions for new capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from thalia.pathways.axonal_projection import AxonalProjection, SourceSpec

if TYPE_CHECKING:
    from thalia.core.dynamic_brain import DynamicBrain
    from thalia.core.protocols.component import LearnableComponent


def add_region_to_trained_brain(
    brain: "DynamicBrain",
    region_name: str,
    region: "LearnableComponent",
    input_pathway_from: Optional[str] = None,
    output_pathway_to: Optional[str] = None,
) -> None:
    """Add a new region to a trained brain.

    This enables continual learning by growing new capabilities
    without disrupting existing learned features.

    Process:
    1. Add new region to brain
    2. Create input pathway (if specified)
    3. Create output pathway (if specified)
    4. Initialize with small random weights
    5. Old regions continue functioning normally

    Args:
        brain: The brain to modify
        region_name: Name for new region
        region: Initialized region component
        input_pathway_from: Source region for input pathway (optional)
        output_pathway_to: Target region for output pathway (optional)

    Example:
        >>> from thalia.regions.motor import MotorCortex, MotorCortexConfig
        >>>
        >>> # Create new motor cortex region
        >>> motor_config = MotorCortexConfig(
        ...     n_input=100,
        ...     n_output=50,
        ...     device=brain.device,
        ... )
        >>> motor_cortex = MotorCortex(motor_config)
        >>>
        >>> # Add to trained brain
        >>> add_region_to_trained_brain(
        ...     brain,
        ...     region_name="motor_cortex",
        ...     region=motor_cortex,
        ...     input_pathway_from="cortex",
        ...     output_pathway_to="cerebellum",
        ... )
        >>>
        >>> # Now brain has motor cortex capability
        >>> # Cortex → Motor Cortex → Cerebellum
    """
    # Register new region
    setattr(brain, region_name, region)
    region.to(brain.device)
    region.reset_state()

    print(f"➕ Added new region: {region_name}")

    # Create input pathway if specified
    if input_pathway_from is not None:
        _create_input_pathway(
            brain,
            source_region_name=input_pathway_from,
            target_region=region,
            target_region_name=region_name,
        )

    # Create output pathway if specified
    if output_pathway_to is not None:
        _create_output_pathway(
            brain,
            source_region=region,
            source_region_name=region_name,
            target_region_name=output_pathway_to,
        )


def _create_input_pathway(
    brain: "DynamicBrain",
    source_region_name: str,
    target_region: "LearnableComponent",
    target_region_name: str,
) -> None:
    """Create pathway from existing region to new region."""
    from .lesion import _get_region

    # Get source region
    source_region = _get_region(brain, source_region_name)
    source_size = source_region.config.n_output

    # Create AxonalProjection (pure routing, no weights)
    # Weights will be in target region's synaptic_weights dict
    sources = [SourceSpec(region_name=source_region_name, port=None, size=source_size)]

    pathway = AxonalProjection(
        sources=sources,
        axonal_delay_ms=3.0,  # Default 3ms delay
        device=brain.device,
    )
    pathway.to(brain.device)

    # Add connection to brain
    brain.add_connection(source_region_name, target_region_name, pathway)

    print(f"  ➕ Created input pathway: {source_region_name} → {target_region_name}")


def _create_output_pathway(
    brain: "DynamicBrain",
    source_region: "LearnableComponent",
    source_region_name: str,
    target_region_name: str,
) -> None:
    """Create pathway from new region to existing region."""
    from .lesion import _get_region

    # Get source region size
    source_size = source_region.config.n_output

    # Create AxonalProjection (pure routing, no weights)
    # Weights will be in target region's synaptic_weights dict
    sources = [SourceSpec(region_name=source_region_name, port=None, size=source_size)]

    pathway = AxonalProjection(
        sources=sources,
        axonal_delay_ms=3.0,  # Default 3ms delay
        device=brain.device,
    )
    pathway.to(brain.device)

    # Add connection to brain
    brain.add_connection(source_region_name, target_region_name, pathway)

    print(f"  ➕ Created output pathway: {source_region_name} → {target_region_name}")
