"""
Brain Surgery Tools - Modify trained brains for neuroscience experiments.

This module provides tools for performing controlled interventions on trained
brains, enabling neuroscience-style lesion studies, ablation experiments, and
incremental growth research.

Key Functions:
==============
- lesion_region(): Silence a brain region (simulate lesion)
- ablate_pathway(): Remove a pathway connection
- add_region_to_trained_brain(): Grow new region in trained brain
- partial_lesion(): Lesion subset of neurons in region
- temporary_lesion(): Context manager for reversible lesions
- freeze_region(): Disable plasticity in region
- unfreeze_region(): Re-enable plasticity

Use Cases:
==========
1. Neuroscience Experiments:
   - Lesion studies: "What happens if hippocampus is damaged?"
   - Ablation studies: "Is cortex→PFC pathway necessary?"
   - Recovery studies: "Can brain compensate after lesion?"

2. Continual Learning Research:
   - Add new regions for new tasks
   - Protect old regions from catastrophic forgetting
   - Study transfer learning between regions

3. Architecture Search:
   - Test minimal architectures (remove unnecessary components)
   - Identify critical pathways
   - Optimize for efficiency

Example Usage:
==============

    from thalia.surgery import (
        lesion_region, ablate_pathway, add_region_to_trained_brain,
        temporary_lesion, freeze_region
    )
    from thalia.core.dynamic_brain import DynamicBrain
    from thalia.io.checkpoint import BrainCheckpoint

    # Load trained brain
    brain = BrainCheckpoint.load("checkpoints/trained_brain.pkl")

    # Lesion study: Remove hippocampus
    lesion_region(brain, "hippocampus")
    performance_after_lesion = evaluate(brain)

    # Ablation study: Remove cortex→PFC pathway
    ablate_pathway(brain, "cortex_to_pfc")
    performance_without_pathway = evaluate(brain)

    # Temporary lesion (reversible)
    with temporary_lesion(brain, "prefrontal"):
        # PFC is lesioned only in this block
        performance = evaluate(brain)
    # PFC is restored

    # Freeze region (disable learning)
    freeze_region(brain, "cortex")
    # Train on new task (cortex weights won't change)

    # Add new region for new capability
    from thalia.regions.motor import MotorCortexConfig
    new_region_config = MotorCortexConfig(n_input=100, n_output=50)
    add_region_to_trained_brain(
        brain,
        region_name="motor_cortex",
        region_config=new_region_config,
        input_pathway_from="cortex",
    )

Biological Inspiration:
=======================
- Phineas Gage: Frontal lobe damage → personality changes
- H.M.: Hippocampal lesion → no new episodic memories
- Split-brain patients: Corpus callosum cut → hemisphere independence
- Stroke recovery: Brain rewires to compensate

References:
===========
- docs/design/curriculum_strategy.md - Continual learning
- docs/patterns/component-parity.md - Region/pathway consistency

Author: Thalia Project
Date: December 12, 2025 (Tier 3 Implementation)
"""

from __future__ import annotations

from .ablation import (
    ablate_pathway,
    restore_pathway,
)
from .growth import (
    add_region_to_trained_brain,
)
from .lesion import (
    lesion_region,
    partial_lesion,
    restore_region,
    temporary_lesion,
)
from .plasticity import (
    freeze_pathway,
    freeze_region,
    unfreeze_pathway,
    unfreeze_region,
)

__all__ = [
    # Lesion operations
    "lesion_region",
    "partial_lesion",
    "temporary_lesion",
    "restore_region",
    # Ablation operations
    "ablate_pathway",
    "restore_pathway",
    # Plasticity control
    "freeze_region",
    "unfreeze_region",
    "freeze_pathway",
    "unfreeze_pathway",
    # Growth operations
    "add_region_to_trained_brain",
]
