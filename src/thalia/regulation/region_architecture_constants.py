"""
Architectural constants for region-specific ratios and defaults.

These constants define biological ratios (layer sizes, expansion factors)
that are architecturally motivated rather than tunable hyperparameters.

The values here reflect neuroanatomical measurements and computational
principles from neuroscience research. They should only be changed when
there is strong biological evidence or architectural reasoning.

**Usage Pattern**:

.. code-block:: python

    from thalia.regulation.region_architecture_constants import (
        HIPPOCAMPUS_DG_EXPANSION_FACTOR,
    )

    # In config dataclass
    @dataclass
    class HippocampusConfig:
        dg_expansion: float = HIPPOCAMPUS_DG_EXPANSION_FACTOR

**Design Rationale**:

These are **architectural constants** (biological structure) not
**hyperparameters** (tunable learning parameters). They define the
relative sizes and ratios of brain structures based on anatomy.

Author: Thalia Project
Date: December 20, 2025
"""

# =============================================================================
# HIPPOCAMPUS ARCHITECTURE
# =============================================================================

HIPPOCAMPUS_DG_EXPANSION_FACTOR = 3.0
"""Dentate Gyrus expansion factor relative to entorhinal cortex input.

The DG has approximately 3x more neurons than EC, enabling pattern
separation through sparse coding. This ratio is critical for the
orthogonalization function of the hippocampus.

Reference: Amaral et al. (1990) - Hippocampal formation anatomy
"""

# =============================================================================
# CORTICAL LAYER ARCHITECTURE
# =============================================================================

# Dopamine Receptor Density Distribution (Layer-specific sensitivity)
# These values reflect the relative distribution of DA receptors across layers.
# Sum = 1.0 (dopamine is distributed, not multiplied across layers)

CORTEX_L4_DA_FRACTION = 0.2
"""Layer 4 dopamine sensitivity fraction.

Sensory input layer has lowest DA receptor density for input stability.
Prevents reward signals from disrupting sensory processing.

Reference: Seamans & Yang (2004) - DA modulation in PFC
Extracted from LayeredCortex implementation (Architecture Review 2025-12-21)
"""

CORTEX_L23_DA_FRACTION = 0.3
"""Layer 2/3 dopamine sensitivity fraction.

Association layer has moderate DA receptor density for flexible
integration of information with reward context.

Reference: Seamans & Yang (2004) - DA modulation in PFC
Extracted from LayeredCortex implementation (Architecture Review 2025-12-21)
"""

CORTEX_L5_DA_FRACTION = 0.4
"""Layer 5 dopamine sensitivity fraction.

Motor output layer has highest DA receptor density (40% of total).
Critical for reinforcement learning and action selection.

Reference: Seamans & Yang (2004) - DA modulation in PFC
Extracted from LayeredCortex implementation (Architecture Review 2025-12-21)
"""

CORTEX_L6_DA_FRACTION = 0.1
"""Layer 6 dopamine sensitivity fraction.

Feedback/attention layer has lowest DA receptor density for stability
in top-down control signals.

Reference: Seamans & Yang (2004) - DA modulation in PFC
Extracted from LayeredCortex implementation (Architecture Review 2025-12-21)
"""

# =============================================================================
# NEURAL GROWTH CONSTANTS
# =============================================================================

GROWTH_NEW_WEIGHT_SCALE = 0.2
"""Scaling factor for new weights during neurogenesis.

New synaptic weights are initialized at 20% of w_max to allow gradual
integration without disrupting existing network dynamics.

Biological rationale: New neurons start with weak connections that
strengthen through activity-dependent plasticity.

Reference: Common pattern across all regions (Architecture Review 2025-12-21)
"""

# =============================================================================
# ACTIVITY TRACKING CONSTANTS
# =============================================================================

ACTIVITY_HISTORY_DECAY = 0.99
"""Exponential decay factor for activity history tracking.

Used in homeostatic mechanisms to track long-term average activity.
Decay of 0.99 per timestep gives effective time constant of ~100 steps.

Reference: Intrinsic plasticity implementations (Architecture Review 2025-12-21)
"""

ACTIVITY_HISTORY_INCREMENT = 0.01
"""Increment weight for new activity in exponential moving average.

Complement of ACTIVITY_HISTORY_DECAY (1.0 - decay).
Used in: activity = activity * decay + new_value * increment

Reference: Intrinsic plasticity implementations (Architecture Review 2025-12-21)
"""

# =============================================================================
# HIPPOCAMPUS ARCHITECTURE (continued)
# =============================================================================

HIPPOCAMPUS_SPARSITY_TARGET = 0.03
"""Target sparsity for dentate gyrus (3% active neurons).

Biological DG maintains very sparse activity (2-5% active) for
effective pattern separation. This is a key architectural feature,
not a tunable hyperparameter.

Reference: Jung & McNaughton (1993) - Spatial selectivity in DG
"""

# =============================================================================
# MULTISENSORY ARCHITECTURE
# =============================================================================

MULTISENSORY_VISUAL_RATIO = 0.3
"""Visual pool fraction in multisensory integration areas."""

MULTISENSORY_AUDITORY_RATIO = 0.3
"""Auditory pool fraction in multisensory integration areas."""

MULTISENSORY_LANGUAGE_RATIO = 0.2
"""Language pool fraction in multisensory integration areas."""

MULTISENSORY_INTEGRATION_RATIO = 0.2
"""Integration pool fraction (computed as remainder)."""

# =============================================================================
# CEREBELLUM ARCHITECTURE
# =============================================================================

CEREBELLUM_GRANULE_EXPANSION = 4.0
"""Granule cell expansion factor.

Granule layer is 4× larger than mossy fiber input for sparse pattern separation.

Reference: Eccles et al. (1967) - Cerebellum as a neuronal machine
"""

# =============================================================================
# PREFRONTAL CORTEX ARCHITECTURE
# =============================================================================

PFC_WM_CAPACITY_RATIO = 0.3
"""Working memory capacity as ratio of PFC size.

Not all PFC neurons participate in working memory maintenance.
Approximately 30% form stable attractors while others handle
gating and updating.

Reference: Goldman-Rakic (1995) - Cellular basis of working memory
"""

# =============================================================================
# CEREBELLUM ARCHITECTURE
# =============================================================================

CEREBELLUM_GRANULE_EXPANSION = 100.0
"""Granule cell expansion factor (extremely large in biology).

Biological cerebellum has ~50-100 billion granule cells vs ~15 million
Purkinje cells (ratio ~3000-7000). For modeling, we use a much smaller
but still expansive ratio to maintain the sparse coding principle.

Reference: Herculano-Houzel (2010) - Coordinated scaling of cortical and cerebellar neurons
"""

CEREBELLUM_PURKINJE_PER_DCN = 10.0
"""Purkinje cells per deep cerebellar nucleus neuron.

Multiple Purkinje cells converge on each DCN neuron, providing
integration of error signals. 10:1 ratio provides adequate convergence
without excessive computational cost.

Reference: Person & Raman (2012) - Purkinje neuron synchrony
"""

# =============================================================================
# METACOGNITION THRESHOLDS
# =============================================================================

METACOG_ABSTENTION_STAGE1 = 0.5
"""Binary abstention threshold for Stage 1 (toddler).

Stage 1 metacognition uses simple binary threshold: "know it" vs "don't know".
Midpoint threshold (0.5) provides balanced sensitivity.

Reference: Curriculum design - developmental progression
"""

METACOG_ABSTENTION_STAGE2 = 0.3
"""Low confidence threshold for Stage 2 (preschool).

Stage 2 introduces confidence estimation. Lower threshold (0.3) encourages
abstention when uncertainty is moderate or high.

Reference: Curriculum design - developmental progression
"""

METACOG_ABSTENTION_STAGE3 = 0.4
"""Uncertainty threshold for Stage 3 (elementary).

Stage 3 uses uncertainty estimation. Threshold of 0.4 balances
information-seeking behavior with task completion.

Reference: Curriculum design - developmental progression
"""

METACOG_ABSTENTION_STAGE4 = 0.3
"""Calibrated threshold for Stage 4 (adolescent).

Stage 4 uses calibrated confidence with cost-benefit analysis.
Lower threshold (0.3) after calibration reflects improved self-assessment.

Reference: Curriculum design - developmental progression
"""

METACOG_CALIBRATION_LR = 0.01
"""Learning rate for metacognitive calibration network.

Slow learning rate (0.01) prevents overfitting to recent examples
and encourages stable calibration over time.

Reference: Metacognitive learning - standard practice
"""

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Hippocampus
    "HIPPOCAMPUS_DG_EXPANSION_FACTOR",
    "HIPPOCAMPUS_SPARSITY_TARGET",
    # Multisensory
    "MULTISENSORY_VISUAL_RATIO",
    "MULTISENSORY_AUDITORY_RATIO",
    "MULTISENSORY_LANGUAGE_RATIO",
    "MULTISENSORY_INTEGRATION_RATIO",
    # Cerebellum
    "CEREBELLUM_GRANULE_EXPANSION",
    # Prefrontal
    "PFC_WM_CAPACITY_RATIO",
    # Cerebellum
    "CEREBELLUM_GRANULE_EXPANSION",
    "CEREBELLUM_PURKINJE_PER_DCN",
    # Metacognition
    "METACOG_ABSTENTION_STAGE1",
    "METACOG_ABSTENTION_STAGE2",
    "METACOG_ABSTENTION_STAGE3",
    "METACOG_ABSTENTION_STAGE4",
    "METACOG_CALIBRATION_LR",
]
