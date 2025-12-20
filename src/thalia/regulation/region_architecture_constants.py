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
        CORTEX_L23_RATIO,
    )

    # In config dataclass
    @dataclass
    class HippocampusConfig:
        dg_expansion: float = HIPPOCAMPUS_DG_EXPANSION_FACTOR
        ca3_size_ratio: float = HIPPOCAMPUS_CA3_SIZE_RATIO

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

HIPPOCAMPUS_CA3_SIZE_RATIO = 0.5
"""CA3 size as ratio of dentate gyrus size.

CA3 has approximately 50% of DG neuron count but with extensive
recurrent connections for pattern completion (autoassociative memory).

Reference: Treves & Rolls (1994) - Computational analysis
"""

HIPPOCAMPUS_CA1_SIZE_RATIO = 0.5
"""CA1 size as ratio of CA3 size.

CA1 acts as output/comparison layer, typically similar size to CA3
or slightly smaller. Default 0.5 provides adequate capacity for
match/mismatch detection.

Reference: Colgin (2013) - Theta-gamma coupling
"""

HIPPOCAMPUS_SPARSITY_TARGET = 0.03
"""Target sparsity for dentate gyrus (3% active neurons).

Biological DG maintains very sparse activity (2-5% active) for
effective pattern separation. This is a key architectural feature,
not a tunable hyperparameter.

Reference: Jung & McNaughton (1993) - Spatial selectivity in DG
"""

# =============================================================================
# CORTEX LAYER RATIOS (based on mammalian neocortex)
# =============================================================================

CORTEX_L4_RATIO = 1.0
"""Layer 4 size ratio (input layer, baseline).

L4 receives thalamic input and is the reference layer for computing
other layer sizes. Ratio of 1.0 means L4 size is directly determined
by the desired cortical column size.

Reference: Douglas & Martin (2004) - Canonical microcircuit
"""

CORTEX_L23_RATIO = 1.5
"""Layer 2/3 size ratio (processing layer, 1.5x L4).

L2/3 is larger than L4 in most cortical areas, with extensive
lateral connections. The 1.5x ratio reflects typical anatomical
measurements in visual and prefrontal cortex.

Reference: DeFelipe et al. (2002) - Microanatomy of cerebral cortex
"""

CORTEX_L5_RATIO = 1.0
"""Layer 5 size ratio (output layer, same as L4).

L5 contains large pyramidal neurons projecting to subcortical targets.
Similar size to L4 but with different projection patterns.

Reference: Douglas & Martin (2004) - Canonical microcircuit
"""

CORTEX_L6_RATIO = 0.5
"""Layer 6 size ratio (feedback layer, 0.5x L4).

L6 provides corticothalamic feedback and is typically smaller than
other layers. The 0.5x ratio is conservative, allowing sufficient
feedback capacity without overcomplicating the model.

Reference: Sherman & Guillery (2013) - Functional organization of thalamus
"""

# =============================================================================
# STRIATUM ARCHITECTURE
# =============================================================================

STRIATUM_NEURONS_PER_ACTION = 10
"""Number of neurons per action in population coding.

Population coding provides robustness through redundancy. 10 neurons
per action is a reasonable balance between computational efficiency
and noise resistance.

Reference: Computational modeling convention
"""

STRIATUM_D1_D2_RATIO = 0.5
"""D1/D2 pathway size ratio (equal populations).

Biological striatum has roughly equal numbers of D1-MSNs (direct pathway)
and D2-MSNs (indirect pathway), each ~50% of MSN population. The 0.5
ratio creates balanced opponent processing.

Reference: Gerfen & Surmeier (2011) - Modulation of striatal projection systems
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
    "HIPPOCAMPUS_CA3_SIZE_RATIO",
    "HIPPOCAMPUS_CA1_SIZE_RATIO",
    "HIPPOCAMPUS_SPARSITY_TARGET",
    # Cortex
    "CORTEX_L4_RATIO",
    "CORTEX_L23_RATIO",
    "CORTEX_L5_RATIO",
    "CORTEX_L6_RATIO",
    # Striatum
    "STRIATUM_NEURONS_PER_ACTION",
    "STRIATUM_D1_D2_RATIO",
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
