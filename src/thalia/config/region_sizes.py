"""
Standard region size ratios based on neuroscience literature.

This module defines biologically-motivated ratios for brain region sizes,
documenting the neuroscience basis for architectural choices.

Biological Basis:
=================

Hippocampus Ratios:
------------------
Based on rodent hippocampal anatomy (Amaral & Witter, 1989):

- Dentate Gyrus (DG): ~1 million granule cells (rat)
- Entorhinal Cortex (EC): ~200k-300k layer II/III neurons
- DG/EC ratio: ~4:1 (pattern separation via expansion)

- CA3: ~300k pyramidal neurons (rat)
- CA3/DG ratio: ~0.3-0.5 (pattern completion via recurrence)

- CA1: ~300k-400k pyramidal neurons (rat)
- CA1/CA3 ratio: ~1:1 (output comparison layer)

Cortex Layer Ratios:
-------------------
Based on primate neocortex (Douglas & Martin, 2004):

- Layer 4 (input): Dense granular layer, ~30% of cortical neurons
- Layer 2/3 (integration): Largest layers, ~40% of neurons
- Layer 5 (output): Pyramidal output, ~15% of neurons
- Layer 6 (feedback): ~15% of neurons

Default Sizes:
-------------
These are reasonable defaults for prototyping. For production models,
scale based on task complexity and available compute:

- Simple tasks (MNIST): 128-256 neurons per region
- Moderate tasks (language): 512-1024 neurons per region
- Complex tasks (multimodal): 2048+ neurons per region

References:
-----------
- Amaral & Witter (1989): Hippocampal anatomy
- Douglas & Martin (2004): Canonical cortical microcircuit
- Treves & Rolls (1994): Computational analysis of hippocampus

Usage:
======
    from thalia.config.region_sizes import (
        DG_TO_EC_EXPANSION, DEFAULT_CORTEX_SIZE
    )

    dg_size = int(ec_size * DG_TO_EC_EXPANSION)
    cortex_config = LayeredCortexConfig(
        n_output=DEFAULT_CORTEX_SIZE,
        l4_size=int(DEFAULT_CORTEX_SIZE * 0.4),
        l23_size=int(DEFAULT_CORTEX_SIZE * 0.6),
        l5_size=int(DEFAULT_CORTEX_SIZE * 0.4),
        l6_size=int(DEFAULT_CORTEX_SIZE * 0.2),
    )

Author: Thalia Project
Date: December 11, 2025
"""

# =============================================================================
# HIPPOCAMPUS SIZE RATIOS
# =============================================================================

DG_TO_EC_EXPANSION = 4.0
"""Dentate Gyrus to Entorhinal Cortex expansion ratio.

DG has ~4x more neurons than EC input layer.
This expansion enables pattern separation: similar inputs map to
distinct DG representations, reducing interference in CA3.

Biological range: 3-5x depending on species and counting method.
"""

CA3_TO_DG_RATIO = 0.5
"""CA3 to Dentate Gyrus size ratio.

CA3 has ~50% as many neurons as DG.
Smaller size enables dense recurrent connectivity (all-to-all feasible).
This supports pattern completion: partial cues → full memory.

Biological range: 0.3-0.5 (rat: ~300k CA3 vs ~1M DG).
"""

CA2_TO_DG_RATIO = 0.25
"""CA2 to Dentate Gyrus size ratio.

CA2 has ~25% as many neurons as DG (~50% of CA3 size).
Small but crucial region for social memory and temporal context.
Weak CA3→CA2 plasticity creates stability hub.

Biological basis: CA2 is the smallest hippocampal subfield.
References: Hitti & Siegelbaum (2014), Chevaleyre & Siegelbaum (2010)
"""

CA1_TO_CA3_RATIO = 1.0
"""CA1 to CA3 size ratio.

CA1 roughly matches CA3 in size.
CA1 receives both direct EC input and CA3 output, comparing them
for match/mismatch detection (novelty, retrieval confidence).

Biological range: 0.8-1.2 depending on region measured.
"""

CA1_TO_DG_RATIO = CA3_TO_DG_RATIO * CA1_TO_CA3_RATIO  # ~0.5
"""CA1 to Dentate Gyrus size ratio (computed).

Derived from CA3/DG and CA1/CA3 ratios for consistency.
"""

# =============================================================================
# CORTEX LAYER RATIOS
# =============================================================================

L4_TO_INPUT_RATIO = 1.5
"""Layer 4 to input size ratio.

L4 (granular input layer) expands thalamic input by ~1.5x.
Increases representational capacity while maintaining specificity.

Biological basis: L4 is dense but not the largest layer.
"""

L23_TO_L4_RATIO = 2.0
"""Layer 2/3 to Layer 4 size ratio.

L2/3 are the largest cortical layers, ~2x L4 size.
Enable extensive lateral integration and recurrent processing.

Biological basis: L2/3 comprise ~40% of cortical neurons.
"""

L5_TO_L23_RATIO = 0.5
"""Layer 5 to Layer 2/3 size ratio.

L5 (output layer) is ~50% of L2/3 size.
Projects to subcortical structures (striatum, thalamus, brainstem).

Biological basis: L5 ~15% of cortical neurons vs L2/3 ~40%.
"""

L6_TO_L5_RATIO = 1.0
"""Layer 6 to Layer 5 size ratio (if modeling L6).

L6 roughly matches L5 in size.
Provides feedback to thalamus and local modulation.

Note: Currently not modeled explicitly in Thalia (future work).
"""

# =============================================================================
# STRIATUM RATIOS
# =============================================================================

MSN_TO_CORTEX_RATIO = 0.5
"""Medium Spiny Neurons to cortical input ratio.

Striatum has roughly 50% as many MSNs as cortical input neurons.
Provides dimensionality reduction for action selection.

Biological basis: Striatum receives convergent input from wide
cortical areas but performs winner-take-all selection.
"""

NEURONS_PER_ACTION_DEFAULT = 10
"""Default neurons per action in population coding.

Using ~10 neurons per action provides:
- Noise reduction (averaging across population)
- Graded confidence (population spike count)
- Redundancy (robustness to individual neuron failure)

Biological basis: Striatal neurons form functional ensembles
representing specific actions/contexts.
"""

# =============================================================================
# DEFAULT REGION SIZES (for prototyping)
# =============================================================================

DEFAULT_INPUT_SIZE = 128
"""Default size for sensory input layer.

Reasonable for prototyping with MNIST (28x28 = 784 → sparse encoder).
Scale up for more complex sensory processing.
"""

DEFAULT_CORTEX_SIZE = 256
"""Default size for cortical regions.

Sufficient for moderate complexity pattern recognition.
Adjust based on task:
- Simple (MNIST): 128-256
- Moderate (language): 512-1024
- Complex (multimodal): 1024-2048
"""

DEFAULT_HIPPOCAMPUS_SIZE = 128
"""Default size for hippocampus output (CA1).

Episodic memory capacity scales with neuron count.
Adjust based on memory requirements:
- Simple: 64-128 (few distinct patterns)
- Moderate: 256-512 (hundreds of patterns)
- Complex: 1024+ (thousands of patterns)
"""

DEFAULT_PFC_SIZE = 128
"""Default size for prefrontal cortex.

Working memory capacity limited by attractor stability.
Biological PFC is large (~30% of cortex) but can prototype with less.
"""

DEFAULT_N_ACTIONS = 4
"""Default number of discrete actions.

Typical for simple tasks:
- 4-way navigation (up, down, left, right)
- 4-class classification
Scale up for more complex action spaces.
"""

DEFAULT_CEREBELLUM_SIZE = 256
"""Default size for cerebellum.

Cerebellum has vastly more neurons than cortex (~80% of brain's neurons).
For modeling, we use modest size focused on error correction.
"""

# =============================================================================
# CONVENIENCE FUNCTIONS (DEPRECATED - Use LayerSizeCalculator instead)
# =============================================================================

import warnings


def compute_hippocampus_sizes(ec_input_size: int) -> dict:
    """Compute hippocampus layer sizes from EC input size.

    .. deprecated:: 0.3.0
        Use :class:`LayerSizeCalculator.hippocampus_from_input()` instead.
        This function will be removed in v0.4.0.

    Args:
        ec_input_size: Size of entorhinal cortex input

    Returns:
        Dict with dg_size, ca3_size, ca2_size, ca1_size
    """
    warnings.warn(
        "compute_hippocampus_sizes() is deprecated. "
        "Use LayerSizeCalculator().hippocampus_from_input() instead. "
        "This function will be removed in v0.4.0.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Import here to avoid circular dependency
    from thalia.config.size_calculator import LayerSizeCalculator
    calc = LayerSizeCalculator()
    result = calc.hippocampus_from_input(ec_input_size)

    # Return only the keys that old function returned (backward compatibility)
    return {
        "dg_size": result["dg_size"],
        "ca3_size": result["ca3_size"],
        "ca2_size": result["ca2_size"],
        "ca1_size": result["ca1_size"],
    }


def compute_cortex_layer_sizes(input_size: int) -> dict:
    """Compute cortex layer sizes from input size.

    .. deprecated:: 0.3.0
        Use :class:`LayerSizeCalculator.cortex_from_input()` instead.
        This function will be removed in v0.4.0.

    Args:
        input_size: Size of thalamic/sensory input

    Returns:
        Dict with l4_size, l23_size, l5_size

    Warning:
        This function does NOT compute L6a/L6b sizes. Use LayerSizeCalculator
        for complete layer size computation.
    """
    warnings.warn(
        "compute_cortex_layer_sizes() is deprecated. "
        "Use LayerSizeCalculator().cortex_from_input() instead. "
        "This function will be removed in v0.4.0.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Import here to avoid circular dependency
    from thalia.config.size_calculator import LayerSizeCalculator
    calc = LayerSizeCalculator()
    result = calc.cortex_from_input(input_size)

    # Return only the keys that old function returned (backward compatibility)
    return {
        "l4_size": result["l4_size"],
        "l23_size": result["l23_size"],
        "l5_size": result["l5_size"],
        "total_size": result["total_neurons"] - result["l6a_size"] - result["l6b_size"],
    }


def compute_striatum_sizes(
    n_actions: int,
    neurons_per_action: int = NEURONS_PER_ACTION_DEFAULT,
    d1_d2_ratio: float = 0.5,
) -> dict:
    """Compute explicit striatum pathway sizes.

    .. deprecated:: 0.3.0
        Use :class:`LayerSizeCalculator.striatum_from_actions()` instead.
        This function will be removed in v0.4.0.

    Args:
        n_actions: Number of discrete actions
        neurons_per_action: Neurons per action (population coding)
        d1_d2_ratio: Ratio of D1 to D2 pathway sizes (default 0.5 = equal)

    Returns:
        Dict with d1_size, d2_size, total_size, n_actions, neurons_per_action

    Note:
        With neurons_per_action=1, each action still gets 1 neuron in D1 and 1 in D2
        (total 2 neurons per action). The d1_d2_ratio is applied when neurons_per_action >= 2.
    """
    warnings.warn(
        "compute_striatum_sizes() is deprecated. "
        "Use LayerSizeCalculator().striatum_from_actions() instead. "
        "This function will be removed in v0.4.0.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Import here to avoid circular dependency
    from thalia.config.size_calculator import LayerSizeCalculator
    calc = LayerSizeCalculator()
    result = calc.striatum_from_actions(n_actions, neurons_per_action)

    # Return only the keys that old function returned (backward compatibility)
    return {
        "d1_size": result["d1_size"],
        "d2_size": result["d2_size"],
        "total_size": result["total_neurons"],
        "n_actions": result["n_actions"],
        "neurons_per_action": result["neurons_per_action"],
    }


def compute_thalamus_sizes(relay_size: int, trn_ratio: float = 0.3) -> dict:
    """Compute thalamus layer sizes.

    .. deprecated:: 0.3.0
        Use :class:`LayerSizeCalculator.thalamus_from_relay()` instead.
        This function will be removed in v0.4.0.

    Args:
        relay_size: Size of relay neuron population (output size)
        trn_ratio: TRN size as fraction of relay size (default 0.3)

    Returns:
        Dict with relay_size, trn_size
    """
    warnings.warn(
        "compute_thalamus_sizes() is deprecated. "
        "Use LayerSizeCalculator().thalamus_from_relay() instead. "
        "This function will be removed in v0.4.0.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Import here to avoid circular dependency
    from thalia.config.size_calculator import LayerSizeCalculator
    calc = LayerSizeCalculator()
    result = calc.thalamus_from_relay(relay_size)

    # Return only the keys that old function returned (backward compatibility)
    return {
        "relay_size": result["relay_size"],
        "trn_size": result["trn_size"],
    }


def compute_multisensory_sizes(
    total_size: int,
    visual_ratio: float = 0.3,
    auditory_ratio: float = 0.3,
    language_ratio: float = 0.2,
) -> dict:
    """Compute multisensory pool sizes.

    Args:
        total_size: Total number of neurons
        visual_ratio: Fraction for visual pool (default 0.3)
        auditory_ratio: Fraction for auditory pool (default 0.3)
        language_ratio: Fraction for language pool (default 0.2)

    Returns:
        Dict with visual_size, auditory_size, language_size, integration_size
    """
    visual_size = int(total_size * visual_ratio)
    auditory_size = int(total_size * auditory_ratio)
    language_size = int(total_size * language_ratio)
    integration_size = total_size - visual_size - auditory_size - language_size

    return {
        "visual_size": visual_size,
        "auditory_size": auditory_size,
        "language_size": language_size,
        "integration_size": integration_size,
        "total_size": total_size,
    }


def compute_cerebellum_sizes(
    purkinje_size: int,
    granule_expansion: float = 4.0,
) -> dict:
    """Compute cerebellum layer sizes.

    .. deprecated:: 0.3.0
        Use :class:`LayerSizeCalculator.cerebellum_from_output()` instead.
        This function will be removed in v0.4.0.

    Args:
        purkinje_size: Number of Purkinje cells (= output size)
        granule_expansion: Granule cell expansion factor (default 4.0)

    Returns:
        Dict with granule_size, purkinje_size
    """
    warnings.warn(
        "compute_cerebellum_sizes() is deprecated. "
        "Use LayerSizeCalculator().cerebellum_from_output() instead. "
        "This function will be removed in v0.4.0.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Import here to avoid circular dependency
    from thalia.config.size_calculator import LayerSizeCalculator
    calc = LayerSizeCalculator()
    result = calc.cerebellum_from_output(purkinje_size)

    # Return only the keys that old function returned (backward compatibility)
    return {
        "granule_size": result["granule_size"],
        "purkinje_size": result["purkinje_size"],
    }
