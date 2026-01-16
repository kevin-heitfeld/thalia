"""
Standard region size ratios based on neuroscience literature.

This module defines biologically-motivated ratios for brain region sizes,
documenting the neuroscience basis for architectural choices.

DEPRECATED: Use :class:`LayerSizeCalculator` in :mod:`thalia.config.size_calculator`
"""

from __future__ import annotations

import warnings

from thalia.config.size_calculator import LayerSizeCalculator


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
    calc = LayerSizeCalculator()
    result = calc.hippocampus_from_input(ec_input_size)

    # Return only the keys that old function returned (backward compatibility)
    # NOTE: Also including input_size for new (config, sizes, device) pattern
    return {
        "input_size": result["input_size"],
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
    neurons_per_action: int = 10,
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

    # If custom trn_ratio provided, compute directly
    if trn_ratio != 0.3:
        trn_size = int(relay_size * trn_ratio)
        return {
            "relay_size": relay_size,
            "trn_size": trn_size,
        }

    # Import here to avoid circular dependency
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

    # If custom granule_expansion provided, compute directly
    if granule_expansion != 4.0:
        granule_size = int(purkinje_size * granule_expansion)
        return {
            "granule_size": granule_size,
            "purkinje_size": purkinje_size,
        }

    # Import here to avoid circular dependency
    calc = LayerSizeCalculator()
    result = calc.cerebellum_from_output(purkinje_size)

    # Return only the keys that old function returned (backward compatibility)
    return {
        "granule_size": result["granule_size"],
        "purkinje_size": result["purkinje_size"],
    }
