"""
Factory functions for creating standard neuron populations.

This module provides convenience factory functions for creating common neuron types,
reducing boilerplate configuration code and centralizing biological parameter choices.

Usage:
======
    from thalia.components.neurons import create_pyramidal_neurons, create_relay_neurons

    # Simple usage with defaults
    dg_neurons = create_pyramidal_neurons(n_neurons=128, device=device)

    # Custom overrides for regional specialization
    ca3_neurons = create_pyramidal_neurons(
        n_neurons=32,
        device=device,
        adapt_increment=0.1,  # CA3-specific adaptation
        tau_adapt=100.0,
    )

    # Relay neurons with specific configuration
    relay = create_relay_neurons(n_neurons=64, device=device)

Author: Thalia Project
Date: December 12, 2025
"""

import torch

from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.components.neurons.neuron_constants import (
    G_LEAK_STANDARD,
    TAU_SYN_EXCITATORY,
    TAU_SYN_INHIBITORY,
    V_THRESHOLD_STANDARD,
    V_RESET_STANDARD,
    E_LEAK,
    E_EXCITATORY,
    E_INHIBITORY,
    TAU_MEM_STANDARD,
)


def create_pyramidal_neurons(
    n_neurons: int,
    device: torch.device,
    **overrides,
) -> ConductanceLIF:
    """Create standard pyramidal neuron population.

    Standard pyramidal neurons with typical cortical/hippocampal parameters:
    - τ_mem = 20ms (standard integration window)
    - Fast excitatory (AMPA, 5ms)
    - Slower inhibitory (GABA_A, 10ms)
    - Standard threshold and reversal potentials

    Args:
        n_neurons: Number of neurons in population
        device: Device for tensor allocation
        **overrides: Custom parameters to override defaults (e.g., adapt_increment, tau_adapt)

    Returns:
        ConductanceLIF neuron population with standard pyramidal configuration

    Examples:
        # Standard pyramidal neurons (hippocampus DG, CA1)
        >>> dg_neurons = create_pyramidal_neurons(128, device)

        # With spike-frequency adaptation (CA3, to prevent runaway recurrence)
        >>> ca3_neurons = create_pyramidal_neurons(
        ...     32, device, adapt_increment=0.1, tau_adapt=100.0
        ... )

        # Cortical L2/3 with strong adaptation
        >>> l23_neurons = create_pyramidal_neurons(
        ...     256, device, adapt_increment=0.3, tau_adapt=150.0
        ... )
    """
    config = ConductanceLIFConfig(
        g_L=G_LEAK_STANDARD,
        tau_E=TAU_SYN_EXCITATORY,
        tau_I=TAU_SYN_INHIBITORY,
        v_threshold=V_THRESHOLD_STANDARD,
        v_reset=V_RESET_STANDARD,
        E_L=E_LEAK,
        E_E=E_EXCITATORY,
        E_I=E_INHIBITORY,
        tau_mem=TAU_MEM_STANDARD,
        **overrides,  # Allow customization
    )
    return ConductanceLIF(n_neurons=n_neurons, config=config)


def create_relay_neurons(
    n_neurons: int,
    device: torch.device,
    **overrides,
) -> ConductanceLIF:
    """Create thalamic relay neuron population.

    Thalamic relay neurons with typical sensory relay properties:
    - Standard membrane dynamics (τ_mem = 20ms)
    - Fast excitatory transmission (5ms, sensory input)
    - Slower inhibitory (10ms, from TRN)
    - Standard excitability parameters

    Args:
        n_neurons: Number of relay neurons
        device: Device for tensor allocation
        **overrides: Custom parameters to override defaults

    Returns:
        ConductanceLIF neuron population configured for thalamic relay

    Examples:
        # Standard relay neurons
        >>> relay_neurons = create_relay_neurons(64, device)

        # Matrix relay with custom threshold
        >>> matrix_relay = create_relay_neurons(32, device, v_threshold=0.9)
    """
    config = ConductanceLIFConfig(
        v_threshold=V_THRESHOLD_STANDARD,
        v_reset=V_RESET_STANDARD,
        E_L=E_LEAK,
        E_E=E_EXCITATORY,
        E_I=E_INHIBITORY,
        g_L=G_LEAK_STANDARD,
        tau_mem=TAU_MEM_STANDARD,
        tau_E=5.0,  # Fast excitatory (sensory input)
        tau_I=10.0,  # Slower inhibitory (from TRN)
        **overrides,
    )
    return ConductanceLIF(n_neurons=n_neurons, config=config)


def create_trn_neurons(
    n_neurons: int,
    device: torch.device,
    **overrides,
) -> ConductanceLIF:
    """Create thalamic reticular nucleus (TRN) inhibitory neuron population.

    TRN neurons are inhibitory (GABAergic) with faster dynamics than relay neurons:
    - Faster membrane dynamics (τ_mem = 16ms, 0.8x standard)
    - Faster conductance leak (1.2x standard for quicker responses)
    - Very fast excitatory (4ms, from relay/cortex)
    - Fast inhibitory (8ms, recurrent TRN)

    Args:
        n_neurons: Number of TRN neurons
        device: Device for tensor allocation
        **overrides: Custom parameters to override defaults

    Returns:
        ConductanceLIF neuron population configured for TRN

    Examples:
        # Standard TRN neurons
        >>> trn_neurons = create_trn_neurons(32, device)
    """
    config = ConductanceLIFConfig(
        v_threshold=V_THRESHOLD_STANDARD,
        v_reset=V_RESET_STANDARD,
        E_L=E_LEAK,
        E_E=E_EXCITATORY,
        E_I=E_INHIBITORY,
        g_L=G_LEAK_STANDARD * 1.2,  # Slightly faster dynamics
        tau_mem=TAU_MEM_STANDARD * 0.8,  # Faster membrane
        tau_E=4.0,  # Very fast excitatory
        tau_I=8.0,  # Fast inhibitory
        **overrides,
    )
    return ConductanceLIF(n_neurons=n_neurons, config=config)


def create_cortical_layer_neurons(
    n_neurons: int,
    layer: str,
    device: torch.device,
    **overrides,
) -> ConductanceLIF:
    """Create layer-specific cortical neurons.

    Creates neurons configured for specific cortical layers with appropriate
    properties:
    - L4: Fast sensory integration, standard threshold
    - L2/3: Recurrent processing with strong adaptation (prevents frozen attractors)
    - L5: Output layer, slightly lower threshold for reliable output

    Args:
        n_neurons: Number of neurons in the layer
        layer: Layer identifier ("L4", "L2/3", "L5")
        device: Device for tensor allocation
        **overrides: Custom parameters to override layer defaults

    Returns:
        ConductanceLIF neuron population configured for the specified layer

    Examples:
        # L4 sensory neurons
        >>> l4 = create_cortical_layer_neurons(512, "L4", device)

        # L2/3 with custom adaptation
        >>> l23 = create_cortical_layer_neurons(
        ...     256, "L2/3", device, adapt_increment=0.4
        ... )

        # L5 output layer
        >>> l5 = create_cortical_layer_neurons(128, "L5", device)

    Raises:
        ValueError: If layer is not one of "L4", "L2/3", "L5"
    """
    # Base config for all layers
    base_config = {
        "tau_E": 5.0,
        "tau_I": 10.0,
        "E_E": 3.0,
        "E_I": -0.5,
        "g_L": G_LEAK_STANDARD,
        "tau_mem": TAU_MEM_STANDARD,
    }

    # Layer-specific customization
    if layer == "L4":
        # L4: Fast integration, sensitive to sensory input
        layer_config = {
            **base_config,
            "v_threshold": 1.0,
        }
    elif layer == "L2/3":
        # L2/3: Recurrent processing with adaptation
        layer_config = {
            **base_config,
            "v_threshold": 1.0,
            "adapt_increment": 0.3,  # Strong SFA to prevent frozen attractors
            "tau_adapt": 150.0,
        }
    elif layer == "L5":
        # L5: Output layer, slightly lower threshold
        layer_config = {
            **base_config,
            "v_threshold": 0.9,
        }
    else:
        raise ValueError(
            f"Unknown cortical layer '{layer}'. Must be one of: 'L4', 'L2/3', 'L5'"
        )

    # Apply overrides
    layer_config.update(overrides)

    config = ConductanceLIFConfig(**layer_config)
    return ConductanceLIF(n_neurons=n_neurons, config=config)
