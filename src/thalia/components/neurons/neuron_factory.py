"""
Factory functions for creating standard neuron populations.

This module provides convenience factory functions for creating common neuron types,
reducing boilerplate configuration code and centralizing biological parameter choices.

Usage:
======
    from thalia.components.neurons import NeuronFactory, create_pyramidal_neurons

    # Direct function call (traditional approach)
    dg_neurons = create_pyramidal_neurons(n_neurons=128, device=device)

    # Using registry (dynamic approach)
    neurons = NeuronFactory.create("pyramidal", n_neurons=128, device=device)

    # List available neuron types
    available = NeuronFactory.list_types()
    print(available)  # ['pyramidal', 'relay', 'trn', 'cortical_layer']

    # Custom overrides for regional specialization
    ca3_neurons = NeuronFactory.create(
        "pyramidal",
        n_neurons=32,
        device=device,
        adapt_increment=0.1,  # CA3-specific adaptation
        tau_adapt=100.0,
    )

    # Relay neurons with specific configuration
    relay = NeuronFactory.create("relay", n_neurons=64, device=device)

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch

from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.constants.neuron import (
    G_LEAK_STANDARD,
    E_EXCITATORY,
    E_INHIBITORY,
    E_LEAK,
    FAST_SPIKING_INTERNEURON,
    TAU_EXCITATORY_CONDUCTANCE,
    TAU_INHIBITORY_CONDUCTANCE,
    TAU_MEM_STANDARD,
    TAU_MEM_FAST,
    TAU_REF_FAST,
    TAU_SYN_EXCITATORY,
    TAU_SYN_INHIBITORY,
    V_RESET_STANDARD,
    V_THRESHOLD_STANDARD,
)


class NeuronFactory:
    """
    Centralized neuron factory registry.

    Provides standardized neuron creation methods used across Thalia.
    Supports both direct function calls and dynamic registry-based creation.

    Examples:
        # Registry-based creation
        >>> pyramidal = NeuronFactory.create("pyramidal", n_neurons=100, device=device)
        >>> relay = NeuronFactory.create("relay", n_neurons=50, device=device)

        # List available types
        >>> types = NeuronFactory.list_types()
        >>> print(types)  # ['pyramidal', 'relay', 'trn', 'cortical_layer']

        # Check if type exists
        >>> if NeuronFactory.has_type("pyramidal"):
        ...     neurons = NeuronFactory.create("pyramidal", 100, device)
    """

    # Registry of neuron factory functions
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, neuron_type: str):
        """Decorator to register a neuron factory function.

        Args:
            neuron_type: Identifier for the neuron type (e.g., "pyramidal", "relay")

        Examples:
            >>> @NeuronFactory.register("custom")
            ... def create_custom_neurons(n_neurons, device, **overrides):
            ...     config = ConductanceLIFConfig(...)
            ...     return ConductanceLIF(n_neurons, config)
        """
        def decorator(func: Callable) -> Callable:
            cls._registry[neuron_type] = func
            return func
        return decorator

    @classmethod
    def create(
        cls,
        neuron_type: str,
        n_neurons: int,
        device: torch.device,
        **overrides
    ) -> ConductanceLIF:
        """Create neurons by type name.

        Args:
            neuron_type: Type identifier (e.g., "pyramidal", "relay", "trn", "cortical_layer")
            n_neurons: Number of neurons to create
            device: Device for tensor allocation
            **overrides: Custom parameters to override defaults

        Returns:
            ConductanceLIF neuron population

        Raises:
            ValueError: If neuron_type is not registered

        Examples:
            >>> neurons = NeuronFactory.create("pyramidal", 128, device)
            >>> relay = NeuronFactory.create("relay", 64, device, v_threshold=0.9)
            >>> l23 = NeuronFactory.create(
            ...     "cortical_layer", 256, device, layer="L2/3", adapt_increment=0.4
            ... )
        """
        if neuron_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown neuron type: '{neuron_type}'. "
                f"Available types: {available}. "
                f"Use NeuronFactory.register() to add custom types."
            )
        return cls._registry[neuron_type](n_neurons=n_neurons, device=device, **overrides)

    @classmethod
    def list_types(cls) -> List[str]:
        """Get list of available neuron types.

        Returns:
            Sorted list of registered neuron type names

        Examples:
            >>> types = NeuronFactory.list_types()
            >>> print(types)
            ['cortical_layer', 'pyramidal', 'relay', 'trn']
        """
        return sorted(cls._registry.keys())

    @classmethod
    def has_type(cls, neuron_type: str) -> bool:
        """Check if a neuron type is registered.

        Args:
            neuron_type: Type identifier to check

        Returns:
            True if the type is registered, False otherwise

        Examples:
            >>> if NeuronFactory.has_type("pyramidal"):
            ...     neurons = NeuronFactory.create("pyramidal", 100, device)
        """
        return neuron_type in cls._registry


@NeuronFactory.register("pyramidal")
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
    neurons = ConductanceLIF(n_neurons=n_neurons, config=config)
    neurons.to(device)
    return neurons


@NeuronFactory.register("relay")
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
        tau_E=TAU_EXCITATORY_CONDUCTANCE,  # Fast excitatory (sensory input)
        tau_I=TAU_INHIBITORY_CONDUCTANCE,  # Slower inhibitory (from TRN)
        **overrides,
    )
    neurons = ConductanceLIF(n_neurons=n_neurons, config=config)
    neurons.to(device)
    return neurons


@NeuronFactory.register("trn")
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
    neurons = ConductanceLIF(n_neurons=n_neurons, config=config)
    neurons.to(device)
    return neurons


@NeuronFactory.register("cortical_layer")
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
    - L6a: Corticothalamic type I (projects to TRN, inhibitory modulation)
    - L6b: Corticothalamic type II (projects to relay, excitatory modulation)

    Args:
        n_neurons: Number of neurons in the layer
        layer: Layer identifier ("L4", "L2/3", "L5", "L6a", "L6b", or legacy "L6")
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

        # L6a/L6b corticothalamic feedback
        >>> l6a = create_cortical_layer_neurons(32, "L6a", device)
        >>> l6b = create_cortical_layer_neurons(16, "L6b", device)

    Raises:
        ValueError: If layer is not one of "L4", "L2/3", "L5", "L6a", "L6b", "L6"
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
    elif layer in ["L6", "L6a", "L6b"]:
        # L6/L6a/L6b: Corticothalamic feedback layers
        # L6a (type I) → TRN: Inhibitory modulation, low gamma (25-35 Hz)
        # L6b (type II) → relay: Excitatory modulation, high gamma (60-80 Hz)

        if layer == "L6a":
            # L6a: Slower firing for low gamma (25-35 Hz)
            # Longer refractory period (10ms) limits maximum rate to ~100Hz
            # Combined with inhibition → sparse firing → low gamma oscillations
            layer_config = {
                **base_config,
                "v_threshold": 1.0,
                "tau_mem": 15.0,      # Slower dynamics
                "tau_ref": 10.0,      # Long refractory for low-frequency firing
            }
        else:
            # L6b/L6: Standard configuration for higher frequency firing
            layer_config = {
                **base_config,
                "v_threshold": 1.0,
                "tau_mem": 15.0,      # Slower dynamics for feedback control
            }
    else:
        raise ValueError(
            f"Unknown cortical layer '{layer}'. Must be one of: 'L4', 'L2/3', 'L5', 'L6a', 'L6b', 'L6'"
        )

    # Apply overrides
    layer_config.update(overrides)

    config = ConductanceLIFConfig(**layer_config)
    neurons = ConductanceLIF(n_neurons=n_neurons, config=config)
    # Move neurons to specified device (neurons are created on CPU by default)
    neurons.to(device)
    return neurons


@NeuronFactory.register("fast_spiking")
def create_fast_spiking_neurons(
    n_neurons: int,
    device: torch.device,
    **overrides
) -> ConductanceLIF:
    """Create fast-spiking interneuron population (parvalbumin+).

    Fast-spiking interneurons (FSI) are characterized by:
    - Fast membrane time constant (tau_mem ~5-10ms)
    - Short refractory period (tau_ref ~2ms)
    - High leak conductance (fast membrane decay)
    - Provide feedforward/feedback inhibition
    - Dense gap junction networks for synchronization
    - Critical for gamma oscillations (30-80 Hz)

    Common uses:
    - Striatal FSI (~2% of striatum, parvalbumin+)
    - Cortical basket cells and chandelier cells
    - Hippocampal basket cells
    - Cerebellar basket cells

    Biology: Koós & Tepper (1999), Gittis et al. (2010)

    Args:
        n_neurons: Number of FSI neurons to create
        device: Device for tensor allocation ("cpu" or "cuda")
        **overrides: Custom parameters to override defaults

    Returns:
        ConductanceLIF neuron population configured as FSI

    Examples:
        >>> from thalia.components.neurons import create_fast_spiking_neurons
        >>> fsi = create_fast_spiking_neurons(n_neurons=20, device="cpu")

        # With custom parameters
        >>> fsi = create_fast_spiking_neurons(
        ...     n_neurons=20,
        ...     device="cpu",
        ...     tau_mem=8.0,  # Slightly slower FSI
        ... )

        # Via registry
        >>> from thalia.components.neurons import NeuronFactory
        >>> fsi = NeuronFactory.create("fast_spiking", n_neurons=20, device="cpu")
    """
    # Fast-spiking configuration (parvalbumin+ interneurons)
    # Start with preset and apply overrides
    fsi_config = {**FAST_SPIKING_INTERNEURON}
    fsi_config.update(overrides)

    # Build conductance-based config
    config = ConductanceLIFConfig(
        v_threshold=fsi_config.get("v_threshold", V_THRESHOLD_STANDARD),
        v_reset=fsi_config.get("v_reset", V_RESET_STANDARD),
        E_L=fsi_config.get("v_rest", E_LEAK),
        E_E=E_EXCITATORY,
        E_I=E_INHIBITORY,
        tau_E=TAU_SYN_EXCITATORY,
        tau_I=TAU_SYN_INHIBITORY,
        tau_ref=fsi_config.get("tau_ref", TAU_REF_FAST),  # Fast refractory
        g_L=fsi_config.get("g_leak", G_LEAK_STANDARD),  # High leak
        tau_mem=fsi_config.get("tau_mem", TAU_MEM_FAST),  # Fast dynamics
    )

    neurons = ConductanceLIF(n_neurons=n_neurons, config=config)
    neurons.to(device)
    return neurons


__all__ = [
    "NeuronFactory",
    "create_pyramidal_neurons",
    "create_relay_neurons",
    "create_trn_neurons",
    "create_cortical_layer_neurons",
    "create_fast_spiking_neurons",
]
