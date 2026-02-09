"""
Factory functions for creating standard neuron populations.

This module provides convenience factory functions for creating common neuron types,
reducing boilerplate configuration code and centralizing biological parameter choices.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List

import torch

from .neuron import ConductanceLIF, ConductanceLIFConfig


class NeuronType(Enum):
    """Enumeration of standard neuron types for factory registration."""

    PYRAMIDAL = "pyramidal"
    RELAY = "relay"
    TRN = "trn"
    CORTICAL_LAYER = "cortical_layer"
    FAST_SPIKING = "fast_spiking"
    MSN_D1 = "msn_d1"
    MSN_D2 = "msn_d2"


class NeuronFactory:
    """
    Centralized neuron factory registry.

    Provides standardized neuron creation methods used across Thalia.
    Supports both direct function calls and dynamic registry-based creation.
    """

    # Registry of neuron factory functions
    _registry: Dict[NeuronType, Callable] = {}

    @classmethod
    def register(cls, neuron_type: NeuronType) -> Callable[[Callable], Callable]:
        """Decorator to register a neuron factory function.

        Args:
            neuron_type: Identifier for the neuron type (e.g., "pyramidal", "relay")
        """

        def decorator(func: Callable) -> Callable:
            cls._registry[neuron_type] = func
            return func

        return decorator

    @classmethod
    def create(
        cls,
        neuron_type: NeuronType,
        n_neurons: int,
        device: torch.device,
        **overrides: dict[str, Any],
    ) -> ConductanceLIF:
        """Create neurons by type name.

        Args:
            neuron_type: Type identifier for the neuron population
            n_neurons: Number of neurons to create
            device: Device for tensor allocation
            **overrides: Custom parameters to override defaults

        Returns:
            ConductanceLIF neuron population

        Raises:
            ValueError: If neuron_type is not registered
        """
        if neuron_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown neuron type: '{neuron_type}'. "
                f"Available types: {available}. "
                f"Use NeuronFactory.register() to add custom types."
            )
        result: ConductanceLIF = cls._registry[neuron_type](
            n_neurons=n_neurons, device=device, **overrides
        )
        return result

    @classmethod
    def list_neuron_types(cls) -> List[str]:
        """Get list of available neuron types.

        Returns:
            Sorted list of registered neuron type names
        """
        return sorted([key.name for key in cls._registry.keys()])

    @classmethod
    def has_type(cls, neuron_type: NeuronType) -> bool:
        """Check if a neuron type is registered.

        Args:
            neuron_type: Type identifier to check

        Returns:
            True if the type is registered, False otherwise
        """
        return neuron_type in cls._registry

    @staticmethod
    def create_pyramidal_neurons(
        n_neurons: int,
        device: torch.device,
        **overrides: dict[str, Any],
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
        """
        config = ConductanceLIFConfig(
            g_L=0.05,
            tau_E=5.0,
            tau_I=10.0,
            v_threshold=1.0,
            v_reset=0.0,
            E_L=0.0,
            E_E=3.0,
            E_I=-0.5,
            tau_mem=20.0,
            **overrides,
        )
        neurons = ConductanceLIF(n_neurons=n_neurons, config=config, device=device)
        return neurons

    @staticmethod
    def create_relay_neurons(
        n_neurons: int,
        device: torch.device,
        **overrides: dict[str, Any],
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
        """
        config = ConductanceLIFConfig(
            v_threshold=1.0,
            v_reset=0.0,
            E_L=0.0,
            E_E=3.0,
            E_I=-0.5,
            g_L=0.05,
            tau_mem=20.0,
            tau_E=5.0,  # Fast excitatory (sensory input)
            tau_I=10.0,  # Slower inhibitory (from TRN)
            tau_ref=2.0,  # Short refractory period for relay neurons (2ms vs 5ms cortical)
            **overrides,
        )
        neurons = ConductanceLIF(n_neurons=n_neurons, config=config, device=device)
        return neurons

    @staticmethod
    def create_trn_neurons(
        n_neurons: int,
        device: torch.device,
        **overrides: dict[str, Any],
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
        """
        config = ConductanceLIFConfig(
            v_threshold=1.0,
            v_reset=0.0,
            E_L=0.0,
            E_E=3.0,
            E_I=-0.5,
            g_L=0.06,  # Slightly faster dynamics
            tau_mem=16.0,  # Faster membrane
            tau_E=4.0,  # Very fast excitatory
            tau_I=8.0,  # Fast inhibitory
            **overrides,
        )
        neurons = ConductanceLIF(n_neurons=n_neurons, config=config, device=device)
        return neurons

    @staticmethod
    def create_cortical_layer_neurons(
        n_neurons: int,
        layer: str,
        device: torch.device,
        **overrides: dict[str, Any],
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

        Raises:
            ValueError: If layer is not one of "L4", "L2/3", "L5", "L6a", "L6b", "L6"
        """
        # Base config for all layers
        base_config = {
            "tau_E": 5.0,
            "tau_I": 10.0,
            "E_E": 3.0,
            "E_I": -0.5,
            "g_L": 0.05,
            "tau_mem": 20.0,
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
                    "tau_mem": 15.0,  # Slower dynamics
                    "tau_ref": 10.0,  # Long refractory for low-frequency firing
                }
            else:
                # L6b/L6: Standard configuration for higher frequency firing
                layer_config = {
                    **base_config,
                    "v_threshold": 1.0,
                    "tau_mem": 15.0,  # Slower dynamics for feedback control
                }
        else:
            raise ValueError(
                f"Unknown cortical layer '{layer}'. Must be one of: 'L4', 'L2/3', 'L5', 'L6a', 'L6b', 'L6'"
            )

        # Apply overrides
        layer_config.update(overrides)

        config = ConductanceLIFConfig(**layer_config)
        neurons = ConductanceLIF(n_neurons=n_neurons, config=config, device=device)
        return neurons

    @staticmethod
    def create_fast_spiking_neurons(
        n_neurons: int,
        device: torch.device,
        **overrides: dict[str, Any],
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
        """
        # Fast-spiking configuration (parvalbumin+ interneurons)
        # Start with preset and apply overrides
        fsi_config = {
            "tau_mem": 10.0,
            "v_rest": 0.0,
            "v_reset": 0.0,
            "v_threshold": 1.0,
            "tau_ref": 1.0,
            "g_leak": 0.10,
        }
        fsi_config.update(overrides)

        config = ConductanceLIFConfig(
            v_threshold=fsi_config["v_threshold"],
            v_reset=fsi_config["v_reset"],
            E_L=fsi_config["v_rest"],
            E_E=3.0,
            E_I=-0.5,
            tau_E=5.0,
            tau_I=10.0,
            tau_ref=fsi_config["tau_ref"],  # Fast refractory
            g_L=fsi_config["g_leak"],  # High leak
            tau_mem=fsi_config["tau_mem"],  # Fast dynamics
        )

        neurons = ConductanceLIF(n_neurons=n_neurons, config=config, device=device)
        return neurons

    @staticmethod
    def create_msn_d1_neurons(
        n_neurons: int,
        device: torch.device,
        **overrides: dict[str, Any],
    ) -> ConductanceLIF:
        """Create D1-MSN population (direct pathway / GO).

        D1-type Medium Spiny Neurons express D1 dopamine receptors and form
        the direct (GO) pathway in the basal ganglia:
        - DA burst (positive RPE) → LTP → stronger GO signal
        - DA dip (negative RPE) → LTD → weaker GO signal

        Biology:
        - Comprise ~50% of striatal neurons
        - Project to GPi/SNr (output nuclei)
        - Enable action selection via disinhibition
        - Show spike-frequency adaptation (Ca2+-dependent K+ channels)

        Key parameters:
        - Standard membrane dynamics (τ_mem = 20ms)
        - Spike-frequency adaptation (τ_adapt = 100ms, increment = 0.1)
        - Standard threshold and reversal potentials

        Reference: Gerfen & Surmeier (2011), Tepper et al. (2010)

        Args:
            n_neurons: Number of D1-MSN neurons to create
            device: Device for tensor allocation ("cpu" or "cuda")
            **overrides: Custom parameters to override defaults

        Returns:
            ConductanceLIF neuron population configured as D1-MSN
        """
        config = ConductanceLIFConfig(
            v_threshold=1.0,
            v_reset=0.0,
            E_L=0.0,
            E_E=3.0,
            E_I=-0.5,
            tau_E=5.0,
            tau_I=10.0,
            tau_ref=2.0,
            tau_mem=20.0,
            tau_adapt=100.0,
            adapt_increment=0.1,
            **overrides,
        )
        neurons = ConductanceLIF(n_neurons=n_neurons, config=config, device=device)
        return neurons

    @staticmethod
    def create_msn_d2_neurons(
        n_neurons: int,
        device: torch.device,
        **overrides: dict[str, Any],
    ) -> ConductanceLIF:
        """Create D2-MSN population (indirect pathway / NOGO).

        D2-type Medium Spiny Neurons express D2 dopamine receptors with
        INVERTED dopamine response, forming the indirect (NOGO) pathway:
        - DA burst (positive RPE) → LTD → weaker NOGO signal
        - DA dip (negative RPE) → LTP → stronger NOGO signal

        This inversion is the key biological insight:
        - When wrong action is punished → D2 strengthens → inhibits action next time
        - When correct action is rewarded → D2 weakens → allows action

        Biology:
        - Comprise ~50% of striatal neurons
        - Project to GPe (external pallidum) → indirect pathway
        - Oppose action selection via increased inhibition
        - Show spike-frequency adaptation (identical to D1)

        Key parameters:
        - Identical neuron properties to D1-MSN
        - Differentiation is in synaptic learning (three-factor rule)
        - Standard membrane dynamics (τ_mem = 20ms)
        - Spike-frequency adaptation (τ_adapt = 100ms, increment = 0.1)

        Reference: Gerfen & Surmeier (2011), Tepper et al. (2010)

        Args:
            n_neurons: Number of D2-MSN neurons to create
            device: Device for tensor allocation ("cpu" or "cuda")
            **overrides: Custom parameters to override defaults

        Returns:
            ConductanceLIF neuron population configured as D2-MSN
        """
        config = ConductanceLIFConfig(
            v_threshold=1.0,
            v_reset=0.0,
            E_L=0.0,
            E_E=3.0,
            E_I=-0.5,
            tau_E=5.0,
            tau_I=10.0,
            tau_ref=2.0,
            tau_mem=20.0,
            tau_adapt=100.0,
            adapt_increment=0.1,
            **overrides,
        )
        neurons = ConductanceLIF(n_neurons=n_neurons, config=config, device=device)
        return neurons


# Register standard neuron types
NeuronFactory.register(NeuronType.PYRAMIDAL)(NeuronFactory.create_pyramidal_neurons)
NeuronFactory.register(NeuronType.RELAY)(NeuronFactory.create_relay_neurons)
NeuronFactory.register(NeuronType.TRN)(NeuronFactory.create_trn_neurons)
NeuronFactory.register(NeuronType.CORTICAL_LAYER)(NeuronFactory.create_cortical_layer_neurons)
NeuronFactory.register(NeuronType.FAST_SPIKING)(NeuronFactory.create_fast_spiking_neurons)
NeuronFactory.register(NeuronType.MSN_D1)(NeuronFactory.create_msn_d1_neurons)
NeuronFactory.register(NeuronType.MSN_D2)(NeuronFactory.create_msn_d2_neurons)
