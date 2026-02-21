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

    @classmethod
    def create(
        cls,
        region_name: str,
        population_name: str,
        neuron_type: NeuronType,
        n_neurons: int,
        device: torch.device,
        **overrides: dict[str, Any],
    ) -> ConductanceLIF:
        """Create neurons by type name.

        Args:
            region_name: Brain region identifier (e.g., "cortex", "thalamus") - REQUIRED for RNG independence
            population_name: Population identifier (e.g., "FSI") - REQUIRED for RNG independence
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
            region_name=region_name,
            population_name=population_name,
            n_neurons=n_neurons,
            device=device,
            **overrides
        )
        return result

    @staticmethod
    def create_pyramidal_neurons(
        region_name: str,
        population_name: str,
        n_neurons: int,
        device: torch.device,
        **overrides: dict[str, Any],
    ) -> ConductanceLIF:
        """Create standard pyramidal neuron population.

        Standard pyramidal neurons with typical cortical/hippocampal parameters:
        - τ_mem = 20ms ± 15% heterogeneity (breaks synchrony)
        - Fast excitatory (AMPA, 5ms)
        - Slower inhibitory (GABA_A, 10ms)
        - Standard threshold and reversal potentials

        For layer-specific tau_mem, use create_cortical_layer_neurons() instead:
        - L2/3: 25ms (longer integration for recurrent processing)
        - L4: 20ms (fast sensory integration)
        - L5: 30ms (slower for output stability)

        Args:
            region_name: Brain region identifier (e.g., "cortex", "thalamus") - REQUIRED for RNG independence
            population_name: Population identifier (e.g., "FSI") - REQUIRED for RNG independence
            n_neurons: Number of neurons in population
            device: Device for tensor allocation
            **overrides: Custom parameters to override defaults (e.g., adapt_increment, tau_adapt, tau_mem)

        Returns:
            ConductanceLIF neuron population with standard pyramidal configuration
        """
        # Add temporal heterogeneity to prevent synchronization
        # Biology: Pyramidal neurons show 15-20% variability in tau_mem
        if "tau_mem" not in overrides:
            tau_mem_mean = 20.0
            tau_mem_std = 3.0  # 15% CV
            tau_mem_heterogeneous = torch.normal(
                mean=tau_mem_mean,
                std=tau_mem_std,
                size=(n_neurons,),
                device=device
            ).clamp(min=15.0, max=25.0)  # Biological range
        else:
            # Allow override for uniform tau_mem if explicitly specified
            tau_mem_heterogeneous = overrides.pop("tau_mem")  # Remove from overrides

        config_params = {
            "region_name": region_name,
            "population_name": population_name,
            "device": device,
            "tau_mem": tau_mem_heterogeneous,  # Per-neuron heterogeneity
            "g_L": 0.05,
            "tau_E": 5.0,
            "tau_I": 10.0,
            "v_reset": 0.0,
            "E_L": 0.0,
            "E_E": 3.0,
            "E_I": -0.5,
        }
        config_params.update(overrides)
        config = ConductanceLIFConfig(**config_params)
        neurons = ConductanceLIF(n_neurons=n_neurons, config=config)
        return neurons

    @staticmethod
    def create_relay_neurons(
        region_name: str,
        population_name: str,
        n_neurons: int,
        device: torch.device,
        **overrides: dict[str, Any],
    ) -> ConductanceLIF:
        """Create thalamic relay neuron population.

        Thalamic relay neurons with typical sensory relay properties:
        - Faster membrane dynamics (τ_mem = 18ms) for responsive relay
        - Fast excitatory transmission (5ms, sensory input)
        - Slower inhibitory (10ms, from TRN)
        - Standard excitability parameters

        Args:
            region_name: Brain region identifier (e.g., "cortex", "thalamus") - REQUIRED for RNG independence
            population_name: Population identifier (e.g., "FSI") - REQUIRED for RNG independence
            n_neurons: Number of relay neurons
            device: Device for tensor allocation
            **overrides: Custom parameters to override defaults

        Returns:
            ConductanceLIF neuron population configured for thalamic relay
        """
        config_params = {
            "region_name": region_name,
            "population_name": population_name,
            "device": device,
            "v_threshold": 0.8,  # LOWERED from 1.0 to allow T-channel bursts to trigger spikes
            "v_reset": 0.0,
            "E_L": 0.0,
            "E_E": 3.0,
            "E_I": -0.5,
            "g_L": 0.05,
            "tau_mem": 18.0,  # Slightly faster than pyramidal (18ms vs 20ms)
            "tau_E": 5.0,  # Fast excitatory (sensory input)
            "tau_I": 10.0,  # Slower inhibitory (from TRN)
            "tau_ref": 4.0,  # Biological refractory period (3-5ms range, prevents >250Hz firing)
            "adapt_increment": 0.15,  # Add adaptation to prevent sustained high firing from dense input
            "tau_adapt": 100.0,  # Moderate adaptation decay (~100ms)
            # T-type Ca²⁺ channels for intrinsic oscillations (7-14 Hz alpha/theta)
            "enable_t_channels": True,
            "g_T": 0.15,  # Moderate T-channel conductance for alpha oscillations
            "E_Ca": 4.0,  # High Ca²⁺ reversal for strong depolarizing bursts
            "tau_h_T_ms": 50.0,  # 50ms de-inactivation for ~10 Hz rhythms
            "V_half_h_T": -0.3,  # De-inactivates when hyperpolarized below -0.3
            "k_h_T": 0.15,  # Smooth de-inactivation curve
        }
        config_params.update(overrides)
        config = ConductanceLIFConfig(**config_params)
        neurons = ConductanceLIF(n_neurons=n_neurons, config=config)
        return neurons

    @staticmethod
    def create_trn_neurons(
        region_name: str,
        population_name: str,
        n_neurons: int,
        device: torch.device,
        **overrides: dict[str, Any],
    ) -> ConductanceLIF:
        """Create thalamic reticular nucleus (TRN) inhibitory neuron population.

        TRN neurons are inhibitory (GABAergic) with faster dynamics than relay neurons:
        - Faster membrane dynamics for quick attentional gating
        - Faster conductance leak for quicker responses
        - Fast excitatory
        - Fast inhibitory

        Args:
            region_name: Brain region identifier (e.g., "cortex", "thalamus") - REQUIRED for RNG independence
            population_name: Population identifier (e.g., "FSI") - REQUIRED for RNG independence
            n_neurons: Number of TRN neurons
            device: Device for tensor allocation
            **overrides: Custom parameters to override defaults

        Returns:
            ConductanceLIF neuron population configured for TRN
        """
        config_params = {
            "region_name": region_name,
            "population_name": population_name,
            "device": device,
            "v_threshold": 1.6,  # DRASTICALLY INCREASED from 1.0→1.4→1.6 to reduce hyperactivity (18% → 5-15%)
            "v_reset": 0.0,
            "E_L": 0.0,
            "E_E": 3.0,
            "E_I": -0.5,
            "g_L": 0.10,  # INCREASED from 0.06 for faster return to baseline (reduces integration)
            "tau_mem": 12.0,  # Faster membrane for attentional gating
            "tau_E": 4.0,  # Fast excitatory (from relay/cortex)
            "tau_I": 6.0,  # Fast inhibitory (recurrent TRN)
            "adapt_increment": 0.25,  # Strong adaptation to prevent sustained firing (28% FR → 5-15%)
            "tau_adapt": 80.0,  # Medium-slow decay to persist across multiple input events
            # T-type Ca²⁺ channels for stronger pacemaker activity (8-12 Hz alpha)
            "enable_t_channels": True,
            "g_T": 0.20,  # Stronger T-channel conductance for robust alpha generation
            "E_Ca": 4.0,  # High Ca²⁺ reversal for strong depolarizing bursts
            "tau_h_T_ms": 50.0,  # 50ms de-inactivation for ~10 Hz rhythms
            "V_half_h_T": -0.3,  # De-inactivates when hyperpolarized below -0.3
            "k_h_T": 0.15,  # Smooth de-inactivation curve
        }
        config_params.update(overrides)
        config = ConductanceLIFConfig(**config_params)
        neurons = ConductanceLIF(n_neurons=n_neurons, config=config)
        return neurons

    @staticmethod
    def create_cortical_layer_neurons(
        region_name: str,
        population_name: str,
        n_neurons: int,
        device: torch.device,
        **overrides: dict[str, Any],
    ) -> ConductanceLIF:
        """Create layer-specific cortical neurons with temporal heterogeneity.

        Args:
            region_name: Brain region identifier (e.g., "cortex") - REQUIRED for RNG independence
            population_name: Population identifier (e.g., "L2/3", "L4", "L5") - REQUIRED for RNG independence
            n_neurons: Number of neurons in the layer
            device: Device for tensor allocation
            **overrides: Custom parameters to override layer defaults

        Returns:
            ConductanceLIF neuron population configured for the specified layer

        Raises:
            ValueError: If layer is not one of "L2/3", "L4", "L5", "L6a", "L6b"
        """
        # Add temporal heterogeneity to break synchrony and enable emergent oscillations
        # Biology: Cortical pyramidal neurons show 15-20% CV in tau_mem and ~10% in threshold

        # Extract tau_mem from overrides to add heterogeneity
        if "tau_mem" in overrides:
            tau_mem_mean = overrides.pop("tau_mem")  # Remove from overrides
            tau_mem_std = tau_mem_mean * 0.15  # 15% CV (biological variability)
            tau_mem_heterogeneous = torch.normal(
                mean=tau_mem_mean,
                std=tau_mem_std,
                size=(n_neurons,),
                device=device
            ).clamp(min=tau_mem_mean * 0.7, max=tau_mem_mean * 1.3)  # ±30% range
        else:
            # Fallback if no tau_mem specified
            tau_mem_heterogeneous = 20.0

        # Extract v_threshold to add heterogeneity
        if "v_threshold" in overrides:
            v_threshold_mean = overrides.pop("v_threshold")
            v_threshold_std = v_threshold_mean * 0.10  # 10% CV
            v_threshold_heterogeneous = torch.normal(
                mean=v_threshold_mean,
                std=v_threshold_std,
                size=(n_neurons,),
                device=device
            ).clamp(min=v_threshold_mean * 0.85, max=v_threshold_mean * 1.15)
        else:
            v_threshold_heterogeneous = 1.0

        # Base config shared by all layers
        config_params = {
            "region_name": region_name,
            "population_name": population_name,
            "rng_seed": overrides.pop("rng_seed", None),
            "device": device,
            "tau_E": 5.0,  # Fast AMPA (standard for all layers)
            "tau_I": 10.0,  # Slower GABA_A (standard for all layers)
            "E_E": 3.0,  # Excitatory reversal potential
            "E_I": -0.5,  # Inhibitory reversal potential
            "g_L": 0.05,  # Leak conductance
            "tau_mem": tau_mem_heterogeneous,  # Per-neuron temporal diversity
            "v_threshold": v_threshold_heterogeneous,  # Per-neuron threshold diversity
        }

        # Apply caller-provided overrides (remaining parameters)
        config_params.update(overrides)

        config = ConductanceLIFConfig(**config_params)
        neurons = ConductanceLIF(n_neurons=n_neurons, config=config)
        return neurons

    @staticmethod
    def create_fast_spiking_neurons(
        region_name: str,
        population_name: str,
        n_neurons: int,
        device: torch.device,
        **overrides: dict[str, Any],
    ) -> ConductanceLIF:
        """Create fast-spiking interneuron population (parvalbumin+).

        Fast-spiking interneurons (FSI) are characterized by:
        - Very fast membrane time constant (tau_mem = 5ms, ~40 Hz resonance)
        - Ultra-short refractory period (tau_ref = 2ms, allows up to 200 Hz)
        - Fast synaptic kinetics (tau_E/I = 3ms)
        - High leak conductance (g_L = 0.10 for fast membrane decay)
        - Provide feedforward/feedback inhibition
        - Dense gap junction networks for synchronization
        - Critical for gamma oscillations (40-80 Hz)

        Common uses:
        - Striatal FSI (~2% of striatum, parvalbumin+)
        - Cortical basket cells and chandelier cells (PV+)
        - Hippocampal basket cells (generate gamma)
        - Cerebellar basket cells

        Args:
            region_name: Brain region identifier (e.g., "cortex", "thalamus") - REQUIRED for RNG independence
            population_name: Population identifier (e.g., "FSI") - REQUIRED for RNG independence
            n_neurons: Number of FSI neurons to create
            device: Device for tensor allocation ("cpu" or "cuda")
            **overrides: Custom parameters to override defaults

        Returns:
            ConductanceLIF neuron population configured as FSI
        """
        # Fast-spiking configuration (parvalbumin+ interneurons)
        # Optimized for gamma oscillations (40-80 Hz)
        # CRITICAL: Use heterogeneous tau_mem to prevent pathological synchrony
        # Biology: FSI neurons show 30-50% variability in membrane time constants
        # This breaks lockstep firing that creates unrealistic high-frequency peaks

        # Create per-neuron tau_mem with heterogeneity (6-10ms range, mean 8ms)
        # With tau_mem=6ms + tau_ref=2.5ms → max 118 Hz (safe)
        # With tau_mem=8ms + tau_ref=3.5ms → ~87 Hz (gamma range)
        # Allow override if tau_mem is explicitly provided (e.g., for uniform fast dynamics)
        if "tau_mem" not in overrides:
            tau_mem_mean = 8.0  # Increased to prevent ultra-fast spiking
            tau_mem_std = 1.2   # ~15% CV (biological: 15-30%)
            tau_mem_heterogeneous = torch.normal(
                mean=tau_mem_mean,
                std=tau_mem_std,
                size=(n_neurons,),
                device=device
            ).clamp(min=6.0, max=10.0)  # Clamp to 6-10ms range (40-100 Hz gamma)
        else:
            tau_mem_heterogeneous = overrides["tau_mem"]

        config_params = {
            "region_name": region_name,
            "population_name": population_name,
            "device": device,
            "tau_mem": tau_mem_heterogeneous,  # Per-neuron tensor for desynchronization!
            "v_threshold": 1.0,
            "v_reset": 0.0,
            "E_L": 0.0,
            "E_E": 3.0,
            "E_I": -0.5,
            "tau_E": 3.0,  # Fast AMPA (2-4ms biological range)
            "tau_I": 3.0,  # Fast GABA_A (2-4ms for gamma)
            "tau_ref": 2.5,  # Slightly increased from 2.0 for more variability with 60% CV
            "g_L": 0.10,  # High leak conductance for fast decay
        }
        config_params.update(overrides)
        config = ConductanceLIFConfig(**config_params)
        neurons = ConductanceLIF(n_neurons=n_neurons, config=config)
        return neurons

    @staticmethod
    def create_msn_d1_neurons(
        region_name: str,
        population_name: str,
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

        Args:
            region_name: Brain region identifier (e.g., "cortex", "thalamus") - REQUIRED for RNG independence
            population_name: Population identifier (e.g., "FSI") - REQUIRED for RNG independence
            n_neurons: Number of D1-MSN neurons to create
            device: Device for tensor allocation ("cpu" or "cuda")
            **overrides: Custom parameters to override defaults

        Returns:
            ConductanceLIF neuron population configured as D1-MSN
        """
        config_params = {
            "region_name": region_name,
            "population_name": population_name,
            "device": device,
            "v_threshold": 1.0,
            "v_reset": 0.0,
            "E_L": 0.0,
            "E_E": 3.0,
            "E_I": -0.5,
            "tau_E": 5.0,
            "tau_I": 10.0,
            "tau_ref": 2.0,
            "tau_mem": 20.0,
            "tau_adapt": 100.0,
            "adapt_increment": 0.1,
        }
        config_params.update(overrides)
        config = ConductanceLIFConfig(**config_params)
        neurons = ConductanceLIF(n_neurons=n_neurons, config=config)
        return neurons

    @staticmethod
    def create_msn_d2_neurons(
        region_name: str,
        population_name: str,
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

        Args:
            region_name: Brain region identifier (e.g., "cortex", "thalamus") - REQUIRED for RNG independence
            population_name: Population identifier (e.g., "FSI") - REQUIRED for RNG independence
            n_neurons: Number of D2-MSN neurons to create
            device: Device for tensor allocation ("cpu" or "cuda")
            **overrides: Custom parameters to override defaults

        Returns:
            ConductanceLIF neuron population configured as D2-MSN
        """
        config_params = {
            # RNG configuration for independent noise streams
            "region_name": region_name,
            "population_name": population_name,
            "device": device,
            "v_threshold": 1.0,
            "v_reset": 0.0,
            "E_L": 0.0,
            "E_E": 3.0,
            "E_I": -0.5,
            "tau_E": 5.0,
            "tau_I": 10.0,
            "tau_ref": 2.0,
            "tau_mem": 20.0,
            "tau_adapt": 100.0,
            "adapt_increment": 0.1,
        }
        config_params.update(overrides)
        config = ConductanceLIFConfig(**config_params)
        neurons = ConductanceLIF(n_neurons=n_neurons, config=config)
        return neurons


# Register standard neuron types
NeuronFactory.register(NeuronType.PYRAMIDAL)(NeuronFactory.create_pyramidal_neurons)
NeuronFactory.register(NeuronType.RELAY)(NeuronFactory.create_relay_neurons)
NeuronFactory.register(NeuronType.TRN)(NeuronFactory.create_trn_neurons)
NeuronFactory.register(NeuronType.CORTICAL_LAYER)(NeuronFactory.create_cortical_layer_neurons)
NeuronFactory.register(NeuronType.FAST_SPIKING)(NeuronFactory.create_fast_spiking_neurons)
NeuronFactory.register(NeuronType.MSN_D1)(NeuronFactory.create_msn_d1_neurons)
NeuronFactory.register(NeuronType.MSN_D2)(NeuronFactory.create_msn_d2_neurons)
