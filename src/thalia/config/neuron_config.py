"""
Shared neuron configuration base classes.

This module provides BaseNeuronConfig, which extracts common parameters
shared across all neuron models (LIF, ConductanceLIF, pathway neurons).

Design Pattern: Template Method + DRY
- Reduces duplication of tau_mem, v_rest, v_reset, v_threshold, tau_ref
- Single source of truth for common neuron parameters
- Child configs inherit and add model-specific parameters
"""

from __future__ import annotations

from dataclasses import dataclass

from thalia.config.base import BaseConfig
from thalia.constants.neuron import (
    TAU_MEM_STANDARD,
    TAU_REF_STANDARD,
    V_RESET_STANDARD,
    V_REST_STANDARD,
    V_THRESHOLD_STANDARD,
)


@dataclass
class BaseNeuronConfig(BaseConfig):
    """Shared neuron parameters across all neuron types.

    This base config extracts the common parameters that appear in every
    neuron model: membrane time constant, resting/reset/threshold voltages,
    and refractory period.

    All specific neuron configs (LIFConfig, ConductanceLIFConfig, etc.)
    should inherit from this to avoid parameter duplication.

    Attributes:
        tau_mem: Membrane time constant in ms (default: 20.0)
            Controls how quickly the membrane potential decays toward rest.
            Larger values = slower decay = longer memory of inputs.
            Standard pyramidal neurons: 15-30ms
            Fast-spiking interneurons: 5-15ms

        v_rest: Resting membrane potential (default: 0.0)
            Membrane potential with no input.
            In normalized units where threshold = 1.0
            Biological equivalent: ~-65mV

        v_reset: Reset potential after spike (default: 0.0)
            Where membrane is set after spike emission.
            Typically equals v_rest for simplicity.
            Biological equivalent: ~-70mV

        v_threshold: Spike threshold (default: 1.0)
            Membrane potential at which spike is emitted.
            In normalized units (threshold = 1.0 by convention)
            Biological equivalent: ~-55mV

        tau_ref: Absolute refractory period in ms (default: 2.0)
            Duration during which neuron cannot fire after a spike.
            Biological range: 1-5ms depending on neuron type
    """

    tau_mem: float = TAU_MEM_STANDARD  # Membrane time constant (ms)
    v_rest: float = V_REST_STANDARD  # Resting potential
    v_reset: float = V_RESET_STANDARD  # Reset after spike
    v_threshold: float = V_THRESHOLD_STANDARD  # Spike threshold
    tau_ref: float = TAU_REF_STANDARD  # Refractory period (ms)
