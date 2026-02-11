"""Unit types for dimensional analysis in neural computations.

Prevents mixing incompatible quantities (currents vs conductances vs voltages).
Uses Python's NewType for zero-runtime-cost type checking with mypy/pyright.

Example usage:
    from thalia.units import Conductance, Current, Voltage

    def neuron_forward(g_exc: Conductance, v_mem: Voltage) -> Current:
        return g_exc * v_mem  # Type checker will verify units match
"""

from typing import NewType
import torch

# =============================================================================
# ELECTRICAL UNITS
# =============================================================================

Voltage = NewType("Voltage", float)
"""Membrane potential or reversal potential (normalized, dimensionless).

In Thalia, voltages are normalized:
- v_rest = 0.0 (resting potential)
- v_threshold = 1.0 (spike threshold)
- E_E = 3.0 (excitatory reversal)
- E_I = -0.5 (inhibitory reversal)

Physical interpretation: ~70 mV range mapped to [0, 1]
"""

Conductance = NewType("Conductance", float)
"""Synaptic or membrane conductance (normalized by leak conductance).

Units: dimensionless (normalized by g_L)
- g_L = 1.0 (leak conductance, reference)
- g_exc ~ 0.1-0.5 (excitatory synaptic conductance)
- g_inh ~ 0.1-0.5 (inhibitory synaptic conductance)

Physical interpretation: Actual conductance / leak conductance
"""

Current = NewType("Current", float)
"""Membrane current (normalized).

Units: dimensionless (normalized by g_L * V_threshold)
- Positive = depolarizing (inward for cations)
- Negative = hyperpolarizing (outward)

Physical interpretation: I = g × (E - V)
For conductance-based models, currents are DERIVED from conductances and voltages.
"""

# =============================================================================
# TENSOR TYPES (for batch operations)
# =============================================================================

VoltageTensor = NewType("VoltageTensor", torch.Tensor)
"""Tensor of voltages [n_neurons] or [batch, n_neurons]."""

ConductanceTensor = NewType("ConductanceTensor", torch.Tensor)
"""Tensor of conductances [n_neurons] or [batch, n_neurons]."""

CurrentTensor = NewType("CurrentTensor", torch.Tensor)
"""Tensor of currents [n_neurons] or [batch, n_neurons]."""

# =============================================================================
# TEMPORAL UNITS
# =============================================================================

TimeMS = NewType("TimeMS", float)
"""Time in milliseconds."""

Frequency = NewType("Frequency", float)
"""Frequency in Hz (spikes per second)."""

FiringRate = NewType("FiringRate", float)
"""Instantaneous firing rate (spikes per timestep, range [0, 1])."""

# =============================================================================
# CONVERSION FUNCTIONS
# =============================================================================

def conductance_to_current(
    g: Conductance | ConductanceTensor,
    v_mem: Voltage | VoltageTensor,
    e_reversal: Voltage,
) -> Current | CurrentTensor:
    """Convert conductance to current using Ohm's law.

    I = g × (E - V)

    Args:
        g: Conductance
        v_mem: Membrane potential
        e_reversal: Reversal potential for this conductance

    Returns:
        Current (type-checked)
    """
    if isinstance(g, torch.Tensor):
        if isinstance(v_mem, torch.Tensor):
            return CurrentTensor(g * (e_reversal - v_mem))
        else:
            return CurrentTensor(g * (e_reversal - v_mem))
    else:
        if isinstance(v_mem, float):
            return Current(g * (e_reversal - v_mem))
        else:
            return CurrentTensor(g * (e_reversal - v_mem))


def current_to_conductance_invalid(i: Current) -> Conductance:
    """INVALID: Cannot convert current to conductance without voltage!

    This function exists to make the type error explicit.
    If you're calling this, you're doing something wrong.

    Raises:
        TypeError: Always, because this conversion is invalid
    """
    raise TypeError(
        "Cannot convert current to conductance without knowing voltage!\n"
        "Current depends on both conductance AND voltage: I = g × (E - V)\n"
        "Use conductance directly instead of passing currents to conductance-based neurons."
    )


# =============================================================================
# TYPE GUARDS
# =============================================================================

def is_conductance(value: float | torch.Tensor) -> bool:
    """Check if value is in valid conductance range.

    Valid conductances are non-negative (g ≥ 0).
    Typical range: 0.0 - 5.0 (normalized by g_L)
    """
    if isinstance(value, torch.Tensor):
        return bool((value >= 0).all())
    return value >= 0.0


def is_voltage(value: float | torch.Tensor) -> bool:
    """Check if value is in valid voltage range.

    Valid voltages are typically in range [-1.0, 4.0] (normalized).
    - Below -1.0: Unusually hyperpolarized
    - Above 4.0: Beyond typical reversal potentials
    """
    if isinstance(value, torch.Tensor):
        return bool((value >= -1.0).all() and (value <= 4.0).all())
    return -1.0 <= value <= 4.0


# =============================================================================
# DOCUMENTATION
# =============================================================================

__all__ = [
    # Basic units
    "Voltage",
    "Conductance",
    "Current",
    "TimeMS",
    "Frequency",
    "FiringRate",
    # Tensor types
    "VoltageTensor",
    "ConductanceTensor",
    "CurrentTensor",
    # Conversions
    "conductance_to_current",
    "current_to_conductance_invalid",
    # Guards
    "is_conductance",
    "is_voltage",
]
