"""Unit types for dimensional analysis in neural computations.

Prevents mixing incompatible quantities (currents vs conductances vs voltages).
Uses Python's NewType for zero-runtime-cost type checking with mypy/pyright.
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

# =============================================================================
# TENSOR TYPES
# =============================================================================

VoltageTensor = NewType("VoltageTensor", torch.Tensor)
"""Tensor of voltages [n_neurons] or [batch, n_neurons]."""

ConductanceTensor = NewType("ConductanceTensor", torch.Tensor)
"""Tensor of conductances [n_neurons] or [batch, n_neurons]."""

# =============================================================================
# GAP JUNCTION TYPES
# =============================================================================

GapJunctionConductance = NewType("GapJunctionConductance", torch.Tensor)
"""Gap junction conductance tensor [n_neurons].

Gap junctions are electrical synapses with bidirectional current flow.
Unlike chemical synapses with fixed reversals, gap junctions couple to
neighbor voltages dynamically.
"""

GapJunctionReversal = NewType("GapJunctionReversal", torch.Tensor)
"""Dynamic reversal potential for gap junctions [n_neurons].

For gap junctions, the "reversal" is the weighted average of neighbor
voltages, making it time-varying and neuron-specific.

Physics: I_gap[i] = g_gap × (E_eff[i] - V[i])
    where E_eff[i] = Σ_j [g_gap[i,j] × V[j]] / Σ_j g_gap[i,j]
"""
