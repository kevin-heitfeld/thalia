"""
Short-Term Plasticity (STP) - Tsodyks-Markram Model.

Short-term plasticity modulates synaptic strength on fast timescales (10ms-1s)
based on recent presynaptic activity. Unlike long-term plasticity (STDP),
STP is transient and automatically recovers.

Two key phenomena:
1. SHORT-TERM DEPRESSION (STD): Repeated firing depletes vesicles
   - High-frequency bursts → progressively weaker responses
   - Implements a temporal high-pass filter
   - Dominant in cortical pyramidal → interneuron synapses

2. SHORT-TERM FACILITATION (STF): Residual calcium enhances release
   - High-frequency bursts → progressively stronger responses
   - Implements a temporal low-pass filter / coincidence detection
   - Dominant in some cortical pyramidal → pyramidal synapses

The balance between STD and STF determines synapse type:
- Depressing (D): High initial U, fast depression, slow recovery
- Facilitating (F): Low initial U, strong facilitation, fast recovery
- Mixed (E1/E2/E3): Various combinations

Biological basis:
- Vesicle depletion and recycling (depression)
- Residual calcium in presynaptic terminal (facilitation)
- Vesicle pool dynamics (readily releasable, recycling, reserve)

Computational benefits:
- Temporal filtering (high-pass or low-pass depending on synapse type)
- Gain control (automatic normalization of bursts)
- Working memory (facilitation can maintain activity)
- Novelty detection (depressing synapses respond to change)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Dict, Optional

import torch
import torch.nn as nn


class STPType(StrEnum):
    """Predefined synapse types based on Markram et al. (1998) classification."""

    # Depressing synapses (high U, dominant depression)
    DEPRESSING = "depressing"  # Strong initial, rapid fatigue (U=0.5)
    DEPRESSING_MODERATE = "depressing_moderate"  # Moderate depression (U=0.4, thalamic sensory)
    DEPRESSING_STRONG = "depressing_strong"  # Strong depression (U=0.7, thalamic L6 feedback)
    DEPRESSING_FAST = "depressing_fast"  # Very fast depression, quick recovery (U=0.8)

    # Facilitating synapses (low U, dominant facilitation)
    FACILITATING = "facilitating"  # Weak initial, builds up with activity
    FACILITATING_MODERATE = "facilitating_moderate"  # Moderate facilitation (U=0.15, thalamic L6 feedback)
    FACILITATING_STRONG = "facilitating_strong"  # Very strong facilitation

    # Mixed dynamics
    PSEUDOLINEAR = "pseudolinear"  # Balanced, roughly linear response

    # No STP (pass-through)
    NONE = "none"


@dataclass
class STPConfig:
    """Configuration for short-term plasticity.

    The Tsodyks-Markram model uses three variables:
    - u: Utilization (release probability), increases with facilitation
    - x: Available resources (vesicles), decreases with depression
    - Effective transmission = u * x * w

    Dynamics:
    - On pre-spike: u → u + U(1-u), then x → x - u*x
    - Between spikes: u decays to U with τ_f, x recovers to 1 with τ_d

    Attributes:
        U: Baseline release probability (0-1)
            High U (~0.5-0.9) → depressing synapse
            Low U (~0.05-0.2) → facilitating synapse

        tau_d: Depression recovery time constant (ms)
            Time for vesicle pool to refill after depletion.
            Typical: 200-800ms for cortical synapses

        tau_f: Facilitation decay time constant (ms)
            Time for residual calcium to clear.
            Typical: 50-200ms for cortical synapses
    """

    U: float = 0.5  # Baseline release probability
    tau_d: float = 200.0  # Depression recovery (ms)
    tau_f: float = 50.0  # Facilitation decay (ms)

    device: torch.device = torch.device("cpu")

    @classmethod
    def from_type(cls, stp_type: STPType) -> STPConfig:
        """Create config from predefined synapse type.

        Parameters based on Markram et al. (1998) fits to cortical data.
        """
        if stp_type == STPType.DEPRESSING:
            return cls(U=0.5, tau_d=800.0, tau_f=20.0)
        elif stp_type == STPType.DEPRESSING_MODERATE:
            return cls(U=0.4, tau_d=700.0, tau_f=30.0)
        elif stp_type == STPType.DEPRESSING_STRONG:
            return cls(U=0.7, tau_d=600.0, tau_f=15.0)
        elif stp_type == STPType.DEPRESSING_FAST:
            return cls(U=0.8, tau_d=200.0, tau_f=10.0)
        elif stp_type == STPType.FACILITATING:
            return cls(U=0.15, tau_d=200.0, tau_f=200.0)
        elif stp_type == STPType.FACILITATING_MODERATE:
            return cls(U=0.1, tau_d=300.0, tau_f=300.0)
        elif stp_type == STPType.FACILITATING_STRONG:
            return cls(U=0.05, tau_d=100.0, tau_f=500.0)
        elif stp_type == STPType.PSEUDOLINEAR:
            return cls(U=0.3, tau_d=400.0, tau_f=100.0)
        else:  # NONE
            return cls(U=1.0, tau_d=1e6, tau_f=1e-6)  # Effectively no STP


class ShortTermPlasticity(nn.Module):
    """Tsodyks-Markram short-term plasticity model.

    Modulates synaptic efficacy based on recent presynaptic activity.
    Returns a multiplicative factor (0-1) to apply to synaptic weights.

    The model tracks two state variables per synapse:
    - u: Release probability (facilitation variable)
    - x: Available resources (depression variable)

    On each presynaptic spike:
    1. u increases (facilitation): u → u + U(1-u)
    2. x decreases (depression): x → x - u*x
    3. Transmission efficacy = u * x (before the x update)

    Between spikes:
    - u decays toward U with time constant τ_f
    - x recovers toward 1 with time constant τ_d

    Args:
        n_pre: Number of presynaptic neurons
        n_post: Number of postsynaptic neurons (for per-synapse STP)
            If None, uses per-presynaptic-neuron STP (shared across targets)
        config: STP configuration parameters
    """

    def __init__(self, n_pre: int, n_post: int, config: STPConfig):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.config = config

        # Register constants
        self.register_buffer("U", torch.tensor(self.config.U, dtype=torch.float32))

        # Decay factors (computed from dt in update_temporal_parameters)
        # Initialize to dummy values - must call update_temporal_parameters() before use
        self._dt_ms: Optional[float] = None
        self.register_buffer("decay_d", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("decay_f", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("recovery_d", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("recovery_f", torch.tensor(0.0, dtype=torch.float32))

        # Initialize state variables
        shape = (self.n_pre, self.n_post)
        # Release probability (facilitation)
        self.u: torch.Tensor = torch.full(shape, self.U.item(), device=self.config.device, dtype=torch.float32)
        # Available resources (depression)
        self.x: torch.Tensor = torch.ones(shape, device=self.config.device, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Compute STP efficacy for current timestep (ADR-005: 1D tensors).

        Args:
            pre_spikes: Presynaptic spikes [n_pre] (1D)

        Returns:
            Efficacy factor:
            - [n_pre, n_post] (2D matrix)

            Multiply this with synaptic weights to get effective transmission.
        """
        assert (
            pre_spikes.dim() == 1
        ), f"STP.forward: Expected 1D pre_spikes (ADR-005), got shape {pre_spikes.shape}"
        assert (
            pre_spikes.shape[0] == self.n_pre
        ), f"STP.forward: pre_spikes has {pre_spikes.shape[0]} neurons, expected {self.n_pre}"

        # [n_pre] → [n_pre, 1] for broadcasting to [n_pre, n_post]
        spikes = pre_spikes.unsqueeze(-1)

        # === Continuous dynamics (between spikes) ===
        # u decays toward U: u(t+dt) = u(t)*decay_f + U*(1-decay_f)
        # x recovers toward 1: x(t+dt) = x(t)*decay_d + 1*(1-decay_d)

        self.u = self.u * self.decay_f + self.recovery_f
        self.x = self.x * self.decay_d + self.recovery_d

        # === Spike-triggered dynamics ===
        # Compute efficacy BEFORE applying spike effects
        # This is the u*x at the moment of the spike
        efficacy = self.u * self.x

        # On spike: u jumps up (facilitation), x drops (depression)
        # u → u + U(1-u) = u(1-U) + U = lerp toward 1 with step U
        # x → x(1-u) = x - u*x (release fraction u of available pool)

        # IMPORTANT: The x update uses the OLD value of u (before facilitation)
        # This is the Tsodyks-Markram convention: depression is based on the
        # release probability at the time of the spike, not after facilitation

        # Apply spike effects (only where spikes occurred)
        # Depression first (uses current u): x decreases by u*x
        # OPTIMIZATION: Compute x_release directly without clone
        x_release = self.u * self.x  # u*x at spike time (before facilitation)
        self.x = self.x - spikes * x_release

        # Facilitation second: u increases toward 1
        # u → u + U(1-u) when spike occurs
        u_jump = self.U * (1.0 - self.u)
        self.u = self.u + spikes * u_jump

        # Clamp to valid range (numerical safety)
        # OPTIMIZATION: Use clamp_ (in-place) to avoid allocation
        self.u.clamp_(0.0, 1.0)
        self.x.clamp_(0.0, 1.0)

        # Return efficacy (not modulated by spikes - that's the synaptic response)
        # The caller uses this to scale the PSP/PSC
        return efficacy

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update decay factors when brain timestep changes.

        Recomputes cached decay/recovery factors based on new dt.
        Called by DynamicBrain.set_timestep().

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        self._dt_ms = dt_ms
        device = self.U.device

        # Recompute depression decay: exp(-dt / tau_d)
        self.decay_d = torch.tensor(
            float(torch.exp(torch.tensor(-dt_ms / self.config.tau_d)).item()),
            device=device,
            dtype=torch.float32,
        )

        # Recompute facilitation decay: exp(-dt / tau_f)
        self.decay_f = torch.tensor(
            float(torch.exp(torch.tensor(-dt_ms / self.config.tau_f)).item()),
            device=device,
            dtype=torch.float32,
        )

        # Compute recovery rates
        self.recovery_d = torch.tensor(
            1.0 - self.decay_d.item(), device=device, dtype=torch.float32
        )
        self.recovery_f = torch.tensor(
            (1.0 - self.decay_f.item()) * self.config.U, device=device, dtype=torch.float32
        )


# =============================================================================
# STP presets and utilities
# =============================================================================

@dataclass(frozen=True)
class STPPreset:
    """Immutable preset for STP configuration with biological documentation.

    Attributes:
        name: Human-readable name of the preset
        U: Baseline release probability (0-1)
        tau_u: Facilitation time constant (ms) - same as tau_f
        tau_x: Depression recovery time constant (ms) - same as tau_d
        description: Biological context and use case
    """

    name: str
    U: float
    tau_u: float  # Facilitation decay (tau_f in STPConfig)
    tau_x: float  # Depression recovery (tau_d in STPConfig)
    description: str

    def configure(self) -> STPConfig:
        """Create STPConfig with this preset's parameters.

        Returns:
            Configured STPConfig instance
        """
        return STPConfig(
            U=self.U,
            tau_d=self.tau_x,  # tau_x → tau_d (depression recovery)
            tau_f=self.tau_u,  # tau_u → tau_f (facilitation decay)
        )


# =============================================================================
# HIPPOCAMPAL PATHWAY PRESETS
# =============================================================================

MOSSY_FIBER_PRESET = STPPreset(
    name="Mossy Fiber (DG→CA3)",
    U=0.01,  # Reduced from 0.03 to reduce facilitation buildup
    tau_u=400.0,  # Reduced from 800.0 to decay facilitation faster
    tau_x=200.0,
    description=(
        "Dentate gyrus mossy fiber to CA3 pyramidal cells. "
        "Very strong facilitation (low U, long tau_u). "
        "Critical for pattern separation and rapid encoding. "
        "Weak baseline, strong burst response."
    ),
)

SCHAFFER_COLLATERAL_PRESET = STPPreset(
    name="Schaffer Collateral (CA3→CA1)",
    U=0.5,
    tau_u=400.0,
    tau_x=700.0,
    description=(
        "CA3 Schaffer collateral to CA1 pyramidal cells. "
        "Moderate depression (medium U). "
        "Main output pathway from CA3 attractor to CA1 comparator. "
        "Balances reliability with dynamic range."
    ),
)

# =============================================================================
# STRIATAL PATHWAY PRESETS
# =============================================================================

CORTICOSTRIATAL_PRESET = STPPreset(
    name="Corticostriatal",
    U=0.4,
    tau_u=150.0,
    tau_x=250.0,
    description=(
        "Cortical to medium spiny neuron connections. "
        "Moderate depression. "
        "Main input pathway for action selection. "
        "Provides context for reinforcement learning."
    ),
)

THALAMO_STRIATAL_PRESET = STPPreset(
    name="Thalamostriatal (Thalamus→Striatum)",
    U=0.25,
    tau_u=300.0,
    tau_x=400.0,
    description=(
        "Thalamic to striatal connections. "
        "Weak facilitation. "
        "Provides direct sensory and motivational input. "
        "Complements cortical input."
    ),
)

# =============================================================================
# PRESET REGISTRY
# =============================================================================

STP_PRESETS: Dict[str, STPPreset] = {
    # Hippocampal pathways
    "mossy_fiber": MOSSY_FIBER_PRESET,
    "schaffer_collateral": SCHAFFER_COLLATERAL_PRESET,
    # Striatal pathways
    "corticostriatal": CORTICOSTRIATAL_PRESET,
    "thalamostriatal": THALAMO_STRIATAL_PRESET,
}


def get_stp_config(pathway_type: str) -> STPConfig:
    """Get STPConfig for a specific biological pathway.

    Args:
        pathway_type: Name of pathway preset

    Returns:
        Configured STPConfig instance

    Raises:
        KeyError: If pathway_type is not recognized
    """
    if pathway_type not in STP_PRESETS:
        available = ", ".join(STP_PRESETS.keys())
        raise KeyError(f"Unknown pathway_type: {pathway_type}. " f"Available presets: {available}")

    return STP_PRESETS[pathway_type].configure()
