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
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from thalia.utils import clamp_weights

from .weight_init import WeightInitializer


class STPType(Enum):
    """Predefined synapse types based on Markram et al. (1998) classification."""

    # Depressing synapses (high U, dominant depression)
    DEPRESSING = "depressing"  # Strong initial, rapid fatigue (U=0.5)
    DEPRESSING_MODERATE = "depressing_moderate"  # Moderate depression (U=0.4, thalamic sensory)
    DEPRESSING_STRONG = "depressing_strong"  # Strong depression (U=0.7, thalamic L6 feedback)
    DEPRESSING_FAST = "depressing_fast"  # Very fast depression, quick recovery (U=0.8)

    # Facilitating synapses (low U, dominant facilitation)
    FACILITATING = "facilitating"  # Weak initial, builds up with activity
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
        per_synapse: If True, track u,x per synapse; if False, per pre-neuron
    """

    def __init__(
        self,
        n_pre: int,
        n_post: Optional[int] = None,
        config: Optional[STPConfig] = None,
        per_synapse: bool = False,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.config = config or STPConfig()
        self.per_synapse = per_synapse and (n_post is not None)

        # Register constants
        self.register_buffer("U", torch.tensor(self.config.U, dtype=torch.float32))

        # Decay factors (computed from dt in update_temporal_parameters)
        # Initialize to dummy values - must call update_temporal_parameters() before use
        self._dt_ms: Optional[float] = None
        self.register_buffer("decay_d", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("decay_f", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("recovery_d", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("recovery_f", torch.tensor(0.0, dtype=torch.float32))

        # State variables (initialized on first forward)
        self.u: Optional[torch.Tensor] = None  # Release probability (facilitation)
        self.x: Optional[torch.Tensor] = None  # Available resources (depression)

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

    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Compute STP efficacy for current timestep (ADR-005: 1D tensors).

        Args:
            pre_spikes: Presynaptic spikes [n_pre] (1D)

        Returns:
            Efficacy factor:
            - per_synapse=True: [n_pre, n_post] (2D matrix)
            - per_synapse=False: [n_pre] (1D vector)

            Multiply this with synaptic weights to get effective transmission.
        """
        assert (
            pre_spikes.dim() == 1
        ), f"STP.forward: Expected 1D pre_spikes (ADR-005), got shape {pre_spikes.shape}"
        assert (
            pre_spikes.shape[0] == self.n_pre
        ), f"STP.forward: pre_spikes has {pre_spikes.shape[0]} neurons, expected {self.n_pre}"

        # Expand pre_spikes if per_synapse
        if self.per_synapse:
            # [n_pre] → [n_pre, 1] for broadcasting to [n_pre, n_post]
            spikes = pre_spikes.unsqueeze(-1)
        else:
            spikes = pre_spikes

        # === Continuous dynamics (between spikes) ===
        # u decays toward U: u(t+dt) = u(t)*decay_f + U*(1-decay_f)
        # x recovers toward 1: x(t+dt) = x(t)*decay_d + 1*(1-decay_d)

        # Initialize state variables if needed (first forward pass or after loading)
        if self.u is None:
            shape = (self.n_pre, self.n_post) if self.per_synapse else (self.n_pre,)
            self.u = torch.full(shape, self.U.item(), device=pre_spikes.device, dtype=torch.float32)
        if self.x is None:
            shape = (self.n_pre, self.n_post) if self.per_synapse else (self.n_pre,)
            self.x = torch.ones(shape, device=pre_spikes.device, dtype=torch.float32)

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
        u_tensor: torch.Tensor = self.u
        x_tensor: torch.Tensor = self.x
        U_tensor: torch.Tensor = self.U

        u_for_release = u_tensor.clone()  # Save u before facilitation
        x_release = u_for_release * x_tensor
        self.x = x_tensor - spikes * x_release

        # Facilitation second: u increases toward 1
        u_jump = U_tensor * (1.0 - u_tensor)
        self.u = u_tensor + spikes * u_jump

        # Clamp to valid range (numerical safety)
        u_clamped: torch.Tensor = self.u
        x_clamped: torch.Tensor = self.x
        self.u = torch.clamp(u_clamped, 0.0, 1.0)
        self.x = torch.clamp(x_clamped, 0.0, 1.0)

        # Return efficacy (not modulated by spikes - that's the synaptic response)
        # The caller uses this to scale the PSP/PSC
        if self.per_synapse:
            return efficacy
        else:
            return efficacy

    def get_efficacy(self) -> torch.Tensor:
        """Get current efficacy without updating state.

        Useful for analysis or when you need the current state
        without advancing the simulation.
        """
        return self.u * self.x

    def grow(self, n_new: int, target: str = "pre") -> None:
        """Grow STP dimensions by adding new neurons.

        Args:
            n_new: Number of neurons to add
            target: 'pre' or 'post' - which dimension to grow

        Effects:
            - Updates n_pre or n_post
            - Expands state tensors (u, x) with baseline values
        """
        if target == "pre":
            old_n_pre = self.n_pre
            self.n_pre = old_n_pre + n_new

            if self.u is not None and self.x is not None:
                if self.per_synapse:
                    # Add new rows: [old_n_pre, n_post] → [new_n_pre, n_post]
                    u_device = self.U.device
                    assert self.n_post is not None, "n_post must be set for per_synapse mode"
                    shape_2d: tuple[int, int] = (n_new, self.n_post)  # Explicit type for mypy
                    new_u = torch.full(
                        shape_2d,
                        self.config.U,
                        device=u_device,
                        dtype=torch.float32,
                    )
                    x_device = self.x.device
                    new_x = torch.ones(
                        shape_2d,
                        device=x_device,
                        dtype=torch.float32,
                    )
                    self.u = torch.cat([self.u, new_u], dim=0)
                    self.x = torch.cat([self.x, new_x], dim=0)
                else:
                    # Add new elements: [old_n_pre] → [new_n_pre]
                    u_device_1d = self.u.device
                    x_device_1d = self.x.device
                    new_u = torch.full(
                        (n_new,),
                        self.config.U,
                        device=u_device_1d,
                        dtype=torch.float32,
                    )
                    new_x = torch.ones(
                        (n_new,),
                        device=x_device_1d,
                        dtype=torch.float32,
                    )
                    self.u = torch.cat([self.u, new_u], dim=0)
                    self.x = torch.cat([self.x, new_x], dim=0)

        elif target == "post":
            if self.n_post is None:
                raise ValueError("Cannot grow 'post' dimension when n_post is None")

            old_n_post = self.n_post
            self.n_post = old_n_post + n_new

            if self.u is not None and self.x is not None:
                if self.per_synapse:
                    # Add new columns: [n_pre, old_n_post] → [n_pre, new_n_post]
                    u_device_post = self.u.device
                    x_device_post = self.x.device
                    new_u = torch.full(
                        (self.n_pre, n_new),
                        self.config.U,
                        device=u_device_post,
                        dtype=torch.float32,
                    )
                    new_x = torch.ones(
                        (self.n_pre, n_new),
                        device=x_device_post,
                        dtype=torch.float32,
                    )
                    self.u = torch.cat([self.u, new_u], dim=1)
                    self.x = torch.cat([self.x, new_x], dim=1)
                else:
                    # No change needed - per-pre state doesn't depend on n_post
                    pass
        else:
            raise ValueError(f"Unknown target: {target}. Use 'pre' or 'post'.")


class STPSynapse(nn.Module):
    """Synapse with integrated short-term plasticity.

    Combines static weights with dynamic STP modulation.
    The effective weight at any time is: w_eff = w * u * x

    This is a convenience wrapper that handles both the weight matrix
    and the STP dynamics in a single module.

    Args:
        n_pre: Number of presynaptic neurons
        n_post: Number of postsynaptic neurons
        stp_config: STP configuration
        w_init_mean: Mean of initial weights
        w_init_std: Std of initial weights
        w_min: Minimum weight
        w_max: Maximum weight
        per_synapse_stp: If True, track STP per synapse (more memory)
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        stp_config: Optional[STPConfig] = None,
        w_init_mean: float = 0.3,
        w_init_std: float = 0.1,
        w_min: float = 0.0,
        w_max: float = 1.0,
        per_synapse_stp: bool = False,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.w_min = w_min
        self.w_max = w_max

        # Initialize weights
        weights = WeightInitializer.gaussian(
            n_output=n_pre, n_input=n_post, mean=w_init_mean, std=w_init_std, device="cpu"
        )
        weights = clamp_weights(weights, w_min, w_max, inplace=False)
        self.weight = nn.Parameter(weights, requires_grad=False)

        # STP module
        self.stp = ShortTermPlasticity(
            n_pre=n_pre,
            n_post=n_post if per_synapse_stp else None,
            config=stp_config,
            per_synapse=per_synapse_stp,
        )
        self.per_synapse_stp = per_synapse_stp

    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Transmit spikes through synapse with STP modulation.

        Args:
            pre_spikes: Presynaptic spikes, shape (batch, n_pre)

        Returns:
            Postsynaptic current, shape (batch, n_post)
        """
        # Get STP efficacy
        efficacy = self.stp(pre_spikes)

        # Clamp weights
        w = clamp_weights(self.weight, self.w_min, self.w_max, inplace=False)

        if self.per_synapse_stp:
            # efficacy is (batch, n_pre, n_post)
            # w is (n_pre, n_post)
            # pre_spikes is (batch, n_pre)
            effective_w = w * efficacy
            # (batch, n_pre) @ (batch, n_pre, n_post) needs einsum
            post_current = torch.einsum("bi,bij->bj", pre_spikes, effective_w)
        else:
            # efficacy is (batch, n_pre)
            # Modulate spikes by efficacy, then project
            modulated_spikes = pre_spikes * efficacy
            post_current = torch.matmul(modulated_spikes, w)

        return post_current

    def get_effective_weights(self) -> torch.Tensor:
        """Get current effective weights (static × STP)."""
        w = clamp_weights(self.weight, self.w_min, self.w_max, inplace=False)
        efficacy = self.stp.get_efficacy()

        if self.per_synapse_stp:
            # efficacy is (batch, n_pre, n_post)
            return w * efficacy
        else:
            # efficacy is (batch, n_pre), broadcast to (batch, n_pre, 1)
            return w * efficacy.unsqueeze(-1)


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
        references: Key papers validating these parameters
    """

    name: str
    U: float
    tau_u: float  # Facilitation decay (tau_f in STPConfig)
    tau_x: float  # Depression recovery (tau_d in STPConfig)
    description: str
    references: str

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
    U=0.03,
    tau_u=800.0,
    tau_x=200.0,
    description=(
        "Dentate gyrus mossy fiber to CA3 pyramidal cells. "
        "Very strong facilitation (low U, long tau_u). "
        "Critical for pattern separation and rapid encoding. "
        "Weak baseline, strong burst response."
    ),
    references="Salin et al. (1996); Nicoll & Schmitz (2005)",
)

SCHAFFER_COLLATERAL_PRESET = STPPreset(
    name="Schaffer Collateral (CA3→CA1)",
    U=0.5,
    tau_u=400.0,
    tau_x=500.0,
    description=(
        "CA3 Schaffer collateral to CA1 pyramidal cells. "
        "Moderate depression (medium U). "
        "Main output pathway from CA3 attractor to CA1 comparator. "
        "Balances reliability with dynamic range."
    ),
    references="Salin et al. (1996); Dobrunz & Stevens (1997)",
)

EC_CA1_PRESET = STPPreset(
    name="Perforant Path (EC→CA1)",
    U=0.35,
    tau_u=300.0,
    tau_x=400.0,
    description=(
        "Entorhinal cortex direct pathway to CA1 pyramidal cells. "
        "Mild depression. "
        "Provides baseline input for match/mismatch comparison. "
        "Less plastic than CA3→CA1 pathway."
    ),
    references="Bartesaghi & Gessi (2004)",
)

CA3_RECURRENT_PRESET = STPPreset(
    name="CA3 Recurrent (CA3→CA3)",
    U=0.4,
    tau_u=200.0,
    tau_x=300.0,
    description=(
        "CA3 recurrent collaterals (auto-associative connections). "
        "Moderate depression with fast recovery. "
        "Supports pattern completion while preventing runaway excitation. "
        "Short time constants for rapid attractor dynamics."
    ),
    references="Miles & Wong (1986); Debanne et al. (1996)",
)

# =============================================================================
# CORTICAL PATHWAY PRESETS
# =============================================================================

CORTICAL_FF_PRESET = STPPreset(
    name="Cortical Feedforward (L4→L2/3)",
    U=0.2,
    tau_u=200.0,
    tau_x=300.0,
    description=(
        "Thalamocortical and layer 4 to layer 2/3 connections. "
        "Weak facilitation. "
        "Transmits sensory information with temporal filtering. "
        "Responds preferentially to stimulus changes."
    ),
    references="Tsodyks & Markram (1997); Reyes et al. (1998)",
)

CORTICAL_RECURRENT_PRESET = STPPreset(
    name="Cortical Recurrent (L2/3→L2/3)",
    U=0.6,
    tau_u=100.0,
    tau_x=200.0,
    description=(
        "Recurrent excitatory connections within cortical layers. "
        "Strong depression with fast dynamics. "
        "Provides gain control and prevents runaway activity. "
        "Implements divisive normalization."
    ),
    references="Markram et al. (1998); Tsodyks et al. (2000)",
)

CORTICAL_FB_PRESET = STPPreset(
    name="Cortical Feedback (L5→L2/3)",
    U=0.3,
    tau_u=250.0,
    tau_x=350.0,
    description=(
        "Feedback connections from deep to superficial layers. "
        "Mild depression with moderate recovery. "
        "Carries top-down predictions and attentional modulation. "
        "Balanced for sustained modulation."
    ),
    references="Thomson & Bannister (2003)",
)

CORTICAL_PYRAMIDAL_INTERNEURON_PRESET = STPPreset(
    name="Pyramidal→Interneuron",
    U=0.7,
    tau_u=50.0,
    tau_x=150.0,
    description=(
        "Pyramidal cell to inhibitory interneuron. "
        "Strong depression with very fast dynamics. "
        "Provides feedforward inhibition for gain control. "
        "Quick response, rapid fatigue."
    ),
    references="Gupta et al. (2000); Galarreta & Hestrin (1998)",
)

# =============================================================================
# STRIATAL PATHWAY PRESETS
# =============================================================================

CORTICOSTRIATAL_PRESET = STPPreset(
    name="Corticostriatal (Cortex→Striatum)",
    U=0.4,
    tau_u=150.0,
    tau_x=250.0,
    description=(
        "Cortical to medium spiny neuron connections. "
        "Moderate depression. "
        "Main input pathway for action selection. "
        "Provides context for reinforcement learning."
    ),
    references="Charpier et al. (1999); Partridge et al. (2000)",
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
    references="Ding et al. (2008)",
)

# =============================================================================
# PREFRONTAL PATHWAY PRESETS
# =============================================================================

PFC_RECURRENT_PRESET = STPPreset(
    name="PFC Recurrent",
    U=0.15,
    tau_u=400.0,
    tau_x=300.0,
    description=(
        "Prefrontal cortex recurrent connections. "
        "Weak facilitation for working memory. "
        "Long time constants support persistent activity. "
        "Critical for delay period maintenance."
    ),
    references="Wang et al. (2013); Compte et al. (2000)",
)

PFC_TO_STRIATUM_PRESET = STPPreset(
    name="PFC→Striatum",
    U=0.35,
    tau_u=200.0,
    tau_x=300.0,
    description=(
        "Prefrontal to striatal top-down modulation. "
        "Mild depression. "
        "Gates action selection based on goals. "
        "Provides contextual biasing signal."
    ),
    references="Haber et al. (2000)",
)

# =============================================================================
# PRESET REGISTRY
# =============================================================================

STP_PRESETS: Dict[str, STPPreset] = {
    # Hippocampal pathways
    "mossy_fiber": MOSSY_FIBER_PRESET,
    "schaffer_collateral": SCHAFFER_COLLATERAL_PRESET,
    "ec_ca1": EC_CA1_PRESET,
    "ca3_recurrent": CA3_RECURRENT_PRESET,
    # Cortical pathways
    "cortical_ff": CORTICAL_FF_PRESET,
    "cortical_recurrent": CORTICAL_RECURRENT_PRESET,
    "cortical_fb": CORTICAL_FB_PRESET,
    "cortical_pyr_int": CORTICAL_PYRAMIDAL_INTERNEURON_PRESET,
    # Striatal pathways
    "corticostriatal": CORTICOSTRIATAL_PRESET,
    "thalamostriatal": THALAMO_STRIATAL_PRESET,
    # Prefrontal pathways
    "pfc_recurrent": PFC_RECURRENT_PRESET,
    "pfc_striatum": PFC_TO_STRIATUM_PRESET,
}


def get_stp_config(pathway_type: str) -> STPConfig:
    """Get STPConfig for a specific biological pathway.

    Args:
        pathway_type: Name of pathway preset (e.g., "mossy_fiber", "schaffer_collateral")

    Returns:
        Configured STPConfig instance

    Raises:
        KeyError: If pathway_type is not recognized
    """
    if pathway_type not in STP_PRESETS:
        available = ", ".join(STP_PRESETS.keys())
        raise KeyError(f"Unknown pathway_type: {pathway_type}. " f"Available presets: {available}")

    return STP_PRESETS[pathway_type].configure()


def list_presets() -> Dict[str, str]:
    """List all available STP presets with descriptions.

    Returns:
        Dictionary mapping preset names to descriptions
    """
    return {name: preset.description for name, preset in STP_PRESETS.items()}


def sample_heterogeneous_stp_params(
    base_preset: str,
    n_synapses: int,
    variability: float = 0.3,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample heterogeneous STP parameters from biological distributions.

    Biological basis:
        Even within a single pathway type, individual synapses show substantial
        variability in STP parameters (Dobrunz & Stevens 1997, Markram et al. 1998).
        Release probability U can vary 10-fold within the same connection.

        This heterogeneity enables richer temporal filtering: some synapses act as
        high-pass filters (strong depression), others as low-pass (weak depression),
        creating a diverse population of temporal feature detectors.

    Args:
        base_preset: Name of base STP preset (e.g., "corticostriatal")
        n_synapses: Number of synapses to sample parameters for
        variability: Coefficient of variation (std/mean) for parameter sampling
                     Typical biological range: 0.2-0.5
        seed: Random seed for reproducibility

    Returns:
        Tuple of (U_array, tau_d_array, tau_f_array) each shape [n_synapses]
        All arrays contain per-synapse parameters sampled from lognormal distributions
    """
    if seed is not None:
        np.random.seed(seed)

    # Get base configuration
    base_config = get_stp_config(base_preset)

    # Sample from lognormal distributions (ensures positive values)
    # lognormal(mu, sigma) where mean = exp(mu + sigma^2/2)
    # For CV = sigma_y / mu_y = variability, we have:
    # sigma = sqrt(log(1 + CV^2))
    # mu = log(mean) - sigma^2 / 2

    def sample_lognormal(mean_val: float, cv: float, size: int) -> np.ndarray:
        """Sample from lognormal with specified mean and coefficient of variation."""
        sigma = np.sqrt(np.log(1 + cv**2))
        mu = np.log(mean_val) - sigma**2 / 2
        return np.random.lognormal(mu, sigma, size=size)

    # Sample U (release probability)
    U_samples = sample_lognormal(base_config.U, variability, n_synapses)
    U_samples = np.clip(U_samples, 0.01, 0.99)  # Biological bounds

    # Sample tau_d (depression recovery time constant)
    tau_d_samples = sample_lognormal(base_config.tau_d, variability, n_synapses)
    tau_d_samples = np.clip(tau_d_samples, 50.0, 2000.0)  # Biological bounds

    # Sample tau_f (facilitation decay time constant)
    tau_f_samples = sample_lognormal(base_config.tau_f, variability, n_synapses)
    tau_f_samples = np.clip(tau_f_samples, 10.0, 2000.0)  # Biological bounds

    return U_samples, tau_d_samples, tau_f_samples


def create_heterogeneous_stp_configs(
    base_preset: str,
    n_synapses: int,
    variability: float = 0.3,
    seed: Optional[int] = None,
) -> list[STPConfig]:
    """Create list of heterogeneous STP configs for per-synapse dynamics.

    Convenience wrapper around sample_heterogeneous_stp_params() that returns
    a list of STPConfig objects, one per synapse.

    Args:
        base_preset: Name of base STP preset
        n_synapses: Number of synapses
        variability: Coefficient of variation (0.2-0.5 typical)
        seed: Random seed for reproducibility

    Returns:
        List of STPConfig objects, one per synapse
    """
    U_samples, tau_d_samples, tau_f_samples = sample_heterogeneous_stp_params(
        base_preset=base_preset,
        n_synapses=n_synapses,
        variability=variability,
        seed=seed,
    )

    # Create STPConfig for each synapse
    configs = []
    for i in range(n_synapses):
        config = STPConfig(
            U=float(U_samples[i]),
            tau_d=float(tau_d_samples[i]),
            tau_f=float(tau_f_samples[i]),
        )
        configs.append(config)

    return configs
