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

References:
- Tsodyks & Markram (1997): The neural code between neocortical pyramidal neurons
- Markram et al. (1998): Differential signaling via the same axon
- Abbott & Regehr (2004): Synaptic computation (review)

Computational benefits:
- Temporal filtering (high-pass or low-pass depending on synapse type)
- Gain control (automatic normalization of bursts)
- Working memory (facilitation can maintain activity)
- Novelty detection (depressing synapses respond to change)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn

from thalia.utils.core_utils import clamp_weights


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

        dt: Simulation timestep (ms)
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

    Example:
        >>> # Per-synapse STP (most accurate) - ADR-005: 1D tensors
        >>> stp = ShortTermPlasticity(n_pre=100, n_post=50, per_synapse=True)
        >>> stp.reset_state()
        >>>
        >>> for t in range(100):
        ...     pre_spikes = ...  # [n_pre] (1D)
        ...     efficacy = stp(pre_spikes)  # [n_pre, n_post] or [n_pre]
        ...     # Modulate weights: effective_w = w * efficacy

        >>> # Depressing synapse for pyramidal→interneuron
        >>> config = STPConfig.from_type(STPType.DEPRESSING)
        >>> stp_dep = ShortTermPlasticity(n_pre=80, n_post=20, config=config)
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

        # State variables (initialized on first forward or reset)
        self.u: Optional[torch.Tensor] = None  # Release probability (facilitation)
        self.x: Optional[torch.Tensor] = None  # Available resources (depression)

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update decay factors when brain timestep changes.

        Recomputes cached decay/recovery factors based on new dt.
        Called by DynamicBrain._broadcast_temporal_update().

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        self._dt_ms = dt_ms
        device = self.U.device  # type: ignore[union-attr]

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
        self.recovery_d = torch.tensor(1.0 - self.decay_d.item(), device=device, dtype=torch.float32)
        self.recovery_f = torch.tensor(
            (1.0 - self.decay_f.item()) * self.config.U, device=device, dtype=torch.float32
        )

    def reset_state(self) -> None:
        """Reset STP state to baseline (ADR-005: 1D tensors).

        - u starts at U (baseline release probability)
        - x starts at 1 (full vesicle pool)
        - Uses 1D tensors per single-brain architecture
        """
        if self._dt_ms is None:
            raise RuntimeError(
                "ShortTermPlasticity.reset_state() called before update_temporal_parameters(). "
                "Brain must call update_temporal_parameters() during initialization."
            )

        device = self.U.device  # type: ignore[union-attr]

        shape: tuple[int, int] | tuple[int]
        if self.per_synapse:
            assert self.n_post is not None, "n_post must be set for per_synapse mode"
            shape = (self.n_pre, self.n_post)
        else:
            shape = (self.n_pre,)

        # Cast device from parameter for mypy (register_buffer returns Union[Tensor, Module])
        dev = torch.device(str(device)) if device else torch.device("cpu")
        self.u = torch.full(shape, self.config.U, device=dev, dtype=torch.float32)
        self.x = torch.ones(shape, device=dev, dtype=torch.float32)

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

        if self.u is None:
            self.reset_state()

        # Expand pre_spikes if per_synapse
        if self.per_synapse:
            # [n_pre] → [n_pre, 1] for broadcasting to [n_pre, n_post]
            spikes = pre_spikes.unsqueeze(-1)
        else:
            spikes = pre_spikes

        # === Continuous dynamics (between spikes) ===
        # u decays toward U: u(t+dt) = u(t)*decay_f + U*(1-decay_f)
        # x recovers toward 1: x(t+dt) = x(t)*decay_d + 1*(1-decay_d)
        self.u = self.u * self.decay_f + self.recovery_f  # type: ignore[operator]
        self.x = self.x * self.decay_d + self.recovery_d  # type: ignore[operator]

        # === Spike-triggered dynamics ===
        # Compute efficacy BEFORE applying spike effects
        # This is the u*x at the moment of the spike
        efficacy = self.u * self.x  # type: ignore[operator]

        # On spike: u jumps up (facilitation), x drops (depression)
        # u → u + U(1-u) = u(1-U) + U = lerp toward 1 with step U
        # x → x(1-u) = x - u*x (release fraction u of available pool)

        # IMPORTANT: The x update uses the OLD value of u (before facilitation)
        # This is the Tsodyks-Markram convention: depression is based on the
        # release probability at the time of the spike, not after facilitation

        # Apply spike effects (only where spikes occurred)
        # Depression first (uses current u): x decreases by u*x
        u_tensor: torch.Tensor = self.u  # type: ignore[assignment]
        x_tensor: torch.Tensor = self.x  # type: ignore[assignment]
        U_tensor: torch.Tensor = self.U  # type: ignore[assignment]

        u_for_release = u_tensor.clone()  # Save u before facilitation
        x_release = u_for_release * x_tensor
        self.x = x_tensor - spikes * x_release  # type: ignore[assignment]

        # Facilitation second: u increases toward 1
        u_jump = U_tensor * (1.0 - u_tensor)
        self.u = u_tensor + spikes * u_jump  # type: ignore[assignment]

        # Clamp to valid range (numerical safety)
        u_clamped: torch.Tensor = self.u  # type: ignore[assignment]
        x_clamped: torch.Tensor = self.x  # type: ignore[assignment]
        self.u = torch.clamp(u_clamped, 0.0, 1.0)  # type: ignore[assignment]
        self.x = torch.clamp(x_clamped, 0.0, 1.0)  # type: ignore[assignment]

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
        if self.u is None or self.x is None:
            raise RuntimeError("STP state not initialized. Call reset_state() first.")
        return self.u * self.x

    def get_state(self) -> dict[str, Optional[torch.Tensor]]:
        """Get current STP state for analysis/saving."""
        efficacy_value: Optional[torch.Tensor] = None
        if self.u is not None and self.x is not None:
            u_tensor: torch.Tensor = self.u
            x_tensor: torch.Tensor = self.x
            efficacy_value = (u_tensor * x_tensor).clone()

        return {
            "u": self.u.clone() if self.u is not None else None,
            "x": self.x.clone() if self.x is not None else None,
            "efficacy": efficacy_value,
        }

    def load_state(self, state: dict[str, Optional[torch.Tensor]]) -> None:
        """Restore STP state from checkpoint.

        Args:
            state: Dictionary from get_state()
        """
        if state["u"] is not None:
            u_device = self.U.device  # type: ignore[union-attr]
            self.u = state["u"].to(u_device)  # type: ignore[arg-type]
        if state["x"] is not None:
            x_device = self.U.device  # type: ignore[union-attr]
            self.x = state["x"].to(x_device)  # type: ignore[arg-type]

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
                    u_device = self.U.device  # type: ignore[union-attr]
                    assert self.n_post is not None, "n_post must be set for per_synapse mode"
                    shape_2d: tuple[int, int] = (n_new, self.n_post)  # Explicit type for mypy
                    new_u = torch.full(
                        shape_2d,
                        self.config.U,
                        device=u_device,  # type: ignore[arg-type]
                        dtype=torch.float32,
                    )
                    x_device = self.x.device  # type: ignore[union-attr]
                    new_x = torch.ones(
                        shape_2d,
                        device=x_device,  # type: ignore[arg-type]
                        dtype=torch.float32,
                    )
                    self.u = torch.cat([self.u, new_u], dim=0)
                    self.x = torch.cat([self.x, new_x], dim=0)
                else:
                    # Add new elements: [old_n_pre] → [new_n_pre]
                    u_device_1d = self.u.device  # type: ignore[union-attr]
                    x_device_1d = self.x.device  # type: ignore[union-attr]
                    new_u = torch.full(
                        (n_new,),
                        self.config.U,
                        device=u_device_1d,  # type: ignore[arg-type]
                        dtype=torch.float32,
                    )
                    new_x = torch.ones(
                        (n_new,),
                        device=x_device_1d,  # type: ignore[arg-type]
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
                    u_device_post = self.u.device  # type: ignore[union-attr]
                    x_device_post = self.x.device  # type: ignore[union-attr]
                    new_u = torch.full(
                        (self.n_pre, n_new),
                        self.config.U,
                        device=u_device_post,  # type: ignore[arg-type]
                        dtype=torch.float32,
                    )
                    new_x = torch.ones(
                        (self.n_pre, n_new),
                        device=x_device_post,  # type: ignore[arg-type]
                        dtype=torch.float32,
                    )
                    self.u = torch.cat([self.u, new_u], dim=1)
                    self.x = torch.cat([self.x, new_x], dim=1)
                else:
                    # No change needed - per-pre state doesn't depend on n_post
                    pass
        else:
            raise ValueError(f"Unknown target: {target}. Use 'pre' or 'post'.")

    def __repr__(self) -> str:
        synapse_str = f"{self.n_pre}→{self.n_post}" if self.n_post else f"{self.n_pre}"
        return (
            f"ShortTermPlasticity({synapse_str}, "
            f"U={self.config.U:.2f}, τ_d={self.config.tau_d:.0f}ms, "
            f"τ_f={self.config.tau_f:.0f}ms)"
        )


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

    Example:
        >>> synapse = STPSynapse(
        ...     n_pre=100, n_post=50,
        ...     stp_config=STPConfig.from_type(STPType.DEPRESSING)
        ... )
        >>> synapse.reset_state(batch_size=1)
        >>>
        >>> for t in range(100):
        ...     pre_spikes = ...
        ...     post_current = synapse(pre_spikes)
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
        weights = torch.randn(n_pre, n_post, requires_grad=False) * w_init_std + w_init_mean
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

    def reset_state(self) -> None:
        """Reset STP state."""
        self.stp.reset_state()

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

    def get_stp_state(self) -> dict:
        """Get STP state for analysis."""
        return self.stp.get_state()

    def __repr__(self) -> str:
        return (
            f"STPSynapse({self.n_pre}→{self.n_post}, "
            f"U={self.stp.config.U:.2f}, "
            f"τ_d={self.stp.config.tau_d:.0f}ms, "
            f"τ_f={self.stp.config.tau_f:.0f}ms)"
        )
