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
from typing import Optional, Union

import torch
import torch.nn as nn

from thalia import GlobalConfig
from thalia.utils import decay_tensor


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

    U: float  # Baseline release probability
    tau_d: float  # Depression recovery (ms)
    tau_f: float  # Facilitation decay (ms)

    def steady_state_utilization(self, rate_hz: float) -> float:
        """Compute the steady-state synaptic utilization factor at a given firing rate.

        Returns ``u_eff = u_ss * x_ss``: the fraction of nominal synaptic weight
        that is actually delivered per spike when the presynaptic neuron fires
        tonically at *rate_hz*.

        Uses the Tsodyks-Markram closed-form steady state:

        .. math::

            u_{ss} = \\frac{U \\,(1 + r \\tau_f)}{1 + U \\, r \\tau_f}, \\quad
            x_{ss} = \\frac{1}{1 + u_{ss} \\, r \\, \\tau_d}, \\quad
            u_{eff} = u_{ss} \\, x_{ss}

        where :math:`r = rate\\_hz / 1000` (spikes/ms).

        Args:
            rate_hz: Mean presynaptic firing rate in Hz.

        Returns:
            Steady-state effective utilization in (0, 1].
        """
        r_ms = rate_hz / 1000.0
        u_ss = (self.U * (1.0 + r_ms * self.tau_f)) / (1.0 + self.U * r_ms * self.tau_f)
        x_ss = 1.0 / (1.0 + u_ss * r_ms * self.tau_d)
        return float(u_ss * x_ss)

    def steady_state_ux(self, rate_hz: float) -> tuple[float, float]:
        """Return the individual steady-state variables ``(u_ss, x_ss)`` at *rate_hz*.

        Unlike :meth:`steady_state_utilization` which returns only the product,
        this returns the two variables separately so callers can pre-load STP
        state tensors (e.g. to avoid onset transients at simulation start).

        Args:
            rate_hz: Mean presynaptic firing rate in Hz.

        Returns:
            ``(u_ss, x_ss)`` — both scalars in (0, 1].
        """
        r_ms = rate_hz / 1000.0
        u_ss = (self.U * (1.0 + r_ms * self.tau_f)) / (1.0 + self.U * r_ms * self.tau_f)
        x_ss = 1.0 / (1.0 + u_ss * r_ms * self.tau_d)
        return float(u_ss), float(x_ss)


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

    def __init__(
        self,
        n_pre: int,
        config: STPConfig,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.config = config

        # Register constants
        self.U: torch.Tensor
        self.register_buffer("U", torch.tensor(self.config.U, device=device, dtype=torch.float32))

        # Decay factors (computed from dt in update_temporal_parameters)
        # Initialize to dummy values - must call update_temporal_parameters() before use
        self._dt_ms: Optional[float] = None

        self.decay_d: torch.Tensor
        self.decay_f: torch.Tensor
        self.recovery_d: torch.Tensor
        self.recovery_f: torch.Tensor
        self.register_buffer("decay_d", torch.tensor(0.0, device=device, dtype=torch.float32))
        self.register_buffer("decay_f", torch.tensor(0.0, device=device, dtype=torch.float32))
        self.register_buffer("recovery_d", torch.tensor(0.0, device=device, dtype=torch.float32))
        self.register_buffer("recovery_f", torch.tensor(0.0, device=device, dtype=torch.float32))

        # Initialize state variables as [n_pre] — STP is per-presynaptic-neuron.
        # The n_post dimension is unnecessary: all postsynaptic targets of the same
        # pre-neuron receive identical efficacy (STP dynamics depend only on pre activity).
        shape = (self.n_pre,)
        # Release probability (facilitation)
        self.u: torch.Tensor = torch.full(shape, self.U.item(), device=device, dtype=torch.float32)
        # Available resources (depression)
        self.x: torch.Tensor = torch.ones(shape, device=device, dtype=torch.float32)

        self.to(device)

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

        # Cache registered buffers once — avoids nn.Module.__getattr__ per access
        U       = self.U
        decay_f = self.decay_f
        decay_d = self.decay_d
        recov_f = self.recovery_f
        recov_d = self.recovery_d

        # === Continuous dynamics (in-place to avoid nn.Module.__setattr__ overhead) ===
        # u decays toward U: u(t+dt) = u(t)*decay_f + U*(1-decay_f)
        # x recovers toward 1: x(t+dt) = x(t)*decay_d + 1*(1-decay_d)
        self.u.mul_(decay_f).add_(recov_f)
        self.x.mul_(decay_d).add_(recov_d)

        # === Spike-triggered dynamics ===
        # Capture pre-spike efficacy (u*x) — this is the correct return value.
        # Must be saved before x is depleted and u is facilitated.
        efficacy = self.u * self.x  # [n_pre]

        # On spike: x drops (depression), u jumps (facilitation)
        # IMPORTANT: depression uses the OLD u — Tsodyks-Markram convention.
        # Depression: x -= spikes * u*x = spikes * efficacy
        self.x.addcmul_(pre_spikes, efficacy, value=-1.0)

        # Facilitation: u += spikes * U*(1-u)
        # U*(1-u) as a small [n_pre] temp — cheap since there is no n_post dimension.
        self.u.addcmul_(pre_spikes, U - U * self.u)

        # Clamp to valid range (numerical safety)
        self.u.clamp_(0.0, 1.0)
        self.x.clamp_(0.0, 1.0)

        # Return pre-spike efficacy u*x — caller scales synaptic weights by this.
        return efficacy  # [n_pre]

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update decay factors when brain timestep changes.

        Recomputes cached decay/recovery factors based on new dt.
        Called by Brain.set_timestep().

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        self._dt_ms = dt_ms
        device = self.U.device

        # Compute decay factors for continuous dynamics: exp(-dt / tau)
        self.decay_d = decay_tensor(dt_ms, self.config.tau_d, device=device)
        self.decay_f = decay_tensor(dt_ms, self.config.tau_f, device=device)

        # Compute recovery rates
        self.recovery_d = torch.tensor(
            1.0 - self.decay_d.item(), device=device, dtype=torch.float32
        )
        self.recovery_f = torch.tensor(
            (1.0 - self.decay_f.item()) * self.config.U, device=device, dtype=torch.float32
        )

    @torch.no_grad()
    def initialize_to_steady_state(self, rate_hz: float) -> None:
        """Set u and x to their Tsodyks-Markram steady-state values for *rate_hz*.

        Prevents the large onset transient that occurs when simulations start
        with u=U, x=1 (resting state) while weights are scaled for steady-state
        STP depletion.  After this call the first spike delivers the same
        conductance as every subsequent spike at the given firing rate.

        Call this after :meth:`update_temporal_parameters` has been called (i.e.
        after :meth:`BrainBuilder.build`).

        Args:
            rate_hz: Expected presynaptic firing rate in Hz.
        """
        u_ss, x_ss = self.config.steady_state_ux(rate_hz)
        self.u.fill_(u_ss)
        self.x.fill_(x_ss)
