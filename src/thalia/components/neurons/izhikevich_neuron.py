"""Izhikevich Neuron Model - Biologically Plausible Spiking with Rich Dynamics.

The Izhikevich model combines computational efficiency with biological realism,
capable of reproducing all known cortical neuron firing patterns including:
- Tonic spiking (regular firing)
- Phasic spiking (burst then quiet)
- Tonic bursting (repeated bursts)
- Phasic bursting (initial burst)
- Mixed mode (bursts and spikes)
- Spike frequency adaptation
- Class 1 and 2 excitability
- Resonance, rebound spike, subthreshold oscillations, etc.

Model equations:
    dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    du/dt = a*(b*v - u)

    if v >= 30 mV:
        v := c
        u := u + d

Parameters:
    a: recovery time constant (smaller = slower recovery)
    b: sensitivity of recovery variable u to voltage v
    c: after-spike reset value for voltage
    d: after-spike reset increment for recovery variable

This is ideal for dopamine neurons with pacemaker dynamics that need to
coexist with tonic inhibition.

Reference: Izhikevich, E.M. (2003). Simple model of spiking neurons.
IEEE Transactions on Neural Networks, 14(6), 1569-1572.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from thalia.units import ConductanceTensor, VoltageTensor


@dataclass
class IzhikevichNeuronConfig:
    """Configuration for Izhikevich neuron model.

    For dopamine-like pacemaker neurons, use:
        a=0.02, b=0.2, c=-65, d=8 (regular spiking with adaptation)
    Or:
        a=0.02, b=0.25, c=-55, d=0.05 (tonic spiking / pacemaker)

    Note: Voltages in Izhikevich are typically in mV (not normalized).
    We'll use a semi-normalized version: v in [-100, 40] maps to [-1, 0.4] normalized.
    """
    device: torch.device = torch.device("cpu")

    # Izhikevich parameters for pacemaker/tonic spiking
    a: float = 0.02  # Recovery time constant (smaller = slower)
    b: float = 0.25  # Recovery sensitivity to voltage
    c: float = -55.0  # Reset voltage (mV)
    d: float = 0.05  # Reset recovery increment

    # Thresholds and scaling
    v_threshold: float = 30.0  # Spike threshold (mV)
    v_rest: float = -65.0  # Resting potential (mV)

    # Input current scaling (converts conductance to effective current)
    # g_exc * current_scale gives excitatory current
    # g_inh * current_scale * (-1) gives inhibitory current
    excitatory_current_scale: float = 100.0  # Amplification for excitatory input
    inhibitory_current_scale: float = 80.0  # Amplification for inhibitory input

    # Tonic pacemaker current (intrinsic depolarizing drive)
    i_tonic: float = 5.0  # Constant input current for pacemaking (~4-5 Hz)

    # Noise for stochastic firing
    noise_std: float = 1.0  # Standard deviation of current noise (mV)


class IzhikevichNeuron(nn.Module):
    """Izhikevich neuron with pacemaker dynamics for dopamine neurons.

    This model naturally handles:
    1. Intrinsic pacemaking (via i_tonic)
    2. Tonic inhibition (just reduces effective current, no shunting!)
    3. Burst/pause dynamics (via large positive/negative currents)
    4. Spike-frequency adaptation (via recovery variable u)

    Unlike conductance-based LIF, inhibitory currents don't create shunting,
    so tonic inhibition can coexist with pacemaking without silencing.
    """

    def __init__(self, n_neurons: int, config: IzhikevichNeuronConfig):
        """Initialize Izhikevich neuron population.

        Args:
            n_neurons: Number of neurons in population
            config: Configuration parameters
        """
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config
        self.device = config.device

        # State variables
        self.v = torch.full(
            (n_neurons,), config.v_rest, device=self.device, dtype=torch.float32
        )  # Voltage (mV)
        self.u = torch.full(
            (n_neurons,), config.b * config.v_rest, device=self.device, dtype=torch.float32
        )  # Recovery variable

        # Initialize with heterogeneous voltages for desynchronization
        self.v += torch.randn(n_neurons, device=self.device) * 5.0

    @property
    def membrane(self) -> VoltageTensor:
        """Return normalized membrane voltage for compatibility."""
        # Convert from mV to normalized units: v_mV ∈ [-100, 40] → v_norm ∈ [-1, 0.4]
        v_normalized = (self.v + 100.0) / 100.0 - 1.0
        return VoltageTensor(v_normalized)

    def __call__(self, *args, **kwds):
        assert False, f"{self.__class__.__name__} instances should not be called directly. Use forward() instead."
        return super().__call__(*args, **kwds)

    @torch.no_grad()
    def forward(
        self,
        g_exc_input: Optional[ConductanceTensor],
        g_inh_input: Optional[ConductanceTensor],
        rpe_drive: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, VoltageTensor]:
        """Update Izhikevich neurons for one timestep (1ms).

        Args:
            g_exc_input: Excitatory conductance (converted to current)
            g_inh_input: Inhibitory conductance (converted to current)
            rpe_drive: RPE modulation (optional, per-neuron tensor or scalar)

        Returns:
            (spikes, voltage): Boolean spike tensor and voltage tensor
        """
        # Convert conductances to currents (no shunting in Izhikevich!)
        I_exc = 0.0
        if g_exc_input is not None:
            I_exc = g_exc_input * self.config.excitatory_current_scale

        I_inh = 0.0
        if g_inh_input is not None:
            # Inhibition is negative current
            I_inh = -g_inh_input * self.config.inhibitory_current_scale

        # Tonic pacemaker current
        I_tonic = self.config.i_tonic

        # RPE modulation (adds/subtracts current for burst/pause)
        I_rpe = 0.0
        if rpe_drive is not None:
            # Scale RPE: ±1 RPE → ±20 mV current change
            if isinstance(rpe_drive, torch.Tensor):
                I_rpe = rpe_drive * 20.0
            else:
                I_rpe = torch.full((self.n_neurons,), rpe_drive * 20.0, device=self.device)

        # Noise for stochastic firing
        I_noise = torch.randn(self.n_neurons, device=self.device) * self.config.noise_std

        # Total input current
        I_total = I_exc + I_inh + I_tonic + I_rpe + I_noise

        # Izhikevich dynamics (dt = 1ms)
        # Use smaller sub-steps for numerical stability
        dt = 0.5  # 0.5ms sub-steps
        for _ in range(2):  # 2 steps = 1ms total
            # dv/dt = 0.04*v^2 + 5*v + 140 - u + I
            dv = (0.04 * self.v**2 + 5.0 * self.v + 140.0 - self.u + I_total) * dt
            self.v = self.v + dv

            # du/dt = a*(b*v - u)
            du = self.config.a * (self.config.b * self.v - self.u) * dt
            self.u = self.u + du

        # Spike detection and reset
        spikes = self.v >= self.config.v_threshold

        # Reset spiked neurons
        self.v = torch.where(spikes, torch.tensor(self.config.c, device=self.device), self.v)
        self.u = torch.where(spikes, self.u + self.config.d, self.u)

        return spikes, self.membrane
