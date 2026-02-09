"""Dopamine Neuron: Specialized Neuron Type with Pacemaking and Burst/Pause Dynamics.

Dopamine neurons in the VTA exhibit characteristic firing patterns that encode
reward prediction errors (RPE) through two distinct modes:

1. **Tonic Pacemaking** (4-5 Hz baseline):
   - Intrinsic oscillation driven by I_h (HCN channels)
   - Represents background motivation/mood state
   - Provides baseline dopamine tone

2. **Phasic Modulation**:
   - **Burst** (15-20 Hz): Positive RPE (unexpected reward)
   - **Pause** (<1 Hz): Negative RPE (expected reward omitted)
   - Duration: 100-200 ms

Biophysical Mechanisms:
=======================
- **I_h (HCN channels)**: Depolarizing leak current that drives pacemaking
- **SK channels**: Calcium-activated K+ channels that provide adaptation
- **Thin dendrites**: Fast membrane time constant (tau_mem ~ 15 ms)
- **Low threshold**: Enables autonomous firing without input

This specialized neuron type is used exclusively by the VTA region to encode
reward prediction errors through burst and pause dynamics.

Author: Thalia Project
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig


@dataclass
class DopamineNeuronConfig(ConductanceLIFConfig):
    """Configuration for dopamine neurons with pacemaking dynamics.

    Extends base LIF with parameters for:
    - I_h pacemaking current (HCN channels)
    - SK calcium-activated K+ channels (adaptation)
    - RPE-driven burst/pause modulation

    Biological parameters based on:
    - Grace & Bunney (1984): Midbrain DA neuron electrophysiology
    - Hyland et al. (2002): SK channels in DA neurons
    - Ungless & Grace (2012): VTA DA neuron properties
    """

    # Membrane properties (fast dynamics due to thin dendrites)
    tau_mem: float = 15.0  # Faster than typical pyramidal neurons
    v_rest: float = 0.0
    v_reset: float = -0.1  # Slight hyperpolarization after spike
    v_threshold: float = 1.0
    tau_ref: float = 2.0  # Short refractory period

    # Leak conductance (tuned for tonic firing)
    g_L: float = 0.067  # tau_m = C_m/g_L = 15ms

    # Reversal potentials
    E_L: float = 0.0
    E_E: float = 3.0
    E_I: float = -0.5

    # Synaptic time constants
    tau_E: float = 5.0
    tau_I: float = 10.0

    # I_h pacemaking current (HCN channels)
    # This provides the depolarizing drive that causes tonic firing
    i_h_conductance: float = 0.03  # Depolarizing leak
    i_h_reversal: float = 0.8  # Mixed cation channel (above rest, below threshold)

    # SK calcium-activated K+ channels (spike-frequency adaptation)
    # Prevents runaway bursting and provides natural return to baseline
    sk_conductance: float = 0.02
    sk_reversal: float = -0.5  # Hyperpolarizing (like E_I)
    ca_decay: float = 0.95  # Calcium decay per timestep (fast)
    ca_influx_per_spike: float = 0.15  # Calcium increase per spike

    # RPE modulation parameters
    rpe_to_current_gain: float = 15.0  # mV equivalent per RPE unit
    # +1 RPE → +15 mV drive (depolarize → burst)
    # -1 RPE → -15 mV drive (hyperpolarize → pause)

    # Disable adaptation from base class (we use SK instead)
    adapt_increment: float = 0.0

    # Add slight noise for biological realism
    noise_std: float = 0.01


class DopamineNeuron(ConductanceLIF):
    """Dopamine neuron with intrinsic pacemaking and RPE-driven burst/pause.

    Key features:
    1. Autonomous firing at 4-5 Hz (I_h pacemaking)
    2. Bursts to 15-20 Hz on positive RPE
    3. Pauses (silence) on negative RPE
    4. SK adaptation prevents runaway bursting
    5. Fast return to baseline after phasic response

    Usage:
        ```python
        da_neurons = DopamineNeuron(
            n_neurons=20000,
            config=DopamineNeuronConfig(),
            device=torch.device("cpu")
        )

        # Tonic firing (no RPE)
        da_neurons.forward(i_synaptic=0.0, rpe_drive=0.0)

        # Burst on positive RPE
        da_neurons.forward(i_synaptic=0.0, rpe_drive=1.0)  # Strong burst

        # Pause on negative RPE
        da_neurons.forward(i_synaptic=0.0, rpe_drive=-1.0)  # Silence
        ```
    """

    def __init__(
        self,
        n_neurons: int,
        config: DopamineNeuronConfig,
        device: torch.device,
    ):
        """Initialize dopamine neuron population.

        Args:
            n_neurons: Number of dopamine neurons (~20-30k in human VTA)
            config: Configuration with pacemaking and SK parameters
            device: PyTorch device for tensor allocation
        """
        super().__init__(n_neurons, config, device)

        # Store specialized config
        self.da_config = config

        # SK channel state (calcium-activated K+ for adaptation)
        self.ca_concentration = torch.zeros(n_neurons, device=device)
        self.sk_activation = torch.zeros(n_neurons, device=device)

        # Initialize some neurons with varied phases for realistic population dynamics
        # This prevents artificial synchronization
        self.v_mem = torch.rand(n_neurons, device=device) * config.v_threshold * 0.5

    def forward(
        self, i_synaptic: torch.Tensor | float = 0.0, rpe_drive: float = 0.0
    ) -> torch.Tensor:
        """Update dopamine neurons with RPE modulation.

        Args:
            i_synaptic: Synaptic input current [n_neurons] or scalar
                       (Typically near zero for VTA DA neurons - mostly intrinsic)
            rpe_drive: Reward prediction error drive (normalized scalar)
                      +1.0 = strong positive RPE (burst)
                      -1.0 = strong negative RPE (pause)
                       0.0 = no prediction error (tonic)

        Returns:
            Spike tensor [n_neurons], dtype=bool
        """
        # Convert scalar synaptic input to tensor if needed
        if isinstance(i_synaptic, (int, float)):
            i_synaptic = torch.full(
                (self.n_neurons,), float(i_synaptic), device=self.device
            )

        # === Intrinsic Currents ===

        # I_h pacemaking current (depolarizing, drives tonic firing)
        # Current flows when V < E_h (pulls membrane toward E_h)
        i_pacemaker = self.da_config.i_h_conductance * (
            self.da_config.i_h_reversal - self.v_mem
        )

        # SK adaptation current (hyperpolarizing, prevents runaway bursting)
        # Activated by calcium influx during spikes
        i_adaptation = (
            -self.da_config.sk_conductance
            * self.sk_activation
            * (self.v_mem - self.da_config.sk_reversal)
        )

        # === RPE Modulation ===

        # Convert RPE to current drive (scaled by gain parameter)
        # Positive RPE → positive current → depolarization → burst
        # Negative RPE → negative current → hyperpolarization → pause
        i_rpe = rpe_drive * self.da_config.rpe_to_current_gain

        # === Total Current ===

        i_total = i_synaptic + i_pacemaker + i_adaptation + i_rpe

        # Update membrane potential and check for spikes using parent class
        # This handles threshold crossing, reset, refractory period, etc.
        super().forward(i_total)

        # === Update Calcium and SK Channels ===

        # Calcium influx on spike (spike-triggered calcium entry)
        self.ca_concentration += (
            self.spikes.float() * self.da_config.ca_influx_per_spike
        )

        # Calcium decay (fast buffering/extrusion)
        self.ca_concentration *= self.da_config.ca_decay

        # SK activation (sigmoidal function of calcium)
        # More calcium → more SK activation → more adaptation
        self.sk_activation = self.ca_concentration / (self.ca_concentration + 0.5)

        return self.spikes

    def get_firing_rate_hz(self, window_ms: int = 1000) -> float:
        """Get population firing rate in Hz.

        Useful for monitoring whether neurons are in tonic/burst/pause mode.

        Args:
            window_ms: Time window for rate calculation (not used in single timestep)

        Returns:
            Mean firing rate across population in Hz
        """
        # Single timestep rate
        spike_rate = self.spikes.float().mean().item()

        # Convert to Hz (spikes per second)
        # spike_rate is spikes per timestep (dt_ms)
        firing_rate_hz = spike_rate * (1000.0 / self.config.dt_ms)

        return firing_rate_hz


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_dopamine_neurons(
    n_neurons: int,
    device: Optional[torch.device] = None,
    **config_overrides,
) -> DopamineNeuron:
    """Factory function for creating dopamine neuron populations.

    Args:
        n_neurons: Number of neurons to create
        device: PyTorch device (defaults to CPU)
        **config_overrides: Override default config parameters

    Returns:
        DopamineNeuron instance

    Example:
        ```python
        # Default configuration
        da_neurons = create_dopamine_neurons(n_neurons=20000)

        # Custom configuration
        da_neurons = create_dopamine_neurons(
            n_neurons=10000,
            device=torch.device("cuda"),
            i_h_conductance=0.04,  # Stronger pacemaking
            rpe_to_current_gain=20.0  # Stronger RPE response
        )
        ```
    """
    config = DopamineNeuronConfig(device=str(device or "cpu"), **config_overrides)

    return DopamineNeuron(
        n_neurons=n_neurons, config=config, device=device or torch.device("cpu")
    )
