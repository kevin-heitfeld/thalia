"""Acetylcholine Neuron: Specialized Neuron Type for NB with Fast, Brief Bursts.

Acetylcholine neurons in the Nucleus Basalis (NB) exhibit characteristic firing
patterns that encode attention and encoding/retrieval mode switching:

1. **Tonic Pacemaking** (2-5 Hz baseline):
   - Moderate baseline firing (higher than NE, lower than DA)
   - Represents retrieval mode / low attention state
   - Provides background cholinergic tone

2. **Phasic Bursting**:
   - **Burst** (10-20 Hz): High attention, novelty, prediction errors
   - **Duration**: 50-100ms (brief, fast bursts)
   - **Effect**: Switch to encoding mode, enhance attention

Biophysical Mechanisms:
=======================
- **I_h (HCN channels)**: Moderate strength (2-5 Hz baseline)
- **Fast bursts**: Rapid depolarization → brief high-frequency firing
- **SK channels**: Fast adaptation (limits burst duration to 50-100ms)
- **Selective projections**: Mainly to cortex and hippocampus (not striatum)

This specialized neuron type is used exclusively by the NB region to encode
attention shifts and prediction errors through fast, brief bursts.

References:
- Hasselmo & McGaughy (2004): ACh and cortical function
- Sarter & Parikh (2005): Cholinergic systems and attention
- Hangya et al. (2015): Central cholinergic neurons
- Gu & Yakel (2011): Cholinergic coordination of prefrontal cortex

Author: Thalia Project
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig


@dataclass
class AcetylcholineNeuronConfig(ConductanceLIFConfig):
    """Configuration for acetylcholine neurons with fast burst dynamics.

    Extends base LIF with parameters for:
    - Moderate I_h pacemaking (2-5 Hz baseline)
    - Fast SK adaptation (brief bursts)
    - Prediction error-driven bursts

    Biological parameters based on:
    - Hangya et al. (2015): BF cholinergic neuron properties
    - Sarter & Parikh (2005): Cholinergic neuron electrophysiology
    - Hasselmo & McGaughy (2004): ACh dynamics
    """

    # Membrane properties
    tau_mem: float = 12.0  # Fast membrane (faster than DA/NE)
    v_rest: float = 0.0
    v_reset: float = -0.08  # Shallow reset (allows rapid bursting)
    v_threshold: float = 1.0
    tau_ref: float = 1.5  # Very short refractory (enables fast bursts)

    # Leak conductance (tuned for moderate tonic firing)
    g_L: float = 0.083  # tau_m = C_m/g_L = 12ms

    # Reversal potentials
    E_L: float = 0.0
    E_E: float = 3.0
    E_I: float = -0.5

    # Synaptic time constants
    tau_E: float = 4.0  # Fast excitation
    tau_I: float = 8.0

    # I_h pacemaking current (HCN channels) - MODERATE
    # Between NE and DA for 2-5 Hz baseline
    i_h_conductance: float = 0.022
    i_h_reversal: float = 0.75

    # SK calcium-activated K+ channels - FAST ADAPTATION
    # Strong and fast → limits burst duration to 50-100ms
    sk_conductance: float = 0.035  # Stronger than DA/NE
    sk_reversal: float = -0.5
    ca_decay: float = 0.88  # Very fast calcium decay (brief bursts)
    ca_influx_per_spike: float = 0.25  # High influx (rapid SK activation)

    # Prediction error modulation parameters
    prediction_error_to_current_gain: float = 25.0  # Strong response to PE
    # High PE → strong burst
    # Low PE → baseline

    # Disable adaptation from base class (we use SK instead)
    adapt_increment: float = 0.0

    # Noise for biological realism
    noise_std: float = 0.02


class AcetylcholineNeuron(ConductanceLIF):
    """Acetylcholine neuron with fast, brief bursts.

    Key features:
    1. Autonomous firing at 2-5 Hz (moderate I_h)
    2. Fast brief bursts (10-20 Hz for 50-100ms) on prediction errors
    3. Fast SK adaptation limits burst duration
    4. Rapid return to baseline after burst

    Usage:
        ```python
        ach_neurons = AcetylcholineNeuron(
            n_neurons=3000,
            config=AcetylcholineNeuronConfig(),
            device=torch.device("cpu")
        )

        # Tonic firing (retrieval mode)
        ach_neurons.forward(i_synaptic=0.0, prediction_error_drive=0.0)

        # Fast burst on prediction error
        ach_neurons.forward(i_synaptic=0.0, prediction_error_drive=1.0)
        ```
    """

    def __init__(
        self,
        n_neurons: int,
        config: AcetylcholineNeuronConfig,
        device: torch.device,
    ):
        """Initialize acetylcholine neuron population.

        Args:
            n_neurons: Number of ACh neurons (~3,000-5,000 in human NB)
            config: Configuration with pacemaking and fast burst parameters
            device: PyTorch device for tensor allocation
        """
        super().__init__(n_neurons, config, device)

        # Store specialized config
        self.ach_config = config

        # SK channel state (calcium-activated K+ for fast adaptation)
        self.ca_concentration = torch.zeros(n_neurons, device=device)
        self.sk_activation = torch.zeros(n_neurons, device=device)

        # Initialize with varied phases (prevent artificial synchronization)
        self.v_mem = torch.rand(n_neurons, device=device) * config.v_threshold * 0.4

    def forward(
        self,
        i_synaptic: torch.Tensor | float = 0.0,
        prediction_error_drive: float = 0.0,
    ) -> torch.Tensor:
        """Update acetylcholine neurons with prediction error modulation.

        Args:
            i_synaptic: Synaptic input current [n_neurons] or scalar
                       (NB receives input from PFC, amygdala for prediction errors)
            prediction_error_drive: Prediction error magnitude (normalized scalar)
                                  +1.0 = high prediction error (burst)
                                   0.0 = low prediction error (tonic)
                                  (Note: ACh responds to |PE|, not signed RPE)

        Returns:
            Spike tensor [n_neurons], dtype=bool
        """
        # Convert scalar synaptic input to tensor if needed
        if isinstance(i_synaptic, (int, float)):
            i_synaptic = torch.full(
                (self.n_neurons,), float(i_synaptic), device=self.device
            )

        # === Intrinsic Currents ===

        # I_h pacemaking current (moderate strength → 2-5 Hz baseline)
        i_pacemaker = self.ach_config.i_h_conductance * (
            self.ach_config.i_h_reversal - self.v_mem
        )

        # SK adaptation current (fast and strong → brief bursts)
        i_adaptation = (
            -self.ach_config.sk_conductance
            * self.sk_activation
            * (self.v_mem - self.ach_config.sk_reversal)
        )

        # === Prediction Error Modulation ===
        # Convert prediction error magnitude to current drive
        # High |PE| → positive current → depolarization → burst
        # ACh responds to surprise magnitude, not valence
        i_prediction_error = (
            prediction_error_drive * self.ach_config.prediction_error_to_current_gain
        )

        # === Total Current ===
        i_total = i_synaptic + i_pacemaker + i_adaptation + i_prediction_error

        # Add noise for biological realism
        if self.ach_config.noise_std > 0:
            noise = torch.randn_like(i_total) * self.ach_config.noise_std
            i_total = i_total + noise

        # Call parent's forward to update membrane potential and generate spikes
        spikes, _ = super().forward(i_total)

        # === Update Calcium and SK Activation ===
        # Large calcium influx on spike (fast SK activation)
        self.ca_concentration += spikes.float() * self.ach_config.ca_influx_per_spike

        # Fast calcium decay (limits burst duration to 50-100ms)
        self.ca_concentration *= self.ach_config.ca_decay

        # SK activation (sigmoidal function of calcium)
        # High calcium → high SK → hyperpolarization → terminates burst
        self.sk_activation = self.ca_concentration / (self.ca_concentration + 0.3)

        # Store spikes for diagnostic access
        self.spikes = spikes

        return spikes

    def get_firing_rate_hz(self, window_ms: int = 100) -> float:
        """Get average firing rate.

        Args:
            window_ms: Time window for rate estimation (ms, not used in single timestep)

        Returns:
            Average firing rate in Hz
        """
        # Check if spikes have been computed
        if not hasattr(self, "spikes") or self.spikes is None:
            return 0.0

        # Single timestep rate
        spike_rate = self.spikes.float().mean().item()

        # Convert to Hz (spikes per second)
        firing_rate_hz = spike_rate * (1000.0 / self.ach_config.dt_ms)

        return firing_rate_hz

    def reset_state(self):
        """Reset neuron state to baseline."""
        super().reset_state()
        self.ca_concentration.zero_()
        self.sk_activation.zero_()
        # Re-randomize phases
        self.v_mem = (
            torch.rand(self.n_neurons, device=self.device)
            * self.ach_config.v_threshold
            * 0.4
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_acetylcholine_neurons(
    n_neurons: int,
    device: torch.device | None = None,
    **config_overrides,
) -> AcetylcholineNeuron:
    """Factory function for creating acetylcholine neuron populations.

    Args:
        n_neurons: Number of neurons to create
        device: PyTorch device (defaults to CPU)
        **config_overrides: Override default config parameters

    Returns:
        AcetylcholineNeuron instance

    Example:
        ```python
        # Default configuration
        ach_neurons = create_acetylcholine_neurons(n_neurons=3000)

        # Custom configuration
        ach_neurons = create_acetylcholine_neurons(
            n_neurons=5000,
            device=torch.device("cuda"),
            i_h_conductance=0.025,  # Slightly higher baseline
            sk_conductance=0.04,  # Even faster adaptation
        )
        ```
    """
    config = AcetylcholineNeuronConfig(device=str(device or "cpu"), **config_overrides)

    return AcetylcholineNeuron(
        n_neurons=n_neurons, config=config, device=device or torch.device("cpu")
    )
