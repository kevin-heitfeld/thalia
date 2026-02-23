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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from thalia.typing import ConductanceTensor, VoltageTensor

from .conductance_lif_neuron import ConductanceLIF, ConductanceLIFConfig


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
    # Must dominate g_L for pacemaking
    i_h_conductance: float = 0.45
    i_h_reversal: float = 0.77

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

    # Noise for biological realism and tonic firing
    # Tuned to 0.07 for ~2-5 Hz firing rate
    noise_std: float = 0.07


class AcetylcholineNeuron(ConductanceLIF):
    """Acetylcholine neuron with fast, brief bursts.

    Key features:
    1. Autonomous firing at 2-5 Hz (moderate I_h)
    2. Fast brief bursts (10-20 Hz for 50-100ms) on prediction errors
    3. Fast SK adaptation limits burst duration
    4. Rapid return to baseline after burst
    """

    def __init__(self, n_neurons: int, config: AcetylcholineNeuronConfig):
        """Initialize acetylcholine neuron population.

        Args:
            n_neurons: Number of ACh neurons (~3,000-5,000 in human NB)
            config: Configuration with pacemaking and fast burst parameters
        """
        super().__init__(n_neurons, config)

        # SK channel state (calcium-activated K+ for fast adaptation)
        self.ca_concentration = torch.zeros(n_neurons, device=self.device)
        self.sk_activation = torch.zeros(n_neurons, device=self.device)

        # Initialize with varied phases (prevent artificial synchronization)
        self.v_mem = torch.rand(n_neurons, device=self.device) * config.v_threshold * 0.4

    @torch.no_grad()
    def forward(
        self,
        g_ampa_input: Optional[ConductanceTensor],
        g_gaba_a_input: Optional[ConductanceTensor],
        g_nmda_input: Optional[ConductanceTensor],
        prediction_error_drive: float,
    ) -> tuple[torch.Tensor, VoltageTensor]:
        """Update acetylcholine neurons with prediction error modulation.

        Args:
            g_ampa_input: AMPA (fast excitatory) conductance input [n_neurons]
            g_gaba_a_input: GABA_A (fast inhibitory) conductance input [n_neurons]
            g_nmda_input: NMDA (slow excitatory) conductance input [n_neurons] (not used for ACh neurons)
            prediction_error_drive: Prediction error magnitude (normalized scalar)
                                  +1.0 = high prediction error (burst)
                                   0.0 = low prediction error (tonic)

        Returns:
            (spikes, membrane): Spike tensor and membrane potentials
        """
        # Store PE for conductance calculation
        self._current_pe = prediction_error_drive

        # Call parent's forward
        spikes, _membrane = super().forward(
            g_ampa_input=g_ampa_input,
            g_gaba_a_input=g_gaba_a_input,
            g_nmda_input=g_nmda_input,
        )

        # === Update Calcium and SK Activation ===
        # Large calcium influx on spike (fast SK activation)
        self.ca_concentration += spikes.float() * self.config.ca_influx_per_spike

        # Fast calcium decay (limits burst duration to 50-100ms)
        self.ca_concentration *= self.config.ca_decay

        # SK activation (sigmoidal function of calcium)
        # High calcium → high SK → hyperpolarization → terminates burst
        self.sk_activation = self.ca_concentration / (self.ca_concentration + 0.3)

        # Store spikes for diagnostic access
        self.spikes = spikes

        return spikes, self.membrane

    def _get_additional_conductances(self) -> list[tuple[torch.Tensor, float]]:
        """Compute I_h and SK conductances.

        Prediction error modulates I_h for rapid bursting.

        Returns:
            List of (conductance, reversal) tuples
        """
        # I_h pacemaker (modulated by prediction error)
        pe = getattr(self, '_current_pe', 0.0)
        g_ih_base = self.config.i_h_conductance
        g_ih_modulated = g_ih_base * (1.0 + 0.8 * pe)  # Strong PE modulation
        g_ih = torch.full((self.n_neurons,), max(0.0, g_ih_modulated), device=self.device)

        # SK adaptation (strong and fast for brief bursts)
        g_sk = self.config.sk_conductance * self.sk_activation

        return [
            (g_ih, self.config.i_h_reversal),  # I_h pacemaker
            (g_sk, self.config.sk_reversal),   # SK adaptation
        ]
