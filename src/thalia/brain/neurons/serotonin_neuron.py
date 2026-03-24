"""Serotonin Neuron: Specialized Neuron Type for DRN with Autoreceptor Feedback.

Serotonin (5-HT) neurons in the Dorsal Raphe Nucleus (DRN) exhibit characteristic
firing patterns that encode patience, mood, and behavioral inhibition:

1. **Tonic Pacemaking** (2-4 Hz baseline):
   - Moderate baseline firing (between NE and DA)
   - Represents calm, patient state; background 5-HT tone
   - Driven by intrinsic I_h (HCN channels)

2. **Autoreceptor Self-Inhibition** (5-HT1A):
   - Each spike releases 5-HT → activates 5-HT1A somatodendritic autoreceptors
   - 5-HT1A couples to Gi protein → opens GIRK (K+) channels → hyperpolarization
   - Slow GPCR kinetics (τ ~ 200 ms) → gradual self-inhibition at sustained firing
   - Prevents runaway saturation; limits burst duration

3. **LHb-Driven Suppression** (punishment signal):
   - Lateral habenula (LHb) activity → inhibits DRN (disynaptic or direct)
   - Punishment → LHb burst → DRN pause → disinhibition of aversive circuits
   - Implements the anti-reward serotonin withdrawal mechanism

Biophysical Mechanisms:
=======================
- **I_h (HCN channels)**: Moderate strength (2-4 Hz baseline; stronger than NE)
- **5-HT1A autoreceptor**: Slow inhibitory K+ current (GIRK) proportional to
  cumulative recent spiking; τ_autoreceptor ~ 200 ms (GPCR coupling latency)
- **SK channels**: Moderate spike-frequency adaptation
- **No gap junctions**: DRN neurons are not electrically coupled (unlike LC)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch

from thalia.typing import ConductanceTensor, PopulationName, RegionName, VoltageTensor

from .conductance_lif_neuron import ConductanceLIF, ConductanceLIFConfig


@dataclass
class SerotoninNeuronConfig(ConductanceLIFConfig):
    """Configuration for serotonin neurons with autoreceptor feedback.

    Extends base LIF with parameters for:
    - Moderate I_h pacemaking (2-4 Hz baseline)
    - 5-HT1A autoreceptor GIRK channel (slow self-inhibition)
    - Moderate SK adaptation
    - No gap junctions (unlike LC)
    """
    # =========================================================================
    # Membrane properties
    # =========================================================================
    tau_mem_ms: Union[float, torch.Tensor] = 15.0  # Intermediate membrane time constant
    v_reset: Union[float, torch.Tensor] = -0.10    # Moderate hyperpolarization reset
    v_threshold: Union[float, torch.Tensor] = 1.0  # Standard threshold
    tau_ref: float = 2.0                           # Moderate refractory period
    g_L: Union[float, torch.Tensor] = 0.067        # Leak conductance

    # =========================================================================
    # Reversal potentials (normalized units, E_L = 0 by convention)
    # =========================================================================
    E_E: float = 3.0   # Excitatory (≈ 0mV, well above threshold)
    E_I: float = -0.5  # Inhibitory (≈ -70mV, below rest)

    # =========================================================================
    # Synaptic time constants
    # =========================================================================
    tau_E: float = 5.0         # Fast excitation
    tau_I: float = 10.0

    # NMDA conductance (slow excitation for temporal integration)
    tau_nmda: float = 100.0    # NMDA decay time constant (80-150ms biologically)
    E_nmda: float = 3.0        # NMDA reversal potential (same as AMPA)

    # GABA_B slow inhibitory channel (metabotropic K⁺)
    # Biology: tau_decay ~250-800 ms, deeper hyperpolarisation (E_GABA_B ~ -90 mV)
    tau_GABA_B: float = 400.0  # GABA_B conductance decay (ms); 250-800 ms biologically
    E_GABA_B: float = -0.8     # GABA_B reversal (normalised; more negative than E_I = -0.5)

    # =========================================================================
    # Noise
    # =========================================================================
    noise_std: Union[float, torch.Tensor] = 0.075
    noise_tau_ms: float = 3.0

    # =========================================================================
    # I_h (HCN) pacemaker current parameters
    # =========================================================================
    i_h_conductance: float = 0.30
    i_h_reversal: float = 0.76

    # =========================================================================
    # SK calcium-activated K+ channels (spike-frequency adaptation)
    # =========================================================================
    sk_conductance: float = 0.060   # Raised 0.042→0.060: DR 5-HT at 3.90 Hz (target ≤3); stronger SK after-spike inhibition
    sk_reversal: float = -0.5
    ca_decay: float = 0.91          # Moderate calcium decay
    ca_influx_per_spike: float = 0.18

    # =========================================================================
    # Serotonin drive modulation parameters
    # =========================================================================
    serotonin_drive_gain: float = 20.0
    """Gain converting external serotonin drive scalar to I_h modulation.

    Positive drive → increased I_h → burst above baseline.
    Negative drive (from LHb punishment) → decreased I_h → pause.
    """

    # Disable base-class adaptation (we use SK instead)
    adapt_increment: Union[float, torch.Tensor] = 0.0

    # =========================================================================
    # 5-HT1A somatodendritic autoreceptor parameters
    # =========================================================================
    # Gi-coupled → GIRK channels → K+ efflux
    # This produces slow, graded self-inhibition proportional to recent firing.
    # Kinetics: GPCR coupling time constant ~200ms (slow)
    autoreceptor_conductance: float = 0.06
    """GIRK (K+) conductance per unit autoreceptor activation.

    Higher values → stronger self-inhibition → lower sustained firing rate.
    """
    autoreceptor_reversal: float = -0.5
    """Reversal potential for GIRK channels (same as E_I, K+ near -90 mV normalised)."""

    autoreceptor_tau_ms: float = 200.0
    """Slow decay time constant for 5-HT1A signalling (GPCR coupling latency)."""

    autoreceptor_gain: float = 0.35
    """Fraction of spiking activity that drives autoreceptor signal.

    Higher values → faster / stronger autoreceptor saturation.
    """


class SerotoninNeuron(ConductanceLIF):
    """Serotonin (5-HT) neuron with autoreceptor feedback and tonic pacemaking.

    Key features:
    1. Autonomous firing at 2-4 Hz (moderate I_h)
    2. 5-HT1A autoreceptor: slow GIRK K+ self-inhibition (τ ~ 200 ms)
    3. SK spike-frequency adaptation (moderate, shorter than NE)
    4. LHb-driven suppression via negative ``serotonin_drive``
    5. No gap junction coupling (unlike LC)
    """

    def __init__(
        self,
        n_neurons: int,
        config: SerotoninNeuronConfig,
        region_name: RegionName,
        population_name: PopulationName,
        device: Union[str, torch.device],
    ):
        """Initialise serotonin neuron population."""
        super().__init__(n_neurons, config, region_name, population_name, device)

        # Autoreceptor signal (slow 5-HT1A / GIRK)
        # Cumulative 5-HT1A activation; decays with autoreceptor_tau_ms
        self._autoreceptor_signal = torch.zeros(n_neurons, device=device)

        # SK channel state
        self.ca_concentration = torch.zeros(n_neurons, device=device)
        self.sk_activation = torch.zeros(n_neurons, device=device)

        # Initialise with staggered phases to prevent artificial synchrony
        self.v_mem = torch.rand(n_neurons, device=device) * config.v_threshold * 0.35

        # Drive cache
        self._current_drive: float = 0.0

    @torch.no_grad()
    def forward(
        self,
        g_ampa_input: Optional[ConductanceTensor],
        g_nmda_input: Optional[ConductanceTensor],
        g_gaba_a_input: Optional[ConductanceTensor],
        g_gaba_b_input: Optional[ConductanceTensor],
        serotonin_drive: float,
    ) -> tuple[torch.Tensor, VoltageTensor]:
        """Update serotonin neurons with external drive.

        Args:
            g_ampa_input: AMPA conductance input [n_neurons] (typically None)
            g_nmda_input: NMDA conductance input [n_neurons] (not used for 5-HT)
            g_gaba_a_input: GABA_A conductance input [n_neurons]
            g_gaba_b_input: GABA_B conductance input [n_neurons] (not used for 5-HT)
            serotonin_drive: External 5-HT drive (normalised scalar).
                +1.0 → strong tonic/burst drive (high reward expectation).
                 0.0 → baseline tonic firing (~2-4 Hz).
                -1.0 → strong inhibition (LHb punishment → DRN pause).

        Returns:
            (spikes, membrane): Spike tensor [n_neurons] and membrane potentials
        """
        self._current_drive = serotonin_drive

        # Parent forward performs LIF dynamics + additional conductances
        spikes, _membrane = super().forward(
            g_ampa_input=g_ampa_input,
            g_nmda_input=g_nmda_input,
            g_gaba_a_input=g_gaba_a_input,
            g_gaba_b_input=g_gaba_b_input,
        )
        spikes_float = spikes.float()

        # Update SK calcium / adaptation
        self.ca_concentration += spikes_float * self.config.ca_influx_per_spike
        self.ca_concentration *= self.config.ca_decay
        self.sk_activation = self.ca_concentration / (self.ca_concentration + 0.3)

        # Update 5-HT1A autoreceptor signal
        # Each spike increments the slow autoreceptor accumulator
        dt_ms: float = getattr(self, '_dt_ms', 1.0)
        alpha_auto = dt_ms / self.config.autoreceptor_tau_ms
        self._autoreceptor_signal += spikes_float * self.config.autoreceptor_gain
        self._autoreceptor_signal -= self._autoreceptor_signal * alpha_auto
        self._autoreceptor_signal.clamp_(min=0.0, max=1.0)

        return spikes, self.V_soma

    def _get_additional_conductances(self) -> list[tuple[torch.Tensor, float]]:
        """Compute I_h, SK, and 5-HT1A autoreceptor conductances.

        I_h is modulated by external serotonin drive:
        - Positive drive → higher I_h → firing above baseline
        - Negative drive (LHb punishment) → lower I_h → pause

        Returns:
            List of (conductance_tensor, reversal_potential) tuples
        """
        drive = self._current_drive
        g_ih_base = self.config.i_h_conductance
        # Clamp so drive cannot fully silence I_h (biology: min ~0.1x baseline)
        g_ih_modulated = g_ih_base * max(0.1, 1.0 + 0.7 * drive)
        g_ih = torch.full((self.n_neurons,), g_ih_modulated, device=self.V_soma.device)

        # SK adaptation
        g_sk = self.config.sk_conductance * self.sk_activation

        # 5-HT1A autoreceptor GIRK (K+) — per-neuron since autoreceptor_signal is per-neuron
        g_girk = self.config.autoreceptor_conductance * self._autoreceptor_signal

        return [
            (g_ih,  self.config.i_h_reversal),       # I_h pacemaker
            (g_sk,  self.config.sk_reversal),         # SK adaptation
            (g_girk, self.config.autoreceptor_reversal),  # 5-HT1A GIRK
        ]
