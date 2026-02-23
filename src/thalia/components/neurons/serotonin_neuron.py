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

References:
===========
- Jacobs & Azmitia (1992): Structure and function of the raphe nuclei
- Haj-Dahmane & Shen (2005): 5-HT1A autoreceptor modulation of DRN neurons
- Liu et al. (2005): Dorsal raphe nucleus electrophysiology
- Miyazaki et al. (2011): Role of serotonin in patience and reward timing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from thalia.typing import ConductanceTensor, VoltageTensor

from .conductance_lif_neuron import ConductanceLIF, ConductanceLIFConfig


@dataclass
class SerotoninNeuronConfig(ConductanceLIFConfig):
    """Configuration for serotonin neurons with autoreceptor feedback.

    Extends base LIF with parameters for:
    - Moderate I_h pacemaking (2-4 Hz baseline)
    - 5-HT1A autoreceptor GIRK channel (slow self-inhibition)
    - Moderate SK adaptation
    - No gap junctions (unlike LC)

    Biological parameters based on:
    - Liu et al. (2005): DRN dorsal raphe neuron properties
    - Haj-Dahmane & Shen (2005): 5-HT1A autoreceptor kinetics
    - Jacobs & Azmitia (1992): Raphe physiology
    """

    # =========================================================================
    # MEMBRANE PROPERTIES
    # =========================================================================
    tau_mem: float = 15.0  # Intermediate membrane time constant
    v_rest: float = 0.0
    v_reset: float = -0.10   # Moderate hyperpolarization reset
    v_threshold: float = 1.0
    tau_ref: float = 2.0    # Moderate refractory period

    # Leak conductance (tuned for 2-4 Hz tonic firing)
    g_L: float = 0.067       # tau_m = C_m/g_L ≈ 15 ms

    # =========================================================================
    # REVERSAL POTENTIALS
    # =========================================================================
    E_L: float = 0.0
    E_E: float = 3.0
    E_I: float = -0.5

    # =========================================================================
    # SYNAPTIC TIME CONSTANTS
    # =========================================================================
    tau_E: float = 5.0       # Moderate excitation
    tau_I: float = 10.0

    # =========================================================================
    # I_H PACEMAKING CURRENT (HCN CHANNELS)
    # =========================================================================
    # Between LC (0.20) and NB/VTA levels — gives 2-4 Hz reliable baseline.
    # Boosts from I_h are used here, similar to NE neuron fix for reliable tonic firing.
    i_h_conductance: float = 0.30
    i_h_reversal: float = 0.76

    # =========================================================================
    # 5-HT1A AUTORECEPTOR (GIRK K+ CHANNEL)
    # =========================================================================
    # 5-HT1A somatodendritic autoreceptors: Gi-coupled → GIRK channels → K+ efflux
    # This produces slow, graded self-inhibition proportional to recent firing.
    # Kinetics: GPCR coupling time constant ~200ms (slow)
    #
    # Biology refs:
    # - Haj-Dahmane & Shen (2005): τ_autoreceptor ~150-250 ms for 5-HT1A
    # - Sprouse & Aghajanian (1987): GIRK current in DRN neurons
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
    Tuned so that tonic 2-4 Hz yields a stable partial self-inhibition (~0.15-0.25
    autoreceptor activation), keeping neurons below saturation.
    """

    # =========================================================================
    # SK CALCIUM-ACTIVATED K+ CHANNELS (SPIKE-FREQUENCY ADAPTATION)
    # =========================================================================
    sk_conductance: float = 0.028   # Moderate adaptation
    sk_reversal: float = -0.5
    ca_decay: float = 0.91          # Moderate calcium decay
    ca_influx_per_spike: float = 0.18

    # =========================================================================
    # SEROTONIN DRIVE MODULATION
    # =========================================================================
    serotonin_drive_gain: float = 20.0
    """Gain converting external serotonin drive scalar to I_h modulation.

    Positive drive → increased I_h → burst above baseline.
    Negative drive (from LHb punishment) → decreased I_h → pause.
    """

    # Disable base-class adaptation (we use SK instead)
    adapt_increment: float = 0.0

    # =========================================================================
    # NOISE
    # =========================================================================
    noise_std: float = 0.075
    """Intrinsic noise for realistic stochastic tonic firing."""


class SerotoninNeuron(ConductanceLIF):
    """Serotonin (5-HT) neuron with autoreceptor feedback and tonic pacemaking.

    Key features:
    1. Autonomous firing at 2-4 Hz (moderate I_h)
    2. 5-HT1A autoreceptor: slow GIRK K+ self-inhibition (τ ~ 200 ms)
    3. SK spike-frequency adaptation (moderate, shorter than NE)
    4. LHb-driven suppression via negative ``serotonin_drive``
    5. No gap junction coupling (unlike LC)
    """

    def __init__(self, n_neurons: int, config: SerotoninNeuronConfig):
        """Initialise serotonin neuron population.

        Args:
            n_neurons: Number of 5-HT neurons (~100,000-200,000 in human DRN)
            config: Configuration with pacemaking and autoreceptor parameters
        """
        super().__init__(n_neurons, config)

        # ── Autoreceptor signal (slow 5-HT1A / GIRK) ───────────────────────
        # Cumulative 5-HT1A activation; decays with autoreceptor_tau_ms
        self._autoreceptor_signal = torch.zeros(n_neurons, device=self.device)

        # ── SK channel state ────────────────────────────────────────────────
        self.ca_concentration = torch.zeros(n_neurons, device=self.device)
        self.sk_activation = torch.zeros(n_neurons, device=self.device)

        # ── Drive cache ─────────────────────────────────────────────────────
        self._current_drive: float = 0.0

        # Initialise with staggered phases to prevent artificial synchrony
        self.v_mem = torch.rand(n_neurons, device=self.device) * config.v_threshold * 0.35

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

        # ── Update SK calcium / adaptation ───────────────────────────────────
        self.ca_concentration += spikes.float() * self.config.ca_influx_per_spike
        self.ca_concentration *= self.config.ca_decay
        self.sk_activation = self.ca_concentration / (self.ca_concentration + 0.3)

        # ── Update 5-HT1A autoreceptor signal ────────────────────────────────
        # Each spike increments the slow autoreceptor accumulator
        dt_ms: float = getattr(self, '_dt_ms', 1.0)
        alpha_auto = dt_ms / self.config.autoreceptor_tau_ms
        self._autoreceptor_signal += spikes.float() * self.config.autoreceptor_gain
        self._autoreceptor_signal -= self._autoreceptor_signal * alpha_auto
        self._autoreceptor_signal.clamp_(min=0.0, max=1.0)

        self.spikes = spikes
        return spikes, self.membrane

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
        g_ih = torch.full((self.n_neurons,), g_ih_modulated, device=self.device)

        # SK adaptation
        g_sk = self.config.sk_conductance * self.sk_activation

        # 5-HT1A autoreceptor GIRK (K+) — per-neuron since autoreceptor_signal is per-neuron
        g_girk = self.config.autoreceptor_conductance * self._autoreceptor_signal

        return [
            (g_ih,  self.config.i_h_reversal),       # I_h pacemaker
            (g_sk,  self.config.sk_reversal),         # SK adaptation
            (g_girk, self.config.autoreceptor_reversal),  # 5-HT1A GIRK
        ]
