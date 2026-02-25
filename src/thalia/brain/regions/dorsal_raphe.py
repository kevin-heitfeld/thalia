"""Dorsal Raphe Nucleus (DRN) — Serotonin Patience and Mood System.

The DRN is the brain's primary source of serotonin (5-HT), broadcasting
signals related to patience, temporal discounting, mood, and behavioural
inhibition.  DRN serotonin neurons project widely to striatum, prefrontal
cortex, hippocampus, amygdala, and thalamus.

Biological Background:
======================
**Anatomy:**
- Location: Midline midbrain/pons, largest serotonergic nucleus
- ~100,000-200,000 neurons in humans (~85% serotonergic, ~15% GABAergic)
- Receives inhibitory input from lateral habenula (LHb), BLA, and cortex
- Sends diffuse ascending projections to forebrain via medial forebrain bundle

**Serotonin Neuron Firing Patterns:**
1. **Tonic Pacemaking** (2-4 Hz baseline):
   - Intrinsic I_h (HCN channels) + membrane bistability
   - Represents patient / calm state; baseline 5-HT tone
   - Suppressed during sleep, pain, punishment

2. **5-HT1A Autoreceptor Self-Inhibition:**
   - Each spike triggers slow GIRK K+ current (τ ~ 200 ms)
   - Limits burst duration and prevents firing rate saturation
   - Encodes negative feedback around sustained serotonergic drive

3. **LHb-Driven Pauses:**
   - Lateral habenula encodes negative RPE (punishment, reward omission)
   - LHb activates GABAergic interneurons in DRN → 5-HT pause
   - Pause disinhibits aversive circuitry (CeA, BLA) for punishment signalling

**Computational Roles:**
- Temporal discounting (γ modulation): High 5-HT → patient, larger γ
- Reward/approach gating: 5-HT × DA interaction in NAc gates motivation
- Hippocampal specificity: 5-HT2C suppresses CA3→CA1 LTP
- PFC impulsivity control: 5-HT suppresses premature response in vmPFC
- DRN↔LHb anti-reward feedback: key loop in punishment learning

**Inputs (expected SynapseId patterns):**
- ``lateral_habenula`` / ``LHbPopulation.PRINCIPAL``: Excitatory → activates local GABA
  → 5-HT pause.  Modelled as a negative drive on DRN 5-HT neurons.

**Outputs:**
- ``'5ht'`` neuromodulator channel broadcast to all subscribers

**References:**
- Jacobs & Azmitia (1992): Structure and function of raphe nuclei
- Miyazaki et al. (2011, 2012): Dorsal raphe 5-HT and patience/reward timing
- Liu et al. (2005): DRN neuron electrophysiology in vivo
- Dayan & Huys (2008): Serotonin, tonic inhibition, and RL
"""

from __future__ import annotations

from typing import ClassVar, Dict, Optional

import torch

from thalia.brain.configs import DorsalRapheNucleusConfig
from thalia.components import (
    SerotoninNeuron,
    SerotoninNeuronConfig,
)
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
    NeuromodulatorType,
    PopulationName,
    PopulationPolarity,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)

from .neuromodulator_source_region import NeuromodulatorSourceRegion
from .population_names import DRNPopulation, LHbPopulation
from .region_registry import register_region


@register_region(
    "dorsal_raphe",
    aliases=["drn", "serotonin_system", "raphe"],
    description="Dorsal raphe nucleus - serotonin patience and mood system",
    version="1.0",
    author="Thalia Project",
    config_class=DorsalRapheNucleusConfig,
)
class DorsalRapheNucleus(NeuromodulatorSourceRegion[DorsalRapheNucleusConfig]):
    """Dorsal Raphe Nucleus — Serotonin Patience and Mood System.

    Broadcasts 5-HT via tonic pacemaking modulated by LHb punishment signals.
    5-HT1A autoreceptors provide per-neuron slow self-inhibition, and local GABA
    interneurons provide homeostatic gain control.

    Input Populations (via SynapseId routing):
    ------------------------------------------
    - ``lateral_habenula`` / ``LHbPopulation.PRINCIPAL``: Punishment signal
      (high LHb → inhibit DRN → 5-HT pause).

    Output Populations:
    -------------------
    - ``DRNPopulation.SEROTONIN``: 5-HT projection neurons (broadcast as ``'5ht'``)
    """

    # Declare neuromodulator output channel
    neuromodulator_outputs: ClassVar[Dict[NeuromodulatorType, PopulationName]] = {
        '5ht': DRNPopulation.SEROTONIN,
    }

    def __init__(
        self,
        config: DorsalRapheNucleusConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
    ):
        super().__init__(config, population_sizes, region_name)

        self.serotonin_neurons_size = population_sizes[DRNPopulation.SEROTONIN]
        self.gaba_neurons_size = population_sizes[DRNPopulation.GABA]

        # ── 5-HT projection neurons ──────────────────────────────────────────
        self.serotonin_neurons = SerotoninNeuron(
            n_neurons=self.serotonin_neurons_size,
            config=SerotoninNeuronConfig(
                region_name=self.region_name,
                population_name=DRNPopulation.SEROTONIN,
                serotonin_drive_gain=config.tonic_drive_gain * 20.0,
            ),
            device=self.device,
        )

        # ── Local GABAergic interneurons (homeostatic inhibition) ────────────
        # These correspond to the ~15% GABA cells in DRN that regulate
        # 5-HT neuron excitability and mediate LHb-driven pauses.
        self._init_gaba_interneurons(DRNPopulation.GABA, self.gaba_neurons_size)

        # ── Drive normalisation state ────────────────────────────────────────
        if config.drive_normalization:
            self._avg_drive: float = 0.5
            self._drive_count: int = 0

        # ====================================================================
        # REGISTER POPULATIONS
        # ====================================================================
        self._register_neuron_population(
            DRNPopulation.SEROTONIN,
            self.serotonin_neurons,
            polarity=PopulationPolarity.ANY,
        )

        self.to(self.device)

    @torch.no_grad()
    def forward(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Compute serotonin output from tonic drive and LHb punishment input.

        Args:
            synaptic_inputs: Routed spike tensors keyed by SynapseId.
                LHb principal spikes (if present) provide punishment / pause signal.
            neuromodulator_inputs: (not used — DRN is a neuromodulator source)

        Returns:
            RegionOutput with ``DRNPopulation.SEROTONIN`` spike tensor.
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        # ── 1. Extract LHb punishment input ──────────────────────────────────
        lhb_spikes: Optional[torch.Tensor] = None
        for sid, spikes in synaptic_inputs.items():
            if (
                sid.source_region == "lateral_habenula"
                and sid.source_population == LHbPopulation.PRINCIPAL
            ):
                lhb_spikes = spikes
                break

        lhb_rate: float = 0.0
        if lhb_spikes is not None:
            lhb_rate = float(lhb_spikes.float().mean().item())

        # ── 2. Compute serotonin drive ─────────────────────────────────────
        # Positive baseline drive (encodes tonic state)
        # Subtracted by normalised LHb signal (punishment → pause)
        raw_drive = self.config.tonic_drive_gain - lhb_rate * self.config.lhb_inhibition_gain

        if self.config.drive_normalization:
            raw_drive = self._normalize_drive(raw_drive)

        # Clip drive to sensible range for I_h modulation
        serotonin_drive = float(torch.tensor(raw_drive).clamp(-1.0, 1.0).item())

        # ── 3. Update 5-HT neurons ────────────────────────────────────────────
        # Apply GABA feedback from the previous timestep's interneuron activity.
        # This closes the homeostatic loop: primary fires → GABA fires → primary
        # is inhibited next step (one-step causal delay, no circular dependency).
        gaba_feedback = self._get_gaba_feedback_conductance(self.serotonin_neurons_size, gain=0.01)
        serotonin_spikes, _ = self.serotonin_neurons.forward(
            g_ampa_input=None,    # No direct AMPA; drive via I_h modulation
            g_nmda_input=None,
            g_gaba_a_input=ConductanceTensor(gaba_feedback),
            g_gaba_b_input=None,
            serotonin_drive=serotonin_drive,
        )
        # ── 4. Update GABA interneurons (homeostatic inhibition) ─────────────
        serotonin_activity = serotonin_spikes.float().mean().item()
        gaba_spikes = self._step_gaba_interneurons(serotonin_activity)

        region_outputs: RegionOutput = {
            DRNPopulation.SEROTONIN: serotonin_spikes,
            DRNPopulation.GABA: gaba_spikes,
        }
        return self._post_forward(region_outputs)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_gaba_drive(self, primary_activity: float) -> torch.Tensor:
        """Compute excitatory drive for GABA interneurons from 5-HT activity.

        No tonic baseline: GABA only fires proportionally when 5-HT overshoots
        its target (gain=2.0).  The previous baseline=0.3 saturated GABA at
        ~420 Hz which completely silenced serotonin neurons via feedback.

        Args:
            primary_activity: Mean serotonin neuron firing rate.

        Returns:
            Drive conductance tensor [gaba_neurons_size]
        """
        # Tonic baseline REMOVED: a constant 0.3 drive pushes GABA neurons to V*=2.25
        # (threshold=0.9), saturating them at ~420 Hz and fully silencing serotonin
        # neurons via massive GABA feedback. GABA should only fire when serotonin
        # overshoots its target. Match the base-class convention: gain 2.0, no baseline.
        feedback = primary_activity * 2.0

        return torch.full((self.gaba_neurons_size,), feedback, device=self.device)

    def _normalize_drive(self, raw_drive: float) -> float:
        """Adaptive normalisation to prevent saturation / silence.

        Maintains a running mean of the drive signal and scales new drives
        relative to that baseline.  Mirrors the approach in LocusCoeruleus and
        NucleusBasalis for consistency.
        """
        self._drive_count += 1
        alpha = min(0.01, 1.0 / self._drive_count)  # Slow adaptation
        self._avg_drive = (1.0 - alpha) * self._avg_drive + alpha * raw_drive

        # Normalise so that the mean drive stays near 0
        normalised = (raw_drive - self._avg_drive) / (abs(self._avg_drive) + 0.1)
        return float(normalised)

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Propagate timestep change to neuron populations and base class."""
        super().update_temporal_parameters(dt_ms)
        self.serotonin_neurons.update_temporal_parameters(dt_ms)
        self.gaba_neurons.update_temporal_parameters(dt_ms)
