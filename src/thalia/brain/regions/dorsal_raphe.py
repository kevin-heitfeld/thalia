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
"""

from __future__ import annotations

from typing import ClassVar, Dict, Optional, Union

import torch

from thalia import GlobalConfig
from thalia.brain.adaptive_normalization import AdaptiveNormalization
from thalia.brain.configs import DorsalRapheNucleusConfig
from thalia.brain.neurons import (
    SerotoninNeuronConfig,
    SerotoninNeuron,
    heterogeneous_dendrite_coupling,
    heterogeneous_noise_std,
    heterogeneous_tau_mem,
    heterogeneous_v_reset,
    heterogeneous_v_threshold,
    heterogeneous_g_L,
)
from thalia.typing import (
    NeuromodulatorInput,
    NeuromodulatorChannel,
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
)
class DorsalRapheNucleus(NeuromodulatorSourceRegion[DorsalRapheNucleusConfig]):
    """Dorsal Raphe Nucleus — Serotonin Patience and Mood System.

    Broadcasts 5-HT via tonic pacemaking modulated by LHb punishment signals
    and PFC top-down excitatory input.
    5-HT1A autoreceptors provide per-neuron slow self-inhibition, and local GABA
    interneurons provide homeostatic gain control.

    Input Populations (via SynapseId routing):
    ------------------------------------------
    - ``lateral_habenula`` / ``LHbPopulation.PRINCIPAL``: Punishment signal
      (high LHb → inhibit DRN → 5-HT pause).
    - ``prefrontal_cortex`` / ``CortexPopulation.L5_PYR``: Top-down excitatory
      control (active PFC → boost DRN → patience/impulse control).

    Output Populations:
    -------------------
    - ``DRNPopulation.SEROTONIN``: 5-HT projection neurons (broadcast as ``'5ht'``)
    """

    # Declare neuromodulator output channel
    neuromodulator_outputs: ClassVar[Dict[NeuromodulatorChannel, PopulationName]] = {
        NeuromodulatorChannel.SHT: DRNPopulation.SEROTONIN,
    }

    def __init__(
        self,
        config: DorsalRapheNucleusConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        super().__init__(config, population_sizes, region_name, device=device)

        self.serotonin_neurons_size = population_sizes[DRNPopulation.SEROTONIN]
        self.gaba_size = population_sizes[DRNPopulation.GABA]

        # ── 5-HT projection neurons ──────────────────────────────────────────
        self.serotonin_neurons: SerotoninNeuron
        self.serotonin_neurons = self._create_and_register_neuron_population(
            population_name=DRNPopulation.SEROTONIN,
            n_neurons=self.serotonin_neurons_size,
            polarity=PopulationPolarity.ANY,
            config=SerotoninNeuronConfig(
                serotonin_drive_gain=config.tonic_drive_gain * 20.0,
                tau_mem_ms=heterogeneous_tau_mem(15.0, self.serotonin_neurons_size, device, cv=0.20),
                v_reset=heterogeneous_v_reset(-0.1, self.serotonin_neurons_size, device),
                v_threshold=heterogeneous_v_threshold(1.0, self.serotonin_neurons_size, device, cv=0.12, clamp_fraction=0.25),
                g_L=heterogeneous_g_L(0.095, self.serotonin_neurons_size, device),
                noise_std=heterogeneous_noise_std(0.075, self.serotonin_neurons_size, device),
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.serotonin_neurons_size, device, cv=0.25),
            ),
        )

        # ── Local GABAergic interneurons (homeostatic inhibition) ────────────
        # These correspond to the ~15% GABA cells in DRN that regulate
        # 5-HT neuron excitability and mediate LHb-driven pauses.
        self._init_gaba_interneurons(DRNPopulation.GABA, self.gaba_size, device)

        # ── Drive normalisation state ────────────────────────────────────────
        self._drive_norm = AdaptiveNormalization(center=True, warmup=False)

        self.to(device)

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Compute serotonin output from tonic drive and LHb punishment input."""
        config = self.config

        # ── 1. Extract synaptic inputs ───────────────────────────────────────
        lhb_spikes: Optional[torch.Tensor] = None
        pfc_spikes: Optional[torch.Tensor] = None
        for sid, spikes in synaptic_inputs.items():
            if (
                sid.source_region == "lateral_habenula"
                and sid.source_population == LHbPopulation.PRINCIPAL
            ):
                lhb_spikes = spikes
            elif sid.source_region == "prefrontal_cortex":
                pfc_spikes = spikes

        lhb_rate: float = 0.0
        if lhb_spikes is not None:
            lhb_rate = float(lhb_spikes.mean().item())

        pfc_rate: float = 0.0
        if pfc_spikes is not None:
            pfc_rate = float(pfc_spikes.mean().item())

        # ── 2. Compute serotonin drive ─────────────────────────────────────
        # Positive baseline drive (encodes tonic state)
        # + PFC top-down excitation (executive control → patience)
        # − LHb punishment signal (punishment → 5-HT pause)
        # Biology: Celada et al. 2001; Hajós et al. 2007.
        raw_drive = (config.tonic_drive_gain
                     + pfc_rate * config.pfc_excitation_gain
                     - lhb_rate * config.lhb_inhibition_gain)

        raw_drive = self._drive_norm(raw_drive)

        # Clip drive to sensible range for I_h modulation
        serotonin_drive = float(torch.tensor(raw_drive).clamp(-1.0, 1.0).item())

        # ── 3. Update 5-HT neurons ────────────────────────────────────────────
        # Apply GABA feedback from the previous timestep's interneuron activity.
        # This closes the homeostatic loop: primary fires → GABA fires → primary
        # is inhibited next step (one-step causal delay, no circular dependency).
        gaba_feedback = self._gaba_feedback(self.serotonin_neurons_size, scale=0.01)
        serotonin_spikes, _ = self.serotonin_neurons.forward(
            g_ampa_input=None,    # No direct AMPA; drive via I_h modulation
            g_nmda_input=None,
            g_gaba_a_input=gaba_feedback,
            g_gaba_b_input=None,
            serotonin_drive=serotonin_drive,
        )

        # ── 4. Update GABA interneurons (homeostatic inhibition) ─────────────
        serotonin_activity = serotonin_spikes.float().mean().item()
        gaba_spikes = self._step_gaba_interneurons(
            serotonin_activity,
            drive_scale=2.0,
            nmda_ratio=0.3,
            self_inhibition_scale=0.5,
        )

        region_outputs: RegionOutput = {
            DRNPopulation.SEROTONIN: serotonin_spikes,
            DRNPopulation.GABA: gaba_spikes,
        }

        return region_outputs
