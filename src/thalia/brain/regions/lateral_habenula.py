"""Lateral Habenula (LHb) - Aversive Prediction Error Encoder.

The lateral habenula is a small but evolutionarily conserved epithalamic nucleus
that encodes negative reward prediction errors (negative RPE). It receives
information about expected versus actual outcomes and, when outcomes are worse
than expected (punishment, reward omission), activates the RMTg to pause
VTA dopamine neurons — implementing the "anti-reward" signal.

Biological Background:
======================
**Anatomy:**
- Location: Epithalamus, dorsal to the thalamus
- ~6,000-10,000 neurons in rodents (glutamatergic principal cells)
- Receives dense excitatory input from SNr and GPi (basal ganglia output)
- Projects primarily to RMTg (GABA) which inhibits VTA DA neurons

**Aversive Signaling Logic:**
- High SNr activity = suppressed basal ganglia output = bad outcome / no reward
  → SNr excites LHb → LHb excites RMTg → RMTg inhibits VTA DA
  → Dopamine pause = negative RPE
- Low SNr activity = disinhibited thalamus = reward / good outcome
  → Less LHb drive → VTA DA maintains tonic / bursts

This implements the "anti-reward" axis complementary to VTA's positive RPE:
    Positive RPE → VTA DA burst (direct cortical/subcortical input)
    Negative RPE → LHb→RMTg→VTA GABA pause

**Why not just directly inhibit VTA?**
The LHb→RMTg→VTA disynaptic pathway provides:
1. Gain control (LHb can modulate RMTg gain)
2. Temporal sharpening (RMTg fast GABA = abrupt pause)
3. Convergence with other aversive signals (amygdala→LHb)

**Inputs:**
- SNr VTA_FEEDBACK: Excitatory (high SNr → bad outcome → LHb excited)
- PFC (future): Expectation signals for predictive negative RPE
- Amygdala (future): Conditioned fear, punishment signals

**Outputs:**
- RMTg GABA: Excitatory (drives dopamine pause)

**Pathophysiology:**
- LHb hyperactivity linked to depression (Elevated negative RPE)
- DBS of LHb shows antidepressant effects
- Habenular lesions → impaired punishment learning
"""

from __future__ import annotations

import torch

from thalia.brain.configs import LateralHabenulaConfig
from thalia.components import ConductanceLIF, ConductanceLIFConfig
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
    PopulationPolarity,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)
from thalia.utils import split_excitatory_conductance

from .neural_region import NeuralRegion
from .population_names import LHbPopulation
from .region_registry import register_region


@register_region(
    "lateral_habenula",
    aliases=["lhb"],
    description="Lateral habenula - aversive prediction error encoder",
    version="1.0",
    author="Thalia Project",
    config_class=LateralHabenulaConfig,
)
class LateralHabenula(NeuralRegion[LateralHabenulaConfig]):
    """Lateral Habenula - Aversive Prediction Error Encoder.

    Glutamatergic principal neurons that encode negative outcomes via
    SNr-driven excitation, projecting to RMTg to pause VTA dopamine.

    Input Populations:
    ------------------
    - SNr VTA_FEEDBACK: Excitatory (high SNr activity = bad outcome)

    Output Populations:
    -------------------
    - "principal": Glutamatergic projection to RMTg GABA neurons
    """

    def __init__(self, config: LateralHabenulaConfig, population_sizes: PopulationSizes, region_name: RegionName):
        super().__init__(config, population_sizes, region_name)

        self.principal_size = population_sizes[LHbPopulation.PRINCIPAL]

        # Glutamatergic principal neurons (mostly silent at baseline)
        self.principal_neurons = ConductanceLIF(
            n_neurons=self.principal_size,
            config=ConductanceLIFConfig(
                region_name=self.region_name,
                population_name=LHbPopulation.PRINCIPAL,
                tau_mem=self.config.tau_mem,
                v_threshold=self.config.v_threshold,
                v_reset=0.0,
                tau_ref=self.config.tau_ref,
                g_L=0.08,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                noise_std=0.003,
            ),
            device=self.device,
        )

        # Low baseline drive (LHb is mostly quiet except during aversive events)
        self.baseline_drive = torch.full(
            (self.principal_size,), config.baseline_drive, device=self.device
        )

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(LHbPopulation.PRINCIPAL, self.principal_neurons, polarity=PopulationPolarity.EXCITATORY)

        # Ensure all tensors are on the correct device
        self.to(self.device)

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Update LHb neurons from SNr input (high SNr = bad outcome = LHb excited)."""
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        # =====================================================================
        # Integrate synaptic inputs at dendrites (all sources → PRINCIPAL)
        # High SNr activity → excites LHb (aversive outcome signal)
        # =====================================================================
        dendrite = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.principal_size,
            filter_by_target_population=LHbPopulation.PRINCIPAL,
        )

        g_exc = self.baseline_drive.clone() + dendrite.g_ampa
        g_gaba_a = dendrite.g_gaba_a

        # LHb is a fast prediction-error encoder: AMPA-only, no NMDA.
        g_ampa, g_nmda = split_excitatory_conductance(g_exc, nmda_ratio=0.0)

        principal_spikes, _ = self.principal_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa),
            g_nmda_input=ConductanceTensor(g_nmda),
            g_gaba_a_input=ConductanceTensor(g_gaba_a),
            g_gaba_b_input=None,
        )

        region_outputs: RegionOutput = {
            LHbPopulation.PRINCIPAL: principal_spikes,
        }

        return self._post_forward(region_outputs)

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes."""
        super().update_temporal_parameters(dt_ms)
        self.principal_neurons.update_temporal_parameters(dt_ms)
