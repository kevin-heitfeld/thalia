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

from typing import TYPE_CHECKING, Optional, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import TonicPacemakerConfig
from thalia.brain.neurons import (
    ConductanceLIFConfig,
    heterogeneous_dendrite_coupling,
    heterogeneous_noise_std,
    heterogeneous_tau_adapt,
    heterogeneous_tau_mem,
    heterogeneous_v_reset,
    heterogeneous_v_threshold,
    heterogeneous_adapt_increment,
    heterogeneous_g_L,
    split_excitatory_conductance,
)
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
    PopulationPolarity,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)

from .neural_region import NeuralRegion
from .population_names import LHbPopulation
from .region_registry import register_region

if TYPE_CHECKING:
    from thalia.brain.neurons import ConductanceLIF


@register_region(
    "lateral_habenula",
    aliases=["lhb"],
    description="Lateral habenula - aversive prediction error encoder",
)
class LateralHabenula(NeuralRegion[TonicPacemakerConfig]):
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

    def __init__(
        self,
        config: TonicPacemakerConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        super().__init__(config, population_sizes, region_name, device=device)

        self.principal_size = population_sizes[LHbPopulation.PRINCIPAL]

        # Glutamatergic principal neurons (mostly silent at baseline)
        self.principal_neurons: ConductanceLIF
        self.principal_neurons = self._create_and_register_neuron_population(
            population_name=LHbPopulation.PRINCIPAL,
            n_neurons=self.principal_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(self.config.tau_mem_ms, self.principal_size, device),
                v_reset=heterogeneous_v_reset(-0.05, self.principal_size, device),
                v_threshold=heterogeneous_v_threshold(self.config.v_threshold, self.principal_size, device),
                tau_ref=self.config.tau_ref,
                g_L=heterogeneous_g_L(0.08, self.principal_size, device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.080, self.principal_size, device),
                noise_tau_ms=3.0,
                tau_adapt_ms=heterogeneous_tau_adapt(400.0, self.principal_size, device),
                adapt_increment=heterogeneous_adapt_increment(0.05, self.principal_size, device),  # Raised 0.01→0.05: SFA was 0.99 (no adaptation) at 24 Hz
                E_adapt=-0.5,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.principal_size, device, cv=0.25),
            ),
        )

        # Low baseline drive (LHb is mostly quiet except during aversive events)
        self.baseline_drive = torch.full((self.principal_size,), config.baseline_drive, device=device)

        # Ensure all tensors are on the correct device
        self.to(device)

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Update LHb neurons from SNr input (high SNr = bad outcome = LHb excited)."""
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

        self._apply_all_population_homeostasis(region_outputs)

        return region_outputs
