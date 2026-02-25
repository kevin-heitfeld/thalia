"""Rostromedial Tegmental Nucleus (RMTg) - Dopamine Pause Mediator.

The RMTg (also called the GABAergic tail of the VTA, or tVTA) is a GABAergic
nucleus that mediates the dopamine pause response to aversive outcomes and
negative reward prediction errors. It serves as the key relay between the
lateral habenula's aversive signal and VTA dopamine neuron inhibition.

Biological Background:
======================
**Anatomy:**
- Location: Posterior to VTA, rostral midbrain
- Dense GABAergic neurons (~3,000-5,000 in rodents)
- Receives the heaviest known projection from lateral habenula
- Projects densely and selectively to VTA DA neurons (not GABA neurons)
- Sometimes called "tVTA" (tail of VTA) due to proximity

**Functional Significance:**
The RMTg is the critical relay for negative RPE signaling:
    Aversive event → SNr↑ → LHb↑ → RMTg↑ → VTA DA pause

Without RMTg, LHb cannot efficiently pause DA neurons because:
- LHb neurons are glutamatergic (they excite, not inhibit VTA)
- RMTg converts this glutamatergic signal into potent GABA inhibition
- RMTg→VTA projection is fast, targeted, and strong

**Timing Properties:**
- RMTg neurons fire 10-30ms before the dopamine pause
- Fast GABA kinetics (GABA_A) → abrupt, precise dopamine pause
- Mirrors the precision of DA bursts (opposite sign, similar timing)

**Inputs:**
- LHb PRINCIPAL: Excitatory (main driver of RMTg activity)
- SNr (direct minor projection, less dominant than LHb→RMTg)

**Outputs:**
- VTA DA neurons: Inhibitory (GABA_A-mediated pause)
- SNc DA neurons (future): Pauses nigrostriatal DA

**Key for Reinforcement Learning:**
Without RMTg, the network cannot properly encode:
- Reward omission (expected reward didn't arrive)
- Punishment (aversive outcome)
- Negative prediction errors (outcome worse than expected)

These are essential for learning to avoid bad outcomes.
"""

from __future__ import annotations

import torch

from thalia.brain.configs import RostromedialTegmentumConfig
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
from .population_names import RMTgPopulation
from .region_registry import register_region


@register_region(
    "rostromedial_tegmentum",
    aliases=["rmtg", "tvta"],
    description="Rostromedial tegmental nucleus - dopamine pause mediator",
    version="1.0",
    author="Thalia Project",
    config_class=RostromedialTegmentumConfig,
)
class RostromedialTegmentum(NeuralRegion[RostromedialTegmentumConfig]):
    """Rostromedial Tegmental Nucleus - Dopamine Pause Mediator.

    GABAergic neurons that receive LHb excitation and project inhibitory
    output to VTA dopamine neurons, implementing the dopamine pause for
    negative reward prediction errors.

    Input Populations:
    ------------------
    - LHb PRINCIPAL: Excitatory (aversive signal drives RMTg)

    Output Populations:
    -------------------
    - "gaba": GABAergic projection to VTA DA neurons (drives pause)
    """

    def __init__(self, config: RostromedialTegmentumConfig, population_sizes: PopulationSizes, region_name: RegionName):
        super().__init__(config, population_sizes, region_name)

        self.gaba_size = population_sizes[RMTgPopulation.GABA]

        # GABAergic neurons that inhibit VTA DA neurons
        self.gaba_neurons = ConductanceLIF(
            n_neurons=self.gaba_size,
            config=ConductanceLIFConfig(
                region_name=self.region_name,
                population_name=RMTgPopulation.GABA,
                tau_mem=self.config.tau_mem,
                v_threshold=self.config.v_threshold,
                v_reset=0.0,
                tau_ref=self.config.tau_ref,
                g_L=0.10,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=4.0,   # Faster AMPA kinetics for precise timing
                tau_I=10.0,
                noise_std=0.005,
            ),
            device=self.device,
        )

        # Moderate baseline drive
        self.baseline_drive = torch.full(
            (self.gaba_size,), config.baseline_drive, device=self.device
        )

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(RMTgPopulation.GABA, self.gaba_neurons, polarity=PopulationPolarity.INHIBITORY)

        # Ensure all tensors are on the correct device
        self.to(self.device)

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Update RMTg from LHb excitation; output inhibits VTA DA neurons."""
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        # =====================================================================
        # Integrate synaptic inputs at dendrites (all sources → GABA population)
        # LHb excitation drives RMTg to fire → inhibits VTA DA (pause)
        # =====================================================================
        dendrite = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.gaba_size,
            filter_by_target_population=RMTgPopulation.GABA,
        )

        g_exc = self.baseline_drive.clone() + dendrite.g_ampa
        g_gaba_a = dendrite.g_gaba_a

        # Fast AMPA-dominant excitation (RMTg needs precise fast responses)
        g_ampa, g_nmda = split_excitatory_conductance(g_exc, nmda_ratio=0.05)

        gaba_spikes, _ = self.gaba_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa),
            g_nmda_input=ConductanceTensor(g_nmda),
            g_gaba_a_input=ConductanceTensor(g_gaba_a),
            g_gaba_b_input=None,
        )

        region_outputs: RegionOutput = {
            RMTgPopulation.GABA: gaba_spikes,
        }

        return self._post_forward(region_outputs)

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes."""
        super().update_temporal_parameters(dt_ms)
        self.gaba_neurons.update_temporal_parameters(dt_ms)
