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

from typing import TYPE_CHECKING, Optional, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import TonicPacemakerConfig
from thalia.brain.neurons import (
    ConductanceLIFConfig,
    heterogeneous_dendrite_coupling,
    heterogeneous_noise_std,
    heterogeneous_tau_mem,
    heterogeneous_v_threshold,
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
from .population_names import RMTgPopulation
from .region_registry import register_region

if TYPE_CHECKING:
    from thalia.brain.neurons import ConductanceLIF


@register_region(
    "rostromedial_tegmentum",
    aliases=["rmtg", "tvta"],
    description="Rostromedial tegmental nucleus - dopamine pause mediator",
)
class RostromedialTegmentum(NeuralRegion[TonicPacemakerConfig]):
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

    def __init__(
        self,
        config: TonicPacemakerConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        super().__init__(config, population_sizes, region_name, device=device)

        self.gaba_size = population_sizes[RMTgPopulation.GABA]

        # GABAergic neurons that inhibit VTA DA neurons
        self.gaba_neurons: ConductanceLIF
        self.gaba_neurons = self._create_and_register_neuron_population(
            population_name=RMTgPopulation.GABA,
            n_neurons=self.gaba_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(self.config.tau_mem_ms, self.gaba_size, device),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(self.config.v_threshold, self.gaba_size, device),
                tau_ref=self.config.tau_ref,
                g_L=heterogeneous_g_L(0.10, self.gaba_size, device, cv=0.08),
                E_E=3.0,
                E_I=-0.5,
                tau_E=4.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.09, self.gaba_size, device),
                noise_tau_ms=3.0,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.gaba_size, device, cv=0.20),
            ),
        )

        # Moderate baseline drive
        self.baseline_drive = torch.full((self.gaba_size,), config.baseline_drive, device=device)

        # Ensure all tensors are on the correct device
        self.to(device)

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Update RMTg from LHb excitation; output inhibits VTA DA neurons."""
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

        return region_outputs
