"""Globus Pallidus Externa (GPe) - Basal Ganglia Indirect Pathway Hub.

The GPe is a key GABAergic nucleus in the basal ganglia that links the indirect
pathway (D2 MSNs) to the subthalamic nucleus (STN) and substantia nigra reticulata
(SNr). It contains two functionally distinct cell types:

- **Prototypic neurons** (~75%): GABAergic, ~50 Hz tonic firing, project to STN and SNr.
  Receive inhibition from D2 MSNs. Form the GPe-STN oscillatory loop.
- **Arkypallidal neurons** (~25%): GABAergic, project back to all striatal
  compartments providing global suppression / action gating.

Biological Background:
======================
**Anatomy:**
- Location: Lateral segment of the globus pallidus (in primates)
- ~50,000 neurons in rodents (~75% prototypic, ~25% arkypallidal)
- Tonic baseline firing ~50 Hz (among highest in basal ganglia)
- Dense gap-junction coupling → coordinated population activity

**Indirect Pathway Operation:**
1. D2-MSNs fire → inhibit GPe PROTOTYPIC
2. Disinhibited STN → glutamatergic burst to SNr (hyperdirect effect propagated)
3. SNr fires more → thalamus suppressed → no action

**Arkypallidal Function:**
- Provide widespread inhibitory feedback to striatum
- Proposed role: Cancel ongoing action representations globally
- Active during action switching and behavioral transitions

**Inputs:**
- Striatum D2-MSNs: Inhibitory (indirect pathway)
- STN: Excitatory (glutamatergic, GPe-STN feedback loop)

**Outputs:**
- STN: Inhibitory (slows STN, GPe-STN loop dynamics)
- SNr: Inhibitory (pallido-nigral pathway, can release SNr gate)
- Striatum (arkypallidal): Inhibitory, global MSN suppression
"""

from __future__ import annotations

import torch

from thalia.brain.configs import GlobusPallidusExternaConfig
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
from .population_names import GPePopulation
from .region_registry import register_region


@register_region(
    "globus_pallidus_externa",
    aliases=["gpe"],
    description="Globus pallidus externa - basal ganglia indirect pathway hub",
    version="1.0",
    author="Thalia Project",
    config_class=GlobusPallidusExternaConfig,
)
class GlobusPallidusExterna(NeuralRegion[GlobusPallidusExternaConfig]):
    """Globus Pallidus Externa - Indirect Pathway Hub.

    Contains prototypic neurons (→STN/SNr) and arkypallidal neurons (→striatum).
    Tonically active GABAergic population that gates basal ganglia output
    through the indirect pathway.

    Input Populations:
    ------------------
    - striatum D2: Inhibitory (indirect pathway activation suppresses GPe)
    - STN: Excitatory (feedback from subthalamic nucleus)

    Output Populations:
    -------------------
    - "prototypic": Projects to STN (inhibitory) and SNr (inhibitory)
    - "arkypallidal": Projects back to striatum (inhibitory, global suppression)
    """

    def __init__(self, config: GlobusPallidusExternaConfig, population_sizes: PopulationSizes, region_name: RegionName):
        super().__init__(config, population_sizes, region_name)

        self.arkypallidal_size = population_sizes[GPePopulation.ARKYPALLIDAL]
        self.prototypic_size = population_sizes[GPePopulation.PROTOTYPIC]

        # Prototypic neurons: ~75% of GPe, ~50 Hz tonic, project to STN + SNr
        prototypic_config = ConductanceLIFConfig(
            region_name=self.region_name,
            population_name=GPePopulation.PROTOTYPIC,
            device=self.device,
            tau_mem=self.config.tau_mem,
            v_threshold=self.config.v_threshold,
            v_reset=0.0,
            v_rest=0.0,
            tau_ref=self.config.tau_ref,
            g_L=0.10,
            E_L=0.0,
            E_E=3.0,
            E_I=-0.5,
            tau_E=5.0,
            tau_I=10.0,
            noise_std=0.007 if self.config.baseline_noise_conductance_enabled else 0.0,
        )
        self.prototypic_neurons = ConductanceLIF(
            n_neurons=self.prototypic_size,
            config=prototypic_config,
        )

        # Arkypallidal neurons: ~25% of GPe, project back to striatum
        arkypallidal_config = ConductanceLIFConfig(
            region_name=self.region_name,
            population_name=GPePopulation.ARKYPALLIDAL,
            device=self.device,
            tau_mem=self.config.tau_mem,
            v_threshold=self.config.v_threshold,
            v_reset=0.0,
            v_rest=0.0,
            tau_ref=self.config.tau_ref,
            g_L=0.10,
            E_L=0.0,
            E_E=3.0,
            E_I=-0.5,
            tau_E=5.0,
            tau_I=10.0,
            noise_std=0.005 if self.config.baseline_noise_conductance_enabled else 0.0,
        )
        self.arkypallidal_neurons = ConductanceLIF(
            n_neurons=self.arkypallidal_size,
            config=arkypallidal_config,
        )

        # Tonic drive for baseline firing (~50 Hz for prototypic)
        self.prototypic_baseline = torch.full(
            (self.prototypic_size,), config.baseline_drive, device=self.device
        )
        self.arkypallidal_baseline = torch.full(
            (self.arkypallidal_size,), config.baseline_drive * 0.857, device=self.device
        )

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(GPePopulation.ARKYPALLIDAL, self.arkypallidal_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(GPePopulation.PROTOTYPIC, self.prototypic_neurons, polarity=PopulationPolarity.INHIBITORY)

        # Ensure all tensors are on the correct device
        self.to(self.device)

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Update GPe neurons based on striatal D2 inhibition and STN excitation."""
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        # =====================================================================
        # PROTOTYPIC NEURONS
        # =====================================================================
        proto_dendrite = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.prototypic_size,
            filter_by_target_population=GPePopulation.PROTOTYPIC,
        )
        # D2-MSN inhibition reduces prototypic activity (indirect pathway)
        # STN excitation provides GPe-STN feedback loop drive
        g_exc_proto = self.prototypic_baseline.clone() + proto_dendrite.g_ampa
        g_inh_proto = proto_dendrite.g_gaba_a

        g_ampa_proto, g_nmda_proto = split_excitatory_conductance(g_exc_proto, nmda_ratio=0.05)
        prototypic_spikes, _ = self.prototypic_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa_proto),
            g_nmda_input=ConductanceTensor(g_nmda_proto),
            g_gaba_a_input=ConductanceTensor(g_inh_proto),
            g_gaba_b_input=None,
        )

        # =====================================================================
        # ARKYPALLIDAL NEURONS
        # =====================================================================
        arky_dendrite = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.arkypallidal_size,
            filter_by_target_population=GPePopulation.ARKYPALLIDAL,
        )
        # D2-MSN inhibition of arkypallidal as well
        g_exc_arky = self.arkypallidal_baseline.clone() + arky_dendrite.g_ampa
        g_inh_arky = arky_dendrite.g_gaba_a

        g_ampa_arky, g_nmda_arky = split_excitatory_conductance(g_exc_arky, nmda_ratio=0.05)
        arkypallidal_spikes, _ = self.arkypallidal_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa_arky),
            g_nmda_input=ConductanceTensor(g_nmda_arky),
            g_gaba_a_input=ConductanceTensor(g_inh_arky),
            g_gaba_b_input=None,
        )

        region_outputs: RegionOutput = {
            GPePopulation.ARKYPALLIDAL: arkypallidal_spikes,
            GPePopulation.PROTOTYPIC: prototypic_spikes,
        }

        return self._post_forward(region_outputs)

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes."""
        super().update_temporal_parameters(dt_ms)
        self.prototypic_neurons.update_temporal_parameters(dt_ms)
        self.arkypallidal_neurons.update_temporal_parameters(dt_ms)
