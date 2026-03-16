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

from typing import Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import TonicPacemakerConfig
from thalia.brain.neurons import (
    ConductanceLIFConfig,
    heterogeneous_adapt_increment,
    heterogeneous_g_L,
    heterogeneous_tau_mem,
    heterogeneous_v_threshold,
)
from thalia.typing import (
    NeuromodulatorInput,
    PopulationPolarity,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)

from .basal_ganglia_output_nucleus import BasalGangliaOutputNucleus
from .population_names import GPePopulation
from .region_registry import register_region


@register_region(
    "globus_pallidus_externa",
    aliases=["gpe"],
    description="Globus pallidus externa - basal ganglia indirect pathway hub",
)
class GlobusPallidusExterna(BasalGangliaOutputNucleus[TonicPacemakerConfig]):
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

    def __init__(
        self,
        config: TonicPacemakerConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        super().__init__(config, population_sizes, region_name, device=device)

        self.arkypallidal_size = population_sizes[GPePopulation.ARKYPALLIDAL]
        self.prototypic_size = population_sizes[GPePopulation.PROTOTYPIC]

        # Prototypic neurons: ~75% of GPe, ~50 Hz tonic, project to STN + SNr
        self.prototypic_neurons = self._create_and_register_neuron_population(
            population_name=GPePopulation.PROTOTYPIC,
            n_neurons=self.prototypic_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(self.config.tau_mem_ms, self.prototypic_size, self.device),
                v_threshold=heterogeneous_v_threshold(self.config.v_threshold, self.prototypic_size, self.device),
                v_reset=0.0,
                tau_ref=self.config.tau_ref,
                g_L=heterogeneous_g_L(0.10, self.prototypic_size, self.device, cv=0.08),
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                E_adapt=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                noise_std=0.007,
                adapt_increment=0.0,
                tau_adapt=100.0,
            ),
        )

        # Arkypallidal neurons: ~25% of GPe, project back to striatum.
        self.arkypallidal_neurons = self._create_and_register_neuron_population(
            population_name=GPePopulation.ARKYPALLIDAL,
            n_neurons=self.arkypallidal_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(self.config.tau_mem_ms, self.arkypallidal_size, self.device),
                v_threshold=heterogeneous_v_threshold(self.config.v_threshold, self.arkypallidal_size, self.device),
                v_reset=0.0,
                tau_ref=self.config.tau_ref,
                g_L=heterogeneous_g_L(0.10, self.arkypallidal_size, self.device, cv=0.08),
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                E_adapt=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                noise_std=0.005,
                adapt_increment=heterogeneous_adapt_increment(0.10, self.arkypallidal_size, self.device),
                tau_adapt=100.0,
            ),
        )

        # Tonic drive: prototypic at full baseline; arkypallidal at reduced drive.
        # Factor 0.95: arkypallidal fire at 5-20 Hz (Abdi et al. 2015), much lower
        # than prototypic 30-80 Hz. Increased from 0.80 after arkypallidals dropped
        # to 3.9 Hz (too low). Factor 0.55 overcorrects catastrophically to 0.12 Hz.
        self.prototypic_baseline = self._make_tonic_baseline(self.prototypic_size)
        self.arkypallidal_baseline = self._make_tonic_baseline(self.arkypallidal_size, factor=0.95)

        # Ensure all tensors are on the correct device
        self.to(device)

    def _step(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Update GPe neurons based on striatal D2 inhibition and STN excitation."""
        prototypic_spikes = self._bg_step_single(
            synaptic_inputs, self.prototypic_size, GPePopulation.PROTOTYPIC,
            self.prototypic_neurons, self.prototypic_baseline,
        )
        arkypallidal_spikes = self._bg_step_single(
            synaptic_inputs, self.arkypallidal_size, GPePopulation.ARKYPALLIDAL,
            self.arkypallidal_neurons, self.arkypallidal_baseline,
        )
        return {
            GPePopulation.ARKYPALLIDAL: arkypallidal_spikes,
            GPePopulation.PROTOTYPIC: prototypic_spikes,
        }
