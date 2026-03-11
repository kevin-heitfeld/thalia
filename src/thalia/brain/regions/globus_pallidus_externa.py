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
from thalia.brain.configs import TonicPacemakerConfig, get_default_gpe_config
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
    version="1.0",
    author="Thalia Project",
    config_class=get_default_gpe_config,
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
        self.prototypic_neurons = self._make_bg_neurons(
            self.prototypic_size, GPePopulation.PROTOTYPIC, noise_std=0.007
        )
        # Arkypallidal neurons: ~25% of GPe, project back to striatum.
        # adapt_increment=0.015, tau_adapt=100ms cap equilibrium below 20 Hz.
        # factor=0.85 reduces baseline drive (was 0.9 → 34 Hz despite adapt_increment=0.005).
        self.arkypallidal_neurons = self._make_bg_neurons(
            self.arkypallidal_size,
            GPePopulation.ARKYPALLIDAL,
            noise_std=0.005,
            adapt_increment=0.015,
            tau_adapt=100.0,
        )

        # Tonic drive: prototypic at full baseline; arkypallidal at 0.9× (just above threshold at rest)
        # With factor=0.9: g_E_ss≈0.055, V_inf≈1.06 → ~10-15 Hz intrinsic; D2 inhibition will
        # bring it down to 5-20 Hz target. Previous 0.5× was sub-threshold (V_inf≈0.70), relying on
        # STN excitation which is chronically depleted (eff≈0.04) at STN's 20 Hz rate with U=0.50 τd=800ms.
        # Previous 0.857 that caused 46.9 Hz was before STN was properly calibrated to 20 Hz.
        self.prototypic_baseline = self._make_tonic_baseline(self.prototypic_size)
        self.arkypallidal_baseline = self._make_tonic_baseline(self.arkypallidal_size, factor=0.85)

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(GPePopulation.ARKYPALLIDAL, self.arkypallidal_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(GPePopulation.PROTOTYPIC, self.prototypic_neurons, polarity=PopulationPolarity.INHIBITORY)

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
