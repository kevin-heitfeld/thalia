"""Shared base class for basal ganglia output nuclei (GPi, SNr).

GPi and SNr share the same computational pattern: tonically firing GABAergic
populations whose output is gated by striatal inhibition and STN excitation.
The only differences are per-population biophysical parameters (noise level,
adaptation, baseline drive, NMDA ratio).

A single class :class:`BasalGangliaOutputNucleus` is registered under two names
(``globus_pallidus_interna``, ``substantia_nigra``) and parameterised via
:class:`~thalia.brain.configs.basal_ganglia.BGOutputConfig`.

.. note::
    GPe (globus pallidus externa) is **not** registered here.  It uses the
    dedicated :class:`~thalia.brain.regions.globus_pallidus_externa.GlobusPollidusExterna`
    subclass which adds electrical gap junction coupling between PROTOTYPIC neurons.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs.basal_ganglia import BGOutputConfig, BGPopulationConfig
from thalia.brain.neurons import (
    ConductanceLIF,
    build_conductance_lif_config,
    split_excitatory_conductance,
)
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)

from .neural_region import NeuralRegion
from .region_registry import register_region


class BasalGangliaOutputNucleus(NeuralRegion[BGOutputConfig]):
    """Basal ganglia output nucleus (GPi, SNr).

    Registered under two names (``globus_pallidus_interna``, ``substantia_nigra``);
    parameterised entirely via :class:`~thalia.brain.configs.basal_ganglia.BGOutputConfig`
    which carries :class:`~thalia.brain.configs.basal_ganglia.BGPopulationConfig` entries
    for each population.

    GPe uses :class:`~thalia.brain.regions.globus_pallidus_externa.GlobusPollidusExterna`
    instead, which adds gap junction coupling on PROTOTYPIC neurons.
    """

    def __init__(
        self,
        config: BGOutputConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ) -> None:
        super().__init__(config, population_sizes, region_name, device=device)

        overrides = config.population_overrides
        assert overrides, (
            f"BGOutputConfig.population_overrides must be non-empty for region '{region_name}'"
        )
        self._population_overrides: dict[str, BGPopulationConfig] = dict(overrides)

        for pop_name, pop_config in self._population_overrides.items():
            n = population_sizes[pop_name]

            self._create_and_register_neuron_population(
                population_name=pop_name,
                n_neurons=n,
                polarity=pop_config.polarity,
                config=build_conductance_lif_config(
                    pop_config, n, device,
                    tau_ref=config.tau_ref, g_L=0.10, g_L_cv=0.08,
                ),
            )

            # Register tonic baseline drive as a named buffer so .to(device) tracks it.
            baseline = torch.full(
                (n,), config.baseline_drive * pop_config.baseline_multiplier, device=device,
            )
            self.register_buffer(f"_baseline_{pop_name}", baseline)

        self.to(device)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        result: RegionOutput = {}
        for pop_name, pop_config in self._population_overrides.items():
            n = self.get_population_size(pop_name)
            neurons = self.neuron_populations[pop_name]
            assert isinstance(neurons, ConductanceLIF)
            baseline: torch.Tensor = getattr(self, f"_baseline_{pop_name}")

            result[pop_name] = self._bg_step_single(
                synaptic_inputs, n, pop_name, neurons,
                baseline=baseline, nmda_ratio=pop_config.nmda_ratio,
            )
        return result

    # ------------------------------------------------------------------
    # Shared single-population step
    # ------------------------------------------------------------------

    def _bg_step_single(
        self,
        synaptic_inputs: SynapticInput,
        n_neurons: int,
        population_name: str,
        neurons: ConductanceLIF,
        *,
        baseline: torch.Tensor,
        nmda_ratio: float,
    ) -> torch.Tensor:
        """Run one timestep for a single BG output population.

        Pattern:
            1. Integrate synaptic inputs at dendrites (filtered to this population).
            2. Add tonic baseline to g_AMPA.
            3. Split combined excitatory conductance into AMPA + NMDA components.
            4. Forward through the ConductanceLIF neuron model.
        """
        dendrite = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=n_neurons,
            filter_by_target_population=population_name,
        )
        g_exc = baseline + dendrite.g_ampa
        g_inh = dendrite.g_gaba_a

        g_ampa, g_nmda = split_excitatory_conductance(g_exc, nmda_ratio=nmda_ratio)
        spikes, _ = neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa),
            g_nmda_input=ConductanceTensor(g_nmda),
            g_gaba_a_input=ConductanceTensor(g_inh),
            g_gaba_b_input=None,
        )

        return spikes


# ---------------------------------------------------------------------------
# Register under the two canonical BG output nucleus names.
# GPe is registered separately in globus_pallidus_externa.py.
# ---------------------------------------------------------------------------

register_region(
    "globus_pallidus_interna",
    aliases=["gpi", "entopeduncular"],
    description="Globus pallidus interna - basal ganglia output nucleus for motor/cognitive loops",
)(BasalGangliaOutputNucleus)

register_region(
    "substantia_nigra",
    aliases=["snr", "substantia_nigra_reticulata"],
    description="Substantia nigra pars reticulata - basal ganglia output nucleus",
)(BasalGangliaOutputNucleus)
