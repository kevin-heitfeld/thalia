"""Globus Pallidus Externa (GPe) - Basal Ganglia Intermediate Relay.

The GPe is **not** an output nucleus.  It is an intermediate relay in the indirect
pathway, distinguished from GPi and SNr by three key properties:

1. **Connectivity role** — receives D2-MSN inhibition and STN excitation, then
   projects back to STN (closing the GPe-STN oscillatory loop) and forwards to
   SNr/GPi.  This bidirectional STN↔GPe loop is absent from GPi and SNr.

2. **Two distinct cell types** (Mallet et al. 2012, Nat Neurosci):
   - **PROTOTYPIC** (~75 %): rhythmic, project to STN / SNr / GPi.
     Electrically coupled via Cx36 gap junctions.
   - **ARKYPALLIDAL** (~25 %): tonically active, project back to striatum.
     Provide the ``cancel`` signal that globally suppresses MSN activity during
     action stopping (Schmidt et al. 2013).

3. **Gap junction coupling** between PROTOTYPIC neurons (Connelly et al. 2010).
   This coupling promotes synchrony within the GPe-STN loop and is the substrate
   for the pathological β-band oscillation seen in Parkinson's disease.

Implementation
--------------
:class:`GlobusPollidusExterna` extends :class:`BasalGangliaOutputNucleus` with:
* A sparse gap-junction weight matrix ``w_prototypic_gap`` between PROTOTYPIC neurons.
* A one-step membrane voltage buffer ``_prototypic_membrane_buffer`` that feeds the
  previous-timestep voltage into the gap-junction computation.
* An overridden :meth:`_step` that routes PROTOTYPIC neurons through the coupled
  path and ARKYPALLIDAL neurons through the standard uncoupled path.

The external connectivity (D2→GPe, GPe→STN, STN→GPe, GPe→SNr/GPi,
ARKY→striatum) is unchanged — it is wired by :mod:`thalia.brain.presets.basal_ganglia`.
"""

from __future__ import annotations

from typing import Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs.basal_ganglia import GPeConfig
from thalia.brain.gap_junctions import GapJunctionCoupling
from thalia.brain.neurons import (
    ConductanceLIF,
    split_excitatory_conductance,
)
from thalia.brain.synapses import WeightInitializer
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
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
    description=(
        "Globus pallidus externa — indirect pathway relay with electrically coupled "
        "PROTOTYPIC neurons forming the GPe-STN oscillatory loop"
    ),
)
class GlobusPollidusExterna(BasalGangliaOutputNucleus):
    """Globus Pallidus Externa with gap junction coupling on PROTOTYPIC neurons.

    Parameterised by :class:`~thalia.brain.configs.basal_ganglia.GPeConfig`.
    All shared basal-ganglia output nucleus logic (neuron creation, baseline drive,
    synaptic integration) is inherited from :class:`BasalGangliaOutputNucleus`.
    This class adds electrical coupling between PROTOTYPIC neurons only.
    """

    def __init__(
        self,
        config: GPeConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ) -> None:
        super().__init__(config, population_sizes, region_name, device=device)

        n_prototypic = population_sizes[GPePopulation.PROTOTYPIC]

        # ------------------------------------------------------------------
        # Gap junction weight matrix (PROTOTYPIC ↔ PROTOTYPIC)
        # ------------------------------------------------------------------
        gap_w = WeightInitializer.sparse_gaussian_no_autapses(
            n_input=n_prototypic,
            n_output=n_prototypic,
            connectivity=config.gap_junction_connectivity,
            mean=config.gap_junctions.coupling_strength,
            std=config.gap_junctions.coupling_strength * 0.3,
            device=device,
        )
        self.prototypic_gap_junctions = GapJunctionCoupling.from_coupling_matrix(gap_w)

        self._gap_junction_scale: float = config.gap_junction_scale

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

            if pop_name == GPePopulation.PROTOTYPIC:
                spikes = self._prototypic_step_with_gap_junctions(
                    synaptic_inputs, n, neurons, baseline, pop_config.nmda_ratio,
                )
            else:
                spikes = self._bg_step_single(
                    synaptic_inputs, n, pop_name, neurons,
                    baseline=baseline, nmda_ratio=pop_config.nmda_ratio,
                )

            result[pop_name] = spikes
        return result

    # ------------------------------------------------------------------
    # PROTOTYPIC step with electrical coupling
    # ------------------------------------------------------------------

    def _prototypic_step_with_gap_junctions(
        self,
        synaptic_inputs: SynapticInput,
        n_neurons: int,
        neurons: ConductanceLIF,
        baseline: torch.Tensor,
        nmda_ratio: float,
    ) -> torch.Tensor:
        """One timestep for PROTOTYPIC neurons with gap-junction coupling.

        Gap junctions are computed as proper (g_gap, E_gap) for the conductance-based
        neuron model: I_gap = g_gap_total * (E_gap - V), where E_gap is the
        conductance-weighted average of neighbor voltages.
        """
        dendrite = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=n_neurons,
            filter_by_target_population=GPePopulation.PROTOTYPIC,
        )

        # Gap junction coupling → proper (g_gap, E_gap) for conductance-based neuron
        g_gap_total, E_gap = self.prototypic_gap_junctions.forward(neurons.V_soma)
        g_gap_scaled = g_gap_total * self._gap_junction_scale

        g_exc = baseline + dendrite.g_ampa
        g_inh = dendrite.g_gaba_a

        g_ampa, g_nmda = split_excitatory_conductance(g_exc, nmda_ratio=nmda_ratio)
        spikes, _ = neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa),
            g_nmda_input=ConductanceTensor(g_nmda),
            g_gaba_a_input=ConductanceTensor(g_inh),
            g_gaba_b_input=None,
            g_gap_input=ConductanceTensor(g_gap_scaled),
            E_gap_reversal=E_gap,
        )

        return spikes
