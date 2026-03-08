"""Substantia Nigra pars Compacta (SNc) - Nigrostriatal Dopamine Pacemaker.

The SNc contains tonically-active dopaminergic neurons that broadcast dopamine
to the dorsal striatum (caudate nucleus + putamen) via the nigrostriatal pathway,
enabling motor skill learning and habitual action sequences.

Biological Background:
======================
**Anatomy:**
- Location: Midbrain (dorsal to VTA)
- ~400,000-600,000 DA neurons in humans (10× more than VTA)
- Dense projection to dorsal striatum (motor/associative loop)
- Separated from VTA mesolimbic/mesocortical projections

**Functional Distinction from VTA:**
- VTA → ventral striatum (NAc): reward motivation, goal-directed learning
- SNc → dorsal striatum (caudate/putamen): motor control, habit learning
- SNc degeneration causes Parkinson's disease (tremor, akinesia, rigidity)

**Dopamine Firing Patterns:**
1. **Tonic pacemaking** (4-6 Hz baseline):
   - Intrinsic Ca²⁺ pacemaker currents (L-type, HCN)
   - Provides dopaminergic tone for movement facilitation
2. **Phasic activity** modulation by sensorimotor context (not RPE)

**Nigrostriatal Pathway:**
- SNc DA → dorsal striatum D1 MSNs: direct pathway facilitation (Go)
- SNc DA → dorsal striatum D2 MSNs: indirect pathway suppression (NoGo)
- Dopamine-gated plasticity for motor sequence learning (Hikosaka et al. 2002)

References:
    Hikosaka, O., Nakamura, K., Sakai, K., & Nakahashi, H. (2002).
        Central mechanisms of motor skill learning. Current Opinion in
        Neurobiology, 12(2), 217-222.
    Björklund, A., & Dunnett, S. B. (2007). Dopamine neuron systems in
        the brain: an update. Trends in Neurosciences, 30(5), 194-202.
"""

from __future__ import annotations

from typing import ClassVar, Dict, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import SubstantiaNigraCompactaConfig
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
    NeuromodulatorChannel,
    PopulationName,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)

from .dopamine_pacemaker_base import DopaminePacemakerBase
from .population_names import SNcPopulation
from .region_registry import register_region


@register_region(
    "substantia_nigra_compacta",
    aliases=["snc", "nigra_compacta"],
    description="SNc - nigrostriatal dopamine pacemaker for motor learning",
    version="1.0",
    author="Thalia Project",
    config_class=SubstantiaNigraCompactaConfig,
)
class SubstantiaNigraCompacta(DopaminePacemakerBase[SubstantiaNigraCompactaConfig]):
    """Substantia Nigra pars Compacta — Nigrostriatal Dopamine System.

    Provides tonic dopaminergic drive to dorsal striatum for motor learning.
    Unlike the VTA, the SNc does NOT compute reward prediction errors; it
    maintains a steady tonic signal modulated by sensorimotor context and
    receives short-loop feedback from dorsal striatal D1/D2 neurons.

    Input Populations:
    ------------------
    - ``SNcPopulation.GABA`` (inhibitory): Striatal D1/D2 feedback, GPe

    Output Populations:
    -------------------
    - ``SNcPopulation.DA``: Dopamine neuron spikes
    - ``SNcPopulation.GABA``: Local GABAergic interneuron spikes

    Neuromodulator Output:
    ----------------------
    - ``da_nigrostriatal``: Routed to dorsal striatum by Brain
    """

    # Declarative neuromodulator channel.  Brain reads this ClassVar
    # at brain-build time to wire NeuromodulatorTract diffusion filters.
    neuromodulator_outputs: ClassVar[Dict[NeuromodulatorChannel, PopulationName]] = {
        NeuromodulatorChannel.DA_NIGROSTRIATAL: SNcPopulation.DA,
    }

    def __init__(
        self,
        config: SubstantiaNigraCompactaConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        super().__init__(config, population_sizes, region_name, device=device)

        self.da_size = population_sizes[SNcPopulation.DA]
        self.gaba_size = population_sizes[SNcPopulation.GABA]

        # DA neurons — ConductanceLIF with spike-frequency adaptation and I_h pacemaker
        # I_h (HCN) contributes to 4-6 Hz tonic rhythm; activates during RMTg-driven pauses
        # and provides a rebound ramp that restores tonic firing (Neuhoff et al. 2002).
        self.da_neurons = self._make_da_neurons(self.da_size, SNcPopulation.DA)

        # GABAergic interneurons (local inhibitory control)
        self._init_gaba_interneurons(SNcPopulation.GABA, self.gaba_size)

        self._register_neuron_population(SNcPopulation.DA, self.da_neurons)

        # Ensure all tensors are on the correct device
        self.to(device)

    def _step(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Advance SNc one timestep.

        Args:
            synaptic_inputs: Inhibitory feedback from striatum D1/D2 and GPe.
            neuromodulator_inputs: Unused (SNc is a neuromodulator source).

        Returns:
            ``RegionOutput`` with ``SNcPopulation.DA`` and ``SNcPopulation.GABA``
            spike tensors.
        """
        # =====================================================================
        # STRIATAL SHORT-LOOP FEEDBACK (D1/D2 → SNc)
        # =====================================================================
        # Tonic 4-6 Hz pacemaking is achieved via baseline_drive + spike-frequency
        # adaptation (see SubstantiaNigraCompactaConfig.baseline_drive):
        #   1. baseline_drive sets V_inf above threshold
        #   2. Each spike increments g_adapt, suppressing re-firing
        #   3. g_adapt decays (tau=300ms) → V_inf rises back → next spike
        # Striatal GABA_A feedback provides transient pauses on top of tonic firing.
        striatal_dendrite = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.da_size,
            filter_by_target_population=SNcPopulation.DA,
        )
        g_inh = striatal_dendrite.g_gaba_a

        baseline = torch.full((self.da_size,), self.config.baseline_drive, device=self.device)

        da_spikes, _ = self.da_neurons.forward(
            g_ampa_input=ConductanceTensor(baseline),
            g_nmda_input=None,
            g_gaba_a_input=ConductanceTensor(g_inh),
            g_gaba_b_input=None,
        )

        # =====================================================================
        # GABA INTERNEURONS
        # =====================================================================
        da_activity = da_spikes.float().mean().item()
        gaba_spikes = self._step_gaba_interneurons(da_activity)

        region_outputs: RegionOutput = {
            SNcPopulation.DA: da_spikes,
            SNcPopulation.GABA: gaba_spikes,
        }

        return region_outputs
