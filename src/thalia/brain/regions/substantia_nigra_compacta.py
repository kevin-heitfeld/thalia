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

from typing import ClassVar, Dict

import torch

from thalia.brain.configs import SubstantiaNigraCompactaConfig
from thalia.components import (
    ConductanceLIFConfig,
    ConductanceLIF,
)
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
    NeuromodulatorType,
    PopulationName,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)
from thalia.utils import split_excitatory_conductance

from .neuromodulator_source_region import NeuromodulatorSourceRegion
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
class SubstantiaNigraCompacta(NeuromodulatorSourceRegion[SubstantiaNigraCompactaConfig]):
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
    - ``da_nigrostriatal``: Routed to dorsal striatum by DynamicBrain
    """

    # Declarative neuromodulator channel.  DynamicBrain reads this ClassVar
    # at brain-build time to wire NeuromodulatorTract diffusion filters.
    neuromodulator_outputs: ClassVar[Dict[NeuromodulatorType, PopulationName]] = {
        "da_nigrostriatal": SNcPopulation.DA,
    }

    def __init__(
        self,
        config: SubstantiaNigraCompactaConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
    ) -> None:
        super().__init__(config, population_sizes, region_name)

        self.da_size = population_sizes[SNcPopulation.DA]
        self.gaba_size = population_sizes[SNcPopulation.GABA]

        # DA neurons — ConductanceLIF with spike-frequency adaptation and I_h pacemaker
        # I_h (HCN) contributes to 4-6 Hz tonic rhythm; activates during RMTg-driven pauses
        # and provides a rebound ramp that restores tonic firing (Neuhoff et al. 2002).
        self.da_neurons = ConductanceLIF(
            n_neurons=self.da_size,
            config=ConductanceLIFConfig(
                region_name=region_name,
                population_name=SNcPopulation.DA,
                tau_mem=config.tau_mem,
                v_threshold=1.0,
                v_reset=0.0,
                tau_ref=config.tau_ref,
                g_L=config.g_L,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                noise_std=config.noise_std,
                adapt_increment=config.adapt_increment,
                tau_adapt=config.tau_adapt,
                E_adapt=-0.5,
                # I_h (HCN) pacemaker — voltage-dependent rebound after inhibitory pauses
                enable_ih=True,
                g_h_max=0.03,
                E_h=-0.3,
                V_half_h=-0.35,
                k_h=0.08,
                tau_h_ms=150.0,
            ),
            device=self.device,
        )

        # GABAergic interneurons (local inhibitory control)
        self._init_gaba_interneurons(SNcPopulation.GABA, self.gaba_size)

        self._register_neuron_population(SNcPopulation.DA, self.da_neurons)

        # Ensure all tensors are on the correct device
        self.to(self.device)

    @torch.no_grad()
    def forward(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Advance SNc one timestep.

        Args:
            synaptic_inputs: Inhibitory feedback from striatum D1/D2 and GPe.
            neuromodulator_inputs: Unused (SNc is a neuromodulator source).

        Returns:
            ``RegionOutput`` with ``SNcPopulation.DA`` and ``SNcPopulation.GABA``
            spike tensors.
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        # =====================================================================
        # TONIC BASELINE DRIVE
        # =====================================================================
        # Intrinsic Ca²⁺ pacemaker keeps DA neurons at 4-6 Hz.
        # Above-threshold drive is balanced by spike-frequency adaptation (I_KCA).
        baseline = torch.full(
            (self.da_size,), self.config.baseline_drive, device=self.device
        )
        g_ampa, g_nmda = split_excitatory_conductance(baseline, nmda_ratio=0.30)

        # =====================================================================
        # STRIATAL SHORT-LOOP FEEDBACK (D1/D2 → SNc)
        # =====================================================================
        # Striatal D1 MSNs provide modest GABAergic feedback to SNc DA neurons,
        # creating a dopamine-to-striatum-to-SNc feedback loop that stabilizes
        # dopaminergic tone during sustained motor activity.
        striatal_dendrite = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.da_size,
            filter_by_target_population=SNcPopulation.DA,
        )
        g_inh = striatal_dendrite.g_gaba_a

        da_spikes, _ = self.da_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa),
            g_nmda_input=ConductanceTensor(g_nmda),
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

        return self._post_forward(region_outputs)

    def _compute_gaba_drive(self, primary_activity: float) -> torch.Tensor:
        """SNc GABA drive: tonic baseline + small DA autoinhibition term."""
        return torch.full((self.gaba_neurons_size,), 0.004 + primary_activity * 0.05, device=self.device)

    def _step_gaba_interneurons(self, primary_activity: float) -> torch.Tensor:
        """Override to use AMPA-only drive for SNc GABA interneurons.

        Same NMDA-buildup fix as VTA. Biology: SNc GABA interneurons have
        minimal NMDA receptors; AMPA-only is the correct pathway here.
        """
        gaba_drive = self._compute_gaba_drive(primary_activity)
        gaba_spikes, _ = self.gaba_neurons.forward(
            g_ampa_input=ConductanceTensor(gaba_drive),
            g_nmda_input=None,
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )
        return gaba_spikes
