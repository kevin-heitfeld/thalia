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
- Dopamine-gated plasticity for motor sequence learning
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import DopaminePacemakerConfig
from thalia.brain.neurons import (
    ConductanceLIFConfig,
    heterogeneous_adapt_increment,
    heterogeneous_g_L,
    heterogeneous_tau_mem,
    heterogeneous_v_threshold,
)
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
    NeuromodulatorChannel,
    PopulationName,
    PopulationPolarity,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapticInput,
)

from .dopamine_pacemaker_base import DopaminePacemakerBase
from .population_names import SNcPopulation
from .region_registry import register_region

if TYPE_CHECKING:
    from thalia.brain.neurons import ConductanceLIF


@register_region(
    "substantia_nigra_compacta",
    aliases=["snc", "nigra_compacta"],
    description="SNc - nigrostriatal dopamine pacemaker for motor learning",
)
class SubstantiaNigraCompacta(DopaminePacemakerBase[DopaminePacemakerConfig]):
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
        config: DopaminePacemakerConfig,
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
        # and provides a rebound ramp that restores tonic firing.
        self.da_neurons: ConductanceLIF
        self.da_neurons = self._create_and_register_neuron_population(
            SNcPopulation.DA,
            self.da_size,
            polarity=PopulationPolarity.ANY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(config.tau_mem_ms, self.da_size, self.device, cv=0.20),
                v_threshold=heterogeneous_v_threshold(1.0, self.da_size, self.device, cv=0.12, clamp_fraction=0.25),
                v_reset=0.0,
                tau_ref=config.tau_ref,
                g_L=heterogeneous_g_L(config.g_L, self.da_size, self.device),
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                noise_std=config.noise_std,
                adapt_increment=heterogeneous_adapt_increment(config.adapt_increment, self.da_size, self.device),
                tau_adapt=config.tau_adapt,
                E_adapt=-0.5,
                # I_h (HCN) pacemaker — see class-level constants for rationale.
                enable_ih=True,
                g_h_max=0.03,
                E_h=0.9,
                V_half_h=-0.35,
                k_h=0.08,
                tau_h_ms=150.0,
            ),
        )

        # GABAergic interneurons (local inhibitory control)
        self.gaba_neurons: ConductanceLIF
        self.gaba_neurons = self._create_and_register_neuron_population(
            population_name=SNcPopulation.GABA,
            n_neurons=self.gaba_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(8.0, self.gaba_size, device=self.device, cv=0.10),
                v_threshold=heterogeneous_v_threshold(1.0, self.gaba_size, device=self.device, cv=0.06),
                v_reset=0.0,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=3.0,
                tau_I=3.0,
                tau_ref=2.5,
                g_L=heterogeneous_g_L(0.10, self.gaba_size, device=self.device, cv=0.08),
            ),
        )

        self._prev_gaba_spikes: torch.Tensor
        self.register_buffer("_prev_gaba_spikes", torch.zeros(self.gaba_size, dtype=torch.bool, device=self.device), persistent=False)

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
        # adaptation (see DopaminePacemakerConfig.baseline_drive):
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
