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

from typing import TYPE_CHECKING, ClassVar, Dict, List, Union

import torch
from torch import nn

from thalia import GlobalConfig
from thalia.brain.configs import DopaminePacemakerConfig
from thalia.brain.neurons import (
    ConductanceLIFConfig,
    heterogeneous_adapt_increment,
    heterogeneous_dendrite_coupling,
    heterogeneous_g_L,
    heterogeneous_noise_std,
    heterogeneous_tau_adapt,
    heterogeneous_tau_mem,
    heterogeneous_v_threshold,
)
from thalia.brain.synapses import WeightInitializer
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

from .neuromodulator_source_region import NeuromodulatorSourceRegion
from .population_names import SNcPopulation
from .region_registry import register_region

if TYPE_CHECKING:
    from thalia.brain.neurons import ConductanceLIF


@register_region(
    "substantia_nigra_compacta",
    aliases=["snc", "nigra_compacta"],
    description="SNc - nigrostriatal dopamine pacemaker for motor learning",
)
class SubstantiaNigraCompacta(NeuromodulatorSourceRegion[DopaminePacemakerConfig]):
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

    # 5-HT from DRN inhibits nigrostriatal DA release via 5-HT1A on DA somata.
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = [
        NeuromodulatorChannel.SHT,
    ]

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
                v_reset=0.0,
                # v_threshold CV raised 0.12→0.25→0.35: tight threshold clustering
                # caused pacemaker phase-lock (100% epileptiform at CV=0.12, 60%
                # at CV=0.25).  Wider spread introduces natural frequency variation
                # that further breaks phase-lock across the population.
                v_threshold=heterogeneous_v_threshold(1.0, self.da_size, self.device, cv=0.35, clamp_fraction=0.25),
                tau_ref=config.tau_ref,
                g_L=heterogeneous_g_L(config.g_L, self.da_size, self.device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(config.noise_std, self.da_size, self.device),
                noise_tau_ms=3.0,
                # tau_adapt CV raised 0.25→0.40: combined with threshold hetero-
                # geneity, varied adaptation time constants further desynchronise
                # pacemaker neurons by giving each a different recovery trajectory.
                tau_adapt_ms=heterogeneous_tau_adapt(config.tau_adapt_ms, self.da_size, self.device, cv=0.40),
                adapt_increment=heterogeneous_adapt_increment(config.adapt_increment, self.da_size, self.device),
                E_adapt=-0.5,
                # I_h (HCN) pacemaker — see class-level constants for rationale.
                # g_h_max: 0.03→0.015→0.008→0.004→0.008 (reverted).  Reducing
                # g_h_max alone didn't fix phase-lock (100% at 0.004).  The real
                # fix is increased v_threshold + tau_adapt heterogeneity above.
                enable_ih=True,
                g_h_max=0.008,
                E_h=0.9,
                V_half_h=-0.35,
                k_h=0.08,
                tau_h_ms=150.0,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.da_size, self.device, cv=0.25),
            ),
        )

        # GABAergic interneurons (local inhibitory control)
        self._init_gaba_interneurons(SNcPopulation.GABA, self.gaba_size, device)

        # =====================================================================
        # SPARSE RECURRENT GABA INHIBITION (desynchronisation mechanism)
        # =====================================================================
        # SNc DA neurons lack D2 autoreceptors (unlike VTA mesolimbic), so they
        # have no self-inhibitory feedback to break I_h pacemaker synchrony.
        # This sparse inhibition matrix models heterogeneous local GABA interneuron
        # projections: when multiple DA neurons co-fire, each receives a different-
        # strength GABA feedback next timestep, creating competitive dynamics that
        # desynchronise the population.  Same proven pattern as VTA mesocortical fix.
        # Reduced 30%/0.012→20%/0.008: full-strength recurrent GABA
        # resolved epileptiform (100%→0%) but over-suppressed DA rate to
        # 1.38 Hz (target 2-8 Hz).  Lighter inhibition should maintain
        # desynchronisation while allowing higher tonic firing.
        self._da_recurrent_inhib_weights = nn.Parameter(
            WeightInitializer.sparse_random_no_autapses(
                n_input=self.da_size,
                n_output=self.da_size,
                connectivity=0.20,
                weight_scale=0.008,
                device=device,
            ),
            requires_grad=False,
        )

        # =====================================================================
        # SEROTONIN RECEPTOR (DRN → SNc, 5-HT1A inhibitory)
        # 5-HT1A on DA somata: serotonin suppresses tonic DA firing
        self._init_receptors_from_config(device)

        # Ensure all tensors are on the correct device
        self.to(device)

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Advance SNc one timestep."""
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

        # Sparse recurrent GABA inhibition: previous-timestep DA spikes →
        # heterogeneous GABA_A conductance.  Each DA neuron receives a
        # different weighted sum of neighbors' spikes, breaking co-activation.
        da_recurrent_gaba = torch.matmul(
            self._da_recurrent_inhib_weights,
            self._prev_spikes(SNcPopulation.DA),
        )
        g_inh = striatal_dendrite.g_gaba_a + da_recurrent_gaba

        # 5-HT1A inhibitory modulation: serotonin reduces DA excitability
        self._update_receptors(neuromodulator_inputs)
        sht_level = self._sht_concentration.mean().item()
        sht_suppression = max(0.0, 1.0 - 0.25 * sht_level)  # -25% at max 5-HT

        baseline = torch.full((self.da_size,), self.config.baseline_drive * sht_suppression, device=self.device)

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
        gaba_spikes = self._step_gaba_interneurons(
            da_activity, drive_scale=0.05, drive_baseline=0.004,
        )

        region_outputs: RegionOutput = {
            SNcPopulation.DA: da_spikes,
            SNcPopulation.GABA: gaba_spikes,
        }

        return region_outputs
