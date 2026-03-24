"""Subiculum — Hippocampal Output Gateway.

The subiculum is the principal output nucleus of the hippocampal formation,
interposed between CA1 and the entorhinal cortex.  It receives the majority
of CA1 axon collaterals and redistributes compressed hippocampal output to a
wide set of downstream targets.

Biological Background:
======================
**Anatomy:**
- Located between CA1 (stratum oriens border) and the presubiculum
- ~75 % of CA1 projections terminate in subiculum (O'Mara et al. 2001)
- Remaining ~25 % exit directly to EC_V (the direct CA1→EC path modelled
  by the previous direct connection that this region replaces)
- ~35,000 neurons in rat; predominantly excitatory pyramidal cells

**Three Physiological Cell Types (collapsed to one population here):**
1. **Regular-spiking** (~40 %): Tonic 5-10 Hz, no bursting
2. **Burst-firing** (~40 %): Initial Ca²⁺-driven doublet/triplet, then tonic
3. **Weak-bursting** (~20 %): Single initial burst, then regular

Heterogeneous `tau_mem_ms` and `v_threshold` in `ConductanceLIF` naturally give
rise to all three modes from a single population tensor.

**Inhibitory Interneurons:**
- PV basket cells (~15% of subicular neurons) provide fast feedback
  inhibition (< 1 ms latency) onto pyramidal somata, preventing runaway
  synchronous bursting and sharpening temporal precision.

**Inputs:**
- ``hippocampus / HippocampusPopulation.CA1``    — excitatory (Schaffer collateral relay)
- ``medial_septum / MedialSeptumPopulation.ACH`` — optional cholinergic modulation

**Outputs (axonal targets):**
- ``EntorhinalCortexPopulation.EC_V`` — perforant path back-projection to neocortex
- ``CortexPopulation.L5_PYR``         — direct hippocampal-to-PFC report (consolidation)
- ``BLAPopulation.PRINCIPAL``         — contextual fear/safety signal to amygdala
"""

from __future__ import annotations

from typing import ClassVar, List, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import SubiculumConfig
from thalia.brain.neurons import (
    ConductanceLIFConfig,
    ConductanceLIF,
    heterogeneous_adapt_increment,
    heterogeneous_dendrite_coupling,
    heterogeneous_g_L,
    heterogeneous_noise_std,
    heterogeneous_tau_adapt,
    heterogeneous_tau_mem,
    heterogeneous_v_threshold,
    split_excitatory_conductance,
)
from thalia.brain.synapses import STPConfig
from thalia.learning import InhibitorySTDPConfig, InhibitorySTDPStrategy
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorChannel,
    NeuromodulatorInput,
    PopulationPolarity,
    PopulationSizes,
    ReceptorType,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)

from .neural_region import InternalConnectionSpec, NeuralRegion
from .population_names import SubiculumPopulation
from .region_registry import register_region


@register_region(
    "subiculum",
    aliases=["sub", "hippocampal_output"],
    description="Subiculum — hippocampal output gateway (CA1 → EC / PFC / BLA relay)",
)
class Subiculum(NeuralRegion[SubiculumConfig]):
    """Subiculum — Hippocampal Output Gateway.

    Receives CA1 excitatory input and relays a regular-spiking transformed
    output to entorhinal cortex (EC_V), prefrontal cortex (L5_PYR), and
    basolateral amygdala (PRINCIPAL).

    Biologically, the subiculum *converts* CA1 complex-spike bursts into
    regular spiking: the burst-onset Ca²⁺ component is absorbed by adaptation
    (``adapt_increment``), and the subsequent regular plateau matches downstream
    expectations in EC and PFC.

    Populations:
    ------------
    - ``SubiculumPopulation.PRINCIPAL``: Excitatory pyramidal cells.
    - ``SubiculumPopulation.PV``: PV basket-cell interneurons (inhibitory).
      Provide fast feedback inhibition onto PRINCIPAL, preventing runaway bursting.

    Internal Connectivity:
    ----------------------
    - PRINCIPAL → PV (AMPA): Pyramidal excitation drives PV interneurons.
    - PV → PRINCIPAL (GABA_A): PV feedback inhibition sharpens temporal precision.

    Input Populations (via SynapseId routing):
    ------------------------------------------
    - ``hippocampus / CA1``           → PRINCIPAL  (main driver)
    - ``medial_septum / ACH``         → PRINCIPAL  (optional ACh arousal)
    - ``entorhinal_cortex / EC_III``  → PRINCIPAL  (optional direct EC input)

    Output Populations:
    -------------------
    - ``SubiculumPopulation.PRINCIPAL`` — spikes forwarded to EC_V, PFC, BLA
    """

    # Subiculum is a structural relay; it does not release neuromodulators.
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = []

    def __init__(
        self,
        config: SubiculumConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ) -> None:
        super().__init__(config, population_sizes, region_name, device=device)

        self.principal_size: int = population_sizes[SubiculumPopulation.PRINCIPAL]
        self.pv_size: int = population_sizes[SubiculumPopulation.PV]

        # ── Principal population: heterogeneous LIF pyramidal neurons ─────────
        self.principal_neurons: ConductanceLIF
        self.principal_neurons = self._create_and_register_neuron_population(
            population_name=SubiculumPopulation.PRINCIPAL,
            n_neurons=self.principal_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(config.tau_mem_ms, self.principal_size, device),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(config.v_threshold, self.principal_size, device),
                tau_ref=5.0,
                g_L=heterogeneous_g_L(0.05, self.principal_size, device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.principal_size, device),
                noise_std=heterogeneous_noise_std(0.005, self.principal_size, device),
                noise_tau_ms=3.0,
                tau_adapt_ms=heterogeneous_tau_adapt(config.tau_adapt_ms, self.principal_size, device),
                adapt_increment=heterogeneous_adapt_increment(config.adapt_increment, self.principal_size, device),
            ),
        )

        # ── PV basket-cell population: fast-spiking inhibitory interneurons ──
        self.pv_neurons: ConductanceLIF
        self.pv_neurons = self._create_and_register_neuron_population(
            population_name=SubiculumPopulation.PV,
            n_neurons=self.pv_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(10.0, self.pv_size, device),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(0.50, self.pv_size, device),
                tau_ref=1.5,  # Short refractory: fast-spiking PV interneurons
                g_L=heterogeneous_g_L(0.08, self.pv_size, device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=3.0,    # Fast AMPA kinetics for PV cells
                tau_I=8.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.pv_size, device),
                noise_std=heterogeneous_noise_std(0.008, self.pv_size, device),
                noise_tau_ms=2.0,
                tau_adapt_ms=heterogeneous_tau_adapt(50.0, self.pv_size, device),
                adapt_increment=heterogeneous_adapt_increment(0.0, self.pv_size, device),
            ),
        )

        # ── Internal connectivity ─────────────────────────────────────────────
        # Inhibitory STDP for PV→Principal (Vogels et al. 2011)
        istdp_cfg = InhibitorySTDPConfig(
            learning_rate=config.istdp_learning_rate,
            tau_istdp=config.istdp_tau_ms,
            alpha=config.istdp_alpha,
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        )
        self.istdp_pv: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)

        # Principal ↔ PV: excitatory feedforward drive + fast GABAergic feedback
        self._add_feedforward_inhibition(
            exc_pop=SubiculumPopulation.PRINCIPAL,
            inh_pop=SubiculumPopulation.PV,
            e_to_i=InternalConnectionSpec(
                connectivity=0.50,
                # Reverted 0.030→0.025: 0.030 still caused 16% PV epileptiform (T110333).
                # PV→E is now 0.09 (3.6× original), so each PV spike delivers much
                # stronger inhibition — fewer, desynchronized PV spikes suffice.
                weight_scale=0.025,
                receptor_type=ReceptorType.AMPA,
                stp_config=STPConfig(U=0.35, tau_d=250.0, tau_f=20.0),
            ),
            i_to_e=InternalConnectionSpec(
                connectivity=0.40,
                # Raised 0.12→0.14: E/I=8.4 (T150804), just over threshold.
                # PV rate 5.58 Hz — each spike needs more inhibitory punch.
                weight_scale=0.14,
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.30, tau_d=300.0, tau_f=15.0),
                learning_strategy=self.istdp_pv,
            ),
            device=device,
        )

        self.to(device)

    # =========================================================================
    # FORWARD
    # =========================================================================

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Advance subiculum one timestep.

        Processing:
        1. Integrate external synaptic inputs for PRINCIPAL dendrites.
        2. Add tonic sub-threshold depolarisation.
        3. Fire principal neurons (with PV feedback from previous step).
        4. Drive PV interneurons with current-step principal spikes.
        5. Use PV spikes for next-step feedback inhibition.
        """
        # ── 1. External dendritic integration for PRINCIPAL ───────────────────
        dendrite_principal = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.principal_size,
            filter_by_target_population=SubiculumPopulation.PRINCIPAL,
        )

        # ── 2. PV → Principal feedback inhibition (from previous step) ────────
        pv_to_principal = SynapseId(
            source_region=self.region_name,
            source_population=SubiculumPopulation.PV,
            target_region=self.region_name,
            target_population=SubiculumPopulation.PRINCIPAL,
            receptor_type=ReceptorType.GABA_A,
        )
        int_pv_principal = self._integrate_single_synaptic_input(pv_to_principal, self._prev_spikes(SubiculumPopulation.PV))

        # ── 3. Combined excitatory drive ──────────────────────────────────────
        exc_drive = dendrite_principal.g_ampa + self.config.tonic_drive
        exc_drive = torch.nn.functional.relu(exc_drive)

        # 25 % NMDA: subicular synapses contain moderate NMDA receptor density
        g_ampa, g_nmda = split_excitatory_conductance(exc_drive, nmda_ratio=0.25)

        g_inh_principal = dendrite_principal.g_gaba_a + int_pv_principal.g_gaba_a

        # ── 4. Fire principal neurons ─────────────────────────────────────────
        principal_spikes, _ = self.principal_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa),
            g_nmda_input=ConductanceTensor(g_nmda + dendrite_principal.g_nmda),
            g_gaba_a_input=ConductanceTensor(g_inh_principal),
            g_gaba_b_input=None,
        )

        # ── 5. PV interneurons: driven by external inputs + principal spikes ──
        dendrite_pv = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.pv_size,
            filter_by_target_population=SubiculumPopulation.PV,
        )

        principal_to_pv = SynapseId(
            source_region=self.region_name,
            source_population=SubiculumPopulation.PRINCIPAL,
            target_region=self.region_name,
            target_population=SubiculumPopulation.PV,
            receptor_type=ReceptorType.AMPA,
        )
        int_principal_pv = self._integrate_synaptic_inputs_at_dendrites(
            {principal_to_pv: principal_spikes},
            n_neurons=self.pv_size,
        )

        g_exc_pv = dendrite_pv.g_ampa + int_principal_pv.g_ampa
        g_exc_pv = torch.nn.functional.relu(g_exc_pv)

        pv_spikes, _ = self.pv_neurons.forward(
            g_ampa_input=ConductanceTensor(g_exc_pv),
            g_nmda_input=ConductanceTensor(dendrite_pv.g_nmda + int_principal_pv.g_nmda),
            g_gaba_a_input=ConductanceTensor(dendrite_pv.g_gaba_a),
            g_gaba_b_input=None,
        )

        return {
            SubiculumPopulation.PRINCIPAL: principal_spikes,
            SubiculumPopulation.PV: pv_spikes,
        }
