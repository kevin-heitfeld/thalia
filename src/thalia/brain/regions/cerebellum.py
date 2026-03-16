"""
Cerebellum - Supervised Error-Corrective Learning for Precise Motor Control.

The cerebellum learns through supervised error signals from climbing fibers,
enabling fast, precise learning of input-output mappings without trial-and-error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, List, Optional, Union

import torch
import torch.nn as nn

from thalia import GlobalConfig
from thalia.brain.configs import CerebellumConfig
from thalia.brain.gap_junctions import (
    GapJunctionConfig,
    GapJunctionCoupling,
)
from thalia.brain.neurons import (
    ConductanceLIFConfig,
    heterogeneous_tau_mem,
    heterogeneous_v_threshold,
    heterogeneous_adapt_increment,
    heterogeneous_g_L,
    split_excitatory_conductance,
)
from thalia.brain.synapses import (
    NMReceptorType,
    make_neuromodulator_receptor,
    STPConfig,
    WeightInitializer,
)
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
from thalia.utils import clamp_weights, decay_tensor

from .neural_region import NeuralRegion
from .population_names import CerebellumPopulation
from .region_registry import register_region

if TYPE_CHECKING:
    from thalia.brain.neurons import ConductanceLIF


@register_region(
    "cerebellum",
    description="Supervised error-corrective learning via climbing fiber error signals",
)
class Cerebellum(NeuralRegion[CerebellumConfig]):
    """Cerebellar region with supervised error-corrective learning.

    Architecture:
    - Granule cells: Internal only (parallel fibers → Purkinje dendrites)
    - Purkinje cells: Internal only (axons → DCN, inhibitory GABAergic)
    - DCN (deep cerebellar nuclei): Output neurons (axons → thalamus/motor)

    Neuromodulator subscriptions:
    - **NE** (β-adrenergic on Purkinje): Increases intrinsic excitability and
      potentiates the LTP window in the MAI rule (Woodward et al. 1991).
    - **DA** (D1 on granule cells): Enhances parallel-fiber → Purkinje LTD,
      sharpening error-driven motor adaptation (Bhatt et al. 2007).
    - **ACh** (muscarinic M2 on Golgi): Suppresses Golgi→granule GABA-A,
      temporarily increasing granule sparsity ceiling during attention.
    """

    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = [
        NeuromodulatorChannel.NE,
        NeuromodulatorChannel.DA_NIGROSTRIATAL,
        NeuromodulatorChannel.ACH,
    ]

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        config: CerebellumConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        """Initialize cerebellum."""
        super().__init__(config, population_sizes, region_name, device=device)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.dcn_size = population_sizes[CerebellumPopulation.DCN]
        self.granule_size = population_sizes[CerebellumPopulation.GRANULE]
        self.purkinje_size = population_sizes[CerebellumPopulation.PURKINJE]
        self.basket_size = population_sizes[CerebellumPopulation.BASKET]
        self.stellate_size = population_sizes[CerebellumPopulation.STELLATE]

        # =====================================================================
        # MOSSY FIBER LAYER (Pontine Nuclei equivalent)
        # =====================================================================
        # Biology: Cortex projects to pontine nuclei → mossy fibers → cerebellum.
        # n_mossy is used for DCN collateral weights (downsampled granule activity);
        # external synaptic inputs target CerebellumPopulation.GRANULE and are
        # already integrated to granule dimensionality by the NeuralRegion machinery.
        self.n_mossy = max(int(self.granule_size * 0.1), 50)  # At least 50 mossy fibers

        # =====================================================================
        # GRANULE CELL LAYER (granule + Golgi microcircuit)
        # =====================================================================

        golgi_size = max(int(self.granule_size * self.config.golgi_ratio), 10)

        # Granule neurons (fast, excitable; fire briefly in sparse bursts)
        self.granule_neurons: ConductanceLIF
        self.granule_neurons = self._create_and_register_neuron_population(
            population_name=CerebellumPopulation.GRANULE,
            n_neurons=self.granule_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=ConductanceLIFConfig(
                v_threshold=heterogeneous_v_threshold(0.85, self.granule_size, device),
                v_reset=0.0,
                tau_mem_ms=heterogeneous_tau_mem(5.0, self.granule_size, device),
                tau_E=2.5,
                tau_I=6.0,
                g_L=heterogeneous_g_L(0.05, self.granule_size, device),
                adapt_increment=heterogeneous_adapt_increment(0.15, self.granule_size, device),
                tau_adapt=100.0,
            ),
        )

        # Golgi neurons (large inhibitory interneurons, tonic 5–10 Hz)
        # Receive mossy-fiber and granule-cell excitation; inhibit granule
        # dendrites via GABA-A to enforce sparse coding.
        self.golgi_neurons: ConductanceLIF
        self.golgi_neurons = self._create_and_register_neuron_population(
            population_name=CerebellumPopulation.GOLGI,
            n_neurons=golgi_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                v_threshold=heterogeneous_v_threshold(0.9, golgi_size, device),
                v_reset=0.0,
                tau_mem_ms=heterogeneous_tau_mem(10.0, golgi_size, device),
                tau_E=3.0,
                tau_I=10.0,
                g_L=heterogeneous_g_L(0.05, golgi_size, device),
            ),
        )

        # Granule → Golgi  (feedforward excitation, sparse)
        # Biology: ~10% of granule cells contact each Golgi axon
        self.granule_golgi_weights: nn.Parameter = nn.Parameter(
            WeightInitializer.sparse_random(
                n_input=self.granule_size,
                n_output=golgi_size,
                connectivity=0.10,
                weight_scale=0.002,
                device=device,
            ),
            requires_grad=False,
        )

        # Golgi → Granule  (feedback/feedforward inhibition, broad coverage)
        # Biology: Each Golgi cell inhibits 300–500 granule cells; reach ~30%
        self.golgi_granule_weights: nn.Parameter = nn.Parameter(
            WeightInitializer.sparse_random(
                n_input=golgi_size,
                n_output=self.granule_size,
                connectivity=0.30,
                weight_scale=0.015,
                device=device,
            ),
            requires_grad=False,
        )

        # Causal delay: hold previous-timestep Golgi spikes to avoid circular dep.
        # Biology: Golgi axons have short myelinated segments (~0.5–1 ms latency);
        # a 1-step delay is the minimum causal representation.
        self._prev_golgi_spikes: torch.Tensor
        self.register_buffer("_prev_golgi_spikes", torch.zeros(golgi_size, dtype=torch.bool, device=device))

        # =====================================================================
        # Molecular Layer Interneurons: Basket + Stellate cells
        # =====================================================================

        # Basket neurons (fast-spiking PV+, inner molecular layer)
        # Biology: quick excitation from PF → high-frequency response;
        # τ_mem ~8 ms, no accommodation (truly non-adapting).
        self.basket_neurons: ConductanceLIF
        self.basket_neurons = self._create_and_register_neuron_population(
            population_name=CerebellumPopulation.BASKET,
            n_neurons=self.basket_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                v_threshold=heterogeneous_v_threshold(0.65, self.basket_size, device, cv=0.06),
                v_reset=0.0,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                g_L=heterogeneous_g_L(0.08, self.basket_size, device, cv=0.08),
                tau_mem_ms=heterogeneous_tau_mem(8.0, self.basket_size, device, cv=0.10),
                tau_E=2.0,
                tau_I=6.0,
                adapt_increment=0.0,
                tau_ref=1.5,
            ),
        )

        # Stellate neurons (fast-spiking PV+, outer molecular layer)
        # Slightly higher threshold than basket (less tonic drive from PF).
        self.stellate_neurons: ConductanceLIF
        self.stellate_neurons = self._create_and_register_neuron_population(
            population_name=CerebellumPopulation.STELLATE,
            n_neurons=self.stellate_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                v_threshold=heterogeneous_v_threshold(0.70, self.stellate_size, device),
                v_reset=0.0,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                g_L=heterogeneous_g_L(0.08, self.stellate_size, device, cv=0.08),
                tau_mem_ms=heterogeneous_tau_mem(10.0, self.stellate_size, device),
                tau_E=2.5,
                tau_I=7.0,
                adapt_increment=0.0,
                tau_ref=1.5,
            ),
        )

        # Granule (parallel fiber) → Basket  [basket_size × granule_size]
        # Biology: each basket cell receives ~3000 parallel fiber contacts
        # (scattered across the folium, ~15–20% of granule cells in range)
        self.granule_basket_weights: nn.Parameter = nn.Parameter(
            WeightInitializer.sparse_random(
                n_input=self.granule_size,
                n_output=self.basket_size,
                connectivity=0.15,
                weight_scale=0.005,
                device=device,
            ),
            requires_grad=False,
        )

        # Granule (parallel fiber) → Stellate  [stellate_size × granule_size]
        # Stellate cells are positioned more distally on the PF beam,
        # receiving slightly fewer contacts per cell.
        self.granule_stellate_weights: nn.Parameter = nn.Parameter(
            WeightInitializer.sparse_random(
                n_input=self.granule_size,
                n_output=self.stellate_size,
                connectivity=0.12,
                weight_scale=0.005,
                device=device,
            ),
            requires_grad=False,
        )

        # Basket → Purkinje soma  [purkinje_size × basket_size]
        # Strong somatic GABA-A ('basket' synapses around the axon initial segment).
        # Lateral connectivity: each basket cell contacts 6–10 adjacent Purkinje.
        self.basket_purkinje_weights: nn.Parameter = nn.Parameter(
            WeightInitializer.sparse_random(
                n_input=self.basket_size,
                n_output=self.purkinje_size,
                connectivity=0.25,
                weight_scale=0.008,
                device=device,
            ),
            requires_grad=False,
        )

        # Stellate → Purkinje dendrite  [purkinje_size × stellate_size]
        # Dendritic (GABA-A) inhibition; slightly weaker per synapse than basket
        # soma synapses, but broader spatial coverage.
        self.stellate_purkinje_weights: nn.Parameter = nn.Parameter(
            WeightInitializer.sparse_random(
                n_input=self.stellate_size,
                n_output=self.purkinje_size,
                connectivity=0.10,
                weight_scale=0.001,
                device=device,
            ),
            requires_grad=False,
        )

        # Basket → Basket lateral inhibition  [basket_size × basket_size]
        # Biology: basket cell axon collaterals contact 5–10 neighbouring
        # basket cells, providing mutual suppression that prevents the entire
        # population from firing synchronously in response to a common parallel
        # fiber beam.
        self.basket_basket_weights: nn.Parameter = nn.Parameter(
            WeightInitializer.sparse_random(
                n_input=self.basket_size,
                n_output=self.basket_size,
                connectivity=0.30,
                weight_scale=0.030,
                device=device,
            ),
            requires_grad=False,
        )

        # Stellate → Stellate lateral inhibition  [stellate_size × stellate_size]
        # Biology: stellate collaterals inhibit neighbouring stellate cells,
        # similar in mechanism to basket lateral inhibition but weaker.
        self.stellate_stellate_weights: nn.Parameter = nn.Parameter(
            WeightInitializer.sparse_random(
                n_input=self.stellate_size,
                n_output=self.stellate_size,
                connectivity=0.25,
                weight_scale=0.020,
                device=device,
            ),
            requires_grad=False,
        )

        # Causal delay buffers: previous-timestep spikes → applied next step
        self._prev_basket_spikes: torch.Tensor
        self._prev_stellate_spikes: torch.Tensor
        self.register_buffer("_prev_basket_spikes", torch.zeros(self.basket_size, dtype=torch.bool, device=device))
        self.register_buffer("_prev_stellate_spikes", torch.zeros(self.stellate_size, dtype=torch.bool, device=device))

        # =====================================================================
        # Purkinje cells
        # =====================================================================
        # Single weight matrix [purkinje_size, granule_size]
        # Biology: Each Purkinje cell receives ~200k of ~50B granule cells (<1%).
        # 12% connectivity reduces shared-input overlap (was 20% → 4% pairwise
        # overlap; now 12% → 1.4%) to decorrelate Purkinje firing.
        self.purkinje_synaptic_weights = nn.Parameter(
            WeightInitializer.sparse_random(
                n_input=self.granule_size,
                n_output=self.purkinje_size,
                connectivity=0.12,
                weight_scale=0.000013,
                device=device,
            ),
            requires_grad=False,
        )

        self.soma_neurons: ConductanceLIF
        self.soma_neurons = self._create_and_register_neuron_population(
            population_name=CerebellumPopulation.PURKINJE,
            n_neurons=self.purkinje_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(20.0, self.purkinje_size, device),
                tau_E=2.0,
                tau_I=10.0,
                tau_nmda=50.0,
                # v_threshold reduced from 1.3 → 0.9: previous value was above
                # threshold_max=1.0 of the homeostatic system, so homeostasis could
                # never lower it far enough.  0.9 allows the adaptive threshold to
                # settle in [0.1, 1.0] and reach 40–100 Hz with the pacemaker drive.
                v_threshold=heterogeneous_v_threshold(0.9, self.purkinje_size, device),
                v_reset=0.0,
                E_L=0.0,
                tau_ref=2.0,
                g_L=heterogeneous_g_L(0.10, self.purkinje_size, device),
                adapt_increment=heterogeneous_adapt_increment(0.10, self.purkinje_size, device),
                tau_adapt=30.0,
            ),
        )

        # Dendritic voltage and calcium [purkinje_size, purkinje_n_dendrites]
        self.dendrite_voltage: torch.Tensor
        self.dendrite_calcium: torch.Tensor
        self.register_buffer("dendrite_voltage", torch.zeros(self.purkinje_size, config.purkinje_n_dendrites, device=device))
        self.register_buffer("dendrite_calcium", torch.zeros(self.purkinje_size, config.purkinje_n_dendrites, device=device))

        # Calcium decay factor (tau ~50ms) and voltage decay factor (tau ~10ms).
        self.calcium_decay: torch.Tensor
        self.voltage_decay: torch.Tensor
        self.register_buffer("calcium_decay", decay_tensor(self.dt_ms, 50.0, device))
        self.register_buffer("voltage_decay", decay_tensor(self.dt_ms, 10.0, device))

        # Per-cell heterogeneous pacemaker drive [purkinje_size]
        # Biology: Resurgent Na⁺ and P/Q-type Ca²⁺ channels provide intrinsic tonic drive
        # with cell-to-cell variability in channel density and kinetics.  A shared
        # scalar pacemaker forces all cells to fire synchronously (ρ→0.84).  Per-cell
        # draws with std≈14% of mean naturally decorrelate tonic firing phases.
        self.pacemaker_g: torch.Tensor
        self.register_buffer("pacemaker_g", torch.empty(self.purkinje_size, device=device).normal_(mean=0.060, std=0.014).clamp_(0.025, 0.095))

        # Complex spike tracking [purkinje_size]
        self.last_complex_spike_time: torch.Tensor
        self.register_buffer("last_complex_spike_time", torch.full((self.purkinje_size,), -1000, dtype=torch.int32, device=device))
        self.complex_spike_refractory_ms = 100.0  # ~10 Hz max
        self.complex_spike_timestep = 0

        # =====================================================================
        # Deep cerebellar nuclei (final output)
        # =====================================================================
        # Receives both Purkinje inhibition and mossy fiber collaterals

        # Purkinje → DCN (inhibitory, convergent)
        # Many Purkinje cells converge onto each DCN neuron
        self._add_internal_connection(
            source_population=CerebellumPopulation.PURKINJE,
            target_population=CerebellumPopulation.DCN,
            weights=WeightInitializer.sparse_random(
                n_input=self.purkinje_size,
                n_output=self.dcn_size,
                connectivity=0.2,
                weight_scale=0.002,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.15, tau_d=60.0, tau_f=400.0),
        )

        # Mossy fiber → DCN (excitatory collaterals)
        # Mossy fibers send collaterals to DCN before reaching granule cells
        self._add_internal_connection(
            source_population=CerebellumPopulation.MOSSY,
            target_population=CerebellumPopulation.DCN,
            weights=WeightInitializer.sparse_random(
                n_input=self.n_mossy,
                n_output=self.dcn_size,
                connectivity=0.1,
                weight_scale=0.003,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.15, tau_d=200.0, tau_f=200.0),
        )

        # DCN neurons (tonic firing, modulated by inhibition)
        # DCN neurons are spontaneously active (tonic firing ~40-60 Hz)
        self.dcn_neurons: ConductanceLIF
        self.dcn_neurons = self._create_and_register_neuron_population(
            population_name=CerebellumPopulation.DCN,
            n_neurons=self.dcn_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=ConductanceLIFConfig(
                v_threshold=heterogeneous_v_threshold(1.0, self.dcn_size, device),
                v_reset=0.0,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                g_L=heterogeneous_g_L(0.10, self.dcn_size, device, cv=0.08),
                tau_mem_ms=heterogeneous_tau_mem(20.0, self.dcn_size, device),
                tau_E=4.0,
                tau_I=10.0,
                tau_ref=12.0,
            ),
        )

        # IO membrane potential for gap junction coupling
        self._io_membrane: Optional[torch.Tensor] = None

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY (Adaptive Gain)
        # =====================================================================
        self._register_homeostasis(CerebellumPopulation.PURKINJE, self.purkinje_size, target_firing_rate=0.045, device=device)

        self.gain: torch.Tensor
        self.register_buffer("gain", torch.ones(self.purkinje_size, device=device))

        # =====================================================================
        # INFERIOR OLIVE NEURONS (Error Signal Generation)
        # =====================================================================
        # One IO neuron per Purkinje cell (1:1 mapping)
        # IO neurons generate climbing fiber signals (complex spikes) that teach
        # Purkinje cells.
        self.io_neurons: ConductanceLIF
        self.io_neurons = self._create_and_register_neuron_population(
            population_name=CerebellumPopulation.INFERIOR_OLIVE,
            n_neurons=self.purkinje_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=ConductanceLIFConfig(
                v_threshold=heterogeneous_v_threshold(0.8, self.purkinje_size, device, cv=0.12, clamp_fraction=0.25),
                v_reset=0.0,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                g_L=heterogeneous_g_L(0.05, self.purkinje_size, device),
                tau_mem_ms=heterogeneous_tau_mem(18.0, self.purkinje_size, device, cv=0.20),
                tau_E=5.0,
                tau_I=10.0,
                tau_ref=4.0,
                adapt_increment=heterogeneous_adapt_increment(0.40, self.purkinje_size, device),
                tau_adapt=100.0,
                enable_t_channels=True,
                g_T=0.08,
                E_Ca=4.0,
                tau_h_T_ms=50.0,
                V_half_h_T=-0.3,
                k_h_T=0.15,
                enable_ih=True,
                g_h_max=0.04,
                E_h=-0.3,
                V_half_h=-0.3,
                k_h=0.10,
                tau_h_ms=100.0,
            ),
        )

        # =====================================================================
        # GAP JUNCTIONS (Inferior Olive Synchronization)
        # =====================================================================
        # IO neurons are densely coupled via gap junctions, creating synchronized
        # complex spikes across multiple Purkinje cells. This is critical for
        # coordinated motor learning across cerebellar modules.
        #
        # Biology: IO forms one of the densest gap junction networks in the brain
        # - Strong coupling (0.18) for population synchronization
        # - Dense connectivity (~80% of nearby neurons)
        # - Synchronization time: <1ms

        gap_junctions_config = GapJunctionConfig(
            coupling_strength=config.gap_junction_strength,
            connectivity_threshold=config.gap_junction_threshold,
            max_neighbors=config.gap_junction_max_neighbors,
            interneuron_only=False,  # IO neurons are projection neurons, not interneurons
        )

        # Initialize gap junctions using Purkinje dendritic weights as connectivity pattern
        # IO neurons corresponding to Purkinje cells with similar parallel fiber inputs
        # are anatomically close and should be electrically coupled
        self.gap_junctions_io = GapJunctionCoupling(
            n_neurons=self.purkinje_size,
            afferent_weights=self.purkinje_synaptic_weights,  # Use parallel fiber connectivity
            config=gap_junctions_config,
            interneuron_mask=None,  # All IO neurons can be coupled
            device=device,
        )

        # DCN tonic pacemaker baseline (approximates intrinsic I_NaP and I_h currents).
        # Purkinje inhibition sculpts this tonic drive; see CerebellumConfig.dcn_baseline_drive.
        self._dcn_baseline = torch.full((self.dcn_size,), config.dcn_baseline_drive, device=device)

        # =====================================================================
        # NOREPINEPHRINE RECEPTOR (β-adrenergic on Purkinje cells)
        # =====================================================================
        # NE from LC → increased cAMP → PKA activation → enhanced Purkinje
        # intrinsic excitability and potentiated LTP window.
        # β-adrenergic (Gs → cAMP → PKA): τ_rise=80 ms, τ_decay=1000 ms.
        self.ne_receptor = make_neuromodulator_receptor(
            NMReceptorType.NE_BETA, n_receptors=self.purkinje_size, dt_ms=self.dt_ms, device=device,
            amplitude_scale=1.5,  # Strong β-adrenergic sensitivity on Purkinje
        )
        self._ne_concentration: torch.Tensor
        self.register_buffer("_ne_concentration", torch.zeros(self.purkinje_size, device=device))

        # =====================================================================
        # DOPAMINE RECEPTOR (D1 on granule cells — via SNc projection)
        # =====================================================================
        # DA from SNc enhances cerebellar LTD (parallel fiber → Purkinje weight
        # depression) via PKA/DARPP-32 cascade on granule somata.
        # D1 (Gs → cAMP → PKA): τ_rise=500 ms, τ_decay=8000 ms.
        self.da_receptor = make_neuromodulator_receptor(
            NMReceptorType.DA_D1, n_receptors=self.granule_size, dt_ms=self.dt_ms, device=device,
            amplitude_scale=2.5,  # Preserve legacy effective amplitude
        )
        self._da_concentration: torch.Tensor
        self.register_buffer("_da_concentration", torch.zeros(self.granule_size, device=device))

        # =====================================================================
        # ACETYLCHOLINE RECEPTOR (muscarinic M2 on Golgi cells)
        # =====================================================================
        # ACh from NB → M2 autoreceptors on Golgi somata → suppresses Golgi
        # GABA-A output → disinhibits granule cells during attentional states.
        # Muscarinic M2 (Gi → GIRK): τ_rise=50 ms, τ_decay=600 ms.
        self.ach_receptor = make_neuromodulator_receptor(
            NMReceptorType.ACH_MUSCARINIC_M2, n_receptors=1, dt_ms=self.dt_ms, device=device,
            amplitude_scale=2.0,  # Scalar receptor; strong per-cell effect
        )
        self._ach_concentration: torch.Tensor
        self.register_buffer("_ach_concentration", torch.zeros(1, device=device))

        # Ensure all tensors are on the correct device
        self.to(device)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _step(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Process input through cerebellar circuit.

        Neuromodulator effects applied each step:
        - NE (β-adrenergic on Purkinje): updates per-Purkinje ``_ne_concentration``
          buffer, which scales the Purkinje gain and LTP amplitude in the MAI rule.
        - DA (D1 on granule): updates ``_da_concentration``, which boosts the
          MAI LTD rate — more DA → stronger climbing-fiber driven weight depression.
        - ACh (muscarinic M2 on Golgi): updates scalar ``_ach_concentration``,
          which suppresses Golgi→granule GABA-A conductance.
        """
        config = self.config

        # =====================================================================
        # NEUROMODULATOR RECEPTOR UPDATES
        # =====================================================================
        ne_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.NE)
        self._ne_concentration = self.ne_receptor.update(ne_spikes)

        da_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.DA_NIGROSTRIATAL)
        self._da_concentration = self.da_receptor.update(da_spikes)

        ach_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.ACH)
        self._ach_concentration = self.ach_receptor.update(ach_spikes)

        # Derive scalar modulation factors from mean concentrations
        # NE: per-Purkinje gain ∈ [1.0, 1.5] — β-adrenergic excitability boost
        ne_purkinje_gain = 1.0 + self._ne_concentration * 0.5  # [purkinje_size]
        # DA: scalar LTD amplification ∈ [1.0, 2.0] — D1/PKA potentiates LTD
        da_ltd_boost = 1.0 + float(self._da_concentration.mean().item()) * 2.0
        # ACh: Golgi inhibition scale ∈ [0.5, 1.0] — M2 suppresses Golgi GABA-A
        golgi_inhibition_scale = max(0.5, 1.0 - float(self._ach_concentration.item()) * 0.5)

        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        # Biology: Different sources (cortex, spinal, brainstem) project to pontine
        # nuclei, which give rise to mossy fibers. We model this as a integration stage.
        # NOTE: Weights project directly to granule layer (granule_size), not to mossy
        # fiber intermediates. The granule layer handles internal expansion/sparsification.
        mossy_fiber_conductances = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.granule_size,
            filter_by_target_population=CerebellumPopulation.GRANULE,
        ).g_ampa

        # =====================================================================
        # GRANULE–GOLGI MICROCIRCUIT
        # =====================================================================
        # Golgi → Granule GABA-A inhibition (causal: previous timestep)
        #    golgi_inhibition_scale < 1.0 → ACh muscarinic suppression of Golgi output
        golgi_inh = golgi_inhibition_scale * torch.mv(self.golgi_granule_weights, self._prev_golgi_spikes.float())

        # Granule cell spiking
        granule_g_ampa, granule_g_nmda = split_excitatory_conductance(mossy_fiber_conductances, nmda_ratio=0.3)
        granule_spikes, _ = self.granule_neurons.forward(
            g_ampa_input=ConductanceTensor(granule_g_ampa),
            g_nmda_input=ConductanceTensor(granule_g_nmda),
            g_gaba_a_input=ConductanceTensor(golgi_inh),
            g_gaba_b_input=None,
        )
        granule_spikes_float = granule_spikes.float()  # [granule_size]

        # Granule → Golgi feedforward excitation (same timestep, Golgi fires later)
        golgi_exc = torch.mv(self.granule_golgi_weights, granule_spikes_float)
        golgi_g_ampa, golgi_g_nmda = split_excitatory_conductance(golgi_exc, nmda_ratio=0.0)
        golgi_spikes, _ = self.golgi_neurons.forward(
            g_ampa_input=ConductanceTensor(golgi_g_ampa),
            g_nmda_input=ConductanceTensor(golgi_g_nmda),
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )

        # Mossy fiber collaterals → DCN: downsample granule activity
        # to the n_mossy count used by the internal Purkinje→DCN weight matrix.
        step = max(1, self.granule_size // self.n_mossy)
        mossy_spikes = granule_spikes[::step][:self.n_mossy]  # [n_mossy]

        # =====================================================================
        # INFERIOR OLIVE → CLIMBING FIBERS (Error Signal)
        # =====================================================================
        # IO neurons generate climbing fiber spikes, synchronized via gap junctions
        # Error signal drives IO activity (high error → depolarization → spike)

        # Apply gap junction coupling to IO neurons
        g_gap_io, E_gap_io = self.gap_junctions_io.forward(self.io_neurons.V_soma)

        # Define additional conductances hook for IO neurons
        def io_get_additional_conductances():
            return [(g_gap_io, E_gap_io)]

        self.io_neurons._get_additional_conductances = io_get_additional_conductances

        io_external = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.purkinje_size,
            filter_by_target_population=CerebellumPopulation.INFERIOR_OLIVE,
        )

        climbing_fiber_spikes, _ = self.io_neurons.forward(
            g_ampa_input=ConductanceTensor(io_external.g_ampa),
            g_nmda_input=None,
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )

        # =====================================================================
        # MOLECULAR LAYER INTERNEURONS (Basket + Stellate)
        # =====================================================================
        # Basket and stellate cells receive feedforward parallel fiber input and
        # inhibit Purkinje soma/dendrites on the next timestep (1-step causal delay).
        # This lateral inhibition is the primary desynchronisation mechanism.

        # Inhibitory conductances to Purkinje from PREVIOUS timestep (causal)
        g_basket_inh = torch.mv(self.basket_purkinje_weights, self._prev_basket_spikes.float())
        g_stellate_inh = torch.mv(self.stellate_purkinje_weights, self._prev_stellate_spikes.float())

        # Current-step basket excitation from parallel fibers + BC→BC lateral inhibition
        basket_exc = torch.mv(self.granule_basket_weights, granule_spikes_float)
        basket_g_ampa, _ = split_excitatory_conductance(basket_exc, nmda_ratio=0.0)
        g_bc_self_inh = torch.mv(self.basket_basket_weights, self._prev_basket_spikes.float())
        basket_spikes, _ = self.basket_neurons.forward(
            g_ampa_input=ConductanceTensor(basket_g_ampa),
            g_nmda_input=None,
            g_gaba_a_input=ConductanceTensor(g_bc_self_inh),
            g_gaba_b_input=None,
        )

        # Current-step stellate excitation from parallel fibers + SC→SC lateral inhibition
        stellate_exc = torch.mv(self.granule_stellate_weights, granule_spikes_float)
        stellate_g_ampa, _ = split_excitatory_conductance(stellate_exc, nmda_ratio=0.0)
        g_sc_self_inh = torch.mv(self.stellate_stellate_weights, self._prev_stellate_spikes.float())
        stellate_spikes, _ = self.stellate_neurons.forward(
            g_ampa_input=ConductanceTensor(stellate_g_ampa),
            g_nmda_input=None,
            g_gaba_a_input=ConductanceTensor(g_sc_self_inh),
            g_gaba_b_input=None,
        )

        g_mli_total = g_basket_inh + g_stellate_inh  # [purkinje_size]

        # =====================================================================
        # GRANULE CELLS → PURKINJE CELLS (Parallel Fibers)
        # =====================================================================
        # Process all Purkinje cells in parallel (vectorized).
        # Climbing fiber spikes from IO neurons trigger complex spikes.
        # MLI inhibition (basket+stellate) is applied at the Purkinje soma.

        # Dendritic integration
        dendrite_input = torch.mv(self.purkinje_synaptic_weights, granule_spikes.float())
        dendrite_noise = torch.randn_like(self.dendrite_voltage).mul_(0.00001)
        dendrite_input_distributed = dendrite_input.unsqueeze(1).expand(-1, config.purkinje_n_dendrites) / config.purkinje_n_dendrites + dendrite_noise
        self.dendrite_voltage.mul_(self.voltage_decay).add_(dendrite_input_distributed)

        # Complex spike detection
        if climbing_fiber_spikes.dim() == 0:
            climbing_fiber_active = (climbing_fiber_spikes > 0.5).expand(self.purkinje_size)
        elif climbing_fiber_spikes.dtype == torch.float32:
            climbing_fiber_active = climbing_fiber_spikes > 0.5
        else:
            climbing_fiber_active = torch.zeros_like(climbing_fiber_spikes, dtype=torch.bool)

        time_since_last = self.complex_spike_timestep - self.last_complex_spike_time
        can_spike = time_since_last > self.complex_spike_refractory_ms
        purkinje_complex_spikes = climbing_fiber_active & can_spike
        self.last_complex_spike_time = torch.where(
            purkinje_complex_spikes,
            torch.tensor(self.complex_spike_timestep, dtype=torch.int32, device=purkinje_complex_spikes.device),
            self.last_complex_spike_time,
        )

        calcium_influx = purkinje_complex_spikes.float().unsqueeze(1).expand(-1, config.purkinje_n_dendrites)
        self.dendrite_calcium = self.calcium_decay * (self.dendrite_calcium + calcium_influx)

        # Simple spike generation
        soma_input = self.dendrite_voltage.sum(dim=1)
        calcium_modulation = 1.0 + 0.2 * self.dendrite_calcium.mean(dim=1)
        soma_conductance = (soma_input * calcium_modulation).clamp(min=0.0)
        # Purkinje pacemaker drive (~47 Hz tonic via resurgent Na⁺ / P/Q-Ca²⁺).
        # Per-cell pacemaker: breaks the shared-drive synchrony
        soma_conductance = soma_conductance + self.pacemaker_g
        soma_g_ampa, soma_g_nmda = split_excitatory_conductance(soma_conductance, nmda_ratio=0.0)
        purkinje_simple_spikes, _ = self.soma_neurons.forward(
            g_ampa_input=ConductanceTensor(soma_g_ampa),
            g_nmda_input=ConductanceTensor(soma_g_nmda),
            g_gaba_a_input=ConductanceTensor(g_mli_total) if g_mli_total is not None else None,
            g_gaba_b_input=None,
        )

        self.complex_spike_timestep += 1

        # =====================================================================
        # PURKINJE + MOSSY COLLATERALS → DCN (Final Output)
        # =====================================================================
        # Biology: DCN receives both Purkinje inhibition and mossy fiber collaterals

        purkinje_conductance = self._integrate_single_synaptic_input(
            SynapseId(
                source_region=self.region_name,
                source_population=CerebellumPopulation.PURKINJE,
                target_region=self.region_name,
                target_population=CerebellumPopulation.DCN,
                receptor_type=ReceptorType.GABA_A,
            ),
            purkinje_simple_spikes,
        ).g_gaba_a

        dcn_total_excitation = self._integrate_single_synaptic_input(
            SynapseId(
                source_region=self.region_name,
                source_population=CerebellumPopulation.MOSSY,
                target_region=self.region_name,
                target_population=CerebellumPopulation.DCN,
                receptor_type=ReceptorType.AMPA,
            ),
            mossy_spikes,
        ).g_ampa

        # DCN spiking: Excitation (mossy + tonic baseline) vs Inhibition (Purkinje)
        # Purkinje provides GABAergic shunting inhibition; baseline drive provides
        # intrinsic pacemaker-like depolarization (approximates I_NaP + I_h currents).
        dcn_total_excitation = dcn_total_excitation + self._dcn_baseline
        dcn_total_g_ampa, dcn_total_g_nmda = split_excitatory_conductance(dcn_total_excitation, nmda_ratio=0.3)

        dcn_spikes, _ = self.dcn_neurons.forward(
            g_ampa_input=ConductanceTensor(dcn_total_g_ampa),
            g_nmda_input=ConductanceTensor(dcn_total_g_nmda),
            g_gaba_a_input=ConductanceTensor(purkinje_conductance),
            g_gaba_b_input=None,
        )

        if not GlobalConfig.LEARNING_DISABLED:
            # ======================================================================
            # APPLY CEREBELLAR LEARNING: Marr-Albus-Ito Rule
            # ======================================================================
            # Biology (Ito 1989):
            #   - Climbing fiber (CF) fires → LTD at co-active parallel fiber (PF) synapses
            #   - PF active, CF silent  → slow normalizing LTP
            #
            # Computation:
            #   dw = ltp_rate × outer(1−cf, pf) − ltd_rate × outer(cf, pf)
            # Shapes: weights [purkinje_size, granule_size]
            climbing_fiber_spikes_float = climbing_fiber_spikes.float()  # [purkinje_size]

            # Apply neuromodulator scaling to MAI learning rates:
            #   - DA boosts LTD (D1/PKA cascade enhances climbing-fiber driven depression)
            #   - NE boosts both LTP and LTD (β-adrenergic: ne_purkinje_gain per-Purkinje)
            effective_ltd_rate = config.mai_ltd_rate * da_ltd_boost
            # ne_purkinje_gain is [purkinje_size]; broadcast with granule_spikes [granule_size]
            ne_gain_col = ne_purkinje_gain.unsqueeze(1)  # [purkinje_size, 1]
            ltd_dw = effective_ltd_rate * ne_gain_col * torch.outer(climbing_fiber_spikes_float, granule_spikes_float)
            ltp_dw = config.mai_ltp_rate * ne_gain_col * torch.outer((1.0 - climbing_fiber_spikes_float).clamp(min=0.0), granule_spikes_float)
            mai_delta = ltp_dw - ltd_dw  # [purkinje_size, granule_size]

            self.purkinje_synaptic_weights.data = clamp_weights(
                weights=self.purkinje_synaptic_weights.data + mai_delta,
                w_min=config.w_min,
                w_max=config.w_max,
                inplace=False,
            )

        region_outputs: RegionOutput = {
            CerebellumPopulation.BASKET: basket_spikes,
            CerebellumPopulation.DCN: dcn_spikes,
            CerebellumPopulation.GRANULE: granule_spikes,
            CerebellumPopulation.GOLGI: golgi_spikes,
            CerebellumPopulation.INFERIOR_OLIVE: climbing_fiber_spikes,
            CerebellumPopulation.PURKINJE: purkinje_simple_spikes,
            CerebellumPopulation.STELLATE: stellate_spikes,
        }

        self._prev_golgi_spikes = golgi_spikes
        self._prev_basket_spikes = basket_spikes
        self._prev_stellate_spikes = stellate_spikes

        return region_outputs

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Propagates dt update to neurons, STP components, and learning strategies.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        super().update_temporal_parameters(dt_ms)
        self.calcium_decay = decay_tensor(dt_ms, 50.0, self.calcium_decay.device)
        self.voltage_decay = decay_tensor(dt_ms, 10.0, self.voltage_decay.device)
