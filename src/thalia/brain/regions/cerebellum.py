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
    heterogeneous_dendrite_coupling,
    heterogeneous_noise_std,
    heterogeneous_tau_adapt,
    split_excitatory_conductance,
)
from thalia.brain.synapses import (
    STPConfig,
    WeightInitializer,
)
from thalia.typing import (
    ConductanceTensor,
    GapJunctionReversal,
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
from thalia.learning import (
    InhibitorySTDPConfig,
    InhibitorySTDPStrategy,
    MaIConfig,
    MaIStrategy,
    STDPConfig,
    STDPStrategy,
)
from thalia.utils import decay_tensor

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
    - **5-HT** (5-HT2A on Purkinje): Enhances Purkinje simple spike
      excitability via Gq/PKC (Strahlendorf et al. 1984).
    """

    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = [
        NeuromodulatorChannel.NE,
        NeuromodulatorChannel.DA_NIGROSTRIATAL,
        NeuromodulatorChannel.ACH,
        NeuromodulatorChannel.SHT,
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
        self.dcn_gaba_size = population_sizes[CerebellumPopulation.DCN_GABA]
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
                tau_mem_ms=heterogeneous_tau_mem(5.0, self.granule_size, device),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(0.85, self.granule_size, device),
                tau_ref=5.0,
                g_L=heterogeneous_g_L(0.05, self.granule_size, device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=2.5,
                tau_I=6.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.08, self.granule_size, device),
                noise_tau_ms=3.0,
                tau_adapt_ms=heterogeneous_tau_adapt(100.0, self.granule_size, device),
                adapt_increment=heterogeneous_adapt_increment(0.15, self.granule_size, device),
                E_adapt=-0.5,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.10, self.granule_size, device),
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
                tau_mem_ms=heterogeneous_tau_mem(10.0, golgi_size, device),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(0.9, golgi_size, device),
                tau_ref=5.0,
                g_L=heterogeneous_g_L(0.05, golgi_size, device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=3.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.08, golgi_size, device),
                noise_tau_ms=3.0,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, golgi_size, device, cv=0.20),
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
        self.golgi_size = golgi_size

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
                tau_mem_ms=heterogeneous_tau_mem(8.0, self.basket_size, device, cv=0.10),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(0.65, self.basket_size, device, cv=0.06),
                tau_ref=1.5,
                g_L=heterogeneous_g_L(0.08, self.basket_size, device, cv=0.08),
                E_E=3.0,
                E_I=-0.5,
                tau_E=2.0,
                tau_I=6.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.08, self.basket_size, device, cv=0.15),
                noise_tau_ms=3.0,
                adapt_increment=0.0,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.basket_size, device, cv=0.20),
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
                tau_mem_ms=heterogeneous_tau_mem(10.0, self.stellate_size, device),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(0.70, self.stellate_size, device),
                tau_ref=1.5,
                g_L=heterogeneous_g_L(0.08, self.stellate_size, device, cv=0.08),
                E_E=3.0,
                E_I=-0.5,
                tau_E=2.5,
                tau_I=7.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.08, self.stellate_size, device, cv=0.15),
                noise_tau_ms=3.0,
                adapt_increment=0.0,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.stellate_size, device, cv=0.20),
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

        # =====================================================================
        # Purkinje cells
        # =====================================================================
        # Single weight matrix [purkinje_size, granule_size]
        # Biology: Each Purkinje cell receives ~200k of ~50B granule cells (<1%).
        # 12% connectivity reduces shared-input overlap (was 20% → 4% pairwise
        # overlap; now 12% → 1.4%) to decorrelate Purkinje firing.
        #
        # Marr-Albus-Ito learning rule (Ito 1989): Climbing fiber error signals
        # drive LTD at co-active parallel fiber synapses; PF-only activity drives
        # slow normalizing LTP.  Neuromodulation: DA boosts LTD (Bhatt 2007),
        # NE enhances both LTP & LTD amplitude (Woodward 1991).
        self._mai_strategy = MaIStrategy(MaIConfig(
            ltd_rate=config.mai_ltd_rate,
            ltp_rate=config.mai_ltp_rate,
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        ))
        self._sid_granule_purkinje = self._add_internal_connection(
            source_population=CerebellumPopulation.GRANULE,
            target_population=CerebellumPopulation.PURKINJE,
            weights=WeightInitializer.sparse_random(
                n_input=self.granule_size,
                n_output=self.purkinje_size,
                connectivity=0.12,
                weight_scale=0.000013,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=None,  # STP NOT applied here — Purkinje dendrite integration is manual
            learning_strategy=self._mai_strategy,
        )
        # Alias for backward compatibility (gap junction init, forward pass mv)
        self.purkinje_synaptic_weights = self.get_synaptic_weights(self._sid_granule_purkinje)

        self.soma_neurons: ConductanceLIF
        self.soma_neurons = self._create_and_register_neuron_population(
            population_name=CerebellumPopulation.PURKINJE,
            n_neurons=self.purkinje_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(20.0, self.purkinje_size, device),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(0.9, self.purkinje_size, device),
                tau_ref=2.0,
                g_L=heterogeneous_g_L(0.10, self.purkinje_size, device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=2.0,
                tau_I=10.0,
                tau_nmda=50.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.02, self.purkinje_size, device),  # Reduced 0.08→0.02:
                # Purkinje cells are intrinsic pacemakers with CV<0.25 (Häusser & Clark 1997).
                # At 0.08 noise with mean pacemaker_g=0.065, noise disrupts metronomic ISI
                # regularity → CV=0.62. Reducing to 0.02 (2.2% of v_threshold) preserves
                # biologically realistic pacemaker CV while still allowing synaptic modulation.
                noise_tau_ms=3.0,
                tau_adapt_ms=heterogeneous_tau_adapt(30.0, self.purkinje_size, device),
                adapt_increment=heterogeneous_adapt_increment(0.10, self.purkinje_size, device),
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.purkinje_size, device, cv=0.30),
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
        # Pacemaker conductance: at 65 Hz target with adapt_increment=0.10, tau_adapt=30ms,
        # the steady-state adaptation current is ~0.195, raising effective threshold to ~1.095.
        # V_inf = g_pace * E_E / (g_L + g_pace) must be just above 1.095:
        #   g_pace=0.065 → V_inf = 0.195/0.165 = 1.18 (above 1.095, ~65 Hz)
        # Higher std (CV≈30%) decorrelates pacemaker phases (ρ < 0.3, Häusser & Clark 1997).
        self.pacemaker_g: torch.Tensor
        self.register_buffer("pacemaker_g", torch.empty(self.purkinje_size, device=device).normal_(mean=0.055, std=0.006).clamp_(0.035, 0.080))
        # Reduced std 0.018→0.006 (CV 32%→11%): with 32% CV, cells had wildly different
        # intrinsic drives → homeostasis couldn't converge all to 65 Hz target (drift=44.6%).
        # 11% CV preserves biological cell-to-cell variability while allowing convergence.
        # Tightened clamp 0.020-0.10 → 0.035-0.080: prevents extreme outliers.

        # Complex spike tracking [purkinje_size]
        self.last_complex_spike_time: torch.Tensor
        self.register_buffer("last_complex_spike_time", torch.full((self.purkinje_size,), -1000, dtype=torch.int32, device=device))
        self.complex_spike_refractory_ms = 100.0  # ~10 Hz max
        self.complex_spike_timestep = 0

        # =====================================================================
        # Deep cerebellar nuclei (final output)
        # =====================================================================
        # Receives both Purkinje inhibition and mossy fiber collaterals

        # Inhibitory STDP for Purkinje→DCN: rebound-dependent plasticity.
        # DCN neurons exhibit post-inhibitory rebound spiking; Purkinje→DCN
        # synapses undergo LTP when Purkinje pauses coincide with DCN rebound
        # bursts, and LTD during tonic Purkinje firing (Aizenman et al. 1998;
        # Pugh & Bhatt 2006; Medina & Mauk 1999).  This slow DCN consolidation
        # complements fast PF→Purkinje MaI learning.
        self._purkinje_dcn_istdp = InhibitorySTDPStrategy(InhibitorySTDPConfig(
            learning_rate=config.learning_rate * 0.1,  # Slow: DCN consolidates over many trials
            tau_istdp=20.0,
            alpha=0.12,
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        ))

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
            # Fixed STP direction: was facilitating (U=0.15, tau_d=60, tau_f=400) — biologically wrong.
            # Purkinje→DCN is a DEPRESSING synapse (Telgkamp & Raman 2002; Pedroarena & Bhatt 2003).
            # Round 1: U=0.30, tau_d=200 caused chronic depletion (x·u=0.046) at Purkinje ~90 Hz.
            # Round 2: U=0.12, tau_d=80 → x_ss=1/(1+0.12*0.09*80)=0.54, x·u=0.064 — depressing
            # but stays above the 0.05 functional threshold at 90 Hz tonic firing.
            stp_config=STPConfig(U=0.12, tau_d=80.0, tau_f=20.0),
            learning_strategy=self._purkinje_dcn_istdp,
        )

        # Mossy fiber → DCN (excitatory collaterals)
        # Mossy fibers send collaterals to DCN before reaching granule cells
        # Plasticity: STDP at mossy→DCN synapses (Pugh & Raman 2006)
        self._mossy_dcn_stdp = STDPStrategy(STDPConfig(
            learning_rate=config.learning_rate * 0.2,
            a_plus=0.003, a_minus=0.0015,
            tau_plus=15.0, tau_minus=15.0,
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        ))
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
            learning_strategy=self._mossy_dcn_stdp,
        )

        # DCN neurons (tonic firing, modulated by inhibition)
        # DCN neurons are spontaneously active (tonic firing ~40-60 Hz)
        self.dcn_neurons: ConductanceLIF
        self.dcn_neurons = self._create_and_register_neuron_population(
            population_name=CerebellumPopulation.DCN,
            n_neurons=self.dcn_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(20.0, self.dcn_size, device),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(1.0, self.dcn_size, device),
                tau_ref=12.0,
                g_L=heterogeneous_g_L(0.10, self.dcn_size, device, cv=0.08),
                E_E=3.0,
                E_I=-0.5,
                tau_E=4.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.08, self.dcn_size, device),
                noise_tau_ms=3.0,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.dcn_size, device, cv=0.25),
            ),
        )

        # DCN GABAergic subpopulation (nucleo-olivary inhibitory projection)
        # Biology: A distinct ~30% subpopulation of DCN neurons are GABAergic
        # and project exclusively to the inferior olive, providing the negative
        # feedback that suppresses error signals once learning converges
        # (De Zeeuw & Berrebi 1995; Fredette & Bhagwandien 1992).
        # Same inputs as glutamatergic DCN (Purkinje inhibition + mossy collaterals).
        self._add_internal_connection(
            source_population=CerebellumPopulation.PURKINJE,
            target_population=CerebellumPopulation.DCN_GABA,
            weights=WeightInitializer.sparse_random(
                n_input=self.purkinje_size,
                n_output=self.dcn_gaba_size,
                connectivity=0.2,
                weight_scale=0.002,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.12, tau_d=80.0, tau_f=20.0),
            learning_strategy=self._purkinje_dcn_istdp,
        )
        self._add_internal_connection(
            source_population=CerebellumPopulation.MOSSY,
            target_population=CerebellumPopulation.DCN_GABA,
            weights=WeightInitializer.sparse_random(
                n_input=self.n_mossy,
                n_output=self.dcn_gaba_size,
                connectivity=0.1,
                weight_scale=0.003,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.15, tau_d=200.0, tau_f=200.0),
            learning_strategy=self._mossy_dcn_stdp,
        )
        self.dcn_gaba_neurons: ConductanceLIF
        self.dcn_gaba_neurons = self._create_and_register_neuron_population(
            population_name=CerebellumPopulation.DCN_GABA,
            n_neurons=self.dcn_gaba_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(20.0, self.dcn_gaba_size, device),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(1.0, self.dcn_gaba_size, device),
                tau_ref=12.0,
                g_L=heterogeneous_g_L(0.10, self.dcn_gaba_size, device, cv=0.08),
                E_E=3.0,
                E_I=-0.5,
                tau_E=4.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.08, self.dcn_gaba_size, device),
                noise_tau_ms=3.0,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.dcn_gaba_size, device, cv=0.25),
            ),
        )

        # IO membrane potential for gap junction coupling
        self._io_membrane: Optional[torch.Tensor] = None

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
                tau_mem_ms=heterogeneous_tau_mem(18.0, self.purkinje_size, device, cv=0.20),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(0.8, self.purkinje_size, device, cv=0.12, clamp_fraction=0.25),
                tau_ref=4.0,
                g_L=heterogeneous_g_L(0.05, self.purkinje_size, device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.08, self.purkinje_size, device),
                noise_tau_ms=3.0,
                tau_adapt_ms=heterogeneous_tau_adapt(150.0, self.purkinje_size, device),
                adapt_increment=heterogeneous_adapt_increment(0.55, self.purkinje_size, device),
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
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.purkinje_size, device, cv=0.25),
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
            coupling_strength=config.gap_junctions.coupling_strength,
            connectivity_threshold=config.gap_junctions.connectivity_threshold,
            max_neighbors=config.gap_junctions.max_neighbors,
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
        self._dcn_gaba_baseline = torch.full((self.dcn_gaba_size,), config.dcn_baseline_drive, device=device)

        # NE β on Purkinje, DA D1 on granule, ACh M2 on Golgi (scalar), 5-HT2A on Purkinje
        self._init_receptors_from_config(device)

        # Ensure all tensors are on the correct device
        self.to(device)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
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
        self._update_receptors(neuromodulator_inputs)

        # Derive scalar modulation factors from mean concentrations
        # NE: per-Purkinje gain ∈ [1.0, 1.5] — β-adrenergic excitability boost
        ne_purkinje_gain = 1.0 + self._ne_concentration * 0.5  # [purkinje_size]
        # DA: scalar LTD amplification ∈ [1.0, 2.0] — D1/PKA potentiates LTD
        da_ltd_boost = 1.0 + float(self._da_concentration.mean().item()) * 2.0
        # ACh: Golgi inhibition scale ∈ [0.5, 1.0] — M2 suppresses Golgi GABA-A
        golgi_inhibition_scale = max(0.5, 1.0 - float(self._ach_concentration.item()) * 0.5)
        # 5-HT: per-Purkinje excitability gain ∈ [1.0, 1.3] — 5-HT2A/PKC
        sht_purkinje_gain = 1.0 + self._sht_concentration * 0.3  # [purkinje_size]

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
        # EXTRACT PREVIOUS-STEP SPIKES (all intra-region connections use prev step)
        # =====================================================================
        prev_golgi = self._prev_spikes(CerebellumPopulation.GOLGI)
        prev_granule = self._prev_spikes(CerebellumPopulation.GRANULE)
        prev_basket = self._prev_spikes(CerebellumPopulation.BASKET)
        prev_stellate = self._prev_spikes(CerebellumPopulation.STELLATE)
        prev_purkinje = self._prev_spikes(CerebellumPopulation.PURKINJE)

        # =====================================================================
        # GRANULE–GOLGI MICROCIRCUIT
        # =====================================================================
        # Golgi → Granule GABA-A inhibition (causal: previous timestep)
        #    golgi_inhibition_scale < 1.0 → ACh muscarinic suppression of Golgi output
        golgi_inh = golgi_inhibition_scale * torch.mv(self.golgi_granule_weights, prev_golgi)

        # Granule cell spiking
        granule_g_ampa, granule_g_nmda = split_excitatory_conductance(mossy_fiber_conductances, nmda_ratio=0.3)
        granule_spikes, _ = self.granule_neurons.forward(
            g_ampa_input=ConductanceTensor(granule_g_ampa),
            g_nmda_input=ConductanceTensor(granule_g_nmda),
            g_gaba_a_input=ConductanceTensor(golgi_inh),
            g_gaba_b_input=None,
        )

        # Granule → Golgi feedforward excitation (prev-step granule spikes)
        golgi_exc = torch.mv(self.granule_golgi_weights, prev_granule)
        golgi_g_ampa, golgi_g_nmda = split_excitatory_conductance(golgi_exc, nmda_ratio=0.0)
        golgi_spikes, _ = self.golgi_neurons.forward(
            g_ampa_input=ConductanceTensor(golgi_g_ampa),
            g_nmda_input=ConductanceTensor(golgi_g_nmda),
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )

        # Mossy fiber collaterals → DCN: downsample prev-step granule activity
        # to the n_mossy count used by the internal Purkinje→DCN weight matrix.
        step = max(1, self.granule_size // self.n_mossy)
        mossy_spikes = prev_granule[::step][:self.n_mossy]  # [n_mossy]

        # =====================================================================
        # INFERIOR OLIVE → CLIMBING FIBERS (Error Signal)
        # =====================================================================
        # IO neurons generate climbing fiber spikes, synchronized via gap junctions
        # Error signal drives IO activity (high error → depolarization → spike)

        # Apply gap junction coupling to IO neurons
        g_gap_io, E_gap_io = self.gap_junctions_io.forward(self.io_neurons.V_soma)

        io_external = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.purkinje_size,
            filter_by_target_population=CerebellumPopulation.INFERIOR_OLIVE,
        )

        climbing_fiber_spikes, _ = self.io_neurons.forward(
            g_ampa_input=ConductanceTensor(io_external.g_ampa),
            g_nmda_input=None,
            g_gaba_a_input=ConductanceTensor(io_external.g_gaba_a),
            g_gaba_b_input=None,
            g_gap_input=ConductanceTensor(g_gap_io),
            E_gap_reversal=GapJunctionReversal(E_gap_io),
        )

        # =====================================================================
        # MOLECULAR LAYER INTERNEURONS (Basket + Stellate)
        # =====================================================================
        # Basket and stellate cells receive feedforward parallel fiber input and
        # inhibit Purkinje soma/dendrites on the next timestep (1-step causal delay).
        # This lateral inhibition is the primary desynchronisation mechanism.

        # Inhibitory conductances to Purkinje from PREVIOUS timestep (causal)
        g_basket_inh = torch.mv(self.basket_purkinje_weights, prev_basket)
        g_stellate_inh = torch.mv(self.stellate_purkinje_weights, prev_stellate)

        # Current-step basket excitation from prev parallel fibers + BC→BC lateral inhibition
        basket_exc = torch.mv(self.granule_basket_weights, prev_granule)
        basket_g_ampa, _ = split_excitatory_conductance(basket_exc, nmda_ratio=0.0)
        g_bc_self_inh = torch.mv(self.basket_basket_weights, prev_basket)
        basket_spikes, _ = self.basket_neurons.forward(
            g_ampa_input=ConductanceTensor(basket_g_ampa),
            g_nmda_input=None,
            g_gaba_a_input=ConductanceTensor(g_bc_self_inh),
            g_gaba_b_input=None,
        )

        # Current-step stellate excitation from prev parallel fibers + SC→SC lateral inhibition
        stellate_exc = torch.mv(self.granule_stellate_weights, prev_granule)
        stellate_g_ampa, _ = split_excitatory_conductance(stellate_exc, nmda_ratio=0.0)
        g_sc_self_inh = torch.mv(self.stellate_stellate_weights, prev_stellate)
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

        # Dendritic integration (prev-step parallel fiber input)
        dendrite_input = torch.mv(self.purkinje_synaptic_weights, prev_granule)
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
        # 5-HT2A modulation: enhanced Purkinje excitability via Gq/PKC
        soma_conductance = soma_conductance * sht_purkinje_gain
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
            prev_purkinje,
        ).g_gaba_a

        dcn_total_excitation = self._integrate_single_synaptic_input(
            SynapseId(
                source_region=self.region_name,
                source_population=CerebellumPopulation.MOSSY,
                target_region=self.region_name,
                target_population=CerebellumPopulation.DCN,
                receptor_type=ReceptorType.AMPA,
            ),
            mossy_spikes,  # already prev-step (derived from prev_granule)
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

        # --- DCN GABAergic subpopulation (nucleo-olivary projection) ---
        # Same inputs as glutamatergic DCN: Purkinje inhibition + mossy excitation + tonic baseline
        dcn_gaba_purkinje_g = self._integrate_single_synaptic_input(
            SynapseId(
                source_region=self.region_name,
                source_population=CerebellumPopulation.PURKINJE,
                target_region=self.region_name,
                target_population=CerebellumPopulation.DCN_GABA,
                receptor_type=ReceptorType.GABA_A,
            ),
            prev_purkinje,
        ).g_gaba_a

        dcn_gaba_mossy_g = self._integrate_single_synaptic_input(
            SynapseId(
                source_region=self.region_name,
                source_population=CerebellumPopulation.MOSSY,
                target_region=self.region_name,
                target_population=CerebellumPopulation.DCN_GABA,
                receptor_type=ReceptorType.AMPA,
            ),
            mossy_spikes,
        ).g_ampa

        dcn_gaba_excitation = dcn_gaba_mossy_g + self._dcn_gaba_baseline
        dcn_gaba_g_ampa, dcn_gaba_g_nmda = split_excitatory_conductance(dcn_gaba_excitation, nmda_ratio=0.3)

        dcn_gaba_spikes, _ = self.dcn_gaba_neurons.forward(
            g_ampa_input=ConductanceTensor(dcn_gaba_g_ampa),
            g_nmda_input=ConductanceTensor(dcn_gaba_g_nmda),
            g_gaba_a_input=ConductanceTensor(dcn_gaba_purkinje_g),
            g_gaba_b_input=None,
        )

        # ======================================================================
        # APPLY CEREBELLAR LEARNING: Marr-Albus-Ito Rule
        # ======================================================================
        # Biology (Ito 1989):
        #   - Climbing fiber (CF) fires → LTD at co-active parallel fiber (PF) synapses
        #   - PF active, CF silent  → slow normalizing LTP
        # Dispatched through the LearningStrategy framework; neuromodulator
        # kwargs are handled by MaIStrategy.compute_update().
        self._apply_learning(
            self._sid_granule_purkinje,
            granule_spikes,
            purkinje_simple_spikes,
            climbing_fiber_spikes=climbing_fiber_spikes,
            da_ltd_boost=da_ltd_boost,
            ne_gain=ne_purkinje_gain,
        )

        region_outputs: RegionOutput = {
            CerebellumPopulation.BASKET: basket_spikes,
            CerebellumPopulation.DCN: dcn_spikes,
            CerebellumPopulation.DCN_GABA: dcn_gaba_spikes,
            CerebellumPopulation.GRANULE: granule_spikes,
            CerebellumPopulation.GOLGI: golgi_spikes,
            CerebellumPopulation.INFERIOR_OLIVE: climbing_fiber_spikes,
            CerebellumPopulation.PURKINJE: purkinje_simple_spikes,
            CerebellumPopulation.STELLATE: stellate_spikes,
        }

        self._apply_all_population_homeostasis(region_outputs)

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
