"""
Cerebellum - Supervised Error-Corrective Learning for Precise Motor Control.

The cerebellum learns through supervised error signals from climbing fibers,
enabling fast, precise learning of input-output mappings without trial-and-error.
"""

from __future__ import annotations

from typing import ClassVar, List, Optional, Tuple, Union

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
    ConductanceLIF,
    NeuronFactory,
    NeuronType,
)
from thalia.brain.synapses import (
    NMReceptorType,
    make_nm_receptor,
    STPConfig,
    STPType,
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
from thalia.utils import clamp_weights, split_excitatory_conductance

from .neural_region import NeuralRegion
from .population_names import CerebellumPopulation
from .region_registry import register_region


class GranuleCellLayer(nn.Module):
    """Granule cell layer: mossy fiber input → sparse parallel fiber expansion.

    Biology: Each granule cell receives 4–5 mossy fiber inputs via claw dendrites.
    Golgi interneurons provide recurrent feedforward and feedback inhibition, enforcing
    the <5% population sparsity required for pattern separation (Marr 1969, Albus 1971).

    Connectivity (causal — one-step delay on Golgi feedback to avoid circular deps):
        Mossy fibers ---AMPA---> Granule cells  (external, wired via synaptic_weights)
        Granule cells --AMPA---> Golgi cells    (feedforward gain control)
        Golgi cells   --GABA_A-> Granule cells  (feedback sparsification, 1-step delay)

    The scalar ``_prev_granule_rate * gain`` approximation used in the pre-P1-4 code
    is replaced here with a proper population of Golgi neurons whose spikes drive
    weight-mediated GABA-A conductances at granule dendrites.
    """

    def __init__(
        self,
        n_granule: int,
        n_golgi: int,
        region_name: str,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ) -> None:
        """Initialise granule / Golgi populations and their weight matrices.

        Args:
            n_granule: Number of granule cells (= parallel fiber count into Purkinje).
            n_golgi: Number of Golgi interneurons.  Biology: ~1 per 5–10 granule cells.
            device: PyTorch device string.
            region_name: Parent region name (used in ConductanceLIFConfig).
        """
        super().__init__()
        self.n_granule = n_granule
        self.n_golgi = n_golgi

        # ------------------------------------------------------------------
        # Granule neurons (fast, excitable; fire briefly in sparse bursts)
        # ------------------------------------------------------------------
        self.granule_neurons = ConductanceLIF(
            n_neurons=n_granule,
            config=ConductanceLIFConfig(
                region_name=region_name,
                population_name=CerebellumPopulation.GRANULE,
                v_threshold=0.85,   # More excitable than pyramidal cells
                v_reset=0.0,
                tau_mem=5.0,        # ms — fast integration (biological ~4–6 ms)
                tau_E=2.5,          # ms — fast AMPA (biological minimum ~2–3 ms)
                tau_I=6.0,          # ms — GABA_A (biological 5–10 ms)
            ),
            device=device,
        )

        # ------------------------------------------------------------------
        # Golgi neurons (large inhibitory interneurons, tonic 5–10 Hz)
        # Receive mossy-fiber and granule-cell excitation; inhibit granule
        # dendrites via GABA-A to enforce sparse coding.
        # ------------------------------------------------------------------
        self.golgi_neurons = ConductanceLIF(
            n_neurons=n_golgi,
            config=ConductanceLIFConfig(
                region_name=region_name,
                population_name="golgi",
                v_threshold=0.9,
                v_reset=0.0,
                tau_mem=10.0,       # ms — slightly slower integration
                tau_E=3.0,          # ms — AMPA
                tau_I=10.0,         # ms — GABA_A
            ),
            device=device,
        )

        # ------------------------------------------------------------------
        # Granule → Golgi  (feedforward excitation, sparse)
        # Biology: ~10% of granule cells contact each Golgi axon
        # ------------------------------------------------------------------
        self.granule_golgi_weights: nn.Parameter = nn.Parameter(
            WeightInitializer.sparse_random(
                n_input=n_granule,
                n_output=n_golgi,
                connectivity=0.10,
                weight_scale=0.002,
                device=device,
            ),
            requires_grad=False,
        )

        # ------------------------------------------------------------------
        # Golgi → Granule  (feedback/feedforward inhibition, broad coverage)
        # Biology: Each Golgi cell inhibits 300–500 granule cells; reach ~30%
        # ------------------------------------------------------------------
        self.golgi_granule_weights: nn.Parameter = nn.Parameter(
            WeightInitializer.sparse_random(
                n_input=n_golgi,
                n_output=n_granule,
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
        self.register_buffer(
            "_prev_golgi_spikes",
            torch.zeros(n_golgi, dtype=torch.bool, device=device),
        )

    @torch.no_grad()
    def forward(self, mf_conductances: torch.Tensor, golgi_inhibition_scale: float = 1.0) -> torch.Tensor:
        """Process mossy fiber input through the granule–Golgi microcircuit.

        Args:
            mf_conductances: Summed mossy-fiber (AMPA) conductance at each granule
                cell dendrite, shape ``[n_granule]``.  Produced by the parent
                ``Cerebellum.forward()`` via ``_integrate_synaptic_inputs_at_dendrites``.
            golgi_inhibition_scale: Multiplicative scale on Golgi→granule GABA-A
                conductance.  ACh (muscarinic M2 on Golgi somata) reduces this
                below 1.0, disinhibiting granule cells.  Default 1.0 (no change).

        Returns:
            granule_spikes: Sparse boolean spike tensor, shape ``[n_granule]``.
        """
        # 1. Golgi → Granule GABA-A inhibition (causal: previous timestep)
        #    golgi_inhibition_scale < 1.0 → ACh muscarinic suppression of Golgi output
        golgi_inh = golgi_inhibition_scale * torch.mv(self.golgi_granule_weights, self._prev_golgi_spikes.float())

        # 2. Granule cell spiking
        g_ampa, g_nmda = split_excitatory_conductance(mf_conductances, nmda_ratio=0.3)
        granule_spikes, _ = self.granule_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa),
            g_nmda_input=ConductanceTensor(g_nmda),
            g_gaba_a_input=ConductanceTensor(golgi_inh),
            g_gaba_b_input=None,
        )

        # 3. Granule → Golgi feedforward excitation (same timestep, Golgi fires later)
        golgi_exc = torch.mv(self.granule_golgi_weights, granule_spikes.float())
        golgi_g_ampa, _ = split_excitatory_conductance(golgi_exc, nmda_ratio=0.0)
        golgi_spikes, _ = self.golgi_neurons.forward(
            g_ampa_input=ConductanceTensor(golgi_g_ampa),
            g_nmda_input=None,
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )

        # 4. Store Golgi spikes for next timestep's feedback inhibition
        self._prev_golgi_spikes = golgi_spikes

        return granule_spikes


class VectorizedPurkinjeLayer(nn.Module):
    """Vectorized Purkinje cell layer for efficient parallel processing.

    Processes all Purkinje cells simultaneously using batched operations.
    Replaces inefficient ModuleList iteration.

    Architecture:
        Input: parallel_fiber_spikes [n_parallel_fibers]
        Weights: synaptic_weights [n_purkinje, n_parallel_fibers]
        Output: purkinje_spikes [n_purkinje]

    Performance:
        - 50x faster than ModuleList approach
        - 58M → ~1M parameters (50x reduction)
        - Vectorized dendrite and calcium dynamics
    """

    def __init__(
        self,
        n_purkinje: int,
        n_parallel_fibers: int,
        n_dendrites: int,
        dt_ms: float,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        """Initialize vectorized Purkinje layer.

        Args:
            n_purkinje: Number of Purkinje cells (100 in default config)
            n_parallel_fibers: Number of parallel fiber inputs (granule cells)
            n_dendrites: Dendrites per cell (for calcium compartments)
            device: torch device
            dt_ms: Timestep duration
        """
        super().__init__()

        self.n_purkinje = n_purkinje
        self.n_parallel_fibers = n_parallel_fibers
        self.n_dendrites = n_dendrites
        self.dt_ms = dt_ms

        # Single weight matrix [n_purkinje, n_parallel_fibers]
        # Biology: Each Purkinje cell receives ~200k parallel fibers (20% connectivity)
        # weight_scale reduced from 0.0008: double-integration amplification (dendritic leaky
        # integrator tau=10ms × tau_E=2ms in soma) creates ~11x gain before NMDA, so
        # 0.0008 was driving g_ampa_ss >> 1.0 → Purkinje firing at 47% (target 4-6%).
        # Registered as nn.Parameter(requires_grad=False) so it appears in state_dict()
        # and moves correctly with .to(device), while being updated manually via .data.
        self.synaptic_weights = nn.Parameter(
            WeightInitializer.sparse_random(
                n_input=n_parallel_fibers,
                n_output=n_purkinje,
                connectivity=0.2,
                weight_scale=0.000013,
                device=device,
            ),
            requires_grad=False,
        )

        self.soma_neurons = ConductanceLIF(
            n_neurons=n_purkinje,
            config=ConductanceLIFConfig(
                region_name="cerebellum",
                population_name=CerebellumPopulation.PURKINJE,
                tau_mem=20.0,
                tau_E=2.0,
                tau_I=10.0,
                tau_nmda=50.0,
                v_threshold=1.0,
                v_reset=0.0,
                E_L=0.0,
                tau_ref=2.0,
                g_L=0.10,
            ),
            device=device,
        )

        # Dendritic voltages [n_purkinje, n_dendrites]
        self.dendrite_voltage: torch.Tensor
        self.register_buffer("dendrite_voltage", torch.zeros(n_purkinje, n_dendrites, device=device))
        # Dendritic calcium [n_purkinje, n_dendrites]
        self.dendrite_calcium: torch.Tensor
        self.register_buffer("dendrite_calcium", torch.zeros(n_purkinje, n_dendrites, device=device))

        # Complex spike tracking [n_purkinje]
        self.last_complex_spike_time: torch.Tensor
        self.register_buffer(
            "last_complex_spike_time",
            torch.full((n_purkinje,), -1000, dtype=torch.int32, device=device),
        )
        self.complex_spike_refractory_ms = 100.0  # ~10 Hz max
        self.timestep = 0

        # Calcium decay factor (tau ~50ms) and voltage decay factor (tau ~10ms).
        # Registered as buffers so .to(device) keeps them on the right device.
        self.calcium_decay: torch.Tensor
        self.voltage_decay: torch.Tensor
        self.register_buffer("calcium_decay", torch.exp(torch.tensor(-dt_ms / 50.0)))
        self.register_buffer("voltage_decay", torch.exp(torch.tensor(-dt_ms / 10.0)))

    @torch.no_grad()
    def forward(
        self,
        parallel_fiber_input: torch.Tensor,  # [n_parallel_fibers] bool
        climbing_fiber_active: torch.Tensor,  # [n_purkinje] bool (or scalar)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process inputs and generate simple/complex spikes for all cells.

        Returns:
            simple_spikes: Regular output spikes [n_purkinje] bool
            complex_spikes: Complex spike occurrence [n_purkinje] bool
        """
        device = parallel_fiber_input.device

        # Dendritic integration
        dendrite_input = torch.mv(self.synaptic_weights, parallel_fiber_input.float())
        dendrite_input_distributed = dendrite_input.unsqueeze(1).expand(-1, self.n_dendrites) / self.n_dendrites
        # noise_std reduced from 0.001: clamp(min=0) rectified bias was dominating signal.
        dendrite_noise = torch.randn_like(self.dendrite_voltage) * 0.00001
        dendrite_input_distributed = dendrite_input_distributed + dendrite_noise
        self.dendrite_voltage = self.voltage_decay * self.dendrite_voltage + dendrite_input_distributed

        # Complex spike detection
        if climbing_fiber_active.dim() == 0:
            climbing_fiber_active = (climbing_fiber_active > 0.5).expand(self.n_purkinje)
        elif climbing_fiber_active.dtype == torch.float32:
            climbing_fiber_active = climbing_fiber_active > 0.5

        time_since_last = self.timestep - self.last_complex_spike_time
        can_spike = time_since_last > self.complex_spike_refractory_ms
        complex_spikes = climbing_fiber_active & can_spike
        self.last_complex_spike_time = torch.where(
            complex_spikes,
            torch.tensor(self.timestep, dtype=torch.int32, device=device),
            self.last_complex_spike_time,
        )

        calcium_influx = complex_spikes.float().unsqueeze(1).expand(-1, self.n_dendrites)
        self.dendrite_calcium = self.calcium_decay * (self.dendrite_calcium + calcium_influx)

        # Simple spike generation
        soma_input = self.dendrite_voltage.sum(dim=1)
        calcium_modulation = 1.0 + 0.2 * self.dendrite_calcium.mean(dim=1)
        soma_conductance = (soma_input * calcium_modulation).clamp(min=0.0)
        # Purkinje pacemaker drive (~60 Hz tonic via resurgent Na⁺ / P/Q-Ca²⁺).
        # g_ampa_pacemaker = 0.030 → V_inf ~1.3 at threshold=1.0; homeostasis tunes rate.
        _PURKINJE_PACEMAKER_G: float = 0.030
        soma_conductance = soma_conductance + _PURKINJE_PACEMAKER_G
        # NMDA omitted: dendritic tau=10ms + soma tau_E=2ms already amplify 11x;
        # adding NMDA (tau=50ms, ×25 additional) would grossly overdrive Purkinje.
        soma_g_ampa, soma_g_nmda = split_excitatory_conductance(soma_conductance, nmda_ratio=0.0)
        simple_spikes, _ = self.soma_neurons.forward(
            g_ampa_input=ConductanceTensor(soma_g_ampa),
            g_nmda_input=ConductanceTensor(soma_g_nmda),
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )

        self.timestep += 1
        return simple_spikes, complex_spikes

    def update_temporal_parameters(self, new_dt_ms: float) -> None:
        """Update temporal parameters when timestep changes."""
        self.dt_ms = new_dt_ms
        device = self.calcium_decay.device
        self.calcium_decay = torch.exp(torch.tensor(-new_dt_ms / 50.0, device=device))
        self.voltage_decay = torch.exp(torch.tensor(-new_dt_ms / 10.0, device=device))


@register_region(
    "cerebellum",
    description="Supervised error-corrective learning via climbing fiber error signals",
    version="1.0",
    author="Thalia Project",
    config_class=CerebellumConfig,
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
        # GranuleCellLayer encapsulates granule neurons and explicit Golgi
        # interneurons (P1-4: replaces scalar _prev_granule_rate approximation).
        n_golgi = max(int(self.granule_size * self.config.golgi_ratio), 10)
        self.granule_layer = GranuleCellLayer(
            n_granule=self.granule_size,
            n_golgi=n_golgi,
            region_name=self.region_name,
            device=device,
        )

        # --------------------------------------------------------------------
        # Purkinje cells (vectorized layer for efficiency)
        # --------------------------------------------------------------------
        # PERFORMANCE: Vectorized implementation ~50x faster than ModuleList
        self.purkinje_layer = VectorizedPurkinjeLayer(
            n_purkinje=self.purkinje_size,
            n_parallel_fibers=self.granule_size,
            n_dendrites=self.config.purkinje_n_dendrites,
            dt_ms=self.config.dt_ms,
            device=device,
        )

        # --------------------------------------------------------------------
        # Deep cerebellar nuclei (final output)
        # --------------------------------------------------------------------
        # Receives both Purkinje inhibition and mossy fiber collaterals

        # Purkinje → DCN (inhibitory, convergent)
        # Many Purkinje cells converge onto each DCN neuron
        # CONDUCTANCE-BASED: Strong inhibition to sculpt DCN output
        self._add_internal_connection(
            source_population=CerebellumPopulation.PURKINJE,
            target_population=CerebellumPopulation.DCN,
            weights=WeightInitializer.sparse_random(
                n_input=self.purkinje_size,
                n_output=self.dcn_size,
                connectivity=0.2,
                weight_scale=0.0008,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            # Biology: Strong Purkinje inhibition shows depression, preventing runaway inhibition
            stp_config=STPConfig.from_type(STPType.DEPRESSING),
        )

        # Mossy fiber → DCN (excitatory collaterals)
        # Mossy fibers send collaterals to DCN before reaching granule cells
        # CONDUCTANCE-BASED: Moderate excitation (Purkinje sculpts final output)
        # Routing key: dcn:cerebellum:mossy (target:region:internal_source)
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
            # Biology: Mossy fiber collaterals show facilitation, reinforcing sustained input
            stp_config=STPConfig.from_type(STPType.FACILITATING),
        )

        # DCN neurons (tonic firing, modulated by inhibition)
        # DCN neurons are spontaneously active (tonic firing ~40-60 Hz)
        # Use NORMALIZED units (threshold=1.0 scale) NOT absolute millivolts
        self.dcn_neurons = ConductanceLIF(
            n_neurons=self.dcn_size,
            config=ConductanceLIFConfig(
                region_name=self.region_name,
                population_name=CerebellumPopulation.DCN,
                v_threshold=1.0,  # Standard normalized threshold
                v_reset=0.0,  # Reset to rest
                E_L=0.0,  # Leak reversal (normalized)
                E_E=3.0,  # Excitatory reversal (normalized, above threshold)
                E_I=-0.5,  # Inhibitory reversal (normalized, hyperpolarizing)
                g_L=0.10,  # Moderate leak conductance
                tau_mem=20.0,  # ms, moderate integration
                tau_E=4.0,  # ms, AMPA kinetics (biological range 2-5ms)
                tau_I=10.0,  # ms, GABA_A kinetics (biological range 5-10ms)
                tau_ref=12.0,  # ms, refractory period (max ~83 Hz ceiling, allows biological 40-60 Hz range)
            ),
            device=device,
        )

        # IO membrane potential for gap junction coupling
        self._io_membrane: Optional[torch.Tensor] = None

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY (Adaptive Gain)
        # =====================================================================
        # EMA tracking of Purkinje firing rates (reserved for future use).
        self._register_homeostasis(CerebellumPopulation.PURKINJE, self.purkinje_size, target_firing_rate=0.045, device=device)

        # Adaptive gains (per neuron) — registered as buffer so .to(device) works.
        self.gain: torch.Tensor
        self.register_buffer("gain", torch.ones(self.purkinje_size, device=device))

        # =====================================================================
        # INFERIOR OLIVE NEURONS (Error Signal Generation)
        # =====================================================================
        # One IO neuron per Purkinje cell (1:1 mapping)
        # IO neurons generate climbing fiber signals (complex spikes) that teach
        # Purkinje cells. Dense gap junction coupling synchronizes IO activity
        # across related cerebellar modules.
        self.io_neurons = NeuronFactory.create(
            region_name=self.region_name,
            population_name=CerebellumPopulation.INFERIOR_OLIVE,
            neuron_type=NeuronType.RELAY,  # Relay neurons have tonic firing suitable for IO baseline activity
            n_neurons=self.purkinje_size,
            device=device,
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
            afferent_weights=self.purkinje_layer.synaptic_weights,  # Use parallel fiber connectivity
            config=gap_junctions_config,
            interneuron_mask=None,  # All IO neurons can be coupled
            device=device,
        )

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(CerebellumPopulation.DCN, self.dcn_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(CerebellumPopulation.GRANULE, self.granule_layer.granule_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(CerebellumPopulation.INFERIOR_OLIVE, self.io_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(CerebellumPopulation.PURKINJE, self.purkinje_layer.soma_neurons, polarity=PopulationPolarity.INHIBITORY)

        # DCN tonic pacemaker baseline (approximates intrinsic I_NaP and I_h currents).
        # Purkinje inhibition sculpts this tonic drive; see CerebellumConfig.dcn_baseline_drive.
        self._dcn_baseline = torch.full((self.dcn_size,), config.dcn_baseline_drive, device=device)

        # =====================================================================
        # NOREPINEPHRINE RECEPTOR (β-adrenergic on Purkinje cells)
        # =====================================================================
        # NE from LC → increased cAMP → PKA activation → enhanced Purkinje
        # intrinsic excitability and potentiated LTP window (Woodward 1991).
        # β-adrenergic (Gs → cAMP → PKA): τ_rise=80 ms, τ_decay=1000 ms.
        self.ne_receptor = make_nm_receptor(
            NMReceptorType.NE_BETA, n_receptors=self.purkinje_size, dt_ms=self.dt_ms, device=device,
            amplitude_scale=1.5,  # Strong β-adrenergic sensitivity on Purkinje (Woodward 1991)
        )
        self._ne_concentration: torch.Tensor
        self.register_buffer("_ne_concentration", torch.zeros(self.purkinje_size, device=device))

        # =====================================================================
        # DOPAMINE RECEPTOR (D1 on granule cells — via SNc projection)
        # =====================================================================
        # DA from SNc enhances cerebellar LTD (parallel fiber → Purkinje weight
        # depression) via PKA/DARPP-32 cascade on granule somata (Bhatt 2007).
        # D1 (Gs → cAMP → PKA): τ_rise=500 ms, τ_decay=8000 ms.
        self.da_receptor = make_nm_receptor(
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
        self.ach_receptor = make_nm_receptor(
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
          which suppresses Golgi→granule GABA-A conductance via
          ``GranuleCellLayer.forward(golgi_inhibition_scale=...))``.
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
        golgi_scale = max(0.5, 1.0 - float(self._ach_concentration.item()) * 0.5)

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
        # STAGE 1a: GRANULE–GOLGI MICROCIRCUIT  (P1-4)
        # =====================================================================
        # GranuleCellLayer replaces the former scalar Golgi approximation and
        # hard top-k truncation.  Explicit Golgi interneurons receive granule
        # excitation and feed back GABA-A inhibition, enforcing <5% sparsity
        # through recurrent dynamics (Marr 1969; Eccles, Ito & Szentágothai 1967).
        # Pass ACh-derived golgi_scale: ACh suppresses Golgi → granule disinhibition.
        granule_spikes = self.granule_layer.forward(mossy_fiber_conductances, golgi_inhibition_scale=golgi_scale)

        # Mossy fiber collaterals → DCN: downsample granule activity
        # to the n_mossy count used by the internal Purkinje→DCN weight matrix.
        step = max(1, self.granule_size // self.n_mossy)
        mossy_spikes = granule_spikes[::step][:self.n_mossy]  # [n_mossy]

        # =====================================================================
        # STAGE 2: INFERIOR OLIVE → CLIMBING FIBERS (Error Signal)
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
        # STAGE 3: GRANULE CELLS → PURKINJE CELLS (Parallel Fibers)
        # =====================================================================
        # Process all Purkinje cells in parallel (vectorized)
        # Climbing fiber spikes from IO neurons trigger complex spikes
        purkinje_simple_spikes, _purkinje_complex_spikes = self.purkinje_layer.forward(
            parallel_fiber_input=granule_spikes,
            climbing_fiber_active=climbing_fiber_spikes,
        )  # [purkinje_size]

        # =====================================================================
        # STAGE 4: PURKINJE + MOSSY COLLATERALS → DCN (Final Output)
        # =====================================================================
        # Biology: DCN receives both Purkinje inhibition and mossy fiber collaterals

        purkinje_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CerebellumPopulation.PURKINJE,
            target_region=self.region_name,
            target_population=CerebellumPopulation.DCN,
            receptor_type=ReceptorType.GABA_A,
        )
        purkinje_conductance = self._integrate_synaptic_inputs_at_dendrites({purkinje_synapse: purkinje_simple_spikes}, n_neurons=self.dcn_size).g_gaba_a

        mossy_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CerebellumPopulation.MOSSY,
            target_region=self.region_name,
            target_population=CerebellumPopulation.DCN,
            receptor_type=ReceptorType.AMPA,
        )
        dcn_total_excitation = self._integrate_synaptic_inputs_at_dendrites({mossy_synapse: mossy_spikes}, n_neurons=self.dcn_size).g_ampa

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

        # ======================================================================
        # APPLY CEREBELLAR LEARNING: Marr-Albus-Ito Rule
        # ======================================================================
        # Biology (Ito 1989):
        #   - Climbing fiber (CF) fires → LTD at co-active parallel fiber (PF) synapses
        #   - PF active, CF silent  → slow normalizing LTP
        #
        # Computation:
        #   dw = ltp_rate × outer(1−cf, pf) − ltd_rate × outer(cf, pf)
        # Shapes: weights [n_purkinje, n_granule]
        pre_f = granule_spikes.float()  # [n_granule]
        cf_f = climbing_fiber_spikes.float()  # [n_purkinje]

        # Apply neuromodulator scaling to MAI learning rates:
        #   - DA boosts LTD (D1/PKA cascade enhances climbing-fiber driven depression)
        #   - NE boosts both LTP and LTD (β-adrenergic: ne_purkinje_gain per-Purkinje)
        effective_ltd_rate = config.mai_ltd_rate * da_ltd_boost
        # ne_purkinje_gain is [purkinje_size]; broadcast with pre_f [granule_size]
        ne_gain_col = ne_purkinje_gain.unsqueeze(1)  # [purkinje_size, 1]
        ltd_dw = effective_ltd_rate * ne_gain_col * torch.outer(cf_f, pre_f)
        ltp_dw = config.mai_ltp_rate * ne_gain_col * torch.outer((1.0 - cf_f).clamp(min=0.0), pre_f)
        mai_delta = ltp_dw - ltd_dw  # [n_purkinje, n_granule]

        self.purkinje_layer.synaptic_weights.data = clamp_weights(
            weights=self.purkinje_layer.synaptic_weights.data + mai_delta,
            w_min=config.w_min,
            w_max=config.w_max,
            inplace=False,
        )

        region_outputs: RegionOutput = {
            CerebellumPopulation.DCN: dcn_spikes,
            CerebellumPopulation.GRANULE: granule_spikes,
            CerebellumPopulation.INFERIOR_OLIVE: climbing_fiber_spikes,
            CerebellumPopulation.PURKINJE: purkinje_simple_spikes,
        }

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
        self.purkinje_layer.update_temporal_parameters(dt_ms)
