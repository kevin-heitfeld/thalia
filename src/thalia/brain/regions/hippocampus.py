"""
Trisynaptic Hippocampus - Biologically-Accurate DG→CA3→CA1 Episodic Memory Circuit.

This implements the classic hippocampal trisynaptic circuit for episodic memory:
- **Dentate Gyrus (DG)**: Pattern SEPARATION via sparse coding (~2-5% active)
- **CA3**: Pattern COMPLETION via recurrent connections (autoassociative memory)
- **CA1**: Output/comparison layer detecting match vs mismatch

**Key Biological Features**:
===========================
1. **THETA MODULATION** (6-10 Hz oscillations):
   - Theta trough (0-Ï€): Encoding phase (CA3 learning enabled)
   - Theta peak (Ï€-2Ï€): Retrieval phase (comparison active)
   - Phase separation prevents interference between encoding and retrieval

2. **FEEDFORWARD INHIBITION**:
   - Stimulus onset triggers transient inhibition
   - Naturally clears residual activity
   - Fast-spiking interneuron-like dynamics

3. **CONTINUOUS DYNAMICS**:
   - Everything flows naturally
   - Membrane potentials decay via LIF dynamics
   - Theta phase advances continuously
   - Smooth transitions between encoding and retrieval phases
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from thalia import GlobalConfig
from thalia.brain.configs import HippocampusConfig
from thalia.brain.neurons import NeuronFactory
from thalia.brain.synapses import (
    NMReceptorType,
    make_nm_receptor,
    WeightInitializer,
)
from thalia.brain.synapses.stp import (
    CA3_RECURRENT_PRESET,
    MOSSY_FIBER_PRESET,
    PV_BASKET_PRESET,
    SCHAFFER_COLLATERAL_PRESET,
)
from thalia.learning import (
    TagAndCaptureConfig,
    TagAndCaptureStrategy,
    ThreeFactorConfig,
    ThreeFactorStrategy,
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
from thalia.utils import (
    CircularDelayBuffer,
    clamp_weights,
    compute_ach_recurrent_suppression,
    compute_ne_gain,
    split_excitatory_conductance,
)

from .hippocampus_inhibitory_network import HippocampalInhibitoryNetwork
from .neural_region import NeuralRegion
from .population_names import HippocampusPopulation
from .region_registry import register_region
from .stimulus_gating import StimulusGating


@register_region(
    "hippocampus",
    aliases=["trisynaptic", "trisynaptic_hippocampus"],
    description="DG→CA3→CA1 trisynaptic circuit with theta-modulated encoding/retrieval and episodic memory",
    version="1.0",
    author="Thalia Project",
    config_class=HippocampusConfig,
)
class Hippocampus(NeuralRegion[HippocampusConfig]):
    """Biologically-accurate hippocampus with DG→CA3→CA1 trisynaptic circuit."""

    # Mesolimbic DA (VTA → hippocampus) modulates Schaffer collateral LTP and replay.
    # NE from LC modulates novelty-driven encoding and theta power.
    # ACh from nucleus basalis gates encoding vs. retrieval modes (nicotinic receptors).
    # 'ach_septal': slow muscarinic (M1) modulation from medial septum cholinergic neurons.
    # 'ach_septal': slow muscarinic (M1) modulation from medial septum cholinergic neurons.
    # '5ht': 5-HT2C receptors on CA3 pyramidals suppress recurrent NMDA drive under
    #         aversive/stress states (Bhagya et al. 2022; Barnes & Sharp 1999).
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = [
        NeuromodulatorChannel.DA_MESOLIMBIC,
        NeuromodulatorChannel.NE,
        NeuromodulatorChannel.ACH,
        NeuromodulatorChannel.ACH_SEPTAL,
        NeuromodulatorChannel.SHT,
    ]

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        config: HippocampusConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        """Initialize trisynaptic hippocampus."""
        super().__init__(config, population_sizes, region_name, device=device)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.dg_size = population_sizes[HippocampusPopulation.DG]
        self.ca3_size = population_sizes[HippocampusPopulation.CA3]
        self.ca2_size = population_sizes[HippocampusPopulation.CA2]
        self.ca1_size = population_sizes[HippocampusPopulation.CA1]

        # =====================================================================
        # INTERNAL STATE VARIABLES (for dynamics and plasticity)
        # =====================================================================
        # Previous encoding/retrieval modulation (from OLM dynamics)
        # Used to maintain causal flow: t-1 OLM activity determines t encoding/retrieval
        # These are simple scalar state variables (not tensors), no buffers needed
        self._prev_encoding_mod: float = 0.5  # Default: balanced state

        # NMDA trace for temporal integration (slow kinetics); None until first forward step
        self.nmda_trace: Optional[torch.Tensor]
        self.register_buffer("nmda_trace", None)

        # Spontaneous replay (sharp-wave ripple) detection.
        # ripple_detected: True on every timestep within the SWR window (not just onset).
        # ripple_duration_steps: countdown in timesteps; refreshed on each high-rate CA3 burst.
        self.ripple_detected: bool = False
        self.ripple_duration_steps: int = 0
        # CA3 attractor pattern captured at ripple onset — re-injected each timestep of the
        # window to sustain multi-timestep replay (Buzsaki 1989; Wilson & McNaughton 1994).
        self._swr_ca3_pattern: Optional[torch.Tensor]
        self.register_buffer("_swr_ca3_pattern", None)

        # CA1 prediction signal: match vs. mismatch between CA3 stored pattern and EC input
        # match_level: fraction of CA1 neurons co-driven by BOTH CA3 Schaffer AND EC direct
        #   → recognition/familiarity signal (suppresses VTA novelty response)
        # mismatch_level: EC input not predicted by CA3
        #   → novelty/prediction-error signal; propagates via Subiculum → VTA DA burst
        # Ref: Kumaran & Maguire 2007; Lisman & Grace 2005 (Hippocampal-VTA loop)
        self.ca1_match_level: float = 0.0
        self.ca1_mismatch_level: float = 0.0

        # =====================================================================
        # HIPPOCAMPAL EXCITATORY NEURONS (LIF with adaptation for sparse coding)
        # =====================================================================
        # Create LIF neurons for each layer using factory functions
        # DG: Sparse coding requires high threshold
        # v_threshold raised 0.9 → 1.6 → 1.8: at 0.9 fired 10 Hz; at 1.6 fired 2.9 Hz
        # (run-06). Target is 0â€“1 Hz (sparse pattern separation). Only the highest-
        # weighted tail neurons should fire; 1.8 further reduces the active fraction.
        # Strong SFA (adapt_increment=0.30, tau_adapt=120ms) prevents sustained firing.
        self.dg_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=self.region_name,
            population_name=HippocampusPopulation.DG,
            n_neurons=self.dg_size,
            device=device,
            v_threshold=config.dg.v_threshold,
            adapt_increment=config.dg.adapt_increment,
            tau_adapt=config.dg.tau_adapt,
        )
        # CA3 gets spike-frequency adaptation to prevent frozen attractors
        # Two-compartment: basal dendrites receive DG mossy fibers + recurrents;
        # apical compartment is available for future EC direct-path feedback.
        self.ca3_neurons = NeuronFactory.create_pyramidal_two_compartment(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA3,
            n_neurons=self.ca3_size,
            device=device,
            v_threshold=config.ca3.v_threshold,
            adapt_increment=config.ca3.adapt_increment,
            tau_adapt=config.ca3.tau_adapt,
        )
        # CA2: Social memory and temporal context - moderate threshold for selectivity
        # Reduced from 1.6 (caused near-silence) → 1.1: slightly above CA3 (1.0) for selectivity
        self.ca2_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA2,
            n_neurons=self.ca2_size,
            device=device,
            v_threshold=config.ca2.v_threshold,
            adapt_increment=config.ca2.adapt_increment,
            tau_adapt=config.ca2.tau_adapt,
        )
        # CA1: Output layer – two-compartment pyramidal (CA3 Schaffer collateral basal, EC direct apical)
        self.ca1_neurons = NeuronFactory.create_pyramidal_two_compartment(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA1,
            n_neurons=self.ca1_size,
            device=device,
            v_threshold=config.ca1.v_threshold,
            adapt_increment=config.ca1.adapt_increment,
            tau_adapt=config.ca1.tau_adapt,
        )

        # =====================================================================
        # HIPPOCAMPAL INHIBITORY NETWORKS (with OLM cells for emergent theta)
        # =====================================================================
        # DG inhibitory network: Minimal inhibition for pattern separation
        # Moderate at 0.20 to prevent avalanches while maintaining sparse coding
        # v_threshold_bistratified=1.00: threshold 0.9 → 2.1 Hz (HIGH, target 0â€“1 Hz);
        # 1.10â€“1.30 → 0 Hz (stochastic silence in short windows); 1.00 aims for ~0.3â€“0.8 Hz.
        self.dg_inhibitory = HippocampalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=HippocampusPopulation.DG_INHIBITORY,
            pyr_size=self.dg_size,
            total_inhib_fraction=config.dg.total_inhib_fraction,
            v_threshold_olm=config.dg.v_threshold_olm,
            v_threshold_bistratified=config.dg.v_threshold_bistratified,
            dt_ms=config.dt_ms,
            device=device,
        )

        # CA3 inhibitory network: Moderate inhibition for pattern completion stability
        # INHIBITION REDUCED: 0.65 → 0.40 → 0.25 (further reduced: CA3 at 0.1 Hz, needs more excitation)
        # OLM/bistratified thresholds lowered vs DG: at sparse CA3 firing (0.75â€“2 Hz) the
        # pyramidal→OLM V_infâ‰ˆ0.18â€“0.45; DG-level thresholds (1.0/0.9) are unreachable.
        self.ca3_inhibitory = HippocampalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA3_INHIBITORY,
            pyr_size=self.ca3_size,
            total_inhib_fraction=config.ca3.total_inhib_fraction,
            v_threshold_olm=config.ca3.v_threshold_olm,
            v_threshold_bistratified=config.ca3.v_threshold_bistratified,
            dt_ms=config.dt_ms,
            device=device,
        )

        # CA2 inhibitory network: Social/temporal context processing
        self.ca2_inhibitory = HippocampalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA2_INHIBITORY,
            pyr_size=self.ca2_size,
            total_inhib_fraction=config.ca2.total_inhib_fraction,
            v_threshold_olm=config.ca2.v_threshold_olm,
            v_threshold_bistratified=config.ca2.v_threshold_bistratified,
            dt_ms=config.dt_ms,
            device=device,
        )

        # CA1 inhibitory network: PV, OLM, Bistratified cells
        # OLM cells phase-lock to septal GABA → emergent encoding/retrieval
        self.ca1_inhibitory = HippocampalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA1_INHIBITORY,
            pyr_size=self.ca1_size,
            total_inhib_fraction=config.ca1.total_inhib_fraction,
            v_threshold_olm=config.ca1.v_threshold_olm,
            v_threshold_bistratified=config.ca1.v_threshold_bistratified,
            dt_ms=config.dt_ms,
            device=device,
        )

        # Stimulus gating module (transient inhibition at stimulus changes)
        self.stimulus_gating = StimulusGating(
            n_neurons=self.dg_size,
            threshold=config.ffi_threshold,
            max_inhibition=config.ffi_strength * 10.0,  # Scale to appropriate range
            decay_rate=1.0 - (1.0 / config.ffi_tau),  # Convert tau to rate
            steepness=10.0,
            device=device,
        )

        # =====================================================================
        # GAP JUNCTIONS (Electrical Synapses) - Config Setup
        # =====================================================================
        # Override weights with trisynaptic circuit weights
        self._init_circuit_weights(device)

        # =====================================================================
        # TEMPORAL STATE BUFFERS (Delay Buffers for All State)
        # =====================================================================
        # Spike state buffers for all subregions
        self._dg_spike_buffer = CircularDelayBuffer(1, self.dg_size, device, torch.bool)
        self._ca3_spike_buffer = CircularDelayBuffer(1, self.ca3_size, device, torch.bool)
        self._ca2_spike_buffer = CircularDelayBuffer(1, self.ca2_size, device, torch.bool)
        self._ca1_spike_buffer = CircularDelayBuffer(1, self.ca1_size, device, torch.bool)

        # Inhibitory population spike buffers (1-step delay for I→I conductances)
        # The parent computes I→I conductances from t-1 inhibitory spikes using the
        # registered pv_to_pv / olm_to_pv weight matrices (with STP), then passes
        # the resulting GABA_A conductances into each inhibitory network's forward().
        self._dg_pv_buffer = CircularDelayBuffer(1, self.dg_inhibitory.n_pv, device, torch.bool)
        self._dg_olm_buffer = CircularDelayBuffer(1, self.dg_inhibitory.n_olm, device, torch.bool)
        self._dg_bistratified_buffer = CircularDelayBuffer(1, self.dg_inhibitory.n_bistratified, device, torch.bool)
        self._ca3_pv_buffer = CircularDelayBuffer(1, self.ca3_inhibitory.n_pv, device, torch.bool)
        self._ca3_olm_buffer = CircularDelayBuffer(1, self.ca3_inhibitory.n_olm, device, torch.bool)
        self._ca3_bistratified_buffer = CircularDelayBuffer(1, self.ca3_inhibitory.n_bistratified, device, torch.bool)
        self._ca2_pv_buffer = CircularDelayBuffer(1, self.ca2_inhibitory.n_pv, device, torch.bool)
        self._ca2_olm_buffer = CircularDelayBuffer(1, self.ca2_inhibitory.n_olm, device, torch.bool)
        self._ca2_bistratified_buffer = CircularDelayBuffer(1, self.ca2_inhibitory.n_bistratified, device, torch.bool)
        self._ca1_pv_buffer = CircularDelayBuffer(1, self.ca1_inhibitory.n_pv, device, torch.bool)
        self._ca1_olm_buffer = CircularDelayBuffer(1, self.ca1_inhibitory.n_olm, device, torch.bool)
        self._ca1_bistratified_buffer = CircularDelayBuffer(1, self.ca1_inhibitory.n_bistratified, device, torch.bool)

        # =====================================================================
        # INTER-LAYER AXONAL DELAYS (using CircularDelayBuffer utility)
        # =====================================================================
        # Create delay buffers for biological signal propagation within circuit
        # DG→CA3 delay: Mossy fiber transmission (~3ms biologically)
        # CA3→CA2 delay: Short proximity-based delay (~2ms biologically)
        # CA2→CA1 delay: Short proximity-based delay (~2ms biologically)
        # CA3→CA1 delay: Schaffer collateral transmission (~3ms biologically, direct bypass)

        # Initialize CircularDelayBuffer for each pathway
        dg_ca3_delay_steps = int(config.dg_to_ca3_delay_ms / config.dt_ms)
        ca3_ca3_delay_steps = int(config.ca3_to_ca3_delay_ms / config.dt_ms)
        ca3_ca2_delay_steps = int(config.ca3_to_ca2_delay_ms / config.dt_ms)
        ca2_ca1_delay_steps = int(config.ca2_to_ca1_delay_ms / config.dt_ms)
        ca3_ca1_delay_steps = int(config.ca3_to_ca1_delay_ms / config.dt_ms)

        # Store delay steps for conditional checks
        self._dg_ca3_delay_steps = dg_ca3_delay_steps
        self._ca3_ca3_delay_steps = ca3_ca3_delay_steps
        self._ca3_ca2_delay_steps = ca3_ca2_delay_steps
        self._ca2_ca1_delay_steps = ca2_ca1_delay_steps
        self._ca3_ca1_delay_steps = ca3_ca1_delay_steps

        self._dg_ca3_buffer = CircularDelayBuffer(
            max_delay=dg_ca3_delay_steps,
            size=self.dg_size,
            device=device,
            dtype=torch.bool,
        )
        self._ca3_ca3_buffer = CircularDelayBuffer(
            max_delay=ca3_ca3_delay_steps,
            size=self.ca3_size,
            device=device,
            dtype=torch.bool,
        )
        self._ca3_ca2_buffer = CircularDelayBuffer(
            max_delay=ca3_ca2_delay_steps,
            size=self.ca3_size,
            device=device,
            dtype=torch.bool,
        )
        self._ca3_ca1_buffer = CircularDelayBuffer(
            max_delay=ca3_ca1_delay_steps,
            size=self.ca3_size,
            device=device,
            dtype=torch.bool,
        )
        self._ca2_ca1_buffer = CircularDelayBuffer(
            max_delay=ca2_ca1_delay_steps,
            size=self.ca2_size,
            device=device,
            dtype=torch.bool,
        )

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY
        # =====================================================================
        # Per-subregion homeostatic state (firing-rate EMA + synaptic scaling).
        # Neurons registered below; looked up lazily in _update_homeostasis().
        # use_synaptic_scaling=True: Turrigiano & Nelson (2004) – multiplicative
        # upscaling of all afferent weights when the population is chronically silent.
        self._register_homeostasis(HippocampusPopulation.DG,  self.dg_size,  target_firing_rate=0.001, device=device)  # <1 Hz (sparse pattern separator)
        self._register_homeostasis(HippocampusPopulation.CA3, self.ca3_size, target_firing_rate=0.003, device=device)  # 1â€“5 Hz
        self._register_homeostasis(HippocampusPopulation.CA2, self.ca2_size, target_firing_rate=0.003, device=device)  # 1â€“5 Hz
        self._register_homeostasis(HippocampusPopulation.CA1, self.ca1_size, target_firing_rate=0.003, device=device)  # 1â€“5 Hz

        # =====================================================================
        # DOPAMINE RECEPTOR (minimal 10% VTA projection)
        # =====================================================================
        # Hippocampus receives minimal DA innervation for novelty/salience modulation
        # Primarily affects CA1 output and CA3 consolidation
        # Biological: VTA DA enhances LTP in novelty-detecting neurons
        total_neurons = self.dg_size + self.ca3_size + self.ca2_size + self.ca1_size
        # D1 receptors dominate HPC (Otmakhova & Lisman 1996); D1 Gs/PKA cascade:
        # Ï„_rise=500 ms, Ï„_decay=8000 ms – DA acts as a long-lasting novelty/LTP gate.
        self.da_receptor = make_nm_receptor(
            NMReceptorType.DA_D1, n_receptors=total_neurons, dt_ms=self.dt_ms, device=device,
            amplitude_scale=6.0,  # HPC receives dense DA; preserve legacy signal strength
        )
        # Per-subregion DA concentration buffers (0.5 = tonic baseline)
        self._da_concentration_dg = torch.full((self.dg_size,), 0.5, device=device)
        self._da_concentration_ca3 = torch.full((self.ca3_size,), 0.5, device=device)
        self._da_concentration_ca2 = torch.full((self.ca2_size,), 0.5, device=device)
        self._da_concentration_ca1 = torch.full((self.ca1_size,), 0.5, device=device)

        # =====================================================================
        # NOREPINEPHRINE RECEPTOR (LC projection for arousal/novelty)
        # =====================================================================
        # Hippocampus receives dense NE innervation from LC
        # NE modulates novelty detection and arousal-dependent memory formation
        # Î±1-adrenergic (Gq): arousal / novelty modulation, Ï„_rise=10 ms, Ï„_decay=150 ms.
        self.ne_receptor = make_nm_receptor(
            NMReceptorType.NE_ALPHA1, n_receptors=total_neurons, dt_ms=self.dt_ms, device=device
        )
        # Per-subregion NE concentration buffers
        self._ne_concentration_dg = torch.zeros(self.dg_size, device=device)
        self._ne_concentration_ca3 = torch.zeros(self.ca3_size, device=device)
        self._ne_concentration_ca2 = torch.zeros(self.ca2_size, device=device)
        self._ne_concentration_ca1 = torch.zeros(self.ca1_size, device=device)

        # =====================================================================
        # ACETYLCHOLINE RECEPTOR (NB projection for encoding/retrieval)
        # =====================================================================
        # Hippocampus receives strong ACh innervation from nucleus basalis
        # ACh controls encoding vs retrieval modes (Hasselmo 1999):
        # - High ACh → encoding mode (suppress recurrence, enhance feedforward)
        # - Low ACh → retrieval mode (enable pattern completion, consolidation)
        # NB projects via nicotinic nAChR (ionotropic, fast: Ï„_rise=3 ms, Ï„_decay=15 ms).
        # Controls encoding vs retrieval mode (Hasselmo 1999).
        self.ach_receptor = make_nm_receptor(
            NMReceptorType.ACH_NICOTINIC, n_receptors=total_neurons, dt_ms=self.dt_ms, device=device,
            amplitude_scale=1.3,  # Preserve legacy ~0.2 effective amplitude
        )
        # Per-subregion ACh concentration buffers
        self._ach_concentration_dg = torch.zeros(self.dg_size, device=device)
        self._ach_concentration_ca3 = torch.zeros(self.ca3_size, device=device)
        self._ach_concentration_ca2 = torch.zeros(self.ca2_size, device=device)
        self._ach_concentration_ca1 = torch.zeros(self.ca1_size, device=device)

        # =====================================================================
        # SEPTAL ACETYLCHOLINE RECEPTOR (muscarinic M1, from medial septum)
        # =====================================================================
        # Septal ACh projects to CA1 via volume transmission, acting on M1-type
        # muscarinic receptors. Effects:
        #   - Enhances NMDA NR2B subunit conductance (slower, CaÂ²âº-permeable)
        #   - Modulates Schaffer collateral LTP threshold (encoding gate)
        # Kinetics: Ï„_rise=20ms, Ï„_decay=300ms (much slower than nicotinic NB ACh)
        # Muscarinic M1 (Gq → PLC/IP3): slow cascade, Ï„_rise=100 ms, Ï„_decay=1500 ms.
        # Phase-locked to theta; boosts NMDA NR2B insertion in CA1 at encoding peaks.
        self.ach_septal_receptor = make_nm_receptor(
            NMReceptorType.ACH_MUSCARINIC_M1, n_receptors=self.ca1_size, dt_ms=self.dt_ms, device=device
        )
        self._ach_septal_concentration_ca1 = torch.zeros(self.ca1_size, device=device)

        # =====================================================================
        # SEROTONIN RECEPTOR (5-HT2C on CA3 pyramidals – from DRN)
        # =====================================================================
        # Biology: 5-HT2C receptors on CA3 somata suppress recurrent NMDA
        # conductance, attenuating pattern completion and Schaffer collateral
        # LTP during aversive / high-serotonin states (Barnes & Sharp 1999).
        # Kinetics: tau_rise ~8 ms, tau_decay ~100 ms (SERT reuptake).
        # 5-HT2C (Gq → PLC): fast metabotropic, Ï„_rise=8 ms, Ï„_decay=100 ms.
        self.sht_receptor_ca3 = make_nm_receptor(
            NMReceptorType.SHT_2C, n_receptors=self.ca3_size, dt_ms=self.dt_ms, device=device
        )
        self._sht_concentration_ca3: torch.Tensor
        self.register_buffer("_sht_concentration_ca3", torch.zeros(self.ca3_size, device=device))

        # =====================================================================
        # LEARNING STRATEGY (Tag-and-Capture wrapping Three-Factor Learning)
        # =====================================================================
        # TagAndCaptureStrategy wraps ThreeFactorStrategy:
        # - compute_update: applies three-factor rule (eligibility Ã— DA Ã— lr)
        #   AND updates the tag matrix as a side effect (no extra calls needed).
        # - consolidate(): DA-gated capture that permanently strengthens tagged
        #   synapses when dopamine is elevated.
        # - tags tensor: readable for spontaneous replay prioritisation.
        self._tag_and_capture_strategy = TagAndCaptureStrategy(
            base_strategy=ThreeFactorStrategy(ThreeFactorConfig(
                learning_rate=0.001,  # Conservative rate for stable learning
                eligibility_tau=100.0,  # Eligibility trace decay (ms)
                modulator_tau=50.0,  # Modulator (dopamine) decay (ms)
            )),
            config=TagAndCaptureConfig(
                tag_decay=0.95,        # ~20 step tag lifetime at 1 ms dt
                tag_threshold=0.1,     # Minimum post-spike to create tag
                consolidation_lr_scale=0.5,  # Half of base lr for capture
                consolidation_da_threshold=0.1,
            ),
        )
        # Eagerly allocate the tags buffer (CA3 recurrent: square matrix)
        self._tag_and_capture_strategy.setup(
            n_pre=self.ca3_size,
            n_post=self.ca3_size,
            device=torch.device(device),
        )

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(HippocampusPopulation.DG, self.dg_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(HippocampusPopulation.CA3, self.ca3_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(HippocampusPopulation.CA2, self.ca2_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(HippocampusPopulation.CA1, self.ca1_neurons, polarity=PopulationPolarity.EXCITATORY)

        self._register_neuron_population(HippocampusPopulation.DG_INHIBITORY_PV, self.dg_inhibitory.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(HippocampusPopulation.DG_INHIBITORY_OLM, self.dg_inhibitory.olm_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(HippocampusPopulation.DG_INHIBITORY_BISTRATIFIED, self.dg_inhibitory.bistratified_neurons, polarity=PopulationPolarity.INHIBITORY)

        self._register_neuron_population(HippocampusPopulation.CA3_INHIBITORY_PV, self.ca3_inhibitory.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(HippocampusPopulation.CA3_INHIBITORY_OLM, self.ca3_inhibitory.olm_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(HippocampusPopulation.CA3_INHIBITORY_BISTRATIFIED, self.ca3_inhibitory.bistratified_neurons, polarity=PopulationPolarity.INHIBITORY)

        self._register_neuron_population(HippocampusPopulation.CA2_INHIBITORY_PV, self.ca2_inhibitory.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(HippocampusPopulation.CA2_INHIBITORY_OLM, self.ca2_inhibitory.olm_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(HippocampusPopulation.CA2_INHIBITORY_BISTRATIFIED, self.ca2_inhibitory.bistratified_neurons, polarity=PopulationPolarity.INHIBITORY)

        self._register_neuron_population(HippocampusPopulation.CA1_INHIBITORY_PV, self.ca1_inhibitory.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(HippocampusPopulation.CA1_INHIBITORY_OLM, self.ca1_inhibitory.olm_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(HippocampusPopulation.CA1_INHIBITORY_BISTRATIFIED, self.ca1_inhibitory.bistratified_neurons, polarity=PopulationPolarity.INHIBITORY)

        # Ensure all tensors are on the correct device
        self.to(device)

    def _init_circuit_weights(self, device: Union[str, torch.device]) -> None:
        """Initialize internal circuit weights.

        Internal circuit weights (DG→CA3, CA3→CA1, etc.) are initialized here.
        """
        # =====================================================================
        # INTERNAL WEIGHTS
        # =====================================================================
        # DG → CA3: Random but less sparse (mossy fibers)
        # Biology: "Detonator synapses" with powerful transmission.
        # Weight scale raised 0.0001 → 0.003 (30Ã—): with MOSSY_FIBER_PRESET
        # initial release probability U=0.01, the effective per-spike weight
        # was ~1e-6 at rest – functionally silent.  Detonator synapses are
        # among the strongest in the brain; the high base weight compensates
        # for the low resting release probability and ensures CA3 responds
        # to DG bursts once the facilitating STP builds during input epochs.
        self._add_internal_connection(
            source_population=HippocampusPopulation.DG,
            target_population=HippocampusPopulation.CA3,
            weights=WeightInitializer.sparse_random(
                n_input=self.dg_size,
                n_output=self.ca3_size,
                connectivity=0.15,
                weight_scale=0.003,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            # Mossy Fibers (DG→CA3): Strong facilitation
            stp_config=MOSSY_FIBER_PRESET.configure(),
        )

        # CA3 → CA3 RECURRENT: Autoassociative memory weights
        # Learning: One-shot Hebbian with fast/slow traces and heterosynaptic LTD
        # Weight scale raised 0.00001 → 0.0005 (50Ã—): at 0.00001 the max weight
        # was 0.00001 (diagnostics-confirmed), making pattern completion
        # impossible.  CA3 recurrent collaterals are among the strongest
        # synapses in hippocampus; this value allows autoassociative dynamics
        # while remaining well below the runaway-excitation regime.
        # STP: CA3_RECURRENT_PRESET – moderate-strong depression (Dobrunz &
        # Stevens 1999; Fioravante & Regehr 2011). Depression at theta
        # frequencies limits runaway excitation during pattern completion.
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA3,
            target_population=HippocampusPopulation.CA3,
            weights=WeightInitializer.sparse_random_no_autapses(
                n_input=self.ca3_size,
                n_output=self.ca3_size,
                connectivity=0.05,
                weight_scale=0.0007,  # Raised (0.0005→0.0007): strengthen CA3 recurrent collaterals
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=CA3_RECURRENT_PRESET.configure(),
        )

        # Create sparse local inhibition: each neuron inhibits nearby neurons
        # Biologically: basket cells have local axonal arbors (~200-300Î¼m radius)
        # We approximate this with random sparse connectivity
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA3,
            target_population=HippocampusPopulation.CA3,
            weights=WeightInitializer.sparse_random_no_autapses(
                n_input=self.ca3_size,
                n_output=self.ca3_size,
                connectivity=0.2,
                weight_scale=0.0001,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=PV_BASKET_PRESET.configure(),
        )

        # Initialize CA2 lateral inhibition weights
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA2,
            target_population=HippocampusPopulation.CA2,
            weights=WeightInitializer.sparse_random_no_autapses(
                n_input=self.ca2_size,
                n_output=self.ca2_size,
                connectivity=0.2,
                weight_scale=0.0005,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=PV_BASKET_PRESET.configure(),
        )

        # =====================================================================
        # CA2 PATHWAYS: Social memory and temporal context
        # =====================================================================
        # CA3 → CA2: Weak plasticity (stability mechanism)
        # CA2 is resistant to CA3 pattern completion interference
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA3,
            target_population=HippocampusPopulation.CA2,
            weights=WeightInitializer.sparse_random(
                n_input=self.ca3_size,
                n_output=self.ca2_size,
                connectivity=0.3,
                weight_scale=0.0006,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=SCHAFFER_COLLATERAL_PRESET.configure(),
        )

        # CA2 → CA1: Output to decision layer
        # Provides temporal/social context to CA1 processing
        # STP: SCHAFFER_COLLATERAL_PRESET – CA2→CA1 axons are Schaffer-like
        # collaterals (moderate depression), not mossy fibers (Zhao et al. 2007;
        # Lee et al. 2014). MOSSY_FIBER_PRESET was incorrect here.
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA2,
            target_population=HippocampusPopulation.CA1,
            weights=WeightInitializer.sparse_random(
                n_input=self.ca2_size,
                n_output=self.ca1_size,
                connectivity=0.2,
                weight_scale=0.0005,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=SCHAFFER_COLLATERAL_PRESET.configure(),
        )

        # CA3 → CA1: Feedforward (retrieved memory)
        # This is the DIRECT bypass pathway (Schaffer collaterals)
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA3,
            target_population=HippocampusPopulation.CA1,
            weights=WeightInitializer.sparse_random(
                n_input=self.ca3_size,
                n_output=self.ca1_size,
                connectivity=0.15,
                weight_scale=0.0015,  # Raised (0.0008→0.0015): Schaffer collaterals must overcome tonic inhibition at CA1
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=SCHAFFER_COLLATERAL_PRESET.configure(),
        )

        # CA1 lateral inhibition for competition
        # Use sparse lateral inhibition (similar to CA3 basket cells)
        # Biologically: CA1 interneurons have local connectivity, not all-to-all
        ca1_ca1_inhib_synapse = self._add_internal_connection(
            source_population=HippocampusPopulation.CA1,
            target_population=HippocampusPopulation.CA1,
            weights=WeightInitializer.sparse_random_no_autapses(
                n_input=self.ca1_size,
                n_output=self.ca1_size,
                connectivity=0.2,
                weight_scale=0.0005,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=PV_BASKET_PRESET.configure(),
        )

        # =====================================================================
        # HIPPOCAMPAL INHIBITORY NETWORK SYNAPTIC CONNECTIONS
        # =====================================================================
        # All E→I, I→E, I→I weight matrices are registered here so they
        # participate in the standard STP, diagnostic, and learning-strategy
        # pipeline.  Weight values match those previously hard-coded inside
        # HippocampalInhibitoryNetwork (now a pure neuron container).
        #
        # STP assignments (literature):
        #   Pyr→PV   AMPA:  PV_BASKET_PRESET   (strong depression, Geiger et al. 1997)
        #   Pyr→OLM  AMPA:  SCHAFFER_COLLATERAL (moderate depression; no facilitating preset)
        #   PV→Pyr   GABA_A: PV_BASKET_PRESET   (strong depression, Kraushaar & Jonas 2000)
        #   PV→Pyr   GABA_B: None               (GABA_B recruits via spill-over; no vesicular STP)
        #   OLM→Pyr  GABA_A: SCHAFFER_COLLATERAL (Cea-del Rio et al. 2011: moderate)
        #   PV→PV    GABA_A: PV_BASKET_PRESET   (strong depression, Bennett & Bhatt 2012)
        #   OLM→PV   GABA_A: SCHAFFER_COLLATERAL (Oliva et al. 2000: moderate)
        # =====================================================================

        # Pre-compute Pyr→PV weight maximums (biologically-constrained conductance budget)
        _pv_w_max_dg = 2.0 * 0.02317 / (0.001 * 5.0 * self.dg_size * 0.5)
        _pv_w_max_ca3 = 2.0 * 0.02317 / (0.002 * 5.0 * self.ca3_size * 0.5)
        _pv_w_max_ca2 = 2.0 * 0.02317 / (0.002 * 5.0 * self.ca2_size * 0.5)
        _pv_w_max_ca1 = 2.0 * 0.02317 / (0.002 * 5.0 * self.ca1_size * 0.5)

        for subregion, pyr_pop, pv_pop, olm_pop, bist_pop, pyr_size, pv_w_max_ei, pv_ie, pv_ie_wmax in [
            (
                "DG",
                HippocampusPopulation.DG,
                HippocampusPopulation.DG_INHIBITORY_PV,
                HippocampusPopulation.DG_INHIBITORY_OLM,
                HippocampusPopulation.DG_INHIBITORY_BISTRATIFIED,
                self.dg_size,
                _pv_w_max_dg,
                0.3,   # PV→Pyr connectivity
                0.005, # PV→Pyr GABA_A w_max
            ),
            (
                "CA3",
                HippocampusPopulation.CA3,
                HippocampusPopulation.CA3_INHIBITORY_PV,
                HippocampusPopulation.CA3_INHIBITORY_OLM,
                HippocampusPopulation.CA3_INHIBITORY_BISTRATIFIED,
                self.ca3_size,
                _pv_w_max_ca3,
                0.7,   # PV→Pyr connectivity (strong: controls runaway recurrence)
                0.015, # PV→Pyr GABA_A w_max
            ),
            (
                "CA2",
                HippocampusPopulation.CA2,
                HippocampusPopulation.CA2_INHIBITORY_PV,
                HippocampusPopulation.CA2_INHIBITORY_OLM,
                HippocampusPopulation.CA2_INHIBITORY_BISTRATIFIED,
                self.ca2_size,
                _pv_w_max_ca2,
                0.5,   # PV→Pyr connectivity
                0.012, # PV→Pyr GABA_A w_max
            ),
            (
                "CA1",
                HippocampusPopulation.CA1,
                HippocampusPopulation.CA1_INHIBITORY_PV,
                HippocampusPopulation.CA1_INHIBITORY_OLM,
                HippocampusPopulation.CA1_INHIBITORY_BISTRATIFIED,
                self.ca1_size,
                _pv_w_max_ca1,
                0.5,   # PV→Pyr connectivity (supports theta modulations)
                0.015, # PV→Pyr GABA_A w_max
            ),
        ]:
            # Retrieve interneuron sizes from the corresponding inhibitory network
            inhib_net = getattr(self, f"{subregion.lower()}_inhibitory")
            n_pv = inhib_net.n_pv
            n_olm = inhib_net.n_olm
            n_bist = inhib_net.n_bistratified

            # ------------------------------------------------------------------
            # E → I  (Pyramidal → Interneurons)
            # ------------------------------------------------------------------
            self._add_internal_connection(
                source_population=pyr_pop,
                target_population=pv_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=pyr_size, n_output=n_pv,
                    connectivity=0.5, w_min=0.0, w_max=pv_w_max_ei, device=device,
                ),
                receptor_type=ReceptorType.AMPA,
                stp_config=PV_BASKET_PRESET.configure(),
            )
            self._add_internal_connection(
                source_population=pyr_pop,
                target_population=olm_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=pyr_size, n_output=n_olm,
                    connectivity=0.3, w_min=0.0, w_max=0.015, device=device,
                ),
                receptor_type=ReceptorType.AMPA,
                stp_config=SCHAFFER_COLLATERAL_PRESET.configure(),  # closest to facilitating
            )
            self._add_internal_connection(
                source_population=pyr_pop,
                target_population=bist_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=pyr_size, n_output=n_bist,
                    connectivity=0.35, w_min=0.0, w_max=0.012, device=device,
                ),
                receptor_type=ReceptorType.AMPA,
                stp_config=PV_BASKET_PRESET.configure(),
            )

            # ------------------------------------------------------------------
            # I → E  (Interneurons → Pyramidal)
            # ------------------------------------------------------------------
            self._add_internal_connection(
                source_population=pv_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=n_pv, n_output=pyr_size,
                    connectivity=pv_ie, w_min=0.0, w_max=pv_ie_wmax, device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=PV_BASKET_PRESET.configure(),
            )
            self._add_internal_connection(
                source_population=pv_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=n_pv, n_output=pyr_size,
                    connectivity=pv_ie, w_min=0.0, w_max=pv_ie_wmax * 0.15, device=device,
                ),
                receptor_type=ReceptorType.GABA_B,
                stp_config=None,  # GABA_B recruits via spill-over; vesicular STP does not apply
            )
            self._add_internal_connection(
                source_population=olm_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=n_olm, n_output=pyr_size,
                    connectivity=0.5, w_min=0.0, w_max=0.001, device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=SCHAFFER_COLLATERAL_PRESET.configure(),
            )
            self._add_internal_connection(
                source_population=bist_pop,
                target_population=pyr_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=n_bist, n_output=pyr_size,
                    connectivity=0.55, w_min=0.0, w_max=0.002, device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=PV_BASKET_PRESET.configure(),
            )

            # ------------------------------------------------------------------
            # I → I  (Lateral inhibition within inhibitory network)
            # ------------------------------------------------------------------
            self._add_internal_connection(
                source_population=pv_pop,
                target_population=pv_pop,
                weights=WeightInitializer.sparse_uniform_no_autapses(
                    n_input=n_pv, n_output=n_pv,
                    connectivity=0.3, w_min=0.0, w_max=0.0005, device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=PV_BASKET_PRESET.configure(),
            )
            self._add_internal_connection(
                source_population=olm_pop,
                target_population=pv_pop,
                weights=WeightInitializer.sparse_uniform(
                    n_input=n_olm, n_output=n_pv,
                    connectivity=0.2, w_min=0.0, w_max=0.0004, device=device,
                ),
                receptor_type=ReceptorType.GABA_A,
                stp_config=SCHAFFER_COLLATERAL_PRESET.configure(),
            )

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _run_hippocampal_inhibitory(
        self,
        inhib_net: HippocampalInhibitoryNetwork,
        prev_pyr_spikes: torch.Tensor,
        prev_pv_spikes_buf: CircularDelayBuffer,
        prev_olm_spikes_buf: CircularDelayBuffer,
        prev_bist_spikes_buf: CircularDelayBuffer,
        pyr_pop: str,
        pv_pop: str,
        olm_pop: str,
        bist_pop: str,
        septal_gaba: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Compute one inhibitory network step using registered STP-weighted connections.

        Args:
            inhib_net: The inhibitory network to run (neuron container + gap junctions).
            prev_pyr_spikes: Previous-step pyramidal spikes [pyr_size] for E→I drive.
            prev_pv_spikes_buf: 1-step buffer of PV spikes for I→I OLM→PV computation.
            prev_olm_spikes_buf: 1-step buffer of OLM spikes for I→I OLM→PV computation.
            prev_bist_spikes_buf: 1-step buffer for bistratified spikes (currently unused in I→I).
            pyr_pop / pv_pop / olm_pop / bist_pop: Population names registered in this region.
            septal_gaba: Optional septal GABA spike tensor [100] for OLM phase-locking.

        Returns:
            dict with keys: perisomatic_gaba_a, perisomatic_gaba_b, dendritic, olm_dendritic,
                            pv_spikes, olm_spikes, bistratified_spikes
        """
        rn = self.region_name

        def _syn(src_pop: str, tgt_pop: str, rxn: ReceptorType) -> SynapseId:
            return SynapseId(rn, src_pop, rn, tgt_pop, receptor_type=rxn)

        # ------------------------------------------------------------------
        # E → I  (Pyr → interneurons, STP-modulated)
        # ------------------------------------------------------------------
        syn_pyr_pv = _syn(pyr_pop, pv_pop, ReceptorType.AMPA)
        syn_pyr_olm = _syn(pyr_pop, olm_pop, ReceptorType.AMPA)
        syn_pyr_bist = _syn(pyr_pop, bist_pop, ReceptorType.AMPA)

        pv_g_exc = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={syn_pyr_pv: prev_pyr_spikes},
            n_neurons=inhib_net.n_pv,
        ).g_ampa
        olm_g_exc = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={syn_pyr_olm: prev_pyr_spikes},
            n_neurons=inhib_net.n_olm,
        ).g_ampa
        bist_g_exc = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs={syn_pyr_bist: prev_pyr_spikes},
            n_neurons=inhib_net.n_bistratified,
        ).g_ampa

        # ------------------------------------------------------------------
        # I → I  (PV→PV lateral + OLM→PV, from prev-step inhibitory spikes)
        # ------------------------------------------------------------------
        syn_pv_pv = _syn(pv_pop, pv_pop, ReceptorType.GABA_A)
        syn_olm_pv = _syn(olm_pop, pv_pop, ReceptorType.GABA_A)

        pv_g_inh = (
            self._integrate_synaptic_inputs_at_dendrites(
                synaptic_inputs={syn_pv_pv: prev_pv_spikes_buf.read(1)},
                n_neurons=inhib_net.n_pv,
            ).g_ampa
            + self._integrate_synaptic_inputs_at_dendrites(
                synaptic_inputs={syn_olm_pv: prev_olm_spikes_buf.read(1)},
                n_neurons=inhib_net.n_pv,
            ).g_ampa
        )

        # ------------------------------------------------------------------
        # Septal GABA → OLM  (external input, no STP – held in inhib_net)
        # ------------------------------------------------------------------
        if septal_gaba is not None:
            olm_g_inh = inhib_net.septal_to_olm @ septal_gaba.float()
        else:
            olm_g_inh = torch.zeros(inhib_net.n_olm, device=self.device)
        bist_g_inh = torch.zeros(inhib_net.n_bistratified, device=self.device)

        # ------------------------------------------------------------------
        # Run interneurons
        # ------------------------------------------------------------------
        inhib_out = inhib_net.forward(
            pv_g_exc=pv_g_exc, pv_g_inh=pv_g_inh,
            olm_g_exc=olm_g_exc, olm_g_inh=olm_g_inh,
            bistratified_g_exc=bist_g_exc, bistratified_g_inh=bist_g_inh,
        )
        pv_spikes = inhib_out["pv_spikes"]
        olm_spikes = inhib_out["olm_spikes"]
        bist_spikes = inhib_out["bistratified_spikes"]

        # ------------------------------------------------------------------
        # I → E  (interneurons → pyramidal, STP-modulated)
        # ------------------------------------------------------------------
        syn_pv_pyr_a = _syn(pv_pop, pyr_pop, ReceptorType.GABA_A)
        syn_pv_pyr_b = _syn(pv_pop, pyr_pop, ReceptorType.GABA_B)
        syn_olm_pyr = _syn(olm_pop, pyr_pop, ReceptorType.GABA_A)
        syn_bist_pyr = _syn(bist_pop, pyr_pop, ReceptorType.GABA_A)

        perisomatic_gaba_a = self._integrate_single_synaptic_input(syn_pv_pyr_a, pv_spikes).g_gaba_a
        perisomatic_gaba_b = self._integrate_single_synaptic_input(syn_pv_pyr_b, pv_spikes).g_gaba_b
        olm_dendritic = self._integrate_single_synaptic_input(syn_olm_pyr, olm_spikes).g_gaba_a
        bist_dendritic = self._integrate_single_synaptic_input(syn_bist_pyr, bist_spikes).g_gaba_a

        dendritic = olm_dendritic + bist_dendritic

        # ------------------------------------------------------------------
        # Store inhibitory spikes for next-step I→I computation
        # ------------------------------------------------------------------
        prev_pv_spikes_buf.write_and_advance(pv_spikes)
        prev_olm_spikes_buf.write_and_advance(olm_spikes)
        prev_bist_spikes_buf.write_and_advance(bist_spikes)

        return {
            "perisomatic_gaba_a": perisomatic_gaba_a,
            "perisomatic_gaba_b": perisomatic_gaba_b,
            "dendritic": dendritic,
            "olm_dendritic": olm_dendritic,
            "pv_spikes": pv_spikes,
            "olm_spikes": olm_spikes,
            "bistratified_spikes": bist_spikes,
        }

    def _step(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Process input spikes through the DG→CA3→CA2→CA1 trisynaptic circuit.

        The full forward pass is delegated to eight focused phase methods:

        1. :meth:`_process_neuromodulators` – update concentration buffers
        2. :meth:`_extract_circuit_inputs`  – integrate synaptic inputs + compute gating
        3. :meth:`_step_dg`                 – pattern separation (Dentate Gyrus)
        4. :meth:`_step_ca3`                – pattern completion + recurrence (CA3)
        5. :meth:`_step_ca2`                – temporal / social context (CA2)
        6. :meth:`_step_ca1`                – memory output layer (CA1)
        7. :meth:`_update_match_mismatch`   – familiarity / prediction-error signals
        8. :meth:`_apply_plasticity`        – Hebbian + three-factor weight updates
        9. :meth:`_update_spike_buffers`    – advance all delay buffers

        **Sharp-wave ripple (SWR) replay path:** When CA3 population rate exceeds
        ``config.ripple_threshold``, a replay window opens (``ripple_detected=True``,
        length ``config.ripple_duration_max_ms``).  The CA3 attractor pattern captured
        at onset is re-injected each timestep (``config.ripple_replay_injection``),
        sustaining CA1 activation via boosted Schaffer collaterals
        (``config.ripple_boost_factor``).  The replayed CA1 spikes propagate
        downstream through Subiculum → EC_V → cortex, implementing offline
        memory consolidation (Buzsaki 1989; Wilson & McNaughton 1994).

        Args:
            synaptic_inputs: Point-to-point synaptic connections from cortex/thalamus.
            neuromodulator_inputs: Broadcast neuromodulatory signals (DA, NE, ACh, 5-HT).
        """
        self._process_neuromodulators(neuromodulator_inputs)

        dg_input, ca3_input, ca1_input, septal_gaba, ffi_factor, encoding_mod, retrieval_mod = (
            self._extract_circuit_inputs(synaptic_inputs)
        )

        dg_spikes,  dg_inhib_output  = self._step_dg(dg_input, ffi_factor, septal_gaba)
        ca3_spikes, ca3_inhib_output = self._step_ca3(ca3_input, septal_gaba)
        ca2_spikes, ca2_inhib_output = self._step_ca2(septal_gaba)
        ca1_spikes, ca1_inhib_output, ca1_basal_g_exc, ca1_apical_g_exc = (
            self._step_ca1(ca1_input, ffi_factor, encoding_mod, retrieval_mod, septal_gaba)
        )

        self._update_match_mismatch(ca1_basal_g_exc, ca1_apical_g_exc, retrieval_mod)

        if not GlobalConfig.LEARNING_DISABLED:
            self._apply_plasticity(
                synaptic_inputs, dg_spikes, ca3_spikes, ca2_spikes, ca1_spikes, encoding_mod
            )

        region_outputs: RegionOutput = {
            HippocampusPopulation.DG:  dg_spikes,
            HippocampusPopulation.CA3: ca3_spikes,
            HippocampusPopulation.CA2: ca2_spikes,
            HippocampusPopulation.CA1: ca1_spikes,
            # Inhibitory populations (for diagnostics tracking)
            HippocampusPopulation.DG_INHIBITORY_PV:           dg_inhib_output["pv_spikes"],
            HippocampusPopulation.DG_INHIBITORY_OLM:          dg_inhib_output["olm_spikes"],
            HippocampusPopulation.DG_INHIBITORY_BISTRATIFIED: dg_inhib_output["bistratified_spikes"],
            HippocampusPopulation.CA3_INHIBITORY_PV:           ca3_inhib_output["pv_spikes"],
            HippocampusPopulation.CA3_INHIBITORY_OLM:          ca3_inhib_output["olm_spikes"],
            HippocampusPopulation.CA3_INHIBITORY_BISTRATIFIED: ca3_inhib_output["bistratified_spikes"],
            HippocampusPopulation.CA2_INHIBITORY_PV:           ca2_inhib_output["pv_spikes"],
            HippocampusPopulation.CA2_INHIBITORY_OLM:          ca2_inhib_output["olm_spikes"],
            HippocampusPopulation.CA2_INHIBITORY_BISTRATIFIED: ca2_inhib_output["bistratified_spikes"],
            HippocampusPopulation.CA1_INHIBITORY_PV:           ca1_inhib_output["pv_spikes"],
            HippocampusPopulation.CA1_INHIBITORY_OLM:          ca1_inhib_output["olm_spikes"],
            HippocampusPopulation.CA1_INHIBITORY_BISTRATIFIED: ca1_inhib_output["bistratified_spikes"],
        }

        self._apply_all_population_homeostasis(region_outputs)
        self._update_spike_buffers(dg_spikes, ca3_spikes, ca2_spikes, ca1_spikes)

        return region_outputs

    # =========================================================================
    # PHASE METHODS (decomposed from _step)
    # =========================================================================

    def _process_neuromodulators(self, neuromodulator_inputs: NeuromodulatorInput) -> None:
        """Update all neuromodulator concentration buffers from incoming spikes."""
        if GlobalConfig.NEUROMODULATION_DISABLED:
            return

        # DOPAMINE (from VTA mesolimbic pathway)
        # Hippocampus receives ~10% DA innervation for novelty / salience gating.
        vta_da_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.DA_MESOLIMBIC)
        da_concentration_full = self.da_receptor.update(vta_da_spikes)
        # Phasic DA (spikes) adds to tonic baseline (0.5).
        self._da_concentration_dg  = 0.5 + da_concentration_full[: self.dg_size] * 1.0
        self._da_concentration_ca3 = 0.5 + da_concentration_full[self.dg_size : self.dg_size + self.ca3_size] * 1.0
        self._da_concentration_ca2 = 0.5 + da_concentration_full[self.dg_size + self.ca3_size : self.dg_size + self.ca3_size + self.ca2_size] * 1.0
        self._da_concentration_ca1 = 0.5 + da_concentration_full[self.dg_size + self.ca3_size + self.ca2_size :] * 1.0

        # NOREPINEPHRINE (from Locus Coeruleus) – arousal / novelty gain
        lc_ne_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.NE)
        ne_concentration_full = self.ne_receptor.update(lc_ne_spikes)
        self._ne_concentration_dg  = ne_concentration_full[: self.dg_size]
        self._ne_concentration_ca3 = ne_concentration_full[self.dg_size : self.dg_size + self.ca3_size]
        self._ne_concentration_ca2 = ne_concentration_full[self.dg_size + self.ca3_size : self.dg_size + self.ca3_size + self.ca2_size]
        self._ne_concentration_ca1 = ne_concentration_full[self.dg_size + self.ca3_size + self.ca2_size :]

        # ACETYLCHOLINE – nicotinic nAChR (from Nucleus Basalis)
        # High ACh → encoding mode (suppresses CA3 recurrence)
        nb_ach_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.ACH)
        ach_concentration_full = self.ach_receptor.update(nb_ach_spikes)
        self._ach_concentration_dg  = ach_concentration_full[: self.dg_size]
        self._ach_concentration_ca3 = ach_concentration_full[self.dg_size : self.dg_size + self.ca3_size]
        self._ach_concentration_ca2 = ach_concentration_full[self.dg_size + self.ca3_size : self.dg_size + self.ca3_size + self.ca2_size]
        self._ach_concentration_ca1 = ach_concentration_full[self.dg_size + self.ca3_size + self.ca2_size :]

        # ACETYLCHOLINE – muscarinic M1 mAChR (from Medial Septum)
        # Slow kinetics (Ï„_decay=300 ms) gate NMDA NR2B insertion in CA1.
        septal_ach_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.ACH_SEPTAL)
        self._ach_septal_concentration_ca1 = self.ach_septal_receptor.update(septal_ach_spikes)

        # SEROTONIN – 5-HT2C (from Dorsal Raphe Nucleus, targets CA3)
        # High 5-HT: suppresses CA3 recurrent NMDA and Schaffer LTP.
        sht_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.SHT)
        self._sht_concentration_ca3 = self.sht_receptor_ca3.update(sht_spikes)

    def _extract_circuit_inputs(
        self, synaptic_inputs: SynapticInput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], float, float, float]:
        """Integrate all multi-source synaptic inputs and compute gating signals.

        Returns:
            ``(dg_input, ca3_input, ca1_input, septal_gaba,
               ffi_factor, encoding_mod, retrieval_mod)``
        """
        dg_input = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, n_neurons=self.dg_size,
            filter_by_target_population=HippocampusPopulation.DG,
        ).g_ampa

        ca3_input = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, n_neurons=self.ca3_size,
            filter_by_target_population=HippocampusPopulation.CA3,
        ).g_ampa

        # TODO: Add direct EC->CA2 input integration here

        ca1_input = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs, n_neurons=self.ca1_size,
            filter_by_target_population=HippocampusPopulation.CA1,
        ).g_ampa

        # Septal GABAergic input for OLM theta phase-locking
        septal_gaba: Optional[torch.Tensor] = synaptic_inputs.get("septal_gaba", None)
        if septal_gaba is not None and septal_gaba.numel() == 0:
            septal_gaba = None

        # Encoding/retrieval modulation from PREVIOUS timestep's OLM state (causal)
        encoding_mod = self._prev_encoding_mod
        retrieval_mod = 1.0 - self._prev_encoding_mod

        # Stimulus-onset feedforward inhibition (FFI)
        ffi = self.stimulus_gating.compute(dg_input, return_tensor=False)
        raw_ffi   = ffi.item() if hasattr(ffi, "item") else float(ffi)
        ffi_strength = min(1.0, raw_ffi / self.stimulus_gating.max_inhibition)
        ffi_factor   = 1.0 - ffi_strength * self.config.ffi_strength

        return dg_input, ca3_input, ca1_input, septal_gaba, ffi_factor, encoding_mod, retrieval_mod

    def _step_dg(
        self,
        dg_input: Any,
        ffi_factor: float,
        septal_gaba: Optional[Any],
    ) -> tuple[Any, dict]:
        """Process one timestep of the Dentate Gyrus (pattern separation).

        Returns:
            ``(dg_spikes, dg_inhib_output)``
        """
        # Apply FFI gating then clamp to non-negative conductance
        dg_g_exc = F.relu(dg_input * ffi_factor)

        dg_inhib_output = self._run_hippocampal_inhibitory(
            inhib_net=self.dg_inhibitory,
            prev_pyr_spikes=self._dg_spike_buffer.read(1),
            prev_pv_spikes_buf=self._dg_pv_buffer,
            prev_olm_spikes_buf=self._dg_olm_buffer,
            prev_bist_spikes_buf=self._dg_bistratified_buffer,
            pyr_pop=HippocampusPopulation.DG,
            pv_pop=HippocampusPopulation.DG_INHIBITORY_PV,
            olm_pop=HippocampusPopulation.DG_INHIBITORY_OLM,
            bist_pop=HippocampusPopulation.DG_INHIBITORY_BISTRATIFIED,
            septal_gaba=septal_gaba,
        )
        dg_perisomatic_inhib = dg_inhib_output["perisomatic_gaba_a"]

        dg_g_ampa, dg_g_nmda = split_excitatory_conductance(dg_g_exc, nmda_ratio=0.05)  # Reduced from 0.2

        dg_spikes, _ = self.dg_neurons.forward(
            g_ampa_input=ConductanceTensor(dg_g_ampa),
            g_nmda_input=ConductanceTensor(dg_g_nmda),
            g_gaba_a_input=ConductanceTensor(dg_perisomatic_inhib),
            g_gaba_b_input=None,
        )
        return dg_spikes, dg_inhib_output

    def _step_ca3(
        self,
        ca3_input: Any,
        septal_gaba: Optional[Any],
    ) -> tuple[Any, dict]:
        """Process one timestep of CA3 (pattern completion + recurrence).

        Reads ``_dg_ca3_buffer`` and ``_ca3_ca3_buffer`` internally.
        Sets ``self.ripple_detected``, ``self.ripple_duration_steps``, and ``self._swr_ca3_pattern``.

        Returns:
            ``(ca3_spikes, ca3_inhib_output)``
        """
        config = self.config

        dg_ca3_synapse = SynapseId(
            self.region_name, HippocampusPopulation.DG,
            self.region_name, HippocampusPopulation.CA3,
            receptor_type=ReceptorType.AMPA,
        )
        ca3_ca3_synapse = SynapseId(
            self.region_name, HippocampusPopulation.CA3,
            self.region_name, HippocampusPopulation.CA3,
            receptor_type=ReceptorType.AMPA,
        )
        dg_ca3_weights  = self.get_synaptic_weights(dg_ca3_synapse)
        ca3_ca3_weights = self.get_synaptic_weights(ca3_ca3_synapse)

        # DG mossy fiber input (with axonal delay + STP facilitation)
        # Mossy fibers are FACILITATING – repeated DG spikes progressively enhance CA3.
        dg_spikes_delayed      = self._dg_ca3_buffer.read(self._dg_ca3_delay_steps)
        dg_spikes_delayed_float = dg_spikes_delayed.float()
        stp_efficacy = self.stp_modules[dg_ca3_synapse].forward(dg_spikes_delayed_float)
        ca3_from_dg  = torch.matmul(dg_ca3_weights * stp_efficacy.T, dg_spikes_delayed_float)

        # NOTE: ca3_input includes ALL external sources (EC, cortex, PFC, thalamus)
        # but NOT DG (computed above with STP).
        ca3_ff = ca3_input + ca3_from_dg

        # CA3 recurrent input (ACh-gated, delayed)
        # High ACh (encoding) suppresses recurrence; low ACh (retrieval) releases it.
        ach_level = self._ach_concentration_ca3.mean().item()
        ach_recurrent_modulation = compute_ach_recurrent_suppression(ach_level)
        ca3_delayed   = self._ca3_ca3_buffer.read(self._ca3_ca3_buffer.max_delay)
        ca3_rec_raw   = torch.matmul(ca3_ca3_weights, ca3_delayed.float())
        ca3_rec       = ca3_rec_raw * ach_recurrent_modulation

        # Lateral CA3→CA3 feedback inhibition (basket cells, causal = prev timestep)
        prev_ca3_spikes = self._ca3_spike_buffer.read(1)
        ca3_ca3_inhib_synapse = SynapseId(
            source_region=self.region_name,
            source_population=HippocampusPopulation.CA3,
            target_region=self.region_name,
            target_population=HippocampusPopulation.CA3,
            receptor_type=ReceptorType.GABA_A,
        )
        ca3_feedback_inhibition = torch.matmul(
            self.get_synaptic_weights(ca3_ca3_inhib_synapse), prev_ca3_spikes.float()
        )

        # NE gain modulation (Î²-adrenergic: high NE → more responsive)
        ne_level = self._ne_concentration_ca3.mean().item()
        ne_gain  = compute_ne_gain(ne_level)
        ca3_excitatory_input = (ca3_ff + ca3_rec) * ne_gain

        # CA3 inhibitory network (PV, OLM, Bistratified; OLM phase-locks to septal theta)
        ca3_inhib_output = self._run_hippocampal_inhibitory(
            inhib_net=self.ca3_inhibitory,
            prev_pyr_spikes=prev_ca3_spikes,
            prev_pv_spikes_buf=self._ca3_pv_buffer,
            prev_olm_spikes_buf=self._ca3_olm_buffer,
            prev_bist_spikes_buf=self._ca3_bistratified_buffer,
            pyr_pop=HippocampusPopulation.CA3,
            pv_pop=HippocampusPopulation.CA3_INHIBITORY_PV,
            olm_pop=HippocampusPopulation.CA3_INHIBITORY_OLM,
            bist_pop=HippocampusPopulation.CA3_INHIBITORY_BISTRATIFIED,
            septal_gaba=septal_gaba,
        )
        ca3_perisomatic_inhib = ca3_inhib_output["perisomatic_gaba_a"]

        # SWR replay injection: during an active ripple window, re-excite CA3 with the
        # pattern captured at ripple onset.  This sustains the attractor across the full
        # window, enabling multi-timestep sequence replay to CA1 (Buzsaki 1989).
        if self.ripple_detected and self._swr_ca3_pattern is not None:
            ca3_excitatory_input = (
                ca3_excitatory_input
                + self._swr_ca3_pattern.float() * config.ripple_replay_injection
            )

        ca3_g_exc = F.relu(ca3_excitatory_input)

        # Conductance validation: biological range [0, 5] relative to g_L = 1
        if ca3_g_exc.max() > 10.0:
            print(f"\u26a0\ufe0f  WARNING: CA3 g_exc={ca3_g_exc.max():.2f} exceeds biological range [0, 5]!")
            print(f"  This indicates current/conductance confusion in synaptic weights.")
            print(f"  ca3_from_dg={ca3_from_dg.mean():.4f}, ca3_rec={ca3_rec.mean():.4f}")
            ca3_g_exc = torch.clamp(ca3_g_exc, 0.0, 5.0)

        # Burst-risk self-inhibition: Only apply when V_mem > 80% of threshold
        v_rest       = self.ca3_neurons.E_L.item()
        v_threshold  = self.ca3_neurons.v_threshold.mean().item()
        v_normalized = torch.clamp((self.ca3_neurons.V_soma - v_rest) / (v_threshold - v_rest), 0.0, 1.0)
        burst_risk   = torch.clamp((v_normalized - 0.8) / 0.2, 0.0, 1.0)

        ca3_g_inh = (
            F.relu(ca3_feedback_inhibition)
            + F.relu(ca3_perisomatic_inhib)
            + config.tonic_inhibition
            + burst_risk * 0.1
        )

        ca3_g_ampa, ca3_g_nmda = split_excitatory_conductance(ca3_g_exc, nmda_ratio=0.2)

        # 5-HT2C: high serotonin suppresses NMDA NR2B-mediated currents on CA3 spines
        sht_level = self._sht_concentration_ca3.mean().item()
        if sht_level > 0.0:
            ca3_g_nmda = ca3_g_nmda * (1.0 - 0.5 * sht_level)

        ca3_spikes, _, _ = self.ca3_neurons.forward(
            g_ampa_basal=ConductanceTensor(ca3_g_ampa),
            g_nmda_basal=ConductanceTensor(ca3_g_nmda),
            g_gaba_a_basal=ConductanceTensor(ca3_g_inh),
            g_gaba_b_basal=ConductanceTensor(ca3_inhib_output["perisomatic_gaba_b"]),
            g_ampa_apical=None,
            g_nmda_apical=None,
            g_gaba_a_apical=None,
        )

        # SWR state: duration-gated detection (replaces single-timestep bool).
        # Biology: SWRs last 50-150 ms; the window sustains replay drive across
        # the full event including low-rate inter-cycle timesteps.
        ca3_pop_rate  = ca3_spikes.float().mean().item()
        swr_max_steps = max(1, round(config.ripple_duration_max_ms / config.dt_ms))
        if ca3_pop_rate > config.ripple_threshold:
            if not self.ripple_detected:
                # Onset: capture the active CA3 attractor pattern for replay injection.
                self._swr_ca3_pattern = ca3_spikes.detach().clone()
            self.ripple_duration_steps = swr_max_steps   # extend/refresh window
            self.ripple_detected = True
        else:
            if self.ripple_duration_steps > 1:
                self.ripple_duration_steps -= 1
                self.ripple_detected = True   # still within the ripple envelope
            else:
                self.ripple_duration_steps = 0
                self.ripple_detected = False  # window expired

        return ca3_spikes, ca3_inhib_output

    def _step_ca2(
        self,
        septal_gaba: Optional[Any],
    ) -> tuple[Any, dict]:
        """Process one timestep of CA2 (social memory / temporal context).

        Reads ``_ca3_ca2_buffer`` internally.

        Returns:
            ``(ca2_spikes, ca2_inhib_output)``
        """
        config = self.config

        ca3_ca2_synapse = SynapseId(
            self.region_name, HippocampusPopulation.CA3,
            self.region_name, HippocampusPopulation.CA2,
            receptor_type=ReceptorType.AMPA,
        )
        ca3_ca2_weights = self.get_synaptic_weights(ca3_ca2_synapse)

        # CA3→CA2 with axonal delay + STP (depressing – stability mechanism)
        ca3_spikes_for_ca2      = self._ca3_ca2_buffer.read(self._ca3_ca2_delay_steps)
        ca3_spikes_for_ca2_float = ca3_spikes_for_ca2.float()
        stp_efficacy     = self.stp_modules[ca3_ca2_synapse].forward(ca3_spikes_for_ca2_float)
        effective_w_ca3_ca2 = ca3_ca2_weights * stp_efficacy.T
        ca2_from_ca3 = torch.matmul(effective_w_ca3_ca2, ca3_spikes_for_ca2_float)

        # CA2 lateral feedback inhibition (prevents runaway amplification)
        prev_ca2_spikes      = self._ca2_spike_buffer.read(1)
        ca2_population_rate  = prev_ca2_spikes.float().mean()
        ca2_ca2_inhib_synapse = SynapseId(
            source_region=self.region_name,
            source_population=HippocampusPopulation.CA2,
            target_region=self.region_name,
            target_population=HippocampusPopulation.CA2,
            receptor_type=ReceptorType.GABA_A,
        )
        local_lateral_ca2 = torch.matmul(
            self.get_synaptic_weights(ca2_ca2_inhib_synapse), prev_ca2_spikes.float()
        )
        ca2_feedback_inhibition = ca2_population_rate * 0.2 + local_lateral_ca2 * 0.1

        ca2_g_exc = F.relu(ca2_from_ca3)

        ca2_inhib_output = self._run_hippocampal_inhibitory(
            inhib_net=self.ca2_inhibitory,
            prev_pyr_spikes=self._ca2_spike_buffer.read(1),
            prev_pv_spikes_buf=self._ca2_pv_buffer,
            prev_olm_spikes_buf=self._ca2_olm_buffer,
            prev_bist_spikes_buf=self._ca2_bistratified_buffer,
            pyr_pop=HippocampusPopulation.CA2,
            pv_pop=HippocampusPopulation.CA2_INHIBITORY_PV,
            olm_pop=HippocampusPopulation.CA2_INHIBITORY_OLM,
            bist_pop=HippocampusPopulation.CA2_INHIBITORY_BISTRATIFIED,
            septal_gaba=septal_gaba,
        )
        ca2_perisomatic_inhib = ca2_inhib_output["perisomatic_gaba_a"]

        # Burst-risk self-inhibition
        v_rest       = self.ca2_neurons.E_L.item()
        v_threshold  = self.ca2_neurons.v_threshold.mean().item()
        v_normalized = torch.clamp((self.ca2_neurons.V_soma - v_rest) / (v_threshold - v_rest), 0.0, 1.0)
        burst_risk   = torch.clamp((v_normalized - 0.8) / 0.2, 0.0, 1.0)
        ca2_g_inh = (
            F.relu(config.tonic_inhibition + ca2_feedback_inhibition + ca2_perisomatic_inhib)
            + burst_risk * 0.1
        )

        ca2_g_ampa, ca2_g_nmda = split_excitatory_conductance(ca2_g_exc, nmda_ratio=0.05)  # Reduced from 0.2

        ca2_spikes, _ = self.ca2_neurons.forward(
            g_ampa_input=ConductanceTensor(ca2_g_ampa),
            g_nmda_input=ConductanceTensor(ca2_g_nmda),
            g_gaba_a_input=ConductanceTensor(ca2_g_inh),
            g_gaba_b_input=ConductanceTensor(ca2_inhib_output["perisomatic_gaba_b"]),
        )
        return ca2_spikes, ca2_inhib_output

    def _step_ca1(
        self,
        ca1_input: Any,
        ffi_factor: float,
        encoding_mod: float,
        retrieval_mod: float,
        septal_gaba: Optional[Any],
    ) -> tuple[Any, dict, Any, Any]:
        """Process one timestep of CA1 (memory output / EC comparison layer).

        Reads ``_ca3_ca1_buffer`` and ``_ca2_ca1_buffer`` internally.
        Sets ``self._prev_encoding_mod`` and ``self.nmda_trace``.

        Returns:
            ``(ca1_spikes, ca1_inhib_output, ca1_basal_g_exc, ca1_apical_g_exc)``
        """
        config = self.config

        ca3_ca1_synapse = SynapseId(
            self.region_name, HippocampusPopulation.CA3,
            self.region_name, HippocampusPopulation.CA1,
            receptor_type=ReceptorType.AMPA,
        )
        ca2_ca1_synapse = SynapseId(
            self.region_name, HippocampusPopulation.CA2,
            self.region_name, HippocampusPopulation.CA1,
            receptor_type=ReceptorType.AMPA,
        )
        ca3_ca1_weights = self.get_synaptic_weights(ca3_ca1_synapse)
        ca2_ca1_weights = self.get_synaptic_weights(ca2_ca1_synapse)

        # CA3→CA1 Schaffer collateral (delayed + STP depressing)
        ca3_spikes_delayed = self._ca3_ca1_buffer.read(self._ca3_ca1_delay_steps)
        stp_efficacy = self.stp_modules[ca3_ca1_synapse].forward(ca3_spikes_delayed.float())
        effective_w_ca3_ca1 = ca3_ca1_weights * stp_efficacy.T
        ca1_from_ca3 = torch.matmul(effective_w_ca3_ca1, ca3_spikes_delayed.float())

        # SWR replay boost (offline memory consolidation via sharp-wave ripples)
        if self.ripple_detected:
            ca1_from_ca3 = ca1_from_ca3 * config.ripple_boost_factor

        # FFI gating clears stale activity on stimulus change
        ca1_from_ec  = ca1_input   * ffi_factor
        ca1_from_ca3 = ca1_from_ca3 * ffi_factor

        # NMDA trace: tracks CA3-induced depolarisation for MgÂ²âº block removal
        if self.nmda_trace is not None:
            nmda_decay = torch.exp(torch.tensor(-config.dt_ms / config.nmda_tau))
            self.nmda_trace = self.nmda_trace * nmda_decay + ca1_from_ca3 * (1.0 - nmda_decay)
        else:
            self.nmda_trace = ca1_from_ca3

        # NMDA gating: stronger during retrieval (theta peak)
        mg_block_removal  = torch.sigmoid(
            (self.nmda_trace - config.nmda_threshold) * config.nmda_steepness
        ) * retrieval_mod
        nmda_conductance  = ca1_from_ec * mg_block_removal

        # Muscarinic NR2B enhancement (septal ACh boosts CaÂ²âº influx during encoding)
        ach_septal_mean  = self._ach_septal_concentration_ca1.mean().item()
        nmda_conductance = nmda_conductance * (1.0 + ach_septal_mean * 0.5)

        ampa_conductance  = ca1_from_ec * config.ampa_ratio
        ca3_contribution  = ca1_from_ca3 * (0.5 + 0.5 * encoding_mod)

        # CA2→CA1 (delayed + STP facilitating – temporal sequences)
        ca2_spikes_delayed      = self._ca2_ca1_buffer.read(self._ca2_ca1_delay_steps)
        ca2_spikes_delayed_float = ca2_spikes_delayed.float()
        stp_efficacy = self.stp_modules[ca2_ca1_synapse].forward(ca2_spikes_delayed_float)
        effective_w_ca2_ca1 = ca2_ca1_weights * stp_efficacy.T
        ca1_from_ca2 = torch.matmul(effective_w_ca2_ca1, ca2_spikes_delayed_float) * ffi_factor

        # CA1 inhibitory network (OLM creates emergent encoding/retrieval separation)
        ca1_inhib_output = self._run_hippocampal_inhibitory(
            inhib_net=self.ca1_inhibitory,
            prev_pyr_spikes=self._ca1_spike_buffer.read(1),
            prev_pv_spikes_buf=self._ca1_pv_buffer,
            prev_olm_spikes_buf=self._ca1_olm_buffer,
            prev_bist_spikes_buf=self._ca1_bistratified_buffer,
            pyr_pop=HippocampusPopulation.CA1,
            pv_pop=HippocampusPopulation.CA1_INHIBITORY_PV,
            olm_pop=HippocampusPopulation.CA1_INHIBITORY_OLM,
            bist_pop=HippocampusPopulation.CA1_INHIBITORY_BISTRATIFIED,
            septal_gaba=septal_gaba,
        )
        ca1_perisomatic_inhib = ca1_inhib_output["perisomatic_gaba_a"]  # PV basket cells
        ca1_dendritic_inhib   = ca1_inhib_output["dendritic"]            # OLM + bistratified
        ca1_olm_inhib         = ca1_inhib_output["olm_dendritic"]        # OLM only

        # Update encoding/retrieval for NEXT timestep from current OLM firing rate.
        # Baseline 0.3 ensures tonic ACh-mediated encoding drive during early training.
        olm_firing_rate = ca1_olm_inhib.mean().item()
        self._prev_encoding_mod = torch.clamp(
            torch.tensor(0.3 + olm_firing_rate * 2.0), 0.0, 1.0
        ).item()

        # -----------------------------------------------------------------------
        # Two-compartment CA1:
        #   Basal (proximal)  – CA3 Schaffer collateral + CA2 temporal context
        #   Apical (distal)   – EC direct perforant path (retrieval route)
        # -----------------------------------------------------------------------
        ca1_basal_g_exc  = ca3_contribution + ca1_from_ca2
        ca1_apical_g_exc = ampa_conductance  + nmda_conductance

        ca1_basal_g_ampa,  ca1_basal_g_nmda  = split_excitatory_conductance(ca1_basal_g_exc,  nmda_ratio=0.2)
        ca1_apical_g_ampa, ca1_apical_g_nmda = split_excitatory_conductance(ca1_apical_g_exc, nmda_ratio=0.3)

        # Perisomatic inhibition (PV basket cells + population feedback + tonic ambient)
        prev_ca1_spikes    = self._ca1_spike_buffer.read(1)
        ca1_feedback_inhib = prev_ca1_spikes.float().mean() * 0.1

        v_rest       = self.ca1_neurons.E_L.item()
        v_threshold  = self.ca1_neurons.v_threshold.mean().item()
        v_normalized = torch.clamp((self.ca1_neurons.V_soma - v_rest) / (v_threshold - v_rest), 0.0, 1.0)
        burst_risk   = torch.clamp((v_normalized - 0.8) / 0.2, 0.0, 1.0)
        ca1_g_inh = (
            F.relu(config.tonic_inhibition + ca1_perisomatic_inhib + ca1_feedback_inhib)
            + burst_risk * 0.1
        )

        # OLM dendritic inhibition targets apical tuft (biologically correct)
        ca1_spikes, _ca1_membrane, _ca1_V_dend = self.ca1_neurons.forward(
            g_ampa_basal=ConductanceTensor(ca1_basal_g_ampa),
            g_nmda_basal=ConductanceTensor(ca1_basal_g_nmda),
            g_gaba_a_basal=ConductanceTensor(ca1_g_inh),
            g_gaba_b_basal=ConductanceTensor(ca1_inhib_output["perisomatic_gaba_b"]),
            g_ampa_apical=ConductanceTensor(ca1_apical_g_ampa),
            g_nmda_apical=ConductanceTensor(ca1_apical_g_nmda),
            g_gaba_a_apical=ConductanceTensor(F.relu(ca1_dendritic_inhib)),
        )

        return ca1_spikes, ca1_inhib_output, ca1_basal_g_exc, ca1_apical_g_exc

    def _update_match_mismatch(
        self,
        ca1_basal_g_exc: Any,
        ca1_apical_g_exc: Any,
        retrieval_mod: float,
    ) -> None:
        """Compare CA3 Schaffer drive vs EC direct drive to produce match/mismatch.

        Updates ``self.ca1_match_level`` and ``self.ca1_mismatch_level``.

        Ref: Kumaran & Maguire 2007; Lisman & Grace 2005 (Hippocampal-VTA loop).
        """
        _basal_peak  = ca1_basal_g_exc.max().item()
        _apical_peak = ca1_apical_g_exc.max().item()
        if _basal_peak > 1e-8 and _apical_peak > 1e-8:
            _basal_norm  = ca1_basal_g_exc  / _basal_peak
            _apical_norm = ca1_apical_g_exc / _apical_peak
            # Phase-gate: comparison is genuine only during retrieval mode.
            # During encoding, apical > basal just means "new input", not mismatch.
            self.ca1_match_level    = (torch.minimum(_basal_norm, _apical_norm).mean() * retrieval_mod).item()
            self.ca1_mismatch_level = (F.relu(_apical_norm - _basal_norm).mean() * retrieval_mod).item()
        elif _apical_peak > 1e-8:
            # EC active, CA3 silent → pure novelty (gated by retrieval mode)
            self.ca1_match_level    = 0.0
            self.ca1_mismatch_level = retrieval_mod
        else:
            self.ca1_match_level    = 0.0
            self.ca1_mismatch_level = 0.0

    def _apply_plasticity(
        self,
        synaptic_inputs: SynapticInput,
        dg_spikes: Any,
        ca3_spikes: Any,
        ca2_spikes: Any,
        ca1_spikes: Any,
        encoding_mod: float,
    ) -> None:
        """Apply all Hebbian and three-factor plasticity rules for the current timestep.

        Must be called BEFORE :meth:`_update_spike_buffers` so that delay buffers
        still hold the pre-update spike trains needed for learning.
        """
        config = self.config

        # Synaptic handles (cheap dataclass construction)
        dg_ca3_synapse  = SynapseId(self.region_name, HippocampusPopulation.DG,  self.region_name, HippocampusPopulation.CA3, receptor_type=ReceptorType.AMPA)
        ca3_ca3_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3, self.region_name, HippocampusPopulation.CA3, receptor_type=ReceptorType.AMPA)
        ca3_ca2_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3, self.region_name, HippocampusPopulation.CA2, receptor_type=ReceptorType.AMPA)
        ca2_ca1_synapse = SynapseId(self.region_name, HippocampusPopulation.CA2, self.region_name, HippocampusPopulation.CA1, receptor_type=ReceptorType.AMPA)

        dg_ca3_weights  = self.get_synaptic_weights(dg_ca3_synapse)
        ca3_ca3_weights = self.get_synaptic_weights(ca3_ca3_synapse)
        ca3_ca2_weights = self.get_synaptic_weights(ca3_ca2_synapse)
        ca2_ca1_weights = self.get_synaptic_weights(ca2_ca1_synapse)

        # Re-read delay buffers (not yet advanced for this timestep)
        dg_spikes_delayed       = self._dg_ca3_buffer.read(self._dg_ca3_delay_steps)
        ca3_spikes_for_ca2      = self._ca3_ca2_buffer.read(self._ca3_ca2_delay_steps)
        ca2_spikes_delayed      = self._ca2_ca1_buffer.read(self._ca2_ca1_delay_steps)
        dg_spikes_delayed_float   = dg_spikes_delayed.float()
        ca3_spikes_for_ca2_float  = ca3_spikes_for_ca2.float()
        ca2_spikes_delayed_float  = ca2_spikes_delayed.float()

        dg_spikes_float  = dg_spikes.float()
        ca3_spikes_float = ca3_spikes.float()
        ca2_spikes_float = ca2_spikes.float()
        ca1_spikes_float = ca1_spikes.float()

        # Per-subregion DA levels
        dg_da_level  = self._da_concentration_dg.mean().item()
        ca3_da_level = self._da_concentration_ca3.mean().item()
        ca2_da_level = self._da_concentration_ca2.mean().item()
        ca1_da_level = self._da_concentration_ca1.mean().item()

        # -----------------------------------------------------------------------
        # CA3→CA3 Hebbian (instantaneous delta + heterosynaptic LTD)
        # -----------------------------------------------------------------------
        if ca3_spikes.any():
            ca3_da_gain  = 0.2 + 1.8 * ca3_da_level  # Range: [0.2, 2.0]
            effective_lr = config.learning_rate * encoding_mod * ca3_da_gain
            dW = torch.outer(ca3_spikes_float, ca3_spikes_float)

            if config.heterosynaptic_ratio > 0:
                inactive_ca3_post = torch.logical_not(ca3_spikes).float()
                hetero_dW = -config.heterosynaptic_ratio * torch.outer(ca3_spikes_float, inactive_ca3_post)
                dW = dW + hetero_dW

            ca3_syn_mask = (ca3_ca3_weights.data > 0).float()
            ca3_ca3_weights.data += dW * effective_lr * ca3_syn_mask
            ca3_ca3_weights.data.fill_diagonal_(0.0)  # No self-connections
            clamp_weights(ca3_ca3_weights.data, config.w_min, config.w_max)

        # -----------------------------------------------------------------------
        # DG→CA3 Mossy fiber (one-shot binding)
        # "Detonator synapses" – rapid, powerful LTP for pattern indexing.
        # -----------------------------------------------------------------------
        if dg_spikes_delayed.any() and ca3_spikes.any():
            ca3_da_gain        = 0.2 + 1.8 * ca3_da_level
            mossy_effective_lr = config.learning_rate * 3.0 * encoding_mod * ca3_da_gain
            dW_mossy       = torch.outer(ca3_spikes_float, dg_spikes_delayed_float)
            dg_ca3_syn_mask = (dg_ca3_weights.data > 0).float()
            dg_ca3_weights.data += dW_mossy * mossy_effective_lr * dg_ca3_syn_mask
            clamp_weights(dg_ca3_weights.data, config.w_min, config.w_max)

        # -----------------------------------------------------------------------
        # CA3→CA2 suppressed Hebbian (RGS14-mediated CaMKII block)
        # Biology: CA2 principal neurons express RGS14, a G-protein signalling
        # suppressor that blocks CaMKII activation specifically at CA3→CA2
        # Schaffer-like synapses (Zhao et al. 2007; Lee et al. 2010).  This
        # prevents CA3 pattern-completion attractors from overwriting the stable
        # social/temporal context held in CA2.
        # Rate is 10× lower than CA3→CA3 recurrent plasticity.
        # NOTE: RGS14 does NOT suppress EC→CA2 plasticity – perforant path LTP is intact.
        # -----------------------------------------------------------------------
        if ca3_spikes_for_ca2.any() and ca2_spikes.any():
            # 0.1× scale: RGS14 suppresses ~90% of normal LTP at this synapse.
            # Reduced DA sensitivity (range [0.5, 1.0] vs [0.2, 2.0] for CA3→CA3):
            # dopamine can modestly relieve RGS14 block but cannot fully overcome it.
            ca2_da_gain  = 0.5 + 0.5 * ca2_da_level  # [0.5, 1.0]
            effective_lr = config.learning_rate * encoding_mod * 0.1 * ca2_da_gain
            dW = torch.outer(ca2_spikes_float, ca3_spikes_for_ca2_float)
            ca3_ca2_syn_mask = (ca3_ca2_weights.data > 0).float()
            ca3_ca2_weights.data += dW * effective_lr * ca3_ca2_syn_mask
            clamp_weights(ca3_ca2_weights.data, config.w_min, config.w_max)

        # -----------------------------------------------------------------------
        # CA2→CA1 Hebbian (temporal / social context binding)
        # -----------------------------------------------------------------------
        if ca2_spikes_delayed.any() and ca1_spikes.any():
            effective_lr = config.learning_rate * encoding_mod
            dW = torch.outer(ca1_spikes_float, ca2_spikes_delayed_float)
            ca2_ca1_syn_mask = (ca2_ca1_weights.data > 0).float()
            ca2_ca1_weights.data += dW * effective_lr * ca2_ca1_syn_mask
            clamp_weights(ca2_ca1_weights.data, config.w_min, config.w_max)

        if encoding_mod > 0.5:
            # -----------------------------------------------------------------------
            # External input plasticity (per-source three-factor Hebbian)
            # EC→hippocampus synapses; only during encoding mode (theta trough).
            # -----------------------------------------------------------------------
            da_modulation = (dg_da_level + ca3_da_level + ca2_da_level + ca1_da_level) / 4.0
            effective_lr  = config.learning_rate * 0.3 * (1.0 + da_modulation)

            for synapse_id, source_input in synaptic_inputs.items():
                weights     = self.get_synaptic_weights(synapse_id)
                ext_syn_mask = (weights.data > 0).float()

                if synapse_id.target_population == HippocampusPopulation.DG:
                    dW_dg = effective_lr * torch.outer(dg_spikes_float, source_input.float())
                    weights.data += dW_dg * ext_syn_mask

                elif synapse_id.target_population == HippocampusPopulation.CA3:
                    dW_ca3 = effective_lr * torch.outer(ca3_spikes_float, source_input.float())
                    weights.data += dW_ca3 * ext_syn_mask

                elif synapse_id.target_population == HippocampusPopulation.CA2:
                    # EC→CA2: normal LTP – perforant path is NOT blocked by RGS14.
                    # Zhao et al. 2007: the LTP resistance is synapse-specific to
                    # CA3-derived Schaffer-like inputs.  EC direct input encodes
                    # social/temporal context and must retain full plasticity.
                    dW_ca2 = effective_lr * torch.outer(ca2_spikes_float, source_input.float())
                    weights.data += dW_ca2 * ext_syn_mask

                elif synapse_id.target_population == HippocampusPopulation.CA1:
                    dW_ca1 = effective_lr * torch.outer(ca1_spikes_float, source_input.float())
                    weights.data += dW_ca1 * ext_syn_mask

                clamp_weights(weights.data, config.w_min, config.w_max)

            # -----------------------------------------------------------------------
            # CA3 recurrent three-factor learning (tag-and-capture)
            # Î” W = eligibility_trace Ã— dopamine Ã— learning_rate
            # -----------------------------------------------------------------------
            ca3_delayed_for_3f = self._ca3_ca3_buffer.read(self._ca3_ca3_delay_steps)
            da_ca3_deviation   = ca3_da_level - 0.5  # Deviation from tonic baseline

            if self.get_learning_strategy(ca3_ca3_synapse) is None:
                self._add_learning_strategy(ca3_ca3_synapse, self._tag_and_capture_strategy, device=self.device)
            self._apply_learning(
                ca3_ca3_synapse, ca3_delayed_for_3f, ca3_spikes,
                modulator=da_ca3_deviation,
            )
            self.get_synaptic_weights(ca3_ca3_synapse).data.fill_diagonal_(0.0)

        # -----------------------------------------------------------------------
        # DA-gated consolidation (tag-and-capture)
        # -----------------------------------------------------------------------
        if ca3_da_level > 0.1:
            ca3_ca3_weights.data = self._tag_and_capture_strategy.consolidate(ca3_da_level, ca3_ca3_weights)
            clamp_weights(ca3_ca3_weights.data, config.w_min, config.w_max)

    def _update_spike_buffers(
        self,
        dg_spikes:  Any,
        ca3_spikes: Any,
        ca2_spikes: Any,
        ca1_spikes: Any,
    ) -> None:
        """Advance all per-region history and inter-layer delay buffers.

        Must be called AFTER :meth:`_apply_plasticity` so that plasticity can
        still read the pre-update spike trains.
        """
        # Per-region 1-step history (used by inhibitory networks next step)
        self._dg_spike_buffer.write_and_advance(dg_spikes)
        self._ca3_spike_buffer.write_and_advance(ca3_spikes)
        self._ca2_spike_buffer.write_and_advance(ca2_spikes)
        self._ca1_spike_buffer.write_and_advance(ca1_spikes)

        # Inter-layer axonal delay buffers
        self._dg_ca3_buffer.write_and_advance(dg_spikes)
        self._ca3_ca3_buffer.write_and_advance(ca3_spikes)
        self._ca3_ca2_buffer.write_and_advance(ca3_spikes)
        self._ca3_ca1_buffer.write_and_advance(ca3_spikes)
        self._ca2_ca1_buffer.write_and_advance(ca2_spikes)
