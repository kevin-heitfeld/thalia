"""
Trisynaptic Hippocampus - Biologically-Accurate DG→CA3→CA1 Episodic Memory Circuit.

This implements the classic hippocampal trisynaptic circuit for episodic memory:
- **Dentate Gyrus (DG)**: Pattern SEPARATION via sparse coding (~2-5% active)
- **CA3**: Pattern COMPLETION via recurrent connections (autoassociative memory)
- **CA1**: Output/comparison layer detecting match vs mismatch

**Key Biological Features**:
===========================
1. **THETA MODULATION** (6-10 Hz oscillations):
   - Theta trough (0-π): Encoding phase (CA3 learning enabled)
   - Theta peak (π-2π): Retrieval phase (comparison active)
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

from typing import ClassVar, List, Optional

import torch
import torch.nn.functional as F

from thalia import GlobalConfig
from thalia.brain.configs import HippocampusConfig
from thalia.components import (
    NeuronFactory,
    NeuromodulatorReceptor,
    WeightInitializer,
    GapJunctionConfig,
    GapJunctionCoupling,
)
from thalia.components.synapses.stp import (
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
    compute_excitability_modulation,
)
from thalia.typing import (
    ConductanceTensor,
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
    """
    Biologically-accurate hippocampus with DG→CA3→CA1 trisynaptic circuit.

    Architecture:
    Input (EC from cortex)
           │
           ├──────────────────────┐ (Direct perforant path)
           ▼                      ▼
    ┌──────────────┐        ┌──────────────┐
    │ Dentate Gyrus│        │     CA3      │  Recurrent connections
    │   (DG)       │─────-->│              │  Pattern COMPLETION: partial cue → full pattern
    │ Pattern SEP  │        └──────┬───────┘
    └──────────────┘               │ ◄──────── (recurrent loop back to CA3)
                                   ▼
                            ┌──────────────┐
                            │     CA2      │  Social memory & temporal context
                            │              │  Weak CA3 plasticity (stability hub)
                            └──────┬───────┘
                                   │           ┌─────── Direct bypass (Schaffer)
                                   ▼           ▼
                            ┌──────────────┐
                            │     CA1      │  Output layer with comparison
                            │              │  COINCIDENCE DETECTION: match vs mismatch
                            └──────────────┘
                                   │
                                   ▼
                            Output (to cortex/striatum)

    Four pathways to CA3:
    - EC→DG→CA3: Pattern-separated (sparse), strong during encoding
    - EC→CA3 direct: Preserves similarity (less sparse), provides retrieval cues

    CA2 layer (social memory):
    - CA3→CA2: Weak plasticity (10x lower) - stability mechanism
    - EC→CA2: Strong direct input for temporal encoding
    - CA2→CA1: Provides temporal/social context to decision layer

    CA1 receives from:
    - CA3 direct (Schaffer collaterals): Pattern completion
    - CA2: Temporal/social context
    - EC direct: Current sensory input
    """

    # Mesolimbic DA (VTA → hippocampus) modulates Schaffer collateral LTP and replay.
    # NE from LC modulates novelty-driven encoding and theta power.
    # ACh from nucleus basalis gates encoding vs. retrieval modes (nicotinic receptors).
    # 'ach_septal': slow muscarinic (M1) modulation from medial septum cholinergic neurons.
    neuromodulator_subscriptions: ClassVar[List[str]] = ['da_mesolimbic', 'ne', 'ach', 'ach_septal']

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: HippocampusConfig, population_sizes: PopulationSizes, region_name: RegionName):
        """Initialize trisynaptic hippocampus."""
        super().__init__(config, population_sizes, region_name)

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

        # Spontaneous replay (sharp-wave ripple) detection
        self.ripple_detected: bool = False

        # =====================================================================
        # GAP JUNCTIONS (Electrical Synapses) - Config Setup
        # =====================================================================
        # Override weights with trisynaptic circuit weights
        self._init_circuit_weights()

        # =====================================================================
        # HIPPOCAMPAL EXCITATORY NEURONS (LIF with adaptation for sparse coding)
        # =====================================================================
        # Create LIF neurons for each layer using factory functions
        # DG: Sparse coding requires high threshold
        # v_threshold raised from 0.9 → 1.6: at 0.9 DG fired at 10 Hz (target <1 Hz).
        # DG granule cells are the most sparsely active cells in the brain; the high
        # threshold combined with strong spike-frequency adaptation enforces the <5%
        # population activity needed for pattern separation.
        self.dg_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=self.region_name,
            population_name=HippocampusPopulation.DG,
            n_neurons=self.dg_size,
            device=self.device,
            v_threshold=1.6,
            adapt_increment=0.30,  # Strong adaptation to enforce sparsity
            tau_adapt=120.0,  # Slow decay to persist across pattern presentations
        )
        # CA3 gets spike-frequency adaptation to prevent frozen attractors
        # Two-compartment: basal dendrites receive DG mossy fibers + recurrents;
        # apical compartment is available for future EC direct-path feedback.
        self.ca3_neurons = NeuronFactory.create_pyramidal_two_compartment(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA3,
            n_neurons=self.ca3_size,
            device=self.device,
            adapt_increment=config.adapt_increment,
            tau_adapt=config.adapt_tau,
            v_threshold=0.50,  # Lowered (1.0→0.50): EC_II alone reaches V_inf≈0.53 at biological rates with STP
        )
        # CA2: Social memory and temporal context - moderate threshold for selectivity
        # Reduced from 1.6 (caused near-silence) → 1.1: slightly above CA3 (1.0) for selectivity
        self.ca2_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA2,
            n_neurons=self.ca2_size,
            device=self.device,
            v_threshold=1.1,
            adapt_increment=0.25,  # Moderate adaptation for selectivity
            tau_adapt=100.0,  # Medium decay for temporal integration
        )
        # CA1: Output layer — two-compartment pyramidal (CA3 Schaffer collateral basal, EC direct apical)
        self.ca1_neurons = NeuronFactory.create_pyramidal_two_compartment(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA1,
            n_neurons=self.ca1_size,
            device=self.device,
            v_threshold=0.30,  # Lowered (0.50→0.30): EC_III V_inf≈0.18 at actual STP-depleted 11 Hz; threshold
            # must be reachable from combined EC_III+CA3 Schaffer+PFC drive when all are sub-target.
            adapt_increment=0.20,  # Moderate adaptation to prevent runaway activity
            tau_adapt=80.0,  # Faster decay for responsive output layer
        )

        # =====================================================================
        # HIPPOCAMPAL INHIBITORY NETWORKS (with OLM cells for emergent theta)
        # =====================================================================
        # DG inhibitory network: Minimal inhibition for pattern separation
        # Moderate at 0.20 to prevent avalanches while maintaining sparse coding
        # v_threshold_bistratified=1.00: threshold 0.9 → 2.1 Hz (HIGH, target 0–1 Hz);
        # 1.10–1.30 → 0 Hz (stochastic silence in short windows); 1.00 aims for ~0.3–0.8 Hz.
        self.dg_inhibitory = HippocampalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=HippocampusPopulation.DG_INHIBITORY,
            pyr_size=self.dg_size,
            total_inhib_fraction=0.20,
            dt_ms=config.dt_ms,
            device=str(self.device),
            v_threshold_bistratified=1.00,
        )

        # CA3 inhibitory network: Moderate inhibition for pattern completion stability
        # INHIBITION REDUCED: 0.65 → 0.40 → 0.25 (further reduced: CA3 at 0.1 Hz, needs more excitation)
        # OLM/bistratified thresholds lowered vs DG: at sparse CA3 firing (0.75–2 Hz) the
        # pyramidal→OLM V_inf≈0.18–0.45; DG-level thresholds (1.0/0.9) are unreachable.
        self.ca3_inhibitory = HippocampalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA3_INHIBITORY,
            pyr_size=self.ca3_size,
            total_inhib_fraction=0.25,
            dt_ms=config.dt_ms,
            device=str(self.device),
            v_threshold_olm=0.35,
            v_threshold_bistratified=0.30,
        )

        # CA2 inhibitory network: Social/temporal context processing
        self.ca2_inhibitory = HippocampalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA2_INHIBITORY,
            pyr_size=self.ca2_size,
            total_inhib_fraction=0.15,
            dt_ms=config.dt_ms,
            device=str(self.device),
            v_threshold_olm=0.35,
            v_threshold_bistratified=0.30,
        )

        # CA1 inhibitory network: PV, OLM, Bistratified cells
        # OLM cells phase-lock to septal GABA → emergent encoding/retrieval
        self.ca1_inhibitory = HippocampalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA1_INHIBITORY,
            pyr_size=self.ca1_size,
            total_inhib_fraction=0.30,
            dt_ms=config.dt_ms,
            device=str(self.device),
            v_threshold_olm=0.35,
            v_threshold_bistratified=0.30,
        )

        # Stimulus gating module (transient inhibition at stimulus changes)
        self.stimulus_gating = StimulusGating(
            n_neurons=self.dg_size,
            device=self.device,
            threshold=config.ffi_threshold,
            max_inhibition=config.ffi_strength * 10.0,  # Scale to appropriate range
            decay_rate=1.0 - (1.0 / config.ffi_tau),  # Convert tau to rate
        )

        # =====================================================================
        # TEMPORAL STATE BUFFERS (Delay Buffers for All State)
        # =====================================================================
        # Spike state buffers for all subregions
        self._dg_spike_buffer = CircularDelayBuffer(1, self.dg_size, self.device, torch.bool)
        self._ca3_spike_buffer = CircularDelayBuffer(1, self.ca3_size, self.device, torch.bool)
        self._ca2_spike_buffer = CircularDelayBuffer(1, self.ca2_size, self.device, torch.bool)
        self._ca1_spike_buffer = CircularDelayBuffer(1, self.ca1_size, self.device, torch.bool)

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
            device=str(self.device),
            dtype=torch.bool,
        )
        self._ca3_ca3_buffer = CircularDelayBuffer(
            max_delay=ca3_ca3_delay_steps,
            size=self.ca3_size,
            device=str(self.device),
            dtype=torch.bool,
        )
        self._ca3_ca2_buffer = CircularDelayBuffer(
            max_delay=ca3_ca2_delay_steps,
            size=self.ca3_size,
            device=str(self.device),
            dtype=torch.bool,
        )
        self._ca3_ca1_buffer = CircularDelayBuffer(
            max_delay=ca3_ca1_delay_steps,
            size=self.ca3_size,
            device=str(self.device),
            dtype=torch.bool,
        )
        self._ca2_ca1_buffer = CircularDelayBuffer(
            max_delay=ca2_ca1_delay_steps,
            size=self.ca2_size,
            device=str(self.device),
            dtype=torch.bool,
        )

        # =====================================================================
        # PLASTICITY AND HOMEOSTASIS
        # =====================================================================
        # Intrinsic plasticity state (threshold adaptation)
        self._ca3_activity_history: torch.Tensor
        self._ca3_threshold_offset: torch.Tensor
        self.register_buffer("_ca3_activity_history", torch.zeros(self.ca3_size, device=self.device))
        self.register_buffer("_ca3_threshold_offset", torch.zeros(self.ca3_size, device=self.device))

        # =========================================================================
        # ADAPTIVE HOMEOSTATIC SCALING (Biologically-Inspired)
        # =========================================================================
        # Based on Turrigiano & Nelson (2004): Homeostatic synaptic scaling
        # - Timescale: Hours to days in biology, 10-30 seconds here (practical speedup)
        # - Mechanism: Multiplicative scaling based on average firing rate
        # - Key property: Preserves selectivity (relative weight ratios maintained)

        # Track average firing rate over sliding window
        self._homeostasis_counter: int = 0
        self._homeostasis_interval: int = 1000  # Check every 1 second (1000ms)
        self._homeostasis_window_size: int = 30  # Average over 30 checks = 30 seconds
        self._homeostasis_write_idx: int = 0     # Ring buffer write position
        self._homeostasis_filled: int = 0        # Values written (capped at buffer size)
        # Pre-allocated tensor ring buffer; survives .to(device) unlike a Python list
        self._homeostasis_history: torch.Tensor
        self.register_buffer(
            "_homeostasis_history",
            torch.zeros(self._homeostasis_window_size * self._homeostasis_interval, device=self.device),
        )

        # (Multi-timescale fast/slow trace buffers removed — they accumulated the
        # full historical ΣdW each step and re-applied it as a weight delta,
        # causing O(n²) weight growth that saturated w_max within ~15 timesteps.
        # DA-gated long-term consolidation is handled by the tag-and-capture
        # strategy (three-factor rule) registered below.)

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY
        # =====================================================================
        # Per-subregion homeostatic state (firing-rate EMA).
        # Neurons registered below; looked up lazily in _update_homeostasis().
        self._register_homeostasis(HippocampusPopulation.DG,  self.dg_size,  target_firing_rate=0.001)  # <1 Hz (sparse pattern separator)
        self._register_homeostasis(HippocampusPopulation.CA3, self.ca3_size, target_firing_rate=0.003)  # 1–5 Hz
        self._register_homeostasis(HippocampusPopulation.CA2, self.ca2_size, target_firing_rate=0.003)  # 1–5 Hz
        self._register_homeostasis(HippocampusPopulation.CA1, self.ca1_size, target_firing_rate=0.003)  # 1–5 Hz

        # =====================================================================
        # DOPAMINE RECEPTOR (minimal 10% VTA projection)
        # =====================================================================
        # Hippocampus receives minimal DA innervation for novelty/salience modulation
        # Primarily affects CA1 output and CA3 consolidation
        # Biological: VTA DA enhances LTP in novelty-detecting neurons
        total_neurons = self.dg_size + self.ca3_size + self.ca2_size + self.ca1_size
        self.da_receptor = NeuromodulatorReceptor(
            n_receptors=total_neurons,
            tau_rise_ms=10.0,  # Fast kinetics for rapid modulation
            tau_decay_ms=50.0,  # Medium clearance for transient effects
            spike_amplitude=0.5,  # Strong amplitude for clear reward signal
            device=self.device,
        )
        # Per-subregion DA concentration buffers (0.5 = tonic baseline)
        self._da_concentration_dg = torch.full((self.dg_size,), 0.5, device=self.device)
        self._da_concentration_ca3 = torch.full((self.ca3_size,), 0.5, device=self.device)
        self._da_concentration_ca2 = torch.full((self.ca2_size,), 0.5, device=self.device)
        self._da_concentration_ca1 = torch.full((self.ca1_size,), 0.5, device=self.device)

        # =====================================================================
        # NOREPINEPHRINE RECEPTOR (LC projection for arousal/novelty)
        # =====================================================================
        # Hippocampus receives dense NE innervation from LC
        # NE modulates novelty detection and arousal-dependent memory formation
        self.ne_receptor = NeuromodulatorReceptor(
            n_receptors=total_neurons,
            tau_rise_ms=8.0,
            tau_decay_ms=150.0,
            spike_amplitude=0.12,
            device=self.device,
        )
        # Per-subregion NE concentration buffers
        self._ne_concentration_dg = torch.zeros(self.dg_size, device=self.device)
        self._ne_concentration_ca3 = torch.zeros(self.ca3_size, device=self.device)
        self._ne_concentration_ca2 = torch.zeros(self.ca2_size, device=self.device)
        self._ne_concentration_ca1 = torch.zeros(self.ca1_size, device=self.device)

        # =====================================================================
        # ACETYLCHOLINE RECEPTOR (NB projection for encoding/retrieval)
        # =====================================================================
        # Hippocampus receives strong ACh innervation from nucleus basalis
        # ACh controls encoding vs retrieval modes (Hasselmo 1999):
        # - High ACh → encoding mode (suppress recurrence, enhance feedforward)
        # - Low ACh → retrieval mode (enable pattern completion, consolidation)
        self.ach_receptor = NeuromodulatorReceptor(
            n_receptors=total_neurons,
            tau_rise_ms=5.0,
            tau_decay_ms=50.0,
            spike_amplitude=0.2,
            device=self.device,
        )
        # Per-subregion ACh concentration buffers
        self._ach_concentration_dg = torch.zeros(self.dg_size, device=self.device)
        self._ach_concentration_ca3 = torch.zeros(self.ca3_size, device=self.device)
        self._ach_concentration_ca2 = torch.zeros(self.ca2_size, device=self.device)
        self._ach_concentration_ca1 = torch.zeros(self.ca1_size, device=self.device)

        # =====================================================================
        # SEPTAL ACETYLCHOLINE RECEPTOR (muscarinic M1, from medial septum)
        # =====================================================================
        # Septal ACh projects to CA1 via volume transmission, acting on M1-type
        # muscarinic receptors. Effects:
        #   - Enhances NMDA NR2B subunit conductance (slower, Ca²⁺-permeable)
        #   - Modulates Schaffer collateral LTP threshold (encoding gate)
        # Kinetics: τ_rise=20ms, τ_decay=300ms (much slower than nicotinic NB ACh)
        self.ach_septal_receptor = NeuromodulatorReceptor(
            n_receptors=self.ca1_size,
            tau_rise_ms=20.0,
            tau_decay_ms=300.0,
            spike_amplitude=0.15,
            device=self.device,
        )
        self._ach_septal_concentration_ca1 = torch.zeros(self.ca1_size, device=self.device)

        # =====================================================================
        # LEARNING STRATEGY (Tag-and-Capture wrapping Three-Factor Learning)
        # =====================================================================
        # TagAndCaptureStrategy wraps ThreeFactorStrategy:
        # - compute_update: applies three-factor rule (eligibility × DA × lr)
        #   AND updates the tag matrix as a side effect (no extra calls needed).
        # - consolidate(): DA-gated capture that permanently strengthens tagged
        #   synapses when dopamine is elevated.
        # - tags tensor: readable for spontaneous replay prioritisation.
        self._tag_and_capture_strategy = TagAndCaptureStrategy(
            base_strategy=ThreeFactorStrategy(ThreeFactorConfig(
                learning_rate=0.001,  # Conservative rate for stable learning
                eligibility_tau=100.0,  # Eligibility trace decay (ms)
                modulator_tau=50.0,  # Modulator (dopamine) decay (ms)
                device=self.device,
            )),
            config=TagAndCaptureConfig(
                tag_decay=0.95,        # ~20 step tag lifetime at 1 ms dt
                tag_threshold=0.1,     # Minimum post-spike to create tag
                consolidation_lr_scale=0.5,  # Half of base lr for capture
                consolidation_da_threshold=0.1,
                device=self.device,
            ),
        )
        # Eagerly allocate the tags buffer (CA3 recurrent: square matrix)
        self._tag_and_capture_strategy.setup(
            n_pre=self.ca3_size,
            n_post=self.ca3_size,
            device=torch.device(self.device),
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
        self.to(self.device)

    def _init_circuit_weights(self) -> None:
        """Initialize internal circuit weights.

        Internal circuit weights (DG→CA3, CA3→CA1, etc.) are initialized here.
        """
        device = self.device

        # =====================================================================
        # INTERNAL WEIGHTS
        # =====================================================================
        # DG → CA3: Random but less sparse (mossy fibers)
        # Biology: "Detonator synapses" with powerful transmission.
        # Weight scale raised 0.0001 → 0.003 (30×): with MOSSY_FIBER_PRESET
        # initial release probability U=0.01, the effective per-spike weight
        # was ~1e-6 at rest — functionally silent.  Detonator synapses are
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
        # Weight scale raised 0.00001 → 0.0005 (50×): at 0.00001 the max weight
        # was 0.00001 (diagnostics-confirmed), making pattern completion
        # impossible.  CA3 recurrent collaterals are among the strongest
        # synapses in hippocampus; this value allows autoassociative dynamics
        # while remaining well below the runaway-excitation regime.
        # STP: CA3_RECURRENT_PRESET — moderate-strong depression (Dobrunz &
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
        # Biologically: basket cells have local axonal arbors (~200-300μm radius)
        # We approximate this with random sparse connectivity
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA3,
            target_population=HippocampusPopulation.CA3,
            weights=WeightInitializer.sparse_random_no_autapses(
                n_input=self.ca3_size,
                n_output=self.ca3_size,
                connectivity=0.2,
                weight_scale=0.0001,
                device=self.device,
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
        # STP: SCHAFFER_COLLATERAL_PRESET — CA2→CA1 axons are Schaffer-like
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

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Process input spikes through DG→CA3→CA1 circuit.

        Args:
            synaptic_inputs: Point-to-point synaptic connections from cortex, thalamus, etc.
            neuromodulator_inputs: Broadcast neuromodulatory signals (DA, NE, ACh)
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        cfg = self.config
        dt_ms = cfg.dt_ms

        if not GlobalConfig.NEUROMODULATION_DISABLED:
            # =====================================================================
            # DOPAMINE RECEPTOR PROCESSING (from VTA)
            # =====================================================================
            # Process VTA dopamine spikes → concentration dynamics
            # Hippocampus receives minimal (10%) DA innervation for novelty/salience
            vta_da_spikes = self._extract_neuromodulator(neuromodulator_inputs, 'da_mesolimbic')
            # Update full receptor array
            da_concentration_full = self.da_receptor.update(vta_da_spikes)
            # Phasic DA (spikes) adds to tonic baseline (0.5)
            self._da_concentration_dg = 0.5 + da_concentration_full[: self.dg_size] * 1.0
            self._da_concentration_ca3 = 0.5 + da_concentration_full[self.dg_size : self.dg_size + self.ca3_size] * 1.0
            self._da_concentration_ca2 = 0.5 + da_concentration_full[self.dg_size + self.ca3_size : self.dg_size + self.ca3_size + self.ca2_size] * 1.0
            self._da_concentration_ca1 = 0.5 + da_concentration_full[self.dg_size + self.ca3_size + self.ca2_size :] * 1.0

            # =====================================================================
            # NOREPINEPHRINE RECEPTOR PROCESSING (from LC)
            # =====================================================================
            # Process LC norepinephrine spikes → arousal and novelty modulation
            lc_ne_spikes = neuromodulator_inputs.get('ne', None)
            ne_concentration_full = self.ne_receptor.update(lc_ne_spikes)
            # Split into per-subregion buffers
            self._ne_concentration_dg = ne_concentration_full[: self.dg_size]
            self._ne_concentration_ca3 = ne_concentration_full[self.dg_size : self.dg_size + self.ca3_size]
            self._ne_concentration_ca2 = ne_concentration_full[self.dg_size + self.ca3_size : self.dg_size + self.ca3_size + self.ca2_size]
            self._ne_concentration_ca1 = ne_concentration_full[self.dg_size + self.ca3_size + self.ca2_size :]

            # =====================================================================
            # ACETYLCHOLINE RECEPTOR PROCESSING (from NB — nicotinic)
            # =====================================================================
            # Process NB acetylcholine spikes → encoding/retrieval mode control
            # High ACh → encoding, Low ACh → consolidation/retrieval
            nb_ach_spikes = neuromodulator_inputs.get('ach', None)
            ach_concentration_full = self.ach_receptor.update(nb_ach_spikes)
            # Split into per-subregion buffers
            self._ach_concentration_dg = ach_concentration_full[: self.dg_size]
            self._ach_concentration_ca3 = ach_concentration_full[self.dg_size : self.dg_size + self.ca3_size]
            self._ach_concentration_ca2 = ach_concentration_full[self.dg_size + self.ca3_size : self.dg_size + self.ca3_size + self.ca2_size]
            self._ach_concentration_ca1 = ach_concentration_full[self.dg_size + self.ca3_size + self.ca2_size :]

            # =====================================================================
            # SEPTAL ACH RECEPTOR PROCESSING (muscarinic M1 — from medial septum)
            # =====================================================================
            # Slow muscarinic kinetics (τ_decay=300ms) gate NMDA NR2B insertion in CA1.
            # Phase-locked to theta: ACh fires at peaks (encoding phase), boosting
            # NMDA-dependent plasticity during Schaffer collateral activation.
            septal_ach_spikes = self._extract_neuromodulator(neuromodulator_inputs, 'ach_septal')
            self._ach_septal_concentration_ca1 = self.ach_septal_receptor.update(septal_ach_spikes)
        else:
            # Neuromodulation disabled: keep baseline concentrations
            pass

        dg_da_level = self._da_concentration_dg.mean().item()
        ca3_da_level = self._da_concentration_ca3.mean().item()
        ca2_da_level = self._da_concentration_ca2.mean().item()
        ca1_da_level = self._da_concentration_ca1.mean().item()

        # =====================================================================
        # INTERNAL WEIGHT HANDLES
        # =====================================================================
        # Cached each forward pass for use in integrate + STDP stages.

        dg_ca3_synapse = SynapseId(self.region_name, HippocampusPopulation.DG, self.region_name, HippocampusPopulation.CA3, receptor_type=ReceptorType.AMPA)
        ca3_ca3_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3, self.region_name, HippocampusPopulation.CA3, receptor_type=ReceptorType.AMPA)
        ca3_ca2_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3, self.region_name, HippocampusPopulation.CA2, receptor_type=ReceptorType.AMPA)
        ca3_ca1_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3, self.region_name, HippocampusPopulation.CA1, receptor_type=ReceptorType.AMPA)
        ca2_ca1_synapse = SynapseId(self.region_name, HippocampusPopulation.CA2, self.region_name, HippocampusPopulation.CA1, receptor_type=ReceptorType.AMPA)

        dg_ca3_weights = self.get_synaptic_weights(dg_ca3_synapse)
        ca3_ca3_weights = self.get_synaptic_weights(ca3_ca3_synapse)
        ca3_ca2_weights = self.get_synaptic_weights(ca3_ca2_synapse)
        ca3_ca1_weights = self.get_synaptic_weights(ca3_ca1_synapse)
        ca2_ca1_weights = self.get_synaptic_weights(ca2_ca1_synapse)

        # ripple_detected is set below, after CA3 spikes are computed
        self.ripple_detected = False

        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        dg_input = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.dg_size,
            filter_by_target_population=HippocampusPopulation.DG
        ).g_ampa

        ca3_input = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.ca3_size,
            filter_by_target_population=HippocampusPopulation.CA3
        ).g_ampa

        # TODO: Integrate CA2 input here as well (currently only from CA3, but could add EC→CA2)

        ca1_input = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.ca1_size,
            filter_by_target_population=HippocampusPopulation.CA1
        ).g_ampa

        # =====================================================================
        # GET SEPTAL INPUT (for OLM phase-locking)
        # =====================================================================
        # Septal GABAergic input drives theta rhythm by phase-locking OLM cells
        # OLM cells rebound at theta troughs → dendritic inhibition → encoding/retrieval
        septal_gaba = synaptic_inputs.get("septal_gaba", None)
        if septal_gaba is not None and septal_gaba.numel() == 0:
            septal_gaba = None  # Treat empty tensor as None

        # Initialize encoding/retrieval modulation from previous timestep's OLM state
        # This ensures causal flow: previous OLM activity determines current encoding/retrieval
        encoding_mod = self._prev_encoding_mod
        retrieval_mod = 1.0 - self._prev_encoding_mod

        # =====================================================================
        # STIMULUS GATING (TRANSIENT INHIBITION)
        # =====================================================================
        # Compute stimulus-onset inhibition based on DG input change
        # Use DG input as proxy for overall input change
        ffi = self.stimulus_gating.compute(dg_input, return_tensor=False)
        raw_ffi = ffi.item() if hasattr(ffi, "item") else float(ffi)
        # Normalize to [0, 1] by dividing by max_inhibition
        ffi_strength = min(1.0, raw_ffi / self.stimulus_gating.max_inhibition)
        ffi_factor = 1.0 - ffi_strength * cfg.ffi_strength

        # =====================================================================
        # DENTATE GYRUS: Pattern Separation
        # =====================================================================
        # DG input already integrated from all sources above
        # Apply FFI: reduce DG drive when input changes significantly
        dg_total_input = dg_input * ffi_factor

        dg_g_exc = F.relu(dg_total_input)  # Ensure non-negative conductance

        # =====================================================================
        # DG INHIBITORY NETWORK
        # =====================================================================
        dg_inhib_output = self.dg_inhibitory.forward(
            pyr_spikes=self._dg_spike_buffer.read(1),
            septal_gaba=septal_gaba,
            external_exc=None,
        )
        dg_perisomatic_inhib = dg_inhib_output["perisomatic"]

        dg_g_ampa, dg_g_nmda = split_excitatory_conductance(dg_g_exc, nmda_ratio=0.05)  # Reduced from 0.2

        dg_spikes, _ = self.dg_neurons.forward(
            g_ampa_input=ConductanceTensor(dg_g_ampa),
            g_nmda_input=ConductanceTensor(dg_g_nmda),
            g_gaba_a_input=ConductanceTensor(dg_perisomatic_inhib),
            g_gaba_b_input=None,
        )

        dg_spikes_delayed = self._dg_ca3_buffer.read(self._dg_ca3_delay_steps)

        dg_spikes_float = dg_spikes.float()
        dg_spikes_delayed_float = dg_spikes_delayed.float()

        # =====================================================================
        # CA3: Pattern Completion via Recurrence + Bistable Dynamics
        # =====================================================================
        # EMERGENT THETA MODULATION via OLM cells:
        # Instead of hardcoded arithmetic, encoding/retrieval separation emerges from:
        # 1. Septal GABA inhibits OLM cells at theta peaks
        # 2. OLM cells rebound at theta troughs (rebound bursting)
        # 3. OLM → CA1 apical dendrites suppresses retrieval pathway
        # 4. When OLM fires (theta trough), dendritic inhibition blocks EC→CA1
        # 5. When OLM silent (theta peak), EC→CA1 flows freely (retrieval)
        #
        # BISTABLE NEURONS: Real CA3 pyramidal neurons have intrinsic bistability
        # via I_NaP (persistent sodium) and I_CAN (Ca²⁺-activated cation) currents.
        # We model this with a persistent activity trace that:
        #   1. Accumulates when neurons fire
        #   2. Decays slowly (τ ~100-200ms)
        #   3. Provides positive feedback (self-sustaining activity)
        # This enables stable attractor states during delay periods.

        # Feedforward from DG (mossy fibers, theta-gated)
        # Mossy fibers are FACILITATING - repeated DG spikes progressively enhance transmission to CA3
        stp_efficacy = self.stp_modules[dg_ca3_synapse].forward(dg_spikes_delayed_float)
        effective_w_dg_ca3 = dg_ca3_weights * stp_efficacy.T
        ca3_from_dg = torch.matmul(effective_w_dg_ca3, dg_spikes_delayed_float)  # [ca3_size]

        # NOTE: ca3_input from _integrate_synaptic_inputs_at_dendrites includes
        # ALL external sources (EC, cortex, PFC, thalamus, etc.) but NOT DG
        # (DG→CA3 is computed separately above to apply STP)
        ca3_ff = ca3_input + ca3_from_dg

        # =====================================================================
        # CA3 RECURRENT INPUT (Internal Computation)
        # =====================================================================
        # CA3 recurrent connections use CircularDelayBuffer with proper 3ms delay.
        # This prevents instant feedback that causes pathological synchronization.
        #
        # ACh modulation applied here (region-level neuromodulation):
        # High ACh (encoding mode): Suppress recurrence
        # Low ACh (retrieval mode): Full recurrence
        ach_level = self._ach_concentration_ca3.mean().item()
        ach_recurrent_modulation = compute_ach_recurrent_suppression(ach_level)

        # Compute recurrent input from DELAYED CA3 activity
        # CRITICAL: Recurrent buffer stores current spikes with N-timestep delay
        # Read delayed spikes (3ms ago, written after previous spike generation)
        ca3_delayed = self._ca3_ca3_buffer.read(self._ca3_ca3_buffer.max_delay)

        # Compute recurrent input from delayed activity
        ca3_rec_raw = torch.matmul(ca3_ca3_weights, ca3_delayed.float())

        # Apply region-level modulation (ACh and strength scaling)
        ca3_rec = ca3_rec_raw * ach_recurrent_modulation  # [ca3_size]

        # =====================================================================
        # ACTIVITY-DEPENDENT FEEDBACK INHIBITION (Biologically Accurate)
        # =====================================================================
        # In real CA3, pyramidal cells recruit basket cell interneurons which
        # provide lateral inhibition back to the pyramidal population.
        # This creates local competition and prevents runaway activity.

        prev_ca3_spikes = self._ca3_spike_buffer.read(1)

        # Local lateral inhibition: sparse connectivity pattern
        ca3_ca3_inhib_synapse = SynapseId(
            source_region=self.region_name,
            source_population=HippocampusPopulation.CA3,
            target_region=self.region_name,
            target_population=HippocampusPopulation.CA3,
            receptor_type=ReceptorType.GABA_A,
        )
        ca3_ca3_inhib_weights = self.get_synaptic_weights(ca3_ca3_inhib_synapse)
        ca3_feedback_inhibition = torch.matmul(ca3_ca3_inhib_weights, prev_ca3_spikes.float())

        # =====================================================================
        # BISTABLE PERSISTENT ACTIVITY (models I_NaP / I_CAN currents)
        # =====================================================================
        # The persistent activity trace provides a "memory" of recent firing.
        # This is computed BEFORE updating spikes so that the persistent
        # contribution reflects the stable pattern, not the current noise.
        #
        # Key insight: The persistent activity acts like a slow capacitor that
        # charges when neurons fire and provides sustained current afterwards.

        # Persistent activity provides additional input current
        # This is the key mechanism for bistability: once a neuron starts firing,
        # its persistent activity helps keep it firing

        # Total CA3 excitatory input = feedforward + recurrent + persistent
        ca3_excitatory_input = ca3_ff + ca3_rec

        # =====================================================================
        # NOREPINEPHRINE GAIN MODULATION (Locus Coeruleus)
        # =====================================================================
        # High NE (arousal/uncertainty): Increase gain → more responsive
        # Low NE (baseline): Normal gain
        # Biological: β-adrenergic receptors increase neuronal excitability
        ne_level = self._ne_concentration_ca3.mean().item()
        # NE gain: 1.0 (baseline) to 1.5 (high arousal)
        ne_gain = compute_ne_gain(ne_level)
        ca3_excitatory_input = ca3_excitatory_input * ne_gain

        # INTRINSIC PLASTICITY: Apply per-neuron threshold offset
        # Neurons that fire too much have higher thresholds (less excitable)
        ca3_excitatory_input = ca3_excitatory_input - self._ca3_threshold_offset

        # =====================================================================
        # CA3 INHIBITORY NETWORK (PV, OLM, Bistratified)
        # =====================================================================
        # Run CA3 inhibitory network with septal input
        # OLM cells phase-lock to septal GABA for theta modulation
        # CRITICAL: Use PREVIOUS timestep's activity (causal inhibition)
        ca3_inhib_output = self.ca3_inhibitory.forward(
            pyr_spikes=prev_ca3_spikes,
            septal_gaba=septal_gaba,
            external_exc=None,  # Optional external drive
        )
        ca3_perisomatic_inhib = ca3_inhib_output["perisomatic"]

        # Run through CA3 neurons (ConductanceLIF expects g_ampa, g_gaba_a, g_nmda)
        # Excitatory conductance from excitatory inputs
        ca3_g_exc = F.relu(ca3_excitatory_input)

        # =====================================================================
        # CONDUCTANCE VALIDATION (Type Safety Check)
        # =====================================================================
        # Ensure we're passing true conductances, not currents
        # Biological conductance range: [0, 5] relative to leak (g_L = 1)
        # If this assertion fails, it indicates current/conductance confusion
        if ca3_g_exc.max() > 10.0:
            # Emergency warning (not assertion to avoid training crashes)
            print(f"\u26a0\ufe0f  WARNING: CA3 g_exc={ca3_g_exc.max():.2f} exceeds biological range [0, 5]!")
            print(f"  This indicates current/conductance confusion in synaptic weights.")
            print(f"  ca3_from_dg={ca3_from_dg.mean():.4f}, ca3_rec={ca3_rec.mean():.4f}")
            # Clamp to prevent immediate crash, but this needs fixing
            ca3_g_exc = torch.clamp(ca3_g_exc, 0.0, 5.0)

        # Apply inhibition: Feedback (population) + Phasic (from interneurons) + Tonic (ambient GABA)
        # Feedback inhibition scales with population activity to prevent runaway synchrony
        # Phasic inhibition from fast-spiking PV interneurons
        # Tonic inhibition from extrasynaptic GABA_A receptors provides constant baseline
        ca3_g_inh = F.relu(ca3_feedback_inhibition) + F.relu(ca3_perisomatic_inhib) + cfg.tonic_inhibition

        # Add instantaneous self-inhibition to prevent burst escalation
        # This uses membrane potential as a proxy for imminent spiking
        # (spikes haven't happened yet, but high V_mem indicates burst risk)
        # REDUCED from 0.4 to 0.1: Strong self-inhibition was blocking normal firing
        # Only apply when V_mem is VERY close to threshold (prevents premature inhibition)
        # Normalize membrane potential using actual neuron parameters
        v_rest = self.ca3_neurons.E_L.item()
        v_threshold = self.ca3_neurons.v_threshold.mean().item()
        v_normalized = (self.ca3_neurons.V_soma - v_rest) / (v_threshold - v_rest)
        v_normalized = torch.clamp(v_normalized, 0.0, 1.0)
        # Only apply strong inhibition when > 80% of way to threshold
        # This allows normal firing while preventing bursts
        burst_risk = torch.clamp((v_normalized - 0.8) / 0.2, 0.0, 1.0)
        ca3_g_inh = ca3_g_inh + burst_risk * 0.1

        # Split excitatory conductance into AMPA (fast) and NMDA (slow)
        ca3_g_ampa, ca3_g_nmda = split_excitatory_conductance(ca3_g_exc, nmda_ratio=0.2)

        # Run through CA3 neurons (two-compartment).
        # All current inputs are perisomatic (basal): DG mossy fibers, CA3 recurrents,
        # persistent activity, and perisomatic interneuron inhibition.
        # Apical compartment is available for future EC direct-path or PFC feedback.
        ca3_spikes, _, _ = self.ca3_neurons.forward(
            g_ampa_basal=ConductanceTensor(ca3_g_ampa),
            g_nmda_basal=ConductanceTensor(ca3_g_nmda),
            g_gaba_a_basal=ConductanceTensor(ca3_g_inh),
            g_gaba_b_basal=ConductanceTensor(ca3_inhib_output["perisomatic_gaba_b"]),
            g_ampa_apical=None,   # No EC direct input to CA3 in current model
            g_nmda_apical=None,
            g_gaba_a_apical=None,
        )

        ca3_spikes_float = ca3_spikes.float()

        # Emergent sharp-wave ripple detection: synchronous CA3 burst > 5% of population
        # Biology: SWRs are characterised by >5% of CA3 pyramidals firing in a 1-5ms window.
        # No explicit trigger — ripples must emerge from network dynamics (attractor recall,
        # low ACh disinhibiting recurrence, GABA_B-terminated bursts).
        self.ripple_detected = ca3_spikes_float.mean().item() > 0.05

        # Learning happens only when there's CA3 activity AND learning is enabled
        if not GlobalConfig.LEARNING_DISABLED and ca3_spikes.any():
            # Strong dopamine gating: 0.0 DA = 20% learning, 1.0 DA = 200% learning
            ca3_da_gain = 0.2 + 1.8 * ca3_da_level  # Range: [0.2, 2.0]
            effective_lr = cfg.learning_rate * encoding_mod * ca3_da_gain

            dW = torch.outer(ca3_spikes_float, ca3_spikes_float)

            # =========================================================
            # HETEROSYNAPTIC PLASTICITY: Weaken inactive synapses
            # =========================================================
            # Synapses to inactive postsynaptic neurons get weakened when
            # nearby neurons fire strongly. This prevents winner-take-all
            # dynamics from permanently dominating.
            if cfg.heterosynaptic_ratio > 0:
                inactive_ca3_post = torch.logical_not(ca3_spikes).float()
                active_ca3_pre = ca3_spikes_float
                hetero_dW = -cfg.heterosynaptic_ratio * torch.outer(active_ca3_pre, inactive_ca3_post)
                dW = dW + hetero_dW

            # Apply instantaneous weight delta only.  The old code applied the FULL
            # accumulated fast trace (τ=60 s, near-zero decay at 1 ms/step) as a
            # weight delta every timestep, giving O(n²) weight growth that saturated
            # w_max within ~15 steps and caused the runaway ca3_rec we observed.
            # DA-gated long-term consolidation is handled by the tag-and-capture
            # three-factor rule below.
            #
            # SYNAPSE MASK: Only update existing synaptic contacts (weight > 0).
            # Hebbian LTP acts on existing synapses.  Adding to zero entries would
            # create de-novo connections from noise, converting the sparse CA3→CA3
            # matrix to dense and causing runaway amplification.
            ca3_syn_mask = (ca3_ca3_weights.data > 0).float()
            ca3_ca3_weights.data += dW * effective_lr * ca3_syn_mask
            ca3_ca3_weights.data.fill_diagonal_(0.0)  # Maintain no self-connections (biological constraint)
            clamp_weights(ca3_ca3_weights.data, cfg.w_min, cfg.w_max)


        # =====================================================================
        # DG→CA3 MOSSY FIBER LEARNING (One-Shot Binding)
        # =====================================================================
        # Biology: Mossy fiber synapses show rapid, powerful LTP
        # They're among the largest synapses in the brain ("detonator" synapses)
        # that reliably drive postsynaptic firing. Critical for binding sparse DG
        # pattern-separated representations to CA3 autoassociative attractors.
        #
        # Learning characteristics:
        # - ONE-SHOT: Single co-activation creates strong LTP (rapid binding)
        # - STRONG: Large synapses with powerful transmission
        # - SPARSE: Only ~50 mossy fiber inputs per CA3 neuron (vs ~12,000 recurrent)
        # - MODULATED: Enhanced by novelty (dopamine) and encoding state (ACh, theta)
        #
        # This learning is what allows DG to "index" memories in CA3 - each sparse
        # DG pattern becomes associated with a dense CA3 attractor pattern.

        # Learning happens only when there's co-activity (DG and CA3 both firing) AND learning is enabled
        if not GlobalConfig.LEARNING_DISABLED and dg_spikes_delayed.any() and ca3_spikes.any():
            # Hebbian outer product: bind DG pattern to CA3 attractor
            # Shape: [ca3_size, dg_size] - each CA3 neuron learns from DG inputs
            dW_mossy = torch.outer(ca3_spikes_float, dg_spikes_delayed_float)

            # ONE-SHOT learning: Use STRONG learning rate (3x normal)
            # Biology: Mossy fiber LTP is rapid and doesn't require multiple pairings
            ca3_da_gain = 0.2 + 1.8 * ca3_da_level  # Strong dopamine gating
            mossy_effective_lr = cfg.learning_rate * 3.0 * encoding_mod * ca3_da_gain

            # Apply instantaneous delta only (same fix as CA3→CA3 above).
            # Mask to existing mossy fiber contacts only (15% connectivity);
            # Hebbian should not create new synaptic contacts from noise.
            dg_ca3_syn_mask = (dg_ca3_weights.data > 0).float()
            dg_ca3_weights.data += dW_mossy * mossy_effective_lr * dg_ca3_syn_mask
            clamp_weights(dg_ca3_weights.data, cfg.w_min, cfg.w_max)

        # =====================================================================
        # CA2: Social Memory and Temporal Context Layer
        # =====================================================================
        # CA2 sits between CA3 and CA1, providing:
        # - Temporal context encoding (when events occurred)
        # - Social information processing (future: agent interactions)
        # - Stability mechanism (weak CA3→CA2 plasticity prevents interference)
        #
        # Key properties:
        # - Receives CA3 input (but resists CA3 pattern completion)
        # - Strong direct EC input (for temporal encoding)
        # - Projects to CA1 (providing context to decision layer)

        # APPLY CA3→CA2 AXONAL DELAY
        ca3_spikes_for_ca2 = self._ca3_ca2_buffer.read(self._ca3_ca2_delay_steps)
        ca3_spikes_for_ca2_float = ca3_spikes_for_ca2.float()

        # CA3→CA2 input with STP (depressing - stability mechanism)
        stp_efficacy = self.stp_modules[ca3_ca2_synapse].forward(ca3_spikes_for_ca2_float)
        effective_w_ca3_ca2 = ca3_ca2_weights * stp_efficacy.T
        ca2_from_ca3 = torch.matmul(effective_w_ca3_ca2, ca3_spikes_for_ca2_float)

        # =====================================================================
        # CA2 FEEDBACK INHIBITION (Prevent Hyperactivity)
        # =====================================================================
        # Similar to CA3, CA2 pyramidal cells recruit interneurons for
        # population-level feedback inhibition to prevent runaway activity.
        # CA2 is particularly prone to amplification due to strong CA3 input.

        # Compute CA2 feedback inhibition with both global and local components
        # BIOLOGICAL FIX: Feedback inhibition is applied as CONDUCTANCE, not subtracted!
        # CA2 lateral inhibition (for spatial selectivity)
        prev_ca2_spikes = self._ca2_spike_buffer.read(1)
        ca2_population_rate = prev_ca2_spikes.float().mean()

        ca2_ca2_inhib_synapse = SynapseId(
            source_region=self.region_name,
            source_population=HippocampusPopulation.CA2,
            target_region=self.region_name,
            target_population=HippocampusPopulation.CA2,
            receptor_type=ReceptorType.GABA_A,
        )
        ca2_ca2_inhib_weights = self.get_synaptic_weights(ca2_ca2_inhib_synapse)
        local_lateral_ca2 = torch.matmul(ca2_ca2_inhib_weights, prev_ca2_spikes.float())
        ca2_feedback_inhibition = ca2_population_rate * 0.2 + local_lateral_ca2 * 0.1

        ca2_g_exc = F.relu(ca2_from_ca3)

        # =====================================================================
        # CA2 INHIBITORY NETWORK
        # =====================================================================
        # Apply moderate inhibition for temporal/social context processing
        # CRITICAL: Use PREVIOUS timestep's activity (causal inhibition)
        ca2_inhib_output = self.ca2_inhibitory.forward(
            pyr_spikes=self._ca2_spike_buffer.read(1),
            septal_gaba=septal_gaba,
            external_exc=ca2_g_exc,
        )
        ca2_perisomatic_inhib = ca2_inhib_output["perisomatic"]
        # Apply inhibition: Tonic + Feedback + Phasic (conductance-based)
        ca2_g_inh = F.relu(cfg.tonic_inhibition + ca2_feedback_inhibition + ca2_perisomatic_inhib)

        # Add instantaneous self-inhibition to prevent burst escalation
        # Uses membrane potential as proxy for imminent spiking
        # REDUCED from 0.4 to 0.1: Strong self-inhibition was blocking normal firing
        # Only apply when V_mem is VERY close to threshold (prevents premature inhibition)
        # Normalize membrane potential using actual neuron parameters
        v_rest = self.ca2_neurons.E_L.item()
        v_threshold = self.ca2_neurons.v_threshold.mean().item()
        v_normalized = (self.ca2_neurons.V_soma - v_rest) / (v_threshold - v_rest)
        v_normalized = torch.clamp(v_normalized, 0.0, 1.0)
        # Only apply strong inhibition when > 80% of way to threshold
        burst_risk = torch.clamp((v_normalized - 0.8) / 0.2, 0.0, 1.0)
        ca2_g_inh = ca2_g_inh + burst_risk * 0.1

        # Split excitatory conductance into AMPA (fast) and NMDA (slow)
        ca2_g_ampa, ca2_g_nmda = split_excitatory_conductance(ca2_g_exc, nmda_ratio=0.05)  # Reduced from 0.2

        ca2_spikes, _ = self.ca2_neurons.forward(
            g_ampa_input=ConductanceTensor(ca2_g_ampa),
            g_nmda_input=ConductanceTensor(ca2_g_nmda),
            g_gaba_a_input=ConductanceTensor(ca2_perisomatic_inhib),
            g_gaba_b_input=ConductanceTensor(ca2_inhib_output["perisomatic_gaba_b"]),
        )
        ca2_spikes_float = ca2_spikes.float()

        # CA3→CA2 WEAK PLASTICITY (stability mechanism)
        # Learning only when there's activity AND learning is enabled
        if not GlobalConfig.LEARNING_DISABLED and ca3_spikes_for_ca2.any() and ca2_spikes.any():
            # Very weak learning rate (stability hub)
            effective_lr = cfg.learning_rate * encoding_mod

            dW = torch.outer(ca2_spikes_float, ca3_spikes_for_ca2_float)

            # Apply instantaneous delta only (same fix as CA3→CA3 above).
            # Mask to existing contacts only.
            ca3_ca2_syn_mask = (ca3_ca2_weights.data > 0).float()
            ca3_ca2_weights.data += dW * effective_lr * ca3_ca2_syn_mask
            clamp_weights(ca3_ca2_weights.data, cfg.w_min, cfg.w_max)

        # =====================================================================
        # APPLY CA3→CA1 AXONAL DELAY
        # =====================================================================
        # Apply biological transmission delay for CA3→CA1 Schaffer collateral pathway
        # If delay is 0, ca3_spikes_delayed = ca3_spikes (instant)
        ca3_spikes_delayed = self._ca3_ca1_buffer.read(self._ca3_ca1_delay_steps)

        # Feedforward from CA3 (retrieved/encoded memory) with optional STP
        # Schaffer collaterals are DEPRESSING - high-frequency CA3 activity
        # causes progressively weaker transmission to CA1
        # NOTE: Use delayed CA3 spikes for biological accuracy
        stp_efficacy = self.stp_modules[ca3_ca1_synapse].forward(ca3_spikes_delayed.float())
        effective_w_ca3_ca1 = ca3_ca1_weights * stp_efficacy.T
        ca1_from_ca3 = torch.matmul(effective_w_ca3_ca1, ca3_spikes_delayed.float())  # [ca1_size]

        # Apply feedforward inhibition: strong input change reduces CA1 drive
        # This clears residual activity naturally
        ca1_from_ec = ca1_input * ffi_factor
        ca1_from_ca3 = ca1_from_ca3 * ffi_factor

        # NMDA trace update (for retrieval gating)
        # Tracks CA3-induced depolarization for Mg²⁺ block removal
        if self.nmda_trace is not None:
            nmda_decay = torch.exp(torch.tensor(-dt_ms / cfg.nmda_tau))
            self.nmda_trace = self.nmda_trace * nmda_decay + ca1_from_ca3 * (1.0 - nmda_decay)
        else:
            self.nmda_trace = ca1_from_ca3

        # NMDA gating: Mg²⁺ block removal based on CA3 depolarization
        # Stronger during retrieval (theta peak)
        mg_block_removal = torch.sigmoid((self.nmda_trace - cfg.nmda_threshold) * cfg.nmda_steepness) * retrieval_mod
        nmda_conductance = ca1_from_ec * mg_block_removal

        # Muscarinic NR2B enhancement: septal ACh promotes NR2B-containing NMDA
        # subunit insertion (slow, ~300ms time constant). This boosts Ca²⁺ influx
        # during encoding, lowering the LTP induction threshold at Schaffer synapses.
        # Scale: up to +50% NMDA conductance at saturating septal ACh levels.
        ach_septal_mean = self._ach_septal_concentration_ca1.mean().item()
        nmda_nr2b_boost = 1.0 + ach_septal_mean * 0.5
        nmda_conductance = nmda_conductance * nmda_nr2b_boost

        # AMPA conductance: fast baseline transmission
        ampa_conductance = ca1_from_ec * cfg.ampa_ratio

        # CA3 contribution: stronger during encoding
        ca3_contribution = ca1_from_ca3 * (0.5 + 0.5 * encoding_mod)

        # APPLY CA2→CA1 AXONAL DELAY
        ca2_spikes_delayed = self._ca2_ca1_buffer.read(self._ca2_ca1_delay_steps)
        ca2_spikes_delayed_float = ca2_spikes_delayed.float()

        # CA2→CA1 contribution with STP (facilitating - temporal sequences)
        stp_efficacy = self.stp_modules[ca2_ca1_synapse].forward(ca2_spikes_delayed_float)
        effective_w_ca2_ca1 = ca2_ca1_weights * stp_efficacy.T
        ca1_from_ca2 = torch.matmul(effective_w_ca2_ca1, ca2_spikes_delayed_float)

        # Apply FFI to CA2 contribution as well
        ca1_from_ca2 = ca1_from_ca2 * ffi_factor

        # =====================================================================
        # CA1 INHIBITORY NETWORK (PV, OLM, Bistratified)
        # =====================================================================
        # Run CA1 inhibitory network with septal input
        # OLM cells create EMERGENT encoding/retrieval separation!
        ca1_inhib_output = self.ca1_inhibitory.forward(
            pyr_spikes=self._ca1_spike_buffer.read(1),
            septal_gaba=septal_gaba,
            external_exc=ca3_contribution,
        )
        ca1_perisomatic_inhib = ca1_inhib_output["perisomatic"]  # PV cells
        ca1_dendritic_inhib = ca1_inhib_output["dendritic"]      # OLM + bistratified
        ca1_olm_inhib = ca1_inhib_output["olm_dendritic"]        # OLM only

        # =====================================================================
        # EMERGENT ENCODING/RETRIEVAL from OLM dynamics
        # =====================================================================
        # Instead of hardcoded sinusoid, encoding/retrieval emerges from OLM activity:
        # - High OLM activity → strong dendritic inhibition → encoding phase
        # - Low OLM activity → weak dendritic inhibition → retrieval phase
        #
        # Compute modulation from OLM firing rate (inverted for retrieval)
        # Store for NEXT timestep (current timestep already used previous values)
        olm_firing_rate = ca1_olm_inhib.mean().item()  # [0, ~1]
        # Encoding high when OLM fires (suppresses retrieval pathway)
        # Add baseline encoding so learning isn't completely blocked
        # when OLM cells are silent during early training. Real hippocampus has tonic
        # acetylcholine that provides baseline encoding drive even without septal input.
        self._prev_encoding_mod = torch.clamp(torch.tensor(0.3 + olm_firing_rate * 2.0), 0.0, 1.0).item()

        # -----------------------------------------------------------------------
        # TWO-COMPARTMENT CA1: Separate basal and apical pathways
        # -----------------------------------------------------------------------
        # Basal (proximal) compartment: CA3 Schaffer collateral + CA2 temporal context.
        # These arrive at basal/oblique dendrites of CA1 pyramidal cells.
        ca1_basal_g_exc = ca3_contribution + ca1_from_ca2
        ca1_basal_g_ampa, ca1_basal_g_nmda = split_excitatory_conductance(ca1_basal_g_exc, nmda_ratio=0.2)

        # Apical (distal) compartment: EC direct perforant path (ampa + nmda retrieval gating).
        # EC→CA1 direct path targets apical tuft dendrites — the key retrieval route.
        ca1_apical_g_exc = ampa_conductance + nmda_conductance
        ca1_apical_g_ampa, ca1_apical_g_nmda = split_excitatory_conductance(ca1_apical_g_exc, nmda_ratio=0.3)

        # Add CA1 feedback inhibition (like CA3) to prevent hyperactivity
        # CA1 pyramidal cells recruit basket cells for lateral inhibition
        # Simple implementation: population activity suppresses all neurons equally
        prev_ca1_spikes = self._ca1_spike_buffer.read(1)
        ca1_feedback_inhib = prev_ca1_spikes.float().mean() * 0.1  # Light feedback strength

        # Perisomatic (basal) inhibition: PV basket cells + feedback + tonic
        ca1_g_inh = F.relu(cfg.tonic_inhibition + ca1_perisomatic_inhib + ca1_feedback_inhib)

        # Add instantaneous self-inhibition to prevent burst escalation
        # Uses membrane potential as proxy for imminent spiking
        # Only apply when V_mem is VERY close to threshold (prevents premature inhibition)
        # Normalize membrane potential using actual neuron parameters
        v_rest = self.ca1_neurons.E_L.item()
        v_threshold = self.ca1_neurons.v_threshold.mean().item()
        v_normalized = (self.ca1_neurons.V_soma - v_rest) / (v_threshold - v_rest)
        v_normalized = torch.clamp(v_normalized, 0.0, 1.0)
        burst_risk = torch.clamp((v_normalized - 0.8) / 0.2, 0.0, 1.0)
        ca1_g_inh = ca1_g_inh + burst_risk * 0.1

        # Run through CA1 neurons (two-compartment).
        # OLM dendritic inhibition goes directly to the apical compartment
        # (biologically correct: OLM axons target distal apical dendrites, NOT soma).
        ca1_spikes, _ca1_membrane, _ca1_V_dend = self.ca1_neurons.forward(
            g_ampa_basal=ConductanceTensor(ca1_basal_g_ampa),
            g_nmda_basal=ConductanceTensor(ca1_basal_g_nmda),
            g_gaba_a_basal=ConductanceTensor(ca1_g_inh),
            g_gaba_b_basal=ConductanceTensor(ca1_inhib_output["perisomatic_gaba_b"]),
            g_ampa_apical=ConductanceTensor(ca1_apical_g_ampa),
            g_nmda_apical=ConductanceTensor(ca1_apical_g_nmda),
            g_gaba_a_apical=ConductanceTensor(F.relu(ca1_dendritic_inhib)),  # OLM → apical
        )
        ca1_spikes_float = ca1_spikes.float()

        if not GlobalConfig.LEARNING_DISABLED:
            # ---------------------------------------------------------
            # HEBBIAN LEARNING: CA2→CA1 plasticity (during encoding)
            # ---------------------------------------------------------
            # CA2 provides temporal/social context to CA1
            if ca2_spikes_delayed.any() and ca1_spikes.any():
                effective_lr = cfg.learning_rate * encoding_mod

                dW = torch.outer(ca1_spikes_float, ca2_spikes_delayed_float)

                # Apply instantaneous delta only (same fix as CA3→CA3 above).
                # Mask to existing contacts only.
                ca2_ca1_syn_mask = (ca2_ca1_weights.data > 0).float()
                ca2_ca1_weights.data += dW * effective_lr * ca2_ca1_syn_mask
                clamp_weights(ca2_ca1_weights.data, cfg.w_min, cfg.w_max)

            if encoding_mod > 0.5:
                # =====================================================================
                # EXTERNAL INPUT PLASTICITY (Per-Source Learning)
                # =====================================================================
                # Apply three-factor learning (eligibility × dopamine × learning_rate)
                # to all external input pathways. This learns which cortical patterns
                # are associated with hippocampal memories.
                #
                # Biological rationale: EC→hippocampus synapses show robust LTP/LTD
                # modulated by dopamine, VTA novelty signals, and behavioral outcomes.
                # This is how the hippocampus learns which sensory/contextual patterns
                # matter for episodic memory formation.
                #
                # Only apply during encoding mode (theta trough) when new memories
                # are being formed. During retrieval (theta peak), external inputs
                # serve as retrieval cues, not learning signals.

                # Get dopamine modulation (average across all subregions)
                da_modulation = (dg_da_level + ca3_da_level + ca2_da_level + ca1_da_level) / 4.0

                # Modulate learning rate by dopamine
                effective_lr = cfg.learning_rate * 0.3 * (1.0 + da_modulation)

                # Apply learning to each external input source
                for synapse_id, source_input in synaptic_inputs.items():
                    weights = self.get_synaptic_weights(synapse_id)

                    # SYNAPSE MASK: Only potentiate existing synaptic contacts.
                    # The outer product produces a dense matrix; adding it to a sparse
                    # weight matrix would create de-novo connections from noise,
                    # converting sparse EC→hippocampus connectivity to dense within ~200ms
                    # of baseline activity and causing runaway CA3 excitation.
                    ext_syn_mask = (weights.data > 0).float()

                    if synapse_id.target_population == HippocampusPopulation.DG:
                        # Learn EC→DG (pattern separation input)
                        # Hebbian learning: pre (source) × post (DG)
                        dW_dg = effective_lr * torch.outer(dg_spikes_float, source_input.float())
                        weights.data += dW_dg * ext_syn_mask

                    elif synapse_id.target_population == HippocampusPopulation.CA3:
                        # Learn EC→CA3 (direct perforant path for retrieval cues)
                        # Hebbian learning: pre (source) × post (CA3)
                        dW_ca3 = effective_lr * torch.outer(ca3_spikes_float, source_input.float())
                        weights.data += dW_ca3 * ext_syn_mask

                    elif synapse_id.target_population == HippocampusPopulation.CA1:
                        # Learn EC→CA1 (direct output pathway)
                        # Hebbian learning: pre (source) × post (CA1)
                        dW_ca1 = effective_lr * torch.outer(ca1_spikes_float, source_input.float())
                        weights.data += dW_ca1 * ext_syn_mask

                    clamp_weights(weights.data, cfg.w_min, cfg.w_max)

                # =====================================================================
                # CA3 RECURRENT PLASTICITY (Three-Factor Learning)
                # =====================================================================
                # Apply dopamine-modulated three-factor learning to CA3→CA3 recurrent weights.
                # This is CRITICAL for sequence learning - the recurrent connections store
                # temporal associations between patterns.
                #
                # Three-factor rule: ΔW = eligibility_trace × dopamine × learning_rate
                # - Eligibility: accumulated Hebbian correlations (STDP-like)
                # - Dopamine: reward/novelty signal from VTA
                # - Learning rate: modulated by encoding phase (theta)
                #
                # Biological rationale: CA3 recurrent synapses exhibit dopamine-gated LTP/LTD
                # that consolidates rewarded sequences into stable attractors.

                # Get delayed CA3 activity (what was just active before current spikes)
                ca3_delayed = self._ca3_ca3_buffer.read(self._ca3_ca3_delay_steps)

                # Three-factor rule needs DEVIATION from baseline (not absolute concentration)
                # Baseline ~0.5 → modulator = 0 (no learning)
                # Reward burst → modulator > 0 (strengthen synapses)
                # Punishment dip → modulator < 0 (weaken synapses)
                da_ca3_deviation = ca3_da_level - 0.5
                if self.get_learning_strategy(ca3_ca3_synapse) is None:
                    self._add_learning_strategy(ca3_ca3_synapse, self._tag_and_capture_strategy)
                self._apply_learning(
                    ca3_ca3_synapse, ca3_delayed, ca3_spikes,
                    modulator=da_ca3_deviation,
                )
                self.get_synaptic_weights(ca3_ca3_synapse).data.fill_diagonal_(0.0)  # Maintain no self-connections (biological constraint)

            # =====================================================================
            # DOPAMINE-GATED CONSOLIDATION (Tag-and-Capture)
            # =====================================================================
            # Consolidate tagged CA3→CA3 synapses when dopamine is elevated.
            # Tags were accumulated inside _apply_learning (via TagAndCaptureStrategy).
            if ca3_da_level > 0.1:
                ca3_ca3_weights.data = self._tag_and_capture_strategy.consolidate(ca3_da_level, ca3_ca3_weights)
                clamp_weights(ca3_ca3_weights.data, cfg.w_min, cfg.w_max)

        # =====================================================================
        # POST-LEARNING HOMEOSTATIC PLASTICITY
        # =====================================================================
        region_outputs: RegionOutput = {
            HippocampusPopulation.DG: dg_spikes,
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

        # Homeostatic adaptation for all populations (intrinsic excitability + threshold)
        self._apply_all_population_homeostasis(region_outputs)

        # Apply synaptic scaling to maintain stability (after all learning updates)
        self._apply_plasticity(region_outputs)

        # =====================================================================
        # UPDATE STATE BUFFERS FOR NEXT TIMESTEP
        # =====================================================================
        self._dg_spike_buffer.write_and_advance(dg_spikes)
        self._ca3_spike_buffer.write_and_advance(ca3_spikes)
        self._ca2_spike_buffer.write_and_advance(ca2_spikes)
        self._ca1_spike_buffer.write_and_advance(ca1_spikes)

        self._dg_ca3_buffer.write_and_advance(dg_spikes)
        self._ca3_ca3_buffer.write_and_advance(ca3_spikes)
        self._ca3_ca2_buffer.write_and_advance(ca3_spikes)
        self._ca3_ca1_buffer.write_and_advance(ca3_spikes)
        self._ca2_ca1_buffer.write_and_advance(ca2_spikes)

        return self._post_forward(region_outputs)

    def _apply_plasticity(self, region_outputs: RegionOutput) -> None:
        """
        Apply homeostatic plasticity to hippocampal synapses.

        Biologically, ALL hippocampal synapses exhibit both Hebbian and homeostatic plasticity:
        - Internal connections (ca3_ca3, ca3_ca2, ca2_ca1): Hebbian learning in forward()
        - External input pathways (EC→DG, EC→CA3, EC→CA1): Hebbian learning in forward()
        - Homeostatic mechanisms: Applied here to ALL pathways for stability

        The internal pathways use one-shot Hebbian (fast episodic binding) while
        external pathways use dopamine-gated learning (consolidates valuable patterns).
        Homeostatic plasticity (synaptic scaling, intrinsic plasticity) prevents
        runaway excitation and maintains stable network dynamics.

        NOTE: During active learning (three-factor rule), homeostatic scaling is
        TEMPORARILY DISABLED to allow learned patterns to consolidate. Otherwise
        the normalization washes out selectivity immediately after learning updates.
        """
        if GlobalConfig.LEARNING_DISABLED:
            return

        ca3_spikes = region_outputs[HippocampusPopulation.CA3]
        ca3_spikes_float = ca3_spikes.float()

        # =====================================================================
        # ADAPTIVE HOMEOSTATIC SYNAPTIC SCALING (Firing Rate-Based)
        # =====================================================================
        # Biological mechanism: Homeostatic synaptic scaling (Turrigiano et al.)
        # - Neurons measure average firing rate over hours
        # - If chronically too high/low, scale ALL synapses multiplicatively
        # - Preserves selectivity: strong synapses remain relatively stronger
        #
        # Implementation:
        # - Track CA3 population firing rate over 30-second sliding window
        # - Only scale if CHRONICALLY outside tolerance (>50% deviation)
        # - Multiplicative scaling (weights *= scale_factor)
        # - Very slow (5% adjustment) to not interfere with fast learning

        self._homeostasis_counter += 1

        # Track current firing rate every timestep
        current_rate = ca3_spikes_float.mean().item()
        # Ring-buffer write (circular overwrite of oldest value when full)
        max_hist = self._homeostasis_history.shape[0]
        self._homeostasis_history[self._homeostasis_write_idx % max_hist] = current_rate
        self._homeostasis_write_idx += 1
        self._homeostasis_filled = min(self._homeostasis_filled + 1, max_hist)

        # Check for chronic hyperactivity/hypoactivity every interval
        if self._homeostasis_counter >= self._homeostasis_interval:
            self._homeostasis_counter = 0

            # Need enough history to make decision
            if self._homeostasis_filled >= self._homeostasis_interval * 5:  # At least 5 seconds
                # Compute average firing rate over window
                hist_slice = self._homeostasis_history if self._homeostasis_filled >= max_hist else self._homeostasis_history[:self._homeostasis_filled]
                avg_firing_rate = hist_slice.mean().item()

                # Compute deviation from target (use per-population CA3 target)
                ca3_target = self._get_target_firing_rate(HippocampusPopulation.CA3)
                deviation = (avg_firing_rate - ca3_target) / (ca3_target + 1e-8)

                # Only act if chronically outside tolerance
                if abs(deviation) > 0.5:
                    ca3_ca3_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3, self.region_name, HippocampusPopulation.CA3, receptor_type=ReceptorType.AMPA)
                    ca3_ca2_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3, self.region_name, HippocampusPopulation.CA2, receptor_type=ReceptorType.AMPA)
                    ca2_ca1_synapse = SynapseId(self.region_name, HippocampusPopulation.CA2, self.region_name, HippocampusPopulation.CA1, receptor_type=ReceptorType.AMPA)

                    # Compute multiplicative scale factor
                    target_scale = ca3_target / (avg_firing_rate + 1e-8)
                    scale_factor = 1.0 + 0.05 * (target_scale - 1.0)
                    scale_factor = torch.clamp(torch.tensor(scale_factor), 0.9, 1.1).item()

                    # Apply MULTIPLICATIVE scaling to CA3 recurrent weights
                    # This preserves selectivity: if A=2*B before, A=2*B after
                    ca3_ca3_weights = self.get_synaptic_weights(ca3_ca3_synapse)
                    ca3_ca3_weights.data *= scale_factor
                    ca3_ca3_weights.data.fill_diagonal_(0.0)

                    # Also scale CA3→CA2 and CA2→CA1 (prevent downstream runaway)
                    self.get_synaptic_weights(ca3_ca2_synapse).data *= scale_factor
                    self.get_synaptic_weights(ca2_ca1_synapse).data *= scale_factor

                    # Apply same scaling to external inputs into CA3
                    for synapse_id in self.synaptic_weights.keys():
                        if synapse_id.target_population == HippocampusPopulation.CA3:
                            self.get_synaptic_weights(synapse_id).data *= scale_factor

                    # Enforce hard bounds after scaling
                    for synapse_id in [ca3_ca3_synapse, ca3_ca2_synapse, ca2_ca1_synapse]:
                        if synapse_id in self.synaptic_weights:
                            self.get_synaptic_weights(synapse_id).data.clamp_(self.config.w_min, self.config.w_max)

        # =====================================================================
        # INTRINSIC PLASTICITY (Threshold Adaptation)
        # =====================================================================
        # Update activity history (exponential moving average)
        self._ca3_activity_history.mul_(0.99).add_(ca3_spikes_float, alpha=0.01)

        # Use homeostasis helper for excitability modulation
        # This computes threshold offset based on activity deviation from target
        excitability_mod = compute_excitability_modulation(
            self._ca3_activity_history,
            activity_target=self._get_target_firing_rate(HippocampusPopulation.CA3),
            tau=100.0,
        )
        # Convert excitability modulation (>1 = easier, <1 = harder) to threshold offset
        # Higher excitability → lower threshold (subtract positive offset)
        # Lower excitability → higher threshold (subtract negative offset)
        self._ca3_threshold_offset = (1.0 - excitability_mod).clamp(-0.5, 0.5)

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

        # Update neurons
        self.dg_neurons.update_temporal_parameters(dt_ms)
        self.ca3_neurons.update_temporal_parameters(dt_ms)
        self.ca2_neurons.update_temporal_parameters(dt_ms)
        self.ca1_neurons.update_temporal_parameters(dt_ms)

        # Update inhibitory network
        self.dg_inhibitory.update_temporal_parameters(dt_ms)
        self.ca3_inhibitory.update_temporal_parameters(dt_ms)
        self.ca2_inhibitory.update_temporal_parameters(dt_ms)
        self.ca1_inhibitory.update_temporal_parameters(dt_ms)
