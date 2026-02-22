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

from typing import Optional

import torch
import torch.nn.functional as F

from thalia.brain.configs import HippocampusConfig
from thalia.brain.regions.population_names import HippocampusPopulation
from thalia.components import (
    NeuronFactory,
    NeuromodulatorReceptor,
    WeightInitializer,
    GapJunctionConfig,
    GapJunctionCoupling,
)
from thalia.components.synapses.stp import (
    MOSSY_FIBER_PRESET,
    SCHAFFER_COLLATERAL_PRESET,
)
from thalia.learning import (
    ThreeFactorConfig,
    ThreeFactorStrategy,
    compute_excitability_modulation,
)
from thalia.typing import (
    NeuromodulatorInput,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.units import ConductanceTensor
from thalia.utils import (
    CircularDelayBuffer,
    clamp_weights,
    compute_ach_recurrent_suppression,
    compute_ne_gain,
)

from .inhibitory_network import HippocampalInhibitoryNetwork
from .spontaneous_replay import SpontaneousReplayGenerator
from .synaptic_tagging import SynapticTagging

from ..neural_region import NeuralRegion
from ..region_registry import register_region
from ..stimulus_gating import StimulusGating


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

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: HippocampusConfig, population_sizes: PopulationSizes, region_name: RegionName):
        """Initialize trisynaptic hippocampus."""
        super().__init__(config, population_sizes, region_name)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.dg_size = population_sizes[HippocampusPopulation.DG.value]
        self.ca3_size = population_sizes[HippocampusPopulation.CA3.value]
        self.ca2_size = population_sizes[HippocampusPopulation.CA2.value]
        self.ca1_size = population_sizes[HippocampusPopulation.CA1.value]

        # =====================================================================
        # INTERNAL STATE VARIABLES (for dynamics and plasticity)
        # =====================================================================
        # Previous encoding/retrieval modulation (from OLM dynamics)
        # Used to maintain causal flow: t-1 OLM activity determines t encoding/retrieval
        # These are simple scalar state variables (not tensors), no buffers needed
        self._prev_encoding_mod: float = 0.5  # Default: balanced state

        # CA3 bistable persistent activity trace
        # Models I_NaP/I_CAN currents that allow neurons to maintain firing
        # without continuous external input. This is essential for stable
        # attractor states during delay periods.
        self.ca3_persistent: torch.Tensor
        self.register_buffer("ca3_persistent", torch.zeros(self.ca3_size, device=self.device))

        # NMDA trace for temporal integration (slow kinetics); None until first forward step
        self.nmda_trace: Optional[torch.Tensor]
        self.register_buffer("nmda_trace", None)

        # Stored DG pattern from sample phase (for match/mismatch detection)
        self.stored_dg_pattern: torch.Tensor
        self.register_buffer("stored_dg_pattern", torch.zeros(self.dg_size, device=self.device))

        # Spontaneous replay (sharp-wave ripple) detection
        self.ripple_detected: bool = False

        # =====================================================================
        # GAP JUNCTIONS (Electrical Synapses) - Config Setup
        # =====================================================================
        # Gap junction module will be initialized in _init_circuit_weights() after synaptic weights are set up
        self.gap_junctions_ca1: Optional[GapJunctionCoupling] = None

        # Override weights with trisynaptic circuit weights
        self._init_circuit_weights()

        # =====================================================================
        # HIPPOCAMPAL EXCITATORY NEURONS (LIF with adaptation for sparse coding)
        # =====================================================================
        # Create LIF neurons for each layer using factory functions
        # DG: Sparse coding requires high threshold
        self.dg_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=self.region_name,
            population_name=HippocampusPopulation.DG.value,
            n_neurons=self.dg_size,
            device=self.device,
            v_threshold=0.9,
            adapt_increment=0.30,  # Strong adaptation to enforce sparsity
            tau_adapt=120.0,  # Slow decay to persist across pattern presentations
        )
        # CA3 gets spike-frequency adaptation to prevent frozen attractors
        self.ca3_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA3.value,
            n_neurons=self.ca3_size,
            device=self.device,
            adapt_increment=config.adapt_increment,
            tau_adapt=config.adapt_tau,
            v_threshold=1.25,
        )
        # CA2: Social memory and temporal context - higher threshold for selectivity
        self.ca2_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA2.value,
            n_neurons=self.ca2_size,
            device=self.device,
            v_threshold=1.6,
            adapt_increment=0.25,  # Moderate adaptation for selectivity
            tau_adapt=100.0,  # Medium decay for temporal integration
        )
        # CA1: Output layer
        self.ca1_neurons = NeuronFactory.create_pyramidal_neurons(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA1.value,
            n_neurons=self.ca1_size,
            device=self.device,
            v_threshold=1.0,
            adapt_increment=0.20,  # Moderate adaptation to prevent runaway activity
            tau_adapt=80.0,  # Faster decay for responsive output layer
        )

        # =====================================================================
        # HIPPOCAMPAL INHIBITORY NETWORKS (with OLM cells for emergent theta)
        # =====================================================================
        # DG inhibitory network: Minimal inhibition for pattern separation
        # Moderate at 0.20 to prevent avalanches while maintaining sparse coding
        self.dg_inhibitory = HippocampalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=HippocampusPopulation.DG_INHIBITORY.value,
            pyr_size=self.dg_size,
            total_inhib_fraction=0.20,
            dt_ms=config.dt_ms,
            device=str(self.device),
        )

        # CA3 inhibitory network: Strong inhibition for pattern completion stability
        # INHIBITION INCREASED: 0.50 → 0.65 to control runaway recurrent activity (+563% growth)
        self.ca3_inhibitory = HippocampalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA3_INHIBITORY.value,
            pyr_size=self.ca3_size,
            total_inhib_fraction=0.65,
            dt_ms=config.dt_ms,
            device=str(self.device),
        )

        # CA2 inhibitory network: Social/temporal context processing
        self.ca2_inhibitory = HippocampalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA2_INHIBITORY.value,
            pyr_size=self.ca2_size,
            total_inhib_fraction=0.15,
            dt_ms=config.dt_ms,
            device=str(self.device),
        )

        # CA1 inhibitory network: PV, OLM, Bistratified cells
        # OLM cells phase-lock to septal GABA → emergent encoding/retrieval
        self.ca1_inhibitory = HippocampalInhibitoryNetwork(
            region_name=self.region_name,
            population_name=HippocampusPopulation.CA1_INHIBITORY.value,
            pyr_size=self.ca1_size,
            total_inhib_fraction=0.30,
            dt_ms=config.dt_ms,
            device=str(self.device),
        )

        # Stimulus gating module (transient inhibition at stimulus changes)
        self.stimulus_gating = StimulusGating(
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

        # Membrane state buffers for all subregions
        self._dg_membrane_buffer = CircularDelayBuffer(1, self.dg_size, self.device, torch.float32)
        self._ca3_membrane_buffer = CircularDelayBuffer(1, self.ca3_size, self.device, torch.float32)
        self._ca2_membrane_buffer = CircularDelayBuffer(1, self.ca2_size, self.device, torch.float32)
        self._ca1_membrane_buffer = CircularDelayBuffer(1, self.ca1_size, self.device, torch.float32)

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
        # CONSOLIDATION MODE
        # =====================================================================
        # Sleep/offline replay state variables
        # When _consolidation_mode=True, forward() spontaneously reactivates stored
        # CA3→CA1 patterns from episodic memory (simulates sharp-wave ripples).
        self._consolidation_mode: bool = False
        self._replay_cue: Optional[int] = None  # Episode index to replay

        # =====================================================================
        # PLASTICITY AND HOMEOSTASIS
        # =====================================================================
        # Intrinsic plasticity state (threshold adaptation)
        self._ca3_activity_history: torch.Tensor
        self._ca3_threshold_offset: torch.Tensor
        self.register_buffer("_ca3_activity_history", torch.zeros(self.ca3_size, device=self.device))
        self.register_buffer("_ca3_threshold_offset", torch.zeros(self.ca3_size, device=self.device))

        # Synaptic tagging for emergent priority
        # Tags mark recently-active synapses for consolidation
        # Provides biological priority mechanism without explicit Episode objects
        self.synaptic_tagging = SynapticTagging(
            n_neurons=self.ca3_size,
            device=self.device,
            tag_decay=0.95,
            tag_threshold=0.1,
        )

        # Spontaneous replay generator (sharp-wave ripples)
        # Occurs during low ACh (sleep/rest) for memory consolidation
        self.spontaneous_replay = SpontaneousReplayGenerator(
            ripple_rate_hz=2.0,  # Biological rate: 1-3 Hz during sleep
            ach_threshold=0.3,   # Ripples only below this ACh level
            ripple_refractory_ms=200.0,  # Minimum 200ms between ripples
            device=self.device,
        )

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

        # =========================================================================
        # MULTI-TIMESCALE CONSOLIDATION
        # =========================================================================
        # DG→CA3 (mossy fiber pathway - one-shot binding)
        # Biology: Mossy fiber synapses are "detonator" synapses with powerful LTP.
        # They bind sparse DG codes to CA3 attractors.
        self._dg_ca3_fast: torch.Tensor
        self._dg_ca3_slow: torch.Tensor
        self.register_buffer("_dg_ca3_fast", torch.zeros(self.ca3_size, self.dg_size, device=self.device))
        self.register_buffer("_dg_ca3_slow", torch.zeros(self.ca3_size, self.dg_size, device=self.device))

        # CA3 recurrent (autoassociative memory)
        self._ca3_ca3_fast: torch.Tensor
        self._ca3_ca3_slow: torch.Tensor
        self.register_buffer("_ca3_ca3_fast", torch.zeros(self.ca3_size, self.ca3_size, device=self.device))
        self.register_buffer("_ca3_ca3_slow", torch.zeros(self.ca3_size, self.ca3_size, device=self.device))

        # CA3→CA2 (temporal context)
        self._ca3_ca2_fast: torch.Tensor
        self._ca3_ca2_slow: torch.Tensor
        self.register_buffer("_ca3_ca2_fast", torch.zeros(self.ca2_size, self.ca3_size, device=self.device))
        self.register_buffer("_ca3_ca2_slow", torch.zeros(self.ca2_size, self.ca3_size, device=self.device))

        # CA2→CA1 (context to output)
        self._ca2_ca1_fast: torch.Tensor
        self._ca2_ca1_slow: torch.Tensor
        self.register_buffer("_ca2_ca1_fast", torch.zeros(self.ca1_size, self.ca2_size, device=self.device))
        self.register_buffer("_ca2_ca1_slow", torch.zeros(self.ca1_size, self.ca2_size, device=self.device))

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY
        # =====================================================================
        # Per-subregion firing rate trackers (exponential moving average)
        self.register_buffer("dg_firing_rate", torch.zeros(self.dg_size, device=self.device))
        self.register_buffer("ca3_firing_rate", torch.zeros(self.ca3_size, device=self.device))
        self.register_buffer("ca2_firing_rate", torch.zeros(self.ca2_size, device=self.device))
        self.register_buffer("ca1_firing_rate", torch.zeros(self.ca1_size, device=self.device))

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
        # LEARNING STRATEGY (Three-Factor Learning with Dopamine Modulation)
        # =====================================================================
        # Default to dopamine-modulated learning for reward-driven memory formation
        # Three-factor rule: ΔW = eligibility_trace × dopamine × learning_rate
        self._three_factor_strategy = ThreeFactorStrategy(ThreeFactorConfig(
            learning_rate=0.001,  # Conservative rate for stable learning
            eligibility_tau=100.0,  # Eligibility trace decay (ms) - matches temporal integration
            modulator_tau=50.0,  # Modulator (dopamine) decay (ms)
            device=self.device,
        ))

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(HippocampusPopulation.DG.value, self.dg_neurons)
        self._register_neuron_population(HippocampusPopulation.CA3.value, self.ca3_neurons)
        self._register_neuron_population(HippocampusPopulation.CA2.value, self.ca2_neurons)
        self._register_neuron_population(HippocampusPopulation.CA1.value, self.ca1_neurons)

        self._register_neuron_population(HippocampusPopulation.DG_INHIBITORY_PV.value, self.dg_inhibitory.pv_neurons)
        self._register_neuron_population(HippocampusPopulation.DG_INHIBITORY_OLM.value, self.dg_inhibitory.olm_neurons)
        self._register_neuron_population(HippocampusPopulation.DG_INHIBITORY_BISTRATIFIED.value, self.dg_inhibitory.bistratified_neurons)

        self._register_neuron_population(HippocampusPopulation.CA3_INHIBITORY_PV.value, self.ca3_inhibitory.pv_neurons)
        self._register_neuron_population(HippocampusPopulation.CA3_INHIBITORY_OLM.value, self.ca3_inhibitory.olm_neurons)
        self._register_neuron_population(HippocampusPopulation.CA3_INHIBITORY_BISTRATIFIED.value, self.ca3_inhibitory.bistratified_neurons)

        self._register_neuron_population(HippocampusPopulation.CA2_INHIBITORY_PV.value, self.ca2_inhibitory.pv_neurons)
        self._register_neuron_population(HippocampusPopulation.CA2_INHIBITORY_OLM.value, self.ca2_inhibitory.olm_neurons)
        self._register_neuron_population(HippocampusPopulation.CA2_INHIBITORY_BISTRATIFIED.value, self.ca2_inhibitory.bistratified_neurons)

        self._register_neuron_population(HippocampusPopulation.CA1_INHIBITORY_PV.value, self.ca1_inhibitory.pv_neurons)
        self._register_neuron_population(HippocampusPopulation.CA1_INHIBITORY_OLM.value, self.ca1_inhibitory.olm_neurons)
        self._register_neuron_population(HippocampusPopulation.CA1_INHIBITORY_BISTRATIFIED.value, self.ca1_inhibitory.bistratified_neurons)

        # =====================================================================
        # POST-INITIALIZATION
        # =====================================================================
        self.__post_init__()

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
        self._add_internal_connection(
            source_population=HippocampusPopulation.DG.value,
            target_population=HippocampusPopulation.CA3.value,
            weights=WeightInitializer.sparse_random(
                n_input=self.dg_size,
                n_output=self.ca3_size,
                connectivity=0.15,
                weight_scale=0.0001,
                device=device,
            ),
            # Mossy Fibers (DG→CA3): Strong facilitation
            stp_config=MOSSY_FIBER_PRESET.configure(),
            is_inhibitory=False,
        )

        # CA3 → CA3 RECURRENT: Autoassociative memory weights
        # Learning: One-shot Hebbian with fast/slow traces and heterosynaptic LTD
        ca3_ca3_weights = WeightInitializer.sparse_random(
            n_input=self.ca3_size,
            n_output=self.ca3_size,
            connectivity=0.05,
            weight_scale=0.002,
            device=device,
        )
        # Apply phase diversity: ±15% weight variation for temporal coding
        # Phase leads/lags enable different neurons to fire at different theta phases
        ca3_ca3_weights = WeightInitializer.add_phase_diversity(ca3_ca3_weights, phase_diversity=0.15)
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA3.value,
            target_population=HippocampusPopulation.CA3.value,
            weights=ca3_ca3_weights,
            stp_config=None,
            is_inhibitory=False,
        )

        # Create sparse local inhibition: each neuron inhibits nearby neurons
        # Biologically: basket cells have local axonal arbors (~200-300μm radius)
        # We approximate this with random sparse connectivity
        ca3_ca3_inhib_weights = WeightInitializer.sparse_random(
            n_input=self.ca3_size,
            n_output=self.ca3_size,
            connectivity=0.2,
            weight_scale=0.001,
            device=self.device,
        )
        ca3_ca3_inhib_weights.fill_diagonal_(0.0)  # Zero out self-connections
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA3.value,
            target_population=HippocampusPopulation.CA3.value,
            weights=ca3_ca3_inhib_weights,
            stp_config=None,
            is_inhibitory=True,
        )

        # Initialize CA2 lateral inhibition weights
        ca2_ca2_inhib_weights = WeightInitializer.sparse_random(
            n_input=self.ca2_size,
            n_output=self.ca2_size,
            connectivity=0.2,
            weight_scale=0.0005,
            device=device,
        )
        ca2_ca2_inhib_weights.fill_diagonal_(0.0)  # Zero out self-connections
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA2.value,
            target_population=HippocampusPopulation.CA2.value,
            weights=ca2_ca2_inhib_weights,
            stp_config=None,
            is_inhibitory=True,
        )

        # =====================================================================
        # CA2 PATHWAYS: Social memory and temporal context
        # =====================================================================
        # CA3 → CA2: Weak plasticity (stability mechanism)
        # CA2 is resistant to CA3 pattern completion interference
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA3.value,
            target_population=HippocampusPopulation.CA2.value,
            weights=WeightInitializer.sparse_random(
                n_input=self.ca3_size,
                n_output=self.ca2_size,
                connectivity=0.3,
                weight_scale=0.0006,
                device=device,
            ),
            stp_config=SCHAFFER_COLLATERAL_PRESET.configure(),
            is_inhibitory=False,
        )

        # CA2 → CA1: Output to decision layer
        # Provides temporal/social context to CA1 processing
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA2.value,
            target_population=HippocampusPopulation.CA1.value,
            weights=WeightInitializer.sparse_random(
                n_input=self.ca2_size,
                n_output=self.ca1_size,
                connectivity=0.2,
                weight_scale=0.0005,
                device=device,
            ),
            stp_config=MOSSY_FIBER_PRESET.configure(),
            is_inhibitory=False,
        )

        # CA3 → CA1: Feedforward (retrieved memory)
        # This is the DIRECT bypass pathway (Schaffer collaterals)
        self._add_internal_connection(
            source_population=HippocampusPopulation.CA3.value,
            target_population=HippocampusPopulation.CA1.value,
            weights=WeightInitializer.sparse_random(
                n_input=self.ca3_size,
                n_output=self.ca1_size,
                connectivity=0.15,
                weight_scale=0.0008,
                device=device,
            ),
            stp_config=SCHAFFER_COLLATERAL_PRESET.configure(),
            is_inhibitory=False,
        )

        # CA1 lateral inhibition for competition
        # Use sparse lateral inhibition (similar to CA3 basket cells)
        # Biologically: CA1 interneurons have local connectivity, not all-to-all
        ca1_inhib_weights = WeightInitializer.sparse_random(
            n_input=self.ca1_size,
            n_output=self.ca1_size,
            connectivity=0.2,
            weight_scale=0.0005,
            device=device,
        )
        ca1_inhib_weights.fill_diagonal_(0.0)  # Zero diagonal (no self-inhibition)
        ca1_ca1_inhib_synapse = self._add_internal_connection(
            source_population=HippocampusPopulation.CA1.value,
            target_population=HippocampusPopulation.CA1.value,
            weights=ca1_inhib_weights,
            stp_config=None,
            is_inhibitory=True,
        )

        # Create gap junction network for CA1 interneurons
        gap_config_ca1=GapJunctionConfig(
            coupling_strength=self.config.gap_junction_strength,
            connectivity_threshold=self.config.gap_junction_threshold,
            max_neighbors=self.config.gap_junction_max_neighbors,
            interneuron_only=True,
        )
        self.gap_junctions_ca1 = GapJunctionCoupling(
            n_neurons=self.ca1_size,
            afferent_weights=self.get_synaptic_weights(ca1_ca1_inhib_synapse),
            config=gap_config_ca1,
            device=device,
        )

    # =========================================================================
    # CONSOLIDATION MODE
    # =========================================================================

    def enter_consolidation_mode(self) -> None:
        """Enter consolidation mode (sleep/offline replay)."""
        self._consolidation_mode = True

    def exit_consolidation_mode(self) -> None:
        """Exit consolidation mode and return to encoding."""
        self._consolidation_mode = False
        self._replay_cue = None

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

        if cfg.enable_neuromodulation:
            # =====================================================================
            # DOPAMINE RECEPTOR PROCESSING (from VTA)
            # =====================================================================
            # Process VTA dopamine spikes → concentration dynamics
            # Hippocampus receives minimal (10%) DA innervation for novelty/salience
            vta_da_spikes = neuromodulator_inputs.get('da', None)
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
            # ACETYLCHOLINE RECEPTOR PROCESSING (from NB)
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
        else:
            # Neuromodulation disabled: keep baseline concentrations
            pass

        dg_da_level = self._da_concentration_dg.mean().item()
        ca3_da_level = self._da_concentration_ca3.mean().item()
        ca2_da_level = self._da_concentration_ca2.mean().item()
        ca1_da_level = self._da_concentration_ca1.mean().item()

        # =====================================================================
        # CONSOLIDATION MODE: Spontaneous CA3→CA1 Replay (Sharp-Wave Ripples)
        # =====================================================================
        # During sleep consolidation, hippocampus spontaneously reactivates stored
        # patterns without external input. This simulates sharp-wave ripples where
        # CA3 recurrent activity triggers CA1 output for cortical replay.
        #
        # Biological mechanism:
        # - LOW acetylcholine enables CA3 spontaneous reactivation
        # - CA3 attractor pattern propagates through Schaffer collaterals to CA1
        # - STP dynamics preserved (biological timing maintained)
        # - CA1 output drives cortical consolidation via back-projections

        dg_ca3_synapse = SynapseId(self.region_name, HippocampusPopulation.DG.value, self.region_name, HippocampusPopulation.CA3.value, is_inhibitory=False)
        ca3_ca3_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3.value, self.region_name, HippocampusPopulation.CA3.value, is_inhibitory=False)
        ca3_ca2_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3.value, self.region_name, HippocampusPopulation.CA2.value, is_inhibitory=False)
        ca3_ca1_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3.value, self.region_name, HippocampusPopulation.CA1.value, is_inhibitory=False)
        ca2_ca1_synapse = SynapseId(self.region_name, HippocampusPopulation.CA2.value, self.region_name, HippocampusPopulation.CA1.value, is_inhibitory=False)

        dg_ca3_weights = self.get_synaptic_weights(dg_ca3_synapse)
        ca3_ca3_weights = self.get_synaptic_weights(ca3_ca3_synapse)
        ca3_ca2_weights = self.get_synaptic_weights(ca3_ca2_synapse)
        ca3_ca1_weights = self.get_synaptic_weights(ca3_ca1_synapse)
        ca2_ca1_weights = self.get_synaptic_weights(ca2_ca1_synapse)

        # Reset ripple detection flag
        self.ripple_detected = False

        # Check if spontaneous replay should occur (low ACh, probabilistic trigger)
        if self.spontaneous_replay is not None:
            should_replay = self.spontaneous_replay.should_trigger_ripple(
                acetylcholine=self._ach_concentration_ca3.mean().item(),
                dt_ms=dt_ms,
            )

            if should_replay:
                # Select pattern to replay based on tags and weight strength
                seed_pattern = self.spontaneous_replay.select_pattern_to_replay(
                    synaptic_tags=self.synaptic_tagging.tags,
                    ca3_weights=ca3_ca3_weights,
                    seed_fraction=0.15,
                )

                # Inject seed pattern into CA3 persistent activity
                self.ca3_persistent = self.ca3_persistent * 0.5 + seed_pattern.float() * 0.5
                self.ripple_detected = True

        if self._consolidation_mode and self._replay_cue is not None:
            self._replay_cue = None
            # TODO: Implement replay pattern generation and CA3→CA1 propagation here
            return {}  # Skip normal processing during replay timestep (CA3→CA1 driven by internal dynamics)

        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        dg_input = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.dg_size,
            filter_by_target_population=HippocampusPopulation.DG.value
        ).g_exc
        ca3_input = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.ca3_size,
            filter_by_target_population=HippocampusPopulation.CA3.value
        ).g_exc
        # TODO: Integrate CA2 input here as well (currently only from CA3, but could add EC→CA2)
        ca1_input = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.ca1_size,
            filter_by_target_population=HippocampusPopulation.CA1.value
        ).g_exc

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
        # Clamp to positive conductance
        dg_total_input = dg_input * ffi_factor
        if cfg.baseline_noise_conductance_enabled:
            noise = torch.randn_like(dg_total_input) * 0.007
            dg_total_input = dg_total_input + noise

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

        dg_g_ampa, dg_g_nmda = self._split_excitatory_conductance(dg_g_exc)

        dg_spikes, _ = self.dg_neurons.forward(
            g_ampa_input=ConductanceTensor(dg_g_ampa),
            g_gaba_a_input=ConductanceTensor(dg_perisomatic_inhib),
            g_nmda_input=ConductanceTensor(dg_g_nmda),
        )

        self._update_homeostasis(spikes=dg_spikes, firing_rate=self.dg_firing_rate, neurons=self.dg_neurons)

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
            source_population=HippocampusPopulation.CA3.value,
            target_region=self.region_name,
            target_population=HippocampusPopulation.CA3.value,
            is_inhibitory=True,
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

        ca3_persistent_input = self.ca3_persistent * cfg.ca3_persistent_gain  # [ca3_size]

        # Total CA3 excitatory input = feedforward + recurrent + persistent
        ca3_excitatory_input = ca3_ff + ca3_rec + ca3_persistent_input

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

        # Apply homeostatic gain and baseline noise (Turrigiano 2008)
        # Add baseline noise (spontaneous miniature EPSPs) with stochastic component
        if cfg.baseline_noise_conductance_enabled:
            stochastic = 0.007 * torch.randn(self.ca3_size, device=self.device)
            ca3_excitatory_input = ca3_excitatory_input + stochastic.abs()

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
        if self.ca3_neurons.membrane is not None:
            # Normalize membrane potential to [0, 1] range (assume v_rest=-1, v_threshold=1.2)
            v_rest = -1.0
            v_threshold = 1.2  # Updated threshold
            v_normalized = (self.ca3_neurons.membrane - v_rest) / (v_threshold - v_rest)
            v_normalized = torch.clamp(v_normalized, 0.0, 1.0)
            # Only apply strong inhibition when > 80% of way to threshold
            # This allows normal firing while preventing bursts
            burst_risk = torch.clamp((v_normalized - 0.8) / 0.2, 0.0, 1.0)
            ca3_g_inh = ca3_g_inh + burst_risk * 0.1

        # Split excitatory conductance into AMPA (fast) and NMDA (slow)
        ca3_g_ampa, ca3_g_nmda = self._split_excitatory_conductance(ca3_g_exc)

        ca3_spikes, _ = self.ca3_neurons.forward(
            g_ampa_input=ConductanceTensor(ca3_g_ampa),
            g_gaba_a_input=ConductanceTensor(ca3_g_inh),
            g_nmda_input=ConductanceTensor(ca3_g_nmda),
        )

        ca3_spikes_float = ca3_spikes.float()

        self._update_homeostasis(spikes=ca3_spikes, firing_rate=self.ca3_firing_rate, neurons=self.ca3_neurons)

        # Update persistent activity AFTER computing new spikes
        # The trace accumulates spike activity with slow decay
        # Using a direct accumulation: trace += spike - decay*trace
        # This ensures spikes have strong immediate effect but decay slowly
        ca3_persistent_decay_rate = dt_ms / cfg.ca3_persistent_tau

        # Update persistent activity: stronger during encoding, decay otherwise
        # Encoding_mod determines how much new spikes contribute vs decay
        # This is biologically motivated: Ca²⁺-dependent currents build up during
        # active encoding, then decay during maintenance/retrieval
        # Continuous modulation: contribution naturally weak when encoding_mod is low
        self.ca3_persistent = (
            self.ca3_persistent * (1.0 - ca3_persistent_decay_rate * (0.5 + 0.5 * retrieval_mod))
            + ca3_spikes_float
            * encoding_mod  # Contribution scaled by encoding strength
        )
        clamp_weights(self.ca3_persistent, cfg.w_min, cfg.w_max)  # Clamp to prevent runaway

        # Store the DG pattern (accumulate over timesteps, scaled by encoding strength)
        # Continuous modulation: storage naturally weak when encoding_mod is low
        self.stored_dg_pattern = self.stored_dg_pattern + dg_spikes_float * encoding_mod

        # Compute decay factors for multi-timescale traces
        fast_decay = dt_ms / cfg.fast_trace_tau_ms
        slow_decay = dt_ms / cfg.slow_trace_tau_ms

        # =========================================================
        # MULTI-TIMESCALE CONSOLIDATION
        # =========================================================
        # Apply decay to fast trace
        self._ca3_ca3_fast = (1.0 - fast_decay) * self._ca3_ca3_fast
        self._ca3_ca3_slow = (1.0 - slow_decay) * self._ca3_ca3_slow + cfg.consolidation_rate * self._ca3_ca3_fast

        # Learning happens only when there's CA3 activity AND learning is enabled
        if WeightInitializer.GLOBAL_LEARNING_ENABLED and ca3_spikes.any():
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

            # Add learning to traces
            # Accumulate new learning into fast trace
            self._ca3_ca3_fast = self._ca3_ca3_fast + dW * effective_lr

            # Combined weight update: Fast (episodic) + Slow (semantic)
            # Fast trace dominates initially, slow trace provides stability
            combined_dW = self._ca3_ca3_fast + cfg.slow_trace_contribution * self._ca3_ca3_slow
            ca3_ca3_weights.data += combined_dW
            ca3_ca3_weights.data.fill_diagonal_(0.0)  # No self-connections
            clamp_weights(ca3_ca3_weights.data, cfg.w_min, cfg.w_max)

            # =========================================================
            # SYNAPTIC TAGGING
            # =========================================================
            # Update synaptic tags based on spike coincidence
            # Tags mark recently-active synapses for potential consolidation
            # Replaces explicit Episode.priority with emergent biological mechanism
            self.synaptic_tagging.update_tags(pre_spikes=ca3_spikes_float, post_spikes=ca3_spikes_float)

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

        self._dg_ca3_fast = (1.0 - fast_decay) * self._dg_ca3_fast
        self._dg_ca3_slow = (1.0 - slow_decay) * self._dg_ca3_slow + cfg.consolidation_rate * self._dg_ca3_fast

        # Learning happens only when there's co-activity (DG and CA3 both firing) AND learning is enabled
        if WeightInitializer.GLOBAL_LEARNING_ENABLED and dg_spikes_delayed.any() and ca3_spikes.any():
            # Hebbian outer product: bind DG pattern to CA3 attractor
            # Shape: [ca3_size, dg_size] - each CA3 neuron learns from DG inputs
            dW_mossy = torch.outer(ca3_spikes_float, dg_spikes_delayed_float)

            # ONE-SHOT learning: Use STRONG learning rate (3x normal)
            # Biology: Mossy fiber LTP is rapid and doesn't require multiple pairings
            ca3_da_gain = 0.2 + 1.8 * ca3_da_level  # Strong dopamine gating
            mossy_effective_lr = cfg.learning_rate * 3.0 * encoding_mod * ca3_da_gain

            # Add to fast trace
            self._dg_ca3_fast = self._dg_ca3_fast + dW_mossy * mossy_effective_lr

            # Combined weight update: Fast (episodic) + Slow (semantic)
            combined_dW = self._dg_ca3_fast + cfg.slow_trace_contribution * self._dg_ca3_slow
            dg_ca3_weights.data += combined_dW
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
            source_population=HippocampusPopulation.CA2.value,
            target_region=self.region_name,
            target_population=HippocampusPopulation.CA2.value,
            is_inhibitory=True,
        )
        ca2_ca2_inhib_weights = self.get_synaptic_weights(ca2_ca2_inhib_synapse)
        local_lateral_ca2 = torch.matmul(ca2_ca2_inhib_weights, prev_ca2_spikes.float())
        ca2_feedback_inhibition = ca2_population_rate * 0.2 + local_lateral_ca2 * 0.1

        # Apply homeostatic gain and baseline noise
        # Add baseline noise (spontaneous miniature EPSPs)
        if cfg.baseline_noise_conductance_enabled:
            noise = torch.randn_like(ca2_from_ca3) * 0.007
            ca2_from_ca3 = ca2_from_ca3 + noise.abs()

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
        if self.ca2_neurons.membrane is not None:
            # Normalize membrane potential to [0, 1] range (assume v_rest=-1, v_threshold=1.0)
            v_rest = -1.0
            v_threshold = 1.0  # CA2 threshold
            v_normalized = (self.ca2_neurons.membrane - v_rest) / (v_threshold - v_rest)
            v_normalized = torch.clamp(v_normalized, 0.0, 1.0)
            # Only apply strong inhibition when > 80% of way to threshold
            burst_risk = torch.clamp((v_normalized - 0.8) / 0.2, 0.0, 1.0)
            ca2_g_inh = ca2_g_inh + burst_risk * 0.1

        # Split excitatory conductance into AMPA (fast) and NMDA (slow)
        ca2_g_ampa, ca2_g_nmda = self._split_excitatory_conductance(ca2_g_exc)

        ca2_spikes, _ = self.ca2_neurons.forward(
            g_ampa_input=ConductanceTensor(ca2_g_ampa),
            g_gaba_a_input=ConductanceTensor(ca2_perisomatic_inhib),
            g_nmda_input=ConductanceTensor(ca2_g_nmda),
        )
        ca2_spikes_float = ca2_spikes.float()

        self._update_homeostasis(spikes=ca2_spikes, firing_rate=self.ca2_firing_rate, neurons=self.ca2_neurons)

        # CA3→CA2 WEAK PLASTICITY (stability mechanism)
        self._ca3_ca2_fast = (1.0 - fast_decay) * self._ca3_ca2_fast
        self._ca3_ca2_slow = (1.0 - slow_decay) * self._ca3_ca2_slow + cfg.consolidation_rate * self._ca3_ca2_fast

        # Learning only when there's activity AND learning is enabled
        if WeightInitializer.GLOBAL_LEARNING_ENABLED and ca3_spikes_for_ca2.any() and ca2_spikes.any():
            # Very weak learning rate (stability hub)
            effective_lr = cfg.learning_rate * encoding_mod

            dW = torch.outer(ca2_spikes_float, ca3_spikes_for_ca2_float)

            self._ca3_ca2_fast = self._ca3_ca2_fast + dW * effective_lr
            combined_dW = self._ca3_ca2_fast + cfg.slow_trace_contribution * self._ca3_ca2_slow
            ca3_ca2_weights.data += combined_dW
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

        # Split excitation and inhibition for ConductanceLIF
        ca1_g_exc = ca3_contribution + ca1_from_ca2 + ampa_conductance + nmda_conductance
        if cfg.baseline_noise_conductance_enabled:
            noise = torch.randn_like(ca1_g_exc) * 0.007
            ca1_g_exc = ca1_g_exc + noise.abs()

        # Apply dendritic inhibition to excitatory input (models apical dendrite suppression)
        # OLM cells target apical dendrites where EC input arrives
        # Normalize dendritic inhibition to biological range [0, 1]
        ca1_dendritic_inhib_normalized = torch.sigmoid((ca1_dendritic_inhib - 1.5) * 2.0)
        suppression_factor = torch.clamp(ca1_dendritic_inhib_normalized.mean() * 0.8, 0.0, 0.8)
        ca1_g_exc = ca1_g_exc * (1.0 - suppression_factor)

        # Add CA1 feedback inhibition (like CA3) to prevent hyperactivity
        # CA1 pyramidal cells recruit basket cells for lateral inhibition
        # Simple implementation: population activity suppresses all neurons equally
        prev_ca1_spikes = self._ca1_spike_buffer.read(1)
        ca1_feedback_inhib = prev_ca1_spikes.float().mean() * 0.1  # Light feedback strength

        # Inhibitory: perisomatic inhibition from PV cells + lateral CA1 inhibition + tonic
        # Tonic inhibition from extrasynaptic GABA_A receptors provides constant baseline
        ca1_g_inh = F.relu(cfg.tonic_inhibition + ca1_perisomatic_inhib + ca1_feedback_inhib)

        # Add instantaneous self-inhibition to prevent burst escalation
        # Uses membrane potential as proxy for imminent spiking
        # Only apply when V_mem is VERY close to threshold (prevents premature inhibition)
        if self.ca1_neurons.membrane is not None:
            # Normalize membrane potential to [0, 1] range (assume v_rest=-1, v_threshold=1.0)
            v_rest = -1.0
            v_threshold = 1.0  # CA1 threshold
            v_normalized = (self.ca1_neurons.membrane - v_rest) / (v_threshold - v_rest)
            v_normalized = torch.clamp(v_normalized, 0.0, 1.0)
            burst_risk = torch.clamp((v_normalized - 0.8) / 0.2, 0.0, 1.0)
            ca1_g_inh = ca1_g_inh + burst_risk * 0.1

        # Split excitatory conductance into AMPA (fast) and NMDA (slow)
        ca1_g_ampa, ca1_g_nmda = self._split_excitatory_conductance(ca1_g_exc)

        # Run through CA1 neurons (ConductanceLIF with E/I separation)
        ca1_spikes, _ca1_membrane = self.ca1_neurons.forward(
            g_ampa_input=ConductanceTensor(ca1_g_ampa),
            g_gaba_a_input=ConductanceTensor(ca1_g_inh),
            g_nmda_input=ConductanceTensor(ca1_g_nmda),
        )
        ca1_spikes_float = ca1_spikes.float()

        self._update_homeostasis(spikes=ca1_spikes, firing_rate=self.ca1_firing_rate, neurons=self.ca1_neurons)

        # ---------------------------------------------------------
        # HEBBIAN LEARNING: CA2→CA1 plasticity (during encoding)
        # ---------------------------------------------------------
        # CA2 provides temporal/social context to CA1
        if WeightInitializer.GLOBAL_LEARNING_ENABLED and ca2_spikes_delayed.any() and ca1_spikes.any():
            effective_lr = cfg.learning_rate * encoding_mod

            dW = torch.outer(ca1_spikes_float, ca2_spikes_delayed_float)

            self._ca2_ca1_fast = (1.0 - fast_decay) * self._ca2_ca1_fast + dW * effective_lr
            self._ca2_ca1_slow = (1.0 - slow_decay) * self._ca2_ca1_slow + cfg.consolidation_rate * self._ca2_ca1_fast

            combined_dW = self._ca2_ca1_fast + cfg.slow_trace_contribution * self._ca2_ca1_slow
            ca2_ca1_weights.data += combined_dW
            clamp_weights(ca2_ca1_weights.data, cfg.w_min, cfg.w_max)

        if WeightInitializer.GLOBAL_LEARNING_ENABLED:
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
            if encoding_mod > 0.5:
                # Get dopamine modulation (average across all subregions)
                da_modulation = (dg_da_level + ca3_da_level + ca2_da_level + ca1_da_level) / 4.0

                # Modulate learning rate by dopamine
                effective_lr = cfg.learning_rate * 0.3 * (1.0 + da_modulation)

                # Apply learning to each external input source
                for synapse_id, source_input in synaptic_inputs.items():
                    weights = self.get_synaptic_weights(synapse_id)

                    if synapse_id.target_population == HippocampusPopulation.DG.value:
                        # Learn EC→DG (pattern separation input)
                        # Hebbian learning: pre (source) × post (DG)
                        dW_dg = effective_lr * torch.outer(dg_spikes_float, source_input.float())
                        weights.data += dW_dg

                    elif synapse_id.target_population == HippocampusPopulation.CA3.value:
                        # Learn EC→CA3 (direct perforant path for retrieval cues)
                        # Hebbian learning: pre (source) × post (CA3)
                        dW_ca3 = effective_lr * torch.outer(ca3_spikes_float, source_input.float())
                        weights.data += dW_ca3

                    elif synapse_id.target_population == HippocampusPopulation.CA1.value:
                        # Learn EC→CA1 (direct output pathway)
                        # Hebbian learning: pre (source) × post (CA1)
                        dW_ca1 = effective_lr * torch.outer(ca1_spikes_float, source_input.float())
                        weights.data += dW_ca1

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

            if encoding_mod > 0.5:
                # Get delayed CA3 activity (what was just active before current spikes)
                ca3_delayed = self._ca3_ca3_buffer.read(self._ca3_ca3_delay_steps)

                # Three-factor rule needs DEVIATION from baseline (not absolute concentration)
                # Baseline ~0.5 → modulator = 0 (no learning)
                # Reward burst → modulator > 0 (strengthen synapses)
                # Punishment dip → modulator < 0 (weaken synapses)
                da_ca3_deviation = ca3_da_level - 0.5
                if self.get_learning_strategy(ca3_ca3_synapse) is None:
                    self.add_learning_strategy(ca3_ca3_synapse, self._three_factor_strategy)
                self.apply_learning(
                    ca3_ca3_synapse, ca3_delayed, ca3_spikes,
                    modulator=da_ca3_deviation,
                )
                self.get_synaptic_weights(ca3_ca3_synapse).data.fill_diagonal_(0.0)  # Maintain no self-connections (biological constraint)

            # =====================================================================
            # DOPAMINE-GATED CONSOLIDATION
            # =====================================================================
            # Apply dopamine-gated consolidation to tagged synapses
            # High dopamine (reward) → strong consolidation of tagged synapses
            # This is the "capture" part of synaptic tagging and capture
            if self.synaptic_tagging is not None and ca3_da_level > 0.1:
                # Consolidate tagged synapses proportional to dopamine
                ca3_ca3_weights.data = self.synaptic_tagging.consolidate_tagged_synapses(
                    weights=ca3_ca3_weights,
                    dopamine=ca3_da_level,
                    learning_rate=cfg.learning_rate * 0.5,  # Half of base LR
                )
                clamp_weights(ca3_ca3_weights.data, cfg.w_min, cfg.w_max)

        region_outputs: RegionOutput = {
            HippocampusPopulation.DG.value: dg_spikes,
            HippocampusPopulation.CA3.value: ca3_spikes,
            HippocampusPopulation.CA2.value: ca2_spikes,
            HippocampusPopulation.CA1.value: ca1_spikes,
        }

        self._apply_plasticity(region_outputs)

        # =====================================================================
        # UPDATE STATE BUFFERS FOR NEXT TIMESTEP
        # =====================================================================
        # Write spikes to state buffers
        self._dg_spike_buffer.write_and_advance(dg_spikes)
        self._ca3_spike_buffer.write_and_advance(ca3_spikes)
        self._ca2_spike_buffer.write_and_advance(ca2_spikes)
        self._ca1_spike_buffer.write_and_advance(ca1_spikes)

        # Write membrane potentials to state buffers
        self._dg_membrane_buffer.write_and_advance(self.dg_neurons.membrane)
        self._ca3_membrane_buffer.write_and_advance(self.ca3_neurons.membrane)
        self._ca2_membrane_buffer.write_and_advance(self.ca2_neurons.membrane)
        self._ca1_membrane_buffer.write_and_advance(self.ca1_neurons.membrane)

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
        if not WeightInitializer.GLOBAL_LEARNING_ENABLED:
            return  # Skip plasticity if globally disabled (e.g., for testing)

        ca3_spikes = region_outputs[HippocampusPopulation.CA3.value]
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

                # Compute deviation from target
                deviation = (avg_firing_rate - self.config.target_firing_rate) / (self.config.target_firing_rate + 1e-8)

                # Only act if chronically outside tolerance
                if abs(deviation) > 0.5:
                    ca3_ca3_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3.value, self.region_name, HippocampusPopulation.CA3.value, is_inhibitory=False)
                    ca3_ca2_synapse = SynapseId(self.region_name, HippocampusPopulation.CA3.value, self.region_name, HippocampusPopulation.CA2.value, is_inhibitory=False)
                    ca2_ca1_synapse = SynapseId(self.region_name, HippocampusPopulation.CA2.value, self.region_name, HippocampusPopulation.CA1.value, is_inhibitory=False)

                    # Compute multiplicative scale factor
                    target_scale = self.config.target_firing_rate / (avg_firing_rate + 1e-8)
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
                    for synapse_id in self._synaptic_weights.keys():
                        if synapse_id.target_population == HippocampusPopulation.CA3.value:
                            self.get_synaptic_weights(synapse_id).data *= scale_factor

                    # Enforce hard bounds after scaling
                    for synapse_id in [ca3_ca3_synapse, ca3_ca2_synapse, ca2_ca1_synapse]:
                        if self.has_synaptic_weights(synapse_id):
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
            activity_target=self.config.target_firing_rate,
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
