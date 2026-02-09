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

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.brain.configs import HippocampusConfig
from thalia.components import (
    NeuronFactory,
    ShortTermPlasticity,
    WeightInitializer,
    get_stp_config,
    GapJunctionConfig,
    GapJunctionCoupling,
)
from thalia.components.synapses.neuromodulator_receptor import NeuromodulatorReceptor
from thalia.diagnostics import compute_plasticity_metrics
from thalia.learning.homeostasis import (
    UnifiedHomeostasis,
    UnifiedHomeostasisConfig,
)
from thalia.diagnostics import DiagnosticsUtils
from thalia.typing import (
    LayerName,
    RegionLayerSizes,
    RegionSpikesDict,
    SpikesSourceKey,
)
from thalia.utils import (
    CircularDelayBuffer,
    clamp_weights,
    compute_ach_recurrent_suppression,
    compute_ne_gain,
    compute_learning_rate_modulation,
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

    # Declarative output ports (auto-registered by base class)
    OUTPUT_PORTS = {
        "dg": "dg_size",
        "ca3": "ca3_size",
        "ca2": "ca2_size",
        "ca1": "ca1_size",
    }

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: HippocampusConfig, region_layer_sizes: RegionLayerSizes):
        """Initialize trisynaptic hippocampus."""
        super().__init__(config=config, region_layer_sizes=region_layer_sizes)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.dg_size = region_layer_sizes["dg_size"]
        self.ca3_size = region_layer_sizes["ca3_size"]
        self.ca2_size = region_layer_sizes["ca2_size"]
        self.ca1_size = region_layer_sizes["ca1_size"]

        # Calculate inhibitory size for septal input routing
        # Septal GABA targets OLM cells (30% of total inhibitory, 25% of pyramidal)
        # Total OLM = CA1_OLM + CA3_OLM
        olm_fraction = 0.30  # OLM cells are 30% of inhibitory population
        inhibitory_fraction = 0.25  # Total inhibitory is 25% of pyramidal count
        self.inhibitory_size = int((self.ca1_size + self.ca3_size) * inhibitory_fraction * olm_fraction)

        # =====================================================================
        # INITIALIZE STATE FIELDS
        # =====================================================================
        # Layer activities (current spikes)
        self.dg_spikes: Optional[torch.Tensor] = None
        self.ca3_spikes: Optional[torch.Tensor] = None
        self.ca2_spikes: Optional[torch.Tensor] = None
        self.ca1_spikes: Optional[torch.Tensor] = None

        # CA3 bistable persistent activity trace
        # Models I_NaP/I_CAN currents that allow neurons to maintain firing
        # without continuous external input. This is essential for stable
        # attractor states during delay periods.
        self.ca3_persistent: torch.Tensor = torch.zeros(self.ca3_size, device=self.device)

        # NMDA trace for temporal integration (slow kinetics)
        self.nmda_trace: Optional[torch.Tensor] = None

        # Stored DG pattern from sample phase (for match/mismatch detection)
        self.stored_dg_pattern: torch.Tensor = torch.zeros(self.dg_size, device=self.device)

        # Spontaneous replay (sharp-wave ripple) detection
        self.ripple_detected: bool = False

        # =====================================================================
        # GAP JUNCTIONS (Electrical Synapses) - Config Setup
        # =====================================================================
        # Store gap junction config BEFORE weight initialization so it can be
        # used during _init_circuit_weights() to create gap junction module.
        self._gap_config_ca1 = GapJunctionConfig(
            coupling_strength=config.gap_junction_strength,
            connectivity_threshold=config.gap_junction_threshold,
            max_neighbors=config.gap_junction_max_neighbors,
            interneuron_only=True,
        )
        self.gap_junctions_ca1: Optional[GapJunctionCoupling] = None

        # Override weights with trisynaptic circuit weights
        self._init_circuit_weights()

        # Create LIF neurons for each layer using factory functions
        # DG and CA1: Standard pyramidal neurons
        self.dg_neurons = NeuronFactory.create_pyramidal_neurons(self.dg_size, self.device)
        # CA3 gets spike-frequency adaptation to prevent frozen attractors
        self.ca3_neurons = NeuronFactory.create_pyramidal_neurons(
            self.ca3_size,
            self.device,
            adapt_increment=config.adapt_increment,  # SFA enabled!
            tau_adapt=config.adapt_tau,
        )
        # CA2: Social memory and temporal context (standard pyramidal)
        self.ca2_neurons = NeuronFactory.create_pyramidal_neurons(self.ca2_size, self.device)
        # CA1: Output layer
        self.ca1_neurons = NeuronFactory.create_pyramidal_neurons(self.ca1_size, self.device)

        # Stimulus gating module (transient inhibition at stimulus changes)
        self.stimulus_gating = StimulusGating(
            threshold=config.ffi_threshold,
            max_inhibition=config.ffi_strength * 10.0,  # Scale to appropriate range
            decay_rate=1.0 - (1.0 / config.ffi_tau),  # Convert tau to rate
        )

        # =====================================================================
        # HIPPOCAMPAL INHIBITORY NETWORKS (with OLM cells for emergent theta)
        # =====================================================================
        # CA1 inhibitory network: PV, OLM, Bistratified cells
        # OLM cells phase-lock to septal GABA → emergent encoding/retrieval
        self.ca1_inhibitory = HippocampalInhibitoryNetwork(
            region_name="ca1",
            pyr_size=self.ca1_size,
            total_inhib_fraction=0.25,  # 25% inhibitory (20% of total neurons)
            device=str(self.device),
            dt_ms=config.dt_ms,
        )

        # CA3 inhibitory network
        self.ca3_inhibitory = HippocampalInhibitoryNetwork(
            region_name="ca3",
            pyr_size=self.ca3_size,
            total_inhib_fraction=0.25,
            device=str(self.device),
            dt_ms=config.dt_ms,
        )

        # =====================================================================
        # SHORT-TERM PLASTICITY (STP)
        # =====================================================================
        self.stp_mossy: Optional[ShortTermPlasticity] = None
        self.stp_schaffer: Optional[ShortTermPlasticity] = None
        self.stp_ec_ca1: Optional[ShortTermPlasticity] = None
        self.stp_ca3_ca2: Optional[ShortTermPlasticity] = None
        self.stp_ca2_ca1: Optional[ShortTermPlasticity] = None
        self.stp_ec_ca2: Optional[ShortTermPlasticity] = None

        # Mossy Fibers (DG→CA3): Strong facilitation
        self.stp_mossy = ShortTermPlasticity(
            n_pre=self.dg_size,
            n_post=self.ca3_size,
            config=get_stp_config("mossy_fiber"),
            per_synapse=True,
        )
        self.stp_mossy.to(self.device)

        # Schaffer Collaterals (CA3→CA1): Depression
        # High-frequency CA3 activity depresses CA1 input - allows novelty
        # detection (novel patterns don't suffer from adaptation)
        self.stp_schaffer = ShortTermPlasticity(
            n_pre=self.ca3_size,
            n_post=self.ca1_size,
            config=get_stp_config("schaffer_collateral"),
            per_synapse=True,
        )
        self.stp_schaffer.to(self.device)

        # EC→CA1 Direct (Temporoammonic): Depression
        # Initial input is strongest - matched comparison happens on first
        # presentation before adaptation kicks in
        # Use ec_l3_input_size if set (for separate raw sensory input),
        # otherwise fall back to input_size
        self.stp_ec_ca1 = ShortTermPlasticity(
            n_pre=config.ec_l3_input_size,
            n_post=self.ca1_size,
            config=get_stp_config("ec_ca1"),
            per_synapse=True,
        )
        self.stp_ec_ca1.to(self.device)

        # =========================================================================
        # CA2 PATHWAYS: Social memory and temporal context
        # =========================================================================
        # CA3→CA2: DEPRESSION - stability mechanism
        # Weak plasticity (10x lower than typical) prevents runaway activity
        # and makes CA2 resistant to CA3 pattern completion interference
        self.stp_ca3_ca2 = ShortTermPlasticity(
            n_pre=self.ca3_size,
            n_post=self.ca2_size,
            config=get_stp_config("schaffer_collateral"),  # Use Schaffer preset (depressing)
            per_synapse=True,
        )
        self.stp_ca3_ca2.to(self.device)

        # CA2→CA1: FACILITATING - temporal sequences
        # Repeated CA2 activity facilitates transmission to CA1,
        # supporting temporal context and sequence encoding
        self.stp_ca2_ca1 = ShortTermPlasticity(
            n_pre=self.ca2_size,
            n_post=self.ca1_size,
            config=get_stp_config("mossy_fiber"),  # Use mossy fiber preset (facilitating)
            per_synapse=True,
        )
        self.stp_ca2_ca1.to(self.device)

        # EC→CA2 Direct: DEPRESSION - similar to EC→CA1
        # Direct cortical input to CA2 for temporal encoding
        # Created with size=0, will grow when sources added
        self.stp_ec_ca2 = ShortTermPlasticity(
            n_pre=0,  # Grow when sources added
            n_post=self.ca2_size,
            config=get_stp_config("ec_ca1"),  # Use EC→CA1 preset (depressing)
            per_synapse=True,
        )
        self.stp_ec_ca2.to(self.device)

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
        ca3_ca2_delay_steps = int(config.ca3_to_ca2_delay_ms / config.dt_ms)
        ca2_ca1_delay_steps = int(config.ca2_to_ca1_delay_ms / config.dt_ms)
        ca3_ca1_delay_steps = int(config.ca3_to_ca1_delay_ms / config.dt_ms)

        self._dg_ca3_buffer = CircularDelayBuffer(
            max_delay=dg_ca3_delay_steps,
            size=self.dg_size,
            device=str(self.device),
            dtype=torch.bool,
        )
        self._ca3_ca2_buffer = CircularDelayBuffer(
            max_delay=ca3_ca2_delay_steps,
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
        self._ca3_ca1_buffer = CircularDelayBuffer(
            max_delay=ca3_ca1_delay_steps,
            size=self.ca3_size,
            device=str(self.device),
            dtype=torch.bool,
        )

        # Store delay steps for conditional checks
        self._dg_ca3_delay_steps = dg_ca3_delay_steps
        self._ca3_ca2_delay_steps = ca3_ca2_delay_steps
        self._ca2_ca1_delay_steps = ca2_ca1_delay_steps
        self._ca3_ca1_delay_steps = ca3_ca1_delay_steps

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
        # Homeostasis for CA3 recurrent synaptic scaling
        self.homeostasis = UnifiedHomeostasis(UnifiedHomeostasisConfig(
            weight_budget=config.weight_budget * self.ca3_size,
            w_min=config.w_min,
            w_max=config.w_max,
            soft_normalization=config.soft_normalization,
            normalization_rate=config.normalization_rate,
            device=str(self.device),
        ))

        # Intrinsic plasticity state (threshold adaptation)
        self._ca3_activity_history: Optional[torch.Tensor] = None
        self._ca3_threshold_offset: Optional[torch.Tensor] = None

        # Synaptic tagging for emergent priority
        # Tags mark recently-active synapses for consolidation
        # Provides biological priority mechanism without explicit Episode objects
        self.synaptic_tagging = SynapticTagging(
            n_neurons=self.ca3_size,
            device=self.device,
            tag_decay=0.95,  # ~20 timestep lifetime
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
        # MULTI-TIMESCALE CONSOLIDATION
        # =========================================================================
        self._ca3_ca3_fast: Optional[torch.Tensor] = None
        self._ca3_ca3_slow: Optional[torch.Tensor] = None
        self._ca3_ca2_fast: Optional[torch.Tensor] = None
        self._ca3_ca2_slow: Optional[torch.Tensor] = None
        # NOTE: EC→CA1 and EC→CA2 learning handled by per-source plasticity rules
        # No separate consolidation traces needed in multi-source architecture
        self._ca2_ca1_fast: Optional[torch.Tensor] = None
        self._ca2_ca1_slow: Optional[torch.Tensor] = None

        # Initialize fast and slow eligibility traces for multi-timescale learning
        # Pattern: {pathway}_fast and {pathway}_slow for each learned connection
        # CA3 recurrent (autoassociative memory)
        self._ca3_ca3_fast = torch.zeros(self.ca3_size, self.ca3_size, device=self.device)
        self._ca3_ca3_slow = torch.zeros(self.ca3_size, self.ca3_size, device=self.device)

        # CA3→CA2 (temporal context)
        self._ca3_ca2_fast = torch.zeros(self.ca2_size, self.ca3_size, device=self.device)
        self._ca3_ca2_slow = torch.zeros(self.ca2_size, self.ca3_size, device=self.device)

        # NOTE: EC→CA1 and EC→CA2 learning handled by per-source weights (cortex_ca1,
        # thalamus_ca1, etc.) with their own plasticity rules. Multi-source architecture
        # eliminates need for single EC pathway consolidation.

        # CA2→CA1 (context to output)
        self._ca2_ca1_fast = torch.zeros(self.ca1_size, self.ca2_size, device=self.device)
        self._ca2_ca1_slow = torch.zeros(self.ca1_size, self.ca2_size, device=self.device)

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY
        # =====================================================================
        # Track per-subregion firing rates and adaptive gains (Turrigiano 2008)
        self._target_rate = config.target_firing_rate
        self._gain_lr = config.gain_learning_rate
        self._firing_rate_alpha = config.dt_ms / config.gain_tau_ms

        # Per-subregion firing rate trackers (exponential moving average)
        self.register_buffer("dg_firing_rate", torch.zeros(self.dg_size, device=self.device))
        self.register_buffer("ca3_firing_rate", torch.zeros(self.ca3_size, device=self.device))
        self.register_buffer("ca2_firing_rate", torch.zeros(self.ca2_size, device=self.device))
        self.register_buffer("ca1_firing_rate", torch.zeros(self.ca1_size, device=self.device))

        # Per-subregion adaptive gains (learnable parameters)
        self.dg_gain = nn.Parameter(torch.ones(self.dg_size, device=self.device, requires_grad=False))
        self.ca3_gain = nn.Parameter(torch.ones(self.ca3_size, device=self.device, requires_grad=False))
        self.ca2_gain = nn.Parameter(torch.ones(self.ca2_size, device=self.device, requires_grad=False))
        self.ca1_gain = nn.Parameter(torch.ones(self.ca1_size, device=self.device, requires_grad=False))

        # Store gain bounds
        self._baseline_noise = config.baseline_noise_current

        # Adaptive threshold plasticity (complementary to gain)
        self._threshold_lr = config.threshold_learning_rate
        self._threshold_min = config.threshold_min
        self._threshold_max = config.threshold_max

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
            spike_amplitude=0.1,  # Moderate amplitude for stable learning
            device=self.device,
        )
        # Per-subregion DA concentration buffers
        self._da_concentration_dg = torch.zeros(self.dg_size, device=self.device)
        self._da_concentration_ca3 = torch.zeros(self.ca3_size, device=self.device)
        self._da_concentration_ca2 = torch.zeros(self.ca2_size, device=self.device)
        self._da_concentration_ca1 = torch.zeros(self.ca1_size, device=self.device)

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
        # INTERNAL WEIGHTS: Migrated to synaptic_weights dict
        # =====================================================================
        # All weights at target dendrites for consistency
        # Pattern: {source}_{target} naming, e.g., "dg_ca3" = DG→CA3 at CA3 dendrites

        # DG → CA3: Random but less sparse (mossy fibers) - AT CA3 DENDRITES
        # Enhancement: Increased weight_scale from 0.5 to 2.0 to enable CA3 attractor bootstrap
        # Biological rationale: Mossy fiber synapses are among the largest in the brain
        # and provide powerful "detonator" synapses that can reliably drive CA3 pyramidal cells
        self.synaptic_weights["dg_ca3"] = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.ca3_size,
                n_input=self.dg_size,
                sparsity=0.5,
                weight_scale=2.0,  # Strengthened to bootstrap CA3 attractor (was 0.5)
                normalize_rows=True,  # Normalize for reliable propagation
                device=device,
            ),
            requires_grad=False,
        )

        # CA3 → CA3 RECURRENT: Autoassociative memory weights (AT CA3 DENDRITES)
        # Receives recurrent input via "ca3_recurrent" port with 2ms axonal delay
        # Learning: One-shot Hebbian with fast/slow traces and heterosynaptic LTD
        ca3_recurrent_weights = WeightInitializer.sparse_random(
            n_output=self.ca3_size,
            n_input=self.ca3_size,
            sparsity=0.25,  # ~25% connectivity
            weight_scale=0.5,  # Strong recurrence for attractor dynamics
            normalize_rows=True,
            device=device,
        )

        # Apply phase diversity: ±15% weight variation for temporal coding
        # Phase leads/lags enable different neurons to fire at different theta phases
        phase_diversity = torch.randn_like(ca3_recurrent_weights) * 0.15
        ca3_recurrent_weights = ca3_recurrent_weights * (1.0 + phase_diversity)
        ca3_recurrent_weights = torch.clamp(ca3_recurrent_weights, min=0.0)  # Keep positive
        ca3_recurrent_weights.requires_grad_(False)

        self.synaptic_weights["ca3_ca3"] = nn.Parameter(ca3_recurrent_weights, requires_grad=False)

        # =====================================================================
        # CA2 PATHWAYS: Social memory and temporal context
        # =====================================================================
        # CA3 → CA2: Weak plasticity (stability mechanism) - AT CA2 DENDRITES
        # CA2 is resistant to CA3 pattern completion interference
        self.synaptic_weights["ca3_ca2"] = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.ca2_size,
                n_input=self.ca3_size,
                sparsity=0.3,  # Moderate connectivity
                weight_scale=0.2,  # Weaker than typical (stability)
                normalize_rows=True,
                device=device,
            ),
            requires_grad=False,
        )

        # CA2 → CA1: Output to decision layer - AT CA1 DENDRITES
        # Provides temporal/social context to CA1 processing
        self.synaptic_weights["ca2_ca1"] = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.ca1_size,
                n_input=self.ca2_size,
                sparsity=0.2,  # Selective projection
                weight_scale=0.3,  # Moderate weights
                normalize_rows=False,  # Pattern-specific
                device=device,
            ),
            requires_grad=False,
        )

        # CA3 → CA1: Feedforward (retrieved memory) - SPARSE! - AT CA1 DENDRITES
        # This is the DIRECT bypass pathway (Schaffer collaterals)
        self.synaptic_weights["ca3_ca1"] = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.ca1_size,
                n_input=self.ca3_size,
                sparsity=0.15,  # Each CA1 sees only 15% of CA3
                weight_scale=0.3,  # Strong individual weights
                normalize_rows=False,  # NO normalization - pattern-specific!
                device=device,
            ),
            requires_grad=False,
        )

        # CA1 lateral inhibition for competition - AT CA1 DENDRITES
        # Use sparse lateral inhibition (similar to CA3 basket cells)
        # Biologically: CA1 interneurons have local connectivity, not all-to-all
        ca1_inhib_weights = torch.rand(self.ca1_size, self.ca1_size, device=device)
        # Make it sparse (20% connectivity)
        mask = torch.rand(self.ca1_size, self.ca1_size, device=device) < 0.2
        ca1_inhib_weights *= mask.float()
        # Zero diagonal (no self-inhibition)
        ca1_inhib_weights.fill_diagonal_(0.0)
        # Normalize rows so inhibition strength is bounded
        row_sums = ca1_inhib_weights.sum(dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1.0)
        ca1_inhib_weights /= row_sums
        # Scale to reasonable strength (not 0.5!)
        ca1_inhib_weights *= 0.05  # Much weaker than excitation

        self.synaptic_weights["ca1_inhib"] = nn.Parameter(ca1_inhib_weights, requires_grad=False)

        # Create gap junction network for CA1 interneurons
        self.gap_junctions_ca1 = GapJunctionCoupling(
            n_neurons=self.ca1_size,
            afferent_weights=self.synaptic_weights["ca1_inhib"],
            config=self._gap_config_ca1,
            device=device,
        )

    # =========================================================================
    # SYNAPTIC WEIGHT MANAGEMENT
    # =========================================================================

    def add_input_source(
        self,
        source_name: SpikesSourceKey,
        target_layer: LayerName,
        n_input: int,
        sparsity: float = 0.8,
        weight_scale: float = 1.0,
    ) -> None:
        """Override to create per-layer weights to DG, CA3, and CA1.

        Hippocampus has multiple entry points for inputs:
        - DG: Pattern separation (sparse encoding)
        - CA3: Direct perforant path (retrieval cues)
        - CA1: Temporoammonic path (cortical bypass)

        This method creates separate weight matrices for each layer.

        Args:
            source_name: Name of input source (e.g., "cortex:l5", "pfc:executive")
            input_port: Port for routing
            n_input: Size of input from this source
            sparsity: Connection sparsity (0-1, higher = more sparse)
            weight_scale: Scaling factor for weight initialization
        """
        # Call parent to register source (routes to default layer via port)
        super().add_input_source(
            source_name=source_name,
            target_layer=target_layer,
            n_input=n_input,
            sparsity=sparsity,
            weight_scale=weight_scale,
        )

        # Grow STP modules for new input sources
        if n_input > 0:
            self.stp_ec_ca2.grow(n_input, target="pre")

        # Initialize EC weights if this is first source (excluding internal)
        if not source_name.startswith("_"):
            # Ensure EC→CA1 weights exist (for direct cortical input to CA1)
            if self.n_input > 0 and "ec_ca2" not in self.synaptic_weights:
                # EC → CA2 weights
                self.synaptic_weights["ec_ca2"] = nn.Parameter(
                    WeightInitializer.sparse_random(
                        n_output=self.ca2_size,
                        n_input=self.n_input,
                        sparsity=0.3,  # Similar to EC→CA3
                        weight_scale=0.4,  # Strong direct encoding
                        normalize_rows=True,
                        device=self.device,
                    ),
                    requires_grad=False,
                )

        # Create additional per-layer weights for hippocampal multi-entry architecture
        # Pattern: {source}_dg, {source}_ca3, {source}_ca1

        # Source → DG: Sparse projections for pattern separation
        self.synaptic_weights[f"{source_name}_dg"] = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.dg_size,
                n_input=n_input,
                sparsity=0.3,  # 30% connectivity for strong pattern separation
                weight_scale=0.5,  # Increased from 0.1 for stronger drive
                normalize_rows=False,  # NO normalization - prevents massive row sums in sparse matrices
                device=self.device,
            ),
            requires_grad=False,
        )

        # Source → CA3: Direct perforant path (layer II) for retrieval cues
        self.synaptic_weights[f"{source_name}_ca3"] = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.ca3_size,
                n_input=n_input,
                sparsity=0.4,  # Less sparse than DG
                weight_scale=0.5,  # Increased from 0.1 for stronger drive
                normalize_rows=False,  # NO normalization - prevents massive row sums in sparse matrices
                device=self.device,
            ),
            requires_grad=False,
        )

        # Source → CA1: Direct pathway for immediate output
        self.synaptic_weights[f"{source_name}_ca1"] = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.ca1_size,
                n_input=n_input,
                sparsity=0.20,  # Relatively dense for rich integration
                weight_scale=2.0,  # INCREASED: Need sufficient drive for CA1 output
                normalize_rows=False,  # Allow pattern-specific weights
                device=self.device,
            ),
            requires_grad=False,
        )

    # =========================================================================
    # CONSOLIDATION MODE
    # =========================================================================

    def enter_consolidation_mode(self) -> None:
        """Enter consolidation mode (sleep/offline replay)."""
        # Enable consolidation mode flag for replay logic in forward()
        self._consolidation_mode = True

    def exit_consolidation_mode(self) -> None:
        """Exit consolidation mode and return to encoding."""
        # Disable consolidation mode
        self._consolidation_mode = False
        self._replay_cue = None

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _forward_internal(self, inputs: RegionSpikesDict) -> None:
        """Process input spikes through DG→CA3→CA1 circuit."""
        # =====================================================================
        # DOPAMINE RECEPTOR PROCESSING (from VTA)
        # =====================================================================
        # Process VTA dopamine spikes → concentration dynamics
        # Hippocampus receives minimal (10%) DA innervation for novelty/salience
        vta_da_spikes = inputs.get("vta:da_output")
        if vta_da_spikes is not None:
            # Update full receptor array
            da_concentration_full = self.da_receptor.update(vta_da_spikes)
            # Split into per-subregion buffers
            total_neurons = self.dg_size + self.ca3_size + self.ca2_size + self.ca1_size
            self._da_concentration_dg = da_concentration_full[: self.dg_size] * 0.1  # 10% projection strength
            self._da_concentration_ca3 = da_concentration_full[self.dg_size : self.dg_size + self.ca3_size] * 0.1
            self._da_concentration_ca2 = da_concentration_full[self.dg_size + self.ca3_size : self.dg_size + self.ca3_size + self.ca2_size] * 0.1
            self._da_concentration_ca1 = da_concentration_full[self.dg_size + self.ca3_size + self.ca2_size :] * 0.1
        else:
            # Just decay if no VTA input
            da_concentration_full = self.da_receptor.update(None)
            self._da_concentration_dg = da_concentration_full[: self.dg_size] * 0.1
            self._da_concentration_ca3 = da_concentration_full[self.dg_size : self.dg_size + self.ca3_size] * 0.1
            self._da_concentration_ca2 = da_concentration_full[self.dg_size + self.ca3_size : self.dg_size + self.ca3_size + self.ca2_size] * 0.1
            self._da_concentration_ca1 = da_concentration_full[self.dg_size + self.ca3_size + self.ca2_size :] * 0.1

        # =====================================================================
        # CONSOLIDATION MODE: Spontaneous CA3→CA1 Replay (Sharp-Wave Ripples)
        # =====================================================================
        # During sleep consolidation, hippocampus spontaneously reactivates stored
        # patterns without external input. This simulates sharp-wave ripples where
        # CA3 recurrent activity triggers CA1 output for cortical replay.
        #
        # Biological mechanism (Hasselmo 1999):
        # - LOW acetylcholine enables CA3 spontaneous reactivation
        # - CA3 attractor pattern propagates through Schaffer collaterals to CA1
        # - STP dynamics preserved (biological timing maintained)
        # - CA1 output drives cortical consolidation via back-projections

        # Reset ripple detection flag
        self.ripple_detected = False

        # Check if spontaneous replay should occur (low ACh, probabilistic trigger)
        if self.spontaneous_replay is not None:
            should_replay = self.spontaneous_replay.should_trigger_ripple(
                acetylcholine=self.neuromodulator_state.acetylcholine,
                dt_ms=self.config.dt_ms,
            )

            if should_replay:
                # Select pattern to replay based on tags and weight strength
                # Only trigger ripple if we can actually select a pattern
                if self.synaptic_tagging is not None and "ca3_ca3" in self.synaptic_weights:
                    # Mark ripple detected
                    self.ripple_detected = True

                    seed_pattern = self.spontaneous_replay.select_pattern_to_replay(
                        synaptic_tags=self.synaptic_tagging.tags,
                        ca3_weights=self.synaptic_weights["ca3_ca3"],
                        seed_fraction=0.15,  # ~15% of CA3 neurons
                    )

                    # Inject seed pattern into CA3 persistent activity
                    # This triggers attractor dynamics for pattern completion
                    self.ca3_persistent = self.ca3_persistent * 0.5 + seed_pattern.float() * 2.0

        if self._consolidation_mode and self._replay_cue is not None:
            self._replay_cue = None
            return  # Skip normal processing during replay timestep (CA3→CA1 driven by internal dynamics)

        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        dg_input = self._integrate_multi_source_synaptic_inputs(
            inputs=inputs,
            n_neurons=self.dg_size,
            weight_key_suffix="_dg",
            apply_stp=False,
        )

        ca3_input = self._integrate_multi_source_synaptic_inputs(
            inputs=inputs,
            n_neurons=self.ca3_size,
            weight_key_suffix="_ca3",
            apply_stp=False,
        )

        ca1_input = self._integrate_multi_source_synaptic_inputs(
            inputs=inputs,
            n_neurons=self.ca1_size,
            weight_key_suffix="_ca1",
            apply_stp=False,
        )

        # =====================================================================
        # GET SEPTAL INPUT (for OLM phase-locking)
        # =====================================================================
        # Septal GABAergic input drives theta rhythm by phase-locking OLM cells
        # OLM cells rebound at theta troughs → dendritic inhibition → encoding/retrieval
        septal_gaba = inputs.get("septal_gaba", None)
        if septal_gaba is not None and septal_gaba.numel() == 0:
            septal_gaba = None  # Treat empty tensor as None

        # Initialize encoding/retrieval modulation (will be updated from OLM dynamics)
        # Default to balanced state if no septal input yet
        encoding_mod = 0.5
        retrieval_mod = 0.5

        # =====================================================================
        # STIMULUS GATING (TRANSIENT INHIBITION)
        # =====================================================================
        # Compute stimulus-onset inhibition based on DG input change
        # Use DG input as proxy for overall input change
        ffi = self.stimulus_gating.compute(dg_input, return_tensor=False)
        raw_ffi = ffi.item() if hasattr(ffi, "item") else float(ffi)
        # Normalize to [0, 1] by dividing by max_inhibition
        ffi_strength = min(1.0, raw_ffi / self.stimulus_gating.max_inhibition)
        # ffi_strength is now normalized to [0, 1], config.ffi_strength controls max suppression
        ffi_factor = 1.0 - ffi_strength * self.config.ffi_strength

        # =====================================================================
        # 1. DENTATE GYRUS: Pattern Separation
        # =====================================================================
        # DG input already integrated from all sources above
        # Apply FFI: reduce DG drive when input changes significantly
        dg_code = dg_input * ffi_factor

        # Apply homeostatic gain and baseline noise (Turrigiano 2008)
        # Add baseline noise (spontaneous miniature EPSPs)
        if self._baseline_noise > 0:
            dg_code = dg_code + self._baseline_noise
        # Apply adaptive gain
        dg_code = dg_code * self.dg_gain

        # Run through DG neurons (ConductanceLIF expects g_exc, g_inh)
        # DG has minimal inhibition - primarily feedforward excitation for pattern separation
        dg_g_exc = F.relu(dg_code)  # Clamp to positive conductance

        dg_spikes, _ = self.dg_neurons(dg_g_exc, g_inh_input=None)

        # Apply extreme winner-take-all sparsity
        # Use pre-spike membrane (not post-spike which resets to v_reset)
        dg_spikes = self._apply_wta_sparsity(
            dg_spikes,
            self.config.dg_sparsity,
            membrane=self.dg_neurons.membrane_pre_spike,
        )

        # Update homeostatic gain for DG
        # Update firing rate (exponential moving average)
        self.dg_firing_rate = (
            1 - self._firing_rate_alpha
        ) * self.dg_firing_rate + self._firing_rate_alpha * dg_spikes.float()
        # Compute rate error
        rate_error = self._target_rate - self.dg_firing_rate
        # Update gain (increase gain if firing too low, decrease if too high)
        # Clamp to prevent negative gains
        self.dg_gain.data = (self.dg_gain + self._gain_lr * rate_error).clamp(min=0.001)
        # Adaptive threshold: lower threshold when underactive, raise when overactive
        self.dg_neurons.v_threshold.data = torch.clamp(
            self.dg_neurons.v_threshold - self._threshold_lr * rate_error,
            min=self._threshold_min,
            max=self._threshold_max,
        )

        # =====================================================================
        # APPLY DG→CA3 AXONAL DELAY
        # =====================================================================
        # Apply biological transmission delay for DG→CA3 mossy fiber pathway
        # If delay is 0, dg_spikes_delayed = dg_spikes (instant, backward compatible)
        if self._dg_ca3_delay_steps > 0:
            # Write current spikes and read delayed spikes
            self._dg_ca3_buffer.write(dg_spikes)
            dg_spikes_delayed = self._dg_ca3_buffer.read(self._dg_ca3_delay_steps)
            self._dg_ca3_buffer.advance()
        else:
            dg_spikes_delayed = dg_spikes

        # Inter-stage shape check: DG output → CA3 input
        assert dg_spikes.shape == (self.dg_size,), (
            f"Hippocampus: DG spikes have shape {dg_spikes.shape} "
            f"but expected ({self.dg_size},). "
            f"Check DG sparsity or EC→DG weights shape."
        )

        # =====================================================================
        # 2. CA3: Pattern Completion via Recurrence + Bistable Dynamics
        # =====================================================================
        # EMERGENT THETA MODULATION via OLM cells:
        # Instead of hardcoded arithmetic, encoding/retrieval separation emerges from:
        # 1. Septal GABA inhibits OLM cells at theta peaks
        # 2. OLM cells rebound at theta troughs (rebound bursting)
        # 3. OLM → CA1 apical dendrites suppresses retrieval pathway
        # 4. When OLM fires (theta trough), dendritic inhibition blocks EC→CA1
        # 5. When OLM silent (theta peak), EC→CA1 flows freely (retrieval)
        #
        # Result: EMERGENT encoding (trough) / retrieval (peak) without hardcoding!
        #
        # BISTABLE NEURONS: Real CA3 pyramidal neurons have intrinsic bistability
        # via I_NaP (persistent sodium) and I_CAN (Ca²⁺-activated cation) currents.
        # We model this with a persistent activity trace that:
        #   1. Accumulates when neurons fire
        #   2. Decays slowly (τ ~100-200ms)
        #   3. Provides positive feedback (self-sustaining activity)
        # This enables stable attractor states during delay periods.

        # Feedforward from DG (mossy fibers, theta-gated) with optional STP
        # NOTE: Use delayed DG spikes for biological accuracy
        if self.stp_mossy is not None:
            # Get STP efficacy for mossy fiber synapses
            # Mossy fibers are FACILITATING - repeated DG spikes progressively
            # enhance transmission to CA3
            stp_efficacy = self.stp_mossy(dg_spikes_delayed.float())
            # Apply STP to weights: (n_post, n_pre) * (n_pre, n_post).T
            effective_w_dg_ca3 = self.synaptic_weights["dg_ca3"] * stp_efficacy.T
            ca3_from_dg = (
                torch.matmul(effective_w_dg_ca3, dg_spikes_delayed.float())
            )  # [ca3_size]
        else:
            # Standard matmul without STP
            ca3_from_dg = (
                torch.matmul(self.synaptic_weights["dg_ca3"], dg_spikes_delayed.float())
            )  # [ca3_size]

        # Direct perforant path from EC (provides retrieval cues)
        # Strong during retrieval to seed the CA3 attractor from partial cues
        # NOTE: ca3_input from _integrate_multi_source_synaptic_inputs includes
        # ALL external sources (EC, cortex, PFC, thalamus, etc.) but NOT DG
        # (DG→CA3 is computed separately above to apply STP)
        ca3_from_external = ca3_input  # [ca3_size]

        # Total feedforward input to CA3
        # Combine DG mossy fibers (with STP) + external sources (EC, etc.)
        ca3_ff = ca3_from_dg + ca3_from_external

        # =====================================================================
        # CA3 RECURRENT INPUT (Internal Computation)
        # =====================================================================
        # CA3 recurrent connections are computed internally using ca3_ca3 weights.
        # Uses previous timestep's CA3 spikes (provides 1 dt delay, ~1-2ms)
        # to compute recurrent excitation for current timestep.
        #
        # ACh modulation applied here (region-level neuromodulation):
        # High ACh (encoding mode): Suppress recurrence (0.3x)
        # Low ACh (retrieval mode): Full recurrence (1.0x)
        ach_level = self.neuromodulator_state.acetylcholine
        ach_recurrent_modulation = compute_ach_recurrent_suppression(ach_level)

        # Compute recurrent input from previous CA3 activity
        if self.ca3_spikes is not None:
            # Use delayed CA3 spikes (from previous timestep)
            ca3_rec_raw = torch.matmul(
                self.synaptic_weights["ca3_ca3"],
                self.ca3_spikes.float()
            )
        else:
            # No previous activity - zero recurrent input
            ca3_rec_raw = torch.zeros(self.ca3_size, device=self.device)

        # Apply region-level modulation (ACh and strength scaling)
        ca3_rec = (
            ca3_rec_raw
            * self.config.ca3_recurrent_strength
            * ach_recurrent_modulation
        )  # [ca3_size]

        # =====================================================================
        # ACTIVITY-DEPENDENT FEEDBACK INHIBITION (Biologically Accurate)
        # =====================================================================
        # In real CA3, pyramidal cells recruit basket cell interneurons which
        # provide lateral inhibition back to the pyramidal population.
        # This creates local competition and prevents runaway activity.
        #
        # Implementation: Each neuron's inhibition is proportional to the
        # activity of ALL other neurons (lateral inhibition), weighted by
        # a local connectivity pattern.

        # Initialize lateral inhibition weight matrix if needed (sparse local connectivity)
        if not hasattr(self, "_ca3_lateral_inhib_weights"):
            # Create sparse local inhibition: each neuron inhibits nearby neurons
            # Biologically: basket cells have local axonal arbors (~200-300μm radius)
            # We approximate this with random sparse connectivity
            self._ca3_lateral_inhib_weights = torch.rand(
                self.ca3_size, self.ca3_size, device=self.device
            )
            # Make it sparse (20% connectivity for basket cell → pyramidal)
            mask = torch.rand(self.ca3_size, self.ca3_size, device=self.device) < 0.2
            self._ca3_lateral_inhib_weights *= mask.float()
            # Zero out self-connections (neurons don't inhibit themselves)
            self._ca3_lateral_inhib_weights.fill_diagonal_(0.0)
            # Normalize rows so total inhibition per neuron is bounded
            row_sums = self._ca3_lateral_inhib_weights.sum(dim=1, keepdim=True)
            row_sums = torch.clamp(row_sums, min=1.0)  # Avoid division by zero
            self._ca3_lateral_inhib_weights /= row_sums

        # Compute per-neuron feedback inhibition based on population activity
        # Each neuron receives inhibition proportional to others' activity
        if self.ca3_spikes is not None:
            # Lateral inhibition: weight matrix @ spike vector
            # Result: each neuron's inhibition is weighted sum of other neurons' spikes
            feedback_inhibition = torch.matmul(
                self._ca3_lateral_inhib_weights,
                self.ca3_spikes.float()
            ) * self.config.ca3_feedback_inhibition
        else:
            feedback_inhibition = torch.zeros(self.ca3_size, device=self.device)

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

        ca3_persistent_input = self.ca3_persistent * self.config.ca3_persistent_gain  # [ca3_size]

        # Total CA3 input = feedforward + recurrent + persistent - inhibition
        ca3_input = ca3_ff + ca3_rec + ca3_persistent_input - feedback_inhibition

        # =====================================================================
        # NOREPINEPHRINE GAIN MODULATION (Locus Coeruleus)
        # =====================================================================
        # High NE (arousal/uncertainty): Increase gain → more responsive
        # Low NE (baseline): Normal gain
        # Biological: β-adrenergic receptors increase neuronal excitability
        ne_level = self.neuromodulator_state.norepinephrine
        # NE gain: 1.0 (baseline) to 1.5 (high arousal)
        ne_gain = compute_ne_gain(ne_level)
        ca3_input = ca3_input * ne_gain

        # INTRINSIC PLASTICITY: Apply per-neuron threshold offset
        # Neurons that fire too much have higher thresholds (less excitable)
        if self._ca3_threshold_offset is not None:
            ca3_input = ca3_input - self._ca3_threshold_offset

        # =====================================================================
        # EMERGENT PHASE CODING (replaces explicit slot gating)
        # =====================================================================
        # Phase preferences emerge naturally from:
        # 1. Weight diversity (timing jitter in initialization)
        # 2. STDP (strengthens connections at successful firing times)
        # 3. Dendritic integration windows (~15ms)
        # 4. Recurrent dynamics (neurons that fire together wire together)
        #
        # No explicit gating needed - temporal structure emerges from dynamics!
        # Gamma amplitude still modulates overall excitability:
        # - High gamma (theta trough): Enhanced responsiveness to inputs
        # - Low gamma (theta peak): Reduced responsiveness, recurrence dominates

        # Gamma amplitude modulation (emergent from oscillator coupling)
        # High amplitude → neurons more responsive to current input
        # Low amplitude → neurons rely more on recurrent memory
        gamma_amplitude = self._gamma_amplitude_effective  # [0, 1]

        # Scale input by gamma (but NO slot-based gating)
        # This creates temporal windows where input is more/less effective
        # Combined with weight diversity, this leads to phase preferences
        scale = 0.5
        gamma_modulation = scale + scale * gamma_amplitude  # [0.5, 1.0]
        ca3_input = ca3_input * gamma_modulation

        # Apply homeostatic gain and baseline noise (Turrigiano 2008)
        # Add baseline noise (spontaneous miniature EPSPs)
        if self._baseline_noise > 0:
            ca3_input = ca3_input + self._baseline_noise
        # Apply adaptive gain
        ca3_input = ca3_input * self.ca3_gain

        # =====================================================================
        # CA3 INHIBITORY NETWORK (PV, OLM, Bistratified)
        # =====================================================================
        # Run CA3 inhibitory network with septal input
        # OLM cells phase-lock to septal GABA for theta modulation
        if hasattr(self, 'ca3_inhibitory'):
            ca3_inhib_output = self.ca3_inhibitory(
                pyr_spikes=self.ca3_spikes if self.ca3_spikes is not None else torch.zeros(self.ca3_size, device=self.device, dtype=torch.bool),
                septal_gaba=septal_gaba,
                external_exc=None,  # Optional external drive
            )
            ca3_perisomatic_inhib = ca3_inhib_output["perisomatic"]
        else:
            ca3_perisomatic_inhib = torch.zeros(self.ca3_size, device=self.device)

        # Run through CA3 neurons (ConductanceLIF expects g_exc, g_inh)
        # Inhibition is handled by inhibitory network
        ca3_g_exc = F.relu(ca3_input)
        ca3_g_inh = F.relu(ca3_perisomatic_inhib)  # Perisomatic inhibition from PV cells

        ca3_spikes, _ = self.ca3_neurons(ca3_g_exc, g_inh_input=ca3_g_inh)

        # CRITICAL FIX FOR OSCILLATION:
        # The LIF neuron resets membrane to v_reset after spiking, which causes
        # neurons that just spiked to have LOW membrane on the next timestep.
        # This makes WTA select DIFFERENT neurons, causing oscillation.
        #
        # Solution: After LIF processing, restore membrane potential for neurons
        # with high persistent activity. This models how I_NaP keeps neurons
        # near threshold even after spiking.
        # Boost membrane for neurons with high persistent activity
        assert self.ca3_neurons.membrane is not None, "CA3 membrane should exist after forward()"
        persistent_boost = self.ca3_persistent * 1.5
        self.ca3_neurons.membrane = self.ca3_neurons.membrane + persistent_boost

        # Apply sparsity - use pre-spike membrane for WTA
        # Persistent boost is applied to membrane before spike check, so pre-spike membrane reflects it
        ca3_spikes = self._apply_wta_sparsity(
            ca3_spikes,
            self.config.ca3_sparsity,
            membrane=self.ca3_neurons.membrane_pre_spike,
        )

        # Update homeostatic gain for CA3
        # Update firing rate (exponential moving average)
        self.ca3_firing_rate = (
            1 - self._firing_rate_alpha
        ) * self.ca3_firing_rate + self._firing_rate_alpha * ca3_spikes.float()
        # Compute rate error
        rate_error = self._target_rate - self.ca3_firing_rate
        # Update gain (increase gain if firing too low, decrease if too high)
        self.ca3_gain.data = (self.ca3_gain + self._gain_lr * rate_error).clamp(min=0.001)
        # Adaptive threshold
        self.ca3_neurons.v_threshold.data = torch.clamp(
            self.ca3_neurons.v_threshold - self._threshold_lr * rate_error,
            min=self._threshold_min,
            max=self._threshold_max,
        )

        # Inter-stage shape check: CA3 output → CA1 input
        assert ca3_spikes.shape == (self.ca3_size,), (
            f"Hippocampus: CA3 spikes have shape {ca3_spikes.shape} "
            f"but expected ({self.ca3_size},). "
            f"Check CA3 sparsity or DG→CA3 weights shape."
        )

        # Update persistent activity AFTER computing new spikes
        # The trace accumulates spike activity with slow decay
        # Using a direct accumulation: trace += spike - decay*trace
        # This ensures spikes have strong immediate effect but decay slowly
        dt_ms = self.config.dt_ms
        decay_rate = dt_ms / self.config.ca3_persistent_tau

        # Update persistent activity: stronger during encoding, decay otherwise
        # Encoding_mod determines how much new spikes contribute vs decay
        # This is biologically motivated: Ca²⁺-dependent currents build up during
        # active encoding, then decay during maintenance/retrieval
        # Continuous modulation: contribution naturally weak when encoding_mod is low
        self.ca3_persistent = (
            self.ca3_persistent * (1.0 - decay_rate * (0.5 + 0.5 * retrieval_mod))
            + ca3_spikes.float()
            * 0.5
            * encoding_mod  # Contribution scaled by encoding strength
        )

        # Clamp to prevent runaway
        self.ca3_persistent = torch.clamp(self.ca3_persistent, 0.0, 3.0)

        # =====================================================================
        # HEBBIAN LEARNING: Apply to CA3 recurrent weights
        # This is how the hippocampus "stores" the pattern - in the weights!
        # Also store the DG pattern for later match/mismatch detection.
        # Modulated by theta encoding strength!
        # =====================================================================
        # Store the DG pattern (accumulate over timesteps, scaled by encoding strength)
        # Continuous modulation: storage naturally weak when encoding_mod is low
        if encoding_mod > 0.01:  # Only accumulate if encoding has minimal presence
            self.stored_dg_pattern = (
                self.stored_dg_pattern + dg_spikes.float() * encoding_mod
            )

        # One-shot Hebbian: strengthen connections between co-active neurons
        # Learning rate modulated by theta phase AND gamma amplitude
        # Continuous: learning automatically weak when encoding_mod is low
        ca3_activity = ca3_spikes.float()  # Already 1D, no squeeze needed

        # =========================================================
        # MULTI-TIMESCALE CONSOLIDATION
        # =========================================================
        # Trace decay happens CONTINUOUSLY based on time, not conditionally
        # on activity. This is biologically accurate - molecular traces
        # decay with their own time constants regardless of neural activity.
        # Compute decay factors
        fast_decay = dt_ms / self.config.fast_trace_tau_ms
        slow_decay = dt_ms / self.config.slow_trace_tau_ms

        # Apply decay to fast trace
        self._ca3_ca3_fast = (1.0 - fast_decay) * self._ca3_ca3_fast

        # Apply decay to slow trace + consolidation transfer
        consolidation = self.config.consolidation_rate * self._ca3_ca3_fast
        self._ca3_ca3_slow = (1.0 - slow_decay) * self._ca3_ca3_slow + consolidation

        # Learning happens only when there's CA3 activity
        if ca3_activity.sum() > 0:
            # Hebbian outer product: neurons that fire together wire together
            #
            # Gamma amplitude modulation: Learning is stronger when gamma
            # is strong (theta trough, encoding phase). This implements
            # the biological finding that synaptic plasticity is enhanced
            # during periods of strong gamma oscillations.
            base_lr = self.config.learning_rate * encoding_mod

            # Apply automatic gamma amplitude modulation
            # Gamma is modulated by ALL slower oscillators (emergent multi-order coupling)
            gamma_mod = self._gamma_amplitude_effective
            effective_lr = compute_learning_rate_modulation(base_lr, gamma_mod)

            # Apply dopamine modulation to learning rate
            # Dopamine gates consolidation strength - higher DA = stronger learning
            # Biological basis: VTA dopamine signals reward/novelty and gates LTP
            da_level = self.neuromodulator_state.dopamine
            # Strong dopamine gating: 0.0 DA = 20% learning, 1.0 DA = 200% learning
            # This creates a 10x range between min and max dopamine
            da_gain = 0.2 + 1.8 * da_level  # Range: [0.2, 2.0]
            effective_lr = effective_lr * da_gain

            dW = effective_lr * torch.outer(ca3_activity, ca3_activity)

            # =========================================================
            # SYNAPTIC TAGGING
            # =========================================================
            # Update synaptic tags based on spike coincidence
            # Tags mark recently-active synapses for potential consolidation
            # Replaces explicit Episode.priority with emergent biological mechanism
            if self.synaptic_tagging is not None:
                self.synaptic_tagging.update_tags(
                    pre_spikes=ca3_activity,
                    post_spikes=ca3_activity,
                )

            # =========================================================
            # HETEROSYNAPTIC PLASTICITY: Weaken inactive synapses
            # =========================================================
            # Synapses to inactive postsynaptic neurons get weakened when
            # nearby neurons fire strongly. This prevents winner-take-all
            # dynamics from permanently dominating.
            #
            # Implementation: For each active presynaptic neuron, weaken
            # its connections to inactive postsynaptic neurons.
            if self.config.heterosynaptic_ratio > 0:
                inactive_post = (ca3_activity < 0.5).float()  # Inactive neurons
                active_pre = ca3_activity  # Active neurons
                # Weaken: pre active but post inactive
                hetero_ltd = self.config.heterosynaptic_ratio * effective_lr
                hetero_dW = -hetero_ltd * torch.outer(active_pre, inactive_post)
                dW = dW + hetero_dW

            # Add learning to traces
            # Accumulate new learning into fast trace
            self._ca3_ca3_fast = self._ca3_ca3_fast + dW

            # Combined weight update: Fast (episodic) + Slow (semantic)
            # Fast trace dominates initially, slow trace provides stability
            combined_dW = (
                self._ca3_ca3_fast + self.config.slow_trace_contribution * self._ca3_ca3_slow
            )
            self.synaptic_weights["ca3_ca3"].data += combined_dW

            self.synaptic_weights["ca3_ca3"].data.fill_diagonal_(0.0)  # No self-connections
            clamp_weights(
                self.synaptic_weights["ca3_ca3"].data, self.config.w_min, self.config.w_max
            )

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
        if self._ca3_ca2_delay_steps > 0:
            self._ca3_ca2_buffer.write(ca3_spikes)
            ca3_spikes_for_ca2 = self._ca3_ca2_buffer.read(self._ca3_ca2_delay_steps)
            self._ca3_ca2_buffer.advance()
        else:
            ca3_spikes_for_ca2 = ca3_spikes

        # CA3→CA2 input with STP (depressing - stability mechanism)
        if self.stp_ca3_ca2 is not None:
            stp_efficacy = self.stp_ca3_ca2(ca3_spikes_for_ca2.float())
            effective_w_ca3_ca2 = self.synaptic_weights["ca3_ca2"] * stp_efficacy.T
            ca2_from_ca3 = torch.matmul(effective_w_ca3_ca2, ca3_spikes_for_ca2.float())
        else:
            ca2_from_ca3 = torch.matmul(
                self.synaptic_weights["ca3_ca2"], ca3_spikes_for_ca2.float()
            )

        # CA2 external input: Currently CA2 doesn't have per-source external weights
        # It receives primarily from CA3. For now, use zero external input.
        # Future: Add per-source CA2 weights if needed for social/contextual processing
        ca2_from_ec = torch.zeros(self.ca2_size, device=self.device)

        # Combine CA2 inputs (both CA3 pattern and direct EC encoding)
        ca2_input = ca2_from_ca3 + ca2_from_ec

        # Apply homeostatic gain and baseline noise (Turrigiano 2008)
        # Add baseline noise (spontaneous miniature EPSPs)
        if self._baseline_noise > 0:
            ca2_input = ca2_input + self._baseline_noise
        # Apply adaptive gain
        ca2_input = ca2_input * self.ca2_gain

        # Run through CA2 neurons
        ca2_g_exc = F.relu(ca2_input)
        ca2_spikes, _ = self.ca2_neurons(ca2_g_exc, g_inh_input=None)

        # Apply sparsity - use pre-spike membrane for WTA
        ca2_spikes = self._apply_wta_sparsity(
            ca2_spikes,
            self.config.ca2_sparsity,
            membrane=self.ca2_neurons.membrane_pre_spike,
        )

        # Update homeostatic gain for CA2
        # Update firing rate (exponential moving average)
        self.ca2_firing_rate = (
            1 - self._firing_rate_alpha
        ) * self.ca2_firing_rate + self._firing_rate_alpha * ca2_spikes.float()
        # Compute rate error
        rate_error = self._target_rate - self.ca2_firing_rate
        # Update gain (increase gain if firing too low, decrease if too high)
        self.ca2_gain.data = (self.ca2_gain + self._gain_lr * rate_error).clamp(min=0.001)
        # Adaptive threshold
        self.ca2_neurons.v_threshold.data = torch.clamp(
            self.ca2_neurons.v_threshold - self._threshold_lr * rate_error,
            min=self._threshold_min,
            max=self._threshold_max,
        )

        # CA3→CA2 WEAK PLASTICITY (stability mechanism)
        # 10x weaker learning than typical - prevents CA2 from being dominated by CA3
        ca3_activity_for_ca2 = ca3_spikes_for_ca2.float()
        ca2_activity = ca2_spikes.float()

        # Multi-timescale trace decay (continuous, time-based)
        fast_decay = dt_ms / self.config.fast_trace_tau_ms
        slow_decay = dt_ms / self.config.slow_trace_tau_ms

        self._ca3_ca2_fast = (1.0 - fast_decay) * self._ca3_ca2_fast
        consolidation = self.config.consolidation_rate * self._ca3_ca2_fast
        self._ca3_ca2_slow = (1.0 - slow_decay) * self._ca3_ca2_slow + consolidation

        # Learning only when there's activity
        if ca3_activity_for_ca2.sum() > 0 and ca2_activity.sum() > 0:
            # Very weak learning rate (stability hub)
            base_lr = self.config.ca3_ca2_learning_rate * encoding_mod

            gamma_mod = self._gamma_amplitude_effective
            effective_lr = compute_learning_rate_modulation(base_lr, gamma_mod)

            dW = effective_lr * torch.outer(ca2_activity, ca3_activity_for_ca2)

            self._ca3_ca2_fast = self._ca3_ca2_fast + dW
            combined_dW = (
                self._ca3_ca2_fast + self.config.slow_trace_contribution * self._ca3_ca2_slow
            )
            self.synaptic_weights["ca3_ca2"].data += combined_dW

            clamp_weights(
                self.synaptic_weights["ca3_ca2"].data, self.config.w_min, self.config.w_max
            )

        cfg = self.config

        # =====================================================================
        # APPLY CA3→CA1 AXONAL DELAY
        # =====================================================================
        # Apply biological transmission delay for CA3→CA1 Schaffer collateral pathway
        # If delay is 0, ca3_spikes_delayed = ca3_spikes (instant, backward compatible)
        if self._ca3_ca1_delay_steps > 0:
            self._ca3_ca1_buffer.write(ca3_spikes)
            ca3_spikes_delayed = self._ca3_ca1_buffer.read(self._ca3_ca1_delay_steps)
            self._ca3_ca1_buffer.advance()
        else:
            ca3_spikes_delayed = ca3_spikes

        # Feedforward from CA3 (retrieved/encoded memory) with optional STP
        # Schaffer collaterals are DEPRESSING - high-frequency CA3 activity
        # causes progressively weaker transmission to CA1
        # NOTE: Use delayed CA3 spikes for biological accuracy
        if self.stp_schaffer is not None:
            stp_efficacy = self.stp_schaffer(ca3_spikes_delayed.float())
            effective_w_ca3_ca1 = self.synaptic_weights["ca3_ca1"] * stp_efficacy.T
            ca1_from_ca3 = torch.matmul(
                effective_w_ca3_ca1, ca3_spikes_delayed.float()
            )  # [ca1_size]
        else:
            # Standard matmul without STP
            ca1_from_ca3 = torch.matmul(
                self.synaptic_weights["ca3_ca1"], ca3_spikes_delayed.float()
            )  # [ca1_size]

        # Multi-source architecture: ca1_input already integrates all external sources
        # Each source (cortex, thalamus, etc.) has separate weights to CA1 with their own learning
        ca1_from_ec = ca1_input  # [ca1_size]

        # Apply feedforward inhibition: strong input change reduces CA1 drive
        # This clears residual activity naturally
        ca1_from_ec = ca1_from_ec * ffi_factor
        ca1_from_ca3 = ca1_from_ca3 * ffi_factor

        # NMDA trace update (for retrieval gating)
        # Tracks CA3-induced depolarization for Mg²⁺ block removal
        if self.nmda_trace is not None:
            nmda_decay = torch.exp(torch.tensor(-dt_ms / cfg.nmda_tau))
            self.nmda_trace = self.nmda_trace * nmda_decay + ca1_from_ca3 * (
                1.0 - nmda_decay
            )
        else:
            self.nmda_trace = ca1_from_ca3.clone()

        # NMDA gating: Mg²⁺ block removal based on CA3 depolarization
        # Stronger during retrieval (theta peak)
        mg_block_removal = (
            torch.sigmoid((self.nmda_trace - cfg.nmda_threshold) * cfg.nmda_steepness)
            * retrieval_mod
        )
        nmda_current = ca1_from_ec * mg_block_removal

        # AMPA current: fast baseline transmission
        ampa_current = ca1_from_ec * cfg.ampa_ratio

        # CA3 contribution: stronger during encoding
        ca3_contribution = ca1_from_ca3 * (0.5 + 0.5 * encoding_mod)

        # APPLY CA2→CA1 AXONAL DELAY
        if self._ca2_ca1_delay_steps > 0:
            self._ca2_ca1_buffer.write(ca2_spikes)
            ca2_spikes_delayed = self._ca2_ca1_buffer.read(self._ca2_ca1_delay_steps)
            self._ca2_ca1_buffer.advance()
        else:
            ca2_spikes_delayed = ca2_spikes

        # CA2→CA1 contribution with STP (facilitating - temporal sequences)
        if self.stp_ca2_ca1 is not None:
            stp_efficacy = self.stp_ca2_ca1(ca2_spikes_delayed.float())
            effective_w_ca2_ca1 = self.synaptic_weights["ca2_ca1"] * stp_efficacy.T
            ca1_from_ca2 = torch.matmul(effective_w_ca2_ca1, ca2_spikes_delayed.float())
        else:
            ca1_from_ca2 = torch.matmul(
                self.synaptic_weights["ca2_ca1"], ca2_spikes_delayed.float()
            )

        # Apply FFI to CA2 contribution as well
        ca1_from_ca2 = ca1_from_ca2 * ffi_factor

        # Total CA1 input (now includes CA2 temporal/social context)
        ca1_input = ca3_contribution + ca1_from_ca2 + ampa_current + nmda_current

        # Apply homeostatic gain and baseline noise (Turrigiano 2008)
        # Add baseline noise (spontaneous miniature EPSPs)
        if self._baseline_noise > 0:
            ca1_input = ca1_input + self._baseline_noise
        # Apply adaptive gain
        ca1_input = ca1_input * self.ca1_gain

        # =====================================================================
        # CA1 INHIBITORY NETWORK (PV, OLM, Bistratified)
        # =====================================================================
        # Run CA1 inhibitory network with septal input
        # OLM cells create EMERGENT encoding/retrieval separation!
        if hasattr(self, 'ca1_inhibitory'):
            ca1_inhib_output = self.ca1_inhibitory(
                pyr_spikes=self.ca1_spikes if self.ca1_spikes is not None else torch.zeros(self.ca1_size, device=self.device, dtype=torch.bool),
                septal_gaba=septal_gaba,
                external_exc=ca3_contribution,  # CA3→CA1 drives inhibitory network
            )
            ca1_perisomatic_inhib = ca1_inhib_output["perisomatic"]  # PV cells
            ca1_dendritic_inhib = ca1_inhib_output["dendritic"]       # OLM + bistratified
            ca1_olm_inhib = ca1_inhib_output["olm_dendritic"]         # OLM only
        else:
            ca1_perisomatic_inhib = torch.zeros(self.ca1_size, device=self.device)
            ca1_dendritic_inhib = torch.zeros(self.ca1_size, device=self.device)
            ca1_olm_inhib = torch.zeros(self.ca1_size, device=self.device)

        # =====================================================================
        # EMERGENT ENCODING/RETRIEVAL from OLM dynamics
        # =====================================================================
        # Instead of hardcoded sinusoid, encoding/retrieval emerges from OLM activity:
        # - High OLM activity → strong dendritic inhibition → encoding phase
        # - Low OLM activity → weak dendritic inhibition → retrieval phase
        #
        # Compute modulation from OLM firing rate (inverted for retrieval)
        olm_firing_rate = ca1_olm_inhib.mean().item()  # [0, ~1]
        # Encoding high when OLM fires (suppresses retrieval pathway)
        # Add baseline encoding (0.3) so learning isn't completely blocked
        # when OLM cells are silent during early training. Real hippocampus has tonic
        # acetylcholine that provides baseline encoding drive even without septal input.
        encoding_mod = torch.clamp(torch.tensor(0.3 + olm_firing_rate * 2.0), 0.0, 1.0).item()
        # Retrieval high when OLM silent (allows EC→CA1 flow)
        retrieval_mod = 1.0 - encoding_mod

        # Split excitation and inhibition for ConductanceLIF
        # Excitatory: CA3 + EC pathways
        ca1_g_exc = F.relu(ca1_input)

        # Apply dendritic inhibition to excitatory input (models apical dendrite suppression)
        # OLM cells target apical dendrites where EC input arrives
        ca1_g_exc = ca1_g_exc * (1.0 - ca1_dendritic_inhib.mean() * 0.5)  # Moderate suppression

        # Inhibitory: perisomatic inhibition from PV cells + lateral CA1 inhibition
        ca1_g_inh = F.relu(ca1_perisomatic_inhib)
        # Run through CA1 neurons (ConductanceLIF with E/I separation)
        ca1_spikes, _ca1_membrane = self.ca1_neurons(ca1_g_exc, ca1_g_inh)

        # Apply sparsity (more lenient during retrieval to allow mismatch detection)
        # Use pre-spike membrane for WTA (not post-spike which resets to v_reset)
        sparsity_factor = (
            1.0 + 0.5 * retrieval_mod
        )  # Higher threshold during retrieval
        ca1_spikes = self._apply_wta_sparsity(
            ca1_spikes,
            cfg.ca1_sparsity * sparsity_factor,
            membrane=self.ca1_neurons.membrane_pre_spike,
        )

        # Update homeostatic gain for CA1
        # Update firing rate (exponential moving average)
        self.ca1_firing_rate = (
            1 - self._firing_rate_alpha
        ) * self.ca1_firing_rate + self._firing_rate_alpha * ca1_spikes.float()
        # Compute rate error
        rate_error = self._target_rate - self.ca1_firing_rate
        # Update gain (increase gain if firing too low, decrease if too high)
        self.ca1_gain.data = (self.ca1_gain + self._gain_lr * rate_error).clamp(min=0.001)
        # Adaptive threshold
        self.ca1_neurons.v_threshold.data = torch.clamp(
            self.ca1_neurons.v_threshold - self._threshold_lr * rate_error,
            min=self._threshold_min,
            max=self._threshold_max,
        )

        # ---------------------------------------------------------
        # HEBBIAN LEARNING: CA2→CA1 plasticity (during encoding)
        # ---------------------------------------------------------
        # CA2 provides temporal/social context to CA1
        # Moderate learning rate (between CA3→CA2 weak and EC→CA1 strong)
        ca2_activity_delayed = ca2_spikes_delayed.float()
        ca1_activity = ca1_spikes.float()

        if ca2_activity_delayed.sum() > 0 and ca1_activity.sum() > 0:
            base_lr = cfg.ca2_ca1_learning_rate * encoding_mod

            gamma_mod = self._gamma_amplitude_effective
            effective_lr = compute_learning_rate_modulation(base_lr, gamma_mod)

            dW = effective_lr * torch.outer(ca1_activity, ca2_activity_delayed)

            # Multi-timescale consolidation for CA2→CA1
            fast_decay = dt_ms / self.config.fast_trace_tau_ms
            self._ca2_ca1_fast = (1.0 - fast_decay) * self._ca2_ca1_fast + dW

            slow_decay = dt_ms / self.config.slow_trace_tau_ms
            consolidation = self.config.consolidation_rate * self._ca2_ca1_fast
            self._ca2_ca1_slow = (1.0 - slow_decay) * self._ca2_ca1_slow + consolidation

            combined_dW = self._ca2_ca1_fast + self.config.slow_trace_contribution * self._ca2_ca1_slow
            self.synaptic_weights["ca2_ca1"].data += combined_dW

            clamp_weights(self.synaptic_weights["ca2_ca1"].data, cfg.w_min, cfg.w_max)

        # =====================================================================
        # Update STDP Traces (for learning, not comparison)
        # =====================================================================

        # Apply continuous plasticity (learning happens as part of forward dynamics)
        self.dg_spikes = dg_spikes
        self.ca3_spikes = ca3_spikes
        self.ca2_spikes = ca2_spikes
        self.ca1_spikes = ca1_spikes
        self._apply_plasticity()

        # =====================================================================
        # DOPAMINE-GATED CONSOLIDATION
        # =====================================================================
        # Apply dopamine-gated consolidation to tagged synapses
        # High dopamine (reward) → strong consolidation of tagged synapses
        # This is the "capture" part of synaptic tagging and capture
        if self.synaptic_tagging is not None and self.neuromodulator_state.dopamine > 0.1:
            # Consolidate tagged synapses proportional to dopamine
            current_weights = self.synaptic_weights["ca3_ca3"]
            new_weights = self.synaptic_tagging.consolidate_tagged_synapses(
                weights=current_weights,
                dopamine=self.neuromodulator_state.dopamine,
                learning_rate=self.config.learning_rate * 0.5,  # Half of base LR
            )
            self.synaptic_weights["ca3_ca3"].data = new_weights

        # =====================================================================
        # SET PORT OUTPUTS
        # =====================================================================
        self.set_port_output("dg", dg_spikes)
        self.set_port_output("ca3", ca3_spikes)
        self.set_port_output("ca2", ca2_spikes)
        self.set_port_output("ca1", ca1_spikes)

    def _apply_plasticity(self) -> None:
        """
        Apply homeostatic plasticity to CA3 recurrent weights.

        CA3 recurrent learning (one-shot Hebbian) happens in forward().
        This method only applies homeostatic mechanisms (synaptic scaling,
        intrinsic plasticity) to maintain stable network dynamics.
        """
        # Apply homeostatic synaptic scaling to CA3 recurrent weights
        self.synaptic_weights["ca3_ca3"].data = self.homeostasis.normalize_weights(
            self.synaptic_weights["ca3_ca3"].data, dim=1
        )
        self.synaptic_weights["ca3_ca3"].data.fill_diagonal_(0.0)  # Maintain no self-connections

        # Apply intrinsic plasticity (threshold adaptation) using homeostasis helper
        if self.ca3_spikes is not None:
            # Initialize if needed
            if self._ca3_activity_history is None:
                self._ca3_activity_history = torch.zeros(self.ca3_size, device=self.device)
            if self._ca3_threshold_offset is None:
                self._ca3_threshold_offset = torch.zeros(self.ca3_size, device=self.device)

            # Update activity history (exponential moving average)
            self._ca3_activity_history.mul_(0.99).add_(self.ca3_spikes.float(), alpha=0.01)

            # Use homeostasis helper for excitability modulation
            # This computes threshold offset based on activity deviation from target
            excitability_mod = self.homeostasis.compute_excitability_modulation(
                self._ca3_activity_history,
                tau=100.0
            )
            # Convert excitability modulation (>1 = easier, <1 = harder) to threshold offset
            # Higher excitability → lower threshold (subtract positive offset)
            # Lower excitability → higher threshold (subtract negative offset)
            self._ca3_threshold_offset = (1.0 - excitability_mod).clamp(-0.5, 0.5)

    def _apply_wta_sparsity(
        self,
        spikes: torch.Tensor,
        target_sparsity: float,
        membrane: Optional[torch.Tensor] = None,
        conductance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply soft winner-take-all with membrane noise (biologically realistic).

        Instead of hard ranking by exact membrane potential, uses:
        1. Membrane noise (1-2mV typical biological fluctuation)
        2. Probabilistic selection via softmax over noisy potentials
        3. Target sparsity as soft constraint, not hard cutoff

        This makes the selection robust to tiny floating point differences
        (<1e-7) while maintaining biological realism. Real hippocampal neurons
        have ~2-5mV membrane fluctuations that dominate selection dynamics.

        **IMPORTANT**: Must use pre-spike membrane (before reset) for WTA selection.
        Neurons now store `membrane_pre_spike` which is captured before spike check.
        Post-spike membrane is reset to v_reset and is useless for selection.

        Args:
            spikes: Spike tensor [n_neurons] (1D)
            target_sparsity: Fraction of neurons to keep active
            membrane: Pre-spike membrane potentials [n_neurons] for selection
                Should be neuron.membrane_pre_spike (captured before spike reset)
            conductance: Optional conductance fallback [n_neurons] for selection
                Used if membrane unavailable (shouldn't happen with new API)
                Higher conductance = stronger drive = should be selected

        Returns:
            Sparse spike tensor [n_neurons] (1D bool)
        """
        n_neurons = spikes.shape[0]
        k = max(1, int(n_neurons * target_sparsity))

        # Create bool tensor (memory efficient)
        sparse_spikes = torch.zeros_like(spikes, dtype=torch.bool)

        # Single sample processing (no batch)
        active = spikes.nonzero(as_tuple=True)[0]

        if len(active) <= k:
            # All spikes pass through
            sparse_spikes = spikes if spikes.dtype == torch.bool else spikes.bool()
        elif membrane is not None:
            # Use pre-spike membrane potential (primary method)
            # Add biological membrane noise (1-2mV ~ 0.001-0.002 normalized units)
            active_v = membrane[active]
            noise = torch.randn_like(active_v) * 0.002  # 2mV std deviation
            noisy_v = active_v + noise

            # Soft WTA: probabilistic selection via softmax (temperature=10mV ~ 0.01)
            probs = torch.softmax(noisy_v / 0.01, dim=0)
            selected = torch.multinomial(probs, k, replacement=False)
            sparse_spikes[active[selected]] = True
        elif conductance is not None:
            # Fallback: Use conductance as proxy for drive
            # Higher conductance = stronger input = should be selected
            active_g = conductance[active]
            noise = torch.randn_like(active_g) * 0.002  # Biological noise
            noisy_g = active_g + noise

            # Softmax selection weighted by conductance
            probs = torch.softmax(noisy_g / 0.01, dim=0)
            selected = torch.multinomial(probs, k, replacement=False)
            sparse_spikes[active[selected]] = True
        else:
            # Need either membrane or conductance for selection
            raise RuntimeError(
                f"_apply_wta_sparsity called without pre-spike membrane or conductance. "
                f"WTA requires pre-spike membrane (neuron.membrane_pre_spike). "
                f"Got {len(active)} active neurons needing selection to {k}."
            )

        return sparse_spikes

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
        if hasattr(self, "dg_neurons") and self.dg_neurons is not None:
            self.dg_neurons.update_temporal_parameters(dt_ms)
        if hasattr(self, "ca3_neurons") and self.ca3_neurons is not None:
            self.ca3_neurons.update_temporal_parameters(dt_ms)
        if hasattr(self, "ca2_neurons") and self.ca2_neurons is not None:
            self.ca2_neurons.update_temporal_parameters(dt_ms)
        if hasattr(self, "ca1_neurons") and self.ca1_neurons is not None:
            self.ca1_neurons.update_temporal_parameters(dt_ms)

        # Update STP components
        if hasattr(self, "stp_mossy") and self.stp_mossy is not None:
            self.stp_mossy.update_temporal_parameters(dt_ms)
        if hasattr(self, "stp_schaffer") and self.stp_schaffer is not None:
            self.stp_schaffer.update_temporal_parameters(dt_ms)
        if hasattr(self, "stp_ec_ca1") and self.stp_ec_ca1 is not None:
            self.stp_ec_ca1.update_temporal_parameters(dt_ms)
        if hasattr(self, "stp_ca3_ca2") and self.stp_ca3_ca2 is not None:
            self.stp_ca3_ca2.update_temporal_parameters(dt_ms)
        if hasattr(self, "stp_ca2_ca1") and self.stp_ca2_ca1 is not None:
            self.stp_ca2_ca1.update_temporal_parameters(dt_ms)
        if hasattr(self, "stp_ec_ca2") and self.stp_ec_ca2 is not None:
            self.stp_ec_ca2.update_temporal_parameters(dt_ms)

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for this region."""
        cfg = self.config

        # Compute plasticity metrics for CA3 recurrent (most important for episodic memory)
        plasticity = compute_plasticity_metrics(
            weights=self.synaptic_weights["ca3_ca3"].data,
            learning_rate=self.config.learning_rate,
        )
        # Add pathway-specific weight statistics
        plasticity["ec_dg_mean"] = float(self.synaptic_weights["ec_dg"].data.mean().item())
        plasticity["dg_ca3_mean"] = float(self.synaptic_weights["dg_ca3"].data.mean().item())
        plasticity["ca3_ca1_mean"] = float(self.synaptic_weights["ca3_ca1"].data.mean().item())
        plasticity["ec_ca1_mean"] = float(self.synaptic_weights["ec_ca1"].data.mean().item())

        # CA3 bistable dynamics
        ca3_persistent = {}
        ca3_persistent.update(DiagnosticsUtils.trace_diagnostics(self.ca3_persistent, ""))
        ca3_persistent["nonzero_count"] = (self.ca3_persistent > 0.1).sum().item()

        # CA1 NMDA comparison mechanism
        nmda_diagnostics = {"threshold": cfg.nmda_threshold}
        if self.nmda_trace is not None:
            nmda_diagnostics.update(DiagnosticsUtils.trace_diagnostics(self.nmda_trace, ""))
            nmda_diagnostics["trace_std"] = self.nmda_trace.std().item()
            nmda_diagnostics["above_threshold_count"] = (
                (self.nmda_trace > cfg.nmda_threshold).sum().item()
            )

            # Compute Mg block removal
            mg_removal = torch.sigmoid((self.nmda_trace - cfg.nmda_threshold) * cfg.nmda_steepness)
            nmda_diagnostics["mg_block_removal_mean"] = mg_removal.mean().item()
            nmda_diagnostics["mg_block_removal_max"] = mg_removal.max().item()
            nmda_diagnostics["gated_neurons"] = (mg_removal > 0.5).sum().item()

        # Synaptic tagging diagnostics
        synaptic_tagging = {}
        if self.synaptic_tagging is not None:
            synaptic_tagging = self.synaptic_tagging.get_diagnostics()

        return {
            "plasticity": plasticity,
            "ca3_persistent": ca3_persistent,
            "nmda": nmda_diagnostics,
            "synaptic_tagging": synaptic_tagging,
        }
