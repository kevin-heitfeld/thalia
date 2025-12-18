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
   - Naturally clears residual activity (no explicit resets needed!)
   - Fast-spiking interneuron-like dynamics

3. **CONTINUOUS DYNAMICS**:
   - Everything flows naturally - no artificial resets
   - Membrane potentials decay via LIF dynamics
   - Theta phase advances continuously
   - Smooth transitions between encoding and retrieval phases

**All processing is spike-based** (no rate accumulation, ADR-004).

FILE ORGANIZATION (2182 lines)
===============================
Lines 1-150:     Module docstring, imports, class registration
Lines 151-400:   __init__() and weight initialization
Lines 401-650:   DG forward pass (pattern separation)
Lines 651-900:   CA3 forward pass (pattern completion + recurrence)
Lines 901-1100:  CA1 forward pass (comparison and output)
Lines 1101-1350: Learning methods (STDP, acetylcholine modulation)
Lines 1351-1600: Episodic memory (store, retrieve, replay)
Lines 1601-1850: Growth and neurogenesis
Lines 1851-2050: Diagnostics and health checks
Lines 2051-2182: Utility methods and state management

NAVIGATION TIP: Use VSCode's "Go to Symbol" (Ctrl+Shift+O) or collapse
regions (Ctrl+K Ctrl+0) to navigate this file efficiently.

WHY THIS FILE IS LARGE
======================
The DG→CA3→CA1 circuit is a single biological computation that must execute
within one theta cycle (~100-150ms). Splitting would:
1. Require passing ~20 intermediate tensors between files
2. Break the narrative flow of the biological computation
3. Obscure the theta-phase-dependent coordination
4. Duplicate device/config management across files

Components ARE extracted where orthogonal:
- HippocampusMemoryComponent: Episodic storage/retrieval (shared concern)
- ReplayEngine: Sequence replay (shared with sleep system)
- FeedforwardInhibition: Stimulus-triggered inhibition (used by multiple regions)

See: docs/decisions/adr-011-large-file-justification.md

References:
- Marr (1971): Simple memory model
- Treves & Rolls (1994): Pattern separation in DG
- Hasselmo et al. (2002): Theta rhythm and encoding/retrieval
- Colgin (2013): Theta-gamma coupling in hippocampus
"""

import math
from dataclasses import replace
from typing import Optional, Dict, Any, List, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.components.neurons import create_pyramidal_neurons
from thalia.managers.base_manager import ManagerContext
from thalia.managers.component_registry import register_region
from thalia.core.errors import CheckpointError, ComponentError
from thalia.components.neurons.neuron_constants import (
    NE_GAIN_RANGE,
)
from thalia.components.synapses.stp import ShortTermPlasticity
from thalia.components.synapses.stp_presets import get_stp_config
from thalia.utils.core_utils import clamp_weights, cosine_similarity_safe
from thalia.components.synapses.traces import update_trace
from thalia.components.synapses.weight_init import WeightInitializer
from thalia.learning.homeostasis.synaptic_homeostasis import UnifiedHomeostasis, UnifiedHomeostasisConfig
from thalia.regions.base import NeuralComponent
from thalia.regions.feedforward_inhibition import FeedforwardInhibition
from .replay_engine import ReplayEngine, ReplayConfig, ReplayMode
from .config import Episode, HippocampusConfig, HippocampusState
from .memory_component import HippocampusMemoryComponent
from .checkpoint_manager import HippocampusCheckpointManager


@register_region(
    "hippocampus",
    aliases=["trisynaptic"],
    description="DG→CA3→CA1 trisynaptic circuit with theta-modulated encoding/retrieval and episodic memory",
    version="2.0",
    author="Thalia Project",
    config_class=HippocampusConfig,
)
class TrisynapticHippocampus(NeuralComponent):
    """
    Biologically-accurate hippocampus with DG→CA3→CA1 trisynaptic circuit.

    Architecture:
    ```
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
                            │     CA1      │  Output layer with comparison
                            │              │  COINCIDENCE DETECTION: match vs mismatch
                            └──────────────┘
                                   │
                                   ▼
                            Output (to cortex/striatum)
    ```

    Two pathways to CA3:
    - EC→DG→CA3: Pattern-separated (sparse), strong during encoding
    - EC→CA3 direct: Preserves similarity (less sparse), provides retrieval cues

    Key biological features:
    1. THETA MODULATION: 6-10 Hz oscillations separate encoding from retrieval
    2. FEEDFORWARD INHIBITION: Stimulus onset triggers transient inhibition
    3. CONTINUOUS DYNAMICS: No artificial resets - everything flows naturally

    All computations are spike-based. No rate accumulation!

    Mixins Provide:
    ---------------
    From DiagnosticsMixin:
        - check_health() → HealthMetrics
        - get_firing_rate(spikes) → float
        - check_weight_health(weights, name) → WeightHealth
        - detect_runaway_excitation(spikes) → bool
        - detect_silence(spikes) → bool

    From NeuralComponent (abstract base):
        - forward(input, **kwargs) → Tensor [must implement]
        - reset_state() → None
        - get_diagnostics() → Dict
        - set_dopamine(level) → None
        - Neuromodulator control methods

    See Also:
        docs/patterns/mixins.md for detailed mixin patterns
    """

    def __init__(self, config: HippocampusConfig):
        """Initialize trisynaptic hippocampus."""
        self.tri_config = config

        # Debug flag for learning investigation (set externally)
        self._debug_hippo = False

        # Compute layer sizes
        self.dg_size = int(config.n_input * config.dg_expansion)
        self.ca3_size = int(self.dg_size * config.ca3_size_ratio)
        self.ca1_size = config.n_output  # CA1 matches output

        # Call parent init
        super().__init__(config)

        # Initialize oscillator-related state
        self._gamma_amplitude_effective: float = 1.0

        # Override weights with trisynaptic circuit weights
        self._init_circuit_weights()

        # Create LIF neurons for each layer using factory functions
        # DG and CA1: Standard pyramidal neurons
        self.dg_neurons = create_pyramidal_neurons(self.dg_size, self.device)
        # CA3 gets spike-frequency adaptation to prevent frozen attractors
        self.ca3_neurons = create_pyramidal_neurons(
            self.ca3_size,
            self.device,
            adapt_increment=config.adapt_increment,  # SFA enabled!
            tau_adapt=config.adapt_tau,
        )
        self.ca1_neurons = create_pyramidal_neurons(self.ca1_size, self.device)

        # Feedforward inhibition module
        self.feedforward_inhibition = FeedforwardInhibition(
            threshold=config.ffi_threshold,
            max_inhibition=config.ffi_strength * 10.0,  # Scale to appropriate range
            decay_rate=1.0 - (1.0 / config.ffi_tau),  # Convert tau to rate
        )

        # =====================================================================
        # SHORT-TERM PLASTICITY (STP)
        # =====================================================================
        # Initialize STP modules for each pathway if enabled
        if config.stp_enabled:
            device = torch.device(config.device)

            # Mossy Fibers (DG→CA3): Strong facilitation
            # Biological U~0.03 means first spikes barely transmit, but repeated
            # DG activity causes massive facilitation (up to 10x enhancement!)
            # Use biologically-validated preset from literature
            self.stp_mossy = ShortTermPlasticity(
                n_pre=self.dg_size,
                n_post=self.ca3_size,
                config=get_stp_config("mossy_fiber", dt=1.0),
                per_synapse=True,
            )
            self.stp_mossy.to(device)

            # Schaffer Collaterals (CA3→CA1): Depression
            # High-frequency CA3 activity depresses CA1 input - allows novelty
            # detection (novel patterns don't suffer from adaptation)
            self.stp_schaffer = ShortTermPlasticity(
                n_pre=self.ca3_size,
                n_post=self.ca1_size,
                config=get_stp_config("schaffer_collateral", dt=1.0),
                per_synapse=True,
            )
            self.stp_schaffer.to(device)

            # EC→CA1 Direct (Temporoammonic): Depression
            # Initial input is strongest - matched comparison happens on first
            # presentation before adaptation kicks in
            # Use ec_l3_input_size if set (for separate raw sensory input),
            # otherwise fall back to n_input
            ec_ca1_input_size = config.ec_l3_input_size if config.ec_l3_input_size > 0 else config.n_input
            self.stp_ec_ca1 = ShortTermPlasticity(
                n_pre=ec_ca1_input_size,
                n_post=self.ca1_size,
                config=get_stp_config("ec_ca1", dt=1.0),
                per_synapse=True,
            )
            self.stp_ec_ca1.to(device)

            # CA3→CA3 Recurrent: FAST DEPRESSION - prevents frozen attractors
            # This is CRITICAL: without STD on recurrent connections, the same
            # neurons fire every timestep because they reinforce themselves.
            # With STD, frequently-used synapses get temporarily weaker,
            # allowing different patterns to emerge for different inputs.
            self.stp_ca3_recurrent = ShortTermPlasticity(
                n_pre=self.ca3_size,
                n_post=self.ca3_size,
                config=get_stp_config("ca3_recurrent", dt=1.0),
                per_synapse=True,
            )
            self.stp_ca3_recurrent.to(device)
        else:
            self.stp_mossy = None
            self.stp_schaffer = None
            self.stp_ec_ca1 = None
            self.stp_ca3_recurrent = None

        # =====================================================================
        # INTER-LAYER AXONAL DELAYS
        # =====================================================================
        # Create delay buffers for biological signal propagation within circuit
        # DG→CA3 delay: Mossy fiber transmission (~3ms biologically)
        # CA3→CA1 delay: Schaffer collateral transmission (~3ms biologically)
        # Uses circular buffer mechanism from AxonalDelaysMixin
        self._dg_ca3_delay_steps = int(config.dg_to_ca3_delay_ms / config.dt_ms)
        self._ca3_ca1_delay_steps = int(config.ca3_to_ca1_delay_ms / config.dt_ms)

        # Initialize delay buffers (lazily initialized on first use)
        self._dg_ca3_delay_buffer: Optional[torch.Tensor] = None
        self._dg_ca3_delay_ptr: int = 0
        self._ca3_ca1_delay_buffer: Optional[torch.Tensor] = None
        self._ca3_ca1_delay_ptr: int = 0

        # Episode buffer for sleep consolidation
        self.episode_buffer: List[Episode] = []

        # =====================================================================
        # MANAGERS: Extract god object logic into focused components
        # =====================================================================
        # Create manager context for plasticity and episode management
        manager_context = ManagerContext(
            device=torch.device(config.device),
            n_output=self.ca3_size,
            dt_ms=config.dt_ms,
            metadata={"ca3_size": self.ca3_size},
        )

        # Homeostasis for CA3 recurrent synaptic scaling
        homeostasis_config = UnifiedHomeostasisConfig(
            weight_budget=config.weight_budget * self.ca3_size,
            w_min=config.w_min,
            w_max=config.w_max,
            soft_normalization=config.soft_normalization,
            normalization_rate=config.normalization_rate,
            device=torch.device(config.device),
        )
        self.homeostasis = UnifiedHomeostasis(homeostasis_config)

        # Intrinsic plasticity state (threshold adaptation)
        self._ca3_activity_history: Optional[torch.Tensor] = None
        self._ca3_threshold_offset: Optional[torch.Tensor] = None

        # Episode manager: handles episodic memory storage and retrieval
        self.memory = HippocampusMemoryComponent(
            config=config,
            context=manager_context,
        )

        # Feedback inhibition state - tracks recent CA3 activity
        self._ca3_activity_trace: Optional[torch.Tensor] = None

        # =====================================================================
        # THETA-GAMMA COUPLING (from centralized oscillators)
        # =====================================================================
        # Store oscillator values received from brain broadcast
        # These replace the local GammaOscillator instance
        if config.theta_gamma_enabled:
            # Storage for broadcast values from brain's OscillatorManager
            self._theta_phase: float = 0.0
            self._gamma_phase: float = 0.0
            self._theta_slot: int = 0
            self._coupled_amplitudes: Dict[str, float] = {}

            # Replay engine for sequence replay (lazy import to avoid circular dependency)
            # NOTE: ReplayEngine receives timing via replay() parameters (gamma_phase, theta_slot)
            # from hippocampus, which gets them from brain's centralized OscillatorManager
            replay_config = ReplayConfig(
                compression_factor=5.0,
                n_slots=config.gamma_n_slots,
                slot_duration_ms=18.0,  # ~18ms per slot for 7 slots in 125ms theta
                mode=ReplayMode.SEQUENCE,
                apply_gating=True,
                pattern_completion=True,
            )
            self.replay_engine = ReplayEngine(replay_config)

            # Assign CA3 neurons to gamma slots for phase coding
            # Neurons in different slots fire at different gamma phases
            self._ca3_slot_assignment = torch.arange(self.ca3_size) % config.gamma_n_slots
        else:
            self._theta_phase = 0.0
            self._gamma_phase = 0.0
            self._theta_slot = 0
            self._coupled_amplitudes = {}
            self.replay_engine = None
            self._ca3_slot_assignment = None

        # Track current sequence position for encoding (auto-advances)
        self._sequence_position: int = 0

        # Theta-driven reset: when True, reset happens at next theta trough
        # This replaces hard resets with biologically-realistic theta-aligned resets
        self._pending_theta_reset: bool = False

        # =====================================================================
        # HINDSIGHT EXPERIENCE REPLAY (HER)
        # =====================================================================
        # Goal relabeling for multi-goal learning
        if config.use_her:
            from thalia.regions.hippocampus.hindsight_relabeling import (
                HippocampalHERIntegration,
                HERConfig,
                HERStrategy,
            )
            her_config = HERConfig(
                strategy=HERStrategy[config.her_strategy.upper()],
                k_hindsight=config.her_k_hindsight,
                replay_ratio=config.her_replay_ratio,
                goal_dim=config.n_output,  # Use output size as goal dimension
                goal_tolerance=config.her_goal_tolerance,
                buffer_size=config.her_buffer_size,
                device=config.device,
            )
            self.her_integration = HippocampalHERIntegration(her_config)
        else:
            self.her_integration = None

        # State
        self.state = HippocampusState()

        # Checkpoint manager for neuromorphic format support
        self.checkpoint_manager = HippocampusCheckpointManager(self)

    def _initialize_weights(self) -> torch.Tensor:
        """Placeholder - real weights created in _init_circuit_weights."""
        return nn.Parameter(torch.zeros(self.tri_config.n_output, self.tri_config.n_input))

    def _create_neurons(self):
        """Placeholder - neurons created in __init__."""
        return None

    def _init_circuit_weights(self) -> None:
        """Initialize all circuit weights."""
        device = torch.device(self.tri_config.device)

        # EC → DG: Random sparse projections (each DG cell sees random subset)
        # This creates orthogonal codes for pattern separation
        # Uses row normalization for reliable activity propagation
        self.w_ec_dg = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.dg_size,
                n_input=self.tri_config.n_input,
                sparsity=0.3,  # 30% connectivity
                weight_scale=0.5,  # Strong weights for propagation
                normalize_rows=True,  # Normalize for reliable propagation
                device=device
            )
        )

        # DG → CA3: Random but less sparse (mossy fibers)
        # Uses row normalization for reliable activity propagation
        self.w_dg_ca3 = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.ca3_size,
                n_input=self.dg_size,
                sparsity=0.5,
                weight_scale=0.5,  # Strong weights for propagation
                normalize_rows=True,  # Normalize for reliable propagation
                device=device
            )
        )

        # EC → CA3: Direct perforant path (layer II)
        # Less sparse than DG→CA3, preserves input similarity for retrieval cues
        # This is CRITICAL for cue-based retrieval: partial cues seed the CA3 attractor
        # Biological: ~40% connectivity, weaker than DG→CA3 but more direct
        self.w_ec_ca3 = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.ca3_size,
                n_input=self.tri_config.n_input,
                sparsity=0.4,  # Less sparse than DG path
                weight_scale=0.3,  # Weaker than DG→CA3
                normalize_rows=True,
                device=device
            )
        )

        # CA3 → CA3: Recurrent connections (autoassociative memory)
        # Initialize with small random values - these will be LEARNED via Hebbian
        # For an attractor network, we need weights strong enough that when ~10% of
        # neurons fire (N=32), the recurrent input exceeds threshold:
        #   N * avg_weight * recurrent_strength > threshold
        #   32 * 0.15 * 0.4 = 1.92 > 1.0 ✓
        # We use slightly larger initial weights to bootstrap the network
        self.w_ca3_ca3 = nn.Parameter(
            WeightInitializer.gaussian(
                n_output=self.ca3_size,
                n_input=self.ca3_size,
                mean=0.05,
                std=0.15,
                device=device
            )
        )
        # No self-connections
        with torch.no_grad():
            self.w_ca3_ca3.data.fill_diagonal_(0.0)
            # Clamp to positive (excitatory recurrent connections)
            self.w_ca3_ca3.data.clamp_(min=0.0)

        # CA3 → CA1: Feedforward (retrieved memory) - SPARSE!
        # CRITICAL: This MUST be sparse so that different CA3 patterns activate
        # different CA1 subpopulations. Otherwise, ALL CA1 neurons get high input
        # from ANY CA3 pattern, and the mg_block is always 1.0.
        #
        # With dense connectivity:
        #   Every CA1 gets ~24 CA3 spikes × 0.15 weight = 3.6 input → mg_block = 1.0
        #
        # With sparse connectivity (15%):
        #   Each CA1 gets ~24 × 0.15 × 0.15 = 0.54 input on average
        #   Only CA1 neurons connected to the active CA3 subset get high input
        #   This creates PATTERN-SPECIFIC CA1 activation → NMDA gating works!
        self.w_ca3_ca1 = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.ca1_size,
                n_input=self.ca3_size,
                sparsity=0.15,  # Each CA1 sees only 15% of CA3
                weight_scale=0.3,  # Strong individual weights
                normalize_rows=False,  # NO normalization - pattern-specific!
                device=device
            )
        )

        # EC → CA1: Direct pathway - SPARSE and PLASTIC!
        # CRITICAL INSIGHT: This pathway must LEARN to align with the indirect
        # EC→DG→CA3→CA1 pathway during memory encoding.
        #
        # We use SPARSE initial connectivity so that:
        # 1. Different EC patterns naturally activate different CA1 subsets
        # 2. Hebbian learning strengthens the connections that are used
        # 3. For a mismatched pattern, EC activates DIFFERENT CA1 neurons
        #    than CA3, so there's no coincidence
        #
        # During SAMPLE phase:
        #   - CA3 drives CA1 (retrieved/encoded pattern)
        #   - CA1 fires for certain neurons
        #   - EC→CA1 weights strengthen for active EC inputs → active CA1 neurons
        #
        # During TEST phase:
        #   - MATCH: Same EC input → same EC→CA1 activation → coincidence with CA3
        #   - MISMATCH: Different EC input → different EC→CA1 activation → no coincidence
        self.w_ec_ca1 = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.ca1_size,
                n_input=self.tri_config.n_input,
                sparsity=0.20,  # Each CA1 sees only 20% of EC
                weight_scale=0.3,  # Strong individual weights
                normalize_rows=False,  # NO normalization - pattern-specific!
                device=device
            )
        )

        # EC Layer III → CA1: Separate pathway for raw sensory input
        # In biology, EC layer III pyramidal cells project directly to CA1
        # (temporoammonic path), carrying raw sensory information that is
        # compared against the retrieved memory from CA3.
        self._ec_l3_input_size = self.tri_config.ec_l3_input_size
        if self._ec_l3_input_size > 0:
            self.w_ec_l3_ca1 = nn.Parameter(
                WeightInitializer.sparse_random(
                    n_output=self.ca1_size,
                    n_input=self._ec_l3_input_size,
                    sparsity=0.20,
                    weight_scale=0.3,
                    device=device,
                )
            )
        else:
            self.w_ec_l3_ca1 = None  # Fall back to w_ec_ca1

        # CA1 lateral inhibition for competition
        self.w_ca1_inhib = nn.Parameter(
            torch.ones(self.ca1_size, self.ca1_size, device=device) * 0.5
        )
        with torch.no_grad():
            self.w_ca1_inhib.data.fill_diagonal_(0.0)

        # Store main weights reference for compatibility
        self.weights = self.w_ca3_ca1

    def reset_state(self) -> None:
        """Reset state for new episode.

        Note: Consider using new_trial() instead, which aligns theta and
        clears input history without fully resetting membrane potentials.
        Full reset is mainly needed between completely unrelated episodes.
        """
        super().reset_state()
        self._init_state()

        # Reset intrinsic plasticity state
        self._ca3_activity_history = None
        self._ca3_threshold_offset = None

    def new_trial(self) -> None:
        """Prepare for a new sequence/episode.

        With continuous learning, most state transitions happen via natural
        dynamics (decay, FFI). This method schedules a reset to occur at the
        next theta trough (encoding phase), which is biologically realistic.

        The actual reset is applied in forward() when transitioning to ENCODE
        phase, mimicking how theta rhythm naturally segments sequences.

        Call this when starting a completely new sequence where the previous
        stored pattern is irrelevant. For continuous text processing within
        a sequence, do NOT call this.
        """
        # Schedule reset for next theta trough (ENCODE phase)
        # This replaces hard resets with theta-aligned resets
        self._pending_theta_reset = True

        # Reset gamma position for new sequence
        self._sequence_position = 0

    def _init_state(self) -> None:
        """Initialize all layer states (internal method)."""
        device = self.device

        self.dg_neurons.reset_state()
        self.ca3_neurons.reset_state()
        self.ca1_neurons.reset_state()

        # Reset STP state for all pathways
        if self.stp_mossy is not None:
            self.stp_mossy.reset_state()
        if self.stp_schaffer is not None:
            self.stp_schaffer.reset_state()
        if self.stp_ec_ca1 is not None:
            self.stp_ec_ca1.reset_state()
        if self.stp_ca3_recurrent is not None:
            self.stp_ca3_recurrent.reset_state()

        # Reset feedback inhibition trace
        self._ca3_activity_trace = torch.zeros(1, device=device)

        # 1D architecture - no batch dimension
        self.state = HippocampusState(
            dg_spikes=torch.zeros(self.dg_size, device=device),
            ca3_spikes=torch.zeros(self.ca3_size, device=device),
            ca1_spikes=torch.zeros(self.ca1_size, device=device),
            ca3_membrane=torch.zeros(self.ca3_size, device=device),
            ca3_persistent=torch.zeros(self.ca3_size, device=device),
            sample_trace=None,  # Set during sample encoding
            dg_trace=torch.zeros(self.dg_size, device=device),
            ca3_trace=torch.zeros(self.ca3_size, device=device),
            nmda_trace=torch.zeros(self.ca1_size, device=device),
            stored_dg_pattern=None,  # Set during sample phase
            ffi_strength=0.0,
        )

    # region Growth and Neurogenesis

    def grow_output(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Grow output dimension by expanding all hippocampal layers proportionally.

        This expands DG, CA3, and CA1 while maintaining the circuit ratios:
        - DG expands by (dg_expansion * n_new)
        - CA3 expands by (ca3_size_ratio * DG_growth)
        - CA1 expands by n_new

        All inter-layer weights are expanded to accommodate new neurons.

        Args:
            n_new: Number of neurons to add to CA1 (output layer)
            initialization: Weight initialization strategy
            sparsity: Sparsity for new connections
        """
        # Calculate proportional growth for all layers
        old_ca1_size = self.ca1_size
        new_ca1_size = old_ca1_size + n_new

        # Maintain circuit ratios
        dg_growth = int(n_new * self.tri_config.dg_expansion / self.tri_config.ca1_size_ratio)
        old_dg_size = self.dg_size
        new_dg_size = old_dg_size + dg_growth

        ca3_growth = int(dg_growth * self.tri_config.ca3_size_ratio / self.tri_config.dg_expansion)
        old_ca3_size = self.ca3_size
        new_ca3_size = old_ca3_size + ca3_growth

        # Helper to create new weights
        def new_weights_for(n_out: int, n_in: int) -> torch.Tensor:
            if initialization == 'xavier':
                return WeightInitializer.xavier(n_out, n_in, device=self.device)
            elif initialization == 'sparse_random':
                return WeightInitializer.sparse_random(n_out, n_in, sparsity, device=self.device)
            else:
                return WeightInitializer.uniform(n_out, n_in, device=self.device)

        # 1. Expand input→DG weights [dg, input]
        # Add rows for new DG neurons
        new_input_dg = new_weights_for(dg_growth, self.tri_config.n_input)
        self.w_ec_dg = nn.Parameter(
            torch.cat([self.w_ec_dg.data, new_input_dg], dim=0)
        )

        # 2. Expand DG→CA3 weights [ca3, dg]
        # Add rows for new CA3 neurons, columns for new DG neurons
        # First expand rows (new CA3 neurons receiving from all DG)
        new_ca3_rows = new_weights_for(ca3_growth, old_dg_size)
        expanded_rows = torch.cat([self.w_dg_ca3.data, new_ca3_rows], dim=0)
        # Then expand columns (all CA3 receiving from new DG)
        new_dg_cols = new_weights_for(new_ca3_size, dg_growth)
        self.w_dg_ca3 = nn.Parameter(
            torch.cat([expanded_rows, new_dg_cols], dim=1)
        )

        # 3. Expand EC→CA3 direct perforant path [ca3, n_input]
        # Only expand rows (CA3), input size is fixed
        new_ec_ca3_rows = new_weights_for(ca3_growth, self.tri_config.n_input)
        self.w_ec_ca3 = nn.Parameter(
            torch.cat([self.w_ec_ca3.data, new_ec_ca3_rows], dim=0)
        )

        # 4. Expand CA3→CA3 recurrent weights [ca3, ca3]
        # Add rows and columns for new CA3 neurons
        new_ca3_recurrent_rows = new_weights_for(ca3_growth, old_ca3_size)
        expanded_recurrent_rows = torch.cat([self.w_ca3_ca3.data, new_ca3_recurrent_rows], dim=0)
        new_ca3_recurrent_cols = new_weights_for(new_ca3_size, ca3_growth)
        self.w_ca3_ca3 = nn.Parameter(
            torch.cat([expanded_recurrent_rows, new_ca3_recurrent_cols], dim=1)
        )

        # 5. Expand CA3→CA1 weights [ca1, ca3]
        # Add rows for new CA1 neurons, columns for new CA3 neurons
        new_ca1_rows = new_weights_for(n_new, old_ca3_size)
        expanded_ca1_rows = torch.cat([self.w_ca3_ca1.data, new_ca1_rows], dim=0)
        new_ca3_cols_to_ca1 = new_weights_for(new_ca1_size, ca3_growth)
        self.w_ca3_ca1 = nn.Parameter(
            torch.cat([expanded_ca1_rows, new_ca3_cols_to_ca1], dim=1)
        )

        # Update main weights reference (for base class compatibility)
        self.weights = self.w_ca3_ca1

        # 5. Expand neurons for all layers using factory functions
        self.dg_size = new_dg_size
        self.dg_neurons = create_pyramidal_neurons(self.dg_size, self.device)

        self.ca3_size = new_ca3_size
        self.ca3_neurons = create_pyramidal_neurons(
            self.ca3_size,
            self.device,
            adapt_increment=self.tri_config.adapt_increment,
            tau_adapt=self.tri_config.adapt_tau,
        )

        self.ca1_size = new_ca1_size
        self.ca1_neurons = create_pyramidal_neurons(self.ca1_size, self.device)

        # 6. Update config
        self.config = replace(self.config, n_output=new_ca1_size)

    def grow_input(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Grow hippocampus input dimension when upstream region grows.

        When upstream regions (e.g., cortex) add neurons, this method expands
        the hippocampus's input weights to accommodate the larger input.

        Args:
            n_new: Number of input neurons to add
            initialization: Weight init strategy
            sparsity: Connection sparsity for new inputs

        Example:
            >>> cortex.grow_output(20)
            >>> cortex_to_hippocampus.grow_output(20)
            >>> hippocampus.grow_input(20)  # Expand EC input weights
        """
        old_n_input = self.config.n_input
        new_n_input = old_n_input + n_new

        # Helper for weight initialization
        def new_weights_for(n_out: int, n_in: int) -> torch.Tensor:
            if initialization == 'xavier':
                return WeightInitializer.xavier(n_out, n_in, device=self.device)
            elif initialization == 'sparse_random':
                return WeightInitializer.sparse_random(n_out, n_in, sparsity, device=self.device)
            else:
                return WeightInitializer.uniform(n_out, n_in, device=self.device)

        # Expand EC→DG weights [dg, input] → [dg, input+n_new]
        new_input_cols = new_weights_for(self.dg_size, n_new)
        self.w_ec_dg = nn.Parameter(
            torch.cat([self.w_ec_dg.data, new_input_cols], dim=1)
        )

        # Expand EC→CA3 direct perforant path [ca3, input] → [ca3, input+n_new]
        new_perforant_cols = new_weights_for(self.ca3_size, n_new)
        self.w_ec_ca3 = nn.Parameter(
            torch.cat([self.w_ec_ca3.data, new_perforant_cols], dim=1)
        )

        # Update config
        self.config = replace(self.config, n_input=new_n_input)

    # endregion

    # region Forward Pass (DG→CA3→CA1)

    def forward(
        self,
        input_spikes: torch.Tensor,
        ec_direct_input: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Process input spikes through DG→CA3→CA1 circuit.

        Args:
            input_spikes: Input spike pattern [n_input] (1D, no batch)
            ec_direct_input: Optional separate input for EC→CA1 direct pathway [n_input] (1D)
                            If None, uses input_spikes (original behavior).
                            When provided, this models EC layer III input which
                            carries raw sensory information to CA1 for comparison.

        Returns:
            CA1 output spikes [n_output] (1D)

        Theta Modulation:
            - Encoding (theta trough): DG→CA3 strong, CA3 recurrence weak
            - Retrieval (theta peak): DG→CA3 suppressed, CA3 recurrence strong
            - Continuous modulation computed from self._theta_phase (set by Brain)

        Features:
            - Feedforward inhibition: Stimulus changes trigger transient inhibition
            - EC layer III: Optional separate input for direct EC→CA1 (biologically
              accurate - EC L3 carries raw sensory info, EC L2 goes through DG)
        """
        # Ensure 1D input (single sample, no batch)
        input_spikes = input_spikes.squeeze()
        assert input_spikes.dim() == 1, (
            f"TrisynapticHippocampus.forward: input must be 1D [n_input], "
            f"got shape {input_spikes.shape}"
        )

        # Convert bool→float if needed (neurons return bool, we need float for matmul)
        input_spikes_float = input_spikes.float() if input_spikes.dtype == torch.bool else input_spikes

        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        assert input_spikes.shape[0] == self.tri_config.n_input, (
            f"TrisynapticHippocampus.forward: input_spikes has shape {input_spikes.shape} "
            f"but n_input={self.tri_config.n_input}. Check that cortex output matches hippocampus input."
        )
        if ec_direct_input is not None:
            ec_direct_input = ec_direct_input.squeeze()
            assert ec_direct_input.dim() == 1, (
                f"TrisynapticHippocampus.forward: ec_direct_input must be 1D, "
                f"got shape {ec_direct_input.shape}"
            )
            expected_ec_size = self._ec_l3_input_size if self._ec_l3_input_size > 0 else self.tri_config.n_input
            assert ec_direct_input.shape[0] == expected_ec_size, (
                f"TrisynapticHippocampus.forward: ec_direct_input has shape {ec_direct_input.shape} "
                f"but expected size={expected_ec_size} (ec_l3_input_size={self._ec_l3_input_size}). "
                f"Check that sensory input matches EC L3 pathway configuration."
            )

        # Ensure state is initialized
        if self.state.dg_spikes is None:
            self._init_state()

        # =====================================================================
        # COMPUTE THETA MODULATION (from oscillator phase set by Brain)
        # =====================================================================
        # encoding_mod: high at theta trough (0°), low at peak (180°)
        # retrieval_mod: low at theta trough, high at theta peak
        encoding_mod = 0.5 * (1.0 + math.cos(self._theta_phase))
        retrieval_mod = 1.0 - encoding_mod

        # =====================================================================
        # THETA-PHASE RESET (prevents frozen attractors + new_trial reset)
        # =====================================================================
        # Detect theta trough transition (encoding_mod > 0.8) for reset logic
        # This is biologically realistic: theta rhythm naturally segments
        # sequences, and new_trial() just schedules reset for next theta trough.
        at_theta_trough = encoding_mod > 0.8

        if at_theta_trough:
            # Check if new_trial() requested a full reset
            if self._pending_theta_reset:
                # Full reset at theta trough - clear stored patterns
                self.state.stored_dg_pattern = None
                self.state.sample_trace = None
                # Stronger persistent reset for new sequences
                if self.state.ca3_persistent is not None:
                    self.state.ca3_persistent = self.state.ca3_persistent * 0.2
                self._pending_theta_reset = False
            elif (self.tri_config.theta_reset_persistent and
                  self.state.ca3_persistent is not None):
                # Normal theta trough: partial decay of persistent activity
                reset_fraction = self.tri_config.theta_reset_fraction
                self.state.ca3_persistent = self.state.ca3_persistent * (1.0 - reset_fraction)

        # =====================================================================
        # FEEDFORWARD INHIBITION
        # =====================================================================
        # Compute feedforward inhibition based on input change
        ffi = self.feedforward_inhibition.compute(input_spikes, return_tensor=False)
        raw_ffi = ffi.item() if hasattr(ffi, 'item') else float(ffi)
        # Normalize to [0, 1] by dividing by max_inhibition
        self.state.ffi_strength = min(1.0, raw_ffi / self.feedforward_inhibition.max_inhibition)

        # =====================================================================
        # 1. DENTATE GYRUS: Pattern Separation
        # =====================================================================
        # Random projections create orthogonal sparse codes
        dg_code = torch.matmul(self.w_ec_dg, input_spikes_float)  # [dg_size]

        # Apply FFI: reduce DG drive when input changes significantly
        # ffi_strength is now normalized to [0, 1]
        ffi_factor = 1.0 - self.state.ffi_strength * 0.5
        dg_code = dg_code * ffi_factor

        # Run through DG neurons (ConductanceLIF expects g_exc, g_inh)
        # DG has minimal inhibition - primarily feedforward excitation for pattern separation
        dg_g_exc = F.relu(dg_code)  # Clamp to positive conductance
        dg_spikes, _ = self.dg_neurons(dg_g_exc, g_inh_input=None)

        # Apply extreme winner-take-all sparsity
        # Note: cast needed because hasattr check doesn't narrow type for Pylance
        membrane_v = getattr(self.dg_neurons, 'v', None)
        dg_spikes = self._apply_wta_sparsity(
            dg_spikes,
            self.tri_config.dg_sparsity,
            membrane_v if isinstance(membrane_v, torch.Tensor) else None,
        )
        self.state.dg_spikes = dg_spikes

        # =====================================================================
        # APPLY DG→CA3 AXONAL DELAY
        # =====================================================================
        # Apply biological transmission delay for DG→CA3 mossy fiber pathway
        # If delay is 0, dg_spikes_delayed = dg_spikes (instant, backward compatible)
        if self._dg_ca3_delay_steps > 0:
            # Initialize buffer on first use
            if self._dg_ca3_delay_buffer is None:
                max_delay_steps = max(1, self._dg_ca3_delay_steps * 2 + 1)
                self._dg_ca3_delay_buffer = torch.zeros(
                    max_delay_steps, self.dg_size,
                    device=dg_spikes.device, dtype=torch.bool
                )
                self._dg_ca3_delay_ptr = 0

            # Store current spikes in circular buffer
            self._dg_ca3_delay_buffer[self._dg_ca3_delay_ptr] = dg_spikes

            # Retrieve delayed spikes
            read_idx = (self._dg_ca3_delay_ptr - self._dg_ca3_delay_steps) % self._dg_ca3_delay_buffer.shape[0]
            dg_spikes_delayed = self._dg_ca3_delay_buffer[read_idx]

            # Advance pointer
            self._dg_ca3_delay_ptr = (self._dg_ca3_delay_ptr + 1) % self._dg_ca3_delay_buffer.shape[0]
        else:
            dg_spikes_delayed = dg_spikes

        # Inter-stage shape check: DG output → CA3 input
        assert dg_spikes.shape == (self.dg_size,), (
            f"TrisynapticHippocampus: DG spikes have shape {dg_spikes.shape} "
            f"but expected ({self.dg_size},). "
            f"Check DG sparsity or EC→DG weights shape."
        )

        # =====================================================================
        # 2. CA3: Pattern Completion via Recurrence + Bistable Dynamics
        # =====================================================================
        # THETA GATING: The real hippocampus uses theta rhythm to temporally
        # separate encoding from retrieval:
        #
        # THETA TROUGH (encoding_mod high, retrieval_mod low):
        #   - DG→CA3 feedforward is STRONG (new patterns drive CA3)
        #   - CA3 recurrence is WEAK (prevents interference from old patterns)
        #   - Hebbian learning strengthens recurrent weights for new pattern
        #
        # THETA PEAK (encoding_mod low, retrieval_mod high):
        #   - DG→CA3 feedforward is SUPPRESSED (new input doesn't drive CA3)
        #   - CA3 recurrence is STRONG (attractors recall stored patterns)
        #   - CA3 outputs the MEMORY, not the current input!
        #
        # BISTABLE NEURONS: Real CA3 pyramidal neurons have intrinsic bistability
        # via I_NaP (persistent sodium) and I_CAN (Ca²⁺-activated cation) currents.
        # We model this with a persistent activity trace that:
        #   1. Accumulates when neurons fire
        #   2. Decays slowly (τ ~100-200ms)
        #   3. Provides positive feedback (self-sustaining activity)
        # This enables stable attractor states during delay periods.

        # Theta-modulated gating (gradual, not binary ON/OFF)
        # DG→CA3 (pattern separation path): Strong during encoding, weak during retrieval
        dg_ca3_gate = 0.1 + 0.9 * encoding_mod  # Range: 0.1 (retrieval) to 1.0 (encoding)

        # EC→CA3 (direct perforant path): Stronger during retrieval for cue-based recall
        # This provides the "seed" for pattern completion from partial cues
        ec_ca3_gate = 0.3 + 0.7 * retrieval_mod  # Range: 0.3 (encoding) to 1.0 (retrieval)

        # CA3 recurrence: Weak during encoding, strong during retrieval
        rec_gate = 0.2 + 0.8 * retrieval_mod  # Range: 0.2 (encoding) to 1.0 (retrieval)

        # Feedforward from DG (mossy fibers, theta-gated) with optional STP
        # NOTE: Use delayed DG spikes for biological accuracy
        if self.stp_mossy is not None:
            # Get STP efficacy for mossy fiber synapses
            # Mossy fibers are FACILITATING - repeated DG spikes progressively
            # enhance transmission to CA3
            stp_efficacy = self.stp_mossy(dg_spikes_delayed.float())
            # Apply STP to weights: (n_post, n_pre) * (n_pre, n_post).T
            effective_w_dg_ca3 = self.w_dg_ca3 * stp_efficacy.T
            ca3_from_dg = torch.matmul(effective_w_dg_ca3, dg_spikes_delayed.float()) * dg_ca3_gate  # [ca3_size]
        else:
            ca3_from_dg = torch.matmul(self.w_dg_ca3, dg_spikes_delayed.float()) * dg_ca3_gate  # [ca3_size]

        # Direct perforant path from EC (provides retrieval cues)
        # Strong during retrieval to seed the CA3 attractor from partial cues
        ca3_from_ec = torch.matmul(self.w_ec_ca3, input_spikes.float()) * ec_ca3_gate  # [ca3_size]

        # Total feedforward input to CA3
        ca3_ff = ca3_from_dg + ca3_from_ec

        # =====================================================================
        # RECURRENT CA3 WITH STP (CRITICAL FOR PREVENTING FROZEN ATTRACTORS)
        # =====================================================================
        # Without STP, recurrent connections cause the same neurons to fire
        # every timestep (frozen attractor). With DEPRESSING STP, frequently-
        # used synapses get temporarily weaker, allowing pattern transitions.

        # =====================================================================
        # ACETYLCHOLINE MODULATION OF CA3 RECURRENCE (Hasselmo 2006)
        # =====================================================================
        # High ACh (encoding mode): Suppress CA3 recurrence to prevent
        # interference from old patterns during new encoding
        # Low ACh (retrieval mode): Enhance CA3 recurrence to enable
        # pattern completion from partial cues
        #
        # Biological mechanism: ACh preferentially blocks recurrent (feedback)
        # connections via muscarinic receptors, while sparing feedforward input
        ach_level = self.state.acetylcholine
        # ACh > 0.5 → encoding mode → suppress recurrence (down to 0.3x)
        # ACh < 0.5 → retrieval mode → full recurrence (1.0x)
        ach_recurrent_modulation = 1.0 - 0.7 * max(0.0, ach_level - 0.5) / 0.5

        if self.stp_ca3_recurrent is not None and self.state.ca3_spikes is not None:
            # Get STP efficacy for CA3 recurrent synapses
            # CA3 recurrent is DEPRESSING - prevents frozen attractors
            # ADR-005: STP now accepts 1D [n_pre] directly
            stp_rec_efficacy = self.stp_ca3_recurrent(self.state.ca3_spikes.float())
            # Apply STP to recurrent weights
            effective_w_ca3_ca3 = self.w_ca3_ca3 * stp_rec_efficacy.T
            ca3_rec = torch.matmul(
                effective_w_ca3_ca3,
                self.state.ca3_spikes.float()
            ) * self.tri_config.ca3_recurrent_strength * rec_gate * ach_recurrent_modulation  # [ca3_size]
        else:
            # Recurrent from previous CA3 activity (theta-gated + ACh-modulated)
            ca3_rec = torch.matmul(
                self.w_ca3_ca3,
                self.state.ca3_spikes.float() if self.state.ca3_spikes is not None else torch.zeros(self.ca3_size, device=input_spikes.device)
            ) * self.tri_config.ca3_recurrent_strength * rec_gate * ach_recurrent_modulation  # [ca3_size]

        # =====================================================================
        # ACTIVITY-DEPENDENT FEEDBACK INHIBITION
        # =====================================================================
        # When many CA3 neurons are active, feedback inhibition increases.
        # This prevents runaway activity and frozen attractors.
        if self._ca3_activity_trace is None:
            self._ca3_activity_trace = torch.zeros(1, device=input_spikes.device)

        # Update activity trace (exponential moving average of total CA3 activity) - in-place
        if self.state.ca3_spikes is not None:
            current_activity = self.state.ca3_spikes.sum()
            self._ca3_activity_trace.mul_(0.9).add_(current_activity, alpha=0.1)

        # Compute feedback inhibition (scales with activity trace)
        feedback_inhibition = self._ca3_activity_trace * self.tri_config.ca3_feedback_inhibition

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

        # Ensure ca3_persistent is initialized
        if self.state.ca3_persistent is None:
            self.state.ca3_persistent = torch.zeros(self.ca3_size, device=input_spikes.device)

        ca3_persistent_input: torch.Tensor = (
            self.state.ca3_persistent * self.tri_config.ca3_persistent_gain
        )

        # Total CA3 input = feedforward + recurrent + persistent - inhibition
        ca3_input = ca3_ff + ca3_rec + ca3_persistent_input - feedback_inhibition

        # =====================================================================
        # NOREPINEPHRINE GAIN MODULATION (Locus Coeruleus)
        # =====================================================================
        # High NE (arousal/uncertainty): Increase gain → more responsive
        # Low NE (baseline): Normal gain
        # Biological: β-adrenergic receptors increase neuronal excitability
        ne_level = self.state.norepinephrine
        # NE gain: 1.0 (baseline) to 1.5 (high arousal)
        ne_gain = 1.0 + NE_GAIN_RANGE * ne_level
        ca3_input = ca3_input * ne_gain

        # INTRINSIC PLASTICITY: Apply per-neuron threshold offset
        # Neurons that fire too much have higher thresholds (less excitable)
        if (self.tri_config.homeostasis_enabled and
            self._ca3_threshold_offset is not None):
            ca3_input = ca3_input - self._ca3_threshold_offset

        # =====================================================================
        # THETA-GAMMA COUPLING: Slot-based gating
        # =====================================================================
        # Apply gamma slot gating to CA3 input. Neurons assigned to the
        # current slot get enhanced, others are suppressed.
        # This enables sequence encoding: items at different positions
        # activate different subsets of CA3 neurons.
        #
        # Two modes:
        # - "item": Slot from _sequence_position (each item gets own slot)
        # - "time": Slot from theta_slot (timing-based, used in replay)
        if self.tri_config.theta_gamma_enabled and self._ca3_slot_assignment is not None:
            cfg = self.tri_config

            # Note: Oscillators advance centrally in Brain, no local advance needed

            # Get current slot based on mode
            if cfg.gamma_slot_mode == "time":
                # Time-based: slot from brain's theta_slot broadcast
                current_slot = self._theta_slot
            else:  # "item" mode (default)
                # Item-based: slot from sequence position
                current_slot = self._sequence_position % cfg.gamma_n_slots

            # Create gating mask based on slot assignment
            # Neurons in current slot get full input, others are suppressed
            slot_match = (self._ca3_slot_assignment == current_slot).float()
            slot_match = slot_match.to(ca3_input.device)

            # ================================================================
            # THETA-MODULATED GAMMA AMPLITUDE
            # ================================================================
            # Gamma amplitude is modulated by theta phase:
            # - Theta trough (encoding): Strong gamma → sharp slot gating
            # - Theta peak (retrieval): Weak gamma → less gating, allows
            #   pattern completion across slots
            #
            # This implements phase-amplitude coupling (PAC) where gamma
            # power is highest at the encoding phase of theta.
            # Automatic coupling: gamma modulated by ALL slower oscillators
            gamma_amplitude = self._gamma_amplitude_effective  # [0, 1]

            # Scale gating strength by gamma amplitude
            # High amplitude → full gating strength
            # Low amplitude → reduced gating, more cross-slot activity
            effective_gating = cfg.gamma_gating_strength * gamma_amplitude

            # Blend: slot neurons get full input, others get reduced
            gamma_gate = (
                slot_match +
                (1.0 - slot_match) * (1.0 - effective_gating)
            )
            ca3_input = ca3_input * gamma_gate

        # Run through CA3 neurons (ConductanceLIF expects g_exc, g_inh)
        # Inhibition is handled by FFI module upstream
        ca3_g_exc = F.relu(ca3_input)  # Clamp to positive conductance
        ca3_spikes, _ = self.ca3_neurons(ca3_g_exc, g_inh_input=None)

        # CRITICAL FIX FOR OSCILLATION:
        # The LIF neuron resets membrane to v_reset after spiking, which causes
        # neurons that just spiked to have LOW membrane on the next timestep.
        # This makes WTA select DIFFERENT neurons, causing oscillation.
        #
        # Solution: After LIF processing, restore membrane potential for neurons
        # with high persistent activity. This models how I_NaP keeps neurons
        # near threshold even after spiking.
        if hasattr(self.ca3_neurons, 'membrane') and self.ca3_neurons.membrane is not None:
            with torch.no_grad():
                # Boost membrane for neurons with high persistent activity
                # This counteracts the post-spike reset
                persistent_boost = self.state.ca3_persistent * 1.5
                self.ca3_neurons.membrane = self.ca3_neurons.membrane + persistent_boost

        # Apply sparsity - now WTA will favor neurons with high persistent activity
        ca3_membrane_v = getattr(self.ca3_neurons, 'membrane', None)
        ca3_spikes = self._apply_wta_sparsity(
            ca3_spikes,
            self.tri_config.ca3_sparsity,
            ca3_membrane_v if isinstance(ca3_membrane_v, torch.Tensor) else None,
        )
        self.state.ca3_spikes = ca3_spikes

        # DEBUG: Track CA3 spike patterns for learning investigation
        if hasattr(self, '_debug_hippo') and self._debug_hippo:
            # Cast needed because PyTorch's tolist() returns Any
            ca3_active_indices = cast(List[int], ca3_spikes.squeeze().nonzero(as_tuple=True)[0].tolist())
            dg_active_count = dg_spikes.sum().item()
            input_sum = input_spikes.sum().item()
            print(f"      [HIPPO FWD] input_sum={input_sum:.0f}, "
                  f"dg_active={dg_active_count:.0f}, ca3_active={len(ca3_active_indices)}, "
                  f"dg_gate={dg_ca3_gate:.3f}, ec_gate={ec_ca3_gate:.3f}, rec_gate={rec_gate:.3f}")
            if len(ca3_active_indices) <= 10:
                print(f"        CA3 indices: {ca3_active_indices}")
            else:
                print(f"        CA3 indices (first 10): {ca3_active_indices[:10]}...")

        # Inter-stage shape check: CA3 output → CA1 input
        assert ca3_spikes.shape == (self.ca3_size,), (
            f"TrisynapticHippocampus: CA3 spikes have shape {ca3_spikes.shape} "
            f"but expected ({self.ca3_size},). "
            f"Check CA3 sparsity or DG→CA3 weights shape."
        )

        # Update persistent activity AFTER computing new spikes
        # The trace accumulates spike activity with slow decay
        # Using a direct accumulation: trace += spike - decay*trace
        # This ensures spikes have strong immediate effect but decay slowly
        dt = self.config.dt_ms
        decay_rate = dt / self.tri_config.ca3_persistent_tau

        # Update persistent activity: stronger during encoding, decay otherwise
        # Encoding_mod determines how much new spikes contribute vs decay
        # This is biologically motivated: Ca²⁺-dependent currents build up during
        # active encoding, then decay during maintenance/retrieval
        # Continuous modulation: contribution naturally weak when encoding_mod is low
        self.state.ca3_persistent = (
            self.state.ca3_persistent * (1.0 - decay_rate * (0.5 + 0.5 * retrieval_mod)) +
            ca3_spikes.float() * 0.5 * encoding_mod  # Contribution scaled by encoding strength
        )

        # Clamp to prevent runaway
        self.state.ca3_persistent = torch.clamp(self.state.ca3_persistent, 0.0, 3.0)

        # =====================================================================
        # HEBBIAN LEARNING: Apply to CA3 recurrent weights
        # This is how the hippocampus "stores" the pattern - in the weights!
        # Also store the DG pattern for later match/mismatch detection.
        # Modulated by theta encoding strength!
        # =====================================================================
        # Store the DG pattern (accumulate over timesteps, scaled by encoding strength)
        # Continuous modulation: storage naturally weak when encoding_mod is low
        if encoding_mod > 0.01:  # Only accumulate if encoding has minimal presence
            if self.state.stored_dg_pattern is None:
                self.state.stored_dg_pattern = dg_spikes.float().clone() * encoding_mod
            else:
                self.state.stored_dg_pattern = self.state.stored_dg_pattern + dg_spikes.float() * encoding_mod

        # One-shot Hebbian: strengthen connections between co-active neurons
        # Learning rate modulated by theta phase AND gamma amplitude
        # Continuous: learning automatically weak when encoding_mod is low
        ca3_activity = ca3_spikes.float()  # Already 1D, no squeeze needed
        if ca3_activity.sum() > 0:
            # Hebbian outer product: neurons that fire together wire together
            #
            # Gamma amplitude modulation: Learning is stronger when gamma
            # is strong (theta trough, encoding phase). This implements
            # the biological finding that synaptic plasticity is enhanced
            # during periods of strong gamma oscillations.
            base_lr = self.tri_config.ca3_recurrent_learning_rate * encoding_mod

            # Apply automatic gamma amplitude modulation
            # Gamma is modulated by ALL slower oscillators (emergent multi-order coupling)
            if self.tri_config.theta_gamma_enabled:
                gamma_mod = self._gamma_amplitude_effective
                effective_lr = base_lr * (0.5 + 0.5 * gamma_mod)  # Range: 50-100% based on gamma
            else:
                effective_lr = base_lr

            dW = effective_lr * torch.outer(ca3_activity, ca3_activity)

            # =========================================================
            # HETEROSYNAPTIC PLASTICITY: Weaken inactive synapses
            # =========================================================
            # Synapses to inactive postsynaptic neurons get weakened when
            # nearby neurons fire strongly. This prevents winner-take-all
            # dynamics from permanently dominating.
            #
            # Implementation: For each active presynaptic neuron, weaken
            # its connections to inactive postsynaptic neurons.
            if self.tri_config.heterosynaptic_ratio > 0:
                inactive_post = (ca3_activity < 0.5).float()  # Inactive neurons
                active_pre = ca3_activity  # Active neurons
                # Weaken: pre active but post inactive
                hetero_ltd = self.tri_config.heterosynaptic_ratio * effective_lr
                hetero_dW = -hetero_ltd * torch.outer(active_pre, inactive_post)
                dW = dW + hetero_dW

            with torch.no_grad():
                self.w_ca3_ca3.data += dW
                self.w_ca3_ca3.data.fill_diagonal_(0.0)  # No self-connections
                clamp_weights(self.w_ca3_ca3.data, self.tri_config.w_min, self.tri_config.w_max)

        # =====================================================================
        # 3. CA1: Coincidence Detection with Plastic EC→CA1 Pathway
        # =====================================================================
        # The key insight: EC→CA1 weights LEARN during sample phase to align
        # with whatever CA1 pattern the indirect pathway produces.
        #
        # ENCODE PHASE:
        #   - CA1 is driven primarily by CA3 (the memory being encoded)
        #   - EC→CA1 weights strengthen via Hebbian learning
        #   - After encoding, EC→CA1 pathway "knows" which EC patterns
        #     should activate which CA1 neurons
        #
        # RETRIEVE PHASE:
        #   - NMDA gating compares EC→CA1 (learned) with CA3→CA1 (recalled)
        #   - MATCH: Same pattern → same EC→CA1 activation → coincidence!
        #   - MISMATCH: Different pattern → different activation → no coincidence

        cfg = self.tri_config

        # =====================================================================
        # APPLY CA3→CA1 AXONAL DELAY
        # =====================================================================
        # Apply biological transmission delay for CA3→CA1 Schaffer collateral pathway
        # If delay is 0, ca3_spikes_delayed = ca3_spikes (instant, backward compatible)
        if self._ca3_ca1_delay_steps > 0:
            # Initialize buffer on first use
            if self._ca3_ca1_delay_buffer is None:
                max_delay_steps = max(1, self._ca3_ca1_delay_steps * 2 + 1)
                self._ca3_ca1_delay_buffer = torch.zeros(
                    max_delay_steps, self.ca3_size,
                    device=ca3_spikes.device, dtype=torch.bool
                )
                self._ca3_ca1_delay_ptr = 0

            # Store current spikes in circular buffer
            self._ca3_ca1_delay_buffer[self._ca3_ca1_delay_ptr] = ca3_spikes

            # Retrieve delayed spikes
            read_idx = (self._ca3_ca1_delay_ptr - self._ca3_ca1_delay_steps) % self._ca3_ca1_delay_buffer.shape[0]
            ca3_spikes_delayed = self._ca3_ca1_delay_buffer[read_idx]

            # Advance pointer
            self._ca3_ca1_delay_ptr = (self._ca3_ca1_delay_ptr + 1) % self._ca3_ca1_delay_buffer.shape[0]
        else:
            ca3_spikes_delayed = ca3_spikes

        # Feedforward from CA3 (retrieved/encoded memory) with optional STP
        # Schaffer collaterals are DEPRESSING - high-frequency CA3 activity
        # causes progressively weaker transmission to CA1
        # NOTE: Use delayed CA3 spikes for biological accuracy
        if self.stp_schaffer is not None:
            stp_efficacy = self.stp_schaffer(ca3_spikes_delayed.float())
            effective_w_ca3_ca1 = self.w_ca3_ca1 * stp_efficacy.T
            ca1_from_ca3 = torch.matmul(effective_w_ca3_ca1, ca3_spikes_delayed.float())  # [ca1_size]
        else:
            ca1_from_ca3 = torch.matmul(self.w_ca3_ca1, ca3_spikes_delayed.float())  # [ca1_size]

        # Direct from EC (current input) - use ec_direct_input if provided
        # ec_direct_input models EC layer III which carries raw sensory info
        # directly to CA1, separate from the EC layer II → DG trisynaptic path
        #
        # If ec_l3_input_size > 0, we have separate weights for EC L3
        # Otherwise, we use the same weights as EC L2
        #
        # EC→CA1 also has STP (depressing) - first presentation is strongest
        if ec_direct_input is not None and self.w_ec_l3_ca1 is not None:
            # Use separate EC L3 weights for raw sensory input
            if self.stp_ec_ca1 is not None:
                stp_efficacy = self.stp_ec_ca1(ec_direct_input.float())
                # Note: EC L3 may have different size, need to handle
                effective_w = self.w_ec_l3_ca1 * stp_efficacy.T
                ca1_from_ec = torch.matmul(effective_w, ec_direct_input.float())  # [ca1_size]
            else:
                ca1_from_ec = torch.matmul(self.w_ec_l3_ca1, ec_direct_input.float())  # [ca1_size]
            ec_input_for_ca1 = ec_direct_input
        elif ec_direct_input is not None:
            # ec_direct_input provided but same size as EC L2
            if self.stp_ec_ca1 is not None:
                stp_efficacy = self.stp_ec_ca1(ec_direct_input.float())
                effective_w = self.w_ec_ca1 * stp_efficacy.T
                ca1_from_ec = torch.matmul(effective_w, ec_direct_input.float())  # [ca1_size]
            else:
                ca1_from_ec = torch.matmul(self.w_ec_ca1, ec_direct_input.float())  # [ca1_size]
            ec_input_for_ca1 = ec_direct_input
        else:
            # Fall back to input_spikes (original behavior)
            # Note: STP for EC→CA1 is sized for ec_l3_input_size, not n_input,
            # so we don't apply STP when falling back to input_spikes
            ca1_from_ec = torch.matmul(self.w_ec_ca1, input_spikes.float())  # [ca1_size]
            ec_input_for_ca1 = input_spikes

        # Apply feedforward inhibition: strong input change reduces CA1 drive
        # This clears residual activity naturally (no explicit reset!)
        # ffi_strength is now normalized to [0, 1], so 0.8 gives max 80% suppression
        ffi_factor = 1.0 - self.state.ffi_strength * 0.8
        ca1_from_ec = ca1_from_ec * ffi_factor
        ca1_from_ca3 = ca1_from_ca3 * ffi_factor

        # CA1 processing varies continuously with theta modulation
        # - High encoding_mod: CA3-driven, learn EC→CA1
        # - High retrieval_mod: NMDA-gated coincidence detection
        # - Neutral: moderate activity

        # NMDA trace update (for retrieval gating)
        # Tracks CA3-induced depolarization for Mg²⁺ block removal
        if self.state.nmda_trace is not None:
            nmda_decay = torch.exp(torch.tensor(-dt / cfg.nmda_tau))
            self.state.nmda_trace = self.state.nmda_trace * nmda_decay + ca1_from_ca3 * (1.0 - nmda_decay)
        else:
            self.state.nmda_trace = ca1_from_ca3.clone()

        # NMDA gating: Mg²⁺ block removal based on CA3 depolarization
        # Stronger during retrieval (theta peak)
        mg_block_removal = torch.sigmoid(
            (self.state.nmda_trace - cfg.nmda_threshold) * cfg.nmda_steepness
        ) * retrieval_mod
        nmda_current = ca1_from_ec * mg_block_removal

        # AMPA current: fast baseline transmission
        ampa_current = ca1_from_ec * cfg.ampa_ratio

        # CA3 contribution: stronger during encoding
        ca3_contribution = ca1_from_ca3 * (0.5 + 0.5 * encoding_mod)

        # Total CA1 input
        ca1_input = ca3_contribution + ampa_current + nmda_current

        # Split excitation and inhibition for ConductanceLIF
        # Excitatory: CA3 + EC pathways
        ca1_g_exc = F.relu(ca1_input)  # Clamp to positive

        # Inhibitory: lateral inhibition from other CA1 neurons
        ca1_g_inh = None
        if self.state.ca1_spikes is not None:
            ca1_g_inh = F.relu(torch.matmul(
                self.state.ca1_spikes.float(),
                self.w_ca1_inhib.t()
            ))

        # Run through CA1 neurons (ConductanceLIF with E/I separation)
        ca1_spikes, _ = self.ca1_neurons(ca1_g_exc, ca1_g_inh)

        # Apply sparsity (more lenient during retrieval to allow mismatch detection)
        ca1_membrane_v = getattr(self.ca1_neurons, 'v', None)
        sparsity_factor = 1.0 + 0.5 * retrieval_mod  # Higher threshold during retrieval
        ca1_spikes = self._apply_wta_sparsity(
            ca1_spikes,
            cfg.ca1_sparsity * sparsity_factor,
            ca1_membrane_v if isinstance(ca1_membrane_v, torch.Tensor) else None,
        )

        # ---------------------------------------------------------
        # HEBBIAN LEARNING: EC→CA1 plasticity (during encoding)
        # ---------------------------------------------------------
        # Strengthen connections: active EC neurons → active CA1 neurons
        # This aligns the direct pathway with the indirect pathway
        # Learning modulated by theta encoding strength (continuous)
        ec_activity = ec_input_for_ca1.float()
        ca1_activity = ca1_spikes.float()

        if ec_activity.sum() > 0 and ca1_activity.sum() > 0:
            # Hebbian outer product: w_ij += lr * post_j * pre_i
            # Learning rate automatically weak when encoding_mod is low
            base_lr = cfg.ec_ca1_learning_rate * encoding_mod

            # Apply automatic gamma amplitude modulation
            if self.tri_config.theta_gamma_enabled:
                gamma_mod = self._gamma_amplitude_effective
                effective_lr = base_lr * (0.5 + 0.5 * gamma_mod)
            else:
                effective_lr = base_lr

            dW = effective_lr * torch.outer(ca1_activity, ec_activity)
            with torch.no_grad():
                # Update the appropriate weight matrix
                if ec_direct_input is not None and self.w_ec_l3_ca1 is not None:
                    self.w_ec_l3_ca1.data += dW
                    clamp_weights(self.w_ec_l3_ca1.data, cfg.w_min, cfg.w_max)
                else:
                    self.w_ec_ca1.data += dW
                    clamp_weights(self.w_ec_ca1.data, cfg.w_min, cfg.w_max)

        self.state.ca1_spikes = ca1_spikes

        # =====================================================================
        # Update STDP Traces (for learning, not comparison)
        # =====================================================================
        if self.state.dg_trace is not None:
            update_trace(self.state.dg_trace, dg_spikes, tau=self.tri_config.tau_plus_ms, dt=dt)
        if self.state.ca3_trace is not None:
            update_trace(self.state.ca3_trace, ca3_spikes, tau=self.tri_config.tau_plus_ms, dt=dt)

        # Store spikes in base state for compatibility
        # CA1 spikes ARE the output - downstream learns from these patterns!
        self.state.spikes = ca1_spikes

        # Apply continuous plasticity (learning happens as part of forward dynamics)
        self._apply_plasticity(input_spikes, ca1_spikes, dt)

        # =====================================================================
        # AUTO-ADVANCE GAMMA SLOT
        # =====================================================================
        # After processing this input, advance to next gamma slot.
        # This is biologically realistic: position in sequence is determined
        # by WHEN the item arrives during the theta cycle, not by an external
        # counter. Each input naturally advances the gamma phase.
        if self.tri_config.theta_gamma_enabled:
            self._sequence_position += 1
            # Oscillators advance centrally in Brain, so we just track position here.
            # The position is used for diagnostics and can be reset by new_trial().

        # Apply axonal delay (biological reality: ALL neural connections have delays)
        delayed_spikes = self._apply_axonal_delay(ca1_spikes, dt)

        # Ensure bool output (CA1 neurons already return bool from Phase 1)
        # WTA sparsity also returns bool
        return delayed_spikes

    def _apply_wta_sparsity(
        self,
        spikes: torch.Tensor,
        target_sparsity: float,
        membrane: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply winner-take-all sparsity to enforce sparse coding.

        Only the top-k neurons (by membrane potential) are allowed to spike.
        Returns bool tensor for consistency with neuron models.

        Args:
            spikes: Spike tensor [n_neurons] (1D)
            target_sparsity: Fraction of neurons to keep active
            membrane: Optional membrane potentials [n_neurons] for selection

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
            # Keep top-k by membrane potential
            active_v = membrane[active]
            _, top_k_idx = torch.topk(active_v, k)
            sparse_spikes[active[top_k_idx]] = True
        else:
            # Random selection if no membrane
            perm = torch.randperm(len(active))[:k]
            sparse_spikes[active[perm]] = True

        return sparse_spikes

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Optional[Dict[str, float]] = None,
        theta_slot: int = 0,
        coupled_amplitudes: Optional[Dict[str, float]] = None,
    ) -> None:
        """Receive oscillator information from brain broadcast.

        Replaces local GammaOscillator with centralized timing from OscillatorManager.
        Gamma effective amplitude implements automatic multiplicative coupling.

        Args:
            phases: Oscillator phases in radians {'theta': ..., 'gamma': ..., etc}
            signals: Oscillator signal values {'theta': ..., 'gamma': ..., etc}
            theta_slot: Current theta slot [0, n_slots-1] for working memory
            coupled_amplitudes: Effective amplitudes per oscillator (pre-computed)

        Note:
            Called automatically by Brain before each forward() call.
            Do not call this manually.
        """
        # Store oscillator values for use in forward()
        self._theta_phase = phases.get('theta', 0.0)
        self._gamma_phase = phases.get('gamma', 0.0)
        self._theta_slot = theta_slot

        # Store effective gamma amplitude (pre-computed by OscillatorManager)
        # Automatic multiplicative coupling:
        # - Gamma modulated by ALL slower oscillators (delta, theta, alpha, beta)
        # OscillatorManager handles the multiplication, we just store the result.
        # This gives emergent delta-theta-alpha-beta-gamma multi-order coupling.
        if coupled_amplitudes is not None:
            self._gamma_amplitude_effective = coupled_amplitudes.get('gamma', 1.0)
        else:
            self._gamma_amplitude_effective = 1.0

    def _apply_plasticity(
        self,
        _input_spikes: torch.Tensor,
        _output_spikes: torch.Tensor,
        _dt: float = 1.0,
    ) -> None:
        """
        Apply homeostatic plasticity to CA3 recurrent weights.

        CA3 recurrent learning (one-shot Hebbian) happens in forward().
        This method only applies homeostatic mechanisms (synaptic scaling,
        intrinsic plasticity) to maintain stable network dynamics.

        Note: _input_spikes, _output_spikes, and _dt are required by the base class
        signature but not used here since we access spikes from self.state.
        """
        if not self.plasticity_enabled:
            return

        cfg = self.tri_config

        # NOTE: All neuromodulators (DA, ACh, NE) are now managed centrally by Brain.
        # VTA updates dopamine, LC updates NE, NB updates ACh.
        # Brain broadcasts to all regions every timestep via _update_neuromodulators().
        # No local decay needed.

        # CA3 recurrent learning happens in forward() using one-shot Hebbian.
        # Here we only apply homeostatic mechanisms (if enabled).

        if not cfg.homeostasis_enabled:
            return

        # Apply homeostatic synaptic scaling to CA3 recurrent weights
        with torch.no_grad():
            self.w_ca3_ca3.data = self.homeostasis.normalize_weights(self.w_ca3_ca3.data, dim=1)
            self.w_ca3_ca3.data.fill_diagonal_(0.0)  # Maintain no self-connections

        # Apply intrinsic plasticity (threshold adaptation) using homeostasis helper
        if self.state.ca3_spikes is not None:
            ca3_spikes_1d = self.state.ca3_spikes.squeeze()

            # Initialize if needed
            if self._ca3_activity_history is None:
                self._ca3_activity_history = torch.zeros(self.ca3_size, device=self.device)
            if self._ca3_threshold_offset is None:
                self._ca3_threshold_offset = torch.zeros(self.ca3_size, device=self.device)

            # Update activity history (exponential moving average)
            self._ca3_activity_history.mul_(0.99).add_(ca3_spikes_1d.float(), alpha=0.01)

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

    # endregion

    # region Episodic Memory

    def get_state(self) -> HippocampusState:
        """Get current state."""
        return self.state

    def store_episode(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        correct: bool,
        context: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority_boost: float = 0.0,
        sequence: Optional[List[torch.Tensor]] = None,
        # HER parameters (automatic integration)
        goal: Optional[torch.Tensor] = None,
        achieved_goal: Optional[torch.Tensor] = None,
        done: bool = False,
    ) -> None:
        """Store an episode in episodic memory for later replay.

        Priority is computed based on reward magnitude and correctness.

        **AUTOMATIC HER INTEGRATION**: If use_her=True and goal/achieved_goal
        are provided, this method AUTOMATICALLY calls add_her_experience()
        to populate the HER buffer. No manual calls needed during training!

        Args:
            state: Final activity pattern at decision time
            action: Selected action
            reward: Received reward
            correct: Whether the action was correct
            context: Optional context/cue pattern
            metadata: Optional additional info
            priority_boost: Extra priority for this episode
            sequence: Optional list of CA3 patterns from each gamma slot
                      during encoding. Enables gamma-driven replay.
            goal: (HER) What the agent was trying to achieve (PFC working memory)
            achieved_goal: (HER) What was actually achieved (CA1 output)
            done: (HER) Whether episode terminated
        """
        # Delegate to memory component (note: method is store_memory, not store_episode)
        self.memory.store_memory(
            state=state,
            action=action,
            reward=reward,
            correct=correct,
            context=context,
            metadata=metadata,
            priority_boost=priority_boost,
            sequence=sequence,
        )

        # =====================================================================
        # AUTOMATIC HER INTEGRATION
        # =====================================================================
        # If HER is enabled AND goal information was provided, automatically
        # add this experience to the HER buffer. This makes HER completely
        # automatic during normal brain operation - no manual calls needed!
        #
        # HER needs:
        # - state: Current state (provided)
        # - action: Action taken (provided)
        # - next_state: For simplicity, use state (single-step experiences)
        # - goal: What we were trying to achieve (from PFC)
        # - achieved_goal: What actually happened (from CA1)
        # - reward: Reward received (provided)
        # - done: Episode termination (provided)
        #
        # During consolidation, HER will relabel failed attempts as successes
        # for the goals that were actually achieved, providing sample-efficient
        # multi-goal learning.
        # =====================================================================
        if self.her_integration is not None and goal is not None:
            self.add_her_experience(
                state=state.clone(),
                action=action,
                next_state=state.clone(),  # Single-step (decision point)
                goal=goal,
                reward=reward,
                done=done,
                achieved_goal=achieved_goal,
            )

    def sample_episodes_prioritized(self, n: int) -> List[Episode]:
        """Sample episodes with probability proportional to priority."""
        return self.memory.sample_episodes_prioritized(n)

    def retrieve_similar(
        self,
        query_state: torch.Tensor,
        query_action: Optional[int] = None,
        k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Retrieve K most similar past experiences from episodic memory.

        For Phase 2 model-based planning: provides outcome predictions based
        on similar past experiences. Uses pattern completion capability of
        hippocampus to predict what will happen next.

        Biology: Hippocampus retrieves similar past episodes during planning
        and decision-making (Johnson & Redish, 2007). CA3 pattern completion
        allows partial cues to retrieve full memories.

        Args:
            query_state: State to find similar experiences for [n] (1D, ADR-005)
            query_action: Optional action to filter by (boosts similarity)
            k: Number of similar experiences to retrieve
            similarity_threshold: Minimum similarity to return (0.0-1.0)

        Returns:
            similar_episodes: List of dicts with keys:
                - 'state': Episode state tensor
                - 'action': Action taken
                - 'next_state': Resulting state (approximated)
                - 'reward': Reward received
                - 'similarity': Cosine similarity score (0.0-1.0)
                - 'context': Optional context tensor
                - 'metadata': Optional metadata dict

        Note:
            Uses cosine similarity in state space. For more sophisticated
            retrieval, could use CA3 recurrent dynamics or DG-CA3-CA1 circuit.
        """
        return self.memory.retrieve_similar(
            query_state=query_state,
            query_action=query_action,
            k=k,
            similarity_threshold=similarity_threshold,
        )

    # =========================================================================
    # HINDSIGHT EXPERIENCE REPLAY (HER)
    # =========================================================================

    def add_her_experience(
        self,
        state: torch.Tensor,
        action: int,
        next_state: torch.Tensor,
        goal: torch.Tensor,
        reward: float,
        done: bool,
        achieved_goal: Optional[torch.Tensor] = None
    ) -> None:
        """Add experience to HER buffer for goal-conditioned learning.

        Args:
            state: Current state (CA3 pattern or cortex output)
            action: Action taken
            next_state: Resulting state
            goal: Intended goal (from PFC working memory)
            reward: Reward received
            done: Episode terminated?
            achieved_goal: What was actually achieved (if None, uses next_state)
        """
        if self.her_integration is None:
            return

        # If achieved_goal not provided, use CA1 output as proxy
        if achieved_goal is None:
            achieved_goal = self.state.ca1_spikes if self.state.ca1_spikes is not None else next_state

        self.her_integration.add_experience(
            state=state,
            action=action,
            next_state=next_state,
            goal=goal,
            reward=reward,
            done=done,
            achieved_goal=achieved_goal
        )

    def enter_consolidation_mode(self) -> None:
        """Enter consolidation/sleep mode for HER replay.

        During consolidation:
        - Hippocampus replays stored experiences
        - Hindsight goals are generated
        - Both real and hindsight experiences available for learning
        """
        if self.her_integration is not None:
            self.her_integration.enter_consolidation()

    def exit_consolidation_mode(self) -> None:
        """Exit consolidation mode, return to active learning."""
        if self.her_integration is not None:
            self.her_integration.exit_consolidation()

    def sample_her_replay_batch(self, batch_size: int = 32) -> List:
        """Sample batch of experiences for HER replay learning.

        Returns mix of real and hindsight-relabeled experiences.
        Only returns data during consolidation mode.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of EpisodeTransition objects (real + hindsight mix)
        """
        if self.her_integration is None:
            return []

        return self.her_integration.replay_for_learning(batch_size)

    def get_her_diagnostics(self) -> Dict[str, Any]:
        """Get HER system diagnostics."""
        if self.her_integration is None:
            return {"her_enabled": False}

        diagnostics = self.her_integration.get_diagnostics()
        diagnostics["her_enabled"] = True
        return diagnostics

    # =========================================================================
    # GAMMA-DRIVEN SEQUENCE REPLAY
    # =========================================================================

    def replay_sequence(
        self,
        episode: Episode,
        compression_factor: float = 5.0,
        dt: float = 1.0,
    ) -> Dict[str, Any]:
        """Replay a sequence using gamma oscillator for time-compressed reactivation.

        Now uses unified ReplayEngine for consistent replay across codebase.

        During sleep, sequences that took seconds to encode are replayed in ~100ms.
        The gamma oscillator drives slot-by-slot reactivation at compressed timing.

        This implements the biological phenomenon where hippocampal sequences
        are replayed during sharp-wave ripples at 5-20x compression.

        Args:
            episode: Episode to replay (must have sequence field populated)
            compression_factor: Time compression (5.0 = 5x faster than encoding)
            dt: Base time step in ms

        Returns:
            Dict with replay metrics:
                - slots_replayed: Number of sequence slots replayed
                - total_activity: Sum of all reactivated patterns
                - gamma_cycles: Number of gamma cycles during replay
                - compression_factor: Actual compression used
                - replayed_patterns: List of replayed pattern tensors
        """
        if self.replay_engine is None:
            raise ComponentError(
                "Hippocampus",
                "Replay engine not available. Set theta_gamma_enabled=True in config."
            )

        # Update compression factor if different from config
        if compression_factor != self.replay_engine.config.compression_factor:
            self.replay_engine.config.compression_factor = compression_factor

        # Update dt if different
        if dt != self.replay_engine.config.dt_ms:
            self.replay_engine.config.dt_ms = dt

        # Switch to time-based mode for replay
        original_mode = self.tri_config.gamma_slot_mode
        self.tri_config.gamma_slot_mode = "time"

        # Pattern processor: forward through CA3 for pattern completion
        # Theta modulation computed internally from self._theta_phase
        def process_pattern(pattern: torch.Tensor) -> torch.Tensor:
            return self.forward(pattern)

        # Gating function: apply gamma gating to patterns
        def get_gating(slot: int) -> float:
            return self._get_gamma_gating(slot)

        # Run replay through unified engine, passing gamma phase from brain
        result = self.replay_engine.replay(
            episode=episode,
            pattern_processor=process_pattern,
            gating_fn=get_gating,
            gamma_phase=self._gamma_phase,  # From brain's OscillatorManager
        )

        # Restore original mode
        self.tri_config.gamma_slot_mode = original_mode

        # Convert ReplayResult to dict format
        return {
            "slots_replayed": result.slots_replayed,
            "total_activity": result.total_activity,
            "gamma_cycles": result.gamma_cycles,
            "compression_factor": result.compression_factor,
            "replayed_patterns": result.replayed_patterns,
        }

    def _get_gamma_gating(self, slot: int) -> float:
        """Get gamma gating strength for a specific slot."""
        if not self.tri_config.theta_gamma_enabled:
            return 1.0

        cfg = self.tri_config
        n_slots = cfg.gamma_n_slots

        # Compute phase difference from target slot
        target_phase = (slot / n_slots) * (2 * math.pi)
        current_gamma_phase = self._gamma_phase  # From brain broadcast

        # Gaussian gating around target phase
        phase_diff = abs(current_gamma_phase - target_phase)
        phase_diff = min(phase_diff, 2 * math.pi - phase_diff)  # Wrap around

        gating = math.exp(-phase_diff ** 2 / (2 * 0.5 ** 2))  # σ = 0.5 radians

        # Scale by gating strength
        return 1.0 - cfg.gamma_gating_strength * (1.0 - gating)

    # endregion

    # region Diagnostics and Health Monitoring

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics using DiagnosticsMixin helpers.

        Returns consolidated diagnostic information about:
        - Layer activity (spikes and rates)
        - CA3 recurrent state and bistable dynamics
        - CA1 comparison mechanism (NMDA trace, Mg block removal)
        - Stored patterns and similarity
        - Weight statistics
        - Theta phase (if applicable)

        This is the primary diagnostic interface for the Hippocampus.
        """
        cfg = self.tri_config
        state = self.state

        # Layer activity using spike_diagnostics helper
        dg_activity: Dict[str, Any] = {"target_sparsity": cfg.dg_sparsity}
        if state.dg_spikes is not None:
            dg_activity.update(self.spike_diagnostics(state.dg_spikes, ""))

        ca3_activity: Dict[str, Any] = {"target_sparsity": cfg.ca3_sparsity}
        if state.ca3_spikes is not None:
            ca3_activity.update(self.spike_diagnostics(state.ca3_spikes, ""))

        ca1_activity: Dict[str, Any] = {"target_sparsity": cfg.ca1_sparsity}
        if state.ca1_spikes is not None:
            ca1_activity.update(self.spike_diagnostics(state.ca1_spikes, ""))

        # CA3 bistable dynamics using trace_diagnostics
        ca3_persistent: Dict[str, Any] = {}
        if state.ca3_persistent is not None:
            ca3_persistent.update(self.trace_diagnostics(state.ca3_persistent, ""))
            ca3_persistent["nonzero_count"] = (state.ca3_persistent > 0.1).sum().item()

        # CA1 NMDA comparison mechanism using trace_diagnostics
        nmda_diagnostics: Dict[str, Any] = {"threshold": cfg.nmda_threshold}
        if state.nmda_trace is not None:
            nmda_diagnostics.update(self.trace_diagnostics(state.nmda_trace, ""))
            nmda_diagnostics["trace_std"] = state.nmda_trace.std().item()
            nmda_diagnostics["above_threshold_count"] = (state.nmda_trace > cfg.nmda_threshold).sum().item()

            # Compute Mg block removal (sigmoid of (trace - threshold) * steepness)
            mg_removal = torch.sigmoid(
                (state.nmda_trace - cfg.nmda_threshold) * cfg.nmda_steepness
            )
            nmda_diagnostics["mg_block_removal_mean"] = mg_removal.mean().item()
            nmda_diagnostics["mg_block_removal_max"] = mg_removal.max().item()
            nmda_diagnostics["gated_neurons"] = (mg_removal > 0.5).sum().item()

        # Stored pattern comparison (for match/mismatch)
        pattern_comparison: Dict[str, Any] = {
            "has_stored_pattern": state.stored_dg_pattern is not None,
        }
        if state.stored_dg_pattern is not None and state.dg_spikes is not None:
            stored = state.stored_dg_pattern.float().squeeze()
            current = state.dg_spikes.float().squeeze()
            # Cosine similarity between stored and current DG patterns
            similarity = cosine_similarity_safe(stored, current)
            pattern_comparison["dg_similarity"] = similarity.item()
            pattern_comparison["stored_active"] = (stored > 0).sum().item()
            pattern_comparison["current_active"] = (current > 0).sum().item()
            pattern_comparison["overlap"] = ((stored > 0) & (current > 0)).sum().item()

        # Feedforward inhibition state
        ffi_state = {
            "current_strength": state.ffi_strength,
        }

        # Custom metrics for hippocampus
        custom = {
            "layer_sizes": {
                "dg": self.dg_size,
                "ca3": self.ca3_size,
                "ca1": self.ca1_size,
            },
            # Layer activity
            "dg": dg_activity,
            "ca3": ca3_activity,
            "ca1": ca1_activity,
            # Dynamics
            "ca3_persistent": ca3_persistent,
            # Comparison mechanism (critical for match/mismatch)
            "nmda": nmda_diagnostics,
            "pattern_comparison": pattern_comparison,
            # Inhibition
            "ffi": ffi_state,
            # Episode buffer
            "episode_buffer_size": len(self.episode_buffer),
            # HER diagnostics
            "her": self.get_her_diagnostics(),
        }

        # Use collect_standard_diagnostics for weight statistics
        return self.collect_standard_diagnostics(
            region_name="hippocampus",
            weight_matrices={
                "ec_dg": self.w_ec_dg.data,
                "dg_ca3": self.w_dg_ca3.data,
                "ca3_ca3": self.w_ca3_ca3.data,
                "ca3_ca1": self.w_ca3_ca1.data,
                "ec_ca1": self.w_ec_ca1.data,
            },
            custom_metrics=custom,
        )

    # =========================================================================
    # CHECKPOINT STATE MANAGEMENT
    # =========================================================================

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for checkpointing.

        Returns all state needed to resume training from this exact point.

        Returns:
            Dictionary with complete region state
        """
        # 1. LEARNABLE PARAMETERS (weights for all pathways)
        weights = {
            "w_ec_dg": self.w_ec_dg.detach().clone(),
            "w_dg_ca3": self.w_dg_ca3.detach().clone(),
            "w_ca3_ca1": self.w_ca3_ca1.detach().clone(),
            "w_ca3_ca3": self.w_ca3_ca3.detach().clone(),
            "w_ec_ca1": self.w_ec_ca1.detach().clone(),
            "w_ec_l3_ca1": self.w_ec_l3_ca1.detach().clone() if self.w_ec_l3_ca1 is not None else None,
            "w_ca1_inhib": self.w_ca1_inhib.detach().clone(),
        }

        # 2. REGION STATE (all three layers)
        neuron_state = {
            "dg": self.dg_neurons.get_state(),
            "ca3": self.ca3_neurons.get_state(),
            "ca1": self.ca1_neurons.get_state(),
        }

        region_state = {
            "neuron_state": neuron_state,
            "ca3_activity_trace": self._ca3_activity_trace.detach().clone() if self._ca3_activity_trace is not None else None,
            "pending_theta_reset": self._pending_theta_reset,
            "sequence_position": self._sequence_position,
            # Intrinsic plasticity state
            "ca3_threshold_offset": self._ca3_threshold_offset.detach().clone() if self._ca3_threshold_offset is not None else None,
            "ca3_activity_history": self._ca3_activity_history.detach().clone() if self._ca3_activity_history is not None else None,
            # Gamma slot assignment
            "ca3_slot_assignment": self._ca3_slot_assignment.detach().clone() if self._ca3_slot_assignment is not None else None,
            # Inter-layer axonal delay buffers
            "dg_ca3_delay_buffer": self._dg_ca3_delay_buffer.detach().clone() if self._dg_ca3_delay_buffer is not None else None,
            "dg_ca3_delay_pointer": self._dg_ca3_delay_ptr,
            "ca3_ca1_delay_buffer": self._ca3_ca1_delay_buffer.detach().clone() if self._ca3_ca1_delay_buffer is not None else None,
            "ca3_ca1_delay_pointer": self._ca3_ca1_delay_ptr,
            "trisynaptic_state": {
                "dg_spikes": self.state.dg_spikes.detach().clone() if self.state.dg_spikes is not None else None,
                "ca3_spikes": self.state.ca3_spikes.detach().clone() if self.state.ca3_spikes is not None else None,
                "ca1_spikes": self.state.ca1_spikes.detach().clone() if self.state.ca1_spikes is not None else None,
                "ca3_membrane": self.state.ca3_membrane.detach().clone() if self.state.ca3_membrane is not None else None,
                "ca3_persistent": self.state.ca3_persistent.detach().clone() if self.state.ca3_persistent is not None else None,
                "sample_trace": self.state.sample_trace.detach().clone() if self.state.sample_trace is not None else None,
                "dg_trace": self.state.dg_trace.detach().clone() if self.state.dg_trace is not None else None,
                "ca3_trace": self.state.ca3_trace.detach().clone() if self.state.ca3_trace is not None else None,
                "nmda_trace": self.state.nmda_trace.detach().clone() if self.state.nmda_trace is not None else None,
                "stored_dg_pattern": self.state.stored_dg_pattern.detach().clone() if self.state.stored_dg_pattern is not None else None,
                "ffi_strength": self.state.ffi_strength,
            }
        }

        # 3. LEARNING STATE (STP for all pathways)
        learning_state = {}
        if self.stp_mossy is not None:
            learning_state["stp_mossy"] = self.stp_mossy.get_state()
        if self.stp_schaffer is not None:
            learning_state["stp_schaffer"] = self.stp_schaffer.get_state()
        if self.stp_ec_ca1 is not None:
            learning_state["stp_ec_ca1"] = self.stp_ec_ca1.get_state()
        if self.stp_ca3_recurrent is not None:
            learning_state["stp_ca3_recurrent"] = self.stp_ca3_recurrent.get_state()

        # 4. OSCILLATOR STATE
        # Note: Oscillator phases now come from brain broadcast, not stored locally.
        # We only save replay_engine state (which still needs migration separately).
        oscillator_state = {}
        if self.replay_engine is not None:
            oscillator_state["replay_engine"] = self.replay_engine.get_state()

        # 5. NEUROMODULATOR STATE
        neuromodulator_state = self.get_neuromodulator_state()

        # 6. EPISODIC MEMORY (episode buffer)
        # Note: Episodes contain tensors, so we need to serialize carefully
        episode_buffer_state = []
        for ep in self.episode_buffer:
            ep_state = {
                "state": ep.state.detach().clone(),
                "context": ep.context.detach().clone() if ep.context is not None else None,
                "action": ep.action,
                "reward": ep.reward,
                "correct": ep.correct,
                "metadata": ep.metadata,
                "priority": ep.priority,
                "timestamp": ep.timestamp,
                "sequence": [s.detach().clone() for s in ep.sequence] if ep.sequence is not None else None,
            }
            episode_buffer_state.append(ep_state)

        return {
            "format": "elastic_tensor",  # Format identifier for hybrid checkpoints
            "weights": weights,
            "region_state": region_state,
            "learning_state": learning_state,
            "oscillator_state": oscillator_state,
            "neuromodulator_state": neuromodulator_state,
            "episode_buffer": episode_buffer_state,
            "config": self.tri_config,
        }

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Restore complete state from checkpoint.

        Args:
            state: Dictionary returned by get_full_state()

        Raises:
            ValueError: If state is incompatible with current configuration
        """
        # Validate configuration compatibility
        saved_config = state["config"]
        if saved_config.n_input != self.tri_config.n_input:
            raise CheckpointError(
                f"Input dimension mismatch: saved={saved_config.n_input}, "
                f"current={self.tri_config.n_input}"
            )
        if saved_config.n_output != self.tri_config.n_output:
            raise CheckpointError(
                f"Output dimension mismatch: saved={saved_config.n_output}, "
                f"current={self.tri_config.n_output}"
            )

        # 1. RESTORE WEIGHTS
        weights = state["weights"]
        self.w_ec_dg.data = weights["w_ec_dg"].to(self.device)
        self.w_dg_ca3.data = weights["w_dg_ca3"].to(self.device)
        self.w_ca3_ca1.data = weights["w_ca3_ca1"].to(self.device)
        self.w_ca3_ca3.data = weights["w_ca3_ca3"].to(self.device)
        self.w_ec_ca1.data = weights["w_ec_ca1"].to(self.device)
        if weights["w_ec_l3_ca1"] is not None and self.w_ec_l3_ca1 is not None:
            self.w_ec_l3_ca1.data = weights["w_ec_l3_ca1"].to(self.device)
        self.w_ca1_inhib.data = weights["w_ca1_inhib"].to(self.device)
        self.weights = self.w_ca3_ca1  # Update reference

        # 2. RESTORE REGION STATE
        region_state = state["region_state"]

        # Restore neuron state for all layers
        neuron_state = region_state["neuron_state"]
        self.dg_neurons.load_state(neuron_state["dg"])
        self.ca3_neurons.load_state(neuron_state["ca3"])
        self.ca1_neurons.load_state(neuron_state["ca1"])

        # Restore other region state
        if region_state["ca3_activity_trace"] is not None:
            self._ca3_activity_trace = region_state["ca3_activity_trace"].to(self.device)
        self._pending_theta_reset = region_state["pending_theta_reset"]
        self._sequence_position = region_state["sequence_position"]
        # Restore intrinsic plasticity state
        if region_state["ca3_threshold_offset"] is not None:
            self._ca3_threshold_offset = region_state["ca3_threshold_offset"].to(self.device)
        if region_state["ca3_activity_history"] is not None:
            self._ca3_activity_history = region_state["ca3_activity_history"].to(self.device)
        if region_state["ca3_slot_assignment"] is not None:
            self._ca3_slot_assignment = region_state["ca3_slot_assignment"].to(self.device)

        # Restore inter-layer delay buffers (if present in checkpoint)
        if "dg_ca3_delay_buffer" in region_state and region_state["dg_ca3_delay_buffer"] is not None:
            self._dg_ca3_delay_buffer = region_state["dg_ca3_delay_buffer"].to(self.device)
            self._dg_ca3_delay_ptr = region_state["dg_ca3_delay_pointer"]
        if "ca3_ca1_delay_buffer" in region_state and region_state["ca3_ca1_delay_buffer"] is not None:
            self._ca3_ca1_delay_buffer = region_state["ca3_ca1_delay_buffer"].to(self.device)
            self._ca3_ca1_delay_ptr = region_state["ca3_ca1_delay_pointer"]

        # Restore HippocampusState
        tri_state = region_state["trisynaptic_state"]
        self.state.dg_spikes = tri_state["dg_spikes"].to(self.device) if tri_state["dg_spikes"] is not None else None
        self.state.ca3_spikes = tri_state["ca3_spikes"].to(self.device) if tri_state["ca3_spikes"] is not None else None
        self.state.ca1_spikes = tri_state["ca1_spikes"].to(self.device) if tri_state["ca1_spikes"] is not None else None
        self.state.ca3_membrane = tri_state["ca3_membrane"].to(self.device) if tri_state["ca3_membrane"] is not None else None
        self.state.ca3_persistent = tri_state["ca3_persistent"].to(self.device) if tri_state["ca3_persistent"] is not None else None
        self.state.sample_trace = tri_state["sample_trace"].to(self.device) if tri_state["sample_trace"] is not None else None
        self.state.dg_trace = tri_state["dg_trace"].to(self.device) if tri_state["dg_trace"] is not None else None
        self.state.ca3_trace = tri_state["ca3_trace"].to(self.device) if tri_state["ca3_trace"] is not None else None
        self.state.nmda_trace = tri_state["nmda_trace"].to(self.device) if tri_state["nmda_trace"] is not None else None
        self.state.stored_dg_pattern = tri_state["stored_dg_pattern"].to(self.device) if tri_state["stored_dg_pattern"] is not None else None
        self.state.ffi_strength = tri_state["ffi_strength"]

        # 3. RESTORE LEARNING STATE (STP)
        learning_state = state["learning_state"]
        if "stp_mossy" in learning_state and self.stp_mossy is not None:
            self.stp_mossy.load_state(learning_state["stp_mossy"])
        if "stp_schaffer" in learning_state and self.stp_schaffer is not None:
            self.stp_schaffer.load_state(learning_state["stp_schaffer"])
        if "stp_ec_ca1" in learning_state and self.stp_ec_ca1 is not None:
            self.stp_ec_ca1.load_state(learning_state["stp_ec_ca1"])
        if "stp_ca3_recurrent" in learning_state and self.stp_ca3_recurrent is not None:
            self.stp_ca3_recurrent.load_state(learning_state["stp_ca3_recurrent"])

        # 4. RESTORE OSCILLATOR STATE
        # Note: Oscillator phases now come from brain broadcast via set_oscillator_phases().
        # We only restore replay_engine state (which still needs migration separately).
        oscillator_state = state["oscillator_state"]
        if "replay_engine" in oscillator_state and self.replay_engine is not None:
            self.replay_engine.load_state(oscillator_state["replay_engine"])

        # 5. RESTORE NEUROMODULATOR STATE
        neuromodulator_state = state["neuromodulator_state"]
        self.state.dopamine = neuromodulator_state["dopamine"]
        self.state.acetylcholine = neuromodulator_state["acetylcholine"]
        self.state.norepinephrine = neuromodulator_state["norepinephrine"]

        # 6. RESTORE EPISODIC MEMORY
        from thalia.regions.hippocampus.config import Episode  # Import here to avoid circular dependency
        self.episode_buffer = []
        for ep_state in state["episode_buffer"]:
            episode = Episode(
                state=ep_state["state"].to(self.device),
                context=ep_state["context"].to(self.device) if ep_state["context"] is not None else None,
                action=ep_state["action"],
                reward=ep_state["reward"],
                correct=ep_state["correct"],
                metadata=ep_state["metadata"],
                priority=ep_state["priority"],
                timestamp=ep_state["timestamp"],
                sequence=[s.to(self.device) for s in ep_state["sequence"]] if ep_state["sequence"] is not None else None,
            )
            self.episode_buffer.append(episode)

    # endregion
