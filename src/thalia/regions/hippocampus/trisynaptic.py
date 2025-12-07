"""
Trisynaptic Hippocampus with biologically-accurate DG→CA3→CA1 circuit.

This implements the classic hippocampal circuit:
- Dentate Gyrus (DG): Pattern SEPARATION via very sparse coding (~2-5% active)
- CA3: Pattern COMPLETION via recurrent connections (autoassociative memory)
- CA1: Output/comparison layer that detects match between memory and input

Key biological features:
1. THETA MODULATION: 6-10 Hz oscillations separate encoding from retrieval
   - Theta trough: Encoding phase (CA3 learning enabled)
   - Theta peak: Retrieval phase (NMDA comparison enabled)

2. FEEDFORWARD INHIBITION: Stimulus onset triggers transient inhibition
   - Naturally clears residual activity (no explicit resets!)
   - Fast-spiking interneuron-like dynamics

3. CONTINUOUS DYNAMICS: Everything flows, no artificial resets
   - Membrane potentials decay naturally via LIF dynamics
   - Theta phase advances continuously
   - Activity transitions smoothly between phases

All processing is spike-based (no rate accumulation).

References:
- Marr (1971): Simple memory model
- Treves & Rolls (1994): Pattern separation in DG
- Hasselmo et al. (2002): Theta rhythm and encoding/retrieval
- Colgin (2013): Theta-gamma coupling in hippocampus
"""

import math
from typing import Optional, Dict, Any, List, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.core.neuron import LIFNeuron, LIFConfig
from thalia.core.stp import ShortTermPlasticity, STPConfig
from thalia.core.utils import ensure_batch_dim, clamp_weights, cosine_similarity_safe
from thalia.core.traces import update_trace
from thalia.core.diagnostics_mixin import DiagnosticsMixin
from thalia.core.weight_init import WeightInitializer
from thalia.regions.base import BrainRegion, LearningRule
from thalia.regions.theta_dynamics import TrialPhase, FeedforwardInhibition
from thalia.regions.gamma_dynamics import GammaOscillator, ThetaGammaConfig
from .config import Episode, TrisynapticConfig, TrisynapticState


class TrisynapticHippocampus(DiagnosticsMixin, BrainRegion):
    """
    Biologically-accurate hippocampus with DG→CA3→CA1 trisynaptic circuit.

    Architecture:
    ```
    Input (from cortex)
           │
           ▼
    ┌──────────────┐
    │ Dentate Gyrus│  Very sparse (~2%), random projections
    │   (DG)       │  Pattern SEPARATION: similar inputs → different codes
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │     CA3      │  Recurrent connections
    │              │  Pattern COMPLETION: partial cue → full pattern
    └──────┬───────┘
           │ ◄──────── (recurrent loop back to CA3)
           ▼
    ┌──────────────┐
    │     CA1      │  Output layer with comparison
    │              │  COINCIDENCE DETECTION: match vs mismatch
    └──────────────┘
           │
           ▼
    Output (to cortex/striatum)
    ```

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
    
    From BrainRegion (abstract base):
        - forward(input, **kwargs) → Tensor [must implement]
        - reset_state() → None
        - get_diagnostics() → Dict
        - set_dopamine(level) → None
        - Neuromodulator control methods
    
    See Also:
        docs/patterns/mixins.md for detailed mixin patterns
    """

    def __init__(self, config: TrisynapticConfig):
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

        # Override weights with trisynaptic circuit weights
        self._init_circuit_weights()

        # Create LIF neurons for each layer
        # CA3 gets spike-frequency adaptation to prevent frozen attractors
        lif_config = LIFConfig(tau_mem=20.0, v_threshold=1.0)
        ca3_lif_config = LIFConfig(
            tau_mem=20.0,
            v_threshold=1.0,
            adapt_increment=config.ca3_adapt_increment,  # SFA enabled!
            tau_adapt=config.ca3_adapt_tau,
        )
        self.dg_neurons = LIFNeuron(self.dg_size, lif_config)
        self.ca3_neurons = LIFNeuron(self.ca3_size, ca3_lif_config)
        self.ca1_neurons = LIFNeuron(self.ca1_size, lif_config)

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
            self.stp_mossy = ShortTermPlasticity(
                n_pre=self.dg_size,
                n_post=self.ca3_size,
                config=STPConfig.from_type(config.stp_mossy_type, dt=1.0),
                per_synapse=True,
            )
            self.stp_mossy.to(device)

            # Schaffer Collaterals (CA3→CA1): Depression
            # High-frequency CA3 activity depresses CA1 input - allows novelty
            # detection (novel patterns don't suffer from adaptation)
            self.stp_schaffer = ShortTermPlasticity(
                n_pre=self.ca3_size,
                n_post=self.ca1_size,
                config=STPConfig.from_type(config.stp_schaffer_type, dt=1.0),
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
                config=STPConfig.from_type(config.stp_ec_ca1_type, dt=1.0),
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
                config=STPConfig.from_type(config.stp_ca3_recurrent_type, dt=1.0),
                per_synapse=True,
            )
            self.stp_ca3_recurrent.to(device)
        else:
            self.stp_mossy = None
            self.stp_schaffer = None
            self.stp_ec_ca1 = None
            self.stp_ca3_recurrent = None

        # Episode buffer for sleep consolidation
        self.episode_buffer: List[Episode] = []

        # Feedback inhibition state - tracks recent CA3 activity
        self._ca3_activity_trace: Optional[torch.Tensor] = None

        # Track phase for theta-phase resets
        self._last_phase: Optional[TrialPhase] = None

        # Intrinsic plasticity: per-neuron threshold adjustment
        # Neurons that fire too much get higher thresholds (less excitable)
        # This operates on LONGER timescales than SFA
        self._ca3_threshold_offset: Optional[torch.Tensor] = None
        self._ca3_activity_history: Optional[torch.Tensor] = None  # EMA of firing rate

        # =====================================================================
        # THETA-GAMMA COUPLING
        # =====================================================================
        # Initialize gamma oscillator for sequence encoding
        if config.theta_gamma_enabled:
            gamma_config = ThetaGammaConfig(
                theta_freq_hz=8.0,  # Standard theta
                gamma_freq_hz=config.gamma_freq_hz,
                n_slots=config.gamma_n_slots,
                coupling_strength=config.gamma_coupling_strength,
            )
            self.gamma_oscillator = GammaOscillator(gamma_config)

            # Replay engine for sequence replay (lazy import to avoid circular dependency)
            from thalia.memory.replay_engine import ReplayEngine, ReplayConfig, ReplayMode
            replay_config = ReplayConfig(
                compression_factor=5.0,
                theta_gamma_config=gamma_config,
                mode=ReplayMode.SEQUENCE,
                apply_gating=True,
                pattern_completion=True,
            )
            self.replay_engine = ReplayEngine(replay_config)

            # Assign CA3 neurons to gamma slots for phase coding
            # Neurons in different slots fire at different gamma phases
            self._ca3_slot_assignment = torch.arange(self.ca3_size) % config.gamma_n_slots
        else:
            self.gamma_oscillator = None
            self.replay_engine = None
            self._ca3_slot_assignment = None

        # Track current sequence position for encoding (auto-advances)
        self._sequence_position: int = 0

        # Theta-driven reset: when True, reset happens at next theta trough
        # This replaces hard resets with biologically-realistic theta-aligned resets
        self._pending_theta_reset: bool = False

        # State
        self.state = TrisynapticState()

    def _get_learning_rule(self) -> LearningRule:
        return LearningRule.THETA_PHASE

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

        # DG → CA3: Random but less sparse
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
        if self.gamma_oscillator is not None:
            self.gamma_oscillator.set_to_slot(0)

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

        # Reset phase tracking
        self._last_phase = None

        batch_size = 1
        self.state = TrisynapticState(
            dg_spikes=torch.zeros(batch_size, self.dg_size, device=device),
            ca3_spikes=torch.zeros(batch_size, self.ca3_size, device=device),
            ca1_spikes=torch.zeros(batch_size, self.ca1_size, device=device),
            ca3_membrane=torch.zeros(batch_size, self.ca3_size, device=device),
            ca3_persistent=torch.zeros(batch_size, self.ca3_size, device=device),
            sample_trace=None,  # Set during sample encoding
            dg_trace=torch.zeros(batch_size, self.dg_size, device=device),
            ca3_trace=torch.zeros(batch_size, self.ca3_size, device=device),
            nmda_trace=torch.zeros(batch_size, self.ca1_size, device=device),
            stored_dg_pattern=None,  # Set during sample phase
            ffi_strength=0.0,
        )

    def forward(  # type: ignore[override]
        self,
        input_spikes: torch.Tensor,
        phase: TrialPhase,
        encoding_mod: float = 1.0,
        retrieval_mod: float = 1.0,
        dt: float = 1.0,
        ec_direct_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process input spikes through DG→CA3→CA1 circuit.

        Note: This method has a different signature than the base class
        because hippocampus requires explicit trial phase for theta modulation.

        Args:
            input_spikes: Input spike pattern [batch, n_input] for trisynaptic path
            phase: Trial phase (ENCODE, DELAY, or RETRIEVE)
            encoding_mod: Theta modulation for encoding (from BrainSystem)
            retrieval_mod: Theta modulation for retrieval (from BrainSystem)
            dt: Time step in ms
            ec_direct_input: Optional separate input for EC→CA1 direct pathway.
                            If None, uses input_spikes (original behavior).
                            When provided, this models EC layer III input which
                            carries raw sensory information to CA1 for comparison.

        Returns:
            CA1 output spikes [batch, n_output]

        Phase Logic:
            - ENCODE: DG→CA3 encoding, CA3 Hebbian learning, EC→CA1 learning
            - DELAY: CA3 recurrence maintains memory, CA1 decays naturally
            - RETRIEVE: NMDA coincidence detection in CA1

        Features:
            - Theta modulation: Encoding/retrieval strength passed from BrainSystem
            - Feedforward inhibition: Stimulus changes trigger transient inhibition
            - EC layer III: Optional separate input for direct EC→CA1 (biologically
              accurate - EC L3 carries raw sensory info, EC L2 goes through DG)
        """
        input_spikes = ensure_batch_dim(input_spikes)

        batch_size = input_spikes.shape[0]

        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        assert input_spikes.shape[-1] == self.tri_config.n_input, (
            f"TrisynapticHippocampus.forward: input_spikes has shape {input_spikes.shape} "
            f"but n_input={self.tri_config.n_input}. Check that cortex output matches hippocampus input."
        )
        if ec_direct_input is not None:
            expected_ec_size = self._ec_l3_input_size if self._ec_l3_input_size > 0 else self.tri_config.n_input
            assert ec_direct_input.shape[-1] == expected_ec_size, (
                f"TrisynapticHippocampus.forward: ec_direct_input has shape {ec_direct_input.shape} "
                f"but expected size={expected_ec_size} (ec_l3_input_size={self._ec_l3_input_size}). "
                f"Check that sensory input matches EC L3 pathway configuration."
            )
            assert ec_direct_input.shape[0] == batch_size, (
                f"TrisynapticHippocampus.forward: ec_direct_input batch size {ec_direct_input.shape[0]} "
                f"doesn't match input_spikes batch size {batch_size}."
            )

        # Ensure state is initialized
        if self.state.dg_spikes is None:
            self._init_state()

        # =====================================================================
        # THETA-PHASE RESET (prevents frozen attractors + new_trial reset)
        # =====================================================================
        # When transitioning to ENCODE phase (theta trough):
        # 1. Partially reset persistent activity to prevent stale attractors
        # 2. If _pending_theta_reset is set (from new_trial), do full reset
        #
        # This is biologically realistic: theta rhythm naturally segments
        # sequences, and new_trial() just schedules reset for next theta trough.
        at_theta_trough = (phase == TrialPhase.ENCODE and self._last_phase != TrialPhase.ENCODE)

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

        self._last_phase = phase

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
        dg_code = torch.matmul(input_spikes.float(), self.w_ec_dg.t())

        # Apply FFI: reduce DG drive when input changes significantly
        # ffi_strength is now normalized to [0, 1]
        ffi_factor = 1.0 - self.state.ffi_strength * 0.5
        dg_code = dg_code * ffi_factor

        # Run through DG neurons
        dg_spikes, _ = self.dg_neurons(dg_code)

        # Apply extreme winner-take-all sparsity
        # Note: cast needed because hasattr check doesn't narrow type for Pylance
        membrane_v = getattr(self.dg_neurons, 'v', None)
        dg_spikes = self._apply_wta_sparsity(
            dg_spikes,
            self.tri_config.dg_sparsity,
            membrane_v if isinstance(membrane_v, torch.Tensor) else None,
        )
        self.state.dg_spikes = dg_spikes

        # Inter-stage shape check: DG output → CA3 input
        assert dg_spikes.shape == (batch_size, self.dg_size), (
            f"TrisynapticHippocampus: DG spikes have shape {dg_spikes.shape} "
            f"but expected ({batch_size}, {self.dg_size}). "
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

        # Compute theta-gated strengths
        # encoding_mod: high at theta trough (0°), low at peak (180°)
        # retrieval_mod: low at theta trough, high at theta peak

        # Feedforward gate: Strong during encoding, FULLY suppressed during retrieval
        # Use encoding_mod directly - when it's low (retrieval phase), gate is low
        ff_gate = encoding_mod  # 0.0 to 1.0 based on theta phase

        # Recurrence gate: Weak during encoding, strong during retrieval
        # Use retrieval_mod directly - when it's high, recurrence is strong
        rec_gate = 0.2 + 0.8 * retrieval_mod  # Range: 0.2 (encoding) to 1.0 (retrieval)

        # Apply theta gating based on phase for additional control
        if phase == TrialPhase.RETRIEVE:
            # During explicit RETRIEVE phase, ensure feedforward is fully gated
            ff_gate = min(ff_gate, 0.05)  # Cap at 5% even if theta says otherwise
            rec_gate = max(rec_gate, 0.9)  # Ensure strong recurrence
        elif phase == TrialPhase.DELAY:
            # During DELAY, CA3 should maintain pattern via moderate recurrence
            ff_gate = ff_gate * 0.3  # Reduce feedforward during delay too
            rec_gate = max(rec_gate, 0.6)  # Moderate recurrence to maintain

        # Feedforward from DG (theta-gated) with optional STP
        if self.stp_mossy is not None:
            # Get STP efficacy for mossy fiber synapses
            # Mossy fibers are FACILITATING - repeated DG spikes progressively
            # enhance transmission to CA3
            stp_efficacy = self.stp_mossy(dg_spikes.float()).squeeze(0)
            # Apply STP to weights: (n_post, n_pre) * (n_pre, n_post).T
            effective_w_dg_ca3 = self.w_dg_ca3 * stp_efficacy.T
            ca3_ff = torch.matmul(dg_spikes.float(), effective_w_dg_ca3.t()) * ff_gate
        else:
            ca3_ff = torch.matmul(dg_spikes.float(), self.w_dg_ca3.t()) * ff_gate

        # =====================================================================
        # RECURRENT CA3 WITH STP (CRITICAL FOR PREVENTING FROZEN ATTRACTORS)
        # =====================================================================
        # Without STP, recurrent connections cause the same neurons to fire
        # every timestep (frozen attractor). With DEPRESSING STP, frequently-
        # used synapses get temporarily weaker, allowing pattern transitions.
        if self.stp_ca3_recurrent is not None and self.state.ca3_spikes is not None:
            # Get STP efficacy for CA3 recurrent synapses
            # CA3 recurrent is DEPRESSING - prevents frozen attractors
            stp_rec_efficacy = self.stp_ca3_recurrent(
                self.state.ca3_spikes.float()
            ).squeeze(0)
            # Apply STP to recurrent weights
            effective_w_ca3_ca3 = self.w_ca3_ca3 * stp_rec_efficacy.T
            ca3_rec = torch.matmul(
                self.state.ca3_spikes.float(),
                effective_w_ca3_ca3.t()
            ) * self.tri_config.ca3_recurrent_strength * rec_gate
        else:
            # Recurrent from previous CA3 activity (theta-gated)
            ca3_rec = torch.matmul(
                self.state.ca3_spikes.float() if self.state.ca3_spikes is not None else torch.zeros(1, self.ca3_size, device=input_spikes.device),
                self.w_ca3_ca3.t()
            ) * self.tri_config.ca3_recurrent_strength * rec_gate

        # =====================================================================
        # ACTIVITY-DEPENDENT FEEDBACK INHIBITION
        # =====================================================================
        # When many CA3 neurons are active, feedback inhibition increases.
        # This prevents runaway activity and frozen attractors.
        if self._ca3_activity_trace is None:
            self._ca3_activity_trace = torch.zeros(1, device=input_spikes.device)

        # Update activity trace (exponential moving average of total CA3 activity)
        if self.state.ca3_spikes is not None:
            current_activity = self.state.ca3_spikes.sum()
            self._ca3_activity_trace = 0.9 * self._ca3_activity_trace + 0.1 * current_activity

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
            self.state.ca3_persistent = torch.zeros(1, self.ca3_size, device=input_spikes.device)

        ca3_persistent_input: torch.Tensor = (
            self.state.ca3_persistent * self.tri_config.ca3_persistent_gain
        )

        # Total CA3 input = feedforward + recurrent + persistent - inhibition
        ca3_input = ca3_ff + ca3_rec + ca3_persistent_input - feedback_inhibition

        # INTRINSIC PLASTICITY: Apply per-neuron threshold offset
        # Neurons that fire too much have higher thresholds (less excitable)
        if (self.tri_config.intrinsic_plasticity_enabled and
            self._ca3_threshold_offset is not None):
            ca3_input = ca3_input - self._ca3_threshold_offset.unsqueeze(0)

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
        # - "time": Slot from oscillator phase (timing-based, used in replay)
        if self.gamma_oscillator is not None and self._ca3_slot_assignment is not None:
            cfg = self.tri_config

            # Advance gamma oscillator
            self.gamma_oscillator.advance(float(dt))

            # Get current slot based on mode
            if cfg.gamma_slot_mode == "time":
                # Time-based: slot from oscillator phase (used during replay)
                current_slot = self.gamma_oscillator.current_slot
            else:  # "item" mode (default)
                # Item-based: slot from sequence position
                current_slot = self._sequence_position % cfg.gamma_n_slots

            # Create gating mask based on slot assignment
            # Neurons in current slot get full input, others are suppressed
            slot_match = (self._ca3_slot_assignment == current_slot).float()
            slot_match = slot_match.to(ca3_input.device).unsqueeze(0)

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
            gamma_amplitude = self.gamma_oscillator.gamma_amplitude  # [0, 1]

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

        # Run through CA3 neurons
        ca3_spikes, _ = self.ca3_neurons(ca3_input)

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
            print(f"      [HIPPO FWD] phase={phase.name}, input_sum={input_sum:.0f}, "
                  f"dg_active={dg_active_count:.0f}, ca3_active={len(ca3_active_indices)}, "
                  f"ff_gate={ff_gate:.3f}, rec_gate={rec_gate:.3f}")
            if len(ca3_active_indices) <= 10:
                print(f"        CA3 indices: {ca3_active_indices}")
            else:
                print(f"        CA3 indices (first 10): {ca3_active_indices[:10]}...")

        # Inter-stage shape check: CA3 output → CA1 input
        assert ca3_spikes.shape == (batch_size, self.ca3_size), (
            f"TrisynapticHippocampus: CA3 spikes have shape {ca3_spikes.shape} "
            f"but expected ({batch_size}, {self.ca3_size}). "
            f"Check CA3 sparsity or DG→CA3 weights shape."
        )

        # Update persistent activity AFTER computing new spikes
        # The trace accumulates spike activity with slow decay
        # Using a direct accumulation: trace += spike - decay*trace
        # This ensures spikes have strong immediate effect but decay slowly
        decay_rate = dt / self.tri_config.ca3_persistent_tau

        # Only update persistent during ENCODE phase - freeze during DELAY/RETRIEVE
        # This is biologically motivated: Ca²⁺-dependent currents take time to build
        # up during active encoding, then provide sustained drive during maintenance
        if phase == TrialPhase.ENCODE:
            self.state.ca3_persistent = (
                self.state.ca3_persistent * (1.0 - decay_rate) +
                ca3_spikes.float() * 0.5  # Strong spike contribution during encoding
            )
        else:
            # During DELAY and RETRIEVE, just decay (don't update from new spikes)
            # This preserves the encoded pattern as the dominant memory trace
            self.state.ca3_persistent = self.state.ca3_persistent * (1.0 - decay_rate * 0.5)

        # Clamp to prevent runaway
        self.state.ca3_persistent = torch.clamp(self.state.ca3_persistent, 0.0, 3.0)

        # =====================================================================
        # ENCODE PHASE: Apply fast Hebbian learning to CA3 recurrent weights
        # This is how the hippocampus "stores" the pattern - in the weights!
        # Also store the DG pattern for later match/mismatch detection.
        # Modulated by theta encoding strength!
        # =====================================================================
        if phase == TrialPhase.ENCODE:
            # Store the DG pattern (accumulate over timesteps)
            if self.state.stored_dg_pattern is None:
                self.state.stored_dg_pattern = dg_spikes.float().clone()
            else:
                self.state.stored_dg_pattern = self.state.stored_dg_pattern + dg_spikes.float()

            # One-shot Hebbian: strengthen connections between co-active neurons
            # Learning rate modulated by theta phase AND gamma amplitude
            ca3_activity = ca3_spikes.float().squeeze()
            if ca3_activity.sum() > 0:
                # Hebbian outer product: neurons that fire together wire together
                #
                # Gamma amplitude modulation: Learning is stronger when gamma
                # is strong (theta trough, encoding phase). This implements
                # the biological finding that synaptic plasticity is enhanced
                # during periods of strong gamma oscillations.
                base_lr = self.tri_config.learning_rate * encoding_mod

                # Apply gamma amplitude modulation if available
                if self.gamma_oscillator is not None:
                    gamma_mod = self.gamma_oscillator.gamma_amplitude
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

        # Feedforward from CA3 (retrieved/encoded memory) with optional STP
        # Schaffer collaterals are DEPRESSING - high-frequency CA3 activity
        # causes progressively weaker transmission to CA1
        if self.stp_schaffer is not None:
            stp_efficacy = self.stp_schaffer(ca3_spikes.float()).squeeze(0)
            effective_w_ca3_ca1 = self.w_ca3_ca1 * stp_efficacy.T
            ca1_from_ca3 = torch.matmul(ca3_spikes.float(), effective_w_ca3_ca1.t())
        else:
            ca1_from_ca3 = torch.matmul(ca3_spikes.float(), self.w_ca3_ca1.t())

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
                stp_efficacy = self.stp_ec_ca1(ec_direct_input.float()).squeeze(0)
                # Note: EC L3 may have different size, need to handle
                effective_w = self.w_ec_l3_ca1 * stp_efficacy.T
                ca1_from_ec = torch.matmul(ec_direct_input.float(), effective_w.t())
            else:
                ca1_from_ec = torch.matmul(ec_direct_input.float(), self.w_ec_l3_ca1.t())
            ec_input_for_ca1 = ec_direct_input
        elif ec_direct_input is not None:
            # ec_direct_input provided but same size as EC L2
            if self.stp_ec_ca1 is not None:
                stp_efficacy = self.stp_ec_ca1(ec_direct_input.float()).squeeze(0)
                effective_w = self.w_ec_ca1 * stp_efficacy.T
                ca1_from_ec = torch.matmul(ec_direct_input.float(), effective_w.t())
            else:
                ca1_from_ec = torch.matmul(ec_direct_input.float(), self.w_ec_ca1.t())
            ec_input_for_ca1 = ec_direct_input
        else:
            # Fall back to input_spikes (original behavior)
            # Note: STP for EC→CA1 is sized for ec_l3_input_size, not n_input,
            # so we don't apply STP when falling back to input_spikes
            ca1_from_ec = torch.matmul(input_spikes.float(), self.w_ec_ca1.t())
            ec_input_for_ca1 = input_spikes

        # Apply feedforward inhibition: strong input change reduces CA1 drive
        # This clears residual activity naturally (no explicit reset!)
        # ffi_strength is now normalized to [0, 1], so 0.8 gives max 80% suppression
        ffi_factor = 1.0 - self.state.ffi_strength * 0.8
        ca1_from_ec = ca1_from_ec * ffi_factor
        ca1_from_ca3 = ca1_from_ca3 * ffi_factor

        if phase == TrialPhase.DELAY:
            # -------------------------------------------------------------
            # DELAY PHASE: CA3 maintains memory, CA1 idles
            # -------------------------------------------------------------
            # During the delay period between sample and test:
            # - CA3 activity is maintained by recurrence (handled above)
            # - CA1 receives minimal input and just lets membrane decay
            # - No learning happens
            # - This simulates the biological delay where no comparison occurs

            # Minimal CA1 input: just enough to keep neurons "warm"
            ca1_input = ca1_from_ca3 * 0.1  # Very weak CA3→CA1 drive

            # Lateral inhibition (keeps activity sparse)
            if self.state.ca1_spikes is not None:
                ca1_inhib = torch.matmul(
                    self.state.ca1_spikes.float(),
                    self.w_ca1_inhib.t()
                )
                ca1_input = ca1_input - ca1_inhib

            # Run through CA1 neurons - membrane will naturally decay
            ca1_spikes, _ = self.ca1_neurons(F.relu(ca1_input))

            # Light sparsity during delay
            ca1_spikes = self._apply_wta_sparsity(
                ca1_spikes,
                cfg.ca1_sparsity * 0.5,  # Allow even sparser activity
                self.ca1_neurons.membrane if hasattr(self.ca1_neurons, 'membrane') else None,
            )

        elif phase == TrialPhase.ENCODE:
            # -------------------------------------------------------------
            # ENCODE PHASE: CA1 driven by CA3, EC→CA1 learns
            # -------------------------------------------------------------
            # During encoding, CA3 determines which CA1 neurons fire.
            # We strengthen EC→CA1 connections for active EC→CA1 pairs.

            # CA1 input dominated by CA3 during sample (memory encoding)
            ca1_input = ca1_from_ca3 + ca1_from_ec * cfg.ampa_ratio

            # Lateral inhibition
            if self.state.ca1_spikes is not None:
                ca1_inhib = torch.matmul(
                    self.state.ca1_spikes.float(),
                    self.w_ca1_inhib.t()
                )
                ca1_input = ca1_input - ca1_inhib

            # Run through CA1 neurons
            ca1_spikes, _ = self.ca1_neurons(F.relu(ca1_input))

            # Apply sparsity
            ca1_membrane_v = getattr(self.ca1_neurons, 'v', None)
            ca1_spikes = self._apply_wta_sparsity(
                ca1_spikes,
                cfg.ca1_sparsity,
                ca1_membrane_v if isinstance(ca1_membrane_v, torch.Tensor) else None,
            )

            # ---------------------------------------------------------
            # HEBBIAN LEARNING: EC→CA1 plasticity (modulated by theta + gamma)
            # ---------------------------------------------------------
            # Strengthen connections: active EC neurons → active CA1 neurons
            # This aligns the direct pathway with the indirect pathway
            # Learning modulated by both theta phase and gamma amplitude
            ec_activity = ec_input_for_ca1.float().squeeze()
            ca1_activity = ca1_spikes.float().squeeze()

            if ec_activity.sum() > 0 and ca1_activity.sum() > 0:
                # Hebbian outer product: w_ij += lr * post_j * pre_i
                base_lr = cfg.ec_ca1_learning_rate * encoding_mod

                # Apply gamma amplitude modulation if available
                if self.gamma_oscillator is not None:
                    gamma_mod = self.gamma_oscillator.gamma_amplitude
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

        else:  # phase == TrialPhase.RETRIEVE
            # -------------------------------------------------------------
            # RETRIEVE PHASE: NMDA coincidence detection
            # -------------------------------------------------------------
            # Now EC→CA1 is aligned (from sample phase learning).
            # NMDA gating detects if current EC input matches recalled CA3.
            # NMDA effectiveness modulated by theta retrieval strength!

            # Mg²⁺ block removal based on INSTANTANEOUS CA3-induced depolarization
            # The trace provides some temporal smoothing but shouldn't accumulate
            # indefinitely. We use a bounded leaky integrator that approaches
            # equilibrium at the current input level.
            if self.state.nmda_trace is not None:
                # Decay toward 0, then add current input (bounded integration)
                nmda_decay = torch.exp(torch.tensor(-dt / cfg.nmda_tau))
                # Use weighted average instead of pure accumulation
                self.state.nmda_trace = self.state.nmda_trace * nmda_decay + ca1_from_ca3 * (1.0 - nmda_decay)
            else:
                self.state.nmda_trace = ca1_from_ca3.clone()

            # Per-neuron Mg²⁺ block removal based on CA3-induced depolarization
            # Modulated by theta retrieval strength
            mg_block_removal = torch.sigmoid(
                (self.state.nmda_trace - cfg.nmda_threshold) * cfg.nmda_steepness
            ) * retrieval_mod

            # NMDA current: EC→CA1 gated by CA3-induced depolarization
            # High when BOTH pathways target the same CA1 neurons
            nmda_current = ca1_from_ec * mg_block_removal

            # AMPA current: fast, ungated baseline
            ampa_current = ca1_from_ec * cfg.ampa_ratio

            # Total CA1 input
            ca1_input = ampa_current + nmda_current

            # Lateral inhibition
            if self.state.ca1_spikes is not None:
                ca1_inhib = torch.matmul(
                    self.state.ca1_spikes.float(),
                    self.w_ca1_inhib.t()
                )
                ca1_input = ca1_input - ca1_inhib

            # Run through CA1 neurons
            ca1_spikes, _ = self.ca1_neurons(F.relu(ca1_input))

            # In test phase, DON'T apply WTA sparsity!
            # The actual spike count should reflect NMDA gating strength.
            # Match: high NMDA → many neurons above threshold → many spikes
            # Mismatch: low NMDA → few neurons above threshold → few spikes
            # This is the key discrimination signal!

        self.state.ca1_spikes = ca1_spikes

        # =====================================================================
        # Update STDP Traces (for learning, not comparison)
        # =====================================================================
        if self.state.dg_trace is not None:
            update_trace(self.state.dg_trace, dg_spikes, tau=self.tri_config.stdp_tau_plus, dt=dt)
        if self.state.ca3_trace is not None:
            update_trace(self.state.ca3_trace, ca3_spikes, tau=self.tri_config.stdp_tau_plus, dt=dt)

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
        if self.gamma_oscillator is not None:
            self._sequence_position += 1
            # The gamma oscillator already advanced in the CA3 section via
            # gamma_oscillator.advance(dt), so we just track position here.
            # The position is used for diagnostics and can be reset by new_trial().

        return ca1_spikes

    def _apply_wta_sparsity(
        self,
        spikes: torch.Tensor,
        target_sparsity: float,
        membrane: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply winner-take-all sparsity to enforce sparse coding.

        Only the top-k neurons (by membrane potential) are allowed to spike.
        """
        batch_size, n_neurons = spikes.shape
        k = max(1, int(n_neurons * target_sparsity))

        sparse_spikes = torch.zeros_like(spikes)

        for b in range(batch_size):
            active = spikes[b].nonzero(as_tuple=True)[0]

            if len(active) <= k:
                sparse_spikes[b] = spikes[b]
            elif membrane is not None:
                # Keep top-k by membrane potential
                active_v = membrane[b, active]
                _, top_k_idx = torch.topk(active_v, k)
                sparse_spikes[b, active[top_k_idx]] = 1.0
            else:
                # Random selection if no membrane
                perm = torch.randperm(len(active))[:k]
                sparse_spikes[b, active[perm]] = 1.0

        return sparse_spikes

    def clear_sample_trace(self) -> None:
        """Reset for new trial.

        Note: We don't clear CA3 weights - the "memory" is in the weights!
        We only reset the transient state (spikes, traces).
        """
        self.state.sample_trace = None
        # Reset spike states
        self.state.dg_spikes = None
        self.state.ca3_spikes = None
        self.state.ca1_spikes = None
        self.state.dg_trace = None
        self.state.ca3_trace = None

    def _apply_plasticity(
        self,
        _input_spikes: torch.Tensor,
        _output_spikes: torch.Tensor,
        _dt: float = 1.0,
    ) -> None:
        """
        Apply continuous STDP learning to circuit weights.

        Called automatically at each forward() timestep.
        CA3 recurrent connections learn via STDP to create attractors.
        Learning rate is modulated by dopamine (via get_effective_learning_rate).

        Note: _input_spikes, _output_spikes, and _dt are required by the base class
        signature but not used here since we access spikes from self.state.
        """
        if not self.plasticity_enabled:
            return

        cfg = self.tri_config

        # Decay neuromodulators (ACh/NE decay locally, dopamine set by Brain)
        self.decay_neuromodulators(dt_ms=_dt)

        # Get dopamine-modulated learning rate
        effective_lr = self.get_effective_learning_rate(cfg.ca3_learning_rate)
        if effective_lr < 1e-8:
            return

        # CA3 recurrent STDP: strengthen connections between co-active neurons
        if self.state.ca3_trace is not None and self.state.ca3_spikes is not None:
            # LTP: post spike when pre trace is high
            ca3_post = self.state.ca3_spikes.float().squeeze()
            ca3_trace = self.state.ca3_trace.squeeze()

            ltp = torch.outer(ca3_post, ca3_trace)
            ltd = torch.outer(ca3_trace, ca3_post) * 0.5  # Weaker LTD

            dW = effective_lr * (ltp - ltd)

            # DEBUG: Check weight update
            dW_abs_mean = dW.abs().mean().item()
            w_before = self.w_ca3_ca3.data.abs().mean().item()

            with torch.no_grad():
                self.w_ca3_ca3.data += dW
                self.w_ca3_ca3.data.fill_diagonal_(0.0)  # No self-connections
                clamp_weights(self.w_ca3_ca3.data, cfg.w_min, cfg.w_max)

                # =============================================================
                # SYNAPTIC SCALING (Homeostatic)
                # =============================================================
                # Multiplicatively adjust all weights towards target mean.
                # This prevents runaway LTP from causing weight explosion.
                if cfg.synaptic_scaling_enabled:
                    mean_weight = self.w_ca3_ca3.data.mean()
                    scaling = 1.0 + cfg.synaptic_scaling_rate * (cfg.synaptic_scaling_target - mean_weight)
                    self.w_ca3_ca3.data *= scaling
                    self.w_ca3_ca3.data.fill_diagonal_(0.0)  # Maintain no self-connections
                    clamp_weights(self.w_ca3_ca3.data, cfg.w_min, cfg.w_max)

            w_after = self.w_ca3_ca3.data.abs().mean().item()
            if dW_abs_mean > 1e-6:
                print(f"      [CA3 STDP] dW_mean={dW_abs_mean:.6f}, w_before={w_before:.6f}, w_after={w_after:.6f}, eff_lr={effective_lr:.4f}")

        # =====================================================================
        # INTRINSIC PLASTICITY
        # =====================================================================
        # Update per-neuron threshold offsets based on firing history.
        # This operates on LONGER timescales than SFA.
        if cfg.intrinsic_plasticity_enabled and self.state.ca3_spikes is not None:
            ca3_spikes_1d = self.state.ca3_spikes.float().mean(dim=0)  # Average across batch

            # Initialize if needed
            if self._ca3_activity_history is None:
                self._ca3_activity_history = torch.zeros(self.ca3_size, device=ca3_spikes_1d.device)
            if self._ca3_threshold_offset is None:
                self._ca3_threshold_offset = torch.zeros(self.ca3_size, device=ca3_spikes_1d.device)

            # Update activity history (exponential moving average)
            self._ca3_activity_history = (
                0.99 * self._ca3_activity_history + 0.01 * ca3_spikes_1d
            )

            # Adjust threshold: high activity → higher threshold (less excitable)
            rate_error = self._ca3_activity_history - cfg.intrinsic_target_rate
            self._ca3_threshold_offset = (
                self._ca3_threshold_offset + cfg.intrinsic_adaptation_rate * rate_error
            ).clamp(-0.5, 0.5)  # Limit threshold adjustment range

    def get_state(self) -> TrisynapticState:
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
    ) -> None:
        """Store an episode in episodic memory for later replay.

        Priority is computed based on reward magnitude and correctness.

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
        """
        cfg = self.tri_config

        # Compute priority based on reward and correctness
        base_priority = 1.0 + abs(reward)
        if correct:
            base_priority += 0.5  # Boost for correct trials
        base_priority += priority_boost

        # Clone sequence tensors if provided
        sequence_cloned = None
        if sequence is not None:
            sequence_cloned = [s.clone().detach() for s in sequence]

        episode = Episode(
            state=state.clone().detach(),
            context=context.clone().detach() if context is not None else None,
            action=action,
            reward=reward,
            correct=correct,
            metadata=metadata,
            priority=base_priority,
            timestamp=len(self.episode_buffer),
            sequence=sequence_cloned,
        )

        # Buffer management: keep limited episodes
        max_episodes = getattr(cfg, 'max_episodes', 100)
        if len(self.episode_buffer) >= max_episodes:
            # Remove lowest priority
            min_idx = min(range(len(self.episode_buffer)),
                         key=lambda i: self.episode_buffer[i].priority)
            self.episode_buffer.pop(min_idx)

        self.episode_buffer.append(episode)

    def sample_episodes_prioritized(self, n: int) -> List[Episode]:
        """Sample episodes with probability proportional to priority."""
        if not self.episode_buffer:
            return []

        n = min(n, len(self.episode_buffer))
        priorities = torch.tensor([ep.priority for ep in self.episode_buffer])
        probs = priorities / priorities.sum()

        indices = torch.multinomial(probs, n, replacement=False)
        return [self.episode_buffer[i] for i in indices]

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
            raise RuntimeError("Replay engine not available. "
                               "Set theta_gamma_enabled=True in config.")

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
        def process_pattern(pattern: torch.Tensor) -> torch.Tensor:
            return self.forward(pattern, phase=TrialPhase.DELAY)

        # Gating function: apply gamma gating to patterns
        def get_gating(slot: int) -> float:
            return self._get_gamma_gating(slot)

        # Run replay through unified engine
        result = self.replay_engine.replay(
            episode=episode,
            pattern_processor=process_pattern,
            gating_fn=get_gating,
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
        if self.gamma_oscillator is None:
            return 1.0

        cfg = self.tri_config
        n_slots = cfg.gamma_n_slots

        # Compute phase difference from target slot
        target_phase = (slot / n_slots) * (2 * math.pi)
        current_gamma_phase = self.gamma_oscillator.gamma_phase

        # Gaussian gating around target phase
        phase_diff = abs(current_gamma_phase - target_phase)
        phase_diff = min(phase_diff, 2 * math.pi - phase_diff)  # Wrap around

        gating = math.exp(-phase_diff ** 2 / (2 * 0.5 ** 2))  # σ = 0.5 radians

        # Scale by gating strength
        return 1.0 - cfg.gamma_gating_strength * (1.0 - gating)

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

        # Weight statistics using weight_diagnostics helper
        weight_stats: Dict[str, Any] = {}
        weight_stats.update(self.weight_diagnostics(self.w_ec_dg.data, "ec_dg"))
        weight_stats.update(self.weight_diagnostics(self.w_dg_ca3.data, "dg_ca3"))
        weight_stats.update(self.weight_diagnostics(self.w_ca3_ca3.data, "ca3_ca3"))
        weight_stats.update(self.weight_diagnostics(self.w_ca3_ca1.data, "ca3_ca1"))
        weight_stats.update(self.weight_diagnostics(self.w_ec_ca1.data, "ec_ca1"))

        # Feedforward inhibition state
        ffi_state = {
            "current_strength": state.ffi_strength,
        }

        return {
            "region": "hippocampus",
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
            # Weights (now with full stats from mixin)
            "weight_stats": weight_stats,
            # Inhibition
            "ffi": ffi_state,
            # Episode buffer
            "episode_buffer_size": len(self.episode_buffer),
        }

    def get_pattern_similarity(self) -> Optional[float]:
        """Get similarity between stored and current DG patterns.

        This measures how well the current input matches the stored memory,
        providing an intrinsic reward signal for pattern completion.
        High similarity = successful recall = good memory system.

        Returns:
            Cosine similarity (0.0 to 1.0), or None if no stored pattern
        """
        if (self.state.stored_dg_pattern is None or
            self.state.dg_spikes is None):
            return None

        stored = self.state.stored_dg_pattern.float().squeeze()
        current = self.state.dg_spikes.float().squeeze()

        similarity = cosine_similarity_safe(stored, current)
        return similarity.item()

    # =========================================================================
    # THETA-GAMMA COUPLING METHODS
    # =========================================================================

    def set_sequence_position(self, position: int) -> None:
        """Manually override the current sequence position.

        NOTE: Position normally auto-advances on each forward() call.
        Use this only for special cases like:
        - Jumping to a specific position during replay
        - Testing specific gamma slots
        - Synchronizing with external position tracking

        For normal operation, position is implicit from arrival order.

        Args:
            position: Sequence position (0, 1, 2, ...)
        """
        self._sequence_position = position

        if self.gamma_oscillator is not None:
            cfg = self.tri_config
            # Set gamma to the slot for this position
            target_slot = position % cfg.gamma_n_slots
            self.gamma_oscillator.set_to_slot(target_slot)

    def sync_gamma_to_theta(self, theta_phase: float) -> None:
        """Synchronize gamma oscillator to external theta phase.

        Use this when you have a separate ThetaGenerator/ThetaState
        and want the gamma oscillator to stay in sync.

        Args:
            theta_phase: External theta phase in radians
        """
        if self.gamma_oscillator is not None:
            self.gamma_oscillator.sync_to_theta_phase(theta_phase)

    def get_current_gamma_slot(self) -> Optional[int]:
        """Get the current gamma slot (working memory position).

        Slot is determined by sequence position (item-based), not by
        time-based gamma phase. This ensures each item gets its own slot.

        Returns:
            Current slot index [0, n_slots-1], or None if gamma disabled
        """
        if self.gamma_oscillator is not None:
            return self._sequence_position % self.tri_config.gamma_n_slots
        return None

    def get_gamma_diagnostics(self) -> Dict[str, Any]:
        """Get gamma oscillator diagnostics.

        Returns:
            Dictionary with gamma state info including theta-modulated values
        """
        if self.gamma_oscillator is None:
            return {"enabled": False}

        cfg = self.tri_config
        item_slot = self._sequence_position % cfg.gamma_n_slots
        time_slot = self.gamma_oscillator.current_slot

        # Current slot depends on mode
        if cfg.gamma_slot_mode == "time":
            current_slot = time_slot
        else:
            current_slot = item_slot

        # Gamma amplitude is modulated by theta phase
        gamma_amplitude = self.gamma_oscillator.gamma_amplitude

        # Effective gating strength (theta-modulated)
        effective_gating = cfg.gamma_gating_strength * gamma_amplitude

        # Effective learning rate multiplier (theta+gamma modulated)
        # This is the factor applied to base learning rate: 0.5 + 0.5 * gamma_amplitude
        lr_multiplier = 0.5 + 0.5 * gamma_amplitude

        return {
            "enabled": True,
            "slot_mode": cfg.gamma_slot_mode,
            "theta_phase": self.gamma_oscillator.theta_phase,
            "gamma_phase": self.gamma_oscillator.gamma_phase,
            "time_based_slot": time_slot,  # From oscillator timing
            "item_based_slot": item_slot,  # From sequence position
            "current_slot": current_slot,  # Active slot based on mode
            "gamma_amplitude": gamma_amplitude,  # Theta-modulated (strong at trough)
            "effective_gating_strength": effective_gating,  # Actual gating applied
            "learning_rate_multiplier": lr_multiplier,  # LR scaling factor
            "n_slots": cfg.gamma_n_slots,
            "sequence_position": self._sequence_position,
        }

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
            "last_phase": self._last_phase.value if self._last_phase is not None else None,
            "pending_theta_reset": self._pending_theta_reset,
            "sequence_position": self._sequence_position,
            "ca3_threshold_offset": self._ca3_threshold_offset.detach().clone() if self._ca3_threshold_offset is not None else None,
            "ca3_activity_history": self._ca3_activity_history.detach().clone() if self._ca3_activity_history is not None else None,
            "ca3_slot_assignment": self._ca3_slot_assignment.detach().clone() if self._ca3_slot_assignment is not None else None,
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

        # 4. OSCILLATOR STATE (gamma oscillator and replay engine)
        oscillator_state = {}
        if self.gamma_oscillator is not None:
            oscillator_state["gamma"] = self.gamma_oscillator.get_state()
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
            raise ValueError(
                f"Input dimension mismatch: saved={saved_config.n_input}, "
                f"current={self.tri_config.n_input}"
            )
        if saved_config.n_output != self.tri_config.n_output:
            raise ValueError(
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
        if region_state["last_phase"] is not None:
            self._last_phase = TrialPhase(region_state["last_phase"])
        self._pending_theta_reset = region_state["pending_theta_reset"]
        self._sequence_position = region_state["sequence_position"]
        if region_state["ca3_threshold_offset"] is not None:
            self._ca3_threshold_offset = region_state["ca3_threshold_offset"].to(self.device)
        if region_state["ca3_activity_history"] is not None:
            self._ca3_activity_history = region_state["ca3_activity_history"].to(self.device)
        if region_state["ca3_slot_assignment"] is not None:
            self._ca3_slot_assignment = region_state["ca3_slot_assignment"].to(self.device)
        
        # Restore TrisynapticState
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
        oscillator_state = state["oscillator_state"]
        if "gamma" in oscillator_state and self.gamma_oscillator is not None:
            self.gamma_oscillator.load_state(oscillator_state["gamma"])
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
