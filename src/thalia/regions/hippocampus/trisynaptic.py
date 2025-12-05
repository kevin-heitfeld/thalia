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

from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.core.neuron import LIFNeuron, LIFConfig
from thalia.core.stp import ShortTermPlasticity, STPConfig
from thalia.core.utils import ensure_batch_dim, clamp_weights, cosine_similarity_safe
from thalia.core.traces import update_trace
from thalia.core.diagnostics_mixin import DiagnosticsMixin
from thalia.regions.base import BrainRegion, LearningRule
from thalia.regions.theta_dynamics import TrialPhase, FeedforwardInhibition
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
    """

    def __init__(self, config: TrisynapticConfig):
        """Initialize trisynaptic hippocampus."""
        self.tri_config = config

        # Compute layer sizes
        self.dg_size = int(config.n_input * config.dg_expansion)
        self.ca3_size = int(self.dg_size * config.ca3_size_ratio)
        self.ca1_size = config.n_output  # CA1 matches output

        # Call parent init
        super().__init__(config)

        # Override weights with trisynaptic circuit weights
        self._init_circuit_weights()

        # Create LIF neurons for each layer
        lif_config = LIFConfig(tau_mem=20.0, v_threshold=1.0)
        self.dg_neurons = LIFNeuron(self.dg_size, lif_config)
        self.ca3_neurons = LIFNeuron(self.ca3_size, lif_config)
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
        else:
            self.stp_mossy = None
            self.stp_schaffer = None
            self.stp_ec_ca1 = None

        # Episode buffer for sleep consolidation
        self.episode_buffer: List[Episode] = []

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
            self._create_sparse_random_weights(
                self.dg_size,
                self.tri_config.n_input,
                sparsity=0.3,  # 30% connectivity
                device=device,
                weight_scale=0.5,  # Strong weights for propagation
                normalize_rows=True,  # Normalize for reliable propagation
            )
        )

        # DG → CA3: Random but less sparse
        # Uses row normalization for reliable activity propagation
        self.w_dg_ca3 = nn.Parameter(
            self._create_sparse_random_weights(
                self.ca3_size,
                self.dg_size,
                sparsity=0.5,
                device=device,
                weight_scale=0.5,  # Strong weights for propagation
                normalize_rows=True,  # Normalize for reliable propagation
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
            torch.randn(self.ca3_size, self.ca3_size, device=device) * 0.15 + 0.05
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
            self._create_sparse_random_weights(
                self.ca1_size,
                self.ca3_size,
                sparsity=0.15,  # Each CA1 sees only 15% of CA3
                device=device,
                weight_scale=0.3,  # Strong individual weights
                normalize_rows=False,  # NO normalization - pattern-specific!
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
            self._create_sparse_random_weights(
                self.ca1_size,
                self.tri_config.n_input,
                sparsity=0.20,  # Each CA1 sees only 20% of EC
                device=device,
                weight_scale=0.3,  # Strong individual weights
                normalize_rows=False,  # NO normalization - pattern-specific!
            )
        )

        # EC Layer III → CA1: Separate pathway for raw sensory input
        # In biology, EC layer III pyramidal cells project directly to CA1
        # (temporoammonic path), carrying raw sensory information that is
        # compared against the retrieved memory from CA3.
        self._ec_l3_input_size = self.tri_config.ec_l3_input_size
        if self._ec_l3_input_size > 0:
            self.w_ec_l3_ca1 = nn.Parameter(
                self._create_sparse_random_weights(
                    self.ca1_size,
                    self._ec_l3_input_size,
                    sparsity=0.20,
                    device=device,
                    weight_scale=0.3,
                    normalize_rows=False,
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

    def _create_sparse_random_weights(
        self,
        n_post: int,
        n_pre: int,
        sparsity: float,
        device: torch.device,
        weight_scale: float = 0.1,
        normalize_rows: bool = True,
    ) -> torch.Tensor:
        """Create sparse random weight matrix.

        Each postsynaptic neuron receives input from a random subset of
        presynaptic neurons. This creates orthogonal projections for
        pattern separation.

        Args:
            n_post: Number of postsynaptic neurons
            n_pre: Number of presynaptic neurons
            sparsity: Fraction of connections (0-1)
            device: Torch device
            weight_scale: Scale for individual weights
            normalize_rows: If True, normalize row sums for reliable propagation.
                           If False, keep raw weights for pattern-specific activation.
        """
        # Create connectivity mask
        mask = torch.rand(n_post, n_pre, device=device) < sparsity

        # Random weights where connected
        weights = torch.rand(n_post, n_pre, device=device) * weight_scale
        weights = weights * mask.float()

        if normalize_rows:
            # Normalize rows so each neuron has similar total input
            # This ensures reliable activity propagation
            row_sums = weights.sum(dim=1, keepdim=True) + 1e-6
            target_sum = n_pre * sparsity * weight_scale * 0.5  # Scale factor
            weights = weights / row_sums * target_sum

        return weights

    def reset(self) -> None:
        """Reset state for new episode.

        Note: Consider using new_trial() instead, which aligns theta and
        clears input history without fully resetting membrane potentials.
        Full reset is mainly needed between completely unrelated episodes.
        """
        super().reset()
        self._init_state(batch_size=1)

    def new_trial(self) -> None:
        """Prepare for a new trial (biologically-realistic alternative to reset).

        This is the preferred method to call between trials because it:
        1. Clears FFI input history (new stimulus = new comparison baseline)
        2. Clears NMDA trace (prevents accumulation across trials)
        3. Does NOT reset membrane potentials (let natural decay handle it)

        The idea is that real neurons don't "reset" - they just receive new
        input and the dynamics naturally transition. FFI handles the state
        transitions that resets were approximating.

        Note: Theta phase alignment is handled by BrainSystem since theta
        is a global oscillation across all brain regions.
        """
        # Clear FFI history so first stimulus triggers inhibition
        self.feedforward_inhibition.clear()

        # Clear stored patterns (this IS needed for new trial)
        self.state.stored_dg_pattern = None
        self.state.sample_trace = None

        # Clear NMDA trace to prevent accumulation across trials
        # The NMDA trace is used for coincidence detection during RETRIEVE,
        # and should start fresh for each trial's comparison
        if self.state.nmda_trace is not None:
            self.state.nmda_trace.zero_()

    def _init_state(self, batch_size: int = 1) -> None:
        """Initialize all layer states (internal method)."""
        device = self.device

        self.dg_neurons.reset_state(batch_size)
        self.ca3_neurons.reset_state(batch_size)
        self.ca1_neurons.reset_state(batch_size)

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

        # Clear FFI history
        self.feedforward_inhibition.clear()

    def forward(
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

        # Ensure state is initialized
        if self.state.dg_spikes is None:
            self._init_state(batch_size)

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
        dg_spikes = self._apply_wta_sparsity(
            dg_spikes,
            self.tri_config.dg_sparsity,
            self.dg_neurons.v if hasattr(self.dg_neurons, 'v') else None,
        )
        self.state.dg_spikes = dg_spikes

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

        # Recurrent from previous CA3 activity (theta-gated)
        ca3_rec = torch.matmul(
            self.state.ca3_spikes.float(),
            self.w_ca3_ca3.t()
        ) * self.tri_config.ca3_recurrent_strength * rec_gate

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
        ca3_persistent_input = (
            self.state.ca3_persistent * self.tri_config.ca3_persistent_gain
        )

        # Total CA3 input = feedforward + recurrent + bistable persistent
        ca3_input = ca3_ff + ca3_rec + ca3_persistent_input

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
        ca3_spikes = self._apply_wta_sparsity(
            ca3_spikes,
            self.tri_config.ca3_sparsity,
            self.ca3_neurons.membrane if hasattr(self.ca3_neurons, 'membrane') else None,
        )
        self.state.ca3_spikes = ca3_spikes

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
            # Learning rate modulated by theta phase
            ca3_activity = ca3_spikes.float().squeeze()
            if ca3_activity.sum() > 0:
                # Hebbian outer product: neurons that fire together wire together
                effective_lr = self.tri_config.learning_rate * encoding_mod
                dW = effective_lr * torch.outer(ca3_activity, ca3_activity)
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
            ca1_spikes = self._apply_wta_sparsity(
                ca1_spikes,
                cfg.ca1_sparsity,
                self.ca1_neurons.v if hasattr(self.ca1_neurons, 'v') else None,
            )

            # ---------------------------------------------------------
            # HEBBIAN LEARNING: EC→CA1 plasticity (modulated by theta)
            # ---------------------------------------------------------
            # Strengthen connections: active EC neurons → active CA1 neurons
            # This aligns the direct pathway with the indirect pathway
            # Use the appropriate weight matrix based on input type
            ec_activity = ec_input_for_ca1.float().squeeze()
            ca1_activity = ca1_spikes.float().squeeze()

            if ec_activity.sum() > 0 and ca1_activity.sum() > 0:
                # Hebbian outer product: w_ij += lr * post_j * pre_i
                effective_lr = cfg.ec_ca1_learning_rate * encoding_mod
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

    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        dt: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply STDP learning to circuit weights.

        CA3 recurrent connections learn via STDP to create attractors.
        """
        cfg = self.tri_config

        # CA3 recurrent STDP: strengthen connections between co-active neurons
        if self.state.ca3_trace is not None and self.state.ca3_spikes is not None:
            # LTP: post spike when pre trace is high
            ca3_post = self.state.ca3_spikes.float().squeeze()
            ca3_trace = self.state.ca3_trace.squeeze()

            ltp = torch.outer(ca3_post, ca3_trace)
            ltd = torch.outer(ca3_trace, ca3_post) * 0.5  # Weaker LTD

            dW = cfg.ca3_learning_rate * (ltp - ltd)

            with torch.no_grad():
                self.w_ca3_ca3.data += dW
                self.w_ca3_ca3.data.fill_diagonal_(0.0)  # No self-connections
                clamp_weights(self.w_ca3_ca3.data, cfg.w_min, cfg.w_max)

        return {
            "ca3_weight_mean": self.w_ca3_ca3.data.mean().item(),
            "dg_sparsity": self.state.dg_spikes.float().mean().item() if self.state.dg_spikes is not None else 0,
            "ca3_sparsity": self.state.ca3_spikes.float().mean().item() if self.state.ca3_spikes is not None else 0,
        }

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
    ) -> None:
        """Store an episode in episodic memory for later replay.

        Priority is computed based on reward magnitude and correctness.
        """
        cfg = self.tri_config

        # Compute priority based on reward and correctness
        base_priority = 1.0 + abs(reward)
        if correct:
            base_priority += 0.5  # Boost for correct trials
        base_priority += priority_boost

        episode = Episode(
            state=state.clone().detach(),
            context=context.clone().detach() if context is not None else None,
            action=action,
            reward=reward,
            correct=correct,
            metadata=metadata,
            priority=base_priority,
            timestamp=len(self.episode_buffer),
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
