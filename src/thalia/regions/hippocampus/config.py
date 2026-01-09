"""
Hippocampus Configuration and State Dataclasses.

This module contains configuration and state dataclasses for the
trisynaptic hippocampus (DG→CA3→CA1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch

from thalia.core.base.component_config import NeuralComponentConfig
from thalia.config.learning_config import STDPLearningConfig
from thalia.core.region_state import BaseRegionState
from thalia.components.synapses.stp import STPType
from thalia.regulation.learning_constants import LEARNING_RATE_ONE_SHOT
from thalia.regulation.region_architecture_constants import (
    HIPPOCAMPUS_SPARSITY_TARGET,
)


@dataclass
class Episode:
    """An episode stored in episodic memory for replay.

    Episodes are stored with priority for experience replay,
    where more important episodes (high reward, correct trials)
    are replayed more frequently.

    Episodes can store either:
    - A single state (traditional): Just the activity pattern at decision time
    - A sequence (extended): List of states from each gamma slot during encoding

    During sleep replay, sequences are replayed time-compressed using
    the gamma oscillator to drive slot-by-slot reactivation.
    """
    state: torch.Tensor          # Activity pattern at decision time (or final state)
    action: int                   # Selected action
    reward: float                 # Received reward
    correct: bool                 # Whether the action was correct
    context: Optional[torch.Tensor] = None  # Context/cue pattern
    metadata: Optional[Dict[str, Any]] = None  # Additional info
    priority: float = 1.0         # Replay priority
    timestamp: int = 0            # When this episode occurred
    sequence: Optional[List[torch.Tensor]] = None  # Sequence of states for gamma-driven replay


@dataclass
class HippocampusConfig(NeuralComponentConfig, STDPLearningConfig):
    """Configuration for hippocampus (trisynaptic circuit).

    Inherits STDP learning parameters from STDPLearningConfig:
    - learning_rate: Base learning rate (overridden with pathway-specific rates below)
    - learning_enabled: Global learning enable/disable
    - weight_min, weight_max: Weight bounds
    - tau_plus_ms, tau_minus_ms: STDP timing window parameters
    - a_plus, a_minus: LTP/LTD amplitudes
    - use_symmetric: Whether to use symmetric STDP

    The hippocampus has ~5x expansion from EC to DG, then compression back.

    **Size Specification**:
    - Simple: Just specify n_input, layer sizes auto-compute from biological ratios
    - Explicit: Use from_input_size() builder for clarity
    - Full control: Specify all layer sizes manually (dg_size, ca3_size, ca2_size, ca1_size)
    """
    # Override default learning rate with CA3-specific fast learning
    learning_rate: float = LEARNING_RATE_ONE_SHOT  # Fast one-shot learning for CA3 recurrent

    # Layer sizes (auto-computed from n_input if all are 0)
    # Use compute_hippocampus_sizes() helper to calculate from input size
    dg_size: int = field(default=0)   # Dentate Gyrus (pattern separation)
    ca3_size: int = field(default=0)  # CA3 (pattern completion)
    ca2_size: int = field(default=0)  # CA2 (social memory, temporal context)
    ca1_size: int = field(default=0)  # CA1 (output, match/mismatch)

    # DG sparsity (VERY sparse for pattern separation)
    dg_sparsity: float = HIPPOCAMPUS_SPARSITY_TARGET
    dg_inhibition: float = 5.0     # Strong lateral inhibition

    # CA3 recurrent dynamics
    ca3_recurrent_strength: float = 0.4  # Strength of recurrent connections
    ca3_sparsity: float = 0.10           # 10% active

    # CA2 dynamics (social memory and temporal context)
    ca2_sparsity: float = 0.12     # 12% active (slightly higher than CA3)
    ca2_plasticity_resistance: float = 0.1  # CA3→CA2 has 10x weaker plasticity (stability hub)

    # CA1 output
    ca1_sparsity: float = 0.15     # 15% active

    # Coincidence detection for comparison
    coincidence_window: float = 5.0  # ms window for spike coincidence

    # Spillover transmission (volume transmission)
    # Enable in hippocampus CA1 and CA3 where experimentally documented
    # (Vizi et al. 1999, Agnati et al. 2010, Sykova 2004)
    # Hippocampal spillover supports pattern completion and memory integration
    enable_spillover: bool = True  # Override base config (disabled by default)
    spillover_mode: str = "connectivity"  # Use shared inputs for neighborhood
    spillover_strength: float = 0.18  # 18% for CA regions (slightly higher than cortex)
    match_threshold: float = 0.3     # Fraction of coincident spikes for match

    # NMDA receptor parameters for CA1 coincidence detection
    # The threshold must be set high enough that only CA1 neurons with STRONG
    # CA3 input get their Mg²⁺ block removed. With ~48 CA3 spikes and 15%
    # connectivity, each CA1 receives 0.2-2.2 weighted input (mean ~1.0).
    # With tau=50ms and 15 test timesteps, the trace reaches ~40% of equilibrium,
    # so threshold=0.4 ensures only neurons with above-average CA3 input participate.
    nmda_tau: float = 50.0           # NMDA time constant (ms) - slow kinetics
    nmda_threshold: float = 0.4      # Threshold tuned for typical test duration
    nmda_steepness: float = 12.0     # Sharp discrimination above threshold
    ampa_ratio: float = 0.05         # Minimal ungated response (discrimination comes from NMDA)

    # Pathway-specific learning rates
    # Note: learning_rate (inherited from STDPLearningConfig) is used for CA3 recurrent
    ca3_ca2_learning_rate: float = 0.001  # Very weak CA3→CA2 (stability mechanism)
    ec_ca2_learning_rate: float = 0.01    # Strong EC→CA2 direct (temporal encoding)
    ca2_ca1_learning_rate: float = 0.005  # Moderate CA2→CA1 (social context to output)
    ec_ca1_learning_rate: float = 0.5     # Strong learning for EC→CA1 alignment

    # Feedforward inhibition parameters
    ffi_threshold: float = 0.3       # Input change threshold to trigger FFI
    ffi_strength: float = 0.8        # How much FFI suppresses activity
    ffi_tau: float = 5.0             # FFI decay time constant (ms)

    # =========================================================================
    # INTER-LAYER AXONAL DELAYS
    # =========================================================================
    # Biological signal propagation times within hippocampal circuit:
    # - DG→CA3 (mossy fibers): ~3ms
    # - CA3→CA2: ~2ms (shorter due to proximity)
    # - CA2→CA1: ~2ms
    # - CA3→CA1 (Schaffer collaterals): ~3ms (direct bypass)
    # Total circuit latency: ~7ms (slightly longer with CA2)
    #
    # Set to 0.0 for instant processing (current behavior, backward compatible)
    # Set to biological values for realistic temporal dynamics and STDP timing
    dg_to_ca3_delay_ms: float = 0.0  # DG→CA3 axonal delay (0=instant)
    ca3_to_ca2_delay_ms: float = 0.0  # CA3→CA2 axonal delay (0=instant)
    ca2_to_ca1_delay_ms: float = 0.0  # CA2→CA1 axonal delay (0=instant)
    ca3_to_ca1_delay_ms: float = 0.0  # CA3→CA1 axonal delay (0=instant, direct bypass)

    # CA3 Bistable Neuron Parameters
    # Real CA3 pyramidal neurons have intrinsic bistability via I_NaP (persistent
    # sodium) and I_CAN (calcium-activated nonspecific cation) currents. These
    # allow neurons to maintain firing without continuous external input.
    #
    # We model this with a "persistent activity" trace that:
    # 1. Accumulates when a neuron fires (like Ca²⁺ buildup activating I_CAN)
    # 2. Decays slowly (τ ~100-200ms, like Ca²⁺ clearance)
    # 3. Provides additional input current (positive feedback)
    #
    # This creates bistability: once a neuron starts firing, the persistent
    # activity helps keep it firing, stabilizing attractor states.
    ca3_persistent_tau: float = 300.0    # Decay time constant (ms) - very slow decay
    ca3_persistent_gain: float = 3.0     # Strong persistent contribution

    # EC Layer III input size (for direct EC→CA1 pathway)
    # If 0, uses the same input as EC layer II (n_input)
    # If >0, expects separate raw sensory input for the temporoammonic path
    ec_l3_input_size: int = 0

    # =========================================================================
    # THETA-GAMMA COUPLING
    # =========================================================================
    # Enable theta-gamma coupling from centralized oscillator manager
    theta_gamma_enabled: bool = True  # Use centralized oscillators for sequence encoding

    # =========================================================================
    # SHORT-TERM PLASTICITY (STP)
    # =========================================================================
    # Biologically, different hippocampal pathways have distinct STP properties:
    # - Mossy Fibers (DG→CA3): STRONGLY FACILITATING - repeated DG activity
    #   causes progressively stronger CA3 activation (U~0.03, τ_f~500ms)
    # - CA3→CA2: DEPRESSING - stability mechanism, prevents runaway activity
    # - CA2→CA1: FACILITATING - temporal sequences benefit from facilitation
    # - Schaffer Collaterals (CA3→CA1): MIXED/DEPRESSING - high-frequency
    #   activity causes depression, enabling novelty detection
    # - EC→CA1 direct: DEPRESSING - initial stimulus is strongest
    # - EC→CA2 direct: DEPRESSING - similar to EC→CA1
    #
    # References:
    # - Salin et al. (1996): Mossy fiber facilitation (U=0.03!)
    # - Dobrunz & Stevens (1997): Schaffer collateral STP
    # - Chevaleyre & Siegelbaum (2010): CA2 plasticity properties
    stp_enabled: bool = True
    stp_mossy_type: STPType = STPType.FACILITATING_STRONG  # DG→CA3 (MF)
    stp_ca3_ca2_type: STPType = STPType.DEPRESSING         # CA3→CA2 (stability)
    stp_ca2_ca1_type: STPType = STPType.FACILITATING       # CA2→CA1 (sequences)
    stp_ec_ca2_type: STPType = STPType.DEPRESSING          # EC→CA2 direct
    stp_schaffer_type: STPType = STPType.DEPRESSING        # CA3→CA1 (SC)
    stp_ec_ca1_type: STPType = STPType.DEPRESSING          # EC→CA1 direct
    # CA3→CA3 recurrent: DEPRESSING - prevents frozen attractors
    # Without STD, the same neurons fire every timestep because recurrent
    # connections reinforce active neurons. With STD, frequently-firing
    # synapses get temporarily weaker, allowing pattern transitions.
    stp_ca3_recurrent_type: STPType = STPType.DEPRESSING_FAST

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION (SFA)
    # =========================================================================
    # Real CA3 pyramidal neurons show strong adaptation: Ca²⁺ influx during
    # spikes activates K⁺ channels (I_AHP) that hyperpolarize the neuron.
    # This prevents the same neurons from dominating activity.
    # Inherited from base with hippocampus-specific override:
    adapt_increment: float = 0.5  # Very strong (prevents CA3 seizure-like activity)
    # adapt_tau: 100.0 (use base default)

    # =========================================================================
    # ACTIVITY-DEPENDENT INHIBITION
    # =========================================================================
    # Feedback inhibition from interneurons scales with total CA3 activity.
    # When many CA3 neurons fire, inhibition increases, making it harder
    # for the same neurons to fire again.
    ca3_feedback_inhibition: float = 0.3  # Inhibition per total activity

    # =========================================================================
    # GAP JUNCTIONS (Electrical Synapses)
    # =========================================================================
    # Gap junctions between CA1 interneurons (basket cells, bistratified cells)
    # provide fast electrical coupling for theta-gamma synchronization.
    # Critical for precise spike timing in episodic memory encoding/retrieval.
    #
    # Biological evidence:
    # - Fukuda & Kosaka (2000): Gap junctions in hippocampal GABAergic networks
    # - Traub et al. (2003): Electrical coupling essential for gamma in CA1
    # - Hormuzdi et al. (2001): Connexin36-mediated interneuron coupling
    #
    # CA1 interneurons (~10-15% of CA1 population) have dense gap junction
    # networks that synchronize inhibition during theta-gamma nested oscillations.
    gap_junctions_enabled: bool = True  # Enable gap junctions in CA1 interneurons
    gap_junction_strength: float = 0.12  # Coupling strength (biological: 0.05-0.2)
    gap_junction_threshold: float = 0.25  # Neighborhood connectivity threshold
    gap_junction_max_neighbors: int = 8  # Max neighbors per interneuron (biological: 4-12)

    # =========================================================================
    # HETEROSYNAPTIC PLASTICITY
    # =========================================================================
    # Synapses to inactive postsynaptic neurons weaken when nearby neurons
    # fire strongly. This prevents winner-take-all dynamics from freezing.
    heterosynaptic_ratio: float = 0.1  # LTD for inactive synapses

    # =========================================================================
    # THETA-PHASE RESETS
    # =========================================================================
    # Reset persistent activity at the start of each theta cycle to prevent
    # stale attractors from dominating. In real brains, theta troughs
    # (encoding phase) partially reset the network.
    theta_reset_persistent: bool = True  # Reset ca3_persistent at theta trough
    theta_reset_fraction: float = 0.5    # How much to decay (0=none, 1=full)

    # =========================================================================
    # THETA-GAMMA COUPLING (Phase Coding - EMERGENT)
    # =========================================================================
    # Note: Theta-gamma coupling (frequency, strength) is handled by the
    # centralized OscillatorManager. Phase preferences EMERGE from:
    # 1. Synaptic delays (different neurons receive inputs at different times)
    # 2. STDP (neurons strengthen connections at their preferred phase)
    # 3. Dendritic integration (~15ms window naturally filters by timing)
    #
    # Working memory capacity = gamma_freq / theta_freq (~40Hz / 8Hz ≈ 5-7 slots)
    # This emerges automatically - no hardcoded slots needed!

    # Phase diversity initialization: adds timing jitter to initial weights
    # This seeds the emergence of phase preferences (otherwise all neurons identical)
    phase_diversity_init: bool = True     # Initialize weights with timing diversity
    phase_jitter_std_ms: float = 5.0      # Std dev of timing jitter (0-10ms)

    # =========================================================================
    # HINDSIGHT EXPERIENCE REPLAY (HER)
    # =========================================================================
    # Enable goal relabeling for multi-goal learning.
    # "What if my actual outcome WAS my goal?" → learn from every episode
    use_her: bool = True  # Enable hindsight experience replay
    her_k_hindsight: int = 4  # Number of hindsight goals per real experience
    her_replay_ratio: float = 0.8  # Fraction of replays that are hindsight
    her_strategy: str = "future"  # "future", "final", "episode", or "random"
    her_goal_tolerance: float = 0.1  # Distance threshold for goal achievement
    her_buffer_size: int = 1000  # Maximum episodes to store

    def __post_init__(self) -> None:
        """Auto-compute layer sizes from n_input if all are 0, then validate."""
        # Auto-compute if all layer sizes are 0
        if all(s == 0 for s in [self.dg_size, self.ca3_size, self.ca2_size, self.ca1_size]):
            from thalia.config.region_sizes import compute_hippocampus_sizes
            sizes = compute_hippocampus_sizes(self.n_input)
            # Use object.__setattr__ to modify frozen dataclass fields
            object.__setattr__(self, "dg_size", sizes["dg_size"])
            object.__setattr__(self, "ca3_size", sizes["ca3_size"])
            object.__setattr__(self, "ca2_size", sizes["ca2_size"])
            object.__setattr__(self, "ca1_size", sizes["ca1_size"])
            # Update n_output to match CA1 (output layer)
            object.__setattr__(self, "n_output", sizes["ca1_size"])
            # Update n_neurons to total size
            total_neurons = sizes["dg_size"] + sizes["ca3_size"] + sizes["ca2_size"] + sizes["ca1_size"]
            object.__setattr__(self, "n_neurons", total_neurons)
        # Auto-compute n_neurons from layer sizes if not provided (or if default 100)
        elif self.n_neurons == 100:  # Default value
            total_neurons = self.dg_size + self.ca3_size + self.ca2_size + self.ca1_size
            object.__setattr__(self, "n_neurons", total_neurons)

        # Validate after computation
        self.validate()

    def validate(self) -> None:
        """Validate size constraints and biological ratios.

        Raises:
            ValueError: If any layer size is 0 or ratios are unrealistic
        """
        # Check that no layer sizes are 0
        if any(s == 0 for s in [self.dg_size, self.ca3_size, self.ca2_size, self.ca1_size]):
            raise ValueError(
                f"All layer sizes must be > 0. Got dg={self.dg_size}, "
                f"ca3={self.ca3_size}, ca2={self.ca2_size}, ca1={self.ca1_size}. "
                "Either specify all sizes explicitly or let them auto-compute from n_input."
            )

        # Check n_output matches ca1_size (CA1 is output layer)
        if self.n_output != self.ca1_size:
            raise ValueError(
                f"n_output ({self.n_output}) must equal ca1_size ({self.ca1_size}). "
                "CA1 is the output layer of the hippocampus."
            )

        # Check n_neurons matches total
        total = self.dg_size + self.ca3_size + self.ca2_size + self.ca1_size
        if self.n_neurons != total:
            raise ValueError(
                f"n_neurons ({self.n_neurons}) must equal sum of layer sizes ({total})"
            )

        # Warn if ratios are far from biological norms (but don't fail)
        dg_to_input = self.dg_size / max(self.n_input, 1)
        if not (2.0 <= dg_to_input <= 6.0):
            import warnings
            warnings.warn(
                f"DG expansion ratio ({dg_to_input:.1f}x) is outside biological range (2-6x). "
                f"Using dg_size={self.dg_size} for n_input={self.n_input}."
            )

    @classmethod
    def from_input_size(
        cls,
        n_input: int,
        **kwargs
    ) -> "HippocampusConfig":
        """Create config with layer sizes computed from input size.

        Uses biological ratios:
        - DG: 4x expansion from EC (pattern separation)
        - CA3: 0.5x compression from DG (pattern completion)
        - CA2: 0.25x of DG (social memory, temporal context)
        - CA1: 1x of CA3 (output, match/mismatch)

        Args:
            n_input: Input size (from entorhinal cortex)
            **kwargs: Additional config parameters

        Returns:
            HippocampusConfig with computed layer sizes

        Example:
            >>> config = HippocampusConfig.from_input_size(n_input=40, device="cpu")
            >>> config.dg_size  # 160
            >>> config.ca1_size  # 80
        """
        from thalia.config.region_sizes import compute_hippocampus_sizes
        sizes = compute_hippocampus_sizes(n_input)
        return cls(
            n_input=n_input,
            n_output=sizes["ca1_size"],
            n_neurons=sizes["dg_size"] + sizes["ca3_size"] + sizes["ca2_size"] + sizes["ca1_size"],
            dg_size=sizes["dg_size"],
            ca3_size=sizes["ca3_size"],
            ca2_size=sizes["ca2_size"],
            ca1_size=sizes["ca1_size"],
            **kwargs
        )


@dataclass
class HippocampusState(BaseRegionState):
    """State for hippocampus (DG→CA3→CA2→CA1 circuit) with RegionState protocol compliance.

    Extends BaseRegionState with hippocampus-specific state:
    - DG/CA3/CA2/CA1 layer activities and traces
    - CA3 persistent activity (attractor dynamics)
    - Sample trace for memory encoding
    - STDP traces for multiple pathways
    - NMDA trace for temporal integration
    - Stored DG pattern for match/mismatch detection
    - Feedforward inhibition strength
    - Short-term plasticity (STP) state for 7 pathways

    Note: Neuromodulators (dopamine, acetylcholine, norepinephrine) are
    inherited from BaseRegionState.

    The CA1 spikes ARE the output - no interpretation needed!
    Different CA1 spike patterns naturally emerge for match vs mismatch
    through the coincidence detection between CA3 (memory) and EC (current).
    """

    # Layer activities (current spikes)
    dg_spikes: Optional[torch.Tensor] = None
    ca3_spikes: Optional[torch.Tensor] = None
    ca2_spikes: Optional[torch.Tensor] = None  # CA2: Social memory and temporal context
    ca1_spikes: Optional[torch.Tensor] = None

    # CA3 recurrent state
    ca3_membrane: Optional[torch.Tensor] = None

    # CA1 membrane voltages (for gap junction coupling)
    ca1_membrane: Optional[torch.Tensor] = None

    # CA3 bistable persistent activity trace
    # Models I_NaP/I_CAN currents that allow neurons to maintain firing
    # without continuous external input. This is essential for stable
    # attractor states during delay periods.
    ca3_persistent: Optional[torch.Tensor] = None

    # Memory trace (for STDP learning during sample phase)
    sample_trace: Optional[torch.Tensor] = None

    # STDP traces
    dg_trace: Optional[torch.Tensor] = None
    ca3_trace: Optional[torch.Tensor] = None
    ca2_trace: Optional[torch.Tensor] = None

    # NMDA trace for temporal integration (slow kinetics)
    nmda_trace: Optional[torch.Tensor] = None

    # Stored DG pattern from sample phase (for match/mismatch detection)
    stored_dg_pattern: Optional[torch.Tensor] = None

    # Current feedforward inhibition strength
    ffi_strength: float = 0.0

    # Short-term plasticity state for 7 pathways
    stp_mossy_state: Optional[Dict[str, torch.Tensor]] = None         # DG→CA3 facilitation
    stp_ca3_ca2_state: Optional[Dict[str, torch.Tensor]] = None       # CA3→CA2 depression
    stp_ca2_ca1_state: Optional[Dict[str, torch.Tensor]] = None       # CA2→CA1 facilitation
    stp_ec_ca2_state: Optional[Dict[str, torch.Tensor]] = None        # EC→CA2 direct
    stp_schaffer_state: Optional[Dict[str, torch.Tensor]] = None      # CA3→CA1 depression
    stp_ec_ca1_state: Optional[Dict[str, torch.Tensor]] = None        # EC→CA1 direct
    stp_ca3_recurrent_state: Optional[Dict[str, torch.Tensor]] = None # CA3 recurrent

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for checkpointing.

        Returns:
            Dictionary with all state fields, including nested STP states for 7 pathways.
        """
        return {
            # Base region state
            "spikes": self.spikes,
            "membrane": self.membrane,
            "dopamine": self.dopamine,
            "acetylcholine": self.acetylcholine,
            "norepinephrine": self.norepinephrine,
            # Layer activities
            "dg_spikes": self.dg_spikes,
            "ca3_spikes": self.ca3_spikes,
            "ca2_spikes": self.ca2_spikes,
            "ca1_spikes": self.ca1_spikes,
            # CA3 state
            "ca3_membrane": self.ca3_membrane,
            "ca3_persistent": self.ca3_persistent,
            # CA1 state (gap junctions)
            "ca1_membrane": self.ca1_membrane,
            # Memory and traces
            "sample_trace": self.sample_trace,
            "dg_trace": self.dg_trace,
            "ca3_trace": self.ca3_trace,
            "ca2_trace": self.ca2_trace,
            "nmda_trace": self.nmda_trace,
            "stored_dg_pattern": self.stored_dg_pattern,
            "ffi_strength": self.ffi_strength,
            # STP state (nested dicts for 7 pathways)
            "stp_mossy_state": self.stp_mossy_state,
            "stp_ca3_ca2_state": self.stp_ca3_ca2_state,
            "stp_ca2_ca1_state": self.stp_ca2_ca1_state,
            "stp_ec_ca2_state": self.stp_ec_ca2_state,
            "stp_schaffer_state": self.stp_schaffer_state,
            "stp_ec_ca1_state": self.stp_ec_ca1_state,
            "stp_ca3_recurrent_state": self.stp_ca3_recurrent_state,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        device: str = "cpu",
    ) -> "HippocampusState":
        """Deserialize state from dictionary.

        Args:
            data: Dictionary with state fields
            device: Target device string (e.g., 'cpu', 'cuda', 'cuda:0')

        Returns:
            HippocampusState instance with restored state
        """
        def transfer_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return t
            return t.to(device)

        def transfer_nested_dict(d: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
            """Transfer nested dict of tensors to device."""
            if d is None:
                return d
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

        return cls(
            # Base region state
            spikes=transfer_tensor(data.get("spikes")),
            membrane=transfer_tensor(data.get("membrane")),
            dopamine=data.get("dopamine", 0.2),
            acetylcholine=data.get("acetylcholine", 0.0),
            norepinephrine=data.get("norepinephrine", 0.0),
            # Layer activities
            dg_spikes=transfer_tensor(data.get("dg_spikes")),
            ca3_spikes=transfer_tensor(data.get("ca3_spikes")),
            ca2_spikes=transfer_tensor(data.get("ca2_spikes")),  # Backward compatible (None if missing)
            ca1_spikes=transfer_tensor(data.get("ca1_spikes")),
            # CA3 state
            ca3_membrane=transfer_tensor(data.get("ca3_membrane")),
            ca3_persistent=transfer_tensor(data.get("ca3_persistent")),
            # CA1 state (gap junctions, added 2025-01, backward compatible)
            ca1_membrane=transfer_tensor(data.get("ca1_membrane")),
            # Memory and traces
            sample_trace=transfer_tensor(data.get("sample_trace")),
            dg_trace=transfer_tensor(data.get("dg_trace")),
            ca3_trace=transfer_tensor(data.get("ca3_trace")),
            ca2_trace=transfer_tensor(data.get("ca2_trace")),  # Backward compatible (None if missing)
            nmda_trace=transfer_tensor(data.get("nmda_trace")),
            stored_dg_pattern=transfer_tensor(data.get("stored_dg_pattern")),
            ffi_strength=data.get("ffi_strength", 0.0),
            # STP state (nested dicts for 7 pathways, backward compatible)
            stp_mossy_state=transfer_nested_dict(data.get("stp_mossy_state")),
            stp_ca3_ca2_state=transfer_nested_dict(data.get("stp_ca3_ca2_state")),
            stp_ca2_ca1_state=transfer_nested_dict(data.get("stp_ca2_ca1_state")),
            stp_ec_ca2_state=transfer_nested_dict(data.get("stp_ec_ca2_state")),
            stp_schaffer_state=transfer_nested_dict(data.get("stp_schaffer_state")),
            stp_ec_ca1_state=transfer_nested_dict(data.get("stp_ec_ca1_state")),
            stp_ca3_recurrent_state=transfer_nested_dict(data.get("stp_ca3_recurrent_state")),
        )

    def reset(self) -> None:
        """Reset state to default values (in-place mutation)."""
        # Reset base state (spikes, membrane, neuromodulators with DA_BASELINE_STANDARD)
        super().reset()

        # Reset hippocampus-specific state
        self.dg_spikes = None
        self.ca3_spikes = None
        self.ca2_spikes = None
        self.ca1_spikes = None
        self.ca3_membrane = None
        self.ca3_persistent = None
        self.sample_trace = None
        self.dg_trace = None
        self.ca3_trace = None
        self.ca2_trace = None
        self.nmda_trace = None
        self.stored_dg_pattern = None
        self.ffi_strength = 0.0
        self.stp_mossy_state = None
        self.stp_ca3_ca2_state = None
        self.stp_ca2_ca1_state = None
        self.stp_ec_ca2_state = None
        self.stp_schaffer_state = None
        self.stp_ec_ca1_state = None
        self.stp_ca3_recurrent_state = None
