"""
Hippocampus Configuration and State Dataclasses.

This module contains configuration and state dataclasses for the
trisynaptic hippocampus (DG→CA3→CA1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch

from thalia.regions.base import RegionConfig, RegionState
from thalia.core.stp import STPType
from thalia.core.learning_constants import (
    LEARNING_RATE_ONE_SHOT,
    TAU_STDP_PLUS,
    TAU_STDP_MINUS,
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
class HippocampusConfig(RegionConfig):
    """Configuration for hippocampus (trisynaptic circuit).

    The hippocampus has ~5x expansion from EC to DG, then compression back.
    """
    # Layer sizes (relative to input)
    dg_expansion: float = 5.0      # DG has 5x more neurons than input
    ca3_size_ratio: float = 0.5    # CA3 is half of DG
    ca1_size_ratio: float = 0.5    # CA1 matches output

    # DG sparsity (VERY sparse for pattern separation)
    dg_sparsity: float = 0.02      # Only 2% active (biological: 1-5%)
    dg_inhibition: float = 5.0     # Strong lateral inhibition

    # CA3 recurrent dynamics
    ca3_recurrent_strength: float = 0.4  # Strength of recurrent connections
    ca3_sparsity: float = 0.10           # 10% active
    ca3_learning_rate: float = 0.1       # STDP learning rate

    # CA1 output
    ca1_sparsity: float = 0.15     # 15% active

    # Coincidence detection for comparison
    coincidence_window: float = 5.0  # ms window for spike coincidence
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

    # Learning rates
    learning_rate: float = LEARNING_RATE_ONE_SHOT  # Fast one-shot learning for CA3 recurrent
    ec_ca1_learning_rate: float = 0.5  # Strong learning for EC→CA1 alignment
    stdp_tau_plus: float = TAU_STDP_PLUS      # ms
    stdp_tau_minus: float = TAU_STDP_MINUS     # ms

    # Weight bounds
    w_max: float = 2.0
    w_min: float = 0.0

    # Feedforward inhibition parameters
    ffi_threshold: float = 0.3       # Input change threshold to trigger FFI
    ffi_strength: float = 0.8        # How much FFI suppresses activity
    ffi_tau: float = 5.0             # FFI decay time constant (ms)

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
    # SHORT-TERM PLASTICITY (STP)
    # =========================================================================
    # Biologically, different hippocampal pathways have distinct STP properties:
    # - Mossy Fibers (DG→CA3): STRONGLY FACILITATING - repeated DG activity
    #   causes progressively stronger CA3 activation (U~0.03, τ_f~500ms)
    # - Schaffer Collaterals (CA3→CA1): MIXED/DEPRESSING - high-frequency
    #   activity causes depression, enabling novelty detection
    # - EC→CA1 direct: DEPRESSING - initial stimulus is strongest
    #
    # References:
    # - Salin et al. (1996): Mossy fiber facilitation (U=0.03!)
    # - Dobrunz & Stevens (1997): Schaffer collateral STP
    stp_enabled: bool = False
    stp_mossy_type: STPType = STPType.FACILITATING_STRONG  # DG→CA3 (MF)
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
    ca3_adapt_increment: float = 0.5  # Adaptation per spike (0 = disabled)
    ca3_adapt_tau: float = 100.0      # Adaptation time constant (ms)

    # =========================================================================
    # ACTIVITY-DEPENDENT INHIBITION
    # =========================================================================
    # Feedback inhibition from interneurons scales with total CA3 activity.
    # When many CA3 neurons fire, inhibition increases, making it harder
    # for the same neurons to fire again.
    ca3_feedback_inhibition: float = 0.3  # Inhibition per total activity

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
    # SYNAPTIC SCALING (Homeostatic)
    # =========================================================================
    # Synaptic scaling maintains stable firing rates by multiplicatively
    # adjusting all synapses to a neuron. If a neuron fires too much/little,
    # all its input synapses scale down/up together.
    synaptic_scaling_enabled: bool = True
    synaptic_scaling_target: float = 0.3  # Target mean weight
    synaptic_scaling_rate: float = 0.001  # Slow adaptation rate

    # =========================================================================
    # INTRINSIC PLASTICITY
    # =========================================================================
    # Neurons adjust their own excitability based on firing history.
    # High activity → higher threshold (less excitable)
    # Low activity → lower threshold (more excitable)
    # This operates on LONGER timescales than SFA.
    intrinsic_plasticity_enabled: bool = True
    intrinsic_target_rate: float = 0.1    # Target firing rate
    intrinsic_adaptation_rate: float = 0.01  # How fast threshold adapts

    # =========================================================================
    # THETA-GAMMA COUPLING
    # =========================================================================
    # Gamma oscillations (30-100 Hz) nested within theta enable:
    # 1. Sequence encoding: items at different gamma phases within theta
    # 2. Working memory capacity: ~7 gamma slots per theta cycle
    # 3. Phase coding: temporal order encoded in gamma phase
    theta_gamma_enabled: bool = True     # Enable theta-gamma coupling
    gamma_freq_hz: float = 40.0           # Gamma frequency (30-100 Hz typical)
    gamma_n_slots: int = 7                # Working memory slots per theta cycle
    gamma_coupling_strength: float = 0.8  # How much theta modulates gamma
    gamma_gating_strength: float = 0.5    # How much gamma gates CA3 activity

    # Slot mode determines how items are assigned to gamma slots:
    # - "item": Each forward() call advances to next slot (position-based)
    #           Best for discrete token sequences where timing is variable
    # - "time": Slot determined by oscillator phase (time-based)
    #           Best for continuous input or replay where timing matters
    gamma_slot_mode: str = "item"  # "item" or "time"

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


@dataclass
class HippocampusState(RegionState):
    """State for hippocampus (trisynaptic circuit).

    The CA1 spikes ARE the output - no interpretation needed!
    Different CA1 spike patterns naturally emerge for match vs mismatch
    through the coincidence detection between CA3 (memory) and EC (current).
    """
    # Layer activities (current spikes)
    dg_spikes: Optional[torch.Tensor] = None
    ca3_spikes: Optional[torch.Tensor] = None
    ca1_spikes: Optional[torch.Tensor] = None

    # CA3 recurrent state
    ca3_membrane: Optional[torch.Tensor] = None

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

    # NMDA trace for temporal integration (slow kinetics)
    nmda_trace: Optional[torch.Tensor] = None

    # Stored DG pattern from sample phase (for match/mismatch detection)
    stored_dg_pattern: Optional[torch.Tensor] = None

    # Current feedforward inhibition strength
    ffi_strength: float = 0.0


# ============================================================================
# DEPRECATED ALIASES (for backward compatibility)
# ============================================================================
# These will be removed in v0.4.0

import warnings

def _deprecated_alias(old_name: str, new_name: str):
    """Helper to create deprecation warning."""
    warnings.warn(
        f"{old_name} is deprecated and will be removed in v0.4.0. "
        f"Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )
