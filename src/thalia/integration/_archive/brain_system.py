"""
BrainSystem: Integrated multi-region brain with learnable pathways.

.. deprecated:: 0.2.0
   This module is deprecated. Use :class:`thalia.core.brain.EventDrivenBrain`
   instead, which provides event-driven processing with better dimension
   handling, input buffering, and cleaner architecture.

   Migration Example::

       # Old (deprecated)
       from thalia.integration import BrainSystem, BrainSystemConfig
       brain = BrainSystem(config)

       # New (recommended)
       from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig
       brain = EventDrivenBrain(config)

This module provides a complete brain system that composes individual
brain regions (Cortex, Hippocampus, PFC, Striatum, Cerebellum) with
learnable inter-region pathways.

COMPLETE BRAIN CONNECTIVITY MAP:
=================================

                    ┌──────────────────────────────────────────────────────┐
                    │              GLOBAL DOPAMINE SIGNAL                  │
                    │  (Reward/prediction error → modulates ALL regions)   │
                    └────────┬─────────┬─────────┬─────────┬───────────────┘
                             │         │         │         │
                             ▼         ▼         ▼         ▼
    Sensory Input      ┌─────────┐ ┌───────┐ ┌───────┐ ┌────────────┐
         │             │  CORTEX │ │ HIPPO │ │  PFC  │ │  STRIATUM  │
         │             └────┬────┘ └───┬───┘ └───┬───┘ └─────┬──────┘
         │                  │         │         │           │
         ▼                  │         │         │           │
    ┌─────────┐ ◄───────────┼─────────┼─────────┘           │
    │  CORTEX │ ◄───────────┼─────────┘                     │
    └────┬────┘             │  ▲                            │
         │                  │  │ AttentionPathway           │
         │                  │  │ (LEARNABLE: predictive)    │
    ┌────┴──────────────────┼──┼─────────────────────────────┤
    │         │             │  │           │                 │
    ▼         ▼             ▼  │           ▼                 │
┌───────┐ ┌───────┐    ┌────────┐    ┌──────────┐            │
│ HIPPO │◄┤  PFC  │◄───│ HIPPO  │    │ STRIATUM │◄───────────┘
└───┬───┘ └───┬───┘    └────────┘    └────┬─────┘   (Striatum
    │         │     (Hippo→PFC:           │         receives ALL)
    │         │      retrieval)           │
    │         │                           │
    │    ▼    │                           │
    │  ReplayPathway                      │
    │  (LEARNABLE: sleep-gated)           │
    │    │                                │
    │    ▼                                ▼
    │  ┌──────────────────────────────────────┐
    │  │           STRIATUM                   │ ◄── D1/D2 Opponent Process
    │  │   (Cortex + PFC + Hippo → Actions)   │     (LEARNABLE: reward-modulated)
    │  └──────────────┬───────────────────────┘
    │                 │
    │                 ▼
    │          ┌────────────┐
    │          │ Motor Cmd  │
    │          └──────┬─────┘
    │                 │
    │                 ▼
    │          ┌────────────┐              ┌──────────────────┐
    └─────────►│ CEREBELLUM │◄─────────────│ Error Signal     │
               └──────┬─────┘              │ (Climbing Fiber) │
                      │                    └──────────────────┘
                      ▼
                 Motor Output
                      │
                      ├───────────────────► Environment
                      │
                      ▼
            ┌────────────────┐
            │ Prediction vs  │──────────► Cerebellum Error
            │ Actual Outcome │
            └────────────────┘

COMPLETE CONNECTION LIST:
=========================
FEEDFORWARD (bottom-up):
  1. Sensory → Cortex (feature extraction)
  2. Cortex → Hippocampus (encoding)
  3. Cortex → PFC (sensory to WM)
  4. Cortex → Striatum (direct sensory-motor)
  5. Hippocampus → Striatum (memory-guided action)
  6. PFC → Striatum (goal-directed action)
  7. Striatum → Cerebellum (motor command)

FEEDBACK PATHWAYS (top-down, LEARNABLE!):
  8. PFC → Cortex: AttentionPathway (predictive learning)
     - Learns to predict cortex activity from PFC goals
     - Enables top-down attention and expectation
  9. Hippocampus → PFC (episodic retrieval)
  10. Hippocampus → Cortex: ReplayPathway (sleep-gated learning)
      - Only learns during sleep consolidation
      - Enables systems-level memory consolidation
  11. Cerebellum → Motor (refined output)

NEUROMODULATION:
  12. Global Dopamine → ALL regions (reward signal)
      - Modulates learning rates across all regions
      - Enables coordinated credit assignment

LEARNING RULES BY REGION:
=========================
  CORTEX: Unsupervised STDP (feature learning)
  HIPPOCAMPUS: Theta-gated STDP (episodic storage)
  PFC: Dopamine-gated STDP (WM update rules)
  STRIATUM: D1/D2 Opponent Process
     D1 (GO):   DA+ → LTP, DA- → LTD
     D2 (NOGO): DA+ → LTD, DA- → LTP (INVERTED!)
  CEREBELLUM: Error-corrective (motor refinement)

PATHWAY LEARNING RULES:
=======================
  AttentionPathway: PREDICTIVE
     - Strengthens connections that correctly predict cortex activity
     - Enables learned top-down attention
  ReplayPathway: REPLAY_GATED
     - Only updates during sleep (replay_active=True)
     - Dopamine-modulated for reward-biased consolidation
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import warnings
import torch
import torch.nn as nn

from ..regions.cortex import LayeredCortex, LayeredCortexConfig
from ..regions.hippocampus import TrisynapticHippocampus, TrisynapticConfig, Episode
from ..regions.prefrontal import Prefrontal, PrefrontalConfig
from ..regions.striatum import Striatum, StriatumConfig
from ..regions.cerebellum import Cerebellum, CerebellumConfig
from ..regions.base import LearningRule
from ..regions.theta_dynamics import ThetaState, ThetaConfig, TrialPhase, TemporalIntegrationLayer  # Global theta rhythm

from .pathways.spiking_attention import SpikingAttentionPathway, SpikingAttentionPathwayConfig
from .pathways.spiking_replay import SpikingReplayPathway, SpikingReplayPathwayConfig


@dataclass
class BrainSystemConfig:
    """Configuration for the integrated brain system.

    .. deprecated:: 0.2.0
       Use :class:`thalia.core.brain.EventDrivenBrainConfig` instead.

    Attributes:
        # Region sizes
        input_size: Size of sensory input
        cortex_size: Number of cortex neurons
        hippocampus_size: Number of hippocampus neurons
        pfc_size: Number of PFC neurons
        n_actions: Number of possible actions

        # Pathway settings
        attention_strength: PFC→Cortex modulation strength
        replay_strength: Hippo→Cortex replay strength

        # Learning settings
        pathway_learning_rate: Base learning rate for pathways

        # Striatum settings
        neurons_per_action: Population coding size

        device: Compute device
    """
    # Region sizes
    input_size: int = 256
    cortex_size: int = 128
    hippocampus_size: int = 64
    pfc_size: int = 32
    n_actions: int = 2

    # Time resolution (milliseconds per simulation timestep)
    # With 1ms timesteps, a 2ms refractory period = 2 timesteps
    dt_ms: float = 1.0

    # Comparison signal (for match-to-sample task)
    # Increased from 8 to 32 to make comparison signal more prominent
    # in striatum input (was 3.4% of input, now ~12%)
    comparison_size: int = 32

    # Pathway settings
    attention_strength: float = 0.3
    replay_strength: float = 0.5
    pathway_learning_rate: float = 0.01

    # Striatum settings
    neurons_per_action: int = 10

    # Display
    verbose: bool = True  # Print architecture diagram on init

    # Hippocampus NMDA and plasticity parameters
    hippo_nmda_tau: float = 50.0           # NMDA time constant (ms)
    hippo_nmda_threshold: float = 0.1      # Gate opening threshold
    hippo_nmda_steepness: float = 12.0     # Sigmoid sharpness
    hippo_ampa_ratio: float = 0.05         # AMPA ungated contribution
    hippo_ec_ca1_learning_rate: float = 0.5  # EC→CA1 plasticity rate
    hippo_ca3_learning_rate: float = 0.2   # CA3 recurrent learning rate
    hippo_dg_sparsity: float = 0.02        # DG sparsity (2%)
    hippo_ca3_sparsity: float = 0.10       # CA3 sparsity (10%)
    hippo_ca1_sparsity: float = 0.15       # CA1 sparsity (15%)

    # Cortex layer ratios (LayeredCortex: L4→L2/3→L5)
    cortex_l4_ratio: float = 1.0           # L4 size relative to cortex_size
    cortex_l23_ratio: float = 1.5          # L2/3 size (processing layer)
    cortex_l5_ratio: float = 1.0           # L5 size (subcortical output)

    # Short-Term Plasticity (STP) - enables temporal dynamics in hippocampus
    enable_stp: bool = False               # Enable STP in hippocampus pathways

    # BCM sliding threshold - enables metaplasticity in cortex
    enable_bcm: bool = False               # Enable BCM in cortex layers
    bcm_tau_theta: float = 1000.0          # BCM threshold adaptation time constant

    device: str = "cpu"


class BrainSystem(nn.Module):
    """
    Integrated multi-region brain system with learnable pathways.

    This class composes brain regions and inter-region pathways into
    a complete system capable of perception, memory, decision-making,
    and learning.

    Key features:
    1. Modular regions with different learning rules
    2. Learnable inter-region pathways
    3. Global neuromodulation (dopamine)
    4. Support for sleep consolidation

    Example:
        brain = BrainSystem(BrainSystemConfig(
            input_size=256,
            cortex_size=128,
            n_actions=2,
        ))

        # Process sample (encoding)
        sample_result = brain.process_sample(pattern)

        # Delay period (let states decay, PFC maintains with gate closed)
        delay_result = brain.inter_trial_interval(n_timesteps=10, dopamine=-0.3)

        # Test and respond
        test_result = brain.process_test_and_respond(test_pattern)

        # Learn from outcome
        brain.deliver_reward(reward=1.0)

        # Rest between trials
        brain.inter_trial_interval(n_timesteps=50)

        # Realistic sleep consolidation (multiple sleep stages)
        brain.realistic_sleep(n_cycles=3, replays_per_cycle=30)
    """

    def __init__(self, config: BrainSystemConfig):
        super().__init__()
        
        warnings.warn(
            "BrainSystem is deprecated and will be removed in a future version. "
            "Use thalia.core.brain.EventDrivenBrain instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.config = config

        # Global state
        self.global_dopamine = 0.0

        # Global theta rhythm (6-10 Hz, coordinates all brain regions)
        # Theta is generated by medial septum and drives encoding/retrieval phases
        self.theta = ThetaState(ThetaConfig(
            frequency_hz=8.0,  # 8 Hz theta
            dt_ms=config.dt_ms,
            encoding_phase_offset=0.0,     # Encoding at theta trough (0°)
            retrieval_phase_offset=3.14159,  # Retrieval at theta peak (180°)
            modulation_depth=0.5,          # 50% modulation
        ))

        # =====================================================================
        # CREATE BRAIN REGIONS
        # =====================================================================

        # 1. CORTEX: Feature extraction (LayeredCortex with L4→L2/3→L5)
        self.cortex = LayeredCortex(LayeredCortexConfig(
            n_input=config.input_size,
            n_output=config.cortex_size,
            dt_ms=config.dt_ms,
            l4_ratio=config.cortex_l4_ratio,
            l23_ratio=config.cortex_l23_ratio,
            l5_ratio=config.cortex_l5_ratio,
            l4_sparsity=0.15,
            l23_sparsity=0.10,
            l5_sparsity=0.20,
            stdp_lr=0.003,
            # BCM sliding threshold for metaplasticity
            bcm_enabled=config.enable_bcm,
            bcm_tau_theta=config.bcm_tau_theta,
            device=config.device,
        ))
        # For LayeredCortex:
        # - L2/3 output goes to hippocampus, PFC (cortical targets)
        # - L5 output goes to striatum (subcortical target)
        self._cortex_l23_size = self.cortex.l23_size
        self._cortex_l5_size = self.cortex.l5_size
        cortex_to_hippo_size = self._cortex_l23_size  # L2/3 → hippocampus
        cortex_to_striatum_size = self._cortex_l5_size  # L5 → striatum

        # 2. HIPPOCAMPUS: Episodic memory with DG→CA3→CA1 trisynaptic circuit
        self.hippocampus = TrisynapticHippocampus(TrisynapticConfig(
            n_input=cortex_to_hippo_size,  # Receives L2/3 output
            n_output=config.hippocampus_size,
            dt_ms=config.dt_ms,
            # Sparsity settings
            dg_sparsity=config.hippo_dg_sparsity,
            ca3_sparsity=config.hippo_ca3_sparsity,
            ca1_sparsity=config.hippo_ca1_sparsity,
            ca3_recurrent_strength=0.4,
            # NMDA gating parameters
            nmda_tau=config.hippo_nmda_tau,
            nmda_threshold=config.hippo_nmda_threshold,
            nmda_steepness=config.hippo_nmda_steepness,
            ampa_ratio=config.hippo_ampa_ratio,
            # Learning rates
            learning_rate=config.hippo_ca3_learning_rate,
            ec_ca1_learning_rate=config.hippo_ec_ca1_learning_rate,
            # EC layer III input size (raw sensory input for temporoammonic path)
            # This enables proper match/mismatch discrimination by using the
            # original sensory pattern for EC→CA1, not the cortex output
            ec_l3_input_size=config.input_size,
            # Short-Term Plasticity for temporal dynamics
            stp_enabled=config.enable_stp,
            device=config.device,
        ))

        # TEMPORAL INTEGRATION LAYER (Entorhinal Cortex simulation)
        # This layer sits between cortex and hippocampus and solves the problem
        # of sparse, temporally variable cortex output. Real EC layer II/III
        # has slow membrane dynamics (~50-100ms tau) that integrate cortical
        # input over a theta cycle before projecting to hippocampus.
        #
        # Without this: Cortex outputs 1-8 spikes/timestep, different neurons
        # each time. Hippocampus NMDA trace never builds up, EC→CA1 learning
        # is scattered, coincidence detection fails.
        #
        # With this: Accumulated cortex activity creates a STABLE pattern that
        # the same neurons produce each timestep. NMDA trace builds up,
        # EC→CA1 learns coherently, match/mismatch discrimination works.
        self.cortex_to_hippo_integrator = TemporalIntegrationLayer(
            n_neurons=cortex_to_hippo_size,
            tau=50.0,         # 50ms integration (roughly half theta cycle)
            threshold=0.5,    # Threshold for spike generation
            gain=2.0,         # Gain for rate-to-spike conversion
            device=torch.device(config.device),
        )

        # 3. PFC: Working memory
        # PFC receives L2/3 cortical output + hippocampus
        pfc_input_size = cortex_to_hippo_size + config.hippocampus_size
        self.pfc = Prefrontal(PrefrontalConfig(
            n_input=pfc_input_size,
            n_output=config.pfc_size,
            dt_ms=config.dt_ms,
            stdp_lr=0.01,
            dopamine_baseline=0.3,
            gate_threshold=0.5,
            wm_decay_tau_ms=200.0,
            soft_bounds=True,
            synaptic_scaling_enabled=True,
            device=config.device,
        ))

        # =====================================================================
        # MATCH/MISMATCH DETECTOR (Opponent Population Coding)
        # =====================================================================
        # The hippocampus CA1 outputs HIGH activity for match (NMDA gates open)
        # and LOW activity for mismatch (NMDA gates closed). But the striatum
        # can't learn from ABSENCE of signal (zero input = no learning).
        #
        # This detector converts CA1 activity level into GRADED opponent signals:
        # - Match population: baseline + gain * similarity
        # - Mismatch population: baseline + gain * (1 - similarity)
        #
        # This ensures BOTH populations always have some activity, with the
        # RELATIVE activation encoding match vs mismatch. The striatum can
        # learn to map these opponent patterns to different actions.
        #
        # Biological basis: Real brains use opponent coding extensively
        # (ON/OFF cells in retina, D1/D2 in striatum, etc.). This avoids
        # the "silence is ambiguous" problem.
        self._comparison_match_size = config.comparison_size // 2
        self._comparison_mismatch_size = config.comparison_size - self._comparison_match_size

        # Graded opponent coding parameters
        # Increased baseline and gain for stronger signal to striatum
        self._comparison_baseline = 0.3   # Tonic baseline (ensures non-zero activity)
        self._comparison_gain = 0.7       # Gain for similarity modulation

        # Track running average of CA1 activity for adaptive normalization
        self._ca1_activity_ema = 0.0
        self._ca1_activity_ema_alpha = 0.1
        self._ca1_activity_var_ema = 0.01  # Variance EMA for novelty detection

        # Accumulator for CA1 activity during test phase
        # Reset at start of each test, accumulates spikes for robust discrimination
        self._ca1_accumulated = 0.0

        # =====================================================================
        # NOVELTY DETECTOR (Neuromodulatory Gating)
        # =====================================================================
        # Novelty detection is crucial for learning - unexpected events should
        # trigger heightened plasticity. In real brains:
        # - Locus coeruleus (norepinephrine) responds to unexpected stimuli
        # - Basal forebrain (acetylcholine) signals novelty/uncertainty
        # - VTA (dopamine) responds to prediction errors
        #
        # We compute novelty as deviation from expected CA1 activity:
        # - High novelty = CA1 activity far from running average
        # - Novelty boosts learning rate via neuromodulatory gain
        #
        # This helps the system learn faster from surprising events (like
        # the first few mismatch trials) and slower from predictable ones.
        self._novelty_signal = 0.0           # Current novelty level [0, 1]
        self._novelty_baseline_lr = 1.0      # Base learning rate multiplier
        self._novelty_max_boost = 2.0        # Max learning rate boost for novel events
        self._novelty_decay = 0.9            # Decay rate for novelty signal

        # 4. STRIATUM: Action selection with D1/D2
        # Striatum receives:
        # - L5 cortical output (subcortical pathway)
        # - PFC output
        # - Hippocampus output
        # - Match/Mismatch comparison signal (NEW!)
        # The comparison signal explicitly encodes whether CA1 is active or not.
        striatum_input_size = (
            self._cortex_l5_size +  # L5 goes to subcortical (striatum)
            config.pfc_size +
            config.hippocampus_size +
            config.comparison_size  # Match + Mismatch detector neurons
        )
        self.striatum = Striatum(StriatumConfig(
            n_input=striatum_input_size,
            n_output=config.n_actions,
            dt_ms=config.dt_ms,
            neurons_per_action=config.neurons_per_action,
            device=config.device,
        ))

        # 5. CEREBELLUM: Motor refinement
        striatum_output_size = config.n_actions * config.neurons_per_action
        self.cerebellum = Cerebellum(CerebellumConfig(
            n_input=striatum_output_size,
            n_output=config.n_actions,
            dt_ms=config.dt_ms,
            device=config.device,
        ))

        # =====================================================================
        # CREATE LEARNABLE PATHWAYS (all spiking with temporal dynamics)
        # =====================================================================

        # PFC → Cortex (spiking attention with phase coding)
        self.attention_pathway = SpikingAttentionPathway(SpikingAttentionPathwayConfig(
            source_size=config.pfc_size,
            target_size=config.cortex_size,
            input_size=config.input_size,
            cortex_size=config.cortex_size,
            stdp_lr=config.pathway_learning_rate,
        ))

        # Hippocampus → Cortex (spiking replay with temporal coding)
        self.replay_pathway = SpikingReplayPathway(SpikingReplayPathwayConfig(
            source_size=config.hippocampus_size,
            target_size=config.cortex_size,
            replay_gain=config.replay_strength * 5,  # Compensate for spiking sparsity
            stdp_lr=config.pathway_learning_rate * 2,
        ))

        # =====================================================================
        # INITIALIZE WEIGHTS
        # =====================================================================
        self._initialize_all_weights()

        # Working memory state (for delay period)
        self._sample_cortex: Optional[torch.Tensor] = None
        self._sample_hippo: Optional[torch.Tensor] = None

        # Show architecture if verbose
        if config.verbose:
            self.print_architecture()

    def print_architecture(self) -> None:
        """Print ASCII connectivity diagram of the brain system."""
        diagram = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                       BRAIN SYSTEM ARCHITECTURE                              ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║                            SENSORY INPUT                                    ║
║                                 │                                           ║
║                                 ▼                                           ║
║   ┌─────────────────────────────────────────────────────────────────────┐   ║
║   │                            CORTEX                                   │   ║
║   │          (Feature extraction, unsupervised STDP learning)           │   ║
║   │                      Size: {:>4} neurons                            │   ║
║   └───────────┬────────────────────────────────────┬────────────────────┘   ║
║               │                                    │                        ║
║               │                                    │                        ║
║               ▼                                    ▼                        ║
║   ┌─────────────────────────┐          ┌─────────────────────────────┐      ║
║   │      HIPPOCAMPUS        │          │            PFC              │      ║
║   │   (Pattern separation,  │          │   (Working memory + control) │     ║
║   │   episodic memory,      │          │   Size: {:>4} neurons        │      ║
║   │   prioritized replay)   │ ─ ─ ─ ─▶ │   Receives: cortex + hippo   │     ║
║   │   Size: {:>4} neurons    │◀─ ─ ─ ─  │                              │      ║
║   └───────────┬─────────────┘          └───────────────┬─────────────┘      ║
║               │                                        │                    ║
║               │ ⚡ ReplayPathway                        │ ⚡ AttentionPathway ║
║               │  (sleep-gated)                         │  (error-driven)    ║
║               │                                        │                    ║
║               ▼                                        ▼                    ║
║   ┌─────────────────────────────────────────────────────────────────────┐   ║
║   │                           STRIATUM                                  │   ║
║   │   (D1/D2 opponent process, action selection, reward learning)       │   ║
║   │   Actions: {:>2}  |  Neurons/action: {:>2}  |  D1/D2: {}             │   ║
║   └───────────────────────────────┬─────────────────────────────────────┘   ║
║                                   │                                         ║
║                                   ▼                                         ║
║   ┌─────────────────────────────────────────────────────────────────────┐   ║
║   │                          CEREBELLUM                                 │   ║
║   │              (Motor refinement, error correction)                   │   ║
║   │              Output: {:>2} motor commands                            │   ║
║   └─────────────────────────────────────────────────────────────────────┘   ║
║                                                                             ║
║   ───────────────────────────────────────────────────────────────────────   ║
║   ⚡ = Learnable pathway with STDP                                          ║
║   ─▶ = Feedforward projection                                              ║
║   ─ ─▶ = Bidirectional/feedback projection                                 ║
╚═════════════════════════════════════════════════════════════════════════════╝
""".format(
            self.config.cortex_size,
            self.config.pfc_size,
            self.config.hippocampus_size,
            self.config.n_actions,
            self.config.neurons_per_action,
            "ON",  # D1/D2 is always enabled
            self.config.n_actions,
        )
        print(diagram)

    def _initialize_all_weights(self) -> None:
        """Initialize all region and pathway weights."""
        with torch.no_grad():
            # Cortex: diagonal bias for receptive fields
            self._init_cortex_weights()

            # Hippocampus: sparse random for pattern separation
            self._init_hippocampus_weights()

            # PFC: block structure (cortex vs hippo preference)
            self._init_pfc_weights()

            # Striatum: asymmetric for action differentiation
            self._init_striatum_weights()

            # Cerebellum: small uniform
            self.cerebellum.weights.data = (
                torch.rand_like(self.cerebellum.weights) * 0.2 + 0.05
            )

    def _init_cortex_weights(self) -> None:
        """Initialize cortex with diagonal bias."""
        # LayeredCortex handles its own weight initialization
        # but we can add diagonal bias to input→L4 weights
        w = self.cortex.w_input_l4  # type: ignore[union-attr]
        with torch.no_grad():
            w.data = torch.rand_like(w) * 0.2 + 0.05
            n_in = w.shape[1]
            n_out = w.shape[0]
            for i in range(n_out):
                center = int(i * n_in / n_out)
                width = max(1, n_in // n_out) * 2
                for j in range(max(0, center - width), min(n_in, center + width)):
                    distance = abs(j - center) / max(1, width)
                    boost = 0.4 * (1.0 - distance)
                    w.data[i, j] += boost

    def _init_hippocampus_weights(self) -> None:
        """Initialize hippocampus weights.

        NOTE: TrisynapticHippocampus now handles its own weight initialization
        with proper sparse connectivity in _init_circuit_weights(). We skip
        re-initialization here to preserve the biologically-inspired sparsity:

        - EC→DG: 30% sparse (pattern separation)
        - DG→CA3: 50% sparse (mossy fibers)
        - CA3→CA1: 15% sparse (Schaffer collaterals, NOT normalized)
        - EC→CA1: 20% sparse (perforant path, NOT normalized)

        The CA3→CA1 and EC→CA1 pathways are specifically NOT row-normalized
        so that NMDA coincidence detection works properly - only CA1 neurons
        that receive strong input from BOTH pathways should fire.
        """
        # Trisynaptic hippocampus handles its own weight initialization
        pass

    def _init_pfc_weights(self) -> None:
        """Initialize PFC with block structure."""
        self.pfc.weights.data = (
            torch.rand_like(self.pfc.weights) * 0.2 + 0.05
        )

        n_pfc = self.pfc.weights.shape[0]
        cortex_inputs = self.config.cortex_size

        # First half prefers cortex
        self.pfc.weights.data[:n_pfc//2, :cortex_inputs] += 0.25
        # Second half prefers hippocampus
        self.pfc.weights.data[n_pfc//2:, cortex_inputs:] += 0.25

    def _init_striatum_weights(self) -> None:
        """Initialize striatum with asymmetric bias for actions.

        The comparison signal is at the end of the input:
        [L5 cortex | PFC | Hippocampus | Match neurons | Mismatch neurons]

        We bias:
        - Action 0 (MATCH) population weights towards match signal neurons
        - Action 1 (NO-MATCH) population weights towards mismatch signal neurons
        """
        self.striatum.weights.data = (
            torch.rand_like(self.striatum.weights) * 0.3 + 0.1
        )

        neurons_per_action = self.config.neurons_per_action
        n_inputs = self.striatum.weights.shape[1]
        comparison_start = n_inputs - self.config.comparison_size
        match_end = comparison_start + self._comparison_match_size

        # MATCH population (action 0) → match signal neurons
        self.striatum.weights.data[
            :neurons_per_action,
            comparison_start:match_end
        ] += 0.5

        # NO-MATCH population (action 1) → mismatch signal neurons
        self.striatum.weights.data[
            neurons_per_action:,
            match_end:
        ] += 0.5

        # Initialize D1/D2 weights with SYMMETRIC distributions
        # This is critical for balanced learning - any asymmetry causes one pathway to dominate
        self.striatum.d1_weights.data = (
            torch.rand_like(self.striatum.d1_weights) * 0.2 + 0.1
        )
        self.striatum.d2_weights.data = (
            torch.rand_like(self.striatum.d2_weights) * 0.2 + 0.1  # SYMMETRIC with D1!
        )

        # D1 weights: Bias GO signal for appropriate action
        self.striatum.d1_weights.data[
            :neurons_per_action,
            comparison_start:match_end
        ] += 0.3  # Action 0 responds to match
        self.striatum.d1_weights.data[
            neurons_per_action:,
            match_end:
        ] += 0.3  # Action 1 responds to mismatch

        # FORCE INITIAL BALANCE: Even with same distribution, random draws differ.
        # Apply aggressive baseline pressure at init to ensure D1 ≈ D2 per action.
        for _ in range(10):  # Apply pressure 10 times
            self.striatum._apply_baseline_pressure()

    # =========================================================================
    # CORTEX OUTPUT ROUTING HELPERS
    # =========================================================================

    def _get_cortex_to_hippo(self, cortex_out: torch.Tensor) -> torch.Tensor:
        """Get cortex output for hippocampus (L2/3 for layered, full for single).

        Uses temporal integration to create stable patterns from sparse variable
        cortex output. This simulates the Entorhinal Cortex layer II/III which
        has slow membrane dynamics that integrate cortical input before
        projecting to hippocampus.

        The integration solves the problem where cortex L2/3 fires only 1-8
        spikes per timestep with different neurons each time, which breaks
        the hippocampus NMDA coincidence detection mechanism.
        """
        # LayeredCortex returns [L2/3, L5] concatenated
        # Split and return L2/3 for cortical targets (hippocampus)
        raw_l23 = cortex_out[:, :self._cortex_l23_size]

        # Apply temporal integration for stable hippocampus input
        integrated = self.cortex_to_hippo_integrator.integrate(raw_l23)
        return integrated

    def _get_cortex_to_pfc(self, cortex_out: torch.Tensor) -> torch.Tensor:
        """Get cortex output for PFC (L2/3 layer)."""
        return cortex_out[:, :self._cortex_l23_size]

    def _get_cortex_to_striatum(self, cortex_out: torch.Tensor) -> torch.Tensor:
        """Get cortex output for striatum (L5 layer)."""
        return cortex_out[:, self._cortex_l23_size:]

    def _compute_comparison_signal(self, hippo_activity: torch.Tensor, n_timesteps: int = 15, current_timestep: int = 0) -> torch.Tensor:
        """Compute match/mismatch comparison signal using TEMPORAL BURST CODING.

        Instead of rate-coded activation levels, we use synchronized bursts:
        - Accumulates CA1 spikes over the test phase for robust discrimination
        - At decision time (last few timesteps), generates a SYNCHRONIZED BURST
          on either match or mismatch population based on accumulated evidence

        Temporal burst coding advantages:
        - Matches the temporal coding used elsewhere in the system
        - Creates strong, precise eligibility traces in striatum via STDP
        - Synchronized spikes summate more effectively in postsynaptic neurons

        Biological basis:
        - Hippocampal sharp-wave ripples: synchronized population bursts
        - Decision-related bursting in prefrontal and striatal circuits
        - Temporal coincidence detection in striatum

        Args:
            hippo_activity: CA1 spike output for current timestep (batch, hippocampus_size)
            n_timesteps: Total timesteps in test phase (for normalization)
            current_timestep: Current timestep within the test phase

        Returns:
            Comparison signal (comparison_size,) as binary spikes (0 or 1)
        """
        # Accumulate CA1 activity over the test phase
        ca1_sum_instant = hippo_activity.sum().item()
        self._ca1_accumulated += ca1_sum_instant

        # Normalize by maximum possible accumulated spikes over the test phase
        ca1_max_possible = self.config.hippocampus_size * n_timesteps
        ca1_normalized = self._ca1_accumulated / max(1.0, ca1_max_possible)
        ca1_normalized = min(1.0, ca1_normalized)

        # Update running statistics for novelty detection
        old_ema = self._ca1_activity_ema
        self._ca1_activity_ema = (
            self._ca1_activity_ema_alpha * ca1_normalized +
            (1 - self._ca1_activity_ema_alpha) * self._ca1_activity_ema
        )
        deviation_sq = (ca1_normalized - old_ema) ** 2
        self._ca1_activity_var_ema = (
            self._ca1_activity_ema_alpha * deviation_sq +
            (1 - self._ca1_activity_ema_alpha) * self._ca1_activity_var_ema
        )

        # Novelty detection
        std_estimate = max(0.01, self._ca1_activity_var_ema ** 0.5)
        deviation = abs(ca1_normalized - self._ca1_activity_ema)
        raw_novelty = deviation / std_estimate
        self._novelty_signal = float(min(1.0, raw_novelty / 3.0))

        # =====================================================================
        # TEMPORAL BURST CODING
        # =====================================================================
        # Instead of graded activation, we generate synchronized bursts.
        # The decision is made based on accumulated CA1 activity:
        # - High CA1 (match) → burst on match population
        # - Low CA1 (mismatch) → burst on mismatch population
        #
        # Bursts occur in the DECISION WINDOW (last 3-5 timesteps of test phase)
        # This aligns with biological decision-related bursting.

        comparison = torch.zeros(self.config.comparison_size)

        # Scale similarity for discrimination
        similarity = min(1.0, ca1_normalized * 4.0)

        # Determine if this is the decision window (last 5 timesteps)
        decision_window_start = max(0, n_timesteps - 5)
        in_decision_window = current_timestep >= decision_window_start

        # Store similarity for diagnostics (accessible after trial)
        self._last_similarity = similarity

        if in_decision_window:
            # Decision threshold: similarity > 0.5 means MATCH
            is_match = similarity > 0.5

            # Store decision for diagnostics
            self._last_comparison_decision = "MATCH" if is_match else "NOMATCH"

            # Generate synchronized burst on appropriate population
            # Burst = all neurons in the population fire together
            # Add some jitter for biological realism (not all fire every timestep)
            burst_prob = 0.8  # 80% of neurons fire in each burst timestep

            if is_match:
                # Burst on match population
                match_spikes = (torch.rand(self._comparison_match_size) < burst_prob).float()
                comparison[:self._comparison_match_size] = match_spikes
                # Sparse activity on mismatch (background noise)
                comparison[self._comparison_match_size:] = (torch.rand(self._comparison_mismatch_size) < 0.1).float()
            else:
                # Burst on mismatch population
                mismatch_spikes = (torch.rand(self._comparison_mismatch_size) < burst_prob).float()
                comparison[self._comparison_match_size:] = mismatch_spikes
                # Sparse activity on match (background noise)
                comparison[:self._comparison_match_size] = (torch.rand(self._comparison_match_size) < 0.1).float()
        else:
            # Before decision window: sparse background activity
            # This prevents eligibility traces from building up prematurely
            comparison = (torch.rand(self.config.comparison_size) < 0.05).float()

        return comparison

    def get_novelty_learning_boost(self) -> float:
        """Get learning rate multiplier based on current novelty.

        Novelty modulates learning rate via neuromodulatory gating:
        - Low novelty (expected events): learning_rate * 1.0
        - High novelty (unexpected events): learning_rate * max_boost

        This implements the biological principle that surprising events
        should trigger stronger plasticity (via NE, ACh, DA modulation).

        Returns:
            Learning rate multiplier in [1.0, novelty_max_boost]
        """
        # Linear interpolation between baseline and max boost
        boost = (
            self._novelty_baseline_lr +
            self._novelty_signal * (self._novelty_max_boost - self._novelty_baseline_lr)
        )
        return boost

    def inter_trial_interval(
        self,
        n_timesteps: int = 50,
        dopamine: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Run network with null input to let states decay naturally.

        This handles both:
        1. Between-trial rest (default, dopamine=0.0)
        2. Within-trial delay period (dopamine=-0.3 for gate closed)

        What happens during this period:
        1. No sensory input (just baseline noise)
        2. Membrane potentials naturally decay via LIF dynamics
        3. Spike-frequency adaptation (SFA) decays back to baseline
        4. Lateral inhibition (recent_spikes) decays
        5. CA1 membranes decay (critical for match/mismatch detection!)
        6. Striatum membranes decay (prevent carryover between trials!)
        7. Weights (memories) are PRESERVED

        Note: Default of 50 timesteps allows adaptation states to decay
        significantly (SFA: 50 steps at tau=200 → ~78% decay of exp(-50/200)).
        For the cortex lateral inhibition (0.9 decay), 50 steps → 0.9^50 ≈ 0.5%

        Args:
            n_timesteps: Duration of rest period (default 50 = ~50ms)
            dopamine: PFC dopamine level (0.0 = neutral, -0.3 = gate closed)

        Returns:
            Dict with PFC activity
        """
        null_input = torch.zeros(1, self.config.input_size)
        pfc_total = torch.zeros(self.config.pfc_size)

        # Create null input for striatum (L5 + PFC + hippo + comparison)
        striatum_null = torch.zeros(
            1,
            self._cortex_l5_size + self.config.pfc_size +
            self.config.hippocampus_size + self.config.comparison_size
        )

        for _ in range(n_timesteps):
            # Advance global theta rhythm
            self.theta.advance()

            # Get current theta modulation
            enc_mod = self.theta.encoding_strength
            ret_mod = self.theta.retrieval_strength

            # Cortex processes null input - membrane decays naturally
            cortex_out = self.cortex.forward(
                null_input,
                encoding_mod=enc_mod,
                retrieval_mod=ret_mod,
            )

            # Hippocampus in DELAY phase: CA3 maintains, CA1 idles
            # This is NOT test mode - no comparison happens yet
            cortex_to_hippo = self._get_cortex_to_hippo(cortex_out)
            hippo_out = self.hippocampus.forward(
                cortex_to_hippo,
                phase=TrialPhase.DELAY,  # CA1 just idles, no comparison
                encoding_mod=enc_mod,
                retrieval_mod=ret_mod,
            )

            # PFC receives L2/3 cortex + hippo
            cortex_to_pfc = self._get_cortex_to_pfc(cortex_out)
            pfc_input = torch.cat([cortex_to_pfc.squeeze(), hippo_out.squeeze()])
            pfc_out = self.pfc.forward(
                pfc_input.unsqueeze(0),
                dopamine_signal=dopamine,
                encoding_mod=enc_mod,
                retrieval_mod=ret_mod,
            )
            pfc_total += pfc_out.squeeze()

            # CRITICAL: Run striatum with null input so membrane potentials decay!
            # Without this, striatum neurons accumulate activity across trials
            # and become saturated (all neurons fire regardless of input)
            self.striatum.forward(striatum_null, explore=False)

        return {"pfc": pfc_total}

    def _settle_for_replay(self, n_timesteps: int = 5) -> None:
        """Let neural activity settle before a sleep replay.

        During sleep, we don't want to artificially reset() state.
        Instead, we let activity naturally decay through LIF dynamics
        with minimal input. This is biologically realistic:

        1. Between sharp-wave ripples, there's low background activity
        2. Membrane potentials decay toward rest
        3. But previous activity leaves some trace (not hard reset)

        This is a shorter version of inter_trial_interval optimized for
        the rapid succession of replays during sleep.

        Args:
            n_timesteps: Brief settling period (default 5 = ~5ms)
        """
        null_input = torch.zeros(1, self.config.input_size)

        # Striatum null input includes comparison signal
        striatum_null = torch.zeros(
            1,
            self._cortex_l5_size + self.config.pfc_size +
            self.config.hippocampus_size + self.config.comparison_size
        )

        for _ in range(n_timesteps):
            # Run cortex with no input - activity decays naturally
            cortex_out = self.cortex.forward(null_input)

            # Run striatum with no input - also decays naturally
            self.striatum.forward(striatum_null, explore=False)

    def process_sample(
        self,
        sample_pattern: torch.Tensor,
        n_timesteps: int = 15,
    ) -> Dict[str, torch.Tensor]:
        """Process sample pattern (encoding phase).

        Routes information through:
        1. PFC attention → Cortex (top-down modulation)
        2. Cortex → Hippocampus (encoding via Hebbian learning in CA3)
        3. Cortex + Hippo → PFC (working memory update)

        Note: Memory is stored in CA3 recurrent weights via fast Hebbian
        plasticity during this phase. No explicit "trace" is stored.

        Args:
            sample_pattern: Input pattern to encode
            n_timesteps: Number of processing timesteps

        Returns:
            Dict with region activities (for monitoring only)
        """
        sample_spikes = self._rate_to_spikes(sample_pattern, n_timesteps)

        # These are for monitoring only - not used for computation
        # Use L2/3 size for layered cortex monitoring
        cortex_total = torch.zeros(self._cortex_l23_size)
        hippo_total = torch.zeros(self.config.hippocampus_size)
        pfc_total = torch.zeros(self.config.pfc_size)

        pfc_activity = torch.zeros(1, self.config.pfc_size)

        # NOTE: No clearing of hippocampus state!
        # The CA3 recurrent weights accumulate patterns across presentations.
        # This is how memory works - patterns strengthen existing attractors.

        # Reset global theta phase for new trial (align to optimal encoding phase)
        self.theta.reset()

        # Prepare hippocampus for new trial (clears FFI)
        self.hippocampus.new_trial()

        # Clear temporal integration layer for new pattern
        # This ensures each trial starts fresh with no accumulated activity
        self.cortex_to_hippo_integrator.clear()

        # Prepare cortex for new trial (clears FFI input history)
        # This ensures FFI will detect the first stimulus as "new" and
        # transiently suppress any lingering recurrent activity
        if hasattr(self.cortex, 'new_trial'):
            self.cortex.new_trial()

        for t in range(n_timesteps):
            # Advance global theta rhythm
            self.theta.advance()

            # Get current theta modulation values
            enc_mod = self.theta.encoding_strength
            ret_mod = self.theta.retrieval_strength

            # 1. Apply top-down attention from PFC
            # sample_spikes[t] is already (batch, input_size)
            modulated_input = self.attention_pathway.modulate(
                sample_spikes[t],
                pfc_activity,
            )

            # 2. Cortex processes modulated input (theta modulates input gain)
            cortex_out = self.cortex.forward(
                modulated_input,
                encoding_mod=enc_mod,
                retrieval_mod=ret_mod,
            )

            # 3. Route cortex outputs to appropriate targets
            # L2/3 → hippocampus (cortical target)
            # L5 → will go to striatum later (subcortical target)
            cortex_to_hippo = self._get_cortex_to_hippo(cortex_out)
            cortex_to_pfc = self._get_cortex_to_pfc(cortex_out)
            cortex_total += cortex_to_hippo.squeeze()  # Monitor L2/3 output

            # 4. Hippocampus encodes via Hebbian learning in CA3 recurrent weights
            # phase=ENCODE triggers Hebbian learning in CA3 and EC→CA1 alignment
            # Theta modulation: encoding_strength is high at theta trough
            # ec_direct_input: Pass raw sensory pattern for EC L3 → CA1 pathway
            hippo_out = self.hippocampus.forward(
                cortex_to_hippo,
                phase=TrialPhase.ENCODE,
                encoding_mod=enc_mod,
                retrieval_mod=ret_mod,
                ec_direct_input=sample_spikes[t],  # Raw sensory input for EC→CA1
            )
            hippo_total += hippo_out.squeeze()  # For monitoring

            # 5. PFC receives L2/3 cortex + hippocampus (theta modulates gating)
            pfc_input = torch.cat([cortex_to_pfc.squeeze(), hippo_out.squeeze()])
            pfc_out = self.pfc.forward(
                pfc_input.unsqueeze(0),
                dopamine_signal=0.5,  # High DA = gate open
                encoding_mod=enc_mod,
                retrieval_mod=ret_mod,
            )
            pfc_total += pfc_out.squeeze()
            pfc_activity = pfc_out

        return {
            "cortex": cortex_total,
            "hippocampus": hippo_total,
            "pfc": pfc_total,
        }

    def process_test_and_respond(
        self,
        test_pattern: torch.Tensor,
        n_timesteps: int = 15,
        explore: bool = True,
    ) -> Dict[str, Any]:
        """Process test pattern and select action.

        Routes through all regions and computes comparison signal
        to determine if test matches sample.

        The new theta-based system handles CA1 clearing naturally:
        - Feedforward inhibition detects the stimulus change from delay→test
        - FFI transiently suppresses residual CA1 activity
        - This clears the slate for NMDA coincidence detection

        No explicit reset needed anymore!

        Args:
            test_pattern: Test pattern to compare
            n_timesteps: Processing timesteps
            explore: Whether to use exploration

        Returns:
            Dict with activities and selected action
        """
        # NOTE: We no longer explicitly reset cortex recurrent state here!
        # The Feedforward Inhibition (FFI) mechanism in LayeredCortex will
        # automatically detect the stimulus change from delay→test and
        # transiently suppress recurrent activity. This is biologically
        # realistic - fast-spiking interneurons detect input changes and
        # briefly inhibit pyramidal cells.

        # CRITICAL: Clear the temporal integration layer for test phase!
        # The test pattern needs to build its own integrated representation.
        # The sample pattern's integration was for ENCODING (to train EC→CA1).
        # Now we need a fresh integration to CREATE the test input pattern
        # that will be compared against the LEARNED EC→CA1 mapping.
        self.cortex_to_hippo_integrator.clear()

        # Reset CA1 activity accumulator for fresh comparison signal computation
        # This accumulates CA1 spikes over the test phase for robust match/mismatch detection
        self._ca1_accumulated = 0.0

        # Reset striatum neuron state for fresh action selection
        # Without this, membrane potential from previous trial could prevent spiking
        self.striatum.neurons.reset_state(1)
        self.striatum.state.t = 0  # Reset timestep counter

        # Reset D1/D2 vote accumulators for fresh trial-level decision
        self.striatum.reset_accumulated_votes()

        test_spikes = self._rate_to_spikes(test_pattern, n_timesteps)

        cortex_total = torch.zeros(self._cortex_l23_size)  # Monitor L2/3
        hippo_total = torch.zeros(self.config.hippocampus_size)
        pfc_total = torch.zeros(self.config.pfc_size)
        # Striatum output is n_actions * neurons_per_action when using population coding
        striatum_output_size = self.config.n_actions * self.config.neurons_per_action
        striatum_total = torch.zeros(striatum_output_size)
        cerebellum_total = torch.zeros(self.config.n_actions)

        pfc_activity = torch.zeros(1, self.config.pfc_size)
        # Striatum receives L5 + PFC + hippo
        combined_input = torch.zeros(
            self._cortex_l5_size +
            self.config.pfc_size +
            self.config.hippocampus_size
        )

        for t in range(n_timesteps):
            # Advance global theta rhythm
            self.theta.advance()

            # Get current theta modulation
            enc_mod = self.theta.encoding_strength
            ret_mod = self.theta.retrieval_strength

            # 1. Top-down attention
            modulated_input = self.attention_pathway.modulate(
                test_spikes[t],  # Already has batch dimension from _rate_to_spikes
                pfc_activity,
            )

            # 2. Cortex (theta modulates input/recurrence balance)
            cortex_out = self.cortex.forward(
                modulated_input,
                encoding_mod=enc_mod,
                retrieval_mod=ret_mod,
            )

            # 3. Route outputs to appropriate targets
            cortex_to_hippo = self._get_cortex_to_hippo(cortex_out)
            cortex_to_pfc = self._get_cortex_to_pfc(cortex_out)
            cortex_to_striatum = self._get_cortex_to_striatum(cortex_out)
            cortex_total += cortex_to_hippo.squeeze()  # Monitor L2/3

            # 4. Hippocampus RETRIEVE phase (test/retrieval)
            # CA1 spikes ARE the output - different patterns emerge naturally
            # for match vs mismatch through NMDA coincidence detection
            # FFI will clear residual activity on first timestep
            # Theta modulation: retrieval_strength is high at theta peak
            # ec_direct_input: Pass raw test pattern for EC L3 → CA1 pathway
            hippo_out = self.hippocampus.forward(
                cortex_to_hippo,
                phase=TrialPhase.RETRIEVE,
                encoding_mod=enc_mod,
                retrieval_mod=ret_mod,
                ec_direct_input=test_spikes[t],  # Raw sensory input for EC→CA1
            )
            hippo_total += hippo_out.squeeze()

            # 5. PFC receives L2/3 + hippo (theta modulates gating)
            pfc_input = torch.cat([cortex_to_pfc.squeeze(), hippo_out.squeeze()])
            pfc_out = self.pfc.forward(
                pfc_input.unsqueeze(0),
                dopamine_signal=-0.2,
                encoding_mod=enc_mod,
                retrieval_mod=ret_mod,
            )
            pfc_total += pfc_out.squeeze()
            pfc_activity = pfc_out

            # 6. Compute match/mismatch comparison signal using TEMPORAL BURST CODING
            # Accumulates CA1 activity, then generates synchronized bursts in decision window
            comparison_signal = self._compute_comparison_signal(hippo_out, n_timesteps, current_timestep=t)

            # 7. Striatum receives L5 (subcortical) + PFC + hippo + comparison
            # The comparison signal ensures striatum gets distinct input for both
            # match (high CA1 + match signal) and mismatch (low CA1 + mismatch signal)
            combined_input = torch.cat([
                cortex_to_striatum.squeeze(),  # L5 output
                pfc_out.squeeze(),
                hippo_out.squeeze(),
                comparison_signal,  # Match/mismatch detector output
            ])
            striatum_out = self.striatum.forward(
                combined_input.unsqueeze(0),
                explore=explore,
                encoding_mod=enc_mod,
                retrieval_mod=ret_mod,
            )
            striatum_total += striatum_out.squeeze()

            # 7. Cerebellum refines
            cerebellum_out = self.cerebellum.forward(
                striatum_out,
                encoding_mod=enc_mod,
                retrieval_mod=ret_mod,
            )
            cerebellum_total += cerebellum_out.squeeze()

        # Delegate final action selection to Striatum.finalize_action
        # This applies UCB + softmax/argmax + exploration, updates counts ONCE,
        # and sets striatum.last_action for reward learning
        finalize_result = self.striatum.finalize_action(explore=explore)
        selected_action = int(finalize_result["selected_action"])

        return {
            "cortex": cortex_total,
            "hippocampus": hippo_total,
            "pfc": pfc_total,
            "striatum": striatum_total,
            "cerebellum": cerebellum_total,
            "selected_action": selected_action,
            "explored": finalize_result["exploring"],
            "combined_input": combined_input,
            # Diagnostics from finalize_action
            "net_votes": finalize_result["net_votes"],
            "ucb_bonus": finalize_result["ucb_bonus"],
            "exploration_prob": finalize_result["exploration_prob"],
            # Hippocampus comparison diagnostics
            "ca1_similarity": getattr(self, '_last_similarity', 0.0),
            "comparison_decision": getattr(self, '_last_comparison_decision', 'UNKNOWN'),
        }

    def deliver_reward(
        self,
        reward: float,
    ) -> Dict[str, Any]:
        """Deliver reward signal to update weights with novelty modulation.

        The brain ALWAYS LEARNS — there is no separate evaluation mode.
        The reward signal is modulated by the current novelty level:
        - High novelty (unexpected events) → boosted learning rate
        - Low novelty (expected events) → baseline learning rate

        This implements the biological principle that surprising events
        trigger stronger plasticity via neuromodulatory gating (NE, ACh, DA).

        Args:
            reward: Reward value (+1 for correct, -1 for incorrect)

        Returns:
            Dict with learning metrics including novelty information
        """
        # Get novelty-based learning boost
        novelty_boost = self.get_novelty_learning_boost()

        # Modulate reward by novelty for enhanced learning from surprising events
        # Note: We boost the magnitude, not the sign (both positive and negative
        # rewards are boosted for novel events)
        modulated_reward = reward * novelty_boost

        self.global_dopamine = modulated_reward

        # Update striatum with novelty-modulated reward
        striatum_result = self.striatum.deliver_reward(modulated_reward)

        # Reset eligibility after learning (explicit control)
        # NOTE: When using deliver_reward_with_counterfactual, that method
        # handles the reset AFTER counterfactual learning is also done.
        self.striatum.reset_eligibility()

        # Update attention pathway with novelty-modulated dopamine
        attention_result = self.attention_pathway.learn(
            source_activity=torch.zeros(self.config.pfc_size),  # Will use traces
            target_activity=torch.zeros(self.config.cortex_size),
            dopamine=modulated_reward,
        )

        # Apply homeostatic scaling to maintain stable striatum activity
        # This prevents D2 from dominating D1 over time
        homeostatic_result = self.striatum.apply_homeostatic_scaling()

        return {
            "striatum": striatum_result,
            "attention_pathway": attention_result,
            "novelty": self._novelty_signal,
            "novelty_boost": novelty_boost,
            "modulated_reward": modulated_reward,
            "homeostatic": homeostatic_result,
        }

    def deliver_reward_with_counterfactual(
        self,
        reward: float,
        is_match: bool,
        selected_action: int,
        counterfactual_scale: float = 0.5,
    ) -> Dict[str, Any]:
        """Deliver reward with counterfactual learning for the non-selected action.

        This implements model-based RL: after experiencing a real outcome, we also
        simulate "what would have happened if I had chosen differently?" and learn
        from that imagined outcome.

        This solves the problem of asymmetric learning where only the selected
        action gets updated. Now BOTH actions learn on every trial:
        - Selected action: learns from actual outcome
        - Non-selected action: learns from counterfactual (imagined) outcome

        Example:
        - Trial is NOMATCH, we select MATCH (wrong), get punished
        - Real learning: MATCH pathway gets punished (D1↓, D2↑)
        - Counterfactual: "If I had selected NOMATCH, I'd get reward"
        - Counterfactual learning: NOMATCH pathway gets rewarded (D1↑, D2↓)

        This is biologically plausible: the hippocampus + PFC can "simulate"
        outcomes without executing, and dopamine responds to predicted rewards.

        Args:
            reward: Actual reward received
            is_match: Whether this was a match trial (determines counterfactual)
            selected_action: The action that was actually taken (0=MATCH, 1=NOMATCH)
            counterfactual_scale: How much to scale counterfactual learning (0-1)

        Returns:
            Dict with both real and counterfactual learning metrics
        """
        # Get novelty-based learning boost (same as regular deliver_reward)
        novelty_boost = self.get_novelty_learning_boost()
        modulated_reward = reward * novelty_boost
        self.global_dopamine = modulated_reward

        # 1. Real learning: update striatum for SELECTED action
        # NOTE: We call striatum.deliver_reward directly (not self.deliver_reward)
        # to avoid the automatic eligibility reset - we need eligibility for counterfactual!
        real_result = self.striatum.deliver_reward(modulated_reward)

        # 2. Counterfactual learning: what would the OTHER action have gotten?
        other_action = 1 - selected_action  # 0→1 or 1→0

        # Determine counterfactual reward:
        # - If trial is MATCH: MATCH action (0) would get +1, NOMATCH (1) would get -1
        # - If trial is NOMATCH: NOMATCH action (1) would get +1, MATCH (0) would get -1
        correct_action = 0 if is_match else 1  # MATCH=0 is correct for match trials
        counterfactual_reward = 1.0 if (other_action == correct_action) else -1.0

        # Apply counterfactual learning to the non-selected action
        # Uses the SAME eligibility traces (current input state) - this is key!
        counterfactual_result = self.striatum.deliver_counterfactual_reward(
            reward=counterfactual_reward,
            action=other_action,
            counterfactual_scale=counterfactual_scale,
        )

        # 3. NOW reset eligibility after BOTH learnings are done
        self.striatum.reset_eligibility()

        # 4. Also update attention pathway (like regular deliver_reward)
        attention_result = self.attention_pathway.learn(
            source_activity=torch.zeros(self.config.pfc_size),
            target_activity=torch.zeros(self.config.cortex_size),
            dopamine=modulated_reward,
        )

        # 5. Apply homeostatic scaling
        homeostatic_result = self.striatum.apply_homeostatic_scaling()

        return {
            "real": real_result,
            "counterfactual": counterfactual_result,
            "selected_action": selected_action,
            "other_action": other_action,
            "counterfactual_reward": counterfactual_reward,
            "attention_pathway": attention_result,
            "homeostatic": homeostatic_result,
            "novelty_boost": novelty_boost,
        }

    def realistic_sleep(
        self,
        n_cycles: int = 3,
        replays_per_cycle: int = 30,
        n_timesteps: int = 10,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run realistic sleep with proper stage transitions.

        Models biological sleep architecture with all stages:

        SLEEP CYCLE STRUCTURE (per ~90 min cycle in real brain):
        =========================================================
        Wake → N1 (drowsy) → N2 (light sleep) → N3/SWS (deep) → REM → N2 → ...

        Stage Characteristics:
        - N1 (drowsy): Falling asleep, theta waves, minimal consolidation
        - N2 (light): Sleep spindles (12-14 Hz), some consolidation
        - N3/SWS: Delta waves, sharp-wave ripples, MAXIMUM consolidation
        - REM: Theta waves, stochastic replay, generalization

        Sleep Architecture Changes Through Night:
        - Early cycles: More SWS (memory consolidation priority)
        - Late cycles: More REM (generalization, creativity)

        Args:
            n_cycles: Number of sleep cycles (real brain: 4-6 per night)
            replays_per_cycle: Total replays per cycle
            n_timesteps: Timesteps per replay
            verbose: Print cycle details

        Returns:
            Dict with comprehensive sleep metrics
        """
        if len(self.hippocampus.episode_buffer) == 0:
            return {"cycles": 0, "total_replays": 0}

        # Track overall metrics
        total_replays = 0
        total_n2_replays = 0
        total_sws_replays = 0
        total_rem_replays = 0
        # D1/D2 is always enabled
        d1_start = self.striatum.d1_weights.clone()
        d2_start = self.striatum.d2_weights.clone()

        cycle_metrics: List[Dict[str, Any]] = []

        for cycle_idx in range(n_cycles):
            # Biological sleep architecture:
            # - Early night: ~10% N2, 70% SWS, 20% REM
            # - Late night: ~15% N2, 35% SWS, 50% REM
            progress = cycle_idx / max(1, n_cycles - 1)
            n2_fraction = 0.10 + 0.05 * progress
            sws_fraction = 0.70 - 0.35 * progress
            rem_fraction = 1.0 - n2_fraction - sws_fraction

            n_n2 = max(1, int(replays_per_cycle * n2_fraction))
            n_sws = int(replays_per_cycle * sws_fraction)
            n_rem = replays_per_cycle - n_sws - n_n2

            # D1/D2 is always enabled
            cycle_d1_before = self.striatum.d1_weights.clone()

            # ================================================================
            # N1 STAGE (DROWSY / FALLING ASLEEP)
            # ================================================================
            # Characterized by:
            # - Alpha waves (8-12 Hz) transition to theta waves (4-8 Hz)
            # - Reduced responsiveness to external stimuli
            # - Hypnagogic experiences (spontaneous imagery/thoughts)
            # - Muscle tone begins to decrease
            #
            # We simulate this as a brief transition where:
            # - Neural states decay (membrane potentials, adaptation)
            # - Some spontaneous activity occurs (driven by intrinsic noise)
            # - Theta oscillations begin modulating hippocampus

            self.replay_pathway.set_sleep_stage("n1")

            # Run brief transition period (like winding down)
            # This lets adaptation states decay and allows spontaneous activity
            n1_timesteps = 10  # Brief transition
            self._run_drowsy_transition(n1_timesteps)

            # ================================================================
            # N2 STAGE (LIGHT SLEEP WITH SPINDLES)
            # ================================================================
            # Characterized by:
            # - Sleep spindles (12-14 Hz bursts) - important for memory!
            # - K-complexes (high amplitude waves)
            # - Moderate consolidation, prepares for SWS

            self.replay_pathway.set_sleep_stage("n2")
            n2_match = 0
            n2_nomatch = 0

            # N2 has moderate, somewhat random replay (spindle-like)
            n2_episodes = self._sample_episodes_for_n2(n_n2)

            for episode in n2_episodes:
                # Let activity settle naturally (no artificial reset)
                self._settle_for_replay(n_timesteps=5)

                # Moderate learning during N2 (spindle consolidation)
                for _ in range(n_timesteps):
                    striatum_input = episode.state.unsqueeze(0)
                    self.striatum.forward(striatum_input, explore=False)
                    self.striatum.last_action = episode.action

                # Moderate reward strength during N2
                self.striatum.deliver_reward(
                    reward=episode.reward * 0.6,  # 60% strength
                    expected=0.0,
                )

                is_match = episode.metadata.get("is_match", True) if episode.metadata else True
                if is_match:
                    n2_match += 1
                else:
                    n2_nomatch += 1

            total_n2_replays += len(n2_episodes)

            # ================================================================
            # SLOW-WAVE SLEEP (N3/SWS) PHASE
            # ================================================================
            # Characterized by:
            # - Sharp-wave ripples (150-200 Hz bursts)
            # - Priority-weighted replay (reward-biased)
            # - High learning rate for consolidation
            # - Cortical slow oscillations coordinate with hippocampal replay

            self.replay_pathway.set_sleep_stage("sws")
            sws_match = 0
            sws_nomatch = 0

            # Sample priority-weighted episodes for SWS
            episodes = self.hippocampus.sample_episodes_prioritized(n_sws)

            for episode in episodes:
                # Let activity settle naturally (no artificial reset)
                self._settle_for_replay(n_timesteps=5)

                # Replay through striatum with FULL learning
                for _ in range(n_timesteps):
                    striatum_input = episode.state.unsqueeze(0)
                    self.striatum.forward(striatum_input, explore=False)
                    self.striatum.last_action = episode.action

                # Deliver reward with FULL strength during SWS
                self.striatum.deliver_reward(
                    reward=episode.reward,
                    expected=0.0,
                )

                # Hippocampus → Cortex consolidation
                # During SWS, hippocampal replay triggers cortical reactivation
                # This coordinated activity strengthens hippo→cortex connections
                if episode.context is not None:
                    hippo_activity = episode.state[:self.config.hippocampus_size].unsqueeze(0)

                    # Trigger a replay ripple and get the replay signal
                    self.replay_pathway.trigger_ripple()
                    replay_signal = self.replay_pathway.replay_step(dt=1.0)

                    if replay_signal is not None:
                        # Run cortex with replay signal to get real cortical reactivation
                        # Biologically: sharp-wave ripples trigger coordinated cortical response
                        cortex_input = torch.zeros(1, self.config.input_size)
                        # Project replay signal to input space (simplified)
                        min_size = min(replay_signal.shape[1], self.config.input_size)
                        cortex_input[:, :min_size] = replay_signal[:, :min_size]
                        cortex_activity = self.cortex.forward(cortex_input)

                        # Learn hippo→cortex mapping with REAL cortex activity
                        self.replay_pathway.learn(
                            pre_spikes=hippo_activity,
                            post_spikes=cortex_activity,
                            reward=0.5,  # Consolidation reward
                            dt=1.0,
                        )

                # Track
                is_match = episode.metadata.get("is_match", True) if episode.metadata else True
                if is_match:
                    sws_match += 1
                else:
                    sws_nomatch += 1

            total_sws_replays += len(episodes)

            # ================================================================
            # REM SLEEP PHASE
            # ================================================================
            # Characterized by:
            # - Theta oscillations (4-8 Hz)
            # - More stochastic/random replay
            # - Lower learning rate (exploration)
            # - May combine disparate memories
            # - Important for generalization

            self.replay_pathway.set_sleep_stage("rem")
            rem_match = 0
            rem_nomatch = 0

            # Sample more randomly for REM (less priority-biased)
            # This helps with generalization and prevents overfitting
            rem_episodes = self._sample_episodes_for_rem(n_rem)

            for episode in rem_episodes:
                # Let activity settle naturally (no artificial reset)
                self._settle_for_replay(n_timesteps=5)

                # Replay with REDUCED learning during REM
                # REM is characterized by theta oscillations (4-8 Hz)
                # Theta modulates the timing of replay and adds variability
                for t in range(n_timesteps):
                    # Theta modulation (~6 Hz during REM)
                    theta_phase = 2 * 3.14159 * 6.0 * t / 1000.0
                    theta_mod = 0.5 * (1 + torch.cos(torch.tensor(theta_phase)))

                    striatum_input = episode.state.unsqueeze(0)
                    # Theta-modulated stochastic noise for REM-like variability
                    # This creates the "creative" associations of REM sleep
                    noise = torch.randn_like(striatum_input) * 0.1 * theta_mod
                    self.striatum.forward(striatum_input + noise, explore=False)
                    self.striatum.last_action = episode.action

                # Deliver reward with REDUCED strength during REM
                # REM is more about exploration/generalization than consolidation
                self.striatum.deliver_reward(
                    reward=episode.reward * 0.3,  # Reduced reward signal
                    expected=0.0,
                )

                # Track
                is_match = episode.metadata.get("is_match", True) if episode.metadata else True
                if is_match:
                    rem_match += 1
                else:
                    rem_nomatch += 1

            total_rem_replays += len(rem_episodes)
            total_replays += len(n2_episodes) + len(episodes) + len(rem_episodes)

            # Compute per-cycle metrics (D1/D2 always enabled)
            cycle_d1_delta = (self.striatum.d1_weights - cycle_d1_before).abs().mean().item()

            cycle_metrics.append({
                "cycle": cycle_idx + 1,
                "n2_replays": len(n2_episodes),
                "sws_replays": len(episodes),
                "rem_replays": len(rem_episodes),
                "n2_fraction": n2_fraction,
                "sws_fraction": sws_fraction,
                "rem_fraction": rem_fraction,
                "n2_match": n2_match,
                "n2_nomatch": n2_nomatch,
                "sws_match": sws_match,
                "sws_nomatch": sws_nomatch,
                "rem_match": rem_match,
                "rem_nomatch": rem_nomatch,
                "d1_delta": cycle_d1_delta,
            })

            if verbose:
                print(f"  Sleep Cycle {cycle_idx + 1}/{n_cycles}: "
                      f"N2={len(n2_episodes)}, "
                      f"SWS={len(episodes)} (M:{sws_match}/NM:{sws_nomatch}), "
                      f"REM={len(rem_episodes)} (M:{rem_match}/NM:{rem_nomatch}), "
                      f"ΔD1={cycle_d1_delta:.4f}")

        # Return to wake state
        self.replay_pathway.set_sleep_stage("wake")

        # Compute total weight changes (D1/D2 always enabled)
        d1_total_delta = (self.striatum.d1_weights - d1_start).abs().mean().item()
        d2_total_delta = (self.striatum.d2_weights - d2_start).abs().mean().item()

        return {
            "cycles": n_cycles,
            "total_replays": total_replays,
            "n2_replays": total_n2_replays,
            "sws_replays": total_sws_replays,
            "rem_replays": total_rem_replays,
            "d1_delta": d1_total_delta,
            "d2_delta": d2_total_delta,
            "cycle_details": cycle_metrics,
        }

    def _sample_episodes_for_n2(self, n: int) -> List[Episode]:
        """
        Sample episodes for N2 (light sleep with spindles).

        N2 spindles have moderate priority bias - between random and
        fully priority-weighted. Spindles help transfer memories from
        hippocampus to cortex.
        """
        if len(self.hippocampus.episode_buffer) == 0:
            return []

        n = min(n, len(self.hippocampus.episode_buffer))

        # Moderate priority bias for spindle-related consolidation
        priorities = torch.tensor([
            ep.priority for ep in self.hippocampus.episode_buffer
        ])

        # Add moderate noise (less than REM, more than SWS)
        noisy_priorities = priorities + torch.randn_like(priorities) * 0.3

        # Softmax with moderate temperature
        probs = torch.softmax(noisy_priorities * 1.0, dim=0)  # temperature = 1.0

        indices = torch.multinomial(probs, n, replacement=False)
        return [self.hippocampus.episode_buffer[i] for i in indices]

    def _sample_episodes_for_rem(self, n: int) -> List[Episode]:
        """
        Sample episodes for REM sleep with less priority bias.

        REM sleep is characterized by more stochastic/random replay,
        which helps with generalization and creative associations.
        """
        if len(self.hippocampus.episode_buffer) == 0:
            return []

        n = min(n, len(self.hippocampus.episode_buffer))

        # Use softmax with lower temperature for more uniform sampling
        # (less priority-biased than SWS)
        priorities = torch.tensor([
            ep.priority for ep in self.hippocampus.episode_buffer
        ])

        # Add noise to priorities for stochasticity
        noisy_priorities = priorities + torch.randn_like(priorities) * 0.5

        # Softmax with lower temperature (more uniform)
        probs = torch.softmax(noisy_priorities * 0.5, dim=0)  # temperature = 2.0

        indices = torch.multinomial(probs, n, replacement=False)
        return [self.hippocampus.episode_buffer[i] for i in indices]

    def _run_drowsy_transition(self, n_timesteps: int = 10) -> None:
        """
        Run N1 (drowsy) transition period.

        During N1:
        - Alpha waves give way to theta waves (4-8 Hz)
        - Neural activity winds down
        - Spontaneous activity occurs (hypnagogic experiences)
        - This is similar to inter_trial_interval but with theta modulation

        Theta oscillations during drowsiness help:
        - Gate hippocampal activity
        - Prepare for deeper sleep consolidation
        - Allow some spontaneous memory reactivation

        Args:
            n_timesteps: Duration of drowsy transition
        """
        null_input = torch.zeros(1, self.config.input_size)

        for _ in range(n_timesteps):
            # Advance global theta rhythm
            self.theta.advance()

            # Get current theta modulation
            enc_mod = self.theta.encoding_strength
            ret_mod = self.theta.retrieval_strength

            # Theta-modulated spontaneous activity
            # During N1, theta waves (4-8 Hz) begin to dominate
            # Small amount of spontaneous input (noise-driven activity)
            spontaneous_input = null_input + torch.randn_like(null_input) * 0.05 * enc_mod

            # Cortex processes spontaneous activity (theta modulates)
            cortex_out = self.cortex.forward(
                spontaneous_input,
                encoding_mod=enc_mod,
                retrieval_mod=ret_mod,
            )

            # Route to appropriate targets
            cortex_to_hippo = self._get_cortex_to_hippo(cortex_out)
            cortex_to_pfc = self._get_cortex_to_pfc(cortex_out)

            # Hippocampus may have spontaneous retrievals during drowsiness
            # This is the basis of hypnagogic imagery!
            hippo_out = self.hippocampus.forward(
                cortex_to_hippo,
                phase=TrialPhase.RETRIEVE,  # Retrieval mode - may surface old memories
                encoding_mod=enc_mod,
                retrieval_mod=ret_mod,
            )

            # PFC maintains minimal activity (consciousness fading)
            pfc_input = torch.cat([cortex_to_pfc.squeeze(), hippo_out.squeeze()])
            self.pfc.forward(
                pfc_input.unsqueeze(0),
                dopamine_signal=-0.5,  # Very low DA = minimal gating
                encoding_mod=enc_mod,
                retrieval_mod=ret_mod,
            )

    def store_experience(
        self,
        combined_input: torch.Tensor,
        is_match: bool,
        selected_action: int,
        correct: bool,
        reward: float,
        sample_pattern: Optional[torch.Tensor] = None,
        test_pattern: Optional[torch.Tensor] = None,
    ) -> None:
        """Store experience for later replay.

        Delegates to hippocampus with priority boosting for
        rare/important experiences.

        Args:
            combined_input: Striatum input from trial
            is_match: Whether trial was a match
            selected_action: Action that was taken
            correct: Whether action was correct
            reward: Reward received
            sample_pattern: Original sample (optional)
            test_pattern: Test pattern (optional)
        """
        priority_boost = 0.0

        # Boost correct NOMATCH selections (rare!)
        if correct and selected_action == 1:
            priority_boost += 3.0

        # Boost NO-MATCH trials generally
        if not is_match:
            priority_boost += 1.5

        self.hippocampus.store_episode(
            state=combined_input,
            action=selected_action,
            reward=reward,
            correct=correct,
            context=sample_pattern,
            metadata={
                "is_match": is_match,
                "test_pattern": test_pattern.clone() if test_pattern is not None else None,
            },
            priority_boost=priority_boost,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information from all regions.

        This aggregates diagnostics from all brain regions and pathways,
        providing a complete snapshot of brain state for debugging.

        Returns:
            Dict with keys:
            - regions: Per-region diagnostics (striatum, hippocampus, etc.)
            - pathways: Per-pathway diagnostics (attention, replay)
            - state: Global state (dopamine, last action, etc.)
            - summary: Quick overview stats
        """
        # =====================================================================
        # REGION DIAGNOSTICS
        # =====================================================================
        regions = {}

        # Striatum (primary region for action selection)
        if hasattr(self.striatum, 'get_diagnostics'):
            regions["striatum"] = self.striatum.get_diagnostics()
        else:
            regions["striatum"] = {"error": "no get_diagnostics method"}

        # Hippocampus (memory and comparison)
        if hasattr(self.hippocampus, 'get_diagnostics'):
            regions["hippocampus"] = self.hippocampus.get_diagnostics()
        else:
            regions["hippocampus"] = {"error": "no get_diagnostics method"}

        # Cortex (basic firing rate for now)
        regions["cortex"] = {
            "firing_rate": (
                self.cortex.state.spikes.mean().item()
                if self.cortex.state.spikes is not None else 0.0
            ),
            "weight_mean": self.cortex.w_input_l4.data.mean().item() if hasattr(self.cortex, 'w_input_l4') else 0.0,
        }

        # PFC
        regions["pfc"] = {
            "firing_rate": (
                self.pfc.state.spikes.mean().item()
                if self.pfc.state.spikes is not None else 0.0
            ),
        }

        # =====================================================================
        # PATHWAY DIAGNOSTICS
        # =====================================================================
        pathways = {
            "attention": self.attention_pathway.get_diagnostics() if hasattr(self.attention_pathway, 'get_diagnostics') else {},
            "replay": self.replay_pathway.get_diagnostics() if hasattr(self.replay_pathway, 'get_diagnostics') else {},
        }

        # =====================================================================
        # GLOBAL STATE
        # =====================================================================
        state = {
            "dopamine_level": self.global_dopamine,
            "last_ca1_similarity": getattr(self, '_last_ca1_similarity', None),
            "last_comparison_decision": getattr(self, '_last_comparison_decision', None),
        }

        # =====================================================================
        # SUMMARY (quick overview)
        # =====================================================================
        # Extract key metrics for quick debugging
        striatum_diag = regions.get("striatum", {})
        hippo_diag = regions.get("hippocampus", {})

        summary = {
            # Action selection state
            "last_action": striatum_diag.get("last_action"),
            "exploring": striatum_diag.get("exploration", {}).get("exploring", False),

            # D1/D2 balance (per-action)
            "net_weight_means": striatum_diag.get("net_weight_means", []),
            "net_votes": striatum_diag.get("net_votes", []),

            # Hippocampus comparison (critical for match/mismatch)
            "ca1_spikes": hippo_diag.get("ca1", {}).get("spikes", 0),
            "nmda_above_threshold": hippo_diag.get("nmda", {}).get("above_threshold_count", 0),
            "mg_block_removal": hippo_diag.get("nmda", {}).get("mg_block_removal_mean", 0.0),
            "dg_similarity": hippo_diag.get("pattern_comparison", {}).get("dg_similarity", 0.0),

            # Dopamine
            "dopamine": self.global_dopamine,
        }

        return {
            "regions": regions,
            "pathways": pathways,
            "state": state,
            "summary": summary,
        }

    def _rate_to_spikes(
        self,
        pattern: torch.Tensor,
        n_timesteps: int,
        max_rate: float = 0.7,
    ) -> torch.Tensor:
        """Convert rate pattern to spike train."""
        spikes = torch.zeros(n_timesteps, *pattern.shape)
        for t in range(n_timesteps):
            spikes[t] = (torch.rand_like(pattern) < pattern * max_rate).float()
        return spikes
