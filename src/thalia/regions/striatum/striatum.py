"""
Striatum - Reinforcement Learning with Three-Factor Rule

The striatum (part of basal ganglia) learns through dopamine-modulated
plasticity, implementing the classic three-factor learning rule for
reinforcement learning.

Key Features:
=============
1. THREE-FACTOR LEARNING: Δw = eligibility × dopamine
   - Pre-post activity creates eligibility traces
   - Eligibility alone does NOT cause plasticity
   - Dopamine arriving later converts eligibility to weight change
   - DA burst → LTP, DA dip → LTD, No DA → no learning

2. DOPAMINE as REWARD PREDICTION ERROR:
   - Burst: "Better than expected" → reinforce recent actions
   - Dip: "Worse than expected" → weaken recent actions
   - Baseline: "As expected" → maintain current policy

3. LONG ELIGIBILITY TRACES:
   - Biological tau: 500-2000ms (Yagishita et al., 2014)
   - Allows credit assignment for delayed rewards
   - Synaptic tag persists until dopamine arrives

4. **ACTION SELECTION** (Winner-Take-All):
   - Lateral inhibition creates competition between action neurons
   - Winning action's synapses become eligible for learning
   - Dopamine retroactively credits (burst) or blames (dip) the winner
   - Losers' eligibility decays without reinforcement

FILE ORGANIZATION (~1950 lines)
================================
Lines 1-150:     Module docstring, imports, class registration
Lines 151-400:   __init__() and pathway initialization (D1/D2)
Lines 401-650:   Forward pass coordination (D1/D2 integration)
Lines 651-850:   Action selection logic (winner-take-all)
Lines 851-1050:  Three-factor learning (eligibility × dopamine)
Lines 1051-1250: Exploration (UCB-based) and homeostasis
Lines 1251-1450: Growth and neurogenesis
Lines 1451-1650: Diagnostics and health monitoring
Lines 1651-1950: Utility methods and state management

QUICK NAVIGATION
================
VSCode shortcuts:
  • Ctrl+Shift+O (Cmd+Shift+O on Mac) - "Go to Symbol" for method jumping
  • Ctrl+K Ctrl+0 - Collapse all regions to see file outline
  • Ctrl+K Ctrl+J - Expand all regions
  • Ctrl+G - Go to specific line number
  • Ctrl+F - Search within file

Key methods to jump to:
  • __init__() - Initialization and D1/D2 pathway setup
  • forward() - Main forward pass coordination
  • select_action() - Action selection logic
  • update_eligibility() - Eligibility trace management
  • apply_three_factor_learning() - Dopamine-gated plasticity
  • grow_output() / grow_input() - Neurogenesis
  • get_diagnostics() - Health monitoring

WHY THIS FILE IS LARGE
======================
The striatum coordinates two opponent pathways (D1 "Go", D2 "No-Go") that
must interact every timestep for action selection. Splitting would:
1. Require passing D1/D2 votes, eligibility, action selection state
2. Duplicate dopamine broadcast logic
3. Obscure the opponent pathway interaction
4. Break action selection coherence

Components ARE extracted where appropriate:
- D1Pathway, D2Pathway: Parallel pathway implementations (extracted because
  they compute independently, ADR-011)
- StriatumLearningComponent: Three-factor learning logic
- StriatumHomeostasisComponent: E/I balance
- StriatumExplorationComponent: UCB exploration
- ActionSelectionMixin: Winner-take-all logic

See: docs/decisions/adr-011-large-file-justification.md

**Biological Basis**:
====================
- **Medium Spiny Neurons (MSNs)**: 95% of striatal neurons
- **D1-MSNs (direct pathway)**: Express D1 receptors, DA → LTP → "Go" signal
- **D2-MSNs (indirect pathway)**: Express D2 receptors, DA → LTD → "No-Go" signal
- **Opponent Processing**: D1 promotes, D2 suppresses actions
- **References**:
  - Schultz et al. (1997): Dopamine reward prediction error hypothesis
  - Yagishita et al. (2014): Direct evidence for synaptic tagging in vivo
  - Frank (2005): Dynamic dopamine modulation in basal ganglia

**When to Use**:
================
- Reinforcement learning from rewards/punishments (not supervised labels)
- Action selection and sequential decision-making
- Habit learning and procedural memory
- Delayed reward credit assignment (eligibility traces bridge temporal gaps)
- Goal-directed behavior with value-based choice
- Trial-and-error learning scenarios
"""

from __future__ import annotations

import weakref
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from thalia.components.gap_junctions import GapJunctionConfig, GapJunctionCoupling
from thalia.components.neurons import (
    E_EXCITATORY,
    E_INHIBITORY,
    E_LEAK,
    V_RESET_STANDARD,
    V_THRESHOLD_STANDARD,
    ConductanceLIF,
    ConductanceLIFConfig,
    create_fast_spiking_neurons,
)
from thalia.components.synapses import (
    ShortTermPlasticity,
    WeightInitializer,
    create_heterogeneous_stp_configs,
    get_stp_config,
)
from thalia.constants.neuromodulation import compute_ne_gain
from thalia.core.diagnostics_schema import (
    compute_activity_metrics,
    compute_health_metrics,
    compute_plasticity_metrics,
)
from thalia.core.neural_region import NeuralRegion
from thalia.managers.base_manager import ManagerContext
from thalia.managers.component_registry import register_region
from thalia.neuromodulation import ACH_BASELINE, NE_BASELINE
from thalia.regions.striatum.exploration import ExplorationConfig
from thalia.typing import SourceOutputs, StateDict
from thalia.utils.core_utils import clamp_weights
from thalia.utils.oscillator_utils import compute_theta_encoding_retrieval

from .action_selection import ActionSelectionMixin
from .checkpoint_manager import StriatumCheckpointManager
from .config import StriatumConfig, StriatumState
from .d1_pathway import D1Pathway
from .d2_pathway import D2Pathway
from .exploration_component import StriatumExplorationComponent
from .forward_coordinator import ForwardPassCoordinator
from .homeostasis_component import HomeostasisManagerConfig, StriatumHomeostasisComponent
from .learning_component import StriatumLearningComponent
from .pathway_base import StriatumPathwayConfig
from .state_tracker import StriatumStateTracker
from .td_lambda import TDLambdaConfig, TDLambdaLearner


@register_region(
    "striatum",
    aliases=["basal_ganglia"],
    description="Reinforcement learning via dopamine-modulated three-factor rule with D1/D2 opponent pathways",
    version="2.1",  # Updated for NeuralRegion migration
    author="Thalia Project",
    config_class=StriatumConfig,
)
class Striatum(NeuralRegion, ActionSelectionMixin):
    """Striatal region with three-factor reinforcement learning.

    **Phase 2 Migration**: Now inherits from NeuralRegion with biologically-accurate
    synaptic weight placement at target dendrites (not in axonal pathways).

    Implements dopamine-modulated learning:
    - Eligibility traces tag recently active synapses
    - Dopamine signal converts eligibility to plasticity
    - No learning without dopamine (unlike Hebbian)
    - Synaptic weights stored per-source in synaptic_weights dict

    Population Coding (optional):
    - Instead of 1 neuron per action, use N neurons per action
    - Decision = which population has highest total spike count
    - Benefits: noise reduction, redundancy, graded confidence

    Mixins Provide:
    ---------------
    From ActionSelectionMixin:
        - select_action_softmax(q_values, temperature) → int
        - select_action_greedy(q_values, epsilon) → int
        - compute_policy(q_values, temperature) → Tensor
        - add_exploration_noise(q_values, noise_std) → Tensor

    From NeuralRegion (base class):
        - synaptic_weights: ParameterDict (per-source weights)
        - add_input_source(source, n_input, learning_rule) → None
        - _apply_synapses(source, input_spikes) → Tensor
        - forward(inputs: Dict) → Tensor [must implement]

    See Also:
        docs/patterns/mixins.md for detailed mixin patterns
        docs/patterns/component-parity.md for component design patterns
    """

    def __init__(self, config: StriatumConfig, sizes: Dict[str, int], device: str = "cpu"):
        """Initialize Striatum with D1/D2 opponent pathways.

        **New Pattern (January 2026)**: Config + Sizes + Device separation

        Args:
            config: Behavioral configuration (no sizes)
            sizes: Size specification dict from LayerSizeCalculator
            device: Device for tensors ("cpu" or "cuda")
        """
        if not isinstance(config, StriatumConfig):
            raise TypeError(
                "Striatum requires StriatumConfig. "
                "Use: Striatum(config=StriatumConfig(...), sizes={...}, device='cpu')"
            )

        # Store config and device
        self.config: StriatumConfig = config
        self.device = torch.device(device)

        # Extract sizes from dict
        self.n_actions = sizes["n_actions"]
        self.d1_size = sizes["d1_size"]
        self.d2_size = sizes["d2_size"]
        self.input_size = sizes.get("input_size", 0)
        self.neurons_per_action = sizes.get("neurons_per_action", 10)

        # Validate sizes
        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be positive, got {self.n_actions}")
        if self.d1_size <= 0 or self.d2_size <= 0:
            raise ValueError(
                f"Pathway sizes must be positive. Got d1={self.d1_size}, d2={self.d2_size}"
            )

        # Total neurons = D1 + D2
        total_neurons = self.d1_size + self.d2_size

        # =====================================================================
        # INITIALIZE NEURAL REGION (Phase 2)
        # =====================================================================
        # NeuralRegion handles synaptic weights per-source in synaptic_weights dict
        # D1/D2 pathways will be neuron populations only (no weights)
        NeuralRegion.__init__(
            self,
            n_neurons=total_neurons,  # Use actual total (d1+d2)
            default_learning_rule="three_factor",  # Dopamine-modulated
            device=device,
            dt_ms=config.dt_ms,
        )

        # Store n_output for NeuralRegion interface
        self.n_output = total_neurons

        # Store D1/D2 sizes for backward compatibility with code that uses n_d1/n_d2
        self.n_d1 = self.d1_size
        self.n_d2 = self.d2_size

        # =====================================================================
        # MULTI-SOURCE ELIGIBILITY TRACES (Phase 3 + Phase 1 Enhancement)
        # =====================================================================
        # Per-source-pathway eligibility traces for multi-source learning
        # Structure: {"source_d1": tensor, "source_d2": tensor, ...}
        #
        # Phase 1 Enhancement: Multi-timescale eligibility traces
        # Biology: Synaptic tags (eligibility) exist at multiple timescales:
        # - Fast traces (~500ms): Immediate coincidence detection (STDP-like)
        # - Slow traces (~60s): Consolidated long-term tags for delayed reward
        # Combined eligibility = fast + α*slow enables both rapid and multi-second
        # credit assignment. (Yagishita et al. 2014, Shindou et al. 2019)
        self._eligibility_d1: StateDict = {}
        self._eligibility_d2: StateDict = {}

        # Multi-timescale eligibility (optional, enabled via config)
        if config.use_multiscale_eligibility:
            self._eligibility_d1_fast: StateDict = {}
            self._eligibility_d2_fast: StateDict = {}
            self._eligibility_d1_slow: StateDict = {}
            self._eligibility_d2_slow: StateDict = {}
        else:
            # Single-timescale mode: fast traces are the regular eligibility
            # (no separate fast/slow dicts)
            pass

        # Source-specific eligibility tau configuration (optional overrides)
        # If not set, uses biological defaults in _get_source_eligibility_tau()
        self._source_eligibility_tau: Dict[str, float] = {}

        # =====================================================================
        # ELASTIC TENSOR CAPACITY TRACKING (Phase 1 - Growth Support)
        # =====================================================================
        # Track active vs total capacity for elastic tensor checkpoint format
        # n_neurons_active: Number of neurons currently in use (total projection neurons)
        # n_neurons_capacity: Total allocated memory (includes reserved space)
        self.n_neurons_active = self.d1_size + self.d2_size
        if self.config.growth_enabled:
            # Pre-allocate extra capacity for fast growth
            reserve_multiplier = 1.0 + self.config.reserve_capacity
            self.n_neurons_capacity = int(self.n_neurons_active * reserve_multiplier)
        else:
            # No reserved capacity
            self.n_neurons_capacity = self.n_neurons_active

        # =====================================================================
        # NEUROMORPHIC ID TRACKING (Phase 2 - Neuron-Centric Format)
        # =====================================================================
        # Assign persistent IDs to neurons for ID-based checkpoint format
        # IDs persist across resets and growth events
        self._current_step = 0  # Track timestep for creation metadata
        self.neuron_ids: List[str] = []  # Persistent neuron IDs
        self._initialize_neuron_ids()

        # =====================================================================
        # STATE TRACKER - Temporal State Management
        # =====================================================================
        # Consolidates all temporal state: votes, spikes, trials, actions
        self.state_tracker = StriatumStateTracker(
            n_actions=self.n_actions,
            n_output=self.d1_size + self.d2_size,  # Total MSN neurons (D1 + D2)
            device=self.device,
        )

        # =====================================================================
        # EXPLORATION MANAGER (UCB + Adaptive Exploration)
        # =====================================================================
        # Centralized exploration management with UCB tracking and adaptive
        # tonic dopamine adjustment based on performance.
        exploration_config = ExplorationConfig(
            ucb_exploration=self.config.ucb_exploration,
            ucb_coefficient=self.config.ucb_coefficient,
            adaptive_exploration=self.config.adaptive_exploration,
            performance_window=self.config.performance_window,
            min_tonic_dopamine=self.config.min_tonic_dopamine,
            max_tonic_dopamine=self.config.max_tonic_dopamine,
            tonic_modulates_exploration=self.config.tonic_modulates_exploration,
            tonic_exploration_scale=self.config.tonic_exploration_scale,
        )
        # Create manager context for exploration
        exploration_context = ManagerContext(
            device=self.device,
            n_output=self.n_actions,
            dt_ms=config.dt_ms,
        )
        self.exploration = StriatumExplorationComponent(
            config=exploration_config,
            context=exploration_context,
            initial_tonic_dopamine=self.config.tonic_dopamine,
        )

        # =====================================================================
        # D1/D2 OPPONENT PROCESS (Direct/Indirect Pathways) - ALWAYS ENABLED
        # =====================================================================
        # We have TWO sets of weights and eligibility per action:
        # - D1 (direct pathway): Facilitates action selection (GO signal)
        # - D2 (indirect pathway): Suppresses action selection (NOGO signal)
        #
        # Action selection is based on D1 - D2 balance.
        # Learning rules are OPPOSITE for D1 vs D2:
        # - D1: DA+ → LTP, DA- → LTD (standard)
        # - D2: DA+ → LTD, DA- → LTP (inverted!)
        #
        # This naturally solves credit assignment:
        # - When wrong action is punished (DA-), its D2 pathway strengthens
        # - This builds up NOGO signal, making the action less likely
        # - Meanwhile, correct action's D1 weights strengthen when rewarded

        # =====================================================================
        # D1/D2 PATHWAYS - Separate MSN Populations
        # =====================================================================
        # Create pathway-specific configuration
        # Pathways operate at MSN level (d1_size/d2_size), not action level
        d1_pathway_config = StriatumPathwayConfig(
            n_input=self.input_size,
            n_output=self.d1_size,  # MSN neurons, not actions
            w_min=config.w_min,
            w_max=config.w_max,
            eligibility_tau_ms=self.config.eligibility_tau_ms,
            stdp_lr=self.config.stdp_lr,
            device=str(self.device),
        )
        d2_pathway_config = StriatumPathwayConfig(
            n_input=self.input_size,
            n_output=self.d2_size,  # MSN neurons, not actions
            w_min=config.w_min,
            w_max=config.w_max,
            eligibility_tau_ms=self.config.eligibility_tau_ms,
            stdp_lr=self.config.stdp_lr,
            device=str(self.device),
        )

        # Create D1 and D2 pathways (neurons only, weights stored in parent)
        self.d1_pathway = D1Pathway(d1_pathway_config)
        self.d2_pathway = D2Pathway(d2_pathway_config)

        # =====================================================================
        # FSI (FAST-SPIKING INTERNEURONS) - Parvalbumin+ Interneurons
        # =====================================================================
        # FSI are ~2% of striatal neurons, provide feedforward inhibition
        # Critical for action selection timing (Koós & Tepper 1999)
        # Gap junction networks enable ultra-fast synchronization (<0.1ms)

        # Type annotations for optional FSI components
        self.fsi_neurons: Optional[ConductanceLIF]
        self.gap_junctions_fsi: Optional[GapJunctionCoupling]

        if self.config.fsi_enabled:
            total_msn_neurons = self.d1_size + self.d2_size
            self.fsi_size = int(total_msn_neurons * self.config.fsi_ratio)
            # FSI have fast kinetics (tau_mem ~5ms vs ~20ms for MSNs)
            self.fsi_neurons = create_fast_spiking_neurons(
                n_neurons=self.fsi_size,
                device=self.device,
            )

            # Store gap junction config BEFORE weight initialization
            if self.config.gap_junctions_enabled:
                self._gap_config_fsi = GapJunctionConfig(
                    enabled=True,
                    coupling_strength=self.config.gap_junction_strength,
                    connectivity_threshold=self.config.gap_junction_threshold,
                    max_neighbors=self.config.gap_junction_max_neighbors,
                )
                self.gap_junctions_fsi = None  # Will be initialized after weights
            else:
                self.gap_junctions_fsi = None  # Gap junctions disabled
        else:
            self.fsi_size = 0
            self.fsi_neurons = None
            self.gap_junctions_fsi = None

        # =====================================================================
        # SYNAPTIC WEIGHTS - Multi-Source Per-Pathway Architecture
        # =====================================================================
        # Weights are stored in parent's synaptic_weights dict per source-pathway.
        # Structure: synaptic_weights["{source}_d1"] = [n_d1, n_source_size]
        #            synaptic_weights["{source}_d2"] = [n_d2, n_source_size]
        #
        # Examples:
        #   "cortex:l5_d1": [d1_size, n_cortex_l5]  - D1 MSNs ← Cortex L5
        #   "cortex:l5_d2": [d2_size, n_cortex_l5]  - D2 MSNs ← Cortex L5
        #   "hippocampus_d1": [d1_size, n_hippo]    - D1 MSNs ← Hippocampus
        #   "hippocampus_d2": [d2_size, n_hippo]    - D2 MSNs ← Hippocampus
        #
        # This is biologically accurate:
        # - Each MSN has unique synaptic weights per input source
        # - D1 and D2 MSNs receive same inputs but learn differently
        # - D1: DA+ → LTP (reinforce GO)
        # - D2: DA+ → LTD (suppress NOGO)
        #
        # Weights are initialized dynamically when sources are added via
        # add_input_source_striatum() or during brain construction.
        # No default weights created at __init__.

        # Create manager context for learning
        learning_context = ManagerContext(
            device=self.device,
            n_input=self.input_size,
            n_output=self.n_actions,
            dt_ms=config.dt_ms,
        )

        # Create learning manager (will access weights via parent methods)
        self.learning = StriatumLearningComponent(
            config=self.config,
            context=learning_context,
            d1_pathway=self.d1_pathway,
            d2_pathway=self.d2_pathway,
        )

        # =====================================================================
        # HOMEOSTATIC PLASTICITY STATE
        # =====================================================================
        # Track activity for synaptic scaling (maintains stable firing rates)
        # This is critical for preventing D2 runaway inhibition!
        self._activity_ema = 0.5  # Exponential moving average of activity rate
        self._trial_spike_count = 0.0  # Spikes accumulated this trial
        self._trial_timesteps = 0  # Timesteps in current trial
        self._homeostatic_scaling_applied = False  # Track if scaling happened

        # =====================================================================
        # UNIFIED HOMEOSTASIS (Constraint-Based Stability)
        # =====================================================================
        # Instead of many overlapping correction mechanisms (BCM, synaptic scaling,
        # intrinsic plasticity), we use a single constraint-based approach:
        #
        # Key insight: CONSTRAINTS > CORRECTIONS
        # - Correction: "If D2 gets too small, boost learning" (might not work)
        # - Constraint: "D1 + D2 MUST sum to budget" (mathematically guaranteed)
        #
        # After each reward delivery, we normalize D1/D2 to share a fixed budget.
        # This GUARANTEES:
        # - Neither pathway can completely dominate
        # - If D1 grows, D2 must shrink (and vice versa)
        # - Stable equilibrium is enforced mathematically
        #
        # DYNAMIC BUDGET: Computed from actual initialized weights to adapt to
        # any architecture (population_coding on/off, different n_input, etc.)

        # Type annotation for optional homeostasis component
        self.homeostasis: Optional[StriatumHomeostasisComponent]

        if self.config.homeostasis_enabled:
            # Compute budget from initialized weights (per-action sum of D1+D2)
            # This ensures the budget matches the actual weight scale
            # With multi-source architecture, budget is computed after all sources added
            # Use conservative default for now (will be recomputed after first weights added)
            dynamic_budget = 0.5  # Conservative default, recomputed on first learning update

            homeostasis_config = HomeostasisManagerConfig(
                weight_budget=dynamic_budget,
                normalization_rate=self.config.homeostatic_rate,
                baseline_pressure_enabled=self.config.baseline_pressure_enabled,
                baseline_pressure_rate=self.config.baseline_pressure_rate,
                baseline_target_net=self.config.baseline_target_net,
                w_min=self.config.w_min,
                w_max=self.config.w_max,
                device=str(self.device),
            )
            # Create manager context for homeostasis
            homeostasis_context = ManagerContext(
                device=self.device,
                n_output=self.n_actions,
                dt_ms=config.dt_ms,
                metadata={
                    "neurons_per_action": self.neurons_per_action,
                    "d1_size": self.d1_size,
                    "d2_size": self.d2_size,
                },
            )
            self.homeostasis = StriatumHomeostasisComponent(
                config=homeostasis_config,
                context=homeostasis_context,
            )
        else:
            self.homeostasis = None

        # =====================================================================
        # TD(λ) - MULTI-STEP CREDIT ASSIGNMENT (Phase 1 Enhancement)
        # =====================================================================
        # TD(λ) extends temporal credit assignment from ~1 second to 5-10 seconds
        # by combining eligibility traces with multi-step returns.
        #
        # When enabled, replaces basic eligibility traces with TD(λ) traces that
        # accumulate with factor (γλ) instead of simple exponential decay.

        # Type annotations for optional TD(λ) learners
        self.td_lambda_d1: Optional[TDLambdaLearner]
        self.td_lambda_d2: Optional[TDLambdaLearner]

        if self.config.use_td_lambda:
            td_config = TDLambdaConfig(
                lambda_=self.config.td_lambda,
                gamma=self.config.td_gamma,
                accumulating=self.config.td_lambda_accumulating,
                device=self.config.device,
            )

            # Create TD(λ) learner for D1 and D2 pathways
            # Each pathway has its own separate TD(λ) learner
            # D1 learner: sized for d1_size neurons
            # D2 learner: sized for d2_size neurons
            self.td_lambda_d1 = TDLambdaLearner(
                n_actions=self.d1_size,  # D1 pathway neurons only
                n_input=self.input_size,
                config=td_config,
            )
            self.td_lambda_d2 = TDLambdaLearner(
                n_actions=self.d2_size,  # D2 pathway neurons only
                n_input=self.input_size,
                config=td_config,
            )
        else:
            self.td_lambda_d1 = None
            self.td_lambda_d2 = None

        # =====================================================================
        # SHORT-TERM PLASTICITY (STP)
        # =====================================================================
        # Initialize STP modules for cortical and thalamic input pathways.
        # Biology: Different striatal inputs have distinct STP dynamics:
        # - Cortex→MSNs: DEPRESSING (U=0.4) - prevents saturation, novelty detection
        # - Thalamus→MSNs: WEAK FACILITATION (U=0.25) - phasic amplification
        #
        # =====================================================================
        # SHORT-TERM PLASTICITY (Per-Source)
        # =====================================================================
        # Multi-source architecture: Each source-pathway has its own STP module.
        # Different sources have different dynamics:
        # - Cortical inputs: DEPRESSING (U=0.4) - context filtering
        # - Thalamic inputs: FACILITATING (U=0.25) - phasic amplification
        # - Hippocampal inputs: DEPRESSING (U=0.35) - episodic filtering
        #
        # STP modules are created dynamically via add_input_source_striatum()
        # when sources are connected.
        if self.config.stp_enabled:
            self.stp_modules: Dict[str, ShortTermPlasticity] = {}
            # Note: STP modules will be added per-source in add_input_source_striatum()
            # Each source-pathway (e.g., "cortex:l5_d1", "hippocampus_d2") gets its own STP
        else:
            self.stp_modules = {}

        # =====================================================================
        # GOAL-CONDITIONED VALUES (Phase 1 Week 2-3 Enhancement)
        # =====================================================================
        # Enable PFC goal context to modulate striatal action values via gating.
        # Biology: PFC working memory → Striatum modulation (Miller & Cohen 2001)
        # Learning: Three-factor rule extended with goal context:
        #   Δw = eligibility × dopamine × goal_context

        # Type annotations for optional PFC modulation
        self.pfc_modulation_d1: Optional[nn.Parameter]
        self.pfc_modulation_d2: Optional[nn.Parameter]

        if self.config.use_goal_conditioning:
            # Initialize PFC → D1 modulation weights [d1_size, pfc_size]
            self.pfc_modulation_d1 = nn.Parameter(
                WeightInitializer.sparse_random(
                    n_output=self.d1_size,  # D1 neurons only
                    n_input=self.config.pfc_size,
                    sparsity=0.3,
                    device=torch.device(self.config.device),
                ),
                requires_grad=False,
            )
            # Initialize PFC → D2 modulation weights [d2_size, pfc_size]
            self.pfc_modulation_d2 = nn.Parameter(
                WeightInitializer.sparse_random(
                    n_output=self.d2_size,  # D2 neurons only
                    n_input=self.config.pfc_size,
                    sparsity=0.3,
                    device=torch.device(self.config.device),
                ),
                requires_grad=False,
            )
        else:
            self.pfc_modulation_d1 = None
            self.pfc_modulation_d2 = None

        # =====================================================================
        # REWARD PREDICTION ERROR (RPE) - Value Estimates
        # =====================================================================
        # Track expected value per action for computing prediction error.
        # DA = actual_reward - expected_reward
        # This prevents runaway winners (high expectation → small surprise)

        # Type annotation for optional value estimates
        self.value_estimates: Optional[torch.Tensor]

        if self.config.rpe_enabled:
            self.value_estimates = torch.full(
                (self.n_actions,), self.config.rpe_initial_value, device=self.device
            )
        else:
            self.value_estimates = None

        # Initialize tracking variables (also reset in reset_state)
        self._last_rpe = 0.0
        self._last_expected = 0.0
        self._last_d1_spikes = None
        self._last_d2_spikes = None
        self._last_pfc_goal_context = None  # For goal-conditioned learning

        # Initialize recent_spikes tensor for trial activity tracking
        total_msn_neurons = self.d1_size + self.d2_size
        self.recent_spikes = torch.zeros(total_msn_neurons, device=self.device)

        # Initialize rpe_trace if RPE is enabled
        self.rpe_trace = (
            torch.zeros(self.n_actions, device=self.device) if self.config.rpe_enabled else None
        )

        # =====================================================================
        # STATE OBJECT - Required for NeuromodulatorMixin
        # =====================================================================
        # Initialize state object with neuromodulator fields
        self.state = StriatumState(  # type: ignore[assignment]
            dopamine=self.config.tonic_dopamine,
            acetylcholine=ACH_BASELINE,
            norepinephrine=NE_BASELINE,
        )

        # =====================================================================
        # CHECKPOINT MANAGER
        # =====================================================================
        # Handles state serialization/deserialization
        self.checkpoint_manager = StriatumCheckpointManager(self)

        # =====================================================================
        # FORWARD PASS COORDINATOR
        # =====================================================================
        # Handles D1/D2 pathway coordination and modulation during forward pass
        # Note: STP is now applied per-source in forward() loop (Phase 5)
        self.forward_coordinator = ForwardPassCoordinator(
            config=self.config,
            d1_pathway=self.d1_pathway,
            d2_pathway=self.d2_pathway,
            d1_neurons=self.d1_pathway.neurons,
            d2_neurons=self.d2_pathway.neurons,
            homeostasis_manager=self.homeostasis,
            pfc_modulation_d1=self.pfc_modulation_d1,
            pfc_modulation_d2=self.pfc_modulation_d2,
            stp_module=None,  # Deprecated: STP now per-source in forward()
            device=self.device,
            n_actions=self.n_actions,
            d1_size=self.d1_size,
            d2_size=self.d2_size,
            neurons_per_action=self.neurons_per_action,
        )

        # =====================================================================
        # D1/D2 PATHWAY DELAY BUFFERS (Temporal Competition)
        # =====================================================================
        # Implement biologically-accurate transmission delays for opponent pathways:
        # - D1 direct pathway: ~15ms (Striatum → GPi/SNr → Thalamus)
        # - D2 indirect pathway: ~25ms (Striatum → GPe → STN → GPi/SNr → Thalamus)
        # D1 arrives ~10ms before D2, creating temporal competition window.

        # Calculate delay steps from millisecond delays
        self._d1_delay_steps = int(self.config.d1_to_output_delay_ms / self.config.dt_ms)
        self._d2_delay_steps = int(self.config.d2_to_output_delay_ms / self.config.dt_ms)

        # Delay buffers (initialized lazily on first forward pass)
        self._d1_delay_buffer: Optional[torch.Tensor] = None
        self._d2_delay_buffer: Optional[torch.Tensor] = None
        self._d1_delay_ptr: int = 0
        self._d2_delay_ptr: int = 0

        # Ensure all parameters are on correct device
        self.to(device)

    # =========================================================================
    # EXPLORATION STATE PROPERTIES
    # =========================================================================
    # These properties delegate to exploration component for convenience.

    @property
    def _action_counts(self) -> torch.Tensor:
        """UCB action counts (delegates to exploration)."""
        return self.exploration._action_counts

    @property
    def _total_trials(self) -> int:
        """Total trial count (delegates to exploration)."""
        return self.exploration._total_trials

    @property
    def _recent_rewards(self) -> List[float]:
        """Recent reward history (delegates to exploration)."""
        return self.exploration._recent_rewards

    @property
    def _recent_accuracy(self) -> float:
        """Running accuracy estimate (delegates to exploration)."""
        return self.exploration._recent_accuracy

    @property
    def tonic_dopamine(self) -> float:
        """Current tonic dopamine level (delegates to exploration)."""
        return self.exploration.tonic_dopamine

    @tonic_dopamine.setter
    def tonic_dopamine(self, value: float) -> None:
        """Set tonic dopamine level (delegates to exploration)."""
        self.exploration.tonic_dopamine = value

    # =========================================================================
    # PATHWAY DELEGATION PROPERTIES
    # =========================================================================
    # Kept properties for neurons which are accessed frequently

    @property
    def d1_neurons(self) -> ConductanceLIF:
        """D1 neuron population (delegates to d1_pathway)."""
        return self.d1_pathway.neurons  # type: ignore[return-value]

    @property
    def d2_neurons(self) -> ConductanceLIF:
        """D2 neuron population (delegates to d2_pathway)."""
        return self.d2_pathway.neurons  # type: ignore[return-value]

    @property
    def last_action(self) -> Optional[int]:
        """Last selected action (delegates to state_tracker)."""
        action: Optional[int] = self.state_tracker.last_action
        return action

    # =========================================================================
    # VALUE ESTIMATION (for centralized RPE computation in Brain/VTA)
    # =========================================================================

    def get_expected_value(self, action: Optional[int] = None) -> float:
        """Get expected value for an action (used by Brain for RPE computation).

        The striatum maintains value estimates for each action as "Q-values".
        The Brain (acting as VTA) queries these to compute reward prediction error:
            RPE = actual_reward - expected_value

        Args:
            action: Action index to get value for. If None, uses last_action.

        Returns:
            Expected value for the action. Returns 0.0 if:
            - rpe_enabled is False (no value tracking)
            - action is None and no last_action recorded
            - action index is out of range
        """
        if self.value_estimates is None:
            return 0.0

        if action is None:
            action = self.last_action

        if action is None or action < 0 or action >= self.n_actions:
            return 0.0

        return float(self.value_estimates[action].item())

    def update_value_estimate(self, action: int, reward: float) -> None:
        """Update value estimate for an action towards actual reward.

        Called by Brain after computing RPE, to update the Q-value:
            V(a) ← V(a) + α * (reward - V(a))

        Args:
            action: Action index to update
            reward: Actual reward received
        """
        if self.value_estimates is None:
            return

        if action < 0 or action >= self.n_actions:
            return

        cfg = self.config
        self.value_estimates[action] = self.value_estimates[action] + cfg.rpe_learning_rate * (
            reward - self.value_estimates[action]
        )

    def evaluate_state(
        self,
        state: torch.Tensor,
    ) -> float:
        """
        Evaluate state quality using learned action values.

        For Phase 2 model-based planning: predicts how good a simulated state is
        by computing the maximum Q-value (best action value) from that state.

        Uses existing value estimates to evaluate states during mental simulation.
        If goal-conditioning is enabled, modulates values based on PFC component of state.

        Biology: Striatum represents action values (Q-values) learned through
        dopaminergic reinforcement. During planning, these values can evaluate
        simulated future states (Daw et al., 2011).

        Args:
            state: State to evaluate [n_input] (1D, ADR-005)
                   Format: [cortex_l5 | hippocampus | pfc] (concatenated)

        Returns:
            state_value: Maximum action value (best Q-value) from this state

        Note:
            This is a simplified evaluator using cached Q-values. In full
            implementation, would process state through forward() to get
            D1-D2 competition values.
        """
        if self.value_estimates is None:
            # No value estimates available, return neutral value
            return 0.0

        # Get all action values
        action_values = self.value_estimates.clone()

        # If goal conditioning enabled, extract PFC component from state and modulate values
        if (
            self.config.use_goal_conditioning
            and hasattr(self, "pfc_modulation_d1")
            and self.pfc_modulation_d1 is not None
        ):

            # Extract PFC component from concatenated state tensor
            # Format: [cortex_l5 | hippocampus | pfc]
            pfc_size = self.config.pfc_size
            pfc_goal_context = state[-pfc_size:]

            # Shape assertion: PFC goal context must match modulation matrix columns
            expected_pfc_size = self.pfc_modulation_d1.shape[1]
            actual_pfc_size = (
                pfc_goal_context.shape[0]
                if pfc_goal_context.dim() == 1
                else pfc_goal_context.shape[-1]
            )
            assert actual_pfc_size == expected_pfc_size, (
                f"PFC goal context size mismatch: got {actual_pfc_size}, expected {expected_pfc_size}. "
                f"pfc_modulation_d1 shape: {self.pfc_modulation_d1.shape}, pfc_goal_context shape: {pfc_goal_context.shape}. "
                f"Check that config.pfc_size matches actual PFC output size in brain config."
            )

            # Compute goal modulation for D1 (Go pathway)
            goal_mod_d1 = torch.sigmoid(self.pfc_modulation_d1 @ pfc_goal_context)

            # Scale action values by goal relevance
            # Higher goal modulation → boost that action's value
            if self.neurons_per_action > 1:
                # With population coding, modulation applies per action
                for action_idx in range(self.n_actions):
                    start = action_idx * self.neurons_per_action
                    end = start + self.neurons_per_action
                    action_mod = goal_mod_d1[start:end].mean()
                    # Modulate value: centered around 1.0, range [0.5, 1.5]
                    action_values[action_idx] *= 0.5 + action_mod
            else:
                # Direct modulation per action
                action_mod = goal_mod_d1[: self.n_actions]
                action_values *= 0.5 + action_mod

        # Return max value (best action from this state)
        return float(action_values.max().item())

    # =========================================================================
    # SYNAPTIC WEIGHT INITIALIZATION (Phase 2 - Option B)
    # =========================================================================

    def add_input_source_striatum(
        self,
        source_name: str,
        n_input: int,
        sparsity: float = 0.0,
        weight_scale: float = 1.0,
        learning_rule: Optional[str] = "three_factor",
    ) -> None:
        """Add input source with automatic D1/D2 pathway weight creation.

        This is the primary method for connecting input sources to the striatum.
        It creates BOTH D1 and D2 pathway weights for the given source.

        Args:
            source_name: Source name (e.g., "cortex:l5", "hippocampus", "thalamus")
            n_input: Input size from this source
            sparsity: Connection sparsity (0.0 = fully connected)
            weight_scale: Initial weight scale multiplier
            learning_rule: Learning rule name (default: "three_factor")

        Example:
            >>> striatum.add_input_source_striatum("cortex:l5", n_input=256)
            # Creates:
            #   synaptic_weights["cortex:l5_d1"] = [d1_size, 256]
            #   synaptic_weights["cortex:l5_d2"] = [d2_size, 256]
        """
        # Initialize D1 weights for this source
        # Use sparse random initialization with positive weights for reliable excitation
        # Biology: Most synapses are excitatory (glutamatergic) in corticostriatal pathways
        d1_weights = (
            WeightInitializer.sparse_random(
                n_output=self.d1_size,
                n_input=n_input,
                sparsity=0.8,  # 80% connectivity - biologically realistic
                weight_scale=0.25 * weight_scale,  # Scaled to produce sufficient drive
                device=self.device,
            )
            * self.config.w_max
        )

        # Initialize D2 weights for this source (same initialization)
        d2_weights = (
            WeightInitializer.sparse_random(
                n_output=self.d2_size,
                n_input=n_input,
                sparsity=0.8,  # 80% connectivity - biologically realistic
                weight_scale=0.25 * weight_scale,  # Scaled to produce sufficient drive
                device=self.device,
            )
            * self.config.w_max
        )

        # Register D1 pathway weights
        d1_key = f"{source_name}_d1"
        self.add_input_source(d1_key, n_input, sparsity=0.0, weight_scale=1.0)
        self.synaptic_weights[d1_key].data = d1_weights

        # Register D2 pathway weights
        d2_key = f"{source_name}_d2"
        self.add_input_source(d2_key, n_input, sparsity=0.0, weight_scale=1.0)
        self.synaptic_weights[d2_key].data = d2_weights

        # Link pathways to parent on first source (for checkpoint compatibility)
        # Pathways need _parent_striatum_ref and _weight_source to access weights
        if self.d1_pathway._parent_striatum_ref is None:
            self.d1_pathway._parent_striatum_ref = weakref.ref(self)
            self.d1_pathway._weight_source = d1_key
        if self.d2_pathway._parent_striatum_ref is None:
            self.d2_pathway._parent_striatum_ref = weakref.ref(self)
            self.d2_pathway._weight_source = d2_key

        # Initialize eligibility traces for source-pathway combinations
        if hasattr(self, "learning") and self.learning is not None:
            self.learning.add_source_eligibility_traces(source_name, n_input)

        # =====================================================================
        # CREATE STP MODULES FOR SOURCE-PATHWAY (Phase 5 + Phase 1 Enhancement)
        # =====================================================================
        # Each source-pathway gets its own STP module with source-specific config.
        # Biology: Different input pathways have different short-term dynamics.
        # Phase 1 Enhancement: Heterogeneous STP enables per-synapse parameter variability.
        if self.config.stp_enabled:
            # Determine STP type based on source name
            if "cortex" in source_name or "cortical" in source_name:
                stp_type = "corticostriatal"  # Depressing (U=0.4)
            elif "thalamus" in source_name or "thalamic" in source_name:
                stp_type = "thalamostriatal"  # Facilitating (U=0.25)
            elif "hippocampus" in source_name or "hippoc" in source_name:
                stp_type = "schaffer_collateral"  # Depressing (U=0.46) - hippocampal preset
            else:
                stp_type = "corticostriatal"  # Default to cortical

            # Create STP configs (heterogeneous if enabled)
            if self.config.heterogeneous_stp:
                # Phase 1 Enhancement: Sample per-synapse STP parameters from distributions
                # Biology: 10-fold variability in U within same pathway (Dobrunz & Stevens 1997)
                # D1 pathway: Create list of per-synapse STP configs
                d1_configs = create_heterogeneous_stp_configs(
                    base_preset=stp_type,
                    n_synapses=n_input * self.d1_size,  # Total synapses
                    variability=self.config.stp_variability,
                    seed=self.config.stp_seed,
                    dt=self.config.dt_ms,
                )

                # D2 pathway: Create list of per-synapse STP configs
                d2_configs = create_heterogeneous_stp_configs(
                    base_preset=stp_type,
                    n_synapses=n_input * self.d2_size,  # Total synapses
                    variability=self.config.stp_variability,
                    seed=self.config.stp_seed,
                    dt=self.config.dt_ms,
                )

                # Create STP modules with heterogeneous configs
                d1_stp = ShortTermPlasticity(
                    n_pre=n_input,
                    n_post=self.d1_size,
                    config=d1_configs,  # type: ignore[arg-type]  # List of configs for heterogeneous STP
                    per_synapse=True,
                )

                d2_stp = ShortTermPlasticity(
                    n_pre=n_input,
                    n_post=self.d2_size,
                    config=d2_configs,  # type: ignore[arg-type]  # List of configs for heterogeneous STP
                    per_synapse=True,
                )
            else:
                # Standard uniform STP parameters
                d1_stp = ShortTermPlasticity(
                    n_pre=n_input,
                    n_post=self.d1_size,
                    config=get_stp_config(stp_type, dt=self.config.dt_ms),
                    per_synapse=True,
                )

                d2_stp = ShortTermPlasticity(
                    n_pre=n_input,
                    n_post=self.d2_size,
                    config=get_stp_config(stp_type, dt=self.config.dt_ms),
                    per_synapse=True,
                )

            # Register STP modules
            d1_stp.to(self.device)
            self.stp_modules[d1_key] = d1_stp

            d2_stp.to(self.device)
            self.stp_modules[d2_key] = d2_stp

    def add_fsi_source(
        self,
        source_name: str,
        n_input: int,
        weight_scale: float = 1.0,
    ) -> None:
        """Add FSI input source (no D1/D2 separation for interneurons).

        FSI (fast-spiking interneurons) are parvalbumin+ interneurons that provide
        feedforward inhibition. Unlike MSNs, FSI don't have D1/D2 separation.

        Args:
            source_name: Source name (e.g., "cortex", "thalamus")
            n_input: Input size from this source
            weight_scale: Initial weight scale multiplier
        """
        if self.fsi_size == 0:
            return  # FSI disabled

        # Initialize FSI weights (input → FSI)
        fsi_weights = (
            WeightInitializer.xavier(
                n_output=self.fsi_size,
                n_input=n_input,
                gain=0.3 * weight_scale,  # FSI more excitable than MSNs
                device=self.device,
            )
            * self.config.w_max
        )

        # Register FSI source
        fsi_key = f"fsi_{source_name}"
        self.add_input_source(fsi_key, n_input, sparsity=0.0, weight_scale=1.0)
        self.synaptic_weights[fsi_key].data = fsi_weights

        # Create gap junction coupling (if enabled and this is first FSI source)
        if hasattr(self, "_gap_config_fsi") and self.gap_junctions_fsi is None:
            # Use first FSI source weights for gap junction neighborhood computation
            self.gap_junctions_fsi = GapJunctionCoupling(
                n_neurons=self.fsi_size,
                afferent_weights=fsi_weights,  # Use current source weights
                config=self._gap_config_fsi,
                device=self.device,
            )

    def _link_pathway_weights_to_parent(self) -> None:
        """REMOVED: No longer needed with multi-source architecture.

        D1/D2 pathways don't store weights - weights are in parent's synaptic_weights
        dict with keys like "{source}_d1" and "{source}_d2".
        """
        return  # No-op for backward compatibility

    def _sync_pathway_weights_to_parent(self) -> None:
        """REMOVED: No sync needed with multi-source architecture.

        Weights are always in parent's synaptic_weights dict.
        D1/D2 pathways don't maintain local weight copies.
        """
        return  # No-op for backward compatibility

    def get_d1_weights(self, source: str) -> torch.Tensor:
        """Get D1 MSN weights for a given source.

        Args:
            source: Input source name (without "_d1" suffix)
                   Example: "cortex:l5" returns weights for "cortex:l5_d1"

        Returns:
            D1 weights [n_d1, n_input]

        Raises:
            KeyError: If source not found
        """
        d1_key = f"{source}_d1"
        if d1_key not in self.synaptic_weights:
            raise KeyError(
                f"Source '{source}' not found in D1 pathway. "
                f"Available sources: {self._list_d1_sources()}"
            )
        return self.synaptic_weights[d1_key]  # type: ignore[return-value]

    def get_d2_weights(self, source: str) -> torch.Tensor:
        """Get D2 MSN weights for a given source.

        Args:
            source: Input source name (without "_d2" suffix)
                   Example: "cortex:l5" returns weights for "cortex:l5_d2"

        Returns:
            D2 weights [n_d2, n_input]

        Raises:
            KeyError: If source not found
        """
        d2_key = f"{source}_d2"
        if d2_key not in self.synaptic_weights:
            raise KeyError(
                f"Source '{source}' not found in D2 pathway. "
                f"Available sources: {self._list_d2_sources()}"
            )
        return self.synaptic_weights[d2_key]  # type: ignore[return-value]

    def set_d1_weights(self, weights: torch.Tensor, source: str) -> None:
        """Update D1 MSN weights for a given source.

        Args:
            weights: New D1 weights [n_d1, n_input]
            source: Input source name (without "_d1" suffix)

        Raises:
            KeyError: If source not found
        """
        d1_key = f"{source}_d1"
        if d1_key not in self.synaptic_weights:
            raise KeyError(f"Source '{source}' not found in D1 pathway")
        self.synaptic_weights[d1_key].data = weights

    def set_d2_weights(self, weights: torch.Tensor, source: str) -> None:
        """Update D2 MSN weights for a given source.

        Args:
            weights: New D2 weights [n_d2, n_input]
            source: Input source name (without "_d2" suffix)

        Raises:
            KeyError: If source not found
        """
        d2_key = f"{source}_d2"
        if d2_key not in self.synaptic_weights:
            raise KeyError(f"Source '{source}' not found in D2 pathway")
        self.synaptic_weights[d2_key].data = weights

    def _list_d1_sources(self) -> List[str]:
        """List all sources connected to D1 pathway."""
        return [key[:-3] for key in self.synaptic_weights.keys() if key.endswith("_d1")]

    def _list_d2_sources(self) -> List[str]:
        """List all sources connected to D2 pathway."""
        return [key[:-3] for key in self.synaptic_weights.keys() if key.endswith("_d2")]

    # =========================================================================
    # NEUROMORPHIC ID MANAGEMENT (Phase 2)
    # =========================================================================

    def _initialize_neuron_ids(self) -> None:
        """Initialize neuron IDs for initial neurons.

        Creates unique IDs for all neurons at initialization (step 0).
        IDs follow format: "striatum_{d1|d2}_neuron_{index}_step{step}"
        """
        self.neuron_ids = []

        # Use actual D1/D2 sizes (already computed in __init__)
        n_d1 = self.d1_size
        n_d2 = self.d2_size

        # D1 neuron IDs
        for i in range(n_d1):
            neuron_id = f"striatum_d1_neuron_{i}_step{self._current_step}"
            self.neuron_ids.append(neuron_id)

        # D2 neuron IDs
        for i in range(n_d2):
            neuron_id = f"striatum_d2_neuron_{i}_step{self._current_step}"
            self.neuron_ids.append(neuron_id)

    def _generate_new_neuron_ids(self, n_new: int, pathway_type: str = "d1") -> List[str]:
        """Generate unique IDs for new neurons.

        Args:
            n_new: Number of new neurons to create
            pathway_type: "d1" or "d2" for pathway identification

        Returns:
            List of new neuron IDs
        """
        new_ids = []
        # Count existing neurons of this type for indexing
        existing_count = sum(1 for id in self.neuron_ids if f"_{pathway_type}_" in id)

        for i in range(n_new):
            neuron_id = (
                f"striatum_{pathway_type}_neuron_{existing_count + i}_step{self._current_step}"
            )
            new_ids.append(neuron_id)

        return new_ids

    # region Growth and Neurogenesis

    def grow_output(
        self,
        n_new: int,
        initialization: str = "xavier",
        sparsity: float = 0.1,
    ) -> None:
        """Grow output dimension by adding new action neurons to striatum.

        For Striatum with population coding, adds complete action populations.
        If n_new=1, adds neurons_per_action neurons total.

        Expands:
        - D1 and D2 weight matrices [n_output, n_input] → [n_output + n_new, n_input]
        - Eligibility traces (D1 and D2)
        - Spike traces (input and output for both pathways)
        - Neuron populations (D1-MSNs and D2-MSNs)
        - Action-related tracking (value estimates, Q-values, etc.)

        Args:
            n_new: Number of new actions to add (each action = neurons_per_action neurons)
            initialization: Weight init strategy ('xavier', 'sparse_random', 'uniform')
            sparsity: Connection sparsity for new neurons (0.0 = no connections, 1.0 = fully connected)

        Example:
            >>> striatum = Striatum(StriatumConfig(n_output=2, neurons_per_action=10))
            >>> # Currently: 2 actions × 10 neurons = 20 total neurons
            >>> striatum.grow_output(n_new=1)  # Add 1 action
            >>> # Now: 3 actions × 10 neurons = 30 total neurons
        """
        # Calculate actual number of neurons to add (population coding)
        n_new_neurons = n_new * self.neurons_per_action

        # OLD: old_n_output = self.config.n_output (ambiguous - could be actions or neurons)
        # NEW: Use explicit pathway sizes to get current neuron count
        old_n_neurons = self.d1_size + self.d2_size
        new_n_neurons = old_n_neurons + n_new_neurons

        # =====================================================================
        # 0. ASSIGN NEURON IDS TO NEW NEURONS (Phase 2 - Neuromorphic)
        # =====================================================================
        # Generate IDs for new neurons before creating them
        # Half D1, half D2 to maintain balance
        n_new_d1 = n_new_neurons // 2
        n_new_d2 = n_new_neurons - n_new_d1

        new_d1_ids = self._generate_new_neuron_ids(n_new_d1, pathway_type="d1")
        new_d2_ids = self._generate_new_neuron_ids(n_new_d2, pathway_type="d2")
        self.neuron_ids.extend(new_d1_ids)
        self.neuron_ids.extend(new_d2_ids)

        # =====================================================================
        # 1. EXPAND D1 AND D2 WEIGHT MATRICES using base helper
        # =====================================================================
        # Use base class helper with striatum-specific scale (w_max * 0.2)
        # CRITICAL: Use .data to update Parameter in-place, not direct assignment
        # Expand D1 pathway by D1 neurons only
        self.d1_pathway.weights.data = self._expand_weights(
            current_weights=self.d1_pathway.weights.data,  # type: ignore[arg-type]  # Pass data tensor
            n_new=n_new_d1,  # D1 pathway grows by n_new_d1
            initialization=initialization,
            sparsity=sparsity,
            scale=self.config.w_max * 0.2,
        ).data
        # Expand D2 pathway by D2 neurons only
        self.d2_pathway.weights.data = self._expand_weights(
            current_weights=self.d2_pathway.weights.data,  # type: ignore[arg-type]  # Pass data tensor
            n_new=n_new_d2,  # D2 pathway grows by n_new_d2
            initialization=initialization,
            sparsity=sparsity,
            scale=self.config.w_max * 0.2,
        ).data

        # =====================================================================
        # 2. UPDATE CONFIG (DO THIS BEFORE CREATING NEURONS!)
        # =====================================================================
        # Neurons are created based on pathway sizes, so update them first
        # Update instance variables (sizes no longer in config)
        self.n_actions += n_new
        self.d1_size = self.d1_size + n_new_d1
        self.d2_size = self.d2_size + n_new_d2
        self.n_d1 = self.d1_size  # Keep n_d1 in sync with d1_size
        self.n_d2 = self.d2_size  # Keep n_d2 in sync with d2_size
        self.n_output = self.d1_size + self.d2_size

        # Update forward_coordinator's cached sizes (critical for STP slicing!)
        self.forward_coordinator.d1_size = self.d1_size
        self.forward_coordinator.d2_size = self.d2_size
        self.forward_coordinator.n_actions = self.n_actions

        # Update elastic tensor capacity tracking (Phase 1)
        self.n_neurons_active = new_n_neurons
        # Check if we need to expand capacity
        if self.n_neurons_active > self.n_neurons_capacity:
            # Growth exceeded reserved capacity - reallocate with new headroom
            if self.config.growth_enabled:
                reserve_multiplier = 1.0 + self.config.reserve_capacity
                self.n_neurons_capacity = int(self.n_neurons_active * reserve_multiplier)
            else:
                self.n_neurons_capacity = self.n_neurons_active

        # =====================================================================
        # 3. EXPAND STATE TENSORS using base helper
        # =====================================================================
        # D1 and D2 pathways grow separately, so expand their tensors separately

        # Expand D1 pathway tensors
        state_2d_d1 = {
            "d1_eligibility": self.d1_pathway.eligibility,
        }
        if hasattr(self, "td_lambda_d1") and self.td_lambda_d1 is not None:
            state_2d_d1["td_lambda_d1_traces"] = self.td_lambda_d1.traces.traces

        # Filter out None values before expansion
        state_2d_d1 = {k: v for k, v in state_2d_d1.items() if v is not None}
        expanded_2d_d1 = self._expand_state_tensors(state_2d_d1, n_new_d1)
        self.d1_pathway.eligibility = expanded_2d_d1["d1_eligibility"]
        if hasattr(self, "td_lambda_d1") and self.td_lambda_d1 is not None:
            self.td_lambda_d1.traces.traces = expanded_2d_d1["td_lambda_d1_traces"]
            self.td_lambda_d1.traces.n_output = self.d1_size
            self.td_lambda_d1.n_actions = self.n_actions

        # Expand multi-source D1 eligibility traces
        if hasattr(self, "_eligibility_d1"):
            for source_key in list(self._eligibility_d1.keys()):
                old_elig = self._eligibility_d1[source_key]
                expanded = self._expand_state_tensors({"elig": old_elig}, n_new_d1)
                self._eligibility_d1[source_key] = expanded["elig"]

        # Expand D2 pathway tensors
        state_2d_d2 = {
            "d2_eligibility": self.d2_pathway.eligibility,
        }
        if hasattr(self, "td_lambda_d2") and self.td_lambda_d2 is not None:
            state_2d_d2["td_lambda_d2_traces"] = self.td_lambda_d2.traces.traces

        # Filter out None values before expansion
        state_2d_d2 = {k: v for k, v in state_2d_d2.items() if v is not None}
        expanded_2d_d2 = self._expand_state_tensors(state_2d_d2, n_new_d2)
        self.d2_pathway.eligibility = expanded_2d_d2["d2_eligibility"]
        if hasattr(self, "td_lambda_d2") and self.td_lambda_d2 is not None:
            self.td_lambda_d2.traces.traces = expanded_2d_d2["td_lambda_d2_traces"]
            self.td_lambda_d2.traces.n_output = self.d2_size
            self.td_lambda_d2.n_actions = self.n_actions

        # Expand multi-source D2 eligibility traces
        if hasattr(self, "_eligibility_d2"):
            for source_key in list(self._eligibility_d2.keys()):
                old_elig = self._eligibility_d2[source_key]
                expanded = self._expand_state_tensors({"elig": old_elig}, n_new_d2)
                self._eligibility_d2[source_key] = expanded["elig"]

        # Build state dict for all 1D tensors [n_neurons]
        state_1d = {
            "recent_spikes": self.recent_spikes,
        }

        # Expand all 1D state tensors at once
        expanded_1d = self._expand_state_tensors(state_1d, n_new_neurons)
        self.recent_spikes = expanded_1d["recent_spikes"]

        # =====================================================================
        # 4. EXPAND NEURON POPULATIONS using efficient in-place growth (ConductanceLIF)
        # =====================================================================
        # Expand D1-MSN and D2-MSN neuron populations separately
        if hasattr(self, "d1_pathway") and self.d1_pathway.neurons is not None:
            self.d1_pathway.neurons.grow_neurons(n_new_d1)  # D1 pathway grows by n_new_d1
        if hasattr(self, "d2_pathway") and self.d2_pathway.neurons is not None:
            self.d2_pathway.neurons.grow_neurons(n_new_d2)  # D2 pathway grows by n_new_d2

        # Expand FSI neurons (fast-spiking interneurons) if enabled
        if self.fsi_neurons is not None:
            # FSI size scales with n_output (fsi_ratio * n_output)
            new_fsi_size = int(self.n_actions * self.neurons_per_action * self.config.fsi_ratio)
            n_new_fsi = new_fsi_size - self.fsi_size
            if n_new_fsi > 0:
                self.fsi_neurons.grow_neurons(n_new_fsi)
                self.fsi_size = new_fsi_size

                # Grow FSI weights if they exist
                if "fsi" in self.synaptic_weights:
                    old_fsi_weights = self.synaptic_weights["fsi"]
                    n_input = old_fsi_weights.shape[1]

                    # Initialize new weights for new FSI neurons
                    new_fsi_weights = (
                        WeightInitializer.xavier(
                            n_output=n_new_fsi,
                            n_input=n_input,
                            gain=0.3,
                            device=self.device,
                        )
                        * self.config.w_max
                    )

                    # Concatenate with existing weights
                    self.synaptic_weights["fsi"].data = torch.cat(
                        [old_fsi_weights, new_fsi_weights], dim=0
                    )

        # 4.5. GROW STP MODULES (D1 and D2 separately)
        # =====================================================================
        # STP modules are per-pathway (D1/D2), need to expand separately
        # Each source has "_d1" and "_d2" STP modules that track n_post neurons
        for key in list(self.stp_modules.keys()):
            if "_d1" in key:
                # D1 pathway STP: grow by n_new_d1
                self.stp_modules[key].grow(n_new_d1, target="post")
            elif "_d2" in key:
                # D2 pathway STP: grow by n_new_d2
                self.stp_modules[key].grow(n_new_d2, target="post")
        # =====================================================================
        # Homeostasis tracks per-neuron activity, needs to expand D1 and D2 separately
        if self.homeostasis is not None:
            self.homeostasis.grow(n_new_d1, n_new_d2)

        # =====================================================================
        # 5. UPDATE ACTION-RELATED TRACKING (1D per action, not per neuron)
        # =====================================================================
        # Expand state tracker (vote accumulators, recent_spikes, counts)
        self.state_tracker.grow(n_new, n_new_neurons)

        # Expand exploration manager (handles action_counts and other exploration state)
        self.exploration.grow(self.n_actions)

        # Value estimates for new actions (start at 0)
        if hasattr(self, "value_estimates") and self.value_estimates is not None:
            self.value_estimates = torch.cat(
                [self.value_estimates, torch.zeros(n_new, device=self.device)], dim=0
            )

        # RPE traces for new actions (only if rpe_trace is enabled)
        if self.rpe_trace is not None:
            self.rpe_trace = torch.cat(
                [self.rpe_trace, torch.zeros(n_new, device=self.device)], dim=0
            )

        # =====================================================================
        # 6. EXPAND PFC MODULATION WEIGHTS (using base helper)
        # =====================================================================
        # PFC modulation weights need to expand separately for D1 and D2 pathways
        # pfc_modulation_d1: [d1_size, pfc_size] → [d1_size + n_new_d1, pfc_size]
        # pfc_modulation_d2: [d2_size, pfc_size] → [d2_size + n_new_d2, pfc_size]
        if hasattr(self, "pfc_modulation_d1") and self.pfc_modulation_d1 is not None:
            self.pfc_modulation_d1 = self._expand_weights(
                current_weights=self.pfc_modulation_d1,
                n_new=n_new_d1,  # Expand D1 modulation by D1 neurons only
                initialization="sparse_random",
                sparsity=0.3,
                scale=1.0,  # Default scale for PFC modulation
            )

        if hasattr(self, "pfc_modulation_d2") and self.pfc_modulation_d2 is not None:
            self.pfc_modulation_d2 = self._expand_weights(
                current_weights=self.pfc_modulation_d2,
                n_new=n_new_d2,  # Expand D2 modulation by D2 neurons only
                initialization="sparse_random",
                sparsity=0.3,
                scale=1.0,  # Default scale for PFC modulation
            )

        # =====================================================================
        # 7. VALIDATE GROWTH
        # =====================================================================
        # Validate at neuron level (total d1+d2), not action level
        old_total_neurons = (self.d1_size + self.d2_size) - (n_new_d1 + n_new_d2)
        total_new_neurons = n_new_d1 + n_new_d2
        self._validate_output_growth(old_total_neurons, total_new_neurons, check_neurons=False)

    def grow_actions(
        self,
        n_new: int,
        initialization: str = "xavier",
        sparsity: float = 0.1,
    ) -> None:
        """Grow action space by adding new actions (SEMANTIC API).

        This is the semantic interface for striatum growth. Adds complete action
        populations (n_new × neurons_per_action neurons total).

        Args:
            n_new: Number of new actions to add
            initialization: Weight init strategy
            sparsity: Connection sparsity for new neurons

        Example:
            >>> striatum.grow_actions(n_new=2)  # Add 2 actions
        """
        self.grow_output(n_new, initialization, sparsity)

    def grow_input(
        self,
        n_new: int,
        initialization: str = "xavier",
        sparsity: float = 0.1,
    ) -> None:
        """DEPRECATED: Use grow_source() for multi-source architecture.

        Striatum now uses per-source synaptic weights (e.g., "cortex:l5_d1",
        "hippocampus_d2") instead of a unified input space. To grow a specific
        input source, use grow_source(source_name, new_size) instead.

        Args:
            n_new: Number of input neurons to add (IGNORED)
            initialization: Weight init strategy (IGNORED)
            sparsity: Connection sparsity (IGNORED)

        Raises:
            NotImplementedError: Always, directing users to grow_source()

        Example (NEW API):
            >>> cortex.grow_output(20)
            >>> cortex_to_striatum.grow_source('cortex:l5', new_size=cortex.l5_size)
            >>> striatum.grow_source('cortex:l5', new_size=cortex.l5_size)
        """
        raise NotImplementedError(
            "grow_input() is not supported for multi-source striatum architecture. "
            "Use grow_source(source_name, new_size) to expand specific input sources. "
            "Example: striatum.grow_source('cortex:l5', new_size=300)"
        )

    def grow_source(
        self,
        source_name: str,
        new_size: int,
        initialization: str = "xavier",
        sparsity: float = 0.1,
    ) -> None:
        """Grow input size for a specific source (expands both D1 and D2 weights).

        **Multi-Source Architecture**: Each input source (cortex:l5, hippocampus, etc.)
        has separate weight matrices for D1 and D2 pathways. This method expands
        the input dimension (columns) for a specific source while preserving existing
        weights.

        When upstream regions grow their output, call this method to expand the
        corresponding input weights in the striatum.

        Args:
            source_name: Name of the source to grow (e.g., "cortex:l5", "hippocampus")
            new_size: New total size for this source's input dimension
            initialization: Weight init strategy ('xavier', 'sparse_random', 'uniform')
            sparsity: Connection sparsity for new weights (0.0 = no connections)

        Example:
            >>> # Cortex L5 grows from 200 to 220 neurons
            >>> cortex.grow_output(20)  # L5 now has 220 neurons
            >>> # Update axonal projection
            >>> cortex_to_striatum.grow_source('cortex:l5', new_size=220)
            >>> # Update striatum weights
            >>> striatum.grow_source('cortex:l5', new_size=220)

        Raises:
            KeyError: If source not found in synaptic_weights
        """
        # Grow D1 pathway weights for this source
        d1_key = f"{source_name}_d1"
        if d1_key in self.synaptic_weights:
            old_weights_d1 = self.synaptic_weights[d1_key]
            old_n_input = old_weights_d1.shape[1]
            n_new = new_size - old_n_input

            if n_new > 0:
                # Expand weight matrix (add columns - input dimension)
                # Cannot use _expand_weights (adds rows), must expand columns manually
                device = old_weights_d1.device
                w_scale = self.config.w_max * 0.2

                # Initialize new input weights
                if initialization == "xavier":
                    new_cols = (
                        WeightInitializer.xavier(
                            n_output=self.d1_size,
                            n_input=n_new,
                            gain=0.2,
                            device=device,
                        )
                        * self.config.w_max
                    )
                elif initialization == "sparse_random":
                    new_cols = WeightInitializer.sparse_random(
                        n_output=self.d1_size,
                        n_input=n_new,
                        sparsity=sparsity,
                        scale=w_scale,
                        device=device,
                    )
                else:  # uniform
                    new_cols = WeightInitializer.uniform(
                        n_output=self.d1_size,
                        n_input=n_new,
                        low=0.0,
                        high=w_scale,
                        device=device,
                    )

                # Concatenate columns (dim=1 for input dimension)
                new_weights_d1 = torch.cat([old_weights_d1.data, new_cols], dim=1)
                self.synaptic_weights[d1_key].data = new_weights_d1

                # Update input_sources tracking
                self.input_sources[d1_key] = new_size

                # Expand D1 eligibility trace for this source
                if hasattr(self, "_eligibility_d1") and d1_key in self._eligibility_d1:
                    old_elig_d1 = self._eligibility_d1[d1_key]
                    # Expand columns (input dimension)
                    expanded = torch.cat(
                        [old_elig_d1, torch.zeros(old_elig_d1.shape[0], n_new, device=self.device)],
                        dim=1,
                    )
                    self._eligibility_d1[d1_key] = expanded

        # Grow D2 pathway weights for this source
        d2_key = f"{source_name}_d2"
        if d2_key in self.synaptic_weights:
            old_weights_d2 = self.synaptic_weights[d2_key]
            old_n_input = old_weights_d2.shape[1]
            n_new = new_size - old_n_input

            if n_new > 0:
                # Expand weight matrix (add columns - input dimension)
                device = old_weights_d2.device
                w_scale = self.config.w_max * 0.2

                # Initialize new input weights
                if initialization == "xavier":
                    new_cols = (
                        WeightInitializer.xavier(
                            n_output=self.d2_size,
                            n_input=n_new,
                            gain=0.2,
                            device=device,
                        )
                        * self.config.w_max
                    )
                elif initialization == "sparse_random":
                    new_cols = WeightInitializer.sparse_random(
                        n_output=self.d2_size,
                        n_input=n_new,
                        sparsity=sparsity,
                        scale=w_scale,
                        device=device,
                    )
                else:  # uniform
                    new_cols = WeightInitializer.uniform(
                        n_output=self.d2_size,
                        n_input=n_new,
                        low=0.0,
                        high=w_scale,
                        device=device,
                    )

                # Concatenate columns (dim=1 for input dimension)
                new_weights_d2 = torch.cat([old_weights_d2.data, new_cols], dim=1)
                self.synaptic_weights[d2_key].data = new_weights_d2

                # Update input_sources tracking
                self.input_sources[d2_key] = new_size

                # Expand D2 eligibility trace for this source
                if hasattr(self, "_eligibility_d2") and d2_key in self._eligibility_d2:
                    old_elig_d2 = self._eligibility_d2[d2_key]
                    # Expand columns (input dimension)
                    expanded = torch.cat(
                        [old_elig_d2, torch.zeros(old_elig_d2.shape[0], n_new, device=self.device)],
                        dim=1,
                    )
                    self._eligibility_d2[d2_key] = expanded

        # Grow FSI weights for this source (if present)
        fsi_key = f"fsi_{source_name}"
        if fsi_key in self.synaptic_weights:
            old_weights_fsi = self.synaptic_weights[fsi_key]
            old_n_input = old_weights_fsi.shape[1]
            n_new = new_size - old_n_input

            if n_new > 0:
                # Expand FSI weight matrix (add columns - input dimension)
                device = old_weights_fsi.device
                w_scale = self.config.w_max * 0.3  # FSI slightly stronger

                # Initialize new input weights
                if initialization == "xavier":
                    new_cols = (
                        WeightInitializer.xavier(
                            n_output=self.fsi_size,
                            n_input=n_new,
                            gain=0.2,
                            device=device,
                        )
                        * self.config.w_max
                    )
                elif initialization == "sparse_random":
                    new_cols = WeightInitializer.sparse_random(
                        n_output=self.fsi_size,
                        n_input=n_new,
                        sparsity=sparsity,
                        scale=w_scale,
                        device=device,
                    )
                else:  # uniform
                    new_cols = WeightInitializer.uniform(
                        n_output=self.fsi_size,
                        n_input=n_new,
                        low=0.0,
                        high=w_scale,
                        device=device,
                    )

                # Concatenate columns (dim=1 for input dimension)
                new_weights_fsi = torch.cat([old_weights_fsi.data, new_cols], dim=1)
                self.synaptic_weights[fsi_key].data = new_weights_fsi

        # Update total input_size (sum of all unique sources, accounting for D1/D2 duplication)
        # Each base source (e.g., "default") appears as both "default_d1" and "default_d2"
        # We need to count each base source once
        unique_sources = set()
        for key in self.input_sources.keys():
            if key.endswith("_d1"):
                base = key[:-3]
                unique_sources.add(base)
            elif key.endswith("_d2"):
                base = key[:-3]
                unique_sources.add(base)
            else:
                unique_sources.add(key)

        # Use D1 keys to get sizes (D1 and D2 should have same input size)
        self.input_size = sum(
            self.input_sources.get(f"{src}_d1", self.input_sources.get(src, 0))
            for src in unique_sources
        )

        # Grow TD(λ) traces to match new input_size
        if self.td_lambda_d1 is not None:
            old_n_input = self.td_lambda_d1.traces.traces.shape[1]
            if self.input_size > old_n_input:
                n_new_inputs = self.input_size - old_n_input
                # Expand traces [n_output, n_input]
                new_cols = torch.zeros(self.d1_size, n_new_inputs, device=self.device)
                self.td_lambda_d1.traces.traces = torch.cat(
                    [self.td_lambda_d1.traces.traces, new_cols], dim=1
                )
                self.td_lambda_d1.traces.n_input = self.input_size

        if self.td_lambda_d2 is not None:
            old_n_input = self.td_lambda_d2.traces.traces.shape[1]
            if self.input_size > old_n_input:
                n_new_inputs = self.input_size - old_n_input
                # Expand traces [n_output, n_input]
                new_cols = torch.zeros(self.d2_size, n_new_inputs, device=self.device)
                self.td_lambda_d2.traces.traces = torch.cat(
                    [self.td_lambda_d2.traces.traces, new_cols], dim=1
                )
                self.td_lambda_d2.traces.n_input = self.input_size

        # Grow STP modules for this source (Phase 5)
        if self.config.stp_enabled:
            # Grow D1 STP module
            if d1_key in self.stp_modules:
                # STP modules need to expand n_pre dimension
                # Note: ShortTermPlasticity doesn't have a grow method, so recreate
                old_stp = self.stp_modules[d1_key]
                new_stp_d1 = ShortTermPlasticity(
                    n_pre=new_size,
                    n_post=self.d1_size,
                    config=old_stp.config,
                    per_synapse=True,
                )
                new_stp_d1.to(self.device)
                # Copy existing state (u, x) with padding
                if old_stp.u is not None:
                    old_size = old_stp.u.shape[0]
                    new_stp_d1.u[:old_size] = old_stp.u  # type: ignore[index]
                if old_stp.x is not None:
                    old_size = old_stp.x.shape[0]
                    new_stp_d1.x[:old_size] = old_stp.x  # type: ignore[index]
                self.stp_modules[d1_key] = new_stp_d1

            # Grow D2 STP module
            if d2_key in self.stp_modules:
                old_stp = self.stp_modules[d2_key]
                new_stp_d2 = ShortTermPlasticity(
                    n_pre=new_size,
                    n_post=self.d2_size,
                    config=old_stp.config,
                    per_synapse=True,
                )
                new_stp_d2.to(self.device)
                # Copy existing state (u, x) with padding
                if old_stp.u is not None:
                    old_size = old_stp.u.shape[0]
                    new_stp_d2.u[:old_size] = old_stp.u  # type: ignore[index]
                if old_stp.x is not None:
                    old_size = old_stp.x.shape[0]
                    new_stp_d2.x[:old_size] = old_stp.x  # type: ignore[index]
                self.stp_modules[d2_key] = new_stp_d2

        # Note: No need to update self.input_size since we don't track unified input size
        # in multi-source architecture. Each source has its own size.

    def _initialize_pathway_weights(self) -> torch.Tensor:
        """Initialize weights for D1 or D2 pathway with balanced, principled scaling.

        Uses Xavier initialization to ensure consistent input magnitude regardless of
        input size. Both D1 and D2 start with identical distributions - the
        competition between GO and NOGO pathways emerges purely from learning.

        Principles:
        1. Xavier scaling: Normalizes for different input sizes
        2. Equal D1/D2: No baked-in pathway preference
        3. Minimal variance: Near-symmetric start prevents early bias lock-in (gain=0.2)
        4. Moderate values: Keeps neurons in operational firing regime

        This avoids encoding task-specific knowledge into initialization.
        """
        # Use Xavier initialization with reduced gain for minimal variance
        # Gain of 0.2 provides similar variance to old system (0.01)
        weights = WeightInitializer.xavier(
            n_output=self.n_actions,
            n_input=self.input_size,
            gain=0.2,  # Reduced gain for near-symmetric start
            device=self.device,
        )

        # Scale by w_max and clamp to bounds
        weights = weights * self.config.w_max

        return clamp_weights(weights, self.config.w_min, self.config.w_max, inplace=False)

    def _update_d1_d2_eligibility(
        self,
        inputs: SourceOutputs,  # Changed from Dict[str, torch.Tensor] - multi-source spike inputs
        d1_spikes: torch.Tensor,
        d2_spikes: torch.Tensor,
        chosen_action: int | None = None,
    ) -> None:
        """Update separate eligibility traces for D1 and D2 pathways per source.

        **Multi-Source Architecture**: Each input source (cortex:l5, hippocampus, etc.)
        has separate eligibility traces for D1 and D2 pathways.

        With SEPARATE neuron populations:
        - D1 eligibility is computed from input-D1 spike coincidence PER SOURCE
        - D2 eligibility is computed from input-D2 spike coincidence PER SOURCE

        CRITICAL: When chosen_action is provided, eligibility is ONLY built for
        the neurons corresponding to that action. This is biologically correct:
        only synapses where the post-synaptic neuron fired should become eligible.
        This prevents both actions from building eligibility from the same input.

        Timestep (dt) is obtained from self.config.dt_ms for temporal dynamics.

        Args:
            inputs: Dict mapping source names to spike tensors
                   e.g., {"cortex:l5": [n_l5], "hippocampus": [n_hc]}
            d1_spikes: D1 neuron population spikes [n_d1] (1D)
            d2_spikes: D2 neuron population spikes [n_d2] (1D)
            chosen_action: If provided, only build eligibility for this action's neurons
        """
        # Ensure D1/D2 spikes are 1D
        if d1_spikes.dim() != 1:
            d1_spikes = d1_spikes.squeeze()
        if d2_spikes.dim() != 1:
            d2_spikes = d2_spikes.squeeze()

        # Get float versions for trace updates
        d1_output_1d = d1_spikes.float()
        d2_output_1d = d2_spikes.float()

        # CRITICAL: Mask output spikes to ONLY include the chosen action's neurons
        # This is biologically correct: only synapses where post-synaptic neurons
        # actually fired should become eligible. Without this, both actions build
        # eligibility from the same input, causing learning instability.
        if chosen_action is not None:
            action_mask = torch.zeros_like(d1_output_1d)
            if self.neurons_per_action > 1:
                pop_slice = self._get_action_population_indices(chosen_action)
                action_mask[pop_slice] = 1.0
            else:
                action_mask[chosen_action] = 1.0
            d1_output_1d = d1_output_1d * action_mask
            d2_output_1d = d2_output_1d * action_mask

        # Update eligibility traces PER SOURCE
        # Each source-pathway combination has its own eligibility traces
        self._update_pathway_eligibility(inputs, d1_output_1d, "_d1", "_eligibility_d1")
        self._update_pathway_eligibility(inputs, d2_output_1d, "_d2", "_eligibility_d2")

    def _get_source_eligibility_tau(self, source_name: str) -> float:
        """Get source-specific eligibility trace tau.

        Different input sources have different temporal dynamics:
        - Cortical inputs: Long traces (1000ms) for sustained context
        - Hippocampal inputs: Fast traces (300ms) for episodic snapshots
        - Thalamic inputs: Intermediate (500ms) for phasic signals

        Args:
            source_name: Source identifier (e.g., "cortex:l5", "hippocampus")

        Returns:
            Eligibility tau in milliseconds
        """
        # Check if custom tau is configured
        if hasattr(self, "_source_eligibility_tau") and source_name in self._source_eligibility_tau:
            tau_value = self._source_eligibility_tau[source_name]
            # Ensure it's a float (handle tensor or other numeric types)
            if isinstance(tau_value, torch.Tensor):
                return float(tau_value.item())
            return float(tau_value)

        # Apply biological defaults based on source type
        if "cortex" in source_name:
            return 1000.0  # Standard corticostriatal (long traces)
        elif "hippocampus" in source_name or "hippoc" in source_name:
            return 300.0  # Fast episodic context
        elif "thalamus" in source_name or "thal" in source_name:
            return 500.0  # Intermediate phasic signals
        else:
            # Default to config value
            return self.config.eligibility_tau_ms

    def _update_pathway_eligibility(
        self,
        inputs: SourceOutputs,
        post_spikes: torch.Tensor,
        pathway_suffix: str,
        eligibility_attr: str,
    ) -> None:
        """Update eligibility traces for one pathway (D1 or D2).

        This helper consolidates eligibility update logic that is identical between
        D1 and D2 pathways. Biological separation is maintained:
        - D1 and D2 have separate eligibility trace dictionaries
        - D1 and D2 have different post-synaptic spike patterns
        - Dopamine modulation (applied later) differs between pathways

        Phase 1 Enhancement: Multi-timescale eligibility (optional)
        When use_multiscale_eligibility is enabled:
        - Fast traces (~500ms): Immediate coincidence detection
        - Slow traces (~60s): Consolidated long-term tags
        - Consolidation: slow ← slow*decay + fast*consolidation_rate
        - Combined eligibility = fast + α*slow (used in deliver_reward())

        Args:
            inputs: Dict of source_name -> spike_tensor
            post_spikes: Post-synaptic spikes for this pathway [n_neurons]
            pathway_suffix: "_d1" or "_d2" to select pathway weights
            eligibility_attr: "_eligibility_d1" or "_eligibility_d2"

        Biological note:
            Eligibility traces are the "synaptic tags" that mark recently active
            synapses. When dopamine arrives (seconds later), only tagged synapses
            undergo plasticity. This implements the three-factor learning rule.
        """
        # Multi-timescale mode
        if self.config.use_multiscale_eligibility:
            # Get or initialize eligibility dicts for fast and slow traces
            fast_attr = f"{eligibility_attr}_fast"
            slow_attr = f"{eligibility_attr}_slow"

            if not hasattr(self, fast_attr):
                setattr(self, fast_attr, {})
            if not hasattr(self, slow_attr):
                setattr(self, slow_attr, {})

            fast_dict = getattr(self, fast_attr)
            slow_dict = getattr(self, slow_attr)

            # Decay constants
            fast_decay = torch.exp(
                torch.tensor(-self.config.dt_ms / self.config.fast_eligibility_tau_ms)
            )
            slow_decay = torch.exp(
                torch.tensor(-self.config.dt_ms / self.config.slow_eligibility_tau_ms)
            )
            consolidation_rate = self.config.eligibility_consolidation_rate

            for source_name, source_spikes in inputs.items():
                # Ensure 1D
                if source_spikes.dim() != 1:
                    source_spikes = source_spikes.squeeze()
                source_spikes_float = source_spikes.float()

                # Get pathway-specific key
                key = f"{source_name}{pathway_suffix}"
                if key not in self.synaptic_weights:
                    continue

                # Initialize traces if needed
                weight_shape = self.synaptic_weights[key].shape
                if key not in fast_dict:
                    fast_dict[key] = torch.zeros(weight_shape, device=self.device)
                if key not in slow_dict:
                    slow_dict[key] = torch.zeros(weight_shape, device=self.device)

                # Compute STDP-style eligibility update
                eligibility_update = torch.outer(post_spikes, source_spikes_float)

                # Update fast trace: decay + immediate tagging
                fast_dict[key] = (
                    fast_dict[key] * fast_decay + eligibility_update * self.config.stdp_lr
                )

                # Update slow trace: decay + consolidation from fast trace
                # Biology: Fast tags consolidate into persistent slow tags
                slow_dict[key] = slow_dict[key] * slow_decay + fast_dict[key] * consolidation_rate

            # For backward compatibility: set regular eligibility to fast traces
            # (deliver_reward will use combined eligibility when multi-timescale enabled)
            if not hasattr(self, eligibility_attr):
                setattr(self, eligibility_attr, {})
            eligibility_dict = getattr(self, eligibility_attr)
            for key in fast_dict:
                eligibility_dict[key] = fast_dict[key].clone()

        else:
            # Single-timescale mode (original implementation)
            # Get or initialize eligibility dict for this pathway
            if not hasattr(self, eligibility_attr):
                setattr(self, eligibility_attr, {})
            eligibility_dict = getattr(self, eligibility_attr)

            for source_name, source_spikes in inputs.items():
                # Ensure 1D
                if source_spikes.dim() != 1:
                    source_spikes = source_spikes.squeeze()

                # Get float version
                source_spikes_float = source_spikes.float()

                # Get pathway-specific key (e.g., "cortex:l5_d1", "hippocampus_d2")
                key = f"{source_name}{pathway_suffix}"
                if key not in self.synaptic_weights:
                    continue

                # Initialize eligibility trace if needed
                if key not in eligibility_dict:
                    weight_shape = self.synaptic_weights[key].shape
                    eligibility_dict[key] = torch.zeros(weight_shape, device=self.device)

                # Compute STDP-style eligibility: outer product of post and pre
                # This marks synapses where pre-spike and post-spike co-occurred
                eligibility_update = torch.outer(post_spikes, source_spikes_float)

                # Get source-specific decay tau (cortex=1000ms, hippoc=300ms, etc.)
                tau_ms = self._get_source_eligibility_tau(source_name)
                decay = torch.exp(torch.tensor(-self.config.dt_ms / tau_ms))

                # Decay old eligibility and add new trace
                eligibility_dict[key] = (
                    eligibility_dict[key] * decay + eligibility_update * self.config.stdp_lr
                )

    def set_source_eligibility_tau(self, source_name: str, tau_ms: float) -> None:
        """Configure custom eligibility tau for a specific source.

        Override the biological default with a custom value for specific sources.
        Useful for fine-tuning learning dynamics per input pathway.

        Args:
            source_name: Source identifier (e.g., "cortex:l5", "hippocampus")
            tau_ms: Eligibility trace decay time constant in milliseconds

        Example:
            >>> striatum.set_source_eligibility_tau("cortex:l5", 1200.0)
            >>> striatum.set_source_eligibility_tau("hippocampus", 250.0)
        """
        self._source_eligibility_tau[source_name] = tau_ms

    def _update_d1_d2_eligibility_all(
        self, inputs: SourceOutputs, d1_spikes: torch.Tensor, d2_spikes: torch.Tensor
    ) -> None:
        """Update eligibility traces for ALL active D1/D2 neurons.

        Unlike _update_d1_d2_eligibility which masks to a chosen action,
        this version accumulates eligibility for all neurons that fired.
        The action-specific masking is deferred to deliver_reward(), which
        uses last_action (set by finalize_action) to apply learning only
        to the chosen action's synapses.

        This is the correct approach when action selection happens at trial
        end (via finalize_action) rather than per-timestep.

        Timestep (dt) is obtained from self.config.dt_ms for temporal dynamics.

        Args:
            inputs: Dict mapping source names to spike tensors
            d1_spikes: D1 neuron population spikes
            d2_spikes: D2 neuron population spikes
        """
        # Just call the existing method with chosen_action=None
        # which will update eligibility for all active neurons
        self._update_d1_d2_eligibility(inputs, d1_spikes, d2_spikes, chosen_action=None)

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Optional[Dict[str, float]] = None,
        theta_slot: int = 0,
        coupled_amplitudes: Optional[Dict[str, float]] = None,
    ) -> None:
        """Receive oscillator information from brain broadcast.

        Beta oscillations modulate action selection and maintenance:
        - High beta amplitude: Action persistence (D1 dominant)
          * Increases D1 gain → stronger GO signals
          * Decreases D2 gain → weaker NOGO signals
          * Results in action maintenance (harder to switch)

        - Low beta amplitude: Action flexibility (D2 effective)
          * Decreases D1 gain → weaker GO signals
          * Allows D2 to suppress → easier to switch actions
          * Results in action exploration (easier to change)

        Biology:
        - Motor cortex shows high beta during action maintenance
        - Beta desynchronization (ERD) precedes action changes
        - Beta rebound (ERS) stabilizes new action after selection
        - Striatum receives beta-modulated cortico-striatal input

        Args:
            phases: Oscillator phases in radians {'theta': ..., 'beta': ..., 'gamma': ...}
            signals: Oscillator signal values (sin/cos waveforms)
            theta_slot: Current theta slot [0, n_slots-1] for working memory
            coupled_amplitudes: Coupled amplitudes {'beta_by_theta': ..., etc}

        Note:
            Called automatically by Brain before each forward() call.
            Do not call this manually.
        """
        # Use base mixin implementation to store all oscillator data
        super().set_oscillator_phases(phases, signals, theta_slot, coupled_amplitudes)

        # Update forward coordinator with oscillator state
        self.forward_coordinator.set_oscillator_phases(
            theta_phase=self._theta_phase,
            beta_phase=self._beta_phase,
            beta_amplitude=self._beta_amplitude_effective,
        )

        # Update forward coordinator with neuromodulator state
        self.forward_coordinator.set_neuromodulators(
            dopamine=self.tonic_dopamine,
            norepinephrine=self.forward_coordinator._ne_level,
        )

    def _initialize_weights(self) -> Optional[nn.Parameter]:
        """Striatum uses D1/D2 pathway weights, not base class weights.

        Returns None because striatum manages weights via d1_pathway and d2_pathway.
        Each pathway has its own weight matrix initialized via _initialize_pathway_weights().
        """
        return None

    def _create_neurons(self) -> Optional[ConductanceLIF]:
        """Striatum uses D1/D2 pathway neurons, not base class neurons.

        Returns None because striatum manages neurons via d1_pathway.neurons and d2_pathway.neurons.
        Each pathway has its own neuron population created separately in __init__.
        The actual D1/D2 neurons are created by _create_d1_neurons() and _create_d2_neurons().
        """
        return None

    def _create_d1_neurons(self) -> ConductanceLIF:
        """Create D1-MSN population (direct pathway / GO).

        D1-MSNs express D1 dopamine receptors:
        - DA burst (positive RPE) → LTP → stronger GO signal
        - DA dip (negative RPE) → LTD → weaker GO signal

        These neurons receive excitation from d1_weights and their activity
        promotes action selection (GO signal).

        Uses spike-frequency adaptation to prevent runaway excitation and
        provide more stable, differentiated action selection signals.
        """
        neuron_config = ConductanceLIFConfig(
            v_threshold=V_THRESHOLD_STANDARD,
            v_reset=V_RESET_STANDARD,
            E_L=E_LEAK,
            E_E=E_EXCITATORY,
            E_I=E_INHIBITORY,
            tau_E=5.0,
            tau_I=5.0,
            dt_ms=self.config.dt_ms,
            tau_ref=2.0,
            tau_adapt=100.0,  # Adaptation time constant
            adapt_increment=0.1,  # Enable spike-frequency adaptation
        )
        total_msn_neurons = self.d1_size + self.d2_size
        neurons = ConductanceLIF(n_neurons=total_msn_neurons, config=neuron_config)
        neurons.to(self.device)
        return neurons

    def _create_d2_neurons(self) -> ConductanceLIF:
        """Create D2-MSN population (indirect pathway / NOGO).

        D2-MSNs express D2 dopamine receptors with INVERTED dopamine response:
        - DA burst (positive RPE) → LTD → weaker NOGO signal
        - DA dip (negative RPE) → LTP → stronger NOGO signal

        This inversion is the key biological insight:
        - When wrong action is punished → D2 strengthens → inhibits action next time
        - When correct action is rewarded → D2 weakens → allows action

        These neurons receive excitation from d2_weights and their activity
        opposes action selection (NOGO signal).

        Uses spike-frequency adaptation to prevent runaway excitation and
        provide more stable, differentiated action selection signals.
        """
        neuron_config = ConductanceLIFConfig(
            v_threshold=V_THRESHOLD_STANDARD,
            v_reset=V_RESET_STANDARD,
            E_L=E_LEAK,
            E_E=E_EXCITATORY,
            E_I=E_INHIBITORY,
            tau_E=5.0,
            tau_I=5.0,
            dt_ms=self.config.dt_ms,
            tau_ref=2.0,
            tau_adapt=100.0,  # Adaptation time constant
            adapt_increment=0.1,  # Enable spike-frequency adaptation
        )
        total_msn_neurons = self.d1_size + self.d2_size
        neurons = ConductanceLIF(n_neurons=total_msn_neurons, config=neuron_config)
        neurons.to(self.device)
        return neurons

    # endregion

    # region Forward Pass (D1/D2 Integration and Action Selection)

    def _consolidate_inputs(self, inputs: SourceOutputs) -> torch.Tensor:
        """Convert Dict inputs to concatenated tensor for internal processing.

        **Phase 2 Method**: Applies synaptic weights per-source and concatenates.
        This preserves the internal logic which expects concatenated inputs.

        Args:
            inputs: Dict mapping source names to spike tensors
                   e.g., {"cortex": [n_cortex], "hippocampus": [n_hippo], "pfc": [n_pfc]}

        Returns:
            Concatenated current tensor ready for D1/D2 pathway processing
            Format: [cortex_current | hippocampus_current | pfc_current]
        """
        # Infer device from parameters
        device = (
            next(self.parameters()).device
            if len(list(self.parameters())) > 0
            else torch.device(self.device)
        )

        # Apply synaptic weights and accumulate currents
        # Note: We accumulate rather than concatenate because both D1 and D2
        # receive the same inputs (biologically accurate)
        total_current = torch.zeros(self.n_neurons, device=device)

        for source_name, input_spikes in inputs.items():
            if source_name in self.synaptic_weights:
                # Apply synaptic weights at target dendrites
                current: torch.Tensor = self._apply_synapses(source_name, input_spikes)
                total_current += current

        return total_current

    def _integrate_multi_source_inputs(
        self,
        inputs: SourceOutputs,
        pathway_suffix: str,
        n_neurons: int,
    ) -> torch.Tensor:
        """Integrate synaptic currents from multiple sources for D1 or D2 pathway.

        This helper consolidates the synaptic integration logic that is identical
        between D1 and D2 pathways, while maintaining biological separation:
        - D1 and D2 have separate synaptic weights (independent plasticity)
        - D1 and D2 have separate STP modules (pathway-specific dynamics)
        - Same input creates different currents due to different weights

        Args:
            inputs: Dict mapping source names to spike tensors
            pathway_suffix: "_d1" or "_d2" to select pathway weights
            n_neurons: Number of neurons in this pathway (d1_size or d2_size)

        Returns:
            Total synaptic current for this pathway [n_neurons]

        Biological note:
            This consolidation is for code quality only. Biologically, D1 and D2
            MSNs are distinct neurons with independent synaptic weights, so the
            computation MUST remain separate per pathway.
        """
        current = torch.zeros(n_neurons, device=self.device)

        for source_name, source_spikes in inputs.items():
            # Ensure 1D (ADR-005)
            if source_spikes.dim() != 1:
                source_spikes = source_spikes.squeeze()

            # Convert to float for matrix multiplication
            source_spikes_float = (
                source_spikes.float() if source_spikes.dtype == torch.bool else source_spikes
            )

            # Get pathway-specific weights (e.g., "cortex:l5_d1" or "hippocampus_d2")
            key = f"{source_name}{pathway_suffix}"
            if key not in self.synaptic_weights:
                continue

            weights = self.synaptic_weights[key]

            # Apply source-specific STP if enabled
            if self.config.stp_enabled and key in self.stp_modules:
                # STP returns [n_input, n_neurons] efficacy matrix
                efficacy = self.stp_modules[key](source_spikes_float)
                # Modulate weights: effective_w = w * efficacy.T
                # weights: [n_neurons, n_input], efficacy.T: [n_neurons, n_input]
                effective_weights = weights * efficacy.T
                current += effective_weights @ source_spikes_float
            else:
                # No STP: direct weight multiplication
                current += weights @ source_spikes_float

        return current

    def _apply(self, fn, recurse: bool = True):
        """Override _apply to handle non-parameter tensors in TD-lambda learners.

        This is called by .to(), .cpu(), .cuda(), etc. to transfer all tensors.
        We need to explicitly transfer TD-lambda traces since they're not nn.Parameters.

        Args:
            fn: Function to apply (e.g., lambda t: t.cuda())
            recurse: Whether to apply recursively (inherited from nn.Module)

        Returns:
            Self after applying function
        """
        # Apply to all nn.Module parameters and buffers
        super()._apply(fn, recurse)

        # Transfer TD-lambda learners (not nn.Modules, so need manual transfer)
        if hasattr(self, "td_lambda_d1") and self.td_lambda_d1 is not None:
            # Extract device from function by applying to a dummy tensor
            dummy = torch.zeros(1)
            new_device = fn(dummy).device
            self.td_lambda_d1.to(new_device)

        if hasattr(self, "td_lambda_d2") and self.td_lambda_d2 is not None:
            # Extract device from function by applying to a dummy tensor
            dummy = torch.zeros(1)
            new_device = fn(dummy).device
            self.td_lambda_d2.to(new_device)

        return self

    def forward(
        self,
        inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process input and select action using SEPARATE D1/D2 populations.

        **Phase 2 Changes**: Now accepts Dict[str, Tensor] instead of concatenated tensor.
        Synaptic weights are applied per-source at target dendrites (biologically accurate).

        BIOLOGICAL ARCHITECTURE:
        - D1-MSNs: SEPARATE neuron population, receives synaptic currents
        - D2-MSNs: SEPARATE neuron population, receives same synaptic currents
        - Both populations have unique synaptic weights for same input sources
        - Action selection: argmax(D1_activity - D2_activity) per action

        Args:
            inputs: Dict mapping source names to spike tensors
                   e.g., {"cortex:l5": [n_l5], "hippocampus": [n_hc], "thalamus": [n_thal]}
                   Each source has separate D1 and D2 synaptic weights.

        NOTE: Exploration is handled by finalize_action() at trial end, not per-timestep.
        NOTE: Theta modulation computed internally from self._theta_phase (set by Brain)

        With population coding:
        - Each action has N neurons per pathway (neurons_per_action)
        - D1_votes = sum(D1 spikes for action)
        - D2_votes = sum(D2 spikes for action)
        - NET = D1_votes - D2_votes
        - Selected action = argmax(NET)
        """
        # Validate input type - must be dict with multi-source architecture
        if not isinstance(inputs, dict):
            raise TypeError(
                f"Striatum.forward() requires Dict[str, Tensor] input (multi-source architecture). "
                f"Got {type(inputs).__name__}. "
                f"Example: striatum.forward({{'cortex:l5': spikes, 'hippocampus': spikes}})"
            )

        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION - Per-Source D1/D2 Weights
        # =====================================================================
        # Each source (cortex:l5, hippocampus, thalamus) has separate weights
        # for D1 and D2 pathways. Accumulate synaptic currents separately.
        # Biology: D1 and D2 MSNs are distinct neurons with independent synaptic
        # weights, so integration must remain separate per pathway.
        d1_current = self._integrate_multi_source_inputs(inputs, "_d1", self.d1_size)
        d2_current = self._integrate_multi_source_inputs(inputs, "_d2", self.d2_size)

        # =====================================================================
        # FSI (FAST-SPIKING INTERNEURONS) - Feedforward Inhibition
        # =====================================================================
        # FSI process inputs in parallel with MSNs but with:
        # 1. Gap junction coupling for synchronization (<0.1ms)
        # 2. Feedforward inhibition to MSNs (sharpens action timing)
        # Biology: FSI are parvalbumin+ interneurons (~2% of striatum)
        fsi_inhibition = 0.0  # Scalar value broadcast to all MSNs
        if self.fsi_size > 0:
            # Accumulate FSI currents from all sources
            fsi_current = torch.zeros(self.fsi_size, device=self.device)
            for source_name, source_spikes in inputs.items():
                fsi_key = f"fsi_{source_name}"
                if fsi_key in self.synaptic_weights:
                    source_spikes_float = (
                        source_spikes.float()
                        if source_spikes.dtype == torch.bool
                        else source_spikes
                    )
                    fsi_current += self.synaptic_weights[fsi_key] @ source_spikes_float

            # Apply gap junction coupling (if enabled and state available)
            if self.gap_junctions_fsi is not None and self.state.fsi_membrane is not None:
                gap_current = self.gap_junctions_fsi(self.state.fsi_membrane)  # type: ignore[misc]
                fsi_current = fsi_current + gap_current

            # Update FSI neurons (fast kinetics, tau_mem ~5ms)
            fsi_spikes, fsi_membrane = self.fsi_neurons(
                g_exc_input=fsi_current,
                g_inh_input=torch.zeros_like(fsi_current),  # FSI receive minimal inhibition
            )

            # Store FSI membrane for next timestep gap junctions
            self.state.fsi_membrane = fsi_membrane

            # FSI provide feedforward inhibition to ALL MSNs (broadcast)
            # Each FSI spike contributes 0.5 inhibitory conductance (strong!)
            fsi_inhibition = torch.sum(fsi_spikes) * 0.5

        # =====================================================================
        # D1/D2 NEURON ACTIVATION with Modulation
        # =====================================================================
        # Apply all modulation (theta, dopamine, NE, PFC, homeostasis) to currents
        # before neuron execution

        # Theta modulation (encoding/retrieval phases)
        encoding_mod, retrieval_mod = compute_theta_encoding_retrieval(self._theta_phase)
        d1_current = d1_current * (1.0 + 0.2 * encoding_mod)  # D1 enhanced during encoding
        d2_current = d2_current * (1.0 + 0.2 * retrieval_mod)  # D2 enhanced during retrieval

        # Tonic dopamine and NE gain modulation
        da_gain = 1.0 + 0.3 * (self.state.dopamine - 0.5)  # Centered at 0.5 baseline
        ne_gain = compute_ne_gain(self.state.norepinephrine)
        d1_current = d1_current * da_gain * ne_gain
        d2_current = d2_current * da_gain * ne_gain

        # Apply FSI inhibition (broadcast to all MSNs)
        d1_current = d1_current - fsi_inhibition
        d2_current = d2_current - fsi_inhibition

        # Execute D1 and D2 MSN populations
        d1_spikes, _ = self.d1_pathway.neurons(
            g_exc_input=d1_current.clamp(min=0),
            g_inh_input=torch.zeros_like(d1_current),
        )
        d2_spikes, _ = self.d2_pathway.neurons(
            g_exc_input=d2_current.clamp(min=0),
            g_inh_input=torch.zeros_like(d2_current),
        )

        # =====================================================================
        # ACTION SELECTION: D1 - D2 (GO - NOGO) WITH TEMPORAL COMPETITION
        # =====================================================================
        # For each action, compute NET = D1_activity - D2_activity
        # Select action with highest NET value (or sample from softmax)
        # This is the key biological insight: D1 and D2 populations COMPETE

        # Count votes from current timestep spikes
        d1_votes_current = self._count_population_votes(d1_spikes)
        d2_votes_current = self._count_population_votes(d2_spikes)

        # =====================================================================
        # APPLY D1/D2 PATHWAY DELAYS (Biological Realism)
        # =====================================================================
        # D1 direct pathway: ~15ms (arrives first!)
        # D2 indirect pathway: ~25ms (arrives ~10ms later)
        # This creates temporal competition where D1 "Go" signal arrives before
        # D2 "No-Go" signal, explaining impulsivity and action selection timing.

        # Apply D1 delay (if configured)
        if self._d1_delay_steps > 0:
            # Initialize D1 delay buffer on first use
            if self._d1_delay_buffer is None:
                max_delay_steps = max(1, self._d1_delay_steps * 2 + 1)
                self._d1_delay_buffer = torch.zeros(
                    max_delay_steps,
                    self.n_actions,
                    device=self.device,
                    dtype=d1_votes_current.dtype,
                )
                self._d1_delay_ptr = 0

            # Store current D1 votes in circular buffer
            self._d1_delay_buffer[self._d1_delay_ptr] = d1_votes_current

            # Retrieve delayed D1 votes
            read_idx = (self._d1_delay_ptr - self._d1_delay_steps) % self._d1_delay_buffer.shape[0]
            d1_votes = self._d1_delay_buffer[read_idx]

            # Advance pointer
            self._d1_delay_ptr = (self._d1_delay_ptr + 1) % self._d1_delay_buffer.shape[0]
        else:
            d1_votes = d1_votes_current

        # Apply D2 delay (if configured, typically LONGER than D1)
        if self._d2_delay_steps > 0:
            # Initialize D2 delay buffer on first use
            if self._d2_delay_buffer is None:
                max_delay_steps = max(1, self._d2_delay_steps * 2 + 1)
                self._d2_delay_buffer = torch.zeros(
                    max_delay_steps,
                    self.n_actions,
                    device=self.device,
                    dtype=d2_votes_current.dtype,
                )
                self._d2_delay_ptr = 0

            # Store current D2 votes in circular buffer
            self._d2_delay_buffer[self._d2_delay_ptr] = d2_votes_current

            # Retrieve delayed D2 votes (arrives LATER than D1!)
            read_idx = (self._d2_delay_ptr - self._d2_delay_steps) % self._d2_delay_buffer.shape[0]
            d2_votes = self._d2_delay_buffer[read_idx]

            # Advance pointer
            self._d2_delay_ptr = (self._d2_delay_ptr + 1) % self._d2_delay_buffer.shape[0]
        else:
            d2_votes = d2_votes_current

        # ACCUMULATE delayed D1/D2 votes across timesteps for trial-level decision
        # This integrates sparse spiking evidence over time WITH proper temporal dynamics
        self.state_tracker.accumulate_votes(d1_votes, d2_votes)

        # =====================================================================
        # OUTPUT SPIKES: Return BOTH D1 and D2 (concatenated)
        # =====================================================================
        # Biologically: Both D1-MSNs and D2-MSNs are PROJECTION neurons that send
        # axons out of the striatum to different targets:
        # - D1 neurons → GPi/SNr (direct pathway, facilitates movement)
        # - D2 neurons → GPe (indirect pathway, suppresses movement)
        #
        # Output format: [D1_neuron_0, ..., D1_neuron_N, D2_neuron_0, ..., D2_neuron_M]
        # Total size: d1_size + d2_size (matches config.output_size property)
        #
        # Action selection (which action to take) happens in finalize_action() at
        # trial end, but both pathways' activity is visible to downstream regions.
        output_spikes = torch.cat([d1_spikes, d2_spikes], dim=0)

        # =====================================================================
        # STORE GOAL CONTEXT AND SPIKES FOR LEARNING
        # =====================================================================
        # Extract PFC goal context from inputs (if present)
        # PFC output spikes encode working memory state, which represents current goal/context
        # This enables goal-conditioned learning: same input → different actions based on goal
        pfc_goal_context = None
        if "pfc" in inputs:
            # PFC spikes encode working memory (goal context)
            # Convert to float for use in learning (firing rate representation)
            pfc_spikes = inputs["pfc"]
            if pfc_spikes.dtype == torch.bool:
                pfc_goal_context = pfc_spikes.float()
            else:
                pfc_goal_context = pfc_spikes.clone()

        # Store for goal-conditioned learning during reward delivery
        self.state_tracker.store_spikes_for_learning(d1_spikes, d2_spikes, pfc_goal_context)

        # =====================================================================
        # UPDATE ELIGIBILITY TRACES (for all active neurons)
        # =====================================================================
        # Update D1/D2 STDP-style eligibility (always enabled)
        # Eligibility accumulates for ALL neurons that fire during the trial.
        # When reward arrives, deliver_reward() uses last_action (set by finalize_action)
        # to apply learning only to the chosen action's synapses.
        # Pass inputs dict to eligibility update (multi-source aware)
        self._update_d1_d2_eligibility_all(inputs, d1_spikes, d2_spikes)

        # UPDATE TD(λ) ELIGIBILITY (if enabled)
        # TD(λ) traces accumulate with factor (γλ) instead of simple decay,
        # enabling credit assignment over longer delays (5-10 seconds)
        if self.td_lambda_d1 is not None:
            # Accumulate all source spikes (TD-lambda needs combined input)
            combined_input = torch.zeros(self.input_size, device=self.device)
            offset = 0
            for source_name, source_spikes in inputs.items():
                source_size = source_spikes.shape[0]
                combined_input[offset : offset + source_size] = source_spikes.float()
                offset += source_size

            # Update TD(λ) eligibility for D1 pathway
            # Note: We update for ALL neurons here; masking to chosen action
            # happens in deliver_reward() using last_action
            d1_gradient = torch.outer(d1_spikes.float(), combined_input)
            self.td_lambda_d1.traces.update(d1_gradient)

            # Update TD(λ) eligibility for D2 pathway
            d2_gradient = torch.outer(d2_spikes.float(), combined_input)
            self.td_lambda_d2.traces.update(d2_gradient)

        # Update recent spikes and trial activity via state_tracker
        self.state_tracker.update_recent_spikes(d1_spikes, d2_spikes, decay=0.9)
        self.state_tracker.update_trial_activity(d1_spikes, d2_spikes)

        # Store D1 and D2 spikes for learning manager
        self.learning.store_spikes(d1_spikes, d2_spikes)

        # Store output spikes (NeuralRegion pattern, not self.state.spikes)
        self.output_spikes = output_spikes

        # Increment step counter for neuromorphic checkpoint creation timestamps
        self._current_step += 1

        # Return output spikes (D1/D2 delays already handled by forward_coordinator)
        return output_spikes

    def deliver_reward(self, reward: float) -> Dict[str, Any]:
        """Deliver reward signal and trigger D1/D2 learning.

        **Multi-Source Learning**: Applies dopamine-modulated plasticity to each
        source-pathway combination separately using their respective eligibility traces.

        Args:
            reward: Raw reward signal (for adaptive exploration tracking only)

        Returns:
            Metrics dict with dopamine level and weight changes per source.
        """
        # Use dopamine from forward_coordinator (set via set_neuromodulators)
        da_level = self.forward_coordinator._tonic_dopamine

        # Store for diagnostics
        self._last_rpe = da_level
        self._last_expected = 0.0

        # Adjust exploration based on performance (new component API)
        # reward > 0 counts as "correct" for exploration adjustment
        correct = reward > 0
        self.exploration.update_performance(reward, correct)

        # Apply learning per source-pathway using eligibility traces
        # Three-factor rule: Δw = eligibility × dopamine × learning_rate
        # Phase 1 Enhancement: Combined eligibility = fast + α*slow (if multi-timescale enabled)
        d1_total_ltp = 0.0
        d1_total_ltd = 0.0
        d2_total_ltp = 0.0
        d2_total_ltd = 0.0

        # D1 pathway learning (DA+ → LTP, DA- → LTD)
        if hasattr(self, "_eligibility_d1"):
            for source_key, eligibility in self._eligibility_d1.items():
                if source_key in self.synaptic_weights:
                    # Phase 1: Use combined eligibility if multi-timescale enabled
                    if self.config.use_multiscale_eligibility:
                        # Combined eligibility = fast + α*slow
                        # Biology: Fast traces for immediate learning, slow traces for delayed reward
                        fast_trace = self._eligibility_d1_fast.get(source_key, eligibility)
                        slow_trace = self._eligibility_d1_slow.get(
                            source_key, torch.zeros_like(eligibility)
                        )
                        combined_eligibility = (
                            fast_trace + self.config.slow_trace_weight * slow_trace
                        )
                    else:
                        # Single-timescale mode: use standard eligibility
                        combined_eligibility = eligibility

                    # Compute weight update: Δw = eligibility × dopamine × lr
                    weight_update = combined_eligibility * da_level * self.config.stdp_lr

                    # Apply update with weight bounds
                    new_weights = torch.clamp(
                        self.synaptic_weights[source_key] + weight_update,
                        min=self.config.w_min,
                        max=self.config.w_max,
                    )
                    self.synaptic_weights[source_key].data = new_weights

                    # Track LTP/LTD for diagnostics
                    if da_level > 0:
                        d1_total_ltp += weight_update.sum().item()
                    else:
                        d1_total_ltd += weight_update.sum().item()

        # D2 pathway learning (DA+ → LTD, DA- → LTP - INVERTED!)
        if hasattr(self, "_eligibility_d2"):
            for source_key, eligibility in self._eligibility_d2.items():
                if source_key in self.synaptic_weights:
                    # Phase 1: Use combined eligibility if multi-timescale enabled
                    if self.config.use_multiscale_eligibility:
                        # Combined eligibility = fast + α*slow
                        fast_trace = self._eligibility_d2_fast.get(source_key, eligibility)
                        slow_trace = self._eligibility_d2_slow.get(
                            source_key, torch.zeros_like(eligibility)
                        )
                        combined_eligibility = (
                            fast_trace + self.config.slow_trace_weight * slow_trace
                        )
                    else:
                        # Single-timescale mode: use standard eligibility
                        combined_eligibility = eligibility

                    # Compute weight update with INVERTED dopamine
                    weight_update = combined_eligibility * (-da_level) * self.config.stdp_lr

                    # Apply update with weight bounds
                    new_weights = torch.clamp(
                        self.synaptic_weights[source_key] + weight_update,
                        min=self.config.w_min,
                        max=self.config.w_max,
                    )
                    self.synaptic_weights[source_key].data = new_weights

                    # Track LTP/LTD for diagnostics (note inverted dopamine)
                    if da_level > 0:
                        d2_total_ltd += weight_update.sum().item()
                    else:
                        d2_total_ltp += weight_update.sum().item()

        return {
            "d1_ltp": d1_total_ltp,
            "d1_ltd": d1_total_ltd,
            "d2_ltp": d2_total_ltp,
            "d2_ltd": d2_total_ltd,
            "net_change": d1_total_ltp + d1_total_ltd + d2_total_ltp + d2_total_ltd,
            "dopamine": da_level,
        }

    def deliver_counterfactual_reward(
        self,
        reward: float,
        action: int,
        counterfactual_scale: float = 0.5,
    ) -> Dict[str, Any]:
        """Apply learning for a counterfactual (imagined) action outcome.

        Delegates to LearningManager for model-based "what if" learning.

        Args:
            reward: The counterfactual reward (what WOULD have happened)
            action: The alternate action to evaluate (the one NOT taken)
            counterfactual_scale: How much to scale the learning (default 0.5)

        Returns:
            Metrics dict with weight changes
        """
        # Update value estimate for counterfactual action
        if self.value_estimates is not None and 0 <= action < self.n_actions:
            cf_lr = self.config.rpe_learning_rate * counterfactual_scale
            self.value_estimates[action] = self.value_estimates[action] + cf_lr * (
                reward - self.value_estimates[action]
            )

        # Delegate to learning manager
        # Pass the actually chosen action and the counterfactual action as alternate
        # This allows the learning manager to boost eligibility for the alternate action
        # if it would have led to better outcomes
        chosen = self.last_action if self.last_action is not None else 0
        return self.learning.apply_counterfactual_learning(
            chosen_action=chosen,
            alternate_actions=[action],
            dopamine=reward,
        )

    def reset_eligibility(self, action_only: bool = True) -> None:
        """Reset eligibility traces after learning is complete.

        Delegates to StriatumLearningComponent for eligibility trace management.

        Args:
            action_only: If True, only reset chosen action. If False, reset all.
        """
        self.learning.reset_eligibility(self.last_action, action_only)

    def _reset_subsystems(self, *subsystem_names: str) -> None:
        """Reset multiple subsystems by calling their reset_state() methods.

        Convenience helper to avoid repetitive code in reset_state() implementations.

        Args:
            *subsystem_names: Names of attributes to reset (must have reset_state())
        """
        for name in subsystem_names:
            if hasattr(self, name):
                subsystem = getattr(self, name)
                if subsystem is not None and hasattr(subsystem, "reset_state"):
                    subsystem.reset_state()

    def _reset_scalars(self, **scalar_values: Any) -> None:
        """Reset scalar attributes to specified values.

        Convenience helper for resetting counters, accumulators, etc.

        Args:
            **scalar_values: Attribute names and their reset values
        """
        for name, value in scalar_values.items():
            setattr(self, name, value)

    def reset_state(self) -> None:
        """Reset striatum state for new sequence/episode.

        Resets:
        - State tracker (votes, spikes, trials)
        - D1/D2 pathway eligibility and neurons
        - Multi-source eligibility traces
        - TD(λ) traces (if enabled)
        - Delay buffers (if enabled)
        """
        # Reset state tracker (votes, recent spikes, trial stats, last action)
        self.state_tracker.reset_state()

        # Reset D1/D2 pathways (eligibility and neurons)
        self.d1_pathway.reset_state()
        self.d2_pathway.reset_state()

        # Reset multi-source eligibility traces (Phase 3)
        if hasattr(self, "_eligibility_d1"):
            for key in self._eligibility_d1:
                self._eligibility_d1[key].zero_()
        if hasattr(self, "_eligibility_d2"):
            for key in self._eligibility_d2:
                self._eligibility_d2[key].zero_()

        # Reset TD(λ) traces if enabled
        if self.td_lambda_d1 is not None:
            self.td_lambda_d1.reset_episode()
            self.td_lambda_d2.reset_episode()

        # Reset D1/D2 pathway delay buffers
        # Keep buffers allocated (for efficiency) but reset pointers to beginning
        # Buffers will be refilled with zeros on next forward pass naturally
        self._d1_delay_ptr = 0
        self._d2_delay_ptr = 0
        if self._d1_delay_buffer is not None:
            self._d1_delay_buffer.zero_()
        if self._d2_delay_buffer is not None:
            self._d2_delay_buffer.zero_()

        # Reset FSI neurons and state
        if self.fsi_size > 0:
            # Reset FSI neuron dynamics
            if self.fsi_neurons is not None:
                self.fsi_neurons.reset_state()
            # Initialize FSI membrane state for gap junctions
            self.state.fsi_membrane = torch.zeros(self.fsi_size, device=self.device)

    def set_neuromodulators(
        self,
        dopamine: Optional[float] = None,
        norepinephrine: Optional[float] = None,
        acetylcholine: Optional[float] = None,
    ) -> None:
        """Set neuromodulator levels for striatum.

        Delegates to forward_coordinator which manages neuromodulator state for
        forward pass computations. Also updates self.state for diagnostics consistency.

        Args:
            dopamine: Dopamine level (affects D1/D2 gain, learning)
            norepinephrine: Norepinephrine level (affects arousal/gain)
            acetylcholine: Acetylcholine level (not used in striatum)

        Note:
            ForwardCoordinator maintains its own neuromodulator state (_tonic_dopamine,
            _ne_level) for performance in tight forward loops. self.state is updated
            for diagnostics and consistency with other regions.
        """
        # Update forward_coordinator's private state (used in forward pass)
        if hasattr(self, "forward_coordinator"):
            self.forward_coordinator.set_neuromodulators(
                dopamine=dopamine,
                norepinephrine=norepinephrine,
                acetylcholine=acetylcholine,
            )

        # Also update self.state for diagnostics consistency (inherited from mixin)
        # Use mixin's validation by calling super()
        super().set_neuromodulators(
            dopamine=dopamine,
            norepinephrine=norepinephrine,
            acetylcholine=acetylcholine,
        )

    # endregion

    # region Diagnostics and Health Monitoring

    # =========================================================================
    # DIAGNOSTIC METHODS
    # =========================================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics in standardized DiagnosticsDict format.

        Returns consolidated diagnostic information about:
        - Activity: Spike rates and population activity
        - Plasticity: D1/D2 pathway weights and eligibility traces
        - Health: Pathway balance and weight statistics
        - Neuromodulators: Dopamine levels
        - Region-specific: D1/D2 votes, value estimates, exploration state, etc.

        This is the primary diagnostic interface for the Striatum.
        """
        # D1/D2 per-action means and NET
        d1_per_action: list[float] = []
        d2_per_action: list[float] = []
        net_per_action: list[float] = []

        for action in range(self.n_actions):
            pop_slice = self._get_action_population_indices(action)
            d1_mean = self.d1_pathway.weights[pop_slice, :].mean().item()
            d2_mean = self.d2_pathway.weights[pop_slice, :].mean().item()
            d1_per_action.append(d1_mean)
            d2_per_action.append(d2_mean)
            net_per_action.append(d1_mean - d2_mean)

        # Accumulated votes (current trial) - from state_tracker
        d1_votes, d2_votes = self.state_tracker.get_accumulated_votes()
        net_votes = self.state_tracker.get_net_votes()

        d1_votes_list = d1_votes.tolist()
        d2_votes_list = d2_votes.tolist()
        net_votes_list = net_votes.tolist()

        # Compute activity metrics
        recent_spikes = (
            self.state_tracker.recent_spikes
            if self.state_tracker.recent_spikes is not None
            else torch.zeros(self.n_neurons, device=self.device)
        )
        activity = compute_activity_metrics(
            output_spikes=recent_spikes,
            total_neurons=self.n_neurons,
        )

        # Compute plasticity metrics for D1 pathway (representative)
        plasticity = compute_plasticity_metrics(
            weights=self.d1_pathway.weights,
            learning_rate=self.config.learning_rate,
        )
        # Add D2 and NET statistics
        plasticity["d2_weight_mean"] = float(self.d2_pathway.weights.mean().item())
        plasticity["d2_weight_std"] = float(self.d2_pathway.weights.std().item())
        net_weights = self.d1_pathway.weights - self.d2_pathway.weights
        plasticity["net_weight_mean"] = float(net_weights.mean().item())
        plasticity["net_weight_std"] = float(net_weights.std().item())

        # Compute health metrics
        health = compute_health_metrics(
            state_tensors={
                "d1_weights": self.d1_pathway.weights,
                "d2_weights": self.d2_pathway.weights,
                "d1_eligibility": self.d1_pathway.eligibility,
                "d2_eligibility": self.d2_pathway.eligibility,
            },
            firing_rate=activity.get("firing_rate", 0.0),
        )

        # Neuromodulator metrics
        neuromodulators = {
            "dopamine": self.forward_coordinator._tonic_dopamine,
            "norepinephrine": (
                self.forward_coordinator._ne_level
                if hasattr(self.forward_coordinator, "_ne_level")
                else 0.0
            ),
        }

        # Value estimates (if RPE enabled)
        value_estimates = None
        if self.value_estimates is not None:
            value_estimates = self.value_estimates.tolist()

        # TD(λ) diagnostics (if enabled)
        td_lambda_state = {}
        if self.td_lambda_d1 is not None:
            td_lambda_state = {
                "td_lambda_enabled": True,
                "lambda": self.td_lambda_d1.config.lambda_,
                "gamma": self.td_lambda_d1.config.gamma,
                "d1_td_lambda": self.td_lambda_d1.get_diagnostics(),
                "d2_td_lambda": self.td_lambda_d2.get_diagnostics(),
            }
        else:
            td_lambda_state = {"td_lambda_enabled": False}

        # Exploration state - from state_tracker
        exploration_state = {
            "exploring": self.state_tracker.exploring,
            "last_uncertainty": self.state_tracker._last_uncertainty,
            "last_exploration_prob": self.state_tracker._last_exploration_prob,
            "recent_accuracy": self._recent_accuracy,
        }

        # UCB state
        ucb_state = {
            "enabled": self.config.ucb_exploration,
            "total_trials": self._total_trials,
            "action_counts": self._action_counts.tolist(),
        }

        # Region-specific custom metrics
        region_specific = {
            "n_actions": self.n_actions,
            "neurons_per_action": self.neurons_per_action,
            "last_action": self.state_tracker.last_action,
            # Per-action state
            "d1_weight_means": d1_per_action,
            "d2_weight_means": d2_per_action,
            "net_weight_means": net_per_action,
            # Current trial votes
            "d1_votes": d1_votes_list,
            "d2_votes": d2_votes_list,
            "net_votes": net_votes_list,
            # Exploration
            "exploration": exploration_state,
            "ucb": ucb_state,
            # Value estimates
            "value_estimates": value_estimates,
            # TD(λ) state
            "td_lambda": td_lambda_state,
        }

        # Return as dict (DiagnosticsDict is a TypedDict, not a class)
        return {
            "activity": activity,
            "plasticity": plasticity,
            "health": health,
            "neuromodulators": neuromodulators,
            "region_specific": region_specific,
        }

    # =========================================================================
    # CHECKPOINT STATE MANAGEMENT
    # =========================================================================

    def get_state(self) -> StriatumState:
        """Get current state as StriatumState instance (RegionState protocol).

        Extracts complete striatal state including:
        - D1/D2 pathway states
        - Vote accumulation
        - Action selection state
        - Exploration state
        - Value/RPE tracking
        - Goal modulation weights
        - Delay buffers
        - Homeostatic tracking
        - Neuromodulators

        Returns:
            StriatumState instance with all current state

        Note:
            This is the NEW state management API (Phase 3.2).
            For legacy checkpoint format, use get_full_state().
        """
        # Get D1/D2 pathway states (opaque dicts)
        d1_pathway_state = self.d1_pathway.get_state()
        d2_pathway_state = self.d2_pathway.get_state()

        # Get exploration manager state
        exploration_manager_state = (
            self.exploration.get_state() if hasattr(self.exploration, "get_state") else None
        )

        # Get homeostasis manager state
        homeostasis_manager_state = None
        if self.homeostasis is not None and hasattr(
            self.homeostasis.unified_homeostasis, "get_state"
        ):
            homeostasis_manager_state = self.homeostasis.unified_homeostasis.get_state()

        # Get neuron membrane state (concatenate D1 and D2 membranes for compatibility with base tests)
        membrane = None
        if self.d1_pathway.neurons is not None and hasattr(self.d1_pathway.neurons, "membrane"):
            d1_membrane = self.d1_pathway.neurons.membrane
            d2_membrane = (
                self.d2_pathway.neurons.membrane
                if (
                    self.d2_pathway.neurons is not None
                    and hasattr(self.d2_pathway.neurons, "membrane")
                )
                else None
            )

            if d1_membrane is not None and d2_membrane is not None:
                # Concatenate D1 and D2 membranes to match n_output size
                membrane = torch.cat([d1_membrane, d2_membrane], dim=0).detach().clone()
            elif d1_membrane is not None:
                # Fallback: only D1 (shouldn't happen in normal operation)
                membrane = d1_membrane.detach().clone()

        # Get neuromodulator levels from forward_coordinator
        dopamine = (
            self.forward_coordinator._tonic_dopamine
            if hasattr(self.forward_coordinator, "_tonic_dopamine")
            else 0.0
        )
        norepinephrine = (
            self.forward_coordinator._ne_level
            if hasattr(self.forward_coordinator, "_ne_level")
            else 0.0
        )
        acetylcholine = (
            self.forward_coordinator._ach_level
            if hasattr(self.forward_coordinator, "_ach_level")
            else 0.0
        )

        return StriatumState(
            # Base state
            spikes=(
                self.state_tracker.recent_spikes.detach().clone()
                if self.state_tracker.recent_spikes is not None
                else None
            ),
            membrane=membrane,
            # D1/D2 pathways
            d1_pathway_state=d1_pathway_state,
            d2_pathway_state=d2_pathway_state,
            # Vote accumulation
            d1_votes_accumulated=self.state_tracker._d1_votes_accumulated.detach().clone(),
            d2_votes_accumulated=self.state_tracker._d2_votes_accumulated.detach().clone(),
            # Action selection
            last_action=self.state_tracker.last_action,
            recent_spikes=self.state_tracker.recent_spikes.detach().clone(),
            # Exploration
            exploring=self.state_tracker.exploring,
            last_uncertainty=self.state_tracker._last_uncertainty,
            last_exploration_prob=self.state_tracker._last_exploration_prob,
            exploration_manager_state=exploration_manager_state,
            # Value/RPE (optional)
            value_estimates=(
                self.value_estimates.detach().clone()
                if hasattr(self, "value_estimates") and self.value_estimates is not None
                else None
            ),
            last_rpe=(
                self.state_tracker._last_rpe if hasattr(self.state_tracker, "_last_rpe") else None
            ),
            last_expected=(
                self.state_tracker._last_expected
                if hasattr(self.state_tracker, "_last_expected")
                else None
            ),
            # Goal modulation (optional)
            pfc_modulation_d1=(
                self.pfc_modulation_d1.detach().clone()
                if hasattr(self, "pfc_modulation_d1") and self.pfc_modulation_d1 is not None
                else None
            ),
            pfc_modulation_d2=(
                self.pfc_modulation_d2.detach().clone()
                if hasattr(self, "pfc_modulation_d2") and self.pfc_modulation_d2 is not None
                else None
            ),
            # Delay buffers (optional)
            d1_delay_buffer=(
                self._d1_delay_buffer.detach().clone()
                if hasattr(self, "_d1_delay_buffer") and self._d1_delay_buffer is not None
                else None
            ),
            d2_delay_buffer=(
                self._d2_delay_buffer.detach().clone()
                if hasattr(self, "_d2_delay_buffer") and self._d2_delay_buffer is not None
                else None
            ),
            d1_delay_ptr=self._d1_delay_ptr if hasattr(self, "_d1_delay_ptr") else 0,
            d2_delay_ptr=self._d2_delay_ptr if hasattr(self, "_d2_delay_ptr") else 0,
            # Homeostasis
            activity_ema=self._activity_ema if hasattr(self, "_activity_ema") else 0.0,
            trial_spike_count=self._trial_spike_count if hasattr(self, "_trial_spike_count") else 0,
            trial_timesteps=self._trial_timesteps if hasattr(self, "_trial_timesteps") else 0,
            homeostatic_scaling_applied=(
                self._homeostatic_scaling_applied
                if hasattr(self, "_homeostatic_scaling_applied")
                else False
            ),
            homeostasis_manager_state=homeostasis_manager_state,
            # STP state (per-source, Phase 5)
            stp_modules_state=(
                {
                    key: {
                        "u": stp.u.detach().clone() if stp.u is not None else None,
                        "x": stp.x.detach().clone() if stp.x is not None else None,
                    }
                    for key, stp in self.stp_modules.items()
                }
                if hasattr(self, "stp_modules")
                else {}
            ),
            # Neuromodulators
            dopamine=dopamine,
            acetylcholine=acetylcholine,
            norepinephrine=norepinephrine,
        )

    def load_state(self, state: StriatumState) -> None:
        """Restore state from StriatumState instance (RegionState protocol).

        Restores complete striatal state including:
        - D1/D2 pathway states
        - Vote accumulation
        - Action selection state
        - Exploration state
        - Value/RPE tracking
        - Goal modulation weights
        - Delay buffers
        - Homeostatic tracking
        - Neuromodulators

        Args:
            state: StriatumState instance to restore

        Note:
            This is the NEW state management API (Phase 3.2).
            For legacy checkpoint format, use load_full_state().
        """
        # Restore D1/D2 pathway states
        if state.d1_pathway_state is not None:
            self.d1_pathway.load_state(state.d1_pathway_state)
        if state.d2_pathway_state is not None:
            self.d2_pathway.load_state(state.d2_pathway_state)

        # Restore neuron membrane state
        if state.membrane is not None:
            # Split concatenated membrane into D1 and D2 parts
            if self.d1_pathway.neurons is not None and hasattr(self.d1_pathway.neurons, "membrane"):
                d1_membrane = state.membrane[: self.d1_size].to(self.device)
                self.d1_pathway.neurons.membrane.data = d1_membrane

            if self.d2_pathway.neurons is not None and hasattr(self.d2_pathway.neurons, "membrane"):
                d2_membrane = state.membrane[self.d1_size : self.d1_size + self.d2_size].to(
                    self.device
                )
                self.d2_pathway.neurons.membrane.data = d2_membrane

        # Restore vote accumulation
        if state.d1_votes_accumulated is not None:
            self.state_tracker._d1_votes_accumulated.data = state.d1_votes_accumulated.to(
                self.device
            )
        if state.d2_votes_accumulated is not None:
            self.state_tracker._d2_votes_accumulated.data = state.d2_votes_accumulated.to(
                self.device
            )

        # Restore action selection
        self.state_tracker.last_action = state.last_action
        if state.recent_spikes is not None:
            self.state_tracker.recent_spikes.data = state.recent_spikes.to(self.device)

        # Restore exploration
        self.state_tracker.exploring = state.exploring
        self.state_tracker._last_uncertainty = state.last_uncertainty
        self.state_tracker._last_exploration_prob = state.last_exploration_prob
        if state.exploration_manager_state is not None and hasattr(self.exploration, "load_state"):
            self.exploration.load_state(state.exploration_manager_state)

        # Restore value/RPE (optional)
        if state.value_estimates is not None and hasattr(self, "value_estimates"):
            self.value_estimates.data = state.value_estimates.to(self.device)
        if state.last_rpe is not None:
            self.state_tracker._last_rpe = state.last_rpe
        if state.last_expected is not None:
            self.state_tracker._last_expected = state.last_expected

        # Restore STP state (per-source, Phase 5)
        if (
            hasattr(state, "stp_modules_state")
            and state.stp_modules_state
            and hasattr(self, "stp_modules")
        ):
            for key, stp_state in state.stp_modules_state.items():
                if key in self.stp_modules:
                    if stp_state["u"] is not None and self.stp_modules[key].u is not None:
                        self.stp_modules[key].u.data = stp_state["u"].to(self.device)
                    if stp_state["x"] is not None and self.stp_modules[key].x is not None:
                        self.stp_modules[key].x.data = stp_state["x"].to(self.device)

        # Restore goal modulation (optional)
        if state.pfc_modulation_d1 is not None and hasattr(self, "pfc_modulation_d1"):
            self.pfc_modulation_d1.data = state.pfc_modulation_d1.to(self.device)
        if state.pfc_modulation_d2 is not None and hasattr(self, "pfc_modulation_d2"):
            self.pfc_modulation_d2.data = state.pfc_modulation_d2.to(self.device)

        # Restore delay buffers (optional)
        if state.d1_delay_buffer is not None and hasattr(self, "_d1_delay_buffer"):
            self._d1_delay_buffer = state.d1_delay_buffer.to(self.device)
            self._d1_delay_ptr = state.d1_delay_ptr
        if state.d2_delay_buffer is not None and hasattr(self, "_d2_delay_buffer"):
            self._d2_delay_buffer = state.d2_delay_buffer.to(self.device)
            self._d2_delay_ptr = state.d2_delay_ptr

        # Restore homeostasis
        if hasattr(self, "_activity_ema"):
            self._activity_ema = state.activity_ema
        if hasattr(self, "_trial_spike_count"):
            self._trial_spike_count = state.trial_spike_count
        if hasattr(self, "_trial_timesteps"):
            self._trial_timesteps = state.trial_timesteps
        if hasattr(self, "_homeostatic_scaling_applied"):
            self._homeostatic_scaling_applied = state.homeostatic_scaling_applied
        if state.homeostasis_manager_state is not None and self.homeostasis is not None:
            if hasattr(self.homeostasis.unified_homeostasis, "load_state"):
                self.homeostasis.unified_homeostasis.load_state(state.homeostasis_manager_state)

        # Restore neuromodulators to forward_coordinator
        if hasattr(self.forward_coordinator, "_tonic_dopamine"):
            self.forward_coordinator._tonic_dopamine = state.dopamine
        if hasattr(self.forward_coordinator, "_ne_level"):
            self.forward_coordinator._ne_level = state.norepinephrine
        # acetylcholine not used by striatum

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for checkpointing.

        Returns all state needed to resume training from this exact point,
        including weights, eligibility traces, neuron state, homeostasis, etc.

        Returns:
            Dictionary with complete region state
        """
        # Delegate to checkpoint manager
        return self.checkpoint_manager.get_full_state()

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Restore complete state from checkpoint.

        Args:
            state: Dictionary returned by get_full_state()

        Raises:
            ValueError: If state is incompatible with current configuration
        """
        # Delegate to checkpoint manager
        self.checkpoint_manager.load_full_state(state)

        # Restore tonic dopamine if present in neuromodulator state
        if "neuromodulator_state" in state:
            if "tonic_dopamine" in state["neuromodulator_state"]:
                self.tonic_dopamine = state["neuromodulator_state"]["tonic_dopamine"]

    # endregion
