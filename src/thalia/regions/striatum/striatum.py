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

from dataclasses import replace
from typing import Optional, Dict, Any, List, Union
import weakref

import torch
import torch.nn as nn

from thalia.typing import StriatumDiagnostics
from thalia.core.base.component_config import NeuralComponentConfig
from thalia.core.neural_region import NeuralRegion
from thalia.managers.base_manager import ManagerContext
from thalia.managers.component_registry import register_region
from thalia.components.neurons import (
    ConductanceLIF,
    ConductanceLIFConfig,
    V_THRESHOLD_STANDARD,
    V_RESET_STANDARD,
    E_LEAK,
    E_EXCITATORY,
    E_INHIBITORY,
)
from thalia.components.synapses import WeightInitializer, ShortTermPlasticity, get_stp_config
from thalia.utils.core_utils import clamp_weights
from thalia.utils.input_routing import InputRouter
from thalia.regions.striatum.exploration import ExplorationConfig
from thalia.neuromodulation import ACH_BASELINE, NE_BASELINE

from .config import StriatumConfig, StriatumState
from .action_selection import ActionSelectionMixin
from .pathway_base import StriatumPathwayConfig
from .d1_pathway import D1Pathway
from .d2_pathway import D2Pathway
from .homeostasis_component import StriatumHomeostasisComponent, HomeostasisManagerConfig
from .learning_component import StriatumLearningComponent
from .exploration_component import StriatumExplorationComponent
from .checkpoint_manager import StriatumCheckpointManager
from .state_tracker import StriatumStateTracker
from .forward_coordinator import ForwardPassCoordinator
from .td_lambda import TDLambdaLearner, TDLambdaConfig


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

    def __init__(self, config: NeuralComponentConfig):
        """Initialize Striatum with D1/D2 opponent pathways.

        **Phase 2 Changes**: Now initializes NeuralRegion base with synaptic weights
        stored per-source instead of in D1/D2 pathways.

        Args:
            config: Neural component configuration. Will be converted to
                   StriatumConfig if not already one.

        Initialization Steps:
            1. Convert config to StriatumConfig if needed
            2. Setup population coding (n_actions → n_neurons)
            3. Initialize NeuralRegion with total neurons (D1 + D2)
            4. Create D1/D2 pathways (neurons only, NO weights)
            5. Create state tracker for votes/actions/trials
            6. Setup exploration, learning, homeostasis components
            7. Initialize neuromodulator state (dopamine)

        Population Coding:
            When enabled, each action is represented by multiple neurons:
            - config.n_output = number of actions
            - actual neurons = n_actions × neurons_per_action
            - More robust to noise, biological plausibility
        """
        if not isinstance(config, StriatumConfig):
            config = StriatumConfig(
                n_input=config.n_input,
                n_output=config.n_output,
                neuron_type=config.neuron_type,
                learning_rate=config.learning_rate,
                w_max=config.w_max,
                w_min=config.w_min,
                target_firing_rate_hz=config.target_firing_rate_hz,
                dt_ms=config.dt_ms,
                device=config.device,
            )

        self.striatum_config: StriatumConfig = config  # type: ignore

        # Validate configuration
        if config.n_output <= 0:
            raise ValueError(f"n_output must be positive, got {config.n_output}")

        # =====================================================================
        # POPULATION CODING SETUP
        # =====================================================================
        # If population_coding is enabled, we create N neurons per action
        # n_output in config = number of ACTIONS
        # actual neurons = n_actions * neurons_per_action
        self.n_actions = config.n_output
        if self.striatum_config.population_coding:
            self.neurons_per_action = self.striatum_config.neurons_per_action
            # Override n_output to be the total number of neurons
            actual_n_output = self.n_actions * self.neurons_per_action
            # Create modified config with expanded output
            config = replace(config, n_output=actual_n_output)
            self.striatum_config = config
        else:
            self.neurons_per_action = 1

        # =====================================================================
        # INITIALIZE NEURAL REGION (Phase 2)
        # =====================================================================
        # NeuralRegion handles synaptic weights per-source in synaptic_weights dict
        # D1/D2 pathways will be neuron populations only (no weights)
        NeuralRegion.__init__(
            self,
            n_neurons=config.n_output,
            default_learning_rule="three_factor",  # Dopamine-modulated
            device=config.device,
            dt_ms=config.dt_ms,
        )

        # Store config for backward compatibility with code that expects self.config
        self.config = config

        # =====================================================================
        # ELASTIC TENSOR CAPACITY TRACKING (Phase 1 - Growth Support)
        # =====================================================================
        # Track active vs total capacity for elastic tensor checkpoint format
        # n_neurons_active: Number of neurons currently in use
        # n_neurons_capacity: Total allocated memory (includes reserved space)
        self.n_neurons_active = config.n_output
        if self.striatum_config.growth_enabled:
            # Pre-allocate extra capacity for fast growth
            reserve_multiplier = 1.0 + self.striatum_config.reserve_capacity
            self.n_neurons_capacity = int(config.n_output * reserve_multiplier)
        else:
            # No reserved capacity
            self.n_neurons_capacity = config.n_output

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
            n_output=config.n_output,
            device=self.device,
        )

        # =====================================================================
        # EXPLORATION MANAGER (UCB + Adaptive Exploration)
        # =====================================================================
        # Centralized exploration management with UCB tracking and adaptive
        # tonic dopamine adjustment based on performance.
        exploration_config = ExplorationConfig(
            ucb_exploration=self.striatum_config.ucb_exploration,
            ucb_coefficient=self.striatum_config.ucb_coefficient,
            adaptive_exploration=self.striatum_config.adaptive_exploration,
            performance_window=self.striatum_config.performance_window,
            min_tonic_dopamine=self.striatum_config.min_tonic_dopamine,
            max_tonic_dopamine=self.striatum_config.max_tonic_dopamine,
            tonic_modulates_exploration=self.striatum_config.tonic_modulates_exploration,
            tonic_exploration_scale=self.striatum_config.tonic_exploration_scale,
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
            initial_tonic_dopamine=self.striatum_config.tonic_dopamine,
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
        pathway_config = StriatumPathwayConfig(
            n_input=config.n_input,
            n_output=config.n_output,
            w_min=config.w_min,
            w_max=config.w_max,
            eligibility_tau_ms=self.striatum_config.eligibility_tau_ms,
            stdp_lr=self.striatum_config.stdp_lr,
            device=self.device,
        )

        # Create D1 and D2 pathways (neurons only, weights stored in parent)
        self.d1_pathway = D1Pathway(pathway_config)
        self.d2_pathway = D2Pathway(pathway_config)

        # =====================================================================
        # FSI (FAST-SPIKING INTERNEURONS) - Parvalbumin+ Interneurons
        # =====================================================================
        # FSI are ~2% of striatal neurons, provide feedforward inhibition
        # Critical for action selection timing (Koós & Tepper 1999)
        # Gap junction networks enable ultra-fast synchronization (<0.1ms)
        if self.striatum_config.fsi_enabled:
            self.fsi_size = int(config.n_output * self.striatum_config.fsi_ratio)
            # FSI have fast kinetics (tau_mem ~5ms vs ~20ms for MSNs)
            from thalia.components.neurons import create_fast_spiking_neurons
            self.fsi_neurons = create_fast_spiking_neurons(
                n_neurons=self.fsi_size,
                device=self.device,
            )

            # Store gap junction config BEFORE weight initialization
            if self.striatum_config.gap_junctions_enabled:
                from thalia.components.gap_junctions import GapJunctionConfig
                self._gap_config_fsi = GapJunctionConfig(
                    enabled=True,
                    coupling_strength=self.striatum_config.gap_junction_strength,
                    connectivity_threshold=self.striatum_config.gap_junction_threshold,
                    max_neighbors=self.striatum_config.gap_junction_max_neighbors,
                )
                self.gap_junctions_fsi = None  # Will be initialized after weights
            else:
                self.gap_junctions_fsi = None  # Gap junctions disabled
        else:
            self.fsi_size = 0
            self.fsi_neurons = None
            self.gap_junctions_fsi = None

        # =====================================================================
        # INITIALIZE SYNAPTIC WEIGHTS (Phase 2 - Option B)
        # =====================================================================
        # Weights are stored in parent's synaptic_weights dict, NOT in pathways.
        # Structure: synaptic_weights["source"] = [n_d1 + n_d2, n_source]
        # First n_d1 rows = D1 MSNs, next n_d2 rows = D2 MSNs
        # Each MSN has unique weights (biologically accurate).
        #
        # MIGRATION STRATEGY (Part 2):
        # For backward compatibility during migration, we maintain BOTH:
        # 1. Legacy: D1/D2 pathway weights (self.d1_pathway.weights, self.d2_pathway.weights)
        # 2. New: Combined matrix in synaptic_weights["default"]
        #
        # The combined matrix is the SOURCE OF TRUTH - D1/D2 pathway weights are
        # VIEWS into it. When pathways update their weights, they update the parent's
        # combined matrix. This allows incremental migration of all references.
        #
        # Part 3 will update all code to use accessor methods (get_d1_weights, etc.)
        # Part 4 will remove D1/D2 pathway weights entirely.
        self._initialize_default_synaptic_weights(config.n_input)

        # Link D1/D2 pathway weights to parent's combined matrix (temporary bridge)
        self._link_pathway_weights_to_parent()

        # Create manager context for learning
        learning_context = ManagerContext(
            device=self.device,
            n_input=config.n_input,
            n_output=config.n_output,
            dt_ms=config.dt_ms,
        )

        # Create learning manager (will access weights via parent methods)
        self.learning = StriatumLearningComponent(
            config=self.striatum_config,
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
        if self.striatum_config.homeostasis_enabled:
            # Compute budget from initialized weights (per-action sum of D1+D2)
            # This ensures the budget matches the actual weight scale
            # Access D1 and D2 weights separately (Phase 2)
            if "default_d1" in self.synaptic_weights and "default_d2" in self.synaptic_weights:
                d1_sum = self.synaptic_weights["default_d1"].sum()
                d2_sum = self.synaptic_weights["default_d2"].sum()
                dynamic_budget = ((d1_sum + d2_sum) / self.n_actions).item()
            else:
                # Fallback to default budget if no weights initialized yet
                dynamic_budget = 0.5  # Conservative default

            homeostasis_config = HomeostasisManagerConfig(
                weight_budget=dynamic_budget,
                normalization_rate=self.striatum_config.homeostatic_rate,
                baseline_pressure_enabled=self.striatum_config.baseline_pressure_enabled,
                baseline_pressure_rate=self.striatum_config.baseline_pressure_rate,
                baseline_target_net=self.striatum_config.baseline_target_net,
                w_min=self.config.w_min,
                w_max=self.config.w_max,
                device=self.device,
            )
            # Create manager context for homeostasis
            homeostasis_context = ManagerContext(
                device=self.device,
                n_output=self.n_actions,
                dt_ms=config.dt_ms,
                metadata={"neurons_per_action": self.neurons_per_action},
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
        if self.striatum_config.use_td_lambda:
            td_config = TDLambdaConfig(
                lambda_=self.striatum_config.td_lambda,
                gamma=self.striatum_config.td_gamma,
                accumulating=self.striatum_config.td_lambda_accumulating,
                device=self.config.device,
            )

            # Create TD(λ) learner for D1 and D2 pathways
            # Note: Use config.n_output (actual neuron count) not n_actions
            # because with population coding, n_output = n_actions * neurons_per_action
            self.td_lambda_d1 = TDLambdaLearner(
                n_actions=self.config.n_output,  # Total neurons, not actions
                n_input=self.config.n_input,
                config=td_config,
            )
            self.td_lambda_d2 = TDLambdaLearner(
                n_actions=self.config.n_output,  # Total neurons, not actions
                n_input=self.config.n_input,
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
        # NOTE: Striatum doesn't have explicit "cortex" and "thalamus" source names.
        # Instead, it receives concatenated multi-source input. For STP implementation,
        # we assume:
        # - Primary input source uses corticostriatal STP (depressing)
        # - Could be extended to separate cortical/thalamic sources in future
        #
        # For now, apply corticostriatal STP to ALL inputs (biologically reasonable
        # as cortex provides ~95% of striatal input volume).
        if self.striatum_config.stp_enabled:
            device = torch.device(config.device)

            # Corticostriatal STP: DEPRESSING (U=0.4)
            # Applied to D1 and D2 MSN populations
            # Context-dependent filtering prevents saturation from sustained cortical input
            self.stp_corticostriatal = ShortTermPlasticity(
                n_pre=config.n_input,
                n_post=config.n_output,  # Total MSNs (D1 + D2 populations)
                config=get_stp_config("corticostriatal", dt=config.dt_ms),
                per_synapse=True,
            )
            self.stp_corticostriatal.to(device)

            # Thalamostriatal STP: WEAK FACILITATION (U=0.25)
            # Reserved for future use when thalamic inputs are explicitly separated
            # For now, we initialize but don't use it in forward pass
            # (keeping for future multi-source routing enhancement)
            self.stp_thalamostriatal = ShortTermPlasticity(
                n_pre=config.n_input,
                n_post=config.n_output,
                config=get_stp_config("thalamostriatal", dt=config.dt_ms),
                per_synapse=True,
            )
            self.stp_thalamostriatal.to(device)
        else:
            self.stp_corticostriatal = None
            self.stp_thalamostriatal = None

        # =====================================================================
        # GOAL-CONDITIONED VALUES (Phase 1 Week 2-3 Enhancement)
        # =====================================================================
        # Enable PFC goal context to modulate striatal action values via gating.
        # Biology: PFC working memory → Striatum modulation (Miller & Cohen 2001)
        # Learning: Three-factor rule extended with goal context:
        #   Δw = eligibility × dopamine × goal_context
        if self.striatum_config.use_goal_conditioning:
            # Initialize PFC → D1 modulation weights
            self.pfc_modulation_d1 = nn.Parameter(
                WeightInitializer.sparse_random(
                    n_output=self.config.n_output,  # D1 neurons
                    n_input=self.striatum_config.pfc_size,
                    sparsity=0.3,
                    device=torch.device(self.config.device),
                ),
                requires_grad=False
            )
            # Initialize PFC → D2 modulation weights
            self.pfc_modulation_d2 = nn.Parameter(
                WeightInitializer.sparse_random(
                    n_output=self.config.n_output,  # D2 neurons
                    n_input=self.striatum_config.pfc_size,
                    sparsity=0.3,
                    device=torch.device(self.config.device),
                ),
                requires_grad=False
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
        if self.striatum_config.rpe_enabled:
            self.value_estimates = torch.full(
                (self.n_actions,),
                self.striatum_config.rpe_initial_value,
                device=self.device
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
        self.recent_spikes = torch.zeros(config.n_output, device=self.device)

        # Initialize rpe_trace if RPE is enabled
        self.rpe_trace = (
            torch.zeros(self.n_actions, device=self.device)
            if self.striatum_config.rpe_enabled
            else None
        )

        # =====================================================================
        # STATE OBJECT - Required for NeuromodulatorMixin
        # =====================================================================
        # Initialize state object with neuromodulator fields
        self.state = StriatumState(
            dopamine=self.striatum_config.tonic_dopamine,
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
        self.forward_coordinator = ForwardPassCoordinator(
            config=self.striatum_config,
            d1_pathway=self.d1_pathway,
            d2_pathway=self.d2_pathway,
            d1_neurons=self.d1_pathway.neurons,
            d2_neurons=self.d2_pathway.neurons,
            homeostasis_manager=self.homeostasis,
            pfc_modulation_d1=self.pfc_modulation_d1,
            pfc_modulation_d2=self.pfc_modulation_d2,
            stp_module=self.stp_corticostriatal,  # Apply corticostriatal STP
            device=self.device,
        )

        # =====================================================================
        # D1/D2 PATHWAY DELAY BUFFERS (Temporal Competition)
        # =====================================================================
        # Implement biologically-accurate transmission delays for opponent pathways:
        # - D1 direct pathway: ~15ms (Striatum → GPi/SNr → Thalamus)
        # - D2 indirect pathway: ~25ms (Striatum → GPe → STN → GPi/SNr → Thalamus)
        # D1 arrives ~10ms before D2, creating temporal competition window.

        # Calculate delay steps from millisecond delays
        self._d1_delay_steps = int(self.striatum_config.d1_to_output_delay_ms / self.config.dt_ms)
        self._d2_delay_steps = int(self.striatum_config.d2_to_output_delay_ms / self.config.dt_ms)

        # Delay buffers (initialized lazily on first forward pass)
        self._d1_delay_buffer: Optional[torch.Tensor] = None
        self._d2_delay_buffer: Optional[torch.Tensor] = None
        self._d1_delay_ptr: int = 0
        self._d2_delay_ptr: int = 0

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
        return self.d1_pathway.neurons

    @property
    def d2_neurons(self) -> ConductanceLIF:
        """D2 neuron population (delegates to d2_pathway)."""
        return self.d2_pathway.neurons

    @property
    def last_action(self) -> Optional[int]:
        """Last selected action (delegates to state_tracker)."""
        return self.state_tracker.last_action

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

        cfg = self.striatum_config
        self.value_estimates[action] = (
            self.value_estimates[action]
            + cfg.rpe_learning_rate * (reward - self.value_estimates[action])
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
        if (self.striatum_config.use_goal_conditioning and
            hasattr(self, 'pfc_modulation_d1') and
            self.pfc_modulation_d1 is not None):

            # Extract PFC component from concatenated state tensor
            # Format: [cortex_l5 | hippocampus | pfc]
            pfc_size = self.striatum_config.pfc_size
            pfc_goal_context = state[-pfc_size:]

            # Shape assertion: PFC goal context must match modulation matrix columns
            expected_pfc_size = self.pfc_modulation_d1.shape[1]
            actual_pfc_size = pfc_goal_context.shape[0] if pfc_goal_context.dim() == 1 else pfc_goal_context.shape[-1]
            assert actual_pfc_size == expected_pfc_size, \
                f"PFC goal context size mismatch: got {actual_pfc_size}, expected {expected_pfc_size}. " \
                f"pfc_modulation_d1 shape: {self.pfc_modulation_d1.shape}, pfc_goal_context shape: {pfc_goal_context.shape}. " \
                f"Check that striatum_config.pfc_size matches actual PFC output size in brain config."

            # Compute goal modulation for D1 (Go pathway)
            goal_mod_d1 = torch.sigmoid(
                self.pfc_modulation_d1 @ pfc_goal_context
            )

            # Scale action values by goal relevance
            # Higher goal modulation → boost that action's value
            if self.striatum_config.population_coding:
                # With population coding, modulation applies per action
                for action_idx in range(self.n_actions):
                    start = action_idx * self.neurons_per_action
                    end = start + self.neurons_per_action
                    action_mod = goal_mod_d1[start:end].mean()
                    # Modulate value: centered around 1.0, range [0.5, 1.5]
                    action_values[action_idx] *= (0.5 + action_mod)
            else:
                # Direct modulation per action
                action_mod = goal_mod_d1[:self.n_actions]
                action_values *= (0.5 + action_mod)

        # Return max value (best action from this state)
        return action_values.max().item()

    # =========================================================================
    # SYNAPTIC WEIGHT INITIALIZATION (Phase 2 - Option B)
    # =========================================================================

    def _initialize_default_synaptic_weights(self, n_input: int) -> None:
        """Initialize default synaptic weights for D1 and D2 pathways.

        Option B Architecture: D1 and D2 MSNs have SEPARATE weights because they
        learn differently (opposite dopamine responses). Both receive the same inputs.

        Structure:
        - synaptic_weights["default_d1"] = [n_d1, n_input]  # D1 MSN weights
        - synaptic_weights["default_d2"] = [n_d2, n_input]  # D2 MSN weights

        This is biologically accurate: each MSN has unique synaptic weights,
        and D1/D2 differ in their dopamine receptors (and thus learning rules).

        Args:
            n_input: Input dimension size
        """
        n_total = self.config.n_output  # Neurons per pathway
        # D1 and D2 are SEPARATE full-size populations, not split
        # Both pathways have n_output neurons (opponent/parallel processing)
        n_d1 = n_total
        n_d2 = n_total

        # Initialize D1 weights using Xavier initialization
        d1_weights = WeightInitializer.xavier(
            n_output=n_d1,
            n_input=n_input,
            gain=0.2,  # Conservative initialization
            device=self.device,
        ) * self.config.w_max

        # Initialize D2 weights (separate matrix, same initialization)
        d2_weights = WeightInitializer.xavier(
            n_output=n_d2,
            n_input=n_input,
            gain=0.2,
            device=self.device,
        ) * self.config.w_max

        # Register as separate sources (will initialize with sparse_random, we'll override)
        self.add_input_source("default_d1", n_input, sparsity=0.0, weight_scale=1.0)
        self.add_input_source("default_d2", n_input, sparsity=0.0, weight_scale=1.0)

        # Override with our Xavier-initialized weights
        self.synaptic_weights["default_d1"].data = d1_weights
        self.synaptic_weights["default_d2"].data = d2_weights

        # Store sizes for easy access
        self.n_d1 = n_d1
        self.n_d2 = n_d2

        # =====================================================================
        # FSI WEIGHTS AND GAP JUNCTIONS
        # =====================================================================
        # If FSI enabled, create weights from inputs to FSI
        # FSI use these weights for both: (1) spike generation, (2) gap junction neighborhoods
        if self.fsi_size > 0:
            # Initialize FSI weights (input → FSI)
            fsi_weights = WeightInitializer.xavier(
                n_output=self.fsi_size,
                n_input=n_input,
                gain=0.3,  # Slightly stronger than MSNs (FSI are more excitable)
                device=self.device,
            ) * self.config.w_max

            # Register FSI as input source
            self.add_input_source("fsi", n_input, sparsity=0.0, weight_scale=1.0)
            self.synaptic_weights["fsi"].data = fsi_weights

            # Create gap junction coupling (uses fsi weights for neighborhoods)
            if hasattr(self, '_gap_config_fsi'):
                from thalia.components.gap_junctions import GapJunctionCoupling
                self.gap_junctions_fsi = GapJunctionCoupling(
                    n_neurons=self.fsi_size,
                    afferent_weights=self.synaptic_weights["fsi"],
                    config=self._gap_config_fsi,
                    device=self.device,
                )

    def _link_pathway_weights_to_parent(self) -> None:
        """Pass parent reference to D1/D2 pathways for weight access.

        D1/D2 pathways no longer own weights - they access parent's synaptic_weights
        dict via a reference. This implements Option B (biologically accurate).

        Each pathway gets:
        - parent reference (for weight access)
        - source name ("default_d1" or "default_d2")
        """
        # Pass parent reference to pathways (using weakref to avoid circular module references)
        self.d1_pathway._parent_striatum_ref = weakref.ref(self)
        self.d1_pathway._weight_source = "default_d1"

        self.d2_pathway._parent_striatum_ref = weakref.ref(self)
        self.d2_pathway._weight_source = "default_d2"

    def _sync_pathway_weights_to_parent(self) -> None:
        """No-op: Pathways access parent weights directly, no sync needed."""
        pass

    def get_d1_weights(self, source: str = "default_d1") -> torch.Tensor:
        """Get D1 MSN weights for a given source.

        Args:
            source: Input source name (default: "default_d1")

        Returns:
            D1 weights [n_d1, n_input]
        """
        if source not in self.synaptic_weights:
            raise KeyError(f"Source '{source}' not found in synaptic_weights")
        return self.synaptic_weights[source]

    def get_d2_weights(self, source: str = "default_d2") -> torch.Tensor:
        """Get D2 MSN weights for a given source.

        Args:
            source: Input source name (default: "default_d2")

        Returns:
            D2 weights [n_d2, n_input]
        """
        if source not in self.synaptic_weights:
            raise KeyError(f"Source '{source}' not found in synaptic_weights")
        return self.synaptic_weights[source]

    def set_d1_weights(self, weights: torch.Tensor, source: str = "default_d1") -> None:
        """Update D1 MSN weights for a given source.

        Args:
            weights: New D1 weights [n_d1, n_input]
            source: Input source name (default: "default_d1")
        """
        if source not in self.synaptic_weights:
            raise KeyError(f"Source '{source}' not found in synaptic_weights")
        self.synaptic_weights[source].data = weights

    def set_d2_weights(self, weights: torch.Tensor, source: str = "default_d2") -> None:
        """Update D2 MSN weights for a given source.

        Args:
            weights: New D2 weights [n_d2, n_input]
            source: Input source name (default: "default_d2")
        """
        if source not in self.synaptic_weights:
            raise KeyError(f"Source '{source}' not found in synaptic_weights")
        self.synaptic_weights[source].data = weights

    # =========================================================================
    # NEUROMORPHIC ID MANAGEMENT (Phase 2)
    # =========================================================================

    def _initialize_neuron_ids(self) -> None:
        """Initialize neuron IDs for initial neurons.

        Creates unique IDs for all neurons at initialization (step 0).
        IDs follow format: "striatum_{d1|d2}_neuron_{index}_step{step}"
        """
        self.neuron_ids = []
        n_neurons = self.config.n_output

        # Half D1, half D2 (for ID purposes, even if pathways are separate)
        n_d1 = n_neurons // 2
        n_d2 = n_neurons - n_d1

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
            neuron_id = f"striatum_{pathway_type}_neuron_{existing_count + i}_step{self._current_step}"
            new_ids.append(neuron_id)

        return new_ids

    # region Growth and Neurogenesis

    def grow_output(
        self,
        n_new: int,
        initialization: str = 'xavier',
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
        old_n_output = self.config.n_output
        new_n_output = old_n_output + n_new_neurons

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
        self.d1_pathway.weights.data = self._expand_weights(
            current_weights=self.d1_pathway.weights,
            n_new=n_new_neurons,
            initialization=initialization,
            sparsity=sparsity,
            scale=self.config.w_max * 0.2,
        )
        self.d2_pathway.weights.data = self._expand_weights(
            current_weights=self.d2_pathway.weights,
            n_new=n_new_neurons,
            initialization=initialization,
            sparsity=sparsity,
            scale=self.config.w_max * 0.2,
        )

        # =====================================================================
        # 2. UPDATE CONFIG (DO THIS BEFORE CREATING NEURONS!)
        # =====================================================================
        # Neurons are created based on config.n_output, so update it first
        # BOTH config.n_output and striatum_config.n_output store TOTAL NEURONS
        # We track n_actions separately in self.n_actions
        self.n_actions += n_new
        self.config = replace(self.config, n_output=new_n_output)
        self.striatum_config = replace(self.striatum_config, n_output=new_n_output)

        # Update elastic tensor capacity tracking (Phase 1)
        self.n_neurons_active = new_n_output
        # Check if we need to expand capacity
        if self.n_neurons_active > self.n_neurons_capacity:
            # Growth exceeded reserved capacity - reallocate with new headroom
            if self.striatum_config.growth_enabled:
                reserve_multiplier = 1.0 + self.striatum_config.reserve_capacity
                self.n_neurons_capacity = int(self.n_neurons_active * reserve_multiplier)
            else:
                self.n_neurons_capacity = self.n_neurons_active

        # =====================================================================
        # 3. EXPAND STATE TENSORS using base helper
        # =====================================================================
        # Build state dict for all 2D tensors [n_neurons, dim]
        state_2d = {
            'd1_eligibility': self.d1_pathway.eligibility,
            'd2_eligibility': self.d2_pathway.eligibility,
        }

        # Add TD-lambda traces if present
        if hasattr(self, 'td_lambda_d1') and self.td_lambda_d1 is not None:
            state_2d['td_lambda_d1_traces'] = self.td_lambda_d1.traces.traces
        if hasattr(self, 'td_lambda_d2') and self.td_lambda_d2 is not None:
            state_2d['td_lambda_d2_traces'] = self.td_lambda_d2.traces.traces

        # Expand all 2D state tensors at once
        expanded_2d = self._expand_state_tensors(state_2d, n_new_neurons)
        self.d1_pathway.eligibility = expanded_2d['d1_eligibility']
        self.d2_pathway.eligibility = expanded_2d['d2_eligibility']
        if hasattr(self, 'td_lambda_d1') and self.td_lambda_d1 is not None:
            self.td_lambda_d1.traces.traces = expanded_2d['td_lambda_d1_traces']
            self.td_lambda_d1.traces.n_output = new_n_output
            self.td_lambda_d1.n_actions = self.n_actions
        if hasattr(self, 'td_lambda_d2') and self.td_lambda_d2 is not None:
            self.td_lambda_d2.traces.traces = expanded_2d['td_lambda_d2_traces']
            self.td_lambda_d2.traces.n_output = new_n_output
            self.td_lambda_d2.n_actions = self.n_actions

        # Build state dict for all 1D tensors [n_neurons]
        state_1d = {
            'recent_spikes': self.recent_spikes,
        }

        # Expand all 1D state tensors at once
        expanded_1d = self._expand_state_tensors(state_1d, n_new_neurons)
        self.recent_spikes = expanded_1d['recent_spikes']

        # =====================================================================
        # 4. EXPAND NEURON POPULATIONS using efficient in-place growth (ConductanceLIF)
        # =====================================================================
        # Expand D1-MSN and D2-MSN neuron populations
        if hasattr(self, 'd1_pathway') and self.d1_pathway.neurons is not None:
            self.d1_pathway.neurons.grow_neurons(n_new_neurons)
        if hasattr(self, 'd2_pathway') and self.d2_pathway.neurons is not None:
            self.d2_pathway.neurons.grow_neurons(n_new_neurons)

        # =====================================================================
        # 5. UPDATE ACTION-RELATED TRACKING (1D per action, not per neuron)
        # =====================================================================
        # Expand D1/D2 vote accumulators [n_actions] via state_tracker
        self.state_tracker._d1_votes_accumulated = torch.cat([
            self.state_tracker._d1_votes_accumulated,
            torch.zeros(n_new, device=self.device)
        ], dim=0)
        self.state_tracker._d2_votes_accumulated = torch.cat([
            self.state_tracker._d2_votes_accumulated,
            torch.zeros(n_new, device=self.device)
        ], dim=0)

        # Expand exploration manager (handles action_counts and other exploration state)
        self.exploration.grow(self.n_actions)

        # Expand homeostasis manager if it exists
        if self.homeostasis is not None:
            self.homeostasis.grow(n_new_neurons)

        # Value estimates for new actions (start at 0)
        if hasattr(self, 'value_estimates'):
            self.value_estimates = torch.cat([
                self.value_estimates,
                torch.zeros(n_new, device=self.device)
            ], dim=0)

        # RPE traces for new actions (only if rpe_trace is enabled)
        if self.rpe_trace is not None:
            self.rpe_trace = torch.cat([
                self.rpe_trace,
                torch.zeros(n_new, device=self.device)
            ], dim=0)

        # =====================================================================
        # 6. EXPAND PFC MODULATION WEIGHTS (using base helper)
        # =====================================================================
        # PFC modulation weights [n_output, pfc_size] need to expand for new neurons
        if hasattr(self, 'pfc_modulation_d1') and self.pfc_modulation_d1 is not None:
            self.pfc_modulation_d1 = self._expand_weights(
                current_weights=self.pfc_modulation_d1,
                n_new=n_new_neurons,
                initialization='sparse_random',
                sparsity=0.3,
                scale=1.0,  # Default scale for PFC modulation
            )

        if hasattr(self, 'pfc_modulation_d2') and self.pfc_modulation_d2 is not None:
            self.pfc_modulation_d2 = self._expand_weights(
                current_weights=self.pfc_modulation_d2,
                n_new=n_new_neurons,
                initialization='sparse_random',
                sparsity=0.3,
                scale=1.0,  # Default scale for PFC modulation
            )

    def grow_input(
        self,
        n_new: int,
        initialization: str = 'xavier',
        sparsity: float = 0.1,
    ) -> None:
        """Grow striatum input dimension when upstream regions grow.

        When upstream regions (cortex, hippocampus, PFC) add neurons, this
        method expands the striatum's D1 and D2 pathway weights by delegating
        to the pathway objects' grow_input() method.

        Args:
            n_new: Number of input neurons to add
            initialization: Weight init strategy
            sparsity: Connection sparsity for new inputs

        Example:
            >>> cortex.grow_output(20)
            >>> cortex_to_striatum.grow_source('cortex', new_size)
            >>> striatum.grow_input(20)  # Expand D1/D2 input weights
        """
        old_n_input = self.config.n_input
        new_n_input = old_n_input + n_new

        # Delegate to internal pathway grow_input() methods
        # (Pathways handle weight expansion, eligibility reset)
        self.d1_pathway.grow_input(n_new_inputs=n_new, initialization=initialization)
        self.d2_pathway.grow_input(n_new_inputs=n_new, initialization=initialization)

        # Grow TD(λ) traces for both D1 and D2
        if self.td_lambda_d1 is not None:
            self.td_lambda_d1.grow_input(n_new)
        if self.td_lambda_d2 is not None:
            self.td_lambda_d2.grow_input(n_new)

        # Update striatum config
        self.config = replace(self.config, n_input=new_n_input)

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
            n_output=self.config.n_output,
            n_input=self.config.n_input,
            gain=0.2,  # Reduced gain for near-symmetric start
            device=self.device
        )

        # Scale by w_max and clamp to bounds
        weights = weights * self.config.w_max

        return clamp_weights(weights, self.config.w_min, self.config.w_max, inplace=False)

    def _update_d1_d2_eligibility(
        self, input_spikes: torch.Tensor, d1_spikes: torch.Tensor, d2_spikes: torch.Tensor,
        chosen_action: int | None = None
    ) -> None:
        """Update separate eligibility traces for D1 and D2 pathways.

        With SEPARATE neuron populations:
        - D1 eligibility is computed from input-D1 spike coincidence
        - D2 eligibility is computed from input-D2 spike coincidence

        CRITICAL: When chosen_action is provided, eligibility is ONLY built for
        the neurons corresponding to that action. This is biologically correct:
        only synapses where the post-synaptic neuron fired should become eligible.
        This prevents both actions from building eligibility from the same input.

        Timestep (dt) is obtained from self.config.dt_ms for temporal dynamics.

        Args:
            input_spikes: Input spike tensor [n_input] (1D)
            d1_spikes: D1 neuron population spikes [n_d1] (1D)
            d2_spikes: D2 neuron population spikes [n_d2] (1D)
            chosen_action: If provided, only build eligibility for this action's neurons
        """
        # Ensure 1D
        if input_spikes.dim() != 1:
            input_spikes = input_spikes.squeeze()
        if d1_spikes.dim() != 1:
            d1_spikes = d1_spikes.squeeze()
        if d2_spikes.dim() != 1:
            d2_spikes = d2_spikes.squeeze()

        # Get float versions for trace updates
        input_1d = input_spikes.float()
        d1_output_1d = d1_spikes.float()
        d2_output_1d = d2_spikes.float()

        # CRITICAL: Mask output spikes to ONLY include the chosen action's neurons
        # This is biologically correct: only synapses where post-synaptic neurons
        # actually fired should become eligible. Without this, both actions build
        # eligibility from the same input, causing learning instability.
        if chosen_action is not None:
            action_mask = torch.zeros_like(d1_output_1d)
            if self.striatum_config.population_coding:
                pop_slice = self._get_action_population_indices(chosen_action)
                action_mask[pop_slice] = 1.0
            else:
                action_mask[chosen_action] = 1.0
            d1_output_1d = d1_output_1d * action_mask
            d2_output_1d = d2_output_1d * action_mask

        # Update eligibility traces using pathway strategies
        # Note: D1 and D2 pathways have separate learning strategies
        self.d1_pathway.update_eligibility(input_1d, d1_output_1d)
        self.d2_pathway.update_eligibility(input_1d, d2_output_1d)

    def _update_d1_d2_eligibility_all(
        self, input_spikes: torch.Tensor, d1_spikes: torch.Tensor, d2_spikes: torch.Tensor
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
            input_spikes: Input spike tensor
            d1_spikes: D1 neuron population spikes
            d2_spikes: D2 neuron population spikes
        """
        # Just call the existing method with chosen_action=None
        # which will update eligibility for all active neurons
        self._update_d1_d2_eligibility(input_spikes, d1_spikes, d2_spikes, chosen_action=None)

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
            tau_adapt=100.0,        # Adaptation time constant
            adapt_increment=0.1,    # Enable spike-frequency adaptation
        )
        neurons = ConductanceLIF(n_neurons=self.config.n_output, config=neuron_config)
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
            tau_adapt=100.0,        # Adaptation time constant
            adapt_increment=0.1,    # Enable spike-frequency adaptation
        )
        neurons = ConductanceLIF(n_neurons=self.config.n_output, config=neuron_config)
        neurons.to(self.device)
        return neurons

    # endregion

    # region Forward Pass (D1/D2 Integration and Action Selection)

    def _consolidate_inputs(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
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
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device(self.device)

        # Apply synaptic weights and accumulate currents
        # Note: We accumulate rather than concatenate because both D1 and D2
        # receive the same inputs (biologically accurate)
        total_current = torch.zeros(self.n_neurons, device=device)

        for source_name, input_spikes in inputs.items():
            if source_name in self.synaptic_weights:
                # Apply synaptic weights at target dendrites
                current = self._apply_synapses(source_name, input_spikes)
                total_current += current

        return total_current

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
            inputs: Either:
                   - Dict mapping source names to spike tensors
                     e.g., {"cortex": [n_cortex], "hippocampus": [n_hippo]}
                   - Tensor of spikes (auto-wrapped as {"default": tensor}) [n_input]
                   PFC component (if present) is used for goal modulation.

        NOTE: Exploration is handled by finalize_action() at trial end, not per-timestep.
        NOTE: Theta modulation computed internally from self._theta_phase (set by Brain)

        With population coding:
        - Each action has N neurons per pathway (neurons_per_action)
        - D1_votes = sum(D1 spikes for action)
        - D2_votes = sum(D2 spikes for action)
        - NET = D1_votes - D2_votes
        - Selected action = argmax(NET)
        """
        # Concatenate all input sources (D1/D2 pathways apply their own weights)
        input_spikes = InputRouter.concatenate_sources(
            inputs,
            component_name="Striatum",
            n_input=self.config.n_input,
            device=self.device,
        )

        # Ensure 1D (ADR-005)
        if input_spikes.dim() != 1:
            input_spikes = input_spikes.squeeze()

        assert input_spikes.dim() == 1, (
            f"Striatum.forward: Expected 1D input (ADR-005), got shape {input_spikes.shape}. "
            "Thalia uses single-brain architecture with no batch dimension."
        )

        # =====================================================================
        # FSI (FAST-SPIKING INTERNEURONS) - Feedforward Inhibition
        # =====================================================================
        # FSI process inputs in parallel with MSNs but with:
        # 1. Gap junction coupling for synchronization (<0.1ms)
        # 2. Feedforward inhibition to MSNs (sharpens action timing)
        # Biology: FSI are parvalbumin+ interneurons (~2% of striatum)
        fsi_inhibition = torch.zeros(self.config.n_output, device=self.device)
        if self.fsi_size > 0:
            # Compute FSI synaptic current (weights applied at dendrites)
            fsi_weights = self.synaptic_weights["fsi"]
            fsi_synaptic_current = fsi_weights @ input_spikes  # [n_fsi]

            # Apply gap junction coupling (if enabled and state available)
            if self.gap_junctions_fsi is not None and self.state.fsi_membrane is not None:
                gap_current = self.gap_junctions_fsi(self.state.fsi_membrane)
                fsi_synaptic_current = fsi_synaptic_current + gap_current

            # Update FSI neurons (fast kinetics, tau_mem ~5ms)
            fsi_spikes, fsi_membrane = self.fsi_neurons(
                g_exc_input=fsi_synaptic_current,
                g_inh_input=torch.zeros_like(fsi_synaptic_current),  # FSI receive minimal inhibition
            )

            # Store FSI membrane for next timestep gap junctions
            self.state.fsi_membrane = fsi_membrane

            # FSI provide feedforward inhibition to ALL MSNs (broadcast)
            # Each FSI spike contributes 0.5 inhibitory conductance (strong!)
            fsi_inhibition = torch.sum(fsi_spikes) * 0.5

        # =====================================================================
        # FORWARD PASS COORDINATION - Delegate to ForwardPassCoordinator
        # =====================================================================
        # ForwardPassCoordinator handles all the complex modulation logic:
        # - D1/D2 pathway activation computation
        # - Theta/beta oscillator modulation
        # - Tonic dopamine and norepinephrine gain modulation
        # - Goal-conditioned modulation (PFC → Striatum)
        # - Homeostatic excitability modulation
        # - FSI feedforward inhibition (sharpens action timing)
        # - D1/D2 neuron population execution
        d1_spikes, d2_spikes, pfc_goal_context = self.forward_coordinator.forward(
            input_spikes=input_spikes,
            recent_spikes=self.state_tracker.recent_spikes,
            fsi_inhibition=fsi_inhibition,  # FSI feedforward inhibition
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
                    max_delay_steps, self.n_actions,
                    device=self.device, dtype=d1_votes_current.dtype
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
                    max_delay_steps, self.n_actions,
                    device=self.device, dtype=d2_votes_current.dtype
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
        # OUTPUT SPIKES: Return D1 activity (action selection happens in finalize_action)
        # =====================================================================
        # During the trial, we output D1 spikes for all neurons.
        # The final action selection (UCB + softmax + exploration) is handled by
        # finalize_action() at the end of the trial, not per-timestep.
        output_spikes = d1_spikes.clone()

        # =====================================================================
        # STORE GOAL CONTEXT AND SPIKES FOR LEARNING
        # =====================================================================
        # Store PFC goal context and D1/D2 spikes for goal-conditioned learning
        # These will be used in deliver_reward() to modulate weight updates
        self.state_tracker.store_spikes_for_learning(d1_spikes, d2_spikes, pfc_goal_context)

        # =====================================================================
        # UPDATE ELIGIBILITY TRACES (for all active neurons)
        # =====================================================================
        # Update D1/D2 STDP-style eligibility (always enabled)
        # Eligibility accumulates for ALL neurons that fire during the trial.
        # When reward arrives, deliver_reward() uses last_action (set by finalize_action)
        # to apply learning only to the chosen action's synapses.
        self._update_d1_d2_eligibility_all(input_spikes, d1_spikes, d2_spikes)

        # UPDATE TD(λ) ELIGIBILITY (if enabled)
        # TD(λ) traces accumulate with factor (γλ) instead of simple decay,
        # enabling credit assignment over longer delays (5-10 seconds)
        if self.td_lambda_d1 is not None:
            # Convert bool spikes to float for gradient computation
            input_spikes_float = input_spikes.float() if input_spikes.dtype == torch.bool else input_spikes

            # Update TD(λ) eligibility for D1 pathway
            # Note: We update for ALL neurons here; masking to chosen action
            # happens in deliver_reward() using last_action
            d1_gradient = torch.outer(d1_spikes.float(), input_spikes_float)
            self.td_lambda_d1.traces.update(d1_gradient)

            # Update TD(λ) eligibility for D2 pathway
            d2_gradient = torch.outer(d2_spikes.float(), input_spikes_float)
            self.td_lambda_d2.traces.update(d2_gradient)

        # Update D1/D2 eligibility traces for ALL active neurons
        # The action-specific masking happens in deliver_reward(), not here
        self._update_d1_d2_eligibility_all(input_spikes, d1_spikes, d2_spikes)

        # Update recent spikes and trial activity via state_tracker
        self.state_tracker.update_recent_spikes(d1_spikes, decay=0.9)
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

        Delegates to LearningManager for dopamine-gated three-factor learning.

        Args:
            reward: Raw reward signal (for adaptive exploration tracking only)

        Returns:
            Metrics dict with dopamine level and weight changes.
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

        # Delegate to learning manager (new component API)
        goal_context = self._last_pfc_goal_context if hasattr(self, '_last_pfc_goal_context') else None

        return self.learning.apply_learning(da_level, goal_context)


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
            cf_lr = self.striatum_config.rpe_learning_rate * counterfactual_scale
            self.value_estimates[action] = (
                self.value_estimates[action]
                + cf_lr * (reward - self.value_estimates[action])
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
                if subsystem is not None and hasattr(subsystem, 'reset_state'):
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
        - TD(λ) traces (if enabled)
        - Delay buffers (if enabled)
        """
        # Reset state tracker (votes, recent spikes, trial stats, last action)
        self.state_tracker.reset_state()

        # Reset D1/D2 pathways (eligibility and neurons)
        self.d1_pathway.reset_state()
        self.d2_pathway.reset_state()

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
        if hasattr(self, 'forward_coordinator'):
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

    def get_diagnostics(self) -> StriatumDiagnostics:
        """Get comprehensive diagnostics in standardized DiagnosticsDict format.

        Returns consolidated diagnostic information about:
        - Activity: Spike rates and population activity
        - Plasticity: D1/D2 pathway weights and eligibility traces
        - Health: Pathway balance and weight statistics
        - Neuromodulators: Dopamine levels
        - Region-specific: D1/D2 votes, value estimates, exploration state, etc.

        This is the primary diagnostic interface for the Striatum.
        """
        from thalia.core.diagnostics_schema import (
            compute_activity_metrics,
            compute_plasticity_metrics,
            compute_health_metrics,
        )

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
        recent_spikes = self.state_tracker.recent_spikes if self.state_tracker.recent_spikes is not None else torch.zeros(self.n_neurons, device=self.device)
        activity = compute_activity_metrics(
            output_spikes=recent_spikes,
            total_neurons=self.n_neurons,
        )

        # Compute plasticity metrics for D1 pathway (representative)
        plasticity = compute_plasticity_metrics(
            weights=self.d1_pathway.weights,
            learning_rate=self.striatum_config.learning_rate,
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
            "norepinephrine": self.forward_coordinator._ne_level if hasattr(self.forward_coordinator, '_ne_level') else 0.0,
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
            "enabled": self.striatum_config.ucb_exploration,
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

    def get_state(self) -> "StriatumState":
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
        from thalia.regions.striatum.config import StriatumState

        # Get D1/D2 pathway states (opaque dicts)
        d1_pathway_state = self.d1_pathway.get_state()
        d2_pathway_state = self.d2_pathway.get_state()

        # Get exploration manager state
        exploration_manager_state = self.exploration.get_state() if hasattr(self.exploration, 'get_state') else None

        # Get homeostasis manager state
        homeostasis_manager_state = None
        if self.homeostasis is not None and hasattr(self.homeostasis.unified_homeostasis, 'get_state'):
            homeostasis_manager_state = self.homeostasis.unified_homeostasis.get_state()

        # Get neuron membrane state (if neurons exist and have membrane)
        membrane = None
        if self.d1_pathway.neurons is not None and hasattr(self.d1_pathway.neurons, 'membrane'):
            if self.d1_pathway.neurons.membrane is not None:
                membrane = self.d1_pathway.neurons.membrane.detach().clone()

        # Get neuromodulator levels from forward_coordinator
        dopamine = self.forward_coordinator._tonic_dopamine if hasattr(self.forward_coordinator, '_tonic_dopamine') else 0.0
        norepinephrine = self.forward_coordinator._ne_level if hasattr(self.forward_coordinator, '_ne_level') else 0.0
        acetylcholine = self.forward_coordinator._ach_level if hasattr(self.forward_coordinator, '_ach_level') else 0.0

        return StriatumState(
            # Base state
            spikes=self.state_tracker.recent_spikes.detach().clone() if self.state_tracker.recent_spikes is not None else None,
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
            value_estimates=self.value_estimates.detach().clone() if hasattr(self, 'value_estimates') and self.value_estimates is not None else None,
            last_rpe=self.state_tracker._last_rpe if hasattr(self.state_tracker, '_last_rpe') else None,
            last_expected=self.state_tracker._last_expected if hasattr(self.state_tracker, '_last_expected') else None,

            # Goal modulation (optional)
            pfc_modulation_d1=self.pfc_modulation_d1.detach().clone() if hasattr(self, 'pfc_modulation_d1') and self.pfc_modulation_d1 is not None else None,
            pfc_modulation_d2=self.pfc_modulation_d2.detach().clone() if hasattr(self, 'pfc_modulation_d2') and self.pfc_modulation_d2 is not None else None,

            # Delay buffers (optional)
            d1_delay_buffer=self._d1_delay_buffer.detach().clone() if hasattr(self, '_d1_delay_buffer') and self._d1_delay_buffer is not None else None,
            d2_delay_buffer=self._d2_delay_buffer.detach().clone() if hasattr(self, '_d2_delay_buffer') and self._d2_delay_buffer is not None else None,
            d1_delay_ptr=self._d1_delay_ptr if hasattr(self, '_d1_delay_ptr') else 0,
            d2_delay_ptr=self._d2_delay_ptr if hasattr(self, '_d2_delay_ptr') else 0,

            # Homeostasis
            activity_ema=self._activity_ema if hasattr(self, '_activity_ema') else 0.0,
            trial_spike_count=self._trial_spike_count if hasattr(self, '_trial_spike_count') else 0,
            trial_timesteps=self._trial_timesteps if hasattr(self, '_trial_timesteps') else 0,
            homeostatic_scaling_applied=self._homeostatic_scaling_applied if hasattr(self, '_homeostatic_scaling_applied') else False,
            homeostasis_manager_state=homeostasis_manager_state,

            # STP state (optional)
            stp_corticostriatal_u=self.stp_corticostriatal.u.detach().clone() if (self.stp_corticostriatal is not None and self.stp_corticostriatal.u is not None) else None,
            stp_corticostriatal_x=self.stp_corticostriatal.x.detach().clone() if (self.stp_corticostriatal is not None and self.stp_corticostriatal.x is not None) else None,
            stp_thalamostriatal_u=self.stp_thalamostriatal.u.detach().clone() if (self.stp_thalamostriatal is not None and self.stp_thalamostriatal.u is not None) else None,
            stp_thalamostriatal_x=self.stp_thalamostriatal.x.detach().clone() if (self.stp_thalamostriatal is not None and self.stp_thalamostriatal.x is not None) else None,

            # Neuromodulators
            dopamine=dopamine,
            acetylcholine=acetylcholine,
            norepinephrine=norepinephrine,
        )

    def load_state(self, state: "StriatumState") -> None:
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
        if state.membrane is not None and self.d1_pathway.neurons is not None:
            if hasattr(self.d1_pathway.neurons, 'membrane'):
                self.d1_pathway.neurons.membrane.data = state.membrane.to(self.device)

        # Restore vote accumulation
        if state.d1_votes_accumulated is not None:
            self.state_tracker._d1_votes_accumulated.data = state.d1_votes_accumulated.to(self.device)
        if state.d2_votes_accumulated is not None:
            self.state_tracker._d2_votes_accumulated.data = state.d2_votes_accumulated.to(self.device)

        # Restore action selection
        self.state_tracker.last_action = state.last_action
        if state.recent_spikes is not None:
            self.state_tracker.recent_spikes.data = state.recent_spikes.to(self.device)

        # Restore exploration
        self.state_tracker.exploring = state.exploring
        self.state_tracker._last_uncertainty = state.last_uncertainty
        self.state_tracker._last_exploration_prob = state.last_exploration_prob
        if state.exploration_manager_state is not None and hasattr(self.exploration, 'load_state'):
            self.exploration.load_state(state.exploration_manager_state)

        # Restore value/RPE (optional)
        if state.value_estimates is not None and hasattr(self, 'value_estimates'):
            self.value_estimates.data = state.value_estimates.to(self.device)
        if state.last_rpe is not None:
            self.state_tracker._last_rpe = state.last_rpe
        if state.last_expected is not None:
            self.state_tracker._last_expected = state.last_expected

        # Restore STP state (optional)
        if state.stp_corticostriatal_u is not None and self.stp_corticostriatal is not None and self.stp_corticostriatal.u is not None:
            self.stp_corticostriatal.u.data = state.stp_corticostriatal_u.to(self.device)
        if state.stp_corticostriatal_x is not None and self.stp_corticostriatal is not None and self.stp_corticostriatal.x is not None:
            self.stp_corticostriatal.x.data = state.stp_corticostriatal_x.to(self.device)
        if state.stp_thalamostriatal_u is not None and self.stp_thalamostriatal is not None and self.stp_thalamostriatal.u is not None:
            self.stp_thalamostriatal.u.data = state.stp_thalamostriatal_u.to(self.device)
        if state.stp_thalamostriatal_x is not None and self.stp_thalamostriatal is not None and self.stp_thalamostriatal.x is not None:
            self.stp_thalamostriatal.x.data = state.stp_thalamostriatal_x.to(self.device)

        # Restore goal modulation (optional)
        if state.pfc_modulation_d1 is not None and hasattr(self, 'pfc_modulation_d1'):
            self.pfc_modulation_d1.data = state.pfc_modulation_d1.to(self.device)
        if state.pfc_modulation_d2 is not None and hasattr(self, 'pfc_modulation_d2'):
            self.pfc_modulation_d2.data = state.pfc_modulation_d2.to(self.device)

        # Restore delay buffers (optional)
        if state.d1_delay_buffer is not None and hasattr(self, '_d1_delay_buffer'):
            self._d1_delay_buffer = state.d1_delay_buffer.to(self.device)
            self._d1_delay_ptr = state.d1_delay_ptr
        if state.d2_delay_buffer is not None and hasattr(self, '_d2_delay_buffer'):
            self._d2_delay_buffer = state.d2_delay_buffer.to(self.device)
            self._d2_delay_ptr = state.d2_delay_ptr

        # Restore homeostasis
        if hasattr(self, '_activity_ema'):
            self._activity_ema = state.activity_ema
        if hasattr(self, '_trial_spike_count'):
            self._trial_spike_count = state.trial_spike_count
        if hasattr(self, '_trial_timesteps'):
            self._trial_timesteps = state.trial_timesteps
        if hasattr(self, '_homeostatic_scaling_applied'):
            self._homeostatic_scaling_applied = state.homeostatic_scaling_applied
        if state.homeostasis_manager_state is not None and self.homeostasis is not None:
            if hasattr(self.homeostasis.unified_homeostasis, 'load_state'):
                self.homeostasis.unified_homeostasis.load_state(state.homeostasis_manager_state)

        # Restore neuromodulators to forward_coordinator
        if hasattr(self.forward_coordinator, '_tonic_dopamine'):
            self.forward_coordinator._tonic_dopamine = state.dopamine
        if hasattr(self.forward_coordinator, '_ne_level'):
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
