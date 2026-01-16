"""
Prefrontal Cortex - Gated Working Memory and Executive Control.

The prefrontal cortex (PFC) specializes in cognitive control and flexible behavior:
- **Working memory maintenance**: Actively maintain information over delays
- **Rule learning**: Learn context-dependent stimulus-response mappings
- **Executive control**: Top-down attention and behavioral inhibition
- **Goal-directed behavior**: Plan and execute multi-step goal hierarchies

**Key Features**:
=================
1. **GATED WORKING MEMORY**:
   - Active maintenance against decay via recurrent excitation
   - Dopamine gates what enters/updates working memory
   - Similar to LSTM/GRU gating in deep learning, but biological
   - Persistent activity emerges from network dynamics (not explicit state)

2. **DOPAMINE GATING MECHANISM**:
   - DA burst (>threshold) → "update gate open" → new info enters WM
   - DA baseline → "maintain" → protect current WM contents
   - DA dip → "clear" → allow WM to decay
   - Gates both learning AND maintenance

3. **CONTEXT-DEPENDENT LEARNING**:
   - Rule neurons represent abstract task rules
   - Same input → different outputs based on context/rule
   - Enables flexible behavior switching
   - Supports cognitive flexibility and set-shifting

4. **SLOW INTEGRATION**:
   - Longer time constants than sensory cortex (τ ~500ms)
   - Integrates information over longer timescales
   - Supports temporal abstraction and planning

FILE ORGANIZATION (1072 lines)
===============================
Lines 1-85:    Module docstring, imports, PrefrontalConfig
Lines 86-260:  Prefrontal class __init__, weight initialization
Lines 261-430: Forward pass (input → WM update → output)
Lines 431-590: Working memory gating logic (dopamine-modulated)
Lines 591-760: Rule learning and context-dependent processing
Lines 761-900: Growth and neurogenesis (grow_output)
Lines 901-1020: Diagnostics and health monitoring
Lines 1021-1072: Utility methods (reset_state, get_full_state)

NAVIGATION TIP: Use VSCode's "Go to Symbol" (Ctrl+Shift+O) to jump between methods.

Biological Basis:
=================
- Layer 2/3 recurrent circuits for WM maintenance
- D1/D2 receptors modulate gain and gating
- Strong connections with striatum (for action selection)
- Connections with hippocampus (for episodic retrieval)

When to Use:
============
- Working memory tasks (maintain info over delays)
- Rule learning (learn context-dependent responses)
- Sequence generation (use rules to generate behavior)
- Any task requiring flexible, goal-directed control
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union

import torch
import torch.nn as nn

from thalia.components.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.components.synapses import ShortTermPlasticity, STPConfig, STPType, WeightInitializer
from thalia.constants.learning import LEARNING_RATE_STDP, WM_NOISE_STD_DEFAULT
from thalia.constants.oscillator import (
    PFC_FEEDFORWARD_GAIN_MIN,
    PFC_FEEDFORWARD_GAIN_RANGE,
    PFC_RECURRENT_GAIN_MIN,
    PFC_RECURRENT_GAIN_RANGE,
)
from thalia.core.base.component_config import NeuralComponentConfig
from thalia.core.errors import ConfigurationError
from thalia.core.neural_region import NeuralRegion
from thalia.core.region_state import BaseRegionState
from thalia.learning import LearningStrategyRegistry, STDPConfig
from thalia.learning.homeostasis.synaptic_homeostasis import UnifiedHomeostasis, UnifiedHomeostasisConfig
from thalia.managers.component_registry import register_region
from thalia.neuromodulation import compute_ne_gain, DA_BASELINE_STANDARD
from thalia.regions.prefrontal.checkpoint_manager import PrefrontalCheckpointManager
from thalia.regions.prefrontal.hierarchy import (
    Goal,
    GoalHierarchyManager,
    GoalHierarchyConfig,
    HyperbolicDiscounter,
    HyperbolicDiscountingConfig,
)
from thalia.typing import PrefrontalDiagnostics
from thalia.utils.input_routing import InputRouter
from thalia.utils.oscillator_utils import compute_theta_encoding_retrieval, compute_oscillator_modulated_gain


@dataclass
class PrefrontalConfig(NeuralComponentConfig):
    """Configuration specific to prefrontal cortex.

    PFC implements DOPAMINE-GATED STDP:
    - STDP creates eligibility traces from spike timing
    - Dopamine gates what enters working memory and what gets learned
    - High DA → update WM and learn new associations
    - Low DA → maintain WM and protect existing patterns

    **Size Specification** (Semantic-First):
    - Sizes passed via sizes dict: input_size, n_neurons
    - Computed at instantiation: output_size (= n_neurons), total_neurons (= n_neurons)
    - Config contains only behavioral parameters (learning rates, time constants, etc.)
    """

    # Working memory parameters
    wm_decay_tau_ms: float = 500.0  # How fast WM decays (slow!)
    wm_noise_std: float = 0.01  # Noise in WM maintenance

    # Gating parameters
    gate_threshold: float = 0.5  # DA level to open update gate
    gate_strength: float = 2.0  # How strongly gating affects updates

    # Dopamine parameters
    dopamine_tau_ms: float = 100.0  # DA decay time constant
    dopamine_baseline: float = 0.2  # Tonic DA level

    # Learning rates
    wm_lr: float = 0.1  # Learning rate for WM update weights
    rule_lr: float = LEARNING_RATE_STDP  # Learning rate for rule weights
    # Note: STDP parameters (stdp_lr, tau_plus_ms, tau_minus_ms, a_plus, a_minus)
    # and heterosynaptic_ratio (0.3) are inherited from NeuralComponentConfig

    # Recurrent connections for WM maintenance
    recurrent_strength: float = 0.8  # Self-excitation for persistence
    recurrent_inhibition: float = 0.2  # Lateral inhibition

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION (SFA)
    # =========================================================================
    # PFC pyramidal neurons show adaptation. This helps prevent runaway
    # activity during sustained working memory maintenance.
    # Inherited from base, with PFC-specific overrides:
    adapt_increment: float = 0.2  # Moderate (maintains WM while adapting)
    adapt_tau: float = 150.0      # Slower decay (longer timescale for WM)

    # =========================================================================
    # SHORT-TERM PLASTICITY (STP)
    # =========================================================================
    # Feedforward connections: Facilitation for temporal filtering
    stp_feedforward_enabled: bool = True
    """Enable short-term plasticity on feedforward (input) connections."""

    # Recurrent connections: Depression prevents frozen attractors
    stp_recurrent_enabled: bool = True
    """Enable short-term plasticity on recurrent connections."""

    # =========================================================================
    # PHASE 3: HIERARCHICAL GOALS & TEMPORAL ABSTRACTION
    # =========================================================================
    # Hierarchical goal management and hyperbolic discounting
    use_hierarchical_goals: bool = True
    """Enable hierarchical goal structures (Phase 3).

    When True:
        - Maintains goal hierarchy stack in working memory
        - Tracks active goals and subgoals
        - Supports options framework for reusable policies
        - Requires goal_hierarchy_config
    """

    goal_hierarchy_config: Optional["GoalHierarchyConfig"] = None
    """Configuration for goal hierarchy manager (Phase 3)."""

    use_hyperbolic_discounting: bool = True
    """Enable hyperbolic temporal discounting (Phase 3).

    When True:
        - Hyperbolic (not exponential) discounting of delayed rewards
        - Context-dependent k parameter (cognitive load, stress, fatigue)
        - Adaptive k learning from experience
        - Requires hyperbolic_config
    """

    hyperbolic_config: Optional["HyperbolicDiscountingConfig"] = None
    """Configuration for hyperbolic discounter (Phase 3)."""

    # =========================================================================
    # HETEROGENEOUS WORKING MEMORY (Phase 1B Enhancement)
    # =========================================================================
    # Biological reality: PFC neurons show heterogeneous maintenance properties
    # - Stable neurons: Strong recurrence, long time constants (~1-2s)
    # - Flexible neurons: Weak recurrence, short time constants (~100-200ms)
    #
    # This heterogeneity enables:
    # - Stable neurons: Maintain context/goals over long delays
    # - Flexible neurons: Rapid updating for new information
    # - Mixed selectivity: Distributed representations across neuron types
    #
    # References:
    # - Rigotti et al. (2013): Mixed selectivity in prefrontal cortex
    # - Murray et al. (2017): Stable population coding for working memory
    # - Wasmuht et al. (2018): Intrinsic neuronal dynamics in PFC
    use_heterogeneous_wm: bool = False  # Enable heterogeneous WM neurons
    stability_cv: float = 0.3  # Coefficient of variation for recurrent strength
    tau_mem_min: float = 100.0  # Minimum membrane time constant (ms) - flexible neurons
    tau_mem_max: float = 500.0  # Maximum membrane time constant (ms) - stable neurons

    # =========================================================================
    # D1/D2 DOPAMINE RECEPTOR SUBTYPES (Phase 1B Enhancement)
    # =========================================================================
    # Biological reality: PFC has both D1 (excitatory) and D2 (inhibitory) receptors
    # - D1-dominant neurons (~60%): "Go" pathway, enhance signals with DA
    # - D2-dominant neurons (~40%): "NoGo" pathway, suppress noise with DA
    #
    # This enables:
    # - D1: Update WM when DA high (new information is important)
    # - D2: Maintain WM when DA low (protect current state)
    # - Opponent modulation: D1 and D2 have opposite DA responses
    #
    # References:
    # - Seamans & Yang (2004): D1 and D2 dopamine systems in PFC
    # - Durstewitz & Seamans (2008): Neurocomputational perspective on PFC
    # - Cools & D'Esposito (2011): Inverted-U dopamine in working memory
    use_d1_d2_subtypes: bool = False  # Enable D1/D2 receptor subtypes
    d1_fraction: float = 0.6  # Fraction of neurons that are D1-dominant (60%)
    d1_da_gain: float = 0.5  # DA gain for D1 neurons (excitatory, 1.0 + gain*DA)
    d2_da_gain: float = 0.3  # DA suppression for D2 neurons (inhibitory, 1.0 - gain*DA)
    d2_output_weight: float = 0.5  # Weight of D2 output in competition (D1 - weight*D2)

    def __post_init__(self) -> None:
        """Auto-compute output_size and total_neurons from n_neurons."""
        # Properties handle this now - no manual assignment needed
        pass


@dataclass
class PrefrontalState(BaseRegionState):
    """State for prefrontal cortex region.

    Implements RegionState protocol for checkpoint compatibility.
    Inherits from BaseRegionState for common fields (spikes, membrane, neuromodulators).

    PFC-specific state:
    - working_memory: Active maintenance of task-relevant information
    - update_gate: Dopamine-gated update signals
    - active_rule: Current task rule representation
    """

    STATE_VERSION: int = 1

    # Inherited from BaseRegionState:
    # - spikes: Optional[torch.Tensor] = None
    # - membrane: Optional[torch.Tensor] = None
    # - dopamine: float = 0.0
    # - acetylcholine: float = 0.0
    # - norepinephrine: float = 0.0

    # PFC-specific state fields
    working_memory: Optional[torch.Tensor] = None
    """Working memory contents [n_neurons]."""

    update_gate: Optional[torch.Tensor] = None
    """Gate state for WM updates [n_neurons]."""

    active_rule: Optional[torch.Tensor] = None
    """Rule representation [n_neurons]."""

    # STP state (recurrent connections)
    stp_recurrent_state: Optional[Dict[str, Any]] = None
    """Short-term plasticity state for recurrent connections."""

    stp_feedforward_state: Optional[Dict[str, Any]] = None
    """Short-term plasticity state for feedforward connections."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for checkpointing."""
        base_dict = super().to_dict()
        base_dict.update({
            'working_memory': self.working_memory,
            'update_gate': self.update_gate,
            'active_rule': self.active_rule,
            'dopamine': self.dopamine,
            'acetylcholine': self.acetylcholine,
            'norepinephrine': self.norepinephrine,
            'stp_recurrent_state': self.stp_recurrent_state,
            'stp_feedforward_state': self.stp_feedforward_state,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: str) -> "PrefrontalState":
        """Deserialize state from dictionary."""
        # Future: Handle version migration if needed

        # Get base state
        base_state = BaseRegionState.from_dict(data, device)

        # Transfer PFC-specific tensors
        wm = data.get('working_memory')
        if wm is not None and isinstance(wm, torch.Tensor):
            wm = wm.to(device)

        gate = data.get('update_gate')
        if gate is not None and isinstance(gate, torch.Tensor):
            gate = gate.to(device)

        rule = data.get('active_rule')
        if rule is not None and isinstance(rule, torch.Tensor):
            rule = rule.to(device)

        # Transfer nested STP state tensors
        stp_recurrent = data.get('stp_recurrent_state')
        if stp_recurrent is not None:
            stp_recurrent = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in stp_recurrent.items()
            }

        stp_feedforward = data.get('stp_feedforward_state')
        if stp_feedforward is not None:
            stp_feedforward = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in stp_feedforward.items()
            }

        return cls(
            spikes=base_state.spikes,
            membrane=base_state.membrane,
            working_memory=wm,
            update_gate=gate,
            active_rule=rule,
            dopamine=data.get('dopamine', 0.2),
            acetylcholine=data.get('acetylcholine', 0.0),
            norepinephrine=data.get('norepinephrine', 0.0),
            stp_recurrent_state=stp_recurrent,
            stp_feedforward_state=stp_feedforward,
        )

    def reset(self) -> None:
        """Reset state to initial conditions."""
        # Reset base fields (spikes, membrane, neuromodulators with DA_BASELINE_STANDARD)
        super().reset()

        # Reset PFC-specific state
        self.working_memory = None
        self.update_gate = None
        self.active_rule = None
        self.stp_recurrent_state = None


# =============================================================================
# HETEROGENEOUS WM NEURON SAMPLING (Phase 1B Enhancement)
# =============================================================================


def sample_heterogeneous_wm_neurons(
    n_neurons: int,
    stability_cv: float = 0.3,
    tau_mem_min: float = 100.0,
    tau_mem_max: float = 500.0,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Sample heterogeneous working memory neuron properties.

    Creates a distribution of neurons with varying maintenance capabilities:
    - Stable neurons: Strong recurrence, long time constants (~500ms)
    - Flexible neurons: Weak recurrence, short time constants (~100ms)

    This heterogeneity enables:
    - Stable neurons maintain context/goals over long delays
    - Flexible neurons enable rapid updating for new information
    - Mixed selectivity for distributed representations

    Biological motivation:
    - Real PFC neurons show 2-10× variability in maintenance properties
    - Heterogeneity provides robustness and mixed selectivity
    - Enables both persistent representations and flexible updating

    References:
    - Rigotti et al. (2013): Mixed selectivity in prefrontal cortex
    - Murray et al. (2017): Stable population coding for working memory
    - Wasmuht et al. (2018): Intrinsic neuronal dynamics in PFC

    Args:
        n_neurons: Number of neurons to sample
        stability_cv: Coefficient of variation for recurrent strength (default 0.3)
        tau_mem_min: Minimum membrane time constant in ms (default 100.0)
        tau_mem_max: Maximum membrane time constant in ms (default 500.0)
        device: Device for tensors ('cpu' or 'cuda')
        seed: Random seed for reproducibility (optional)

    Returns:
        Dictionary with:
        - recurrent_strength: [n_neurons] tensor of recurrent weights (0.2-1.0 range)
        - tau_mem: [n_neurons] tensor of membrane time constants (100-500ms)
        - neuron_type: [n_neurons] tensor of 0 (flexible) or 1 (stable) labels

    Example:
        >>> props = sample_heterogeneous_wm_neurons(100, stability_cv=0.3)
        >>> props['recurrent_strength']  # Shape: [100], mean ~0.6, CV ~0.3
        >>> props['tau_mem']              # Shape: [100], range 100-500ms
        >>> props['neuron_type']          # Shape: [100], 0=flexible, 1=stable
    """
    if seed is not None:
        torch.manual_seed(seed)

    device_obj = torch.device(device)

    # Sample recurrent strength from lognormal distribution
    # Mean = 0.6 (moderate recurrence), CV = stability_cv
    # This creates a distribution with:
    # - Lower tail: Flexible neurons (weak recurrence ~0.2-0.4)
    # - Upper tail: Stable neurons (strong recurrence ~0.8-1.0)
    mean_recurrent = 0.6
    std_recurrent = mean_recurrent * stability_cv

    # Lognormal parameters: log_mean, log_std
    # For lognormal: mean = exp(μ + σ²/2), var = [exp(σ²) - 1] * exp(2μ + σ²)
    # Solve for μ, σ given desired mean and std
    log_var = torch.log(torch.tensor(1.0 + (std_recurrent / mean_recurrent) ** 2))
    log_std = torch.sqrt(log_var)
    log_mean = torch.log(torch.tensor(mean_recurrent)) - log_var / 2

    # Sample from lognormal
    recurrent_strength = torch.distributions.LogNormal(log_mean, log_std).sample((n_neurons,))

    # Clamp to reasonable range [0.2, 1.0]
    # 0.2 = minimum for any persistent activity
    # 1.0 = maximum stability (approaches attractor)
    recurrent_strength = torch.clamp(recurrent_strength, 0.2, 1.0)

    # Tau_mem scales with recurrent strength
    # Stable neurons (high recurrence) have longer time constants
    # Linear mapping: recurrent 0.2→100ms, recurrent 1.0→500ms
    tau_mem = tau_mem_min + (tau_mem_max - tau_mem_min) * (recurrent_strength - 0.2) / 0.8

    # Classify neurons as flexible (0) or stable (1)
    # Threshold at median: lower half = flexible, upper half = stable
    median_strength = torch.median(recurrent_strength)
    neuron_type = (recurrent_strength > median_strength).long()

    # Move to device
    recurrent_strength = recurrent_strength.to(device_obj)
    tau_mem = tau_mem.to(device_obj)
    neuron_type = neuron_type.to(device_obj)

    return {
        "recurrent_strength": recurrent_strength,
        "tau_mem": tau_mem,
        "neuron_type": neuron_type,
    }


class DopamineGatingSystem:
    """Dopamine-based gating for working memory updates.

    Unlike striatal dopamine (which determines LTP vs LTD direction),
    prefrontal dopamine gates what information enters working memory:
    - High DA → gate open → update WM with new input
    - Low DA → gate closed → maintain current WM
    """

    def __init__(
        self,
        n_neurons: int,
        tau_ms: float = 100.0,
        baseline: float = 0.2,
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.n_neurons = n_neurons
        self.tau_ms = tau_ms
        self.baseline = baseline
        self.threshold = threshold
        self.device = torch.device(device)

        self.level = baseline  # Current DA level

    def reset_state(self) -> None:
        """Reset to baseline."""
        self.level = self.baseline

    def update(self, signal: float, dt: float = 1.0) -> float:
        """Update dopamine level with new signal.

        Args:
            signal: External dopamine signal (-1 to 1)
            dt: Timestep in ms

        Returns:
            Current dopamine level
        """
        # Decay toward baseline
        decay = torch.exp(torch.tensor(-dt / self.tau_ms)).item()
        self.level = self.baseline + (self.level - self.baseline) * decay

        # Add signal
        self.level += signal

        # Clamp to valid range
        self.level = max(0.0, min(1.0, self.level))

        return self.level

    def get_gate(self) -> float:
        """Get current gating value (0-1).

        Returns smooth gate value based on dopamine level.
        """
        # Sigmoid around threshold
        return 1.0 / (1.0 + torch.exp(torch.tensor(
            -10 * (self.level - self.threshold)
        )).item())

    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "level": self.level,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.level = state["level"]


@register_region(
    "prefrontal",
    aliases=["pfc"],
    description="Working memory and executive control with dopamine-gated updates and rule learning",
    version="2.0",
    author="Thalia Project",
    config_class=PrefrontalConfig,
)
class Prefrontal(NeuralRegion):
    """Prefrontal cortex with dopamine-gated working memory.

    Implements:
    - Working memory maintenance via recurrent connections
    - Dopamine gating of updates (similar to LSTM gates)
    - Rule learning and context-dependent behavior
    - Slow integration for temporal abstraction

    Inherited from LearnableComponent:
    ----------------------------------
    From LearningStrategyMixin (via LearnableComponent):
        - add_strategy(strategy) → None
        - apply_strategy_learning(pre, post, **kwargs) → Dict
        - Pluggable learning rules (STDP with dopamine modulation)

    From base class:
        - forward(input, **kwargs) → Tensor [must implement]
        - reset_state() → None
        - get_diagnostics() → Dict
        - set_dopamine(level) → None
        - Neuromodulator control methods

    See Also:
        docs/patterns/mixins.md for detailed mixin patterns
        docs/patterns/state-management.md for PrefrontalState
    """

    def __init__(self, config: PrefrontalConfig, sizes: Dict[str, int], device: str):
        """
        Initialize prefrontal cortex.

        Args:
            config: PFC configuration (behavioral parameters only)
            sizes: Size specification {'input_size': int, 'n_neurons': int}
            device: Device ('cpu' or 'cuda')
        """
        # Store config
        self.pfc_config = config
        self.config = config  # For backward compatibility
        self.device = torch.device(device)

        # Extract sizes
        self.input_size = sizes['input_size']
        self.n_neurons = sizes['n_neurons']
        self.n_output = self.n_neurons  # PFC output = neuron count
        self.total_neurons = self.n_neurons

        # =====================================================================
        # HETEROGENEOUS WORKING MEMORY NEURONS (Phase 1B)
        # =====================================================================
        # Sample heterogeneous recurrent strengths and time constants
        # mimicking biological diversity in WM stability across neurons
        if config.use_heterogeneous_wm:
            wm_properties = sample_heterogeneous_wm_neurons(
                n_neurons=self.n_neurons,
                stability_cv=config.stability_cv,
                tau_mem_min=config.tau_mem_min,
                tau_mem_max=config.tau_mem_max,
                device=self.device,
                seed=None,  # Use random seed for variability
            )
            self._recurrent_strength = wm_properties["recurrent_strength"]
            self._tau_mem_heterogeneous = wm_properties["tau_mem"]
            self._neuron_type = wm_properties["neuron_type"]  # 0=flexible, 1=stable
        else:
            self._recurrent_strength = None
            self._tau_mem_heterogeneous = None
            self._neuron_type = None

        # =====================================================================
        # D1/D2 RECEPTOR SUBTYPES (Phase 1B)
        # =====================================================================
        # Split neurons into D1-dominant (excitatory DA response) and
        # D2-dominant (inhibitory DA response) populations
        if config.use_d1_d2_subtypes:
            n_d1 = int(self.n_neurons * config.d1_fraction)
            self._d1_neurons = torch.arange(n_d1, device=self.device)
            self._d2_neurons = torch.arange(n_d1, self.n_neurons, device=self.device)
        else:
            self._d1_neurons = None
            self._d2_neurons = None

        # Initialize NeuralRegion with total neurons
        super().__init__(
            n_neurons=self.n_neurons,
            neuron_config=ConductanceLIFConfig(
                g_L=0.02,  # Slower leak (τ_m ≈ 50ms)
                tau_E=10.0,  # Slower excitatory (for integration)
                tau_I=15.0,  # Slower inhibitory
                adapt_increment=config.adapt_increment,
                tau_adapt=config.adapt_tau,
            ),
            default_learning_rule="stdp",
            device=device,
            dt_ms=config.dt_ms,
        )

        # Override neurons to add STP (NeuralRegion creates basic neurons)
        self.neurons = self._create_neurons()

        # Learning control (specific to prefrontal cortex)
        self.plasticity_enabled: bool = True

        # Register feedforward input source and initialize weights
        self.add_input_source("default", n_input=self.input_size)
        # Initialize with Xavier (better than NeuralRegion's default)
        self.synaptic_weights["default"].data = WeightInitializer.xavier(
            n_output=self.n_neurons,
            n_input=self.input_size,
            gain=1.0,
            device=self.device
        )

        # Recurrent weights for WM maintenance
        self.rec_weights = nn.Parameter(
            WeightInitializer.gaussian(
                n_output=self.n_neurons,
                n_input=self.n_neurons,
                mean=0.0,
                std=0.1,
                device=self.device
            ),
            requires_grad=False
        )
        # Initialize with self-excitation (heterogeneous if enabled)
        if config.use_heterogeneous_wm:
            # Scale diagonal by heterogeneous recurrent strengths
            diag_strength = torch.diag(self._recurrent_strength)
            self.rec_weights.data += diag_strength
        else:
            # Uniform self-excitation
            self.rec_weights.data += torch.eye(
                self.n_neurons, device=self.device
            ) * config.recurrent_strength

        # Lateral inhibition weights
        self.inhib_weights = nn.Parameter(
            torch.ones(self.n_neurons, self.n_neurons, device=self.device)
            * config.recurrent_inhibition
        )
        self.inhib_weights.data.fill_diagonal_(0.0)

        # Dopamine gating system
        self.dopamine_system = DopamineGatingSystem(
            n_neurons=self.n_neurons,
            tau_ms=config.dopamine_tau_ms,
            baseline=config.dopamine_baseline,
            threshold=config.gate_threshold,
            device=config.device,
        )

        # Initialize checkpoint manager for neuromorphic format support
        self.checkpoint_manager = PrefrontalCheckpointManager(self)

        # Initialize learning strategy (STDP with dopamine gating)
        # Using LearningStrategyRegistry for pluggable learning strategies
        self.learning_strategy = LearningStrategyRegistry.create(
            "stdp",
            STDPConfig(
                learning_rate=config.learning_rate,
                a_plus=config.a_plus,
                a_minus=config.a_minus,
                tau_plus=config.tau_plus_ms,
                tau_minus=config.tau_minus_ms,
                dt_ms=config.dt_ms,
                w_min=config.w_min,
                w_max=config.w_max,
            )
        )

        # Homeostasis for synaptic scaling
        homeostasis_config = UnifiedHomeostasisConfig(
            weight_budget=config.weight_budget * self.input_size,  # Total budget per neuron
            w_min=config.w_min,
            w_max=config.w_max,
            soft_normalization=config.soft_normalization,
            normalization_rate=config.normalization_rate,
            device=device,
        )
        self.homeostasis = UnifiedHomeostasis(homeostasis_config)

        # Initialize neurogenesis history tracking
        # Tracks the creation timestep for each neuron (for checkpoint analysis)
        self._neuron_birth_steps = torch.zeros(self.n_neurons, dtype=torch.long, device=self.device)
        self._current_training_step = 0  # Updated externally by training loop

        # Initialize working memory state (1D tensors, ADR-005)
        self.state = PrefrontalState(
            working_memory=torch.zeros(self.n_neurons, device=self.device),
            update_gate=torch.zeros(self.n_neurons, device=self.device),
            dopamine=config.dopamine_baseline,
        )

        # Initialize theta phase for modulation
        self._theta_phase: float = 0.0

        # Move all components to target device
        self.to(self.device)

        # =====================================================================
        # PHASE 3: HIERARCHICAL GOALS & TEMPORAL ABSTRACTION
        # =====================================================================
        # Initialize goal hierarchy (Phase 3)
        self.goal_manager: Optional["GoalHierarchyManager"] = None
        self.discounter: Optional["HyperbolicDiscounter"] = None

        if config.use_hierarchical_goals:
            gh_config = config.goal_hierarchy_config or GoalHierarchyConfig()
            self.goal_manager = GoalHierarchyManager(gh_config)

            # Hyperbolic discounting
            if config.use_hyperbolic_discounting:
                hd_config = config.hyperbolic_config or HyperbolicDiscountingConfig()
                self.discounter = HyperbolicDiscounter(hd_config)

    def _create_neurons(self) -> ConductanceLIF:
        """Create conductance-based LIF neurons with slow dynamics and SFA.

        PFC neurons have significantly different dynamics than standard pyramidal neurons:
        - Much slower leak (τ_m ≈ 50ms vs 20ms) for temporal integration
        - Slower synaptic time constants for sustained integration
        - Spike-frequency adaptation for stable working memory
        """
        cfg = self.pfc_config
        # Custom config for PFC-specific slow dynamics
        neuron_config = ConductanceLIFConfig(
            g_L=0.02,  # Slower leak (τ_m ≈ 50ms with C_m=1.0)
            tau_E=10.0,  # Slower excitatory (for integration)
            tau_I=15.0,  # Slower inhibitory
            adapt_increment=cfg.adapt_increment,  # SFA enabled!
            tau_adapt=cfg.adapt_tau,
        )
        neurons = ConductanceLIF(self.n_neurons, neuron_config)
        neurons.to(self.device)

        # =====================================================================
        # SHORT-TERM PLASTICITY for feedforward connections
        # =====================================================================
        # PFC feedforward connections show SHORT-TERM FACILITATION/DEPRESSION
        # for temporal filtering and gain control during encoding.
        if cfg.stp_feedforward_enabled:
            self.stp_feedforward = ShortTermPlasticity(
                n_pre=self.input_size,
                n_post=self.n_neurons,
                config=STPConfig.from_type(STPType.FACILITATING, dt=cfg.dt_ms),
                per_synapse=True,
            )
            self.stp_feedforward.to(self.device)
        else:
            self.stp_feedforward = None

        # =====================================================================
        # SHORT-TERM PLASTICITY for recurrent connections
        # =====================================================================
        # PFC recurrent connections show SHORT-TERM DEPRESSION, preventing
        # frozen attractors. This allows working memory to be updated.
        if cfg.stp_recurrent_enabled:
            self.stp_recurrent = ShortTermPlasticity(
                n_pre=self.n_neurons,
                n_post=self.n_neurons,
                config=STPConfig.from_type(STPType.DEPRESSING, dt=cfg.dt_ms),
                per_synapse=True,
            )
            self.stp_recurrent.to(self.device)
        else:
            self.stp_recurrent = None

        # =====================================================================
        # Phase 2 Registration: Opt-in auto-growth for STP modules
        if self.stp_feedforward is not None:
            # Feedforward STP (input -> n_output): grows during grow_input (pre) and grow_output (post)
            self._register_stp('stp_feedforward', direction='both', recurrent=False)

        if self.stp_recurrent is not None:
            # Recurrent STP (n_output -> n_output): ONLY grows during grow_output (both pre and post)
            # NOT during grow_input - recurrent connections track n_output, not n_input
            self._register_stp('stp_recurrent', direction='post', recurrent=True)

        return neurons

    def _reset_subsystems(self, *names: str) -> None:
        """Reset state of named subsystems that have reset_state() method.

        Helper from BrainComponentBase for backward compatibility.
        """
        for name in names:
            if hasattr(self, name):
                subsystem = getattr(self, name)
                if subsystem is not None and hasattr(subsystem, 'reset_state'):
                    subsystem.reset_state()

    def reset_state(self) -> None:
        """Reset state for new episode."""
        # Don't call super().reset_state() because it creates NeuralComponentState
        # Instead, create PrefrontalState directly with proper tensor shapes
        self.state = PrefrontalState(
            working_memory=torch.zeros(self.n_neurons, device=self.device),
            update_gate=torch.zeros(self.n_neurons, device=self.device),
            active_rule=None,  # Optional, can be None
            dopamine=DA_BASELINE_STANDARD,
        )

        # Reset subsystems using helper
        self._reset_subsystems('neurons', 'dopamine_system', 'stp_recurrent', 'stp_feedforward')

    def forward(
        self,
        inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        dopamine_signal: float = 0.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Process input through prefrontal cortex.

        Args:
            inputs: Input spikes - Dict mapping source names to spike tensors,
                   or single Tensor (auto-wrapped as {"default": tensor}) [n_input]
            dopamine_signal: External DA signal for gating (-1 to 1)
            **kwargs: Additional inputs

        Returns:
            Output spikes [n_output] (1D bool tensor, ADR-005)

        Note:
            Theta modulation and timestep (dt_ms) computed internally from config
        """
        # Route input to default port
        routed = InputRouter.route(
            inputs,
            port_mapping={"default": ["default", "input"]},
            defaults={"default": torch.zeros(self.input_size, device=self.device)},
            component_name="PrefrontalCortex",
        )
        input_spikes = routed["default"]

        # Get timestep from config for temporal dynamics
        dt = self.config.dt_ms

        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        assert input_spikes.dim() == 1, (
            f"PrefrontalCortex.forward: input_spikes must be 1D [n_input], "
            f"got shape {input_spikes.shape}. See ADR-005: No Batch Dimension."
        )
        assert input_spikes.shape[0] == self.input_size, (
            f"PrefrontalCortex.forward: input_spikes has shape {input_spikes.shape} "
            f"but input_size={self.input_size}. Check that input matches PFC config."
        )

        # Ensure state is initialized
        if self.state.working_memory is None:
            self.reset_state()

        # Update dopamine and get gate value
        da_level = self.dopamine_system.update(dopamine_signal, dt)
        gate = self.dopamine_system.get_gate()
        self.state.dopamine = da_level

        # NOTE: All neuromodulators (DA, ACh, NE) are now managed centrally by Brain.
        # VTA updates dopamine, LC updates NE, NB updates ACh.
        # Brain broadcasts to all regions every timestep via _update_neuromodulators().
        # No local decay needed.

        # =====================================================================
        # THETA MODULATION
        # =====================================================================
        # Compute theta modulation from current phase (set by Brain's OscillatorManager)
        encoding_mod, retrieval_mod = compute_theta_encoding_retrieval(self._theta_phase)

        # Encoding phase (theta trough): gate new info into WM
        # Retrieval phase (theta peak): maintain WM and boost recurrence
        ff_gain = compute_oscillator_modulated_gain(PFC_FEEDFORWARD_GAIN_MIN, PFC_FEEDFORWARD_GAIN_RANGE, encoding_mod)
        rec_gain = compute_oscillator_modulated_gain(PFC_RECURRENT_GAIN_MIN, PFC_RECURRENT_GAIN_RANGE, retrieval_mod)

        # Feedforward input - modulated by encoding phase
        # Apply STP if enabled (temporal filtering and gain control)
        if hasattr(self, 'stp_feedforward') and self.stp_feedforward is not None:
            # Apply STP to feedforward connections (1D → 2D per-synapse efficacy)
            # stp_efficacy has shape [n_output, n_input] - per-synapse modulation
            stp_efficacy = self.stp_feedforward(input_spikes.float())
            # Effective weights: element-wise multiply with STP efficacy
            effective_ff_weights = self.synaptic_weights["default"] * stp_efficacy.t()
            # Apply synaptic weights: weights[n_output, n_input] @ input[n_input] → [n_output]
            ff_input = (effective_ff_weights @ input_spikes.float()) * ff_gain
        else:
            # No STP: direct feedforward
            # Apply synaptic weights: weights[n_output, n_input] @ input[n_input] → [n_output]
            ff_input = (self.synaptic_weights["default"] @ input_spikes.float()) * ff_gain

        # =====================================================================
        # NOREPINEPHRINE GAIN MODULATION (Locus Coeruleus)
        # =====================================================================
        # High NE (arousal/uncertainty): Increase gain → more responsive WM
        # Low NE (baseline): Normal gain
        # Biological: β-adrenergic receptors modulate PFC excitability and
        # working memory flexibility (Arnsten 2009)
        ne_level = self.state.norepinephrine
        # NE gain: 1.0 (baseline) to 1.5 (high arousal)
        ne_gain = compute_ne_gain(ne_level)
        ff_input = ff_input * ne_gain

        # =====================================================================
        # RECURRENT INPUT WITH STP (prevents frozen WM attractors)
        # =====================================================================
        # Without STP, the same WM pattern is reinforced forever.
        # With DEPRESSING STP, frequently-used synapses get temporarily weaker,
        # allowing WM to be updated with new information.
        if (hasattr(self, 'stp_recurrent') and self.stp_recurrent is not None
            and self.state.working_memory is not None):
            # Apply STP to recurrent connections (1D → 2D per-synapse efficacy)
            # stp_efficacy has shape [n_output, n_output] - per-synapse modulation
            stp_efficacy = self.stp_recurrent(
                self.state.working_memory.float()
            )
            # Effective weights: element-wise multiply rec_weights with STP efficacy
            # rec_weights is [n_output, n_output], stp_efficacy is [n_output, n_output]
            effective_rec_weights = self.rec_weights * stp_efficacy.t()
            # Recurrent: weights[n_output, n_output] @ wm[n_output] → [n_output]
            rec_input = (effective_rec_weights @ self.state.working_memory.float()) * rec_gain
        else:
            # Recurrent input from working memory - modulated by retrieval phase
            # rec_weights[n_output, n_output] @ wm[n_output] → [n_output]
            wm = self.state.working_memory.float() if self.state.working_memory is not None else torch.zeros(self.n_neurons, device=input_spikes.device)
            rec_input = (self.rec_weights @ wm) * rec_gain

        # Lateral inhibition: inhib_weights[n_output, n_output] @ wm[n_output] → [n_output]
        wm = self.state.working_memory.float() if self.state.working_memory is not None else torch.zeros(self.n_neurons, device=input_spikes.device)
        inhib = self.inhib_weights @ wm

        # Total excitation and inhibition
        g_exc = (ff_input + rec_input).clamp(min=0)
        g_inh = inhib.clamp(min=0)

        # Run through neurons (returns 1D bool spikes)
        output_spikes, _ = self.neurons(g_exc, g_inh)

        # =====================================================================
        # D1/D2 RECEPTOR SUBTYPES - Differential Dopamine Modulation (Phase 1B)
        # =====================================================================
        # D1-dominant neurons: DA increases excitability (excitatory response)
        # D2-dominant neurons: DA decreases excitability (inhibitory response)
        # Biological: D1 receptors increase cAMP → enhanced firing
        #            D2 receptors decrease cAMP → reduced firing
        if self.pfc_config.use_d1_d2_subtypes and da_level != 0.0:
            # Create output buffer for modulated activity
            modulated_output = output_spikes.float().clone()

            # D1 neurons: Excitatory DA response (gain boost)
            d1_gain = 1.0 + self.pfc_config.d1_da_gain * da_level
            modulated_output[self._d1_neurons] *= d1_gain

            # D2 neurons: Inhibitory DA response (gain reduction)
            d2_gain = 1.0 - self.pfc_config.d2_da_gain * da_level
            modulated_output[self._d2_neurons] *= d2_gain

            # Convert back to spikes (probabilistic based on modulated activity)
            # High activity → more likely to spike
            spike_probs = modulated_output.clamp(0, 1)
            output_spikes = (torch.rand_like(spike_probs) < spike_probs).bool()

        # Update working memory with gating
        # High gate (high DA) → update with new activity
        # Low gate (low DA) → maintain current WM
        gate_tensor = torch.full_like(self.state.working_memory, gate)
        self.state.update_gate = gate_tensor

        # WM decay
        decay = torch.exp(torch.tensor(-dt / self.pfc_config.wm_decay_tau_ms))

        # Gated update: WM = gate * new_input + (1-gate) * decayed_old
        new_wm = (
            gate_tensor * output_spikes.float() +
            (1 - gate_tensor) * self.state.working_memory * decay
        )

        # Add noise for stochasticity
        wm_noise_std = getattr(self.pfc_config, 'wm_noise_std', WM_NOISE_STD_DEFAULT)
        noise = torch.randn_like(new_wm) * wm_noise_std
        self.state.working_memory = (new_wm + noise).clamp(min=0, max=1)

        # Store state
        self.state.spikes = output_spikes

        # Output shape check
        assert output_spikes.shape == (self.n_neurons,), (
            f"PrefrontalCortex.forward: output_spikes has shape {output_spikes.shape} "
            f"but expected ({self.n_neurons},). "
            f"Check PFC neuron or weight configuration."
        )
        assert output_spikes.dtype == torch.bool, (
            f"PrefrontalCortex.forward: output_spikes must be bool (ADR-004), "
            f"got {output_spikes.dtype}"
        )

        # Apply continuous plasticity (learning happens as part of forward dynamics)
        self._apply_plasticity(input_spikes, output_spikes)

        # Store output (NeuralRegion pattern)
        self.output_spikes = output_spikes

        return output_spikes

    def _apply_plasticity(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
    ) -> None:
        """Apply dopamine-gated STDP learning using strategy pattern.

        This is called automatically at each forward() timestep.
        Uses the learning strategy system for consistent plasticity application.
        """
        if not self.plasticity_enabled:
            return

        cfg = self.pfc_config
        # Input/output are already 1D bool tensors (ADR-005)

        # Apply STDP learning via strategy
        # Dopamine modulation is handled automatically by apply_strategy_learning
        metrics = self.apply_strategy_learning(
            pre_activity=input_spikes,
            post_activity=output_spikes,
            weights=self.synaptic_weights["default"],
        )

        # Optional: Apply synaptic scaling for homeostasis
        if cfg.homeostasis_enabled and metrics:
            self.synaptic_weights["default"].data = self.homeostasis.normalize_weights(
                self.synaptic_weights["default"].data, dim=1
            )

        # ======================================================================
        # Update recurrent weights to strengthen WM patterns
        # ======================================================================
        # Rule learning now happens via dopamine-modulated STDP in _apply_plasticity
        # This simple Hebbian update for recurrent connections maintains WM patterns
        if self.state.working_memory is not None:
            # working_memory is already 1D [n_output] (ADR-005)
            wm = self.state.working_memory  # [n_output]
            dW_rec = cfg.rule_lr * torch.outer(wm, wm)  # [n_output, n_output]
            self.rec_weights.data += dW_rec
            self.rec_weights.data.fill_diagonal_(
                cfg.recurrent_strength
            )  # Maintain self-excitation
            self.rec_weights.data.clamp_(0.0, 1.0)

    def grow_input(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Grow prefrontal input dimension when upstream region grows.

        Expands input weight matrix columns to accept larger input.

        Args:
            n_new: Number of input neurons to add
            initialization: Weight init strategy ('sparse_random', 'xavier', 'uniform')
            sparsity: Connection sparsity for new input neurons (if sparse_random)
        """
        old_n_input = self.input_size

        # Use GrowthMixin helper (Architecture Review 2025-12-24, Tier 2.5)
        self.synaptic_weights["default"].data = self._grow_weight_matrix_cols(
            self.synaptic_weights["default"].data,
            n_new,
            initializer=initialization,
            sparsity=sparsity
        )

        # NOTE: STP auto-growth via Phase 2 registration system:
        # - stp_feedforward (if enabled): auto-grows via _auto_grow_registered_components('input')
        # - stp_recurrent: only grows in grow_output() (tracks n_output, not n_input)

        # Update instance variable
        self.input_size += n_new

        # Auto-grow registered STP modules (Phase 2)
        self._auto_grow_registered_components('input', n_new)

        # Validate growth completed correctly
        self._validate_input_growth(old_n_input, n_new)

    def grow_output(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Grow prefrontal output dimension (working memory capacity).

        Expands working memory neuron population and all associated weights.

        Args:
            n_new: Number of neurons to add
            initialization: Weight init strategy ('sparse_random', 'xavier', 'uniform')
            sparsity: Connection sparsity for new neurons (if sparse_random)
        """
        old_n_output = self.n_neurons
        new_n_output = old_n_output + n_new

        # Use GrowthMixin helpers (Architecture Review 2025-12-24, Tier 2.5)
        # 1. Expand synaptic_weights["default"] [n_output, input] → [n_output+n_new, input]
        self.synaptic_weights["default"].data = self._grow_weight_matrix_rows(
            self.synaptic_weights["default"].data,
            n_new,
            initializer=initialization,
            sparsity=sparsity
        )

        # 2. Expand rec_weights [n_output, n_output] → [n_output+n_new, n_output+n_new]
        # First add rows, then add columns
        expanded_rec = self._grow_weight_matrix_rows(
            self.rec_weights.data,
            n_new,
            initializer=initialization,
            sparsity=sparsity
        )
        self.rec_weights.data = self._grow_weight_matrix_cols(
            expanded_rec,
            n_new,
            initializer=initialization,
            sparsity=sparsity
        )
        # Add self-excitation for new neurons
        for i in range(n_new):
            self.rec_weights.data[old_n_output + i, old_n_output + i] = self.pfc_config.recurrent_strength

        # 3. Expand inhib_weights [n_output, n_output] → [n_output+n_new, n_output+n_new]
        new_inhib_rows = torch.ones(n_new, old_n_output, device=self.device) * self.pfc_config.recurrent_inhibition
        expanded_inhib = torch.cat([self.inhib_weights.data, new_inhib_rows], dim=0)
        new_inhib_cols = torch.ones(new_n_output, n_new, device=self.device) * self.pfc_config.recurrent_inhibition
        self.inhib_weights.data = torch.cat([expanded_inhib, new_inhib_cols], dim=1)
        # Zero diagonal (no self-inhibition)
        self.inhib_weights.data.fill_diagonal_(0.0)

        # 4. Expand neurons using efficient in-place growth
        self.neurons.grow_neurons(n_new)

        # 4.5. Track neurogenesis history for new neurons
        # Record creation timestep for checkpoint analysis
        new_birth_steps = torch.full((n_new,), self._current_training_step, dtype=torch.long, device=self.device)
        self._neuron_birth_steps = torch.cat([self._neuron_birth_steps, new_birth_steps])

        # 5. Update dopamine gating system
        self.dopamine_system = DopamineGatingSystem(
            n_neurons=new_n_output,
            tau_ms=self.pfc_config.dopamine_tau_ms,
            baseline=self.pfc_config.dopamine_baseline,
            threshold=self.pfc_config.gate_threshold,
            device=self.device,
        )

        # 5.5. Expand state tensors for new neurons
        if self.state.working_memory is not None:
            new_wm = torch.zeros(n_new, device=self.device)
            self.state.working_memory = torch.cat([self.state.working_memory, new_wm])

        if self.state.update_gate is not None:
            new_gate = torch.zeros(n_new, device=self.device)
            self.state.update_gate = torch.cat([self.state.update_gate, new_gate])

        if self.state.active_rule is not None:
            new_rule = torch.zeros(n_new, device=self.device)
            self.state.active_rule = torch.cat([self.state.active_rule, new_rule])

        # 5.6. Phase 2: Auto-grow registered STP modules
        self._auto_grow_registered_components('output', n_new)

        # 6. Update instance variables
        self.n_neurons = new_n_output
        self.n_output = new_n_output
        self.total_neurons = new_n_output

        # 7. Validate growth completed correctly (skip config check - using n_neurons not n_output)
        self._validate_output_growth(old_n_output, n_new, check_config=False)

    def grow_neurons(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Grow PFC neuron population (SEMANTIC API).

        Args:
            n_new: Number of PFC neurons to add
            initialization: Weight init strategy
            sparsity: Connection sparsity

        Note:
            Expands working memory capacity and cognitive control population.
        """
        self.grow_output(n_new, initialization, sparsity)

    def set_training_step(self, step: int) -> None:
        """Update the current training step for neurogenesis tracking.

        This should be called by the training loop to keep track of when neurons
        are created during growth events.

        Args:
            step: Current global training step
        """
        self._current_training_step = step

    def get_diagnostics(self) -> PrefrontalDiagnostics:
        """Get diagnostics using DiagnosticsMixin helpers.

        Reports working memory state, gating, and weight statistics.
        """
        cfg = self.pfc_config

        # Custom metrics specific to PFC
        custom = {
            "n_output": self.n_neurons,
            "gate_mean": self.state.update_gate.mean().item() if self.state.update_gate is not None else 0.0,
            "gate_std": self.state.update_gate.std().item() if self.state.update_gate is not None else 0.0,
            "wm_mean": self.state.working_memory.mean().item() if self.state.working_memory is not None else 0.0,
            "wm_std": self.state.working_memory.std().item() if self.state.working_memory is not None else 0.0,
            "wm_active": (self.state.working_memory > 0.1).sum().item() if self.state.working_memory is not None else 0,
            "dopamine_level": self.state.dopamine,
            "config_w_min": cfg.w_min,
            "config_w_max": cfg.w_max,
        }

        # Use collect_standard_diagnostics for weight and spike statistics
        return self.collect_standard_diagnostics(
            region_name="prefrontal",
            weight_matrices={
                "feedforward": self.synaptic_weights["default"].data,
                "recurrent": self.rec_weights.data,
                "inhibition": self.inhib_weights.data,
            },
            spike_tensors={
                "output": self.state.spikes,
            },
            custom_metrics=custom,
        )

    def predict_next_state(
        self,
        current_state: torch.Tensor,
        action: int,
        n_actions: Optional[int] = None
    ) -> torch.Tensor:
        """
        Predict next state using working memory dynamics.

        For Phase 2 model-based planning: simulates what state would result
        from taking an action from current state.

        Uses PFC's recurrent dynamics and working memory to generate predictions.
        This is a simplified predictor - in full implementation, would use
        interaction with hippocampus for episode-based prediction.

        Biology: PFC maintains mental simulations during planning (Daw et al., 2005).
        Prefrontal neurons show prospective coding - representing future states
        before they occur (Fuster, 2001).

        Args:
            current_state: Current state representation [n_output] (1D, ADR-005)
            action: Action index to simulate
            n_actions: Total number of possible actions (for one-hot encoding)

        Returns:
            predicted_next_state: Predicted next state [n_output] (1D, ADR-005)

        Note:
            This is a basic predictor. For more accurate predictions, use
            MentalSimulationCoordinator which combines PFC + Hippocampus +
            Cortex predictive coding.
        """
        # Default n_actions if not provided
        if n_actions is None:
            n_actions = 10  # Default, should be passed from config

        # One-hot encode action
        action_one_hot = torch.zeros(n_actions, device=self.device)
        action_one_hot[action] = 1.0

        # Concatenate state and action
        # State: [n_output], Action: [n_actions] → Combined: [n_output + n_actions]
        state_action = torch.cat([current_state, action_one_hot])

        # Use recurrent weights to predict next state
        # Simple linear prediction (can be enhanced with nonlinearity)
        # Project concatenated state-action through recurrent weights
        if state_action.shape[0] == self.rec_weights.shape[1]:
            # If dimensions match, use recurrent weights directly
            prediction = self.rec_weights @ state_action
        else:
            # If dimensions don't match, project to appropriate size first
            # Use feedforward weights to project to output space, then recurrent
            if hasattr(self, 'weights'):
                # First project state to output space
                state_projection = self.rec_weights @ current_state

                # Simple action modulation: scale by action strength
                action_modulation = 1.0 + 0.1 * (action_one_hot.sum() - 0.5)

                prediction = state_projection * action_modulation
            else:
                # Fallback: simple recurrent prediction
                prediction = self.rec_weights @ current_state

        # Apply nonlinearity (tanh to keep bounded)
        prediction = torch.tanh(prediction)

        # Add small amount of noise (stochastic prediction)
        if self.training:
            wm_noise_std = getattr(self.pfc_config, 'wm_noise_std', WM_NOISE_STD_DEFAULT)
            noise = torch.randn_like(prediction) * wm_noise_std
            prediction = prediction + noise

        return prediction

    # =========================================================================
    # PHASE 3: HIERARCHICAL GOALS & TEMPORAL ABSTRACTION
    # =========================================================================

    def set_goal_hierarchy(self, root_goal: "Goal") -> None:
        """
        Set the top-level goal for hierarchical planning.

        Phase 3 functionality: Enables goal decomposition and hierarchical control.

        Args:
            root_goal: Top-level goal to achieve

        Raises:
            ValueError: If hierarchical goals not enabled

        Example:
            essay_goal = Goal(goal_id=0, name="write_essay", level=3)
            pfc.set_goal_hierarchy(essay_goal)
        """
        if self.goal_manager is None:
            raise ConfigurationError("Hierarchical goals not enabled. Set use_hierarchical_goals=True in config.")
        self.goal_manager.set_root_goal(root_goal)

    def push_goal(self, goal: "Goal") -> None:
        """
        Push a goal onto the active goal stack.

        Phase 3 functionality: Activates a goal for pursuit.

        Args:
            goal: Goal to activate

        Raises:
            ValueError: If hierarchical goals not enabled

        Example:
            subgoal = Goal(goal_id=1, name="research_topic", level=2)
            pfc.push_goal(subgoal)
        """
        if self.goal_manager is None:
            raise ConfigurationError("Hierarchical goals not enabled.")
        self.goal_manager.push_goal(goal)

    def get_active_goals(self) -> List["Goal"]:
        """
        Get list of currently active goals.

        Phase 3 functionality: Returns all goals in the working memory stack.

        Returns:
            List of active goals (empty list if none or manager disabled)

        Example:
            goals = pfc.get_active_goals()
            print(f"Working on {len(goals)} goals")
        """
        if self.goal_manager is None:
            return []
        return self.goal_manager.active_goals.copy()

    def decompose_current_goal(self, state: torch.Tensor) -> List["Goal"]:
        """
        Decompose current goal into subgoals based on state.

        Phase 3 functionality: Enables hierarchical planning.

        Args:
            state: Current state for context-dependent decomposition

        Returns:
            List of subgoals (empty if no current goal or manager disabled)

        Example:
            state = pfc.state.spikes.float()
            subgoals = pfc.decompose_current_goal(state)
            for sg in subgoals:
                pfc.push_goal(sg)
        """
        if self.goal_manager is None:
            return []

        current_goal = self.goal_manager.get_current_goal()
        if current_goal is None:
            return []

        return self.goal_manager.decompose_goal(current_goal, state)

    def update_cognitive_load(self, load: float) -> None:
        """
        Update cognitive load (affects temporal discounting).

        Phase 3 functionality: Higher load increases impulsivity (higher k).

        Args:
            load: Cognitive load level (0-1)

        Example:
            # High working memory load
            pfc.update_cognitive_load(0.8)
            # Now temporal discounting will be steeper (more impulsive)
        """
        if self.discounter is not None:
            self.discounter.update_context(cognitive_load=load)

    def evaluate_delayed_reward(
        self,
        reward: float,
        delay: int
    ) -> float:
        """
        Discount delayed reward (hyperbolic or exponential).

        Phase 3 functionality: If hyperbolic discounting enabled, uses
        context-dependent k parameter. Otherwise falls back to exponential.

        Args:
            reward: Reward magnitude
            delay: Delay in timesteps

        Returns:
            Discounted value of delayed reward

        Example:
            # Under low cognitive load, patient
            pfc.update_cognitive_load(0.1)
            v1 = pfc.evaluate_delayed_reward(10.0, 100)

            # Under high cognitive load, impulsive
            pfc.update_cognitive_load(0.9)
            v2 = pfc.evaluate_delayed_reward(10.0, 100)

            assert v2 < v1  # More discounting when loaded
        """
        if self.discounter is not None:
            # Hyperbolic discounting with context
            return self.discounter.discount(reward, delay)
        else:
            # Fallback: Exponential discounting
            gamma = 0.99
            return reward * (gamma ** delay)

    def get_state(self) -> PrefrontalState:
        """Get current state for checkpointing.

        Returns PrefrontalState compatible with RegionState protocol.
        Includes all dynamic state for checkpoint save/load.

        Returns:
            PrefrontalState with current region state
        """
        # Capture STP state if enabled
        stp_recurrent_state = None
        if self.stp_recurrent is not None:
            stp_recurrent_state = self.stp_recurrent.get_state()

        stp_feedforward_state = None
        if hasattr(self, 'stp_feedforward') and self.stp_feedforward is not None:
            stp_feedforward_state = self.stp_feedforward.get_state()

        return PrefrontalState(
            spikes=self.state.spikes.clone() if self.state.spikes is not None else None,
            membrane=self.neurons.v.clone() if hasattr(self.neurons, 'v') else None,
            working_memory=self.state.working_memory.clone() if self.state.working_memory is not None else None,
            update_gate=self.state.update_gate.clone() if self.state.update_gate is not None else None,
            active_rule=self.state.active_rule.clone() if self.state.active_rule is not None else None,
            dopamine=self.state.dopamine,
            acetylcholine=self.state.acetylcholine,
            norepinephrine=self.state.norepinephrine,
            stp_recurrent_state=stp_recurrent_state,
            stp_feedforward_state=stp_feedforward_state,
        )

    def load_state(self, state: PrefrontalState) -> None:
        """Load state from checkpoint.

        Restores region state from PrefrontalState instance.
        Compatible with RegionState protocol.

        Args:
            state: PrefrontalState to restore
        """
        # Use mixin helpers for common restoration
        super().load_state(state)  # Restores: membrane, conductances, traces, neuromodulators

        # PFC-specific state restoration
        self._load_custom_state(state)

    def _load_custom_state(self, state: PrefrontalState) -> None:
        """Restore PFC-specific state components.

        Args:
            state: PrefrontalState to restore from
        """
        # Restore basic state
        if state.spikes is not None:
            self.state.spikes = state.spikes.to(self.device).clone()
        if state.membrane is not None and hasattr(self.neurons, 'v'):
            self.neurons.v = state.membrane.to(self.device).clone()

        # Restore PFC-specific state
        if state.working_memory is not None:
            self.state.working_memory = state.working_memory.to(self.device).clone()
        if state.update_gate is not None:
            self.state.update_gate = state.update_gate.to(self.device).clone()
        if state.active_rule is not None:
            self.state.active_rule = state.active_rule.to(self.device).clone()

        # Neuromodulators already restored by super().load_state() via _restore_neuromodulators()

        # Restore STP state
        if state.stp_recurrent_state is not None and self.stp_recurrent is not None:
            self.stp_recurrent.load_state(state.stp_recurrent_state)

        if state.stp_feedforward_state is not None and hasattr(self, 'stp_feedforward') and self.stp_feedforward is not None:
            self.stp_feedforward.load_state(state.stp_feedforward_state)

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for checkpointing.

        Returns state dictionary with keys:
        - weights: Feedforward, recurrent, and inhibition weights
        - region_state: Neuron state, working memory, spikes
        - learning_state: STDP eligibility traces, STP state
        - neuromodulator_state: Dopamine gating state
        - config: Configuration for validation
        """
        state_obj = self.get_state()
        state = state_obj.to_dict()

        # Add all weights (required for checkpointing)
        # PFC has both synaptic_weights dict (feedforward) and rec_weights (recurrent)
        state['synaptic_weights'] = {
            name: weights.detach().clone()
            for name, weights in self.synaptic_weights.items()
        }
        if hasattr(self, 'rec_weights'):
            state['rec_weights'] = self.rec_weights.detach().clone()

        return state

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Load complete state from checkpoint.

        Args:
            state: State dictionary from get_full_state()
        """
        state_obj = PrefrontalState.from_dict(state, device=str(self.device))
        self.load_state(state_obj)

        # Restore synaptic weights
        if 'synaptic_weights' in state:
            for name, weights in state['synaptic_weights'].items():
                if name in self.synaptic_weights:
                    self.synaptic_weights[name].data = weights.to(self.device)

        # Restore recurrent weights
        if 'rec_weights' in state and hasattr(self, 'rec_weights'):
            self.rec_weights.data = state['rec_weights'].to(self.device)
