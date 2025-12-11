"""
Prefrontal Cortex - Gated Working Memory and Executive Control

The prefrontal cortex (PFC) is specialized for:
- Working memory maintenance
- Rule learning and flexible switching
- Executive control and decision making
- Goal-directed behavior

Key Features:
=============
1. GATED WORKING MEMORY:
   - Information can be actively maintained against decay
   - Dopamine gates what enters/updates working memory
   - Persistent activity through recurrent connections

2. DOPAMINE GATING:
   - DA burst → "update gate open" → allow new info into WM
   - DA dip → "maintain" → protect current WM contents
   - Similar to LSTM/GRU gating mechanisms in deep learning

3. CONTEXT-DEPENDENT LEARNING:
   - Rule neurons learn to represent abstract rules
   - Same sensory input → different outputs based on context
   - Supports flexible behavior switching

4. SLOW INTEGRATION:
   - Longer time constants than sensory areas
   - Allows integration over longer timescales
   - Supports temporal abstraction

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
from typing import TYPE_CHECKING, Optional, Dict, Any

import torch
import torch.nn as nn

from thalia.learning import LearningStrategyRegistry, STDPConfig

from thalia.core.utils import clamp_weights, cosine_similarity_safe
from thalia.core.stp import ShortTermPlasticity, STPConfig, STPType
from thalia.core.weight_init import WeightInitializer
from thalia.core.component_registry import register_region
from thalia.core.learning_constants import LEARNING_RATE_STDP
from thalia.regions.base import (
    NeuralComponent,
    RegionConfig,
    RegionState,
    LearningRule,
)
from thalia.core.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.core.neuron_constants import NE_GAIN_RANGE

if TYPE_CHECKING:
    from thalia.regions.prefrontal_hierarchy import Goal, GoalHierarchyManager, GoalHierarchyConfig
    from thalia.regions.prefrontal_hierarchy import HyperbolicDiscounter, HyperbolicDiscountingConfig


@dataclass
class PrefrontalConfig(RegionConfig):
    """Configuration specific to prefrontal cortex.

    PFC implements DOPAMINE-GATED STDP:
    - STDP creates eligibility traces from spike timing
    - Dopamine gates what enters working memory and what gets learned
    - High DA → update WM and learn new associations
    - Low DA → maintain WM and protect existing patterns
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
    stdp_lr: float = 0.02  # STDP learning rate for spike-based version

    # STDP parameters
    stdp_tau_ms: float = 20.0  # STDP trace decay
    stdp_a_plus: float = 0.01  # LTP amplitude
    stdp_a_minus: float = 0.012  # LTD amplitude (slightly larger for stability)
    heterosynaptic_ratio: float = 0.3  # LTD for non-active synapses

    # Recurrent connections for WM maintenance
    recurrent_strength: float = 0.8  # Self-excitation for persistence
    recurrent_inhibition: float = 0.2  # Lateral inhibition

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION (SFA)
    # =========================================================================
    # PFC pyramidal neurons show adaptation. This helps prevent runaway
    # activity during sustained working memory maintenance.
    pfc_adapt_increment: float = 0.2  # Adaptation per spike
    pfc_adapt_tau: float = 150.0      # Adaptation time constant (ms)

    # =========================================================================
    # SHORT-TERM PLASTICITY (STP) for recurrent connections
    # =========================================================================
    # PFC recurrent connections show SHORT-TERM DEPRESSION, preventing
    # frozen attractors. This allows working memory to be updated.
    stp_recurrent_enabled: bool = True

    # Weight constraints
    soft_bounds: bool = True

    # Synaptic scaling for homeostasis
    synaptic_scaling_enabled: bool = True
    synaptic_scaling_target: float = 0.4
    synaptic_scaling_rate: float = 0.001

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


@dataclass
class PrefrontalState(RegionState):
    """State for prefrontal cortex region."""
    # Working memory contents
    working_memory: Optional[torch.Tensor] = None

    # Gate state
    update_gate: Optional[torch.Tensor] = None

    # Rule representation
    active_rule: Optional[torch.Tensor] = None

    # Dopamine level (override base)
    dopamine: float = 0.2  # Start at baseline


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

    def reset_state(self):
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
    author="Thalia Project"
)
class Prefrontal(NeuralComponent):
    """Prefrontal cortex with dopamine-gated working memory.

    Implements:
    - Working memory maintenance via recurrent connections
    - Dopamine gating of updates (similar to LSTM gates)
    - Rule learning and context-dependent behavior
    - Slow integration for temporal abstraction

    Inherited from NeuralComponent:
    -------------------------------
    From LearningStrategyMixin (via NeuralComponent):
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

    def __init__(self, config: PrefrontalConfig):
        """
        Initialize prefrontal cortex.

        Args:
            config: PFC configuration
        """
        # Store config before calling super().__init__ which calls abstract methods
        self.pfc_config = config
        super().__init__(config)

        # Recurrent weights for WM maintenance
        self.rec_weights = nn.Parameter(
            WeightInitializer.gaussian(
                n_output=config.n_output,
                n_input=config.n_output,
                mean=0.0,
                std=0.1,
                device=self.device
            )
        )
        # Initialize with some self-excitation
        with torch.no_grad():
            self.rec_weights.data += torch.eye(
                config.n_output, device=self.device
            ) * config.recurrent_strength

        # Lateral inhibition weights
        self.inhib_weights = nn.Parameter(
            torch.ones(config.n_output, config.n_output, device=self.device)
            * config.recurrent_inhibition
        )
        with torch.no_grad():
            self.inhib_weights.data.fill_diagonal_(0.0)

        # Dopamine gating system
        self.dopamine_system = DopamineGatingSystem(
            n_neurons=config.n_output,
            tau_ms=config.dopamine_tau_ms,
            baseline=config.dopamine_baseline,
            threshold=config.gate_threshold,
            device=config.device,
        )

        # Initialize learning strategy (STDP with dopamine gating)
        # Using LearningStrategyRegistry for pluggable learning strategies
        self.learning_strategy = LearningStrategyRegistry.create(
            "stdp",
            STDPConfig(
                learning_rate=config.stdp_lr,
                a_plus=config.stdp_a_plus,
                a_minus=config.stdp_a_minus,
                tau_plus=config.stdp_tau_ms,
                tau_minus=config.stdp_tau_ms,
                dt_ms=config.dt_ms,
                w_min=config.w_min,
                w_max=config.w_max,
                soft_bounds=config.soft_bounds,
            )
        )

        # Initialize working memory state (1D tensors, ADR-005)
        self.state = PrefrontalState(
            working_memory=torch.zeros(config.n_output, device=self.device),
            update_gate=torch.zeros(config.n_output, device=self.device),
            dopamine=config.dopamine_baseline,
        )

        # Initialize theta phase for modulation
        self._theta_phase: float = 0.0

        # =====================================================================
        # PHASE 3: HIERARCHICAL GOALS & TEMPORAL ABSTRACTION
        # =====================================================================
        # Initialize goal hierarchy (Phase 3)
        self.goal_manager: Optional["GoalHierarchyManager"] = None
        self.discounter: Optional["HyperbolicDiscounter"] = None

        if config.use_hierarchical_goals:
            from thalia.regions.prefrontal_hierarchy import (
                GoalHierarchyManager,
                GoalHierarchyConfig,
            )

            gh_config = config.goal_hierarchy_config or GoalHierarchyConfig()
            self.goal_manager = GoalHierarchyManager(gh_config)

            # Hyperbolic discounting
            if config.use_hyperbolic_discounting:
                from thalia.regions.prefrontal_hierarchy import (
                    HyperbolicDiscounter,
                    HyperbolicDiscountingConfig,
                )

                hd_config = config.hyperbolic_config or HyperbolicDiscountingConfig()
                self.discounter = HyperbolicDiscounter(hd_config)

    def _get_learning_rule(self) -> LearningRule:
        """PFC uses dopamine-gated STDP learning."""
        return LearningRule.HEBBIAN

    def _initialize_weights(self) -> torch.Tensor:
        """Initialize feedforward weights."""
        return nn.Parameter(
            WeightInitializer.xavier(
                n_output=self.pfc_config.n_output,
                n_input=self.pfc_config.n_input,
                gain=1.0,
                device=torch.device(self.pfc_config.device)
            )
        )

    def _create_neurons(self) -> ConductanceLIF:
        """Create conductance-based LIF neurons with slow dynamics and SFA."""
        cfg = self.pfc_config
        # Slower dynamics for temporal integration + spike-frequency adaptation
        neuron_config = ConductanceLIFConfig(
            g_L=0.02,  # Slower leak (τ_m ≈ 50ms with C_m=1.0)
            tau_E=10.0,  # Slower excitatory (for integration)
            tau_I=15.0,  # Slower inhibitory
            adapt_increment=cfg.pfc_adapt_increment,  # SFA enabled!
            tau_adapt=cfg.pfc_adapt_tau,
        )
        neurons = ConductanceLIF(cfg.n_output, neuron_config)

        # =====================================================================
        # SHORT-TERM PLASTICITY for recurrent connections
        # =====================================================================
        # PFC recurrent connections show SHORT-TERM DEPRESSION, preventing
        # frozen attractors. This allows working memory to be updated.
        if cfg.stp_recurrent_enabled:
            device = torch.device(cfg.device)
            self.stp_recurrent = ShortTermPlasticity(
                n_pre=cfg.n_output,
                n_post=cfg.n_output,
                config=STPConfig.from_type(STPType.DEPRESSING, dt=cfg.dt_ms),
                per_synapse=True,
            )
            self.stp_recurrent.to(device)
        else:
            self.stp_recurrent = None

        return neurons

    def reset_state(self) -> None:
        """Reset state for new episode."""
        # Don't call super().reset_state() because it creates RegionState
        # Instead, create PrefrontalState directly with proper tensor shapes
        self.state = PrefrontalState(
            working_memory=torch.zeros(self.config.n_output, device=self.device),
            update_gate=torch.zeros(self.config.n_output, device=self.device),
            active_rule=None,  # Optional, can be None
            dopamine=0.2,  # Baseline
        )

        # Reset subsystems using helper
        self._reset_subsystems('neurons', 'dopamine_system', 'stp_recurrent')

    def add_neurons(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Add neurons to prefrontal cortex.

        Expands working memory capacity by adding neurons.

        Args:
            n_new: Number of neurons to add
            initialization: Weight initialization strategy
            sparsity: Sparsity for new connections
        """
        from dataclasses import replace

        old_n_output = self.config.n_output
        new_n_output = old_n_output + n_new

        # =====================================================================
        # 1. EXPAND WEIGHTS using base helper
        # =====================================================================
        self.weights = self._expand_weights(
            current_weights=self.weights,
            n_new=n_new,
            initialization=initialization,
            sparsity=sparsity,
            scale=1.0,  # Default scale for PFC
        )

        # =====================================================================
        # 2. UPDATE CONFIG
        # =====================================================================
        self.config = replace(self.config, n_output=new_n_output)
        if hasattr(self, 'pfc_config'):
            self.pfc_config = replace(self.pfc_config, n_output=new_n_output)

        # =====================================================================
        # 3. EXPAND NEURON POPULATION using base helper
        # =====================================================================
        self.neurons = self._recreate_neurons_with_state(
            neuron_factory=self._create_neurons,
            old_n_output=old_n_output,
        )

        # =====================================================================
        # 4. RESET STATE AND LEARNING
        # =====================================================================
        # Reset learning strategy state (traces, eligibility)
        if hasattr(self, 'learning_strategy') and self.learning_strategy is not None:
            self.learning_strategy.reset_state()

        # All state tensors are 1D [n_output] (ADR-005: No Batch Dimension)
        self.state = PrefrontalState(
            membrane=torch.zeros(self.config.n_output, device=self.device),
            spikes=torch.zeros(self.config.n_output, dtype=torch.bool, device=self.device),
            working_memory=torch.zeros(self.config.n_output, device=self.device),
            update_gate=torch.ones(self.config.n_output, device=self.device),
            dopamine=self.pfc_config.dopamine_baseline,
        )

    def forward(
        self,
        input_spikes: torch.Tensor,
        dopamine_signal: float = 0.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Process input through prefrontal cortex.

        Args:
            input_spikes: Input spike pattern [n_input] (1D bool tensor, ADR-005)
            dopamine_signal: External DA signal for gating (-1 to 1)
            **kwargs: Additional inputs

        Returns:
            Output spikes [n_output] (1D bool tensor, ADR-005)

        Note:
            Theta modulation and timestep (dt_ms) computed internally from config
        """
        # Get timestep from config for temporal dynamics
        dt = self.config.dt_ms

        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        assert input_spikes.dim() == 1, (
            f"PrefrontalCortex.forward: input_spikes must be 1D [n_input], "
            f"got shape {input_spikes.shape}. See ADR-005: No Batch Dimension."
        )
        assert input_spikes.shape[0] == self.pfc_config.n_input, (
            f"PrefrontalCortex.forward: input_spikes has shape {input_spikes.shape} "
            f"but n_input={self.pfc_config.n_input}. Check that input matches PFC config."
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
        encoding_mod = (1 + torch.cos(torch.tensor(self._theta_phase, device=self.device))) / 2
        retrieval_mod = (1 - torch.cos(torch.tensor(self._theta_phase, device=self.device))) / 2

        # Encoding phase (theta trough): gate new info into WM
        # Retrieval phase (theta peak): maintain WM and boost recurrence
        ff_gain = 0.5 + 0.5 * encoding_mod  # 0.5-1.0: boost input during encoding
        rec_gain = 0.5 + 0.5 * retrieval_mod  # 0.5-1.0: boost recurrence during retrieval

        # Feedforward input - modulated by encoding phase
        # 1D matmul: weights[n_output, n_input] @ input[n_input] → [n_output]
        ff_input = (self.weights @ input_spikes.float()) * ff_gain

        # =====================================================================
        # NOREPINEPHRINE GAIN MODULATION (Locus Coeruleus)
        # =====================================================================
        # High NE (arousal/uncertainty): Increase gain → more responsive WM
        # Low NE (baseline): Normal gain
        # Biological: β-adrenergic receptors modulate PFC excitability and
        # working memory flexibility (Arnsten 2009)
        ne_level = self.state.norepinephrine
        # NE gain: 1.0 (baseline) to 1.5 (high arousal)
        ne_gain = 1.0 + NE_GAIN_RANGE * ne_level
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
            wm = self.state.working_memory.float() if self.state.working_memory is not None else torch.zeros(self.pfc_config.n_output, device=input_spikes.device)
            rec_input = (self.rec_weights @ wm) * rec_gain

        # Lateral inhibition: inhib_weights[n_output, n_output] @ wm[n_output] → [n_output]
        wm = self.state.working_memory.float() if self.state.working_memory is not None else torch.zeros(self.pfc_config.n_output, device=input_spikes.device)
        inhib = self.inhib_weights @ wm

        # Total excitation and inhibition
        g_exc = (ff_input + rec_input).clamp(min=0)
        g_inh = inhib.clamp(min=0)

        # Run through neurons (returns 1D bool spikes)
        output_spikes, _ = self.neurons(g_exc, g_inh)

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
        noise = torch.randn_like(new_wm) * self.pfc_config.wm_noise_std
        self.state.working_memory = (new_wm + noise).clamp(min=0, max=1)

        # Store state
        self.state.spikes = output_spikes

        # Output shape check
        assert output_spikes.shape == (self.pfc_config.n_output,), (
            f"PrefrontalCortex.forward: output_spikes has shape {output_spikes.shape} "
            f"but expected ({self.pfc_config.n_output},). "
            f"Check PFC neuron or weight configuration."
        )
        assert output_spikes.dtype == torch.bool, (
            f"PrefrontalCortex.forward: output_spikes must be bool (ADR-004), "
            f"got {output_spikes.dtype}"
        )

        # Apply continuous plasticity (learning happens as part of forward dynamics)
        self._apply_plasticity(input_spikes, output_spikes)

        # Apply axonal delay (biological reality: ALL neural connections have delays)
        delayed_spikes = self._apply_axonal_delay(output_spikes, dt)

        return delayed_spikes

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
            weights=self.weights,
        )

        # Optional: Apply synaptic scaling for homeostasis
        if cfg.synaptic_scaling_enabled and metrics:
            with torch.no_grad():
                mean_weight = self.weights.data.mean()
                scaling = 1.0 + cfg.synaptic_scaling_rate * (cfg.synaptic_scaling_target - mean_weight)
                self.weights.data *= scaling
                clamp_weights(self.weights.data, self.config.w_min, self.config.w_max)

        # ======================================================================
        # Update recurrent weights to strengthen WM patterns
        # ======================================================================
        # Rule learning now happens via dopamine-modulated STDP in _apply_plasticity
        # This simple Hebbian update for recurrent connections maintains WM patterns
        if self.state.working_memory is not None:
            # working_memory is already 1D [n_output] (ADR-005)
            wm = self.state.working_memory  # [n_output]
            dW_rec = cfg.rule_lr * torch.outer(wm, wm)  # [n_output, n_output]
            with torch.no_grad():
                self.rec_weights.data += dW_rec
                self.rec_weights.data.fill_diagonal_(
                    cfg.recurrent_strength
                )  # Maintain self-excitation
                self.rec_weights.data.clamp_(0.0, 1.0)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics using DiagnosticsMixin helpers.

        Reports working memory state, gating, and weight statistics.
        """
        cfg = self.pfc_config

        # Custom metrics specific to PFC
        custom = {
            "n_output": cfg.n_output,
            "gate_mean": self.state.update_gate.mean().item() if self.state.update_gate is not None else 0.0,
            "gate_std": self.state.update_gate.std().item() if self.state.update_gate is not None else 0.0,
            "wm_mean": self.state.working_memory.mean().item() if self.state.working_memory is not None else 0.0,
            "wm_std": self.state.working_memory.std().item() if self.state.working_memory is not None else 0.0,
            "wm_active": (self.state.working_memory > 0.1).sum().item() if self.state.working_memory is not None else 0,
            "dopamine_level": self.state.dopamine,
            "config_w_min": cfg.w_min,
            "config_w_max": cfg.w_max,
            "config_rec_w_min": cfg.recurrent_w_min,
            "config_rec_w_max": cfg.recurrent_w_max,
        }

        # Use collect_standard_diagnostics for weight and spike statistics
        return self.collect_standard_diagnostics(
            region_name="prefrontal",
            weight_matrices={
                "feedforward": self.weights.data,
                "recurrent": self.rec_weights.data,
                "inhibition": self.inhib_weights.data,
            },
            spike_tensors={
                "output": self.state.spikes,
            },
            custom_metrics=custom,
        )

    def set_context(self, context: torch.Tensor) -> None:
        """
        Set the current context/rule in working memory.

        This allows explicit control of PFC state for rule-based tasks.

        Args:
            context: Context pattern [n_output] (1D tensor, ADR-005)
        """
        assert context.dim() == 1, (
            f"set_context: context must be 1D [n_output], got shape {context.shape}"
        )
        assert context.shape[0] == self.config.n_output, (
            f"set_context: context has {context.shape[0]} elements but expected {self.config.n_output}"
        )

        self.state.working_memory = context.to(self.device).float()
        self.state.active_rule = context.to(self.device).float()

    def debug_get_working_memory(self) -> torch.Tensor:
        """Get current working memory contents for debugging/testing (1D tensor, ADR-005).

        ⚠️ DEBUG/TEST ONLY: Do not call this from other brain regions!

        Working memory IS the goal context. In biology, PFC → Striatum
        projections carry goal information via spike patterns through the
        pfc_to_striatum pathway, not as a separate signal.

        The working memory spikes flow through the pathway with proper
        axonal delays and are extracted by striatum from its concatenated input.

        This method is only for:
        - Unit tests that need to inspect WM state
        - Task evaluation (external to brain)
        - Debugging and diagnostics
        """
        if self.state.working_memory is None:
            return torch.zeros(self.config.n_output, device=self.device)
        return self.state.working_memory

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
            noise = torch.randn_like(prediction) * self.pfc_config.wm_noise_std
            prediction = prediction + noise

        return prediction

    def maintain(self, n_steps: int = 10) -> Dict[str, Any]:
        """
        Run maintenance steps without external input.

        Useful for testing WM persistence.

        Args:
            n_steps: Number of maintenance steps

        Returns:
            Metrics about maintenance

        Note:
            Timestep (dt_ms) is obtained from self.config
        """
        if self.state.working_memory is None:
            return {"error": "No working memory to maintain"}

        initial_wm = self.state.working_memory.clone()

        for _ in range(n_steps):
            # No external input, low DA (maintenance mode)
            null_input = torch.zeros(self.config.n_input, device=self.device)
            self.forward(null_input, dopamine_signal=-0.3)

        final_wm = self.state.working_memory

        # Compute retention using safe cosine similarity
        retention = cosine_similarity_safe(
            initial_wm.flatten(), final_wm.flatten()
        ).item()

        return {
            "n_steps": n_steps,
            "retention": retention,
            "initial_activity": initial_wm.mean().item(),
            "final_activity": final_wm.mean().item(),
        }

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
            raise ValueError("Hierarchical goals not enabled. Set use_hierarchical_goals=True in config.")
        self.goal_manager.set_root_goal(root_goal)

    def debug_get_current_goal(self) -> Optional["Goal"]:
        """
        Get currently active goal from hierarchy (for debugging/inspection).

        ⚠️ DEBUG/TEST ONLY: For inspection and diagnostics.

        Phase 3 functionality: Returns the goal at the top of the active stack.

        Returns:
            Current goal or None if no active goals

        Example:
            goal = pfc.debug_get_current_goal()
            if goal is not None:
                print(f"Working on: {goal.name}")
        """
        if self.goal_manager is None:
            return None
        return self.goal_manager.get_current_goal()

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

    def debug_get_goal_manager(self) -> Optional["GoalHierarchyManager"]:
        """
        Get goal hierarchy manager for debugging/inspection.

        ⚠️ DEBUG/TEST ONLY: For inspection and diagnostics.

        Phase 3 functionality: Provides access to goal manager internals.

        Returns:
            GoalHierarchyManager or None if not enabled
        """
        return self.goal_manager

    def debug_get_discounter(self) -> Optional["HyperbolicDiscounter"]:
        """
        Get hyperbolic discounter for debugging/inspection.

        ⚠️ DEBUG/TEST ONLY: For inspection and diagnostics.

        Phase 3 functionality: Provides access to discounter internals.

        Returns:
            HyperbolicDiscounter or None if not enabled
        """
        return self.discounter

    def get_state(self) -> PrefrontalState:
        """Get current state."""
        return self.state

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for checkpointing.

        Returns state dictionary with keys:
        - weights: Feedforward, recurrent, and inhibition weights
        - region_state: Neuron state, working memory, spikes
        - learning_state: STDP eligibility traces, STP state
        - neuromodulator_state: Dopamine gating state
        - config: Configuration for validation
        """
        state_dict = {
            "weights": {
                "feedforward": self.weights.data.clone(),
                "recurrent": self.rec_weights.data.clone(),
                "inhibition": self.inhib_weights.data.clone(),
            },
            "region_state": {
                "neurons": self.neurons.get_state(),
                "working_memory": self.state.working_memory.clone() if self.state.working_memory is not None else None,
                "update_gate": self.state.update_gate.clone() if self.state.update_gate is not None else None,
                "spikes": self.state.spikes.clone() if self.state.spikes is not None else None,
                "membrane": self.state.membrane.clone() if self.state.membrane is not None else None,
                "active_rule": self.state.active_rule.clone() if self.state.active_rule is not None else None,
            },
            "learning_state": {},
            "neuromodulator_state": {
                "dopamine": self.state.dopamine,
                "dopamine_system": self.dopamine_system.get_state(),
            },
            "config": {
                "n_input": self.config.n_input,
                "n_output": self.config.n_output,
            },
        }

        # STDP eligibility traces (if learning strategy has state)
        if hasattr(self, 'learning_strategy') and self.learning_strategy is not None:
            if hasattr(self.learning_strategy, 'get_state'):
                state_dict["learning_state"]["stdp_strategy"] = self.learning_strategy.get_state()

        # STP state
        if self.stp_recurrent is not None:
            state_dict["learning_state"]["stp_recurrent"] = self.stp_recurrent.get_state()

        return state_dict

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Load complete state from checkpoint.

        Args:
            state: State dictionary from get_full_state()

        Raises:
            ValueError: If config dimensions don't match
        """
        # Validate config compatibility
        config = state.get("config", {})
        if config.get("n_input") != self.config.n_input:
            raise ValueError(f"Config mismatch: n_input {config.get('n_input')} != {self.config.n_input}")
        if config.get("n_output") != self.config.n_output:
            raise ValueError(f"Config mismatch: n_output {config.get('n_output')} != {self.config.n_output}")

        # Restore weights
        weights = state["weights"]
        self.weights.data.copy_(weights["feedforward"].to(self.device))
        self.rec_weights.data.copy_(weights["recurrent"].to(self.device))
        self.inhib_weights.data.copy_(weights["inhibition"].to(self.device))

        # Restore neuron state
        region_state = state["region_state"]
        self.neurons.load_state(region_state["neurons"])

        # Restore working memory and gating
        if region_state["working_memory"] is not None:
            self.state.working_memory = region_state["working_memory"].to(self.device)
        if region_state["update_gate"] is not None:
            self.state.update_gate = region_state["update_gate"].to(self.device)
        if region_state["spikes"] is not None:
            self.state.spikes = region_state["spikes"].to(self.device)
        if region_state["membrane"] is not None:
            self.state.membrane = region_state["membrane"].to(self.device)
        if region_state["active_rule"] is not None:
            self.state.active_rule = region_state["active_rule"].to(self.device)

        # Restore learning state
        learning_state = state["learning_state"]
        if "stdp_strategy" in learning_state and hasattr(self, 'learning_strategy'):
            if hasattr(self.learning_strategy, 'load_state'):
                self.learning_strategy.load_state(learning_state["stdp_strategy"])

        if "stp_recurrent" in learning_state and self.stp_recurrent is not None:
            self.stp_recurrent.load_state(learning_state["stp_recurrent"])

        # Restore dopamine gating system
        neuromod = state["neuromodulator_state"]
        self.state.dopamine = neuromod["dopamine"]
        self.dopamine_system.load_state(neuromod["dopamine_system"])
