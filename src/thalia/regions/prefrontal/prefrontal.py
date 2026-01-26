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

Biological Basis:
=================
- Layer 2/3 recurrent circuits for WM maintenance
- D1/D2 receptors modulate gain and gating
- Strong connections with striatum (for action selection)
- Connections with hippocampus (for episodic retrieval)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from thalia.components.neurons import ConductanceLIF, ConductanceLIFConfig
from thalia.components.synapses import ShortTermPlasticity, STPConfig, STPType, WeightInitializer
from thalia.config.region_configs import PrefrontalConfig
from thalia.constants.neuromodulation import DA_BASELINE_STANDARD, compute_ne_gain
from thalia.constants.oscillator import (
    PFC_FEEDFORWARD_GAIN_MIN,
    PFC_FEEDFORWARD_GAIN_RANGE,
    PFC_RECURRENT_GAIN_MIN,
    PFC_RECURRENT_GAIN_RANGE,
)
from thalia.core.diagnostics_schema import DiagnosticsDict
from thalia.core.errors import ConfigurationError
from thalia.core.neural_region import NeuralRegion
from thalia.learning import LearningStrategyRegistry, STDPConfig
from thalia.learning.homeostasis.synaptic_homeostasis import (
    UnifiedHomeostasis,
    UnifiedHomeostasisConfig,
)
from thalia.managers.component_registry import register_region
from thalia.typing import StateDict
from thalia.utils.input_routing import InputRouter
from thalia.utils.oscillator_utils import (
    compute_oscillator_modulated_gain,
    compute_theta_encoding_retrieval,
)

from .checkpoint_manager import PrefrontalCheckpointManager
from .goal_emergence import EmergentGoalSystem
from .state import PrefrontalState


def sample_heterogeneous_wm_neurons(
    n_neurons: int,
    stability_cv: float = 0.3,
    tau_mem_min: float = 100.0,
    tau_mem_max: float = 500.0,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> StateDict:
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

    def update(self, signal: float, dt_ms: float = 1.0) -> float:
        """Update dopamine level with new signal.

        Args:
            signal: External dopamine signal (-1 to 1)
            dt_ms: Timestep in ms

        Returns:
            Current dopamine level
        """
        # Decay toward baseline
        decay = torch.exp(torch.tensor(-dt_ms / self.tau_ms)).item()
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
        gate_value: float = 1.0 / (
            1.0 + torch.exp(torch.tensor(-10 * (self.level - self.threshold))).item()
        )
        return gate_value

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

    Inherited from NeuralRegion:
    ----------------------------------
    From LearningStrategyMixin (via NeuralRegion):
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
        self.input_size = sizes["input_size"]
        self.n_neurons = sizes["n_neurons"]
        self.n_output = self.n_neurons  # PFC output = neuron count
        self.total_neurons = self.n_neurons

        # =====================================================================
        # HETEROGENEOUS WORKING MEMORY NEURONS (Phase 1B)
        # =====================================================================
        self._recurrent_strength: Optional[torch.Tensor] = None
        self._tau_mem_heterogeneous: Optional[torch.Tensor] = None
        self._neuron_type: Optional[torch.Tensor] = None

        # Sample heterogeneous recurrent strengths and time constants
        # mimicking biological diversity in WM stability across neurons
        if config.use_heterogeneous_wm:
            wm_properties = sample_heterogeneous_wm_neurons(
                n_neurons=self.n_neurons,
                stability_cv=config.stability_cv,
                tau_mem_min=config.tau_mem_min,
                tau_mem_max=config.tau_mem_max,
                device=self.device,  # type: ignore[arg-type]
                seed=None,  # Use random seed for variability
            )
            self._recurrent_strength = wm_properties["recurrent_strength"]
            self._tau_mem_heterogeneous = wm_properties["tau_mem"]
            self._neuron_type = wm_properties["neuron_type"]  # 0=flexible, 1=stable

        # =====================================================================
        # D1/D2 RECEPTOR SUBTYPES (Phase 1B)
        # =====================================================================
        self._d1_neurons: Optional[torch.Tensor] = None
        self._d2_neurons: Optional[torch.Tensor] = None

        # Split neurons into D1-dominant (excitatory DA response) and
        # D2-dominant (inhibitory DA response) populations
        if config.use_d1_d2_subtypes:
            n_d1 = int(self.n_neurons * config.d1_fraction)
            self._d1_neurons = torch.arange(n_d1, device=self.device)
            self._d2_neurons = torch.arange(n_d1, self.n_neurons, device=self.device)

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
            default_learning_strategy="stdp",
            device=device,
        )

        # Override neurons to add STP (NeuralRegion creates basic neurons)
        self.neurons = self._create_neurons()

        # Learning control (specific to prefrontal cortex)
        self.plasticity_enabled: bool = True

        # Register feedforward input source and initialize weights
        self.add_input_source("default", n_input=self.input_size)
        # Initialize with Xavier (better than NeuralRegion's default)
        self.synaptic_weights["default"].data = WeightInitializer.xavier(
            n_output=self.n_neurons, n_input=self.input_size, gain=1.0, device=self.device
        )

        # Recurrent weights for WM maintenance
        self.rec_weights = nn.Parameter(
            WeightInitializer.gaussian(
                n_output=self.n_neurons,
                n_input=self.n_neurons,
                mean=0.0,
                std=0.1,
                device=self.device,
            ),
            requires_grad=False,
        )
        # Initialize with self-excitation (heterogeneous if enabled)
        if config.use_heterogeneous_wm:
            # Scale diagonal by heterogeneous recurrent strengths
            diag_strength = torch.diag(self._recurrent_strength)
            self.rec_weights.data += diag_strength
        else:
            # Uniform self-excitation
            self.rec_weights.data += (
                torch.eye(self.n_neurons, device=self.device) * config.recurrent_strength
            )

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
        self.learning_strategy = LearningStrategyRegistry.create(  # type: ignore[assignment]
            "stdp",
            STDPConfig(
                learning_rate=config.learning_rate,
                a_plus=config.a_plus,
                a_minus=config.a_minus,
                tau_plus=config.tau_plus_ms,
                tau_minus=config.tau_minus_ms,
                w_min=config.w_min,
                w_max=config.w_max,
            ),
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
        self.state: PrefrontalState = PrefrontalState(  # type: ignore[assignment]
            working_memory=torch.zeros(self.n_neurons, device=self.device),
            update_gate=torch.zeros(self.n_neurons, device=self.device),
            dopamine=config.dopamine_baseline,
        )

        # Initialize theta phase for modulation
        self._theta_phase: float = 0.0

        # Port-based routing: Register output ports
        self.register_output_port("executive", self.n_neurons)  # Main executive control output

        # Move all components to target device
        self.to(self.device)

        # =====================================================================
        # EMERGENT HIERARCHICAL GOALS (Biologically Plausible)
        # =====================================================================
        # Split neurons into abstract (rostral PFC) and concrete (caudal PFC)
        # This implements the biological rostral-caudal hierarchy
        n_abstract = int(self.n_neurons * 0.3)  # 30% abstract (long tau, slow)
        n_concrete = self.n_neurons - n_abstract  # 70% concrete (short tau, fast)

        # Goals emerge from WM patterns - no symbolic Goal objects
        self.emergent_goals = EmergentGoalSystem(
            n_wm_neurons=self.n_neurons,
            n_abstract=n_abstract,
            n_concrete=n_concrete,
            device=str(self.device),
        )

    def _create_neurons(self) -> ConductanceLIF:
        """Create conductance-based LIF neurons with slow dynamics and SFA.

        PFC neurons have significantly different dynamics than standard pyramidal neurons:
        - Much slower leak (τ_m ≈ 50ms vs 20ms) for temporal integration
        - Slower synaptic time constants for sustained integration
        - Spike-frequency adaptation for stable working memory

        If heterogeneous WM is enabled, neurons will have varying membrane time constants
        (tau_mem) to create populations of stable vs flexible neurons.
        """
        cfg = self.pfc_config
        # Custom config for PFC-specific slow dynamics
        neuron_config = ConductanceLIFConfig(
            g_L=0.02,  # Default leak (will be overridden if heterogeneous)
            tau_E=10.0,  # Slower excitatory (for integration)
            tau_I=15.0,  # Slower inhibitory
            adapt_increment=cfg.adapt_increment,  # SFA enabled!
            tau_adapt=cfg.adapt_tau,
        )
        neurons = ConductanceLIF(self.n_neurons, neuron_config, device=self.device)

        # =====================================================================
        # APPLY HETEROGENEOUS MEMBRANE TIME CONSTANTS (Phase 1B)
        # =====================================================================
        if self._tau_mem_heterogeneous is not None:
            # Convert tau_mem to g_L: tau_mem = C_m / g_L  =>  g_L = C_m / tau_mem
            # C_m = 1.0 (default), so g_L = 1 / tau_mem_ms
            C_m = neurons.C_m.item()

            # Compute per-neuron leak conductance from heterogeneous tau_mem
            # tau_mem in ms, g_L dimensionless (leak per timestep)
            g_L_heterogeneous = C_m / self._tau_mem_heterogeneous

            # Replace scalar g_L buffer with per-neuron tensor
            # Remove the old buffer and register new per-neuron buffer
            delattr(neurons, "g_L")
            neurons.register_buffer("g_L", g_L_heterogeneous.to(self.device))

        # =====================================================================
        # SHORT-TERM PLASTICITY for feedforward connections
        # =====================================================================
        # PFC feedforward connections show SHORT-TERM FACILITATION/DEPRESSION
        # for temporal filtering and gain control during encoding.
        self.stp_feedforward = ShortTermPlasticity(
            n_pre=self.input_size,
            n_post=self.n_neurons,
            config=STPConfig.from_type(STPType.FACILITATING),
            per_synapse=True,
        )
        self.stp_feedforward.to(self.device)

        # =====================================================================
        # SHORT-TERM PLASTICITY for recurrent connections
        # =====================================================================
        # PFC recurrent connections show SHORT-TERM DEPRESSION, preventing
        # frozen attractors. This allows working memory to be updated.
        self.stp_recurrent = ShortTermPlasticity(
            n_pre=self.n_neurons,
            n_post=self.n_neurons,
            config=STPConfig.from_type(STPType.DEPRESSING),
            per_synapse=True,
        )
        self.stp_recurrent.to(self.device)

        # =====================================================================
        # Registration: Opt-in auto-growth for STP modules

        # Feedforward STP (input -> n_output): grows during grow_source (pre) and grow_output (post)
        self._register_stp("stp_feedforward", direction="both", recurrent=False)

        # Recurrent STP (n_output -> n_output): ONLY grows during grow_output (both pre and post)
        # NOT during grow_source - recurrent connections track n_output, not n_input
        self._register_stp("stp_recurrent", direction="post", recurrent=True)

        return neurons

    def _reset_subsystems(self, *names: str) -> None:
        """Reset state of named subsystems that have reset_state() method."""
        for name in names:
            if hasattr(self, name):
                subsystem = getattr(self, name)
                if subsystem is not None and hasattr(subsystem, "reset_state"):
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
        self._reset_subsystems("neurons", "dopamine_system", "stp_recurrent", "stp_feedforward")

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        dopamine_signal: float = 0.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Process input through prefrontal cortex.

        Args:
            inputs: Input spikes - Dict mapping source names to spike tensors [n_input]
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
        dt_ms = self.config.dt_ms

        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        assert input_spikes.dim() == 1, (  # type: ignore[union-attr]
            f"PrefrontalCortex.forward: input_spikes must be 1D [n_input], "
            f"got shape {input_spikes.shape}. See ADR-005: No Batch Dimension."  # type: ignore[union-attr]
        )
        assert input_spikes.shape[0] == self.input_size, (  # type: ignore[union-attr]
            f"PrefrontalCortex.forward: input_spikes has shape {input_spikes.shape} "  # type: ignore[union-attr]
            f"but input_size={self.input_size}. Check that input matches PFC config."
        )

        # Ensure state is initialized
        if self.state.working_memory is None:
            self.reset_state()

        # Update dopamine and get gate value
        da_level = self.dopamine_system.update(dopamine_signal, dt_ms)
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
        ff_gain = compute_oscillator_modulated_gain(
            PFC_FEEDFORWARD_GAIN_MIN, PFC_FEEDFORWARD_GAIN_RANGE, encoding_mod
        )
        rec_gain = compute_oscillator_modulated_gain(
            PFC_RECURRENT_GAIN_MIN, PFC_RECURRENT_GAIN_RANGE, retrieval_mod
        )

        # Feedforward input - modulated by encoding phase
        # Apply STP if enabled (temporal filtering and gain control)
        if hasattr(self, "stp_feedforward") and self.stp_feedforward is not None:
            # Apply STP to feedforward connections (1D → 2D per-synapse efficacy)
            # stp_efficacy has shape [n_output, n_input] - per-synapse modulation
            stp_efficacy = self.stp_feedforward(input_spikes.float())  # type: ignore[union-attr]
            # Effective weights: element-wise multiply with STP efficacy
            effective_ff_weights = self.synaptic_weights["default"] * stp_efficacy.t()
            # Apply synaptic weights: weights[n_output, n_input] @ input[n_input] → [n_output]
            ff_input = (effective_ff_weights @ input_spikes.float()) * ff_gain  # type: ignore[union-attr]
        else:
            # No STP: direct feedforward
            # Apply synaptic weights: weights[n_output, n_input] @ input[n_input] → [n_output]
            ff_input = (self.synaptic_weights["default"] @ input_spikes.float()) * ff_gain  # type: ignore[union-attr]

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
        if (
            hasattr(self, "stp_recurrent")
            and self.stp_recurrent is not None
            and self.state.working_memory is not None
        ):
            # Apply STP to recurrent connections (1D → 2D per-synapse efficacy)
            # stp_efficacy has shape [n_output, n_output] - per-synapse modulation
            stp_efficacy = self.stp_recurrent(self.state.working_memory.float())
            # Effective weights: element-wise multiply rec_weights with STP efficacy
            # rec_weights is [n_output, n_output], stp_efficacy is [n_output, n_output]
            effective_rec_weights = self.rec_weights * stp_efficacy.t()
            # Recurrent: weights[n_output, n_output] @ wm[n_output] → [n_output]
            rec_input = (effective_rec_weights @ self.state.working_memory.float()) * rec_gain
        else:
            # Recurrent input from working memory - modulated by retrieval phase
            # rec_weights[n_output, n_output] @ wm[n_output] → [n_output]
            wm = (
                self.state.working_memory.float()
                if self.state.working_memory is not None
                else torch.zeros(self.n_neurons, device=input_spikes.device)  # type: ignore[union-attr]
            )
            rec_input = (self.rec_weights @ wm) * rec_gain

        # Lateral inhibition: inhib_weights[n_output, n_output] @ wm[n_output] → [n_output]
        wm = (
            self.state.working_memory.float()
            if self.state.working_memory is not None
            else torch.zeros(self.n_neurons, device=input_spikes.device)  # type: ignore[union-attr]
        )
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
        gate_tensor = torch.full_like(self.state.working_memory, gate)  # type: ignore[arg-type]
        self.state.update_gate = gate_tensor

        # WM decay
        decay = torch.exp(torch.tensor(-dt_ms / self.pfc_config.wm_decay_tau_ms))

        # Gated update: WM = gate * new_input + (1-gate) * decayed_old
        new_wm = (
            gate_tensor * output_spikes.float()  # type: ignore[operator]
            + (1 - gate_tensor) * self.state.working_memory * decay  # type: ignore[operator]
        )

        # Add noise for stochasticity
        noise = torch.randn_like(new_wm) * self.pfc_config.wm_noise_std
        self.state.working_memory = (new_wm + noise).clamp(min=0, max=1)

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
        self._apply_plasticity(input_spikes, output_spikes)  # type: ignore[arg-type]

        # =====================================================================
        # EMERGENT GOAL SYSTEM: Tag active WM patterns as goals
        # =====================================================================
        if self.emergent_goals is not None and self.state.working_memory is not None:
            # Tag currently active goal patterns (similar to hippocampal synaptic tagging)
            self.emergent_goals.update_goal_tags(self.state.working_memory)

        # Store output (NeuralRegion pattern)
        self.output_spikes = output_spikes

        # Port-based routing: Set port outputs
        self.clear_port_outputs()
        self.set_port_output("executive", output_spikes)

        return output_spikes  # type: ignore[no-any-return]

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
            self.rec_weights.data.fill_diagonal_(cfg.recurrent_strength)  # Maintain self-excitation
            self.rec_weights.data.clamp_(0.0, 1.0)

        # ======================================================================
        # EMERGENT GOAL LEARNING: Learn transitions and consolidate values
        # ======================================================================
        if self.emergent_goals is not None and self.state.working_memory is not None:
            # Extract abstract and concrete patterns from WM
            abstract_pattern = self.state.working_memory[self.emergent_goals.abstract_neurons]
            concrete_pattern = self.state.working_memory[self.emergent_goals.concrete_neurons]

            # Learn goal transitions via Hebbian association
            # When abstract pattern A is active and concrete pattern B follows,
            # strengthen the transition A→B (emergent goal decomposition)
            if abstract_pattern.sum() > 0.1 and concrete_pattern.sum() > 0.1:
                self.emergent_goals.learn_transition(
                    abstract_pattern,
                    concrete_pattern,
                    learning_rate=cfg.rule_lr,
                )

            # Consolidate valuable goal patterns with dopamine
            # High dopamine → strengthen value associations for tagged goals
            self.emergent_goals.consolidate_valuable_goals(
                dopamine=self.state.dopamine,
                learning_rate=cfg.rule_lr,
            )

    def grow_output(
        self,
        n_new: int,
        initialization: str = "sparse_random",
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
            sparsity=sparsity,
        )

        # 2. Expand rec_weights [n_output, n_output] → [n_output+n_new, n_output+n_new]
        # First add rows, then add columns
        expanded_rec = self._grow_weight_matrix_rows(
            self.rec_weights.data, n_new, initializer=initialization, sparsity=sparsity
        )
        self.rec_weights.data = self._grow_weight_matrix_cols(
            expanded_rec, n_new, initializer=initialization, sparsity=sparsity
        )
        # Add self-excitation for new neurons
        for i in range(n_new):
            self.rec_weights.data[old_n_output + i, old_n_output + i] = (
                self.pfc_config.recurrent_strength
            )

        # 3. Expand inhib_weights [n_output, n_output] → [n_output+n_new, n_output+n_new]
        new_inhib_rows = (
            torch.ones(n_new, old_n_output, device=self.device)
            * self.pfc_config.recurrent_inhibition
        )
        expanded_inhib = torch.cat([self.inhib_weights.data, new_inhib_rows], dim=0)
        new_inhib_cols = (
            torch.ones(new_n_output, n_new, device=self.device)
            * self.pfc_config.recurrent_inhibition
        )
        self.inhib_weights.data = torch.cat([expanded_inhib, new_inhib_cols], dim=1)
        # Zero diagonal (no self-inhibition)
        self.inhib_weights.data.fill_diagonal_(0.0)

        # 4. Expand neurons using efficient in-place growth
        self.neurons.grow_neurons(n_new)

        # 4.5. Track neurogenesis history for new neurons
        # Record creation timestep for checkpoint analysis
        new_birth_steps = torch.full(
            (n_new,), self._current_training_step, dtype=torch.long, device=self.device
        )
        self._neuron_birth_steps = torch.cat([self._neuron_birth_steps, new_birth_steps])

        # 5. Update dopamine gating system
        self.dopamine_system = DopamineGatingSystem(
            n_neurons=new_n_output,
            tau_ms=self.pfc_config.dopamine_tau_ms,
            baseline=self.pfc_config.dopamine_baseline,
            threshold=self.pfc_config.gate_threshold,
            device=self.device,  # type: ignore[arg-type]
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
        self._auto_grow_registered_components("output", n_new)

        # 5.7. Grow emergent goals system (if enabled)
        if self.emergent_goals is not None:
            # Expand goal tags, value weights
            new_tags = torch.zeros(n_new, device=self.device)
            self.emergent_goals.goal_tags = torch.cat([self.emergent_goals.goal_tags, new_tags])

            new_values = WeightInitializer.gaussian(
                n_output=n_new, n_input=1, mean=0.0, std=0.1, device=self.device
            ).squeeze()
            self.emergent_goals.value_weights = torch.cat(
                [self.emergent_goals.value_weights, new_values]
            )

            # Update n_wm_neurons
            self.emergent_goals.n_wm_neurons = new_n_output

            # Determine how to split new neurons between abstract/concrete
            # Maintain ~30% abstract, 70% concrete ratio
            old_abstract = self.emergent_goals.n_abstract
            old_concrete = self.emergent_goals.n_concrete
            target_abstract_ratio = 0.3

            new_abstract_total = int(new_n_output * target_abstract_ratio)
            new_concrete_total = new_n_output - new_abstract_total

            n_new_abstract = new_abstract_total - old_abstract
            n_new_concrete = new_concrete_total - old_concrete

            # Expand transition weights [n_concrete, n_abstract]
            if n_new_concrete > 0:
                # Add rows for new concrete neurons
                new_concrete_rows = torch.zeros(n_new_concrete, old_abstract, device=self.device)
                self.emergent_goals.transition_weights = torch.cat(
                    [self.emergent_goals.transition_weights, new_concrete_rows], dim=0
                )

            if n_new_abstract > 0:
                # Add columns for new abstract neurons
                current_n_concrete = self.emergent_goals.transition_weights.shape[0]
                new_abstract_cols = torch.zeros(
                    current_n_concrete, n_new_abstract, device=self.device
                )
                self.emergent_goals.transition_weights = torch.cat(
                    [self.emergent_goals.transition_weights, new_abstract_cols], dim=1
                )

            # Update neuron counts and indices
            self.emergent_goals.n_abstract = new_abstract_total
            self.emergent_goals.n_concrete = new_concrete_total
            self.emergent_goals.abstract_neurons = torch.arange(
                new_abstract_total, device=self.device
            )
            self.emergent_goals.concrete_neurons = torch.arange(
                new_abstract_total, new_n_output, device=self.device
            )

        # 6. Update instance variables
        self.n_neurons = new_n_output
        self.n_output = new_n_output
        self.total_neurons = new_n_output

        # 6.5. Update port sizes
        # Update registered port sizes to reflect new output size
        self._port_sizes["executive"] = self.n_neurons

        # 7. Validate growth completed correctly (skip config check - using n_neurons not n_output)
        self._validate_output_growth(old_n_output, n_new, check_config=False)

    def grow_neurons(
        self,
        n_new: int,
        initialization: str = "sparse_random",
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

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Propagates dt update to neurons, STP components, and learning strategies.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        # Update neurons
        if hasattr(self, "neurons") and hasattr(self.neurons, "update_temporal_parameters"):
            self.neurons.update_temporal_parameters(dt_ms)

        # Update STP components
        if hasattr(self, "stp_feedforward") and self.stp_feedforward is not None:
            self.stp_feedforward.update_temporal_parameters(dt_ms)
        if hasattr(self, "stp_recurrent") and self.stp_recurrent is not None:
            self.stp_recurrent.update_temporal_parameters(dt_ms)

        # Update learning strategies
        if hasattr(self, "strategies"):
            for strategy in self.strategies.values():
                if hasattr(strategy, "update_temporal_parameters"):
                    strategy.update_temporal_parameters(dt_ms)

    def get_diagnostics(self) -> DiagnosticsDict:
        """Get diagnostics using standard DiagnosticsDict format.

        Reports working memory state, gating, and weight statistics.
        """
        from thalia.core.diagnostics_schema import (
            compute_activity_metrics,
            compute_health_metrics,
            compute_plasticity_metrics,
        )

        cfg = self.pfc_config

        # Activity metrics
        activity = compute_activity_metrics(self.output_spikes, self.n_neurons)

        # Plasticity metrics (if learning enabled)
        plasticity = None
        if self.learning_enabled:
            # Average across all weight matrices
            all_weights = torch.cat(
                [
                    self.synaptic_weights["default"].flatten(),
                    self.rec_weights.flatten(),
                    self.inhib_weights.flatten(),
                ]
            )
            plasticity = compute_plasticity_metrics(
                all_weights, learning_rate=self.stdp.learning_rate
            )

        # Health metrics
        health = compute_health_metrics(
            state_tensors={
                "membrane": self.neurons.membrane,
                "ge": self.neurons.ge,
                "gi": self.neurons.gi,
                "working_memory": self.state.working_memory,
            },
            firing_rate=activity["firing_rate"],
        )

        # Neuromodulator metrics
        neuromodulators = {"dopamine": self.state.dopamine}

        # Region-specific metrics
        region_specific = {
            "gate_mean": (
                self.state.update_gate.mean().item() if self.state.update_gate is not None else 0.0
            ),
            "gate_std": (
                self.state.update_gate.std().item() if self.state.update_gate is not None else 0.0
            ),
            "wm_mean": (
                self.state.working_memory.mean().item()
                if self.state.working_memory is not None
                else 0.0
            ),
            "wm_std": (
                self.state.working_memory.std().item()
                if self.state.working_memory is not None
                else 0.0
            ),
            "wm_active": (
                (self.state.working_memory > 0.1).sum().item()
                if self.state.working_memory is not None
                else 0
            ),
            "config_w_min": cfg.w_min,
            "config_w_max": cfg.w_max,
        }

        return {
            "activity": activity,
            "plasticity": plasticity,
            "health": health,
            "neuromodulators": neuromodulators,
            "region_specific": region_specific,
        }

    # =========================================================================
    # EMERGENT GOAL SYSTEM (NEW - Biologically Plausible)
    # =========================================================================

    def get_current_goal_pattern(self) -> torch.Tensor:
        """Return current working memory pattern (= current goal).

        Goals ARE working memory patterns - no symbolic representation needed.

        Returns:
            Current WM pattern [n_neurons] representing active goal

        Example:
            goal_pattern = pfc.get_current_goal_pattern()
            # This IS the goal - a sustained activity pattern
        """
        if self.state.working_memory is None:
            return torch.zeros(self.n_neurons, device=self.device)
        return self.state.working_memory

    def predict_next_subgoal(self) -> torch.Tensor:
        """Predict next subgoal from current abstract WM pattern.

        Uses learned transitions (emergent goal decomposition), not explicit
        subgoal lists.

        Returns:
            Predicted concrete subgoal pattern [n_concrete]

        Example:
            # After training goal hierarchies via examples
            subgoal = pfc.predict_next_subgoal()
            # Inject into WM to activate subgoal
            pfc.state.working_memory[pfc.emergent_goals.concrete_neurons] = subgoal
        """
        assert self.emergent_goals is not None

        if self.state.working_memory is None:
            return torch.zeros(self.emergent_goals.n_concrete, device=self.device)

        # Extract abstract pattern from WM
        abstract_pattern = self.state.working_memory[self.emergent_goals.abstract_neurons]

        # Predict subgoal via learned transitions
        return self.emergent_goals.predict_subgoal(abstract_pattern)

    def get_goal_value(self, goal_pattern: torch.Tensor) -> float:
        """Get learned value for a goal pattern.

        Value is learned from experience with dopamine, not computed explicitly.

        Args:
            goal_pattern: Goal pattern to evaluate [n_neurons]

        Returns:
            Estimated value of goal pattern

        Raises:
            ConfigurationError: If emergent goals not enabled

        Example:
            pattern = pfc.get_current_goal_pattern()
            value = pfc.get_goal_value(pattern)
            print(f"Current goal value: {value:.2f}")
        """
        if self.emergent_goals is None:
            raise ConfigurationError("Emergent goals not enabled.")

        # Compute value as dot product with learned value weights
        return float(torch.sum(goal_pattern * self.emergent_goals.value_weights).item())

    def reset_goal_tags(self) -> None:
        """Reset synaptic tags for goal patterns.

        Useful for starting new episodes or tasks where previous goal
        history should not influence current goal selection.

        Raises:
            ConfigurationError: If emergent goals not enabled

        Example:
            # Start new episode
            pfc.reset_goal_tags()
            pfc.reset_state()
        """
        if self.emergent_goals is None:
            raise ConfigurationError("Emergent goals not enabled.")

        self.emergent_goals.reset_tags()

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
        if hasattr(self, "stp_feedforward") and self.stp_feedforward is not None:
            stp_feedforward_state = self.stp_feedforward.get_state()

        return PrefrontalState(
            working_memory=(
                self.state.working_memory.clone() if self.state.working_memory is not None else None
            ),
            update_gate=(
                self.state.update_gate.clone() if self.state.update_gate is not None else None
            ),
            active_rule=(
                self.state.active_rule.clone() if self.state.active_rule is not None else None
            ),
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
        # Restore PFC-specific state
        if state.working_memory is not None:
            self.state.working_memory = self._load_tensor(state.working_memory).clone()
        if state.update_gate is not None:
            self.state.update_gate = self._load_tensor(state.update_gate).clone()
        if state.active_rule is not None:
            self.state.active_rule = self._load_tensor(state.active_rule).clone()

        # Neuromodulators already restored by super().load_state() via _restore_neuromodulators()

        # Restore STP state
        if state.stp_recurrent_state is not None and self.stp_recurrent is not None:
            self.stp_recurrent.load_state(state.stp_recurrent_state)

        if (
            state.stp_feedforward_state is not None
            and hasattr(self, "stp_feedforward")
            and self.stp_feedforward is not None
        ):
            self.stp_feedforward.load_state(state.stp_feedforward_state)

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for checkpointing.

        Returns state dictionary with keys:
        - weights: Feedforward, recurrent, and inhibition weights
        - region_state: Neuron state, working memory, spikes
        - learning_state: STDP eligibility traces, STP state
        - neuromodulator_state: Dopamine gating state
        - emergent_goals_state: Emergent goal system state (if enabled)
        - config: Configuration for validation
        """
        state_obj = self.get_state()
        state = state_obj.to_dict()

        # Add all weights (required for checkpointing)
        # PFC has both synaptic_weights dict (feedforward) and rec_weights (recurrent)
        state["synaptic_weights"] = {
            name: weights.detach().clone() for name, weights in self.synaptic_weights.items()
        }
        if hasattr(self, "rec_weights"):
            state["rec_weights"] = self.rec_weights.detach().clone()

        # Save emergent goals state (if enabled)
        if self.emergent_goals is not None:
            state["emergent_goals_state"] = self.emergent_goals.get_state_dict()

        return state

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Load complete state from checkpoint.

        Args:
            state: State dictionary from get_full_state()
        """
        state_obj = PrefrontalState.from_dict(state, device=str(self.device))
        self.load_state(state_obj)

        # Restore synaptic weights
        if "synaptic_weights" in state:
            for name, weights in state["synaptic_weights"].items():
                if name in self.synaptic_weights:
                    self.synaptic_weights[name].data = weights.to(self.device)

        # Restore recurrent weights
        if "rec_weights" in state and hasattr(self, "rec_weights"):
            self.rec_weights.data = state["rec_weights"].to(self.device)

        # Restore emergent goals state (if enabled)
        if "emergent_goals_state" in state and self.emergent_goals is not None:
            self.emergent_goals.load_state_dict(state["emergent_goals_state"])
