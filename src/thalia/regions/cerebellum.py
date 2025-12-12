"""Cerebellum - Supervised Error-Corrective Learning for Precise Motor Control.

The cerebellum learns through supervised error signals from climbing fibers,
enabling fast, precise learning of input-output mappings without trial-and-error.

**Key Features**:
=================
1. **ERROR-CORRECTIVE LEARNING** (Delta Rule):
   - Δw ∝ pre × (target - actual)
   - Direct teaching signal (not reward/punishment like RL)
   - Can learn arbitrary mappings in 1-10 trials (vs hundreds for RL)

2. **CLIMBING FIBER ERROR SIGNAL**:
   - Inferior olive computes mismatch between intended and actual movement
   - Climbing fiber activates Purkinje cell → LTD on active parallel fibers
   - Absence of climbing fiber → LTP (strengthen correct associations)
   - Binary teacher: "correct" or "incorrect"

3. **PRECISE TIMING AND COORDINATION**:
   - Cerebellum is master of temporal precision
   - Can learn sub-millisecond timing patterns
   - Critical for smooth, coordinated movements
   - Predictive timing (anticipates sensory consequences)

4. **FAST SUPERVISED LEARNING**:
   - Unlike RL (needs many trials with delayed rewards)
   - Cerebellum learns in 1-10 trials with immediate feedback
   - Supervised signal provides direct gradient information
   - No exploration needed - direct instruction

Biological Basis:
=================
- Marr (1969) and Albus (1971): Cerebellar learning theory
- Parallel fibers (inputs) → Purkinje cells (outputs)
- Climbing fibers carry error/teaching signals
- LTD at parallel fiber-Purkinje synapses when climbing fiber active

When to Use:
============
- You have explicit target outputs (labels)
- You want to learn arbitrary input→output mappings
- You need fast learning (few trials)
- Precise timing is important
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Dict, Any

import torch

from thalia.core.base.component_config import NeuralComponentConfig
from thalia.core.errors import CheckpointError
from thalia.components.synapses.weight_init import WeightInitializer
from thalia.utils.core_utils import clamp_weights
from thalia.learning.eligibility.trace_manager import EligibilityTraceManager, STDPConfig
from thalia.managers.component_registry import register_region
from thalia.components.neurons.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.components.neurons.neuron_constants import (
    V_THRESHOLD_STANDARD,
    V_RESET_STANDARD,
    E_LEAK,
    E_EXCITATORY,
    E_INHIBITORY,
    NE_GAIN_RANGE,
)
from thalia.regulation.learning_constants import (
    LEARNING_RATE_ERROR_CORRECTIVE,
)
from thalia.learning.homeostasis.synaptic_homeostasis import UnifiedHomeostasis, UnifiedHomeostasisConfig
from thalia.regions.base import (
    NeuralComponent,
)


@dataclass
class CerebellumConfig(NeuralComponentConfig):
    """Configuration specific to cerebellar regions.

    The cerebellum implements ERROR-CORRECTIVE learning through:
    1. Parallel fiber → Purkinje cell connections (learned)
    2. Climbing fiber error signals from inferior olive
    3. LTD when climbing fiber active with parallel fiber

    Key biological features:
    - Error signal triggers immediate learning (not delayed like RL)
    - Can learn arbitrary input-output mappings quickly
    - Uses eligibility traces for temporal credit assignment
    """

    # Learning rates
    learning_rate_ltp: float = LEARNING_RATE_ERROR_CORRECTIVE   # Rate for strengthening correct
    learning_rate_ltd: float = LEARNING_RATE_ERROR_CORRECTIVE   # Rate for weakening incorrect
    # Note: stdp_lr and tau_plus_ms/tau_minus_ms inherited from NeuralComponentConfig

    # Error signaling
    error_threshold: float = 0.01     # Minimum error to trigger learning
    temporal_window_ms: float = 10.0  # Window for coincidence detection

    # Cerebellum uses weaker heterosynaptic competition for faster convergence:
    heterosynaptic_ratio: float = 0.2  # Override base (0.3) - weaker competition

    # Input trace parameters
    input_trace_tau_ms: float = 20.0  # Input trace decay


class ClimbingFiberSystem:
    """Climbing fiber error signaling system.

    Climbing fiber activity means: "You got it WRONG"
    Absence means: "You got it RIGHT (or no feedback)"

    The error signal: target - actual
    - Positive: Should have fired but didn't → strengthen inputs
    - Negative: Fired but shouldn't have → weaken inputs
    """

    def __init__(self, n_output: int, device: str = "cpu"):
        self.n_output = n_output
        self.device = torch.device(device)
        self.error = torch.zeros(n_output, device=self.device)

    def compute_error(
        self,
        actual: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute error signal (climbing fiber activity).

        Args:
            actual: Actual output [n_output] (1D)
            target: Target output [n_output] (1D)

        Returns:
            Error signal [n_output] (1D)
        """
        # Ensure 1D
        if actual.dim() != 1:
            actual = actual.squeeze()
        if target.dim() != 1:
            target = target.squeeze()

        self.error = target - actual
        return self.error

    def reset_state(self) -> None:
        self.error = torch.zeros(self.n_output, device=self.device)

    def get_state(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "error": self.error.clone(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.error = state["error"].to(self.device)


@register_region(
    "cerebellum",
    description="Supervised error-corrective learning via climbing fiber error signals",
    version="2.0",
    author="Thalia Project"
)
class Cerebellum(NeuralComponent):
    """Cerebellar region with supervised error-corrective learning.

    Implements the cerebellar learning rule:
    - Error signal = target - actual (from climbing fibers)
    - Weight change = learning_rate × input × error
    - This is essentially the delta rule / perceptron learning

    Mixins Provide:
    ---------------
    From DiagnosticsMixin:
        - check_health() → HealthMetrics
        - get_firing_rate(spikes) → float
        - check_weight_health(weights, name) → WeightHealth
        - detect_runaway_excitation(spikes) → bool
        - detect_silence(spikes) → bool

    From NeuralComponent (abstract base):
        - forward(input, **kwargs) → Tensor [must implement]
        - reset_state() → None
        - get_diagnostics() → Dict
        - deliver_error(target) → Dict [cerebellum-specific]

    See Also:
        docs/patterns/mixins.md for detailed mixin patterns
    """

    def __init__(self, config: NeuralComponentConfig):
        if not isinstance(config, CerebellumConfig):
            config = CerebellumConfig(
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

        self.cerebellum_config: CerebellumConfig = config  # type: ignore
        super().__init__(config)

        self.climbing_fiber = ClimbingFiberSystem(
            n_output=config.n_output,
            device=config.device,
        )

        # =====================================================================
        # ELIGIBILITY TRACE MANAGER for STDP
        # =====================================================================
        stdp_config = STDPConfig(
            stdp_tau_ms=self.cerebellum_config.stdp_tau_ms,
            eligibility_tau_ms=self.cerebellum_config.eligibility_tau_ms,
            stdp_lr=self.cerebellum_config.stdp_lr,
            a_plus=1.0,
            a_minus=self.cerebellum_config.heterosynaptic_ratio,
            w_min=config.w_min,
            w_max=config.w_max,
            heterosynaptic_ratio=self.cerebellum_config.heterosynaptic_ratio,
        )
        self._trace_manager = EligibilityTraceManager(
            n_input=config.n_input,
            n_output=config.n_output,
            config=stdp_config,
            device=self.device,
        )

        # Beta oscillator phase tracking for motor timing
        self._beta_phase: float = 0.0
        self._gamma_phase: float = 0.0

        # Homeostasis for synaptic scaling
        homeostasis_config = UnifiedHomeostasisConfig(
            weight_budget=config.weight_budget * config.n_input,  # Total budget per neuron
            w_min=config.w_min,
            w_max=config.w_max,
            soft_normalization=config.soft_normalization,
            normalization_rate=config.normalization_rate,
            device=config.device,
        )
        self.homeostasis = UnifiedHomeostasis(homeostasis_config)
        self._theta_phase: float = 0.0
        self._beta_amplitude: float = 1.0
        self._gamma_amplitude: float = 1.0
        self._coupled_amplitudes: Dict[str, float] = {}

    @property
    def input_trace(self) -> torch.Tensor:
        """Input trace (delegated to trace manager)."""
        return self._trace_manager.input_trace

    @property
    def output_trace(self) -> torch.Tensor:
        """Output trace (delegated to trace manager)."""
        return self._trace_manager.output_trace

    @property
    def stdp_eligibility(self) -> torch.Tensor:
        """STDP eligibility (delegated to trace manager)."""
        return self._trace_manager.eligibility

    def _initialize_weights(self) -> torch.Tensor:
        """Initialize weights with small uniform values plus noise."""
        # Small initial weights (10% of max) with slight variation
        weights = WeightInitializer.gaussian(
            n_output=self.config.n_output,
            n_input=self.config.n_input,
            mean=self.config.w_max * 0.1,
            std=0.02,
            device=self.device
        )
        return clamp_weights(weights, self.config.w_min, self.config.w_max, inplace=False)

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Optional[Dict[str, float]] = None,
        theta_slot: int = 0,
        coupled_amplitudes: Optional[Dict[str, float]] = None,
    ) -> None:
        """Receive oscillator information from brain broadcast.

        Beta oscillations are critical for motor control and learning:
        - Pre-movement: beta increases during planning phase
        - Movement initiation: beta desynchronizes (ERD)
        - During movement: low beta allows rapid adjustments
        - Post-movement: beta rebounds (ERS)

        The cerebellum uses beta phase to gate climbing fiber learning:
        - Beta trough (phase = π): Peak learning window (movement initiation)
        - Beta peak (phase = 0/2π): Minimal learning (action maintenance)

        Args:
            phases: Oscillator phases in radians {'theta': ..., 'beta': ..., 'gamma': ...}
            signals: Oscillator signal values (sin/cos waveforms)
            theta_slot: Current theta slot [0, n_slots-1] for working memory
            coupled_amplitudes: Effective amplitudes per oscillator (pre-computed)

        Note:
            Called automatically by Brain before each forward() call.
            Do not call this manually.
        """
        # Store oscillator phases for motor timing
        self._theta_phase = phases.get('theta', 0.0)
        self._beta_phase = phases.get('beta', 0.0)
        self._gamma_phase = phases.get('gamma', 0.0)

        # Store effective amplitudes (pre-computed by OscillatorManager)
        # Automatic multiplicative coupling:
        # - Beta modulated by ALL slower oscillators (delta, theta, alpha)
        # - Gamma modulated by ALL slower oscillators (delta, theta, alpha, beta)
        # OscillatorManager handles the multiplication, we just store the result.
        if coupled_amplitudes is not None:
            self._beta_amplitude = coupled_amplitudes.get('beta', 1.0)
            self._gamma_amplitude = coupled_amplitudes.get('gamma', 1.0)
        else:
            self._beta_amplitude = 1.0
            self._gamma_amplitude = 1.0

    def _compute_beta_gate(self) -> float:
        """Compute beta-gated learning modulation.

        Biology:
        - Climbing fibers from inferior olive fire most effectively during beta trough
        - Beta desynchronization (ERD) signals movement initiation window
        - This is when cerebellum should update motor predictions
        - Automatic multiplicative coupling: beta modulated by all slower oscillators

        Returns:
            Gate value [0, 1] - peak at beta trough (phase = π), modulated by coupling
        """
        # Peak learning at β = π (beta trough = movement initiation)
        phase_diff = abs(self._beta_phase - math.pi)
        width = math.pi / 4  # ±45° learning window
        gate = math.exp(-(phase_diff ** 2) / (2 * width ** 2))

        # Modulate by ALL coupling effects (theta, beta modulation)
        # This gives emergent theta-beta-gamma triple coupling automatically
        gate = gate * self._gamma_amplitude

        return gate

    def _create_neurons(self) -> ConductanceLIF:
        """Create Purkinje-like neurons with constants from neuron_constants.py."""
        neuron_config = ConductanceLIFConfig(
            v_threshold=V_THRESHOLD_STANDARD,
            v_reset=V_RESET_STANDARD,
            E_L=E_LEAK,
            E_E=E_EXCITATORY,
            E_I=E_INHIBITORY,
            tau_E=3.0,  # Faster excitatory for precise timing
            tau_I=8.0,  # Faster inhibitory for precise timing
        )
        neurons = ConductanceLIF(n_neurons=self.config.n_output, config=neuron_config)
        neurons.to(self.device)
        return neurons

    def add_neurons(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Add neurons to cerebellum (Purkinje cells).

        Expands motor learning capacity by adding neurons.

        Args:
            n_new: Number of neurons to add
            initialization: Weight initialization strategy
            sparsity: Sparsity for new connections
        """
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
            scale=1.0,  # Default scale for cerebellum
        )

        # =====================================================================
        # 2. UPDATE CONFIG
        # =====================================================================
        self.config = replace(self.config, n_output=new_n_output)

        # =====================================================================
        # 3. EXPAND NEURON POPULATION using base helper
        # =====================================================================
        self.neurons = self._recreate_neurons_with_state(
            neuron_factory=self._create_neurons,
            old_n_output=old_n_output,
        )

        # =====================================================================
        # 4. GROW TRACE MANAGER
        # =====================================================================
        self._trace_manager = self._trace_manager.add_neurons(n_new, dimension='output')

    def forward(
        self,
        input_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process input through cerebellar circuit.

        Args:
            input_spikes: Input spike pattern [n_input] (1D bool tensor, ADR-005)

        Returns:
            Output spikes [n_output] (1D bool tensor, ADR-004/005)

        Note:
            Theta modulation and timestep (dt_ms) computed internally from config
        """
        # Assert 1D input
        assert input_spikes.dim() == 1, (
            f"Cerebellum.forward: input_spikes must be 1D [n_input], "
            f"got shape {input_spikes.shape}. See ADR-005: No Batch Dimension."
        )
        assert input_spikes.shape[0] == self.config.n_input, (
            f"Cerebellum.forward: input_spikes has {input_spikes.shape[0]} neurons "
            f"but expected {self.config.n_input}."
        )

        if self.neurons.membrane is None:
            self.neurons.reset_state()

        cfg = self.cerebellum_config

        # ======================================================================
        # COMPUTE THETA MODULATION (from oscillator phase set by Brain)
        # ======================================================================
        encoding_mod = 0.5 * (1.0 + math.cos(self._theta_phase))
        _ = 1.0 - encoding_mod  # retrieval_mod - reserved for future output gain modulation

        # Encoding phase: stronger input drive for error detection
        # Retrieval phase: stronger output for motor correction
        input_gain = 0.7 + 0.3 * encoding_mod  # 0.7-1.0
        # Note: retrieval_mod could be used for output gain if needed for motor commands

        # Compute synaptic input - modulated by encoding phase
        dt = self.config.dt_ms
        # 1D matmul: weights[n_output, n_input] @ input[n_input] → [n_output]
        g_exc = (self.weights @ input_spikes.float()) * input_gain

        # =====================================================================
        # NOREPINEPHRINE GAIN MODULATION (Locus Coeruleus)
        # =====================================================================
        # High NE (arousal/stress): Increase motor gain → faster reactions
        # Low NE (baseline): Normal gain
        # Biological: NE modulates cerebellar Purkinje cell excitability
        ne_level = self.state.norepinephrine
        # NE gain: 1.0 (baseline) to 1.5 (high arousal)
        ne_gain = 1.0 + NE_GAIN_RANGE * ne_level
        g_exc = g_exc * ne_gain

        # Forward through neurons (returns 1D bool spikes)
        output_spikes, _ = self.neurons(g_exc, None)

        # NOTE: All neuromodulators (DA, ACh, NE) are now managed centrally by Brain.
        # VTA updates dopamine, LC updates NE, NB updates ACh.
        # Brain broadcasts to all regions every timestep via _update_neuromodulators().
        # No local decay needed.

        # ======================================================================
        # Update STDP eligibility using trace manager
        # ======================================================================
        # Use trace manager for consolidated STDP computation
        self._trace_manager.update_traces(
            input_spikes=input_spikes,
            output_spikes=output_spikes,
            dt_ms=dt,
        )

        # Compute STDP weight change direction (raw LTP/LTD without combining)
        ltp, ltd = self._trace_manager.compute_ltp_ltd_separate(
            input_spikes=input_spikes,
            output_spikes=output_spikes,
        )

        # Combine LTP and LTD with learning rate and heterosynaptic ratio
        stdp_dw = cfg.stdp_lr * (ltp - cfg.heterosynaptic_ratio * ltd)

        # Accumulate into eligibility trace (with decay)
        if isinstance(stdp_dw, torch.Tensor):
            self._trace_manager.accumulate_eligibility(stdp_dw, dt_ms=dt)

        self.state.spikes = output_spikes
        self.state.t += 1

        # Apply axonal delay (biological reality: ALL neural connections have delays)
        delayed_spikes = self._apply_axonal_delay(output_spikes, dt)

        return delayed_spikes

    def _apply_error_learning(
        self,
        output_spikes: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, Any]:
        """Apply error-corrective learning using accumulated eligibility.

        This is the core plasticity mechanism, called by deliver_error().
        Uses STDP eligibility traces modulated by error signal:
        - Eligibility accumulates spike-timing correlations during forward()
        - Error signal (climbing fiber) gates weight changes
        - Positive error → apply eligibility (LTP for correct timing)
        - Negative error → apply anti-eligibility (LTD for incorrect timing)

        Dopamine modulates the overall learning rate (arousal/attention effect).

        Args:
            output_spikes: Output spike pattern [n_output] (1D bool, ADR-005)
            target: Target output [n_output] (1D, ADR-005)

        Returns:
            Dictionary with learning metrics
        """
        # Assert 1D inputs
        assert output_spikes.dim() == 1, (
            f"Cerebellum._apply_error_learning: output_spikes must be 1D, "
            f"got shape {output_spikes.shape}"
        )
        assert target.dim() == 1, (
            f"Cerebellum._apply_error_learning: target must be 1D, "
            f"got shape {target.shape}"
        )

        cfg = self.cerebellum_config

        # ======================================================================
        # Compute error via climbing fiber system
        # ======================================================================
        error = self.climbing_fiber.compute_error(output_spikes.float(), target.float())
        # error is already 1D from compute_error()

        if error.abs().max() < cfg.error_threshold:
            return {"error": 0.0, "ltp": 0.0, "ltd": 0.0}

        # ======================================================================
        # BETA GATING - Motor timing modulation
        # ======================================================================
        # Climbing fiber learning is most effective during beta trough (movement initiation)
        # This implements the biological observation that error signals are processed
        # during movement execution, not during movement maintenance
        beta_gate = self._compute_beta_gate()

        # ======================================================================
        # Error-modulated STDP learning
        # ======================================================================
        # The key insight: use error to select WHICH neurons update
        # - Neurons with positive error (should have fired more) → LTP
        # - Neurons with negative error (fired too much) → LTD

        # Scale eligibility by per-neuron error direction
        # Positive error → apply eligibility (strengthen timing correlations)
        # Negative error → apply anti-eligibility (weaken timing correlations)
        # error is already 1D [n_output]
        error_sign = torch.sign(error).unsqueeze(1)  # [n_output, 1]

        # Modulate eligibility by error magnitude and sign
        # Dopamine provides arousal/attention modulation (from VTA via Brain)
        # Beta gate provides motor timing modulation (movement initiation window)
        effective_lr = self.get_effective_learning_rate() * beta_gate
        dw = self.stdp_eligibility * error_sign * error.abs().unsqueeze(1) * effective_lr

        # Update weights with hard bounds
        # Biological regulation via UnifiedHomeostasis (weight normalization)
        old_weights = self.weights.clone()
        self.weights = clamp_weights(self.weights + dw, self.config.w_min, self.config.w_max, inplace=False)

        # Synaptic scaling for homeostasis using UnifiedHomeostasis
        if cfg.homeostasis_enabled:
            with torch.no_grad():
                self.weights = self.homeostasis.normalize_weights(self.weights, dim=1)

        actual_dw = self.weights - old_weights
        ltp = actual_dw[actual_dw > 0].sum().item() if (actual_dw > 0).any() else 0.0
        ltd = actual_dw[actual_dw < 0].sum().item() if (actual_dw < 0).any() else 0.0

        return {
            "error": error.abs().mean().item(),
            "ltp": ltp,
            "ltd": ltd,
            "net_change": ltp + ltd,
            "eligibility_max": self.stdp_eligibility.abs().max().item(),
            "beta_gate": beta_gate,  # Motor timing window
        }

    def deliver_error(
        self,
        target: torch.Tensor,
        output_spikes: Optional[torch.Tensor] = None,
        target_neuron: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Deliver error signal (climbing fiber) and apply learning.

        This is the main learning API for cerebellum - analogous to
        deliver_reward() for striatum. The pattern is:
        1. Run forward() to build eligibility traces (spike-timing correlations)
        2. Call deliver_error() when target/error is known

        The climbing fiber carries the error signal from inferior olive:
        - Positive error (target - actual > 0): should have fired more → LTP
        - Negative error (target - actual < 0): fired too much → LTD

        Dopamine (set via set_dopamine()) modulates learning rate,
        providing arousal/attention effects on motor learning.

        Args:
            target: Target output pattern [n_output] or [batch, n_output]
            output_spikes: Current output (if None, uses last state from forward())
            target_neuron: Single target neuron index (alternative to target tensor)

        Returns:
            Learning metrics dict with error, ltp, ltd, net_change
        """
        # Handle target_neuron convenience parameter
        if target_neuron is not None:
            target = torch.zeros(1, self.config.n_output, device=self.device)
            target[0, target_neuron] = 1.0

        if output_spikes is None:
            output_spikes = self.state.spikes

        if output_spikes is None:
            return {"error": 0.0, "ltp": 0.0, "ltd": 0.0}

        # Apply error-corrective learning using accumulated eligibility
        return self._apply_error_learning(output_spikes, target)

    def reset_state(self) -> None:
        super().reset_state()

        # Reset subsystems (trace manager handles its own traces)
        self._reset_subsystems('_trace_manager', 'climbing_fiber')

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics using DiagnosticsMixin helpers.

        Reports weight statistics, traces, and error signals.
        """
        cfg = self.cerebellum_config

        # Custom metrics specific to cerebellum
        custom = {
            "n_input": cfg.n_input,
            "n_output": cfg.n_output,
            "config_w_min": cfg.w_min,
            "config_w_max": cfg.w_max,
            "error_mean": self.climbing_fiber.get_state().get("last_error_mean", 0.0),
            "beta_phase": self._beta_phase,
            "gamma_phase": self._gamma_phase,
            "theta_phase": self._theta_phase,
            "beta_amplitude": self._beta_amplitude,
            "gamma_amplitude": self._gamma_amplitude,
        }

        # Use collect_standard_diagnostics for weight, spike, and trace statistics
        return self.collect_standard_diagnostics(
            region_name="cerebellum",
            weight_matrices={
                "parallel_fiber": self.weights.data,
            },
            spike_tensors={
                "output": self.state.spikes,
            },
            trace_tensors={
                "input": self.input_trace,
                "output": self.output_trace,
                "eligibility": self.stdp_eligibility,
            },
            custom_metrics=custom,
        )

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for checkpointing.

        Returns state dictionary with keys:
        - weights: Parallel fiber to Purkinje cell weights
        - region_state: Neuron state, traces
        - learning_state: Eligibility traces, climbing fiber state
        - config: Configuration for validation
        """
        state_dict = {
            "weights": {
                "parallel_fiber_purkinje": self.weights.data.clone(),
            },
            "region_state": {
                "neurons": self.neurons.get_state() if self.neurons is not None else None,
                "trace_manager": self._trace_manager.get_state(),
            },
            "learning_state": {
                "stdp_eligibility": self.stdp_eligibility.clone(),
                "climbing_fiber": self.climbing_fiber.get_state(),
            },
            "config": {
                "n_input": self.config.n_input,
                "n_output": self.config.n_output,
            },
        }

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
            raise CheckpointError(f"Config mismatch: n_input {config.get('n_input')} != {self.config.n_input}")
        if config.get("n_output") != self.config.n_output:
            raise CheckpointError(f"Config mismatch: n_output {config.get('n_output')} != {self.config.n_output}")

        # Restore weights
        weights = state["weights"]
        self.weights.data.copy_(weights["parallel_fiber_purkinje"].to(self.device))

        # Restore neuron state
        region_state = state["region_state"]
        if self.neurons is not None and region_state["neurons"] is not None:
            self.neurons.load_state(region_state["neurons"])

        # Restore trace manager state
        if "trace_manager" in region_state:
            self._trace_manager.load_state(region_state["trace_manager"])
        else:
            # Backward compatibility: old checkpoints had separate traces
            if "input_trace" in region_state:
                self._trace_manager.input_trace.copy_(region_state["input_trace"].to(self.device))
            if "output_trace" in region_state:
                self._trace_manager.output_trace.copy_(region_state["output_trace"].to(self.device))

        # Restore learning state
        learning_state = state["learning_state"]
        self._trace_manager.eligibility.copy_(learning_state["stdp_eligibility"].to(self.device))
        self.climbing_fiber.load_state(learning_state["climbing_fiber"])
